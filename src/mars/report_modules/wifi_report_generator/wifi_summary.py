#!/usr/bin/env python3
"""Summarize Wi-Fi/network forensic artefacts from an exemplar scan."""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import json
import plistlib
import re
import struct
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from mars.utils.debug_logger import logger

from . import wifi_log_merger

if TYPE_CHECKING:
    from collections.abc import Callable

SSID_TOKEN_RE = re.compile(r"<([0-9A-Fa-f ]+)>$")
SSID_BYTES_RE = re.compile(r"\{length\s*=\s*\d+,\s*bytes\s*=\s*0x([0-9A-Fa-f]+)\}$")
TIMESTAMP_RE = re.compile(
    r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?)"
)
SSID_LINE_RE = re.compile(r"""\bSSID\b\s*(?:=|:)\s*(?:"([^"]+)"|([^\s,\]"'=]+))""")
SSID_HEX_RE = re.compile(r"wifi\.ssid\.<([0-9A-Fa-f ]+)>")
AUTOJOIN_NAME_RE = re.compile(r"'([^']+)'")
PLAIN_QUOTED_LINE_RE = re.compile(r'^\s*"([^"]+)"\s*$')
BSSID_RE = re.compile(r"(?:BSSID|bssid)[=:\s]+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})")
NETWORK_ADDED_RE = re.compile(r"Added network\s+(.*)$")
RSSI_RE = re.compile(r"RSSI[=:\s]+(-?\d+)")
SNR_RE = re.compile(r"SNR[=:\s]+(-?\d+)")
CHANNEL_RE = re.compile(r"(?:channel|chan)[=:\s]+(\d+)", re.IGNORECASE)

EVENT_FIELDS = [
    "source",
    "vfs_path",
    "host_interface",
    "host_mac",
    "timestamp",
    "ssid",
    "bssid",
    "oui",
    "vendor",
    "channel",
    "band",
    "rssi",
    "snr",
    "laa",
    "wake_context",
    "ip_address",
    "router_ip",
    "router_mac",
    "lease_start",
    "lease_end",
    "notes",
]

DEFAULT_MAX_PLIST_MB = 50
DEFAULT_OUI_MAP_PATH = Path(__file__).resolve().parent / "oui.csv"
PRESENCE_EVENT_SOURCES: set[str] | None = None

# Month name to number mapping for year inference
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class ParserOptions:
    tzinfo: ZoneInfo | None
    max_plist_bytes: int
    collector: NormalizedEventCollector | None = None
    reference_years: set[int] | None = None  # Years from DHCP leases for log timestamp inference


class NormalizedEventCollector:
    def __init__(self, oui_lookup: Callable[[str | None], tuple[str | None, str | None]]):
        self._events: list[dict[str, Any]] = []
        self._lookup = oui_lookup

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    def add(self, source: str, **kwargs: Any) -> dict[str, Any]:
        event = normalize_event(source, oui_lookup=self._lookup, **kwargs)
        self._events.append(event)
        return event


def _sanitize_mac(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parts = re.findall(r"[0-9A-Fa-f]{2}", value)
    if len(parts) != 6:
        return None
    return ":".join(part.lower() for part in parts)


def normalize_bssid(bssid: str | None) -> str | None:
    if not bssid:
        return None
    text = bssid.strip().lower()
    if not text:
        return None
    parts = [p.zfill(2) for p in text.split(":") if p]
    if len(parts) != 6:
        hex_only = re.sub(r"[^0-9a-f]", "", text)
        if len(hex_only) == 12:
            parts = [hex_only[i : i + 2] for i in range(0, 12, 2)]
    return ":".join(parts) if len(parts) == 6 else None


def channel_to_band(ch: Any, frequency_mhz: int | None = None) -> str | None:
    """Derive WiFi band from channel number and optional frequency.

    Channel number alone is ambiguous for 6GHz (WiFi 6E) since channels 1-233
    overlap with 2.4GHz (1-14) and 5GHz (32-196). When frequency is available,
    it provides unambiguous band identification.

    Args:
        ch: WiFi channel number
        frequency_mhz: Optional center frequency in MHz for disambiguation

    Returns:
        Band string ("2.4GHz", "5GHz", "6GHz") or None if indeterminate
    """
    try:
        value = int(str(ch))
    except (TypeError, ValueError):
        return None

    # If frequency is provided, use it for unambiguous band identification
    if frequency_mhz is not None:
        if 2400 <= frequency_mhz <= 2500:
            return "2.4GHz"
        if 5150 <= frequency_mhz <= 5925:
            return "5GHz"
        if 5925 <= frequency_mhz <= 7125:
            return "6GHz"

    # Channel-only heuristics (some overlap exists)
    # 2.4GHz: channels 1-14 (most common interpretation)
    if 1 <= value <= 14:
        return "2.4GHz"
    # 5GHz: channels 32-196 (standard 5GHz range)
    if 32 <= value <= 196:
        return "5GHz"
    # 6GHz: channels 197-233 are exclusively WiFi 6E (no overlap)
    if 197 <= value <= 233:
        return "6GHz"

    return None


def _derive_band(channel: Any) -> str | None:
    # Backwards compatibility helper retained for other call sites
    return channel_to_band(channel)


def is_locally_administered(bssid: str | None) -> bool:
    normalized = normalize_bssid(bssid)
    if not normalized:
        return False
    try:
        first_byte = int(normalized.split(":")[0], 16)
    except (ValueError, IndexError):
        return False
    return bool(first_byte & 0x02)


def _format_notes(notes: Any) -> str | None:
    if notes is None:
        return None
    if isinstance(notes, (list, tuple, set)):
        flattened = [str(item) for item in notes if item not in {None, ""}]
        return "; ".join(flattened) if flattened else None
    if isinstance(notes, dict):
        try:
            return json.dumps(notes, sort_keys=True)
        except TypeError:
            return "; ".join(f"{k}={v}" for k, v in notes.items())
    return str(notes)


def normalize_event(
    source: str,
    *,
    oui_lookup: Callable[[str | None], tuple[str | None, str | None]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = dict.fromkeys(EVENT_FIELDS)
    event["source"] = source

    for key, value in kwargs.items():
        if key not in event:
            continue
        if key in {"timestamp", "lease_start", "lease_end"} and isinstance(value, datetime):
            event[key] = _format_datetime(value)
        elif key in {"channel", "rssi", "snr"}:
            try:
                event[key] = int(value) if value is not None else None
            except (TypeError, ValueError):
                event[key] = None
        elif key in {"bssid", "host_mac", "router_mac"}:
            event[key] = _sanitize_mac(value) or value
        else:
            event[key] = value

    event["notes"] = _format_notes(event.get("notes"))
    if not event.get("band"):
        event["band"] = _derive_band(event.get("channel"))

    if oui_lookup:
        mac_candidate = event.get("bssid") or event.get("host_mac") or event.get("router_mac")
        oui, vendor = oui_lookup(mac_candidate)
        event["oui"] = event.get("oui") or oui
        event["vendor"] = event.get("vendor") or vendor
    return event


def derive_oui(value: str | None) -> str | None:
    sanitized = _sanitize_mac(value)
    if not sanitized:
        return None
    return sanitized.replace(":", "").upper()[:6]


def enrich_with_oui_vendor(
    oui_map_csv: Path | None,
) -> Callable[[str | None], tuple[str | None, str | None]]:
    mapping: dict[str, str] = {}
    if not oui_map_csv:
        return lambda mac: (derive_oui(mac), None)
    try:
        with oui_map_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [h.strip() for h in reader.fieldnames or []]
            oui_field = None
            vendor_field = None
            for header in headers:
                lower = header.lower()
                if not oui_field and ("oui" in lower or "assignment" in lower or "prefix" in lower):
                    oui_field = header
                if not vendor_field and any(token in lower for token in ("vendor", "organization", "org", "company")):
                    vendor_field = header
            for row in reader:
                if not row:
                    continue
                key = (row.get(oui_field or "") or row.get("Assignment") or "").strip()
                vendor = (row.get(vendor_field or "") or "").strip()
                if not key:
                    continue
                normalized = key.replace(":", "").replace("-", "").upper()
                if len(normalized) >= 6 and vendor:
                    mapping[normalized[:6]] = vendor
    except FileNotFoundError:
        logger.info(f"[warn] OUI map {oui_map_csv} not found, vendor enrichment disabled.")
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[warn] Failed to load OUI map {oui_map_csv}: {exc}")

    def lookup(mac: str | None) -> tuple[str | None, str | None]:
        oui = derive_oui(mac)
        if not oui:
            return (None, None)
        return (oui, mapping.get(oui))

    return lookup


def decode_hex_token(token: str) -> str:
    """Decode SSID from hex representation.

    Handles two formats:
    - Old: <32737465 7032646d 622e6d65 646961>
    - New: {length = 5, bytes = 0x4775657374}
    """
    stripped = token.strip()

    # Try old format: <hex hex hex>
    match = SSID_TOKEN_RE.match(stripped)
    if match:
        hex_str = match.group(1).replace(" ", "")
        try:
            return bytes.fromhex(hex_str).decode("utf-8", "replace") or token
        except ValueError:
            return token

    # Try new format: {length = N, bytes = 0xHEX}
    match = SSID_BYTES_RE.match(stripped)
    if match:
        hex_str = match.group(1)
        try:
            return bytes.fromhex(hex_str).decode("utf-8", "replace") or token
        except ValueError:
            return token

    # Not a recognized hex format, return as-is
    return token


def format_bytes_as_mac(value: Any) -> str | None:
    if isinstance(value, bytes):
        return ":".join(f"{b:02x}" for b in value)
    if isinstance(value, str) and len(value) == 12 and all(c in "0123456789abcdefABCDEF" for c in value):
        return ":".join(value[i : i + 2] for i in range(0, 12, 2))
    return None


def safe_load_plist(path: Path, *, max_bytes: int | None = None) -> tuple[Any | None, dict[str, Any]]:
    meta: dict[str, Any] = {}
    lname = path.name.lower()
    if "provenance" in lname:
        return None, meta
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        return None, meta

    if max_bytes and file_size > max_bytes:
        meta["notes"] = "too_large"
        meta["top_keys"] = _extract_top_level_keys(path)
        return None, meta

    try:
        with path.open("rb") as fh:
            return plistlib.load(fh), meta
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[warn] Failed to parse plist {path}: {exc}")
        return None, meta


def _extract_top_level_keys(path: Path, limit: int = 256_000) -> list[str]:
    try:
        with path.open("rb") as fh:
            head = fh.read(limit)
    except Exception:  # noqa: BLE001
        return []
    if head.startswith(b"bplist"):
        return []
    try:
        text = head.decode("utf-8", "ignore")
    except UnicodeDecodeError:
        return []
    start_idx = text.find("<dict>")
    end_idx = text.find("</dict>")
    snippet = text if start_idx == -1 or end_idx == -1 or end_idx <= start_idx else text[start_idx:end_idx]
    keys = re.findall(r"<key>([^<]+)</key>", snippet)
    return list(dict.fromkeys(keys))


def extract_log_timestamp(line: str) -> str | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return match.group(1)


def infer_year_for_log_timestamp(
    timestamp_str: str,
    reference_years: set[int] | None = None,
    tzinfo: ZoneInfo | None = None,
) -> str | None:
    """Infer full ISO timestamp from syslog-style timestamp using reference years.

    Syslog timestamps like "Mar 15 10:23:45" lack year information. This function
    attempts to construct a full datetime by trying candidate years from reference
    timestamps (e.g., DHCP leases).

    Args:
        timestamp_str: Syslog-style timestamp (e.g., "Mar 15 10:23:45")
        reference_years: Years to try, from sources with known dates (DHCP leases)
        tzinfo: Optional timezone for output

    Returns:
        ISO formatted timestamp string, or None if parsing fails
    """
    if not timestamp_str:
        return None

    # Parse "Mar 15 10:23:45" or "Mar  5 10:23:45.123"
    parts = timestamp_str.split()
    if len(parts) < 3:
        return None

    month_str = parts[0].lower()[:3]
    month = MONTH_MAP.get(month_str)
    if not month:
        return None

    try:
        day = int(parts[1])
        time_parts = parts[2].split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        # Handle optional milliseconds
        sec_parts = time_parts[2].split(".")
        second = int(sec_parts[0])
        microsecond = int(sec_parts[1]) * 1000 if len(sec_parts) > 1 else 0
    except (ValueError, IndexError):
        return None

    # Determine candidate years
    if reference_years:
        # Use years from reference timestamps, plus adjacent years for edge cases
        candidates = set(reference_years)
        candidates.update(y - 1 for y in reference_years)
        candidates.update(y + 1 for y in reference_years)
        candidate_years = sorted(candidates, reverse=True)  # Prefer more recent
    else:
        # Fall back to current year and previous year
        now = datetime.now(UTC)
        candidate_years = [now.year, now.year - 1]

    # Try each candidate year
    for year in candidate_years:
        try:
            dt = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=UTC)
            # Don't accept future dates
            if dt > datetime.now(UTC):
                continue
            # Format with timezone if provided
            if tzinfo:
                dt = dt.replace(tzinfo=tzinfo)
                return dt.isoformat()
            return dt.isoformat() + "Z"
        except ValueError:
            # Invalid date (e.g., Feb 30)
            continue

    return None


def extract_years_from_timestamps(timestamps: list[str | None]) -> set[int]:
    """Extract unique years from ISO timestamp strings.

    Args:
        timestamps: List of ISO format timestamp strings

    Returns:
        Set of years found in the timestamps
    """
    years: set[int] = set()
    for ts in timestamps:
        if not ts:
            continue
        try:
            # ISO format: 2024-03-15T10:23:45Z or 2024-03-15T10:23:45+00:00
            year = int(ts[:4])
            if 1990 <= year <= 2100:  # Sanity check
                years.add(year)
        except (ValueError, IndexError):
            continue
    return years


def _search_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_ssid_and_bssid(line: str) -> tuple[str | None, str | None]:
    ssid = None
    bssid = None
    ssid_match = SSID_LINE_RE.search(line)
    if ssid_match:
        ssid = ssid_match.group(1) or ssid_match.group(2)
        if ssid:
            ssid = ssid.strip().strip("=")
            normalized = ssid.lower()
            if normalized in {"nil", "null", "(null)", "ssid"}:
                ssid = None
    else:
        added_match = NETWORK_ADDED_RE.search(line)
        if added_match:
            candidate = added_match.group(1).strip()
            name_match = AUTOJOIN_NAME_RE.search(candidate)
            ssid = name_match.group(1) if name_match else candidate
    if not ssid:
        hex_match = SSID_HEX_RE.search(line)
        if hex_match:
            ssid = decode_hex_token(f"<{hex_match.group(1)}>").strip()
    if not ssid:
        # Check for new format: {length = X, bytes = 0x...}
        # Be smart about what to extract - avoid IPs (4 bytes) and MACs (6 bytes)
        bytes_match = SSID_BYTES_RE.search(line)
        if bytes_match:
            hex_bytes = bytes_match.group(1)
            byte_length = len(hex_bytes) // 2  # Each byte is 2 hex chars

            # SSIDs are typically 1-32 bytes
            # Exclude common non-SSID sizes:
            # - 4 bytes: IPv4 addresses
            # - 6 bytes: MAC addresses
            # - 8 bytes: timestamps/integers
            # - 16 bytes: UUIDs/GUIDs
            if byte_length > 8 and byte_length <= 32:
                # Reconstruct the full token for decode_hex_token
                full_token = f"{{length = 0, bytes = 0x{hex_bytes}}}"
                decoded = decode_hex_token(full_token).strip()
                # Additional check: decoded value should be mostly printable ASCII
                if decoded and sum(32 <= ord(c) <= 126 for c in decoded) / len(decoded) > 0.7:
                    ssid = decoded
    if not ssid:
        plain_match = PLAIN_QUOTED_LINE_RE.match(line)
        if plain_match:
            ssid = plain_match.group(1).strip()
    if ssid:
        ssid = ssid.strip().strip("\"'[]")
        normalized = ssid.lower()
        if normalized in {"nil", "null", "(null)", "ssid"}:
            ssid = None
    bssid_match = BSSID_RE.search(line)
    if bssid_match:
        bssid = bssid_match.group(1).lower()
    return ssid, bssid


def parse_wifi_logs(log_files: list[Path], opts: ParserOptions) -> dict[str, Any]:
    events = []
    seen: set[str] = set()
    for path in sorted(log_files):
        base_key = path.with_suffix("").name if path.suffix == ".bz2" else path.name
        if base_key in seen:
            continue
        seen.add(base_key)
        try:
            if path.suffix == ".bz2":
                import bz2

                fh = bz2.open(  # noqa: SIM115
                    path, "rt", encoding="utf-8", errors="replace"
                )
            else:
                fh = path.open("rt", encoding="utf-8", errors="replace")
            with fh as fh_text:
                for line in fh_text:
                    line = line.rstrip()
                    if not line:
                        continue
                    raw_timestamp = extract_log_timestamp(line)
                    ssid, bssid = extract_ssid_and_bssid(line)
                    if ssid or bssid or (raw_timestamp and "SSID" in line.upper()):
                        normalized_bssid = normalize_bssid(bssid)
                        channel = _search_int(CHANNEL_RE, line)
                        rssi = _search_int(RSSI_RE, line)
                        snr = _search_int(SNR_RE, line)
                        band = channel_to_band(channel)
                        laa_value = "Yes" if is_locally_administered(normalized_bssid) else "No"

                        # Infer year from reference timestamps (DHCP leases)
                        inferred_timestamp = None
                        if raw_timestamp:
                            inferred_timestamp = infer_year_for_log_timestamp(
                                raw_timestamp, opts.reference_years, opts.tzinfo
                            )

                        # Use inferred timestamp if available, otherwise raw
                        timestamp = inferred_timestamp or raw_timestamp

                        event_payload = {
                            "path": str(path),
                            "timestamp": timestamp,
                            "timestamp_raw": raw_timestamp,  # Preserve original for transparency
                            "ssid": ssid,
                            "bssid": normalized_bssid,
                            "line": line,
                            "laa": laa_value,
                        }
                        if channel is not None:
                            event_payload["channel"] = channel
                        if band:
                            event_payload["band"] = band
                        if rssi is not None:
                            event_payload["rssi"] = rssi
                        if snr is not None:
                            event_payload["snr"] = snr
                        events.append(event_payload)
                        if opts.collector:
                            opts.collector.add(
                                "wifi_log",
                                vfs_path=str(path),
                                timestamp=timestamp,
                                ssid=ssid,
                                bssid=normalized_bssid,
                                channel=channel,
                                band=band,
                                rssi=rssi,
                                snr=snr,
                                laa=laa_value,
                                notes=line,
                            )
        except FileNotFoundError:
            continue
    return {"count": len(events), "events": events[:1000]}


def parse_airport_preferences(paths: list[Path], opts: ParserOptions) -> list[dict[str, Any]]:
    networks: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            placeholder = {
                "source": str(path),
                "notes": "too_large",
                "top_keys": meta.get("top_keys", []),
            }
            networks.append(placeholder)
            if opts.collector:
                opts.collector.add(
                    "airport_known_networks",
                    vfs_path=str(path),
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue

        # Format detection: old format has "KnownNetworks" key
        # Big Sur+ format has direct network entries with keys like "wifi.network.ssid.*"
        has_known_networks = "KnownNetworks" in data

        if has_known_networks:
            # OLD FORMAT (pre-Big Sur): Parse KnownNetworks
            known = data.get("KnownNetworks", {})
            if not isinstance(known, dict):
                continue
            for entry in known.values():
                if not isinstance(entry, dict):
                    continue
                last_connected = _format_datetime(entry.get("LastConnected"), opts.tzinfo)
                payload = {
                    "source": str(path),
                    "ssid": entry.get("SSIDString") or entry.get("SSID") or None,
                    "last_connected": last_connected,
                    "security_type": entry.get("SecurityType"),
                    "auto_join": bool(entry.get("AutoJoin", True)),
                    "channel_history": _format_channel_history(entry.get("ChannelHistory"), opts.tzinfo),
                    "bssid_list": _format_bssid_list(entry.get("BSSIDList")),
                    "passpoint": bool(entry.get("Passpoint", 0)),
                    "hidden": bool(entry.get("PossiblyHiddenNetwork", 0)),
                }
                interface_name = entry.get("Interface")
                if isinstance(interface_name, str):
                    payload["interface"] = interface_name
                networks.append(payload)
                if opts.collector:
                    channel_history = payload.get("channel_history") or []
                    last_channel = (
                        channel_history[-1].get("channel")
                        if channel_history and isinstance(channel_history[-1], dict)
                        else None
                    )
                    notes = _format_notes(
                        [
                            f"security={payload.get('security_type')}",
                            f"auto_join={payload.get('auto_join')}",
                            f"hidden={payload.get('hidden')}",
                            f"passpoint={payload.get('passpoint')}",
                        ]
                    )
                    opts.collector.add(
                        "airport_known_network",
                        vfs_path=str(path),
                        host_interface=payload.get("interface"),
                        timestamp=last_connected,
                        ssid=payload.get("ssid"),
                        channel=last_channel,
                        notes=notes,
                    )
                    for bssid in payload.get("bssid_list", []):
                        opts.collector.add(
                            "airport_known_network_bssid",
                            vfs_path=str(path),
                            host_interface=payload.get("interface"),
                            ssid=payload.get("ssid"),
                            bssid=bssid,
                            timestamp=last_connected,
                            notes="airport_bssid_list",
                        )
        else:
            # NEW FORMAT (Big Sur+): Parse direct network entries
            # Look for keys starting with "wifi.network.ssid."
            for key, entry in data.items():
                if not isinstance(key, str) or not key.startswith("wifi.network.ssid."):
                    continue
                if not isinstance(entry, dict):
                    continue

                # Decode SSID from base64 data field
                ssid_data = entry.get("SSID")
                ssid = _decode_ssid_data(ssid_data)

                # Extract timestamps
                added_at = _format_datetime(entry.get("AddedAt"), opts.tzinfo)
                joined_user = _format_datetime(entry.get("JoinedByUserAt"), opts.tzinfo)
                joined_system = _format_datetime(entry.get("JoinedBySystemAt"), opts.tzinfo)
                last_discovered = _format_datetime(entry.get("LastDiscoveredAt"), opts.tzinfo)
                updated_at = _format_datetime(entry.get("UpdatedAt"), opts.tzinfo)
                last_disconnect_ts = _format_datetime(entry.get("LastDisconnectTimestamp"), opts.tzinfo)

                # Build payload with Big Sur+ fields
                # Add last_connected for compatibility with report generation
                # Use the most recent/relevant timestamp
                last_connected = joined_user or joined_system or added_at

                payload: dict[str, Any] = {
                    "source": str(path),
                    "ssid": ssid,
                    "last_connected": last_connected,  # For report compatibility
                    "add_reason": entry.get("AddReason"),  # Cloud Sync vs WiFi Menu - CRITICAL
                    "added_at": added_at,
                    "joined_by_user_at": joined_user,
                    "joined_by_system_at": joined_system,
                    "last_discovered_at": last_discovered,
                    "updated_at": updated_at,
                    "security_type": entry.get("SupportedSecurityTypes"),
                    "hidden": bool(entry.get("Hidden", False)),
                    "moving": bool(entry.get("Moving", False)),  # Network used while mobile
                    # Fields for report compatibility (not present in Big Sur+ format)
                    "passpoint": False,
                    "auto_join": None,
                }

                # Extract __OSSpecific__ data
                os_specific = entry.get("__OSSpecific__")
                if isinstance(os_specific, dict):
                    payload["channel_history"] = _format_channel_history(os_specific.get("ChannelHistory"), opts.tzinfo)
                    payload["collocated_group"] = os_specific.get("CollocatedGroup")
                    payload["roaming_profile"] = os_specific.get("RoamingProfileType")
                    payload["temporarily_disabled"] = bool(os_specific.get("TemporarilyDisabled", False))

                # Extract BSSList with location data - CRITICAL FORENSIC DATA
                bss_list = _format_bigsur_bsslist(entry.get("BSSList"), opts.tzinfo)
                if bss_list:
                    payload["bss_list"] = bss_list

                # Extract disconnect info
                disconnect_reason = entry.get("LastDisconnectReason")
                if disconnect_reason is not None:
                    payload["last_disconnect_reason"] = disconnect_reason
                    payload["last_disconnect_timestamp"] = last_disconnect_ts

                # Extract seamless SSID list
                seamless_ssids = entry.get("SeamlessSSIDList")
                if isinstance(seamless_ssids, list):
                    decoded_seamless = []
                    for ssid_bytes in seamless_ssids:
                        decoded = _decode_ssid_data(ssid_bytes)
                        if decoded:
                            decoded_seamless.append(decoded)
                    if decoded_seamless:
                        payload["seamless_ssid_list"] = decoded_seamless

                # Extract user preferred network names
                preferred_names = entry.get("UserPreferredNetworkNames")
                if isinstance(preferred_names, dict):
                    # Convert datetime values to strings for JSON serialization
                    formatted_names = {}
                    for name, timestamp in preferred_names.items():
                        formatted_names[name] = _format_datetime(timestamp, opts.tzinfo)
                    payload["user_preferred_names"] = formatted_names

                # Extract cached private MAC address
                private_mac = entry.get("CachedPrivateMACAddress")
                if private_mac:
                    payload["cached_private_mac"] = private_mac
                    mac_updated = _format_datetime(entry.get("CachedPrivateMACAddressUpdatedAt"), opts.tzinfo)
                    if mac_updated:
                        payload["cached_private_mac_updated_at"] = mac_updated

                # Extract captive portal info
                captive_profile = entry.get("CaptiveProfile")
                if isinstance(captive_profile, dict):
                    payload["captive_network"] = bool(captive_profile.get("CaptiveNetwork", False))

                # Extract JoinedBySystemAtWeek
                system_week = entry.get("JoinedBySystemAtWeek")
                if system_week is not None:
                    payload["joined_by_system_at_week"] = system_week

                networks.append(payload)

                # Add to collector for normalized events
                if opts.collector:
                    # Determine primary timestamp (prefer user join, then system join, then added)
                    primary_timestamp = joined_user or joined_system or added_at

                    # Build notes with key forensic indicators
                    notes_parts = []
                    if payload.get("add_reason"):
                        notes_parts.append(f"add_reason={payload['add_reason']}")
                    if payload.get("security_type"):
                        notes_parts.append(f"security={payload['security_type']}")
                    if payload.get("moving"):
                        notes_parts.append("moving=True")
                    if payload.get("temporarily_disabled"):
                        notes_parts.append("disabled=True")

                    notes = _format_notes(notes_parts) if notes_parts else None

                    # Add main network event
                    channel_history = payload.get("channel_history") or []
                    last_channel = (
                        channel_history[-1].get("channel")
                        if channel_history and isinstance(channel_history[-1], dict)
                        else None
                    )

                    opts.collector.add(
                        "airport_known_network_bigsur",
                        vfs_path=str(path),
                        timestamp=primary_timestamp,
                        ssid=ssid,
                        channel=last_channel,
                        notes=notes,
                    )

                    # Add events for each BSSID with location data
                    for bss_entry in bss_list:
                        bssid = bss_entry.get("bssid")
                        location = bss_entry.get("location")
                        if bssid and location:
                            # This is CRITICAL - we have geolocation for this network!
                            lat = location.get("latitude")
                            lon = location.get("longitude")
                            loc_notes = f"lat={lat},lon={lon},map={location.get('map_link_google')}"
                            opts.collector.add(
                                "airport_bssid_location",
                                vfs_path=str(path),
                                ssid=ssid,
                                bssid=bssid,
                                timestamp=location.get("timestamp") or primary_timestamp,
                                notes=loc_notes,
                            )
    return networks


def _format_channel_history(value: Any, tzinfo: ZoneInfo | None) -> list[dict[str, Any]]:
    history = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                history.append(
                    {
                        "channel": item.get("Channel"),
                        "timestamp": _format_datetime(item.get("Timestamp"), tzinfo),
                    }
                )
    return history


def _format_bssid_list(value: Any) -> list[str]:
    bssids = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                raw = item.get("LEAKY_AP_BSSID") or item.get("BSSID")
                normalized = normalize_bssid(raw) if isinstance(raw, str) else None
                if normalized:
                    bssids.append(normalized)
    return bssids


def _decode_ssid_data(value: Any) -> str | None:
    """Decode SSID from base64 data field (Big Sur+ format)"""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _generate_map_link(lat: float, lon: float, provider: str = "google") -> str:
    """Generate map URL from coordinates"""
    if provider == "apple":
        return f"http://maps.apple.com/?ll={lat},{lon}"
    # Default to Google Maps
    return f"https://www.google.com/maps?q={lat},{lon}"


def _format_bigsur_bsslist(value: Any, tzinfo: ZoneInfo | None) -> list[dict[str, Any]]:
    """Format BSSList from Big Sur+ format with location data"""
    bss_list = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                entry: dict[str, Any] = {}
                # Basic BSSID info
                bssid = item.get("BSSID")
                if bssid:
                    entry["bssid"] = bssid
                channel = item.get("Channel")
                if channel is not None:
                    entry["channel"] = channel
                channel_flags = item.get("ChannelFlags")
                if channel_flags is not None:
                    entry["channel_flags"] = channel_flags
                last_assoc = _format_datetime(item.get("LastAssociatedAt"), tzinfo)
                if last_assoc:
                    entry["last_associated"] = last_assoc

                # Extract location if present - CRITICAL FORENSIC DATA
                location = item.get("Location")
                if isinstance(location, dict):
                    lat = location.get("LocationLatitude")
                    lon = location.get("LocationLongitude")
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        loc_data: dict[str, Any] = {
                            "latitude": lat,
                            "longitude": lon,
                        }
                        accuracy = location.get("LocationAccuracy")
                        if accuracy is not None:
                            loc_data["accuracy"] = accuracy
                        loc_timestamp = _format_datetime(location.get("LocationTimestamp"), tzinfo)
                        if loc_timestamp:
                            loc_data["timestamp"] = loc_timestamp
                        # Generate map links for easy geolocation
                        loc_data["map_link_google"] = _generate_map_link(lat, lon, "google")
                        loc_data["map_link_apple"] = _generate_map_link(lat, lon, "apple")
                        entry["location"] = loc_data

                # Add IPv4 network signature if present
                ipv4_sig = item.get("IPv4NetworkSignature")
                if ipv4_sig:
                    entry["ipv4_network_signature"] = ipv4_sig

                # Add DHCP server ID if present
                dhcp_server = item.get("DHCPServerID")
                if isinstance(dhcp_server, bytes) and len(dhcp_server) == 4:
                    # Convert to dot-notation IP
                    entry["dhcp_server_id"] = ".".join(str(b) for b in dhcp_server)

                if entry:  # Only add if we got some data
                    bss_list.append(entry)
    return bss_list


def parse_message_tracer(paths: list[Path], opts: ParserOptions) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            entry: dict[str, Any] = {
                "path": str(path),
                "notes": "too_large",
                "top_keys": meta.get("top_keys", []),
            }
            summaries.append(entry)
            if opts.collector:
                opts.collector.add(
                    "wifi_message_tracer",
                    vfs_path=str(path),
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue
        entry = {"path": str(path)}
        assoc_map = data.get("AssociationSSIDMap")
        if isinstance(assoc_map, dict):
            entry["association_counts"] = _decode_ssid_map(assoc_map)
        roam_map = data.get("RoamSSIDMap")
        if isinstance(roam_map, dict):
            entry["roam_counts"] = _decode_ssid_map(roam_map)
        fail_map = data.get("FailedJoinSSIDMap")
        if isinstance(fail_map, dict):
            entry["failed_join_counts"] = _decode_ssid_map(fail_map)
        summaries.append(entry)
        if opts.collector:
            for record in entry.get("association_counts", []):
                opts.collector.add(
                    "message_tracer_association",
                    vfs_path=str(path),
                    ssid=record.get("ssid"),
                    notes=f"association_count={record.get('count')}",
                )
            for record in entry.get("roam_counts", []):
                opts.collector.add(
                    "message_tracer_roam",
                    vfs_path=str(path),
                    ssid=record.get("ssid"),
                    notes=f"roam_count={record.get('count')}",
                )
            for record in entry.get("failed_join_counts", []):
                opts.collector.add(
                    "message_tracer_failed_join",
                    vfs_path=str(path),
                    ssid=record.get("ssid"),
                    notes=f"failed_join_count={record.get('count')}",
                )
    return summaries


def _decode_ssid_map(mapping: dict[Any, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in mapping.items():
        ssid = decode_hex_token(key) if isinstance(key, str) else str(key)
        out.append({"ssid": ssid, "count": value})
    return out


def extract_mac_from_client_identifier(client_id: bytes) -> str | None:
    """Extract MAC address from DHCP ClientIdentifier field.

    The ClientIdentifier field format (RFC 2132):
    - Byte 0: Hardware type (0x01 = Ethernet)
    - Bytes 1-6: Hardware address (MAC for Ethernet)

    Args:
        client_id: Raw ClientIdentifier bytes from DHCP plist

    Returns:
        Colon-separated MAC address string, or None if invalid
    """
    if len(client_id) >= 7 and client_id[0] == 0x01:  # Ethernet type
        return ":".join(f"{b:02x}" for b in client_id[1:7])
    return None


def parse_dhcp_packet(packet_data: bytes) -> dict[str, Any]:
    """Parse a DHCP packet and extract options.

    Args:
        packet_data: Raw DHCP packet bytes (BOOTP format)

    Returns:
        Dictionary containing parsed DHCP options
    """
    result: dict[str, Any] = {}

    try:
        # DHCP packet structure (RFC 2131):
        # 0-235: BOOTP header
        # 236+: DHCP options (starts with magic cookie 0x63825363)

        if len(packet_data) < 240:
            return result

        # Extract chaddr (client hardware address) from BOOTP header
        # Bytes 28-33 contain the client MAC address (for Ethernet)
        hlen = packet_data[2]  # hardware address length
        if hlen >= 6 and len(packet_data) >= 34:
            chaddr = packet_data[28:34]
            result["chaddr"] = ":".join(f"{b:02x}" for b in chaddr)

        # Check for magic cookie at offset 236
        magic_cookie = packet_data[236:240]
        if magic_cookie != b"\x63\x82\x53\x63":
            return result

        # Parse DHCP options (Type-Length-Value format)
        offset = 240
        while offset < len(packet_data):
            option_type = packet_data[offset]

            # Option 255 = End of options
            if option_type == 255:
                break

            # Option 0 = Pad
            if option_type == 0:
                offset += 1
                continue

            # All other options have a length field
            if offset + 1 >= len(packet_data):
                break

            option_length = packet_data[offset + 1]
            if offset + 2 + option_length > len(packet_data):
                break

            option_data = packet_data[offset + 2 : offset + 2 + option_length]

            # Parse common DHCP options
            if option_type == 1 and option_length == 4:
                # Subnet Mask
                result["subnet_mask"] = ".".join(str(b) for b in option_data)

            elif option_type == 3 and option_length >= 4:
                # Router/Gateway (can be multiple)
                routers = []
                for i in range(0, option_length, 4):
                    if i + 4 <= option_length:
                        routers.append(".".join(str(b) for b in option_data[i : i + 4]))
                result["routers"] = routers

            elif option_type == 6 and option_length >= 4:
                # DNS Servers (can be multiple)
                dns_servers = []
                for i in range(0, option_length, 4):
                    if i + 4 <= option_length:
                        dns_servers.append(".".join(str(b) for b in option_data[i : i + 4]))
                result["dns_servers"] = dns_servers

            elif option_type == 15 and option_length > 0:
                # Domain Name
                with contextlib.suppress(Exception):
                    result["domain_name"] = option_data.decode("ascii", errors="ignore")

            elif option_type == 28 and option_length == 4:
                # Broadcast Address
                result["broadcast_address"] = ".".join(str(b) for b in option_data)

            elif option_type == 42 and option_length >= 4:
                # NTP Servers
                ntp_servers = []
                for i in range(0, option_length, 4):
                    if i + 4 <= option_length:
                        ntp_servers.append(".".join(str(b) for b in option_data[i : i + 4]))
                result["ntp_servers"] = ntp_servers

            elif option_type == 51 and option_length == 4:
                # IP Address Lease Time (seconds)
                lease_time = struct.unpack(">I", option_data)[0]
                result["lease_time_option"] = lease_time

            elif option_type == 53 and option_length == 1:
                # DHCP Message Type
                msg_types = {
                    1: "DISCOVER",
                    2: "OFFER",
                    3: "REQUEST",
                    4: "DECLINE",
                    5: "ACK",
                    6: "NAK",
                    7: "RELEASE",
                    8: "INFORM",
                }
                result["dhcp_message_type"] = msg_types.get(option_data[0], f"Unknown({option_data[0]})")

            elif option_type == 54 and option_length == 4:
                # DHCP Server Identifier
                result["dhcp_server_id"] = ".".join(str(b) for b in option_data)

            elif option_type == 58 and option_length == 4:
                # Renewal Time (T1)
                renewal_time = struct.unpack(">I", option_data)[0]
                result["renewal_time"] = renewal_time

            elif option_type == 59 and option_length == 4:
                # Rebinding Time (T2)
                rebinding_time = struct.unpack(">I", option_data)[0]
                result["rebinding_time"] = rebinding_time

            elif option_type == 60 and option_length > 0:
                # Vendor Class Identifier
                with contextlib.suppress(Exception):
                    result["vendor_class"] = option_data.decode("ascii", errors="ignore")

            offset += 2 + option_length

    except Exception:
        # Return whatever we managed to parse
        pass

    return result


def parse_dhcp_leases(
    paths: list[Path],
    opts: ParserOptions,
    interface_map: dict[str, str] | None = None,
    wifi_interface: str | None = None,
) -> list[dict[str, Any]]:
    """Parse DHCP lease plist files.

    Args:
        paths: List of DHCP lease plist file paths
        opts: Parser options
        interface_map: Optional mapping of MAC address (lowercase) to BSD name
            Used to resolve interface from client MAC address
        wifi_interface: Optional single WiFi interface name (e.g., "en1")
            Used as fallback for leases with SSID/NetworkID when MAC lookup fails

    Returns:
        List of parsed DHCP lease dictionaries
    """
    leases: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            # For too_large files, try filename-based interface extraction
            filename_parts = path.name.split(",")
            fallback_interface = filename_parts[0] if len(filename_parts) >= 2 else None
            leases.append(
                {
                    "path": str(path),
                    "interface": fallback_interface,
                    "notes": "too_large",
                    "top_keys": meta.get("top_keys", []),
                }
            )
            if opts.collector:
                opts.collector.add(
                    "dhcp_lease",
                    vfs_path=str(path),
                    host_interface=fallback_interface,
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue

        lease_start = data.get("LeaseStartDate")
        lease_length = data.get("LeaseLength")
        lease_end = None
        if isinstance(lease_start, datetime) and isinstance(lease_length, int):
            lease_end = lease_start + timedelta(seconds=lease_length)
        lease_start_str = _format_datetime(lease_start, opts.tzinfo)
        lease_end_str = _format_datetime(lease_end, opts.tzinfo)

        # Parse PacketData if available
        packet_info: dict[str, Any] = {}
        packet_data = data.get("PacketData")
        if isinstance(packet_data, bytes):
            # PacketData is already bytes from plist
            packet_info = parse_dhcp_packet(packet_data)
        elif isinstance(packet_data, str):
            # In case it's base64 string (shouldn't happen with plistlib, but be safe)
            try:
                decoded = base64.b64decode(packet_data)
                packet_info = parse_dhcp_packet(decoded)
            except Exception:
                pass

        # Resolve interface using priority chain:
        # 1. ClientIdentifier MAC → interface_map lookup
        # 2. chaddr from PacketData → interface_map lookup
        # 3. WiFi fallback: if lease has SSID/NetworkID and wifi_interface is known
        # 4. Filename fallback
        interface: str | None = None
        client_mac: str | None = None

        # 1. Try ClientIdentifier (modern macOS, Sequoia+)
        client_id = data.get("ClientIdentifier")
        if isinstance(client_id, bytes):
            client_mac = extract_mac_from_client_identifier(client_id)
            if client_mac and interface_map:
                interface = interface_map.get(client_mac.lower())

        # 2. Try chaddr from PacketData
        if not interface and packet_info.get("chaddr") and interface_map:
            chaddr = packet_info["chaddr"]
            if not client_mac:
                client_mac = chaddr
            interface = interface_map.get(chaddr.lower())

        # 3. WiFi fallback: if lease has SSID/NetworkID (indicating WiFi) and we know the WiFi interface
        # This handles cases where ClientIdentifier is a CachedPrivateMACAddress that doesn't match
        if not interface and (data.get("SSID") or data.get("NetworkID")) and wifi_interface:
            interface = wifi_interface

        # 4. Filename fallback
        if not interface:
            filename = path.stem  # Get filename without extension (e.g., "en0" from "en0.plist")
            # Check for direct interface name (e.g., "en0.plist", "en1.plist")
            if filename.startswith("en") and filename[2:].isdigit():
                interface = filename
            else:
                # Check for "en0,MAC/format" or "en1-1,MAC/format" style
                filename_parts = path.name.split(",")
                if len(filename_parts) >= 2 and filename_parts[0].startswith("en"):
                    # Strip "-N" suffix (e.g., "en0-1" → "en0", "en1-1" → "en1")
                    iface_part = filename_parts[0]
                    if "-" in iface_part:
                        iface_part = iface_part.split("-")[0]
                    interface = iface_part

        # Build lease entry with both plist fields and parsed packet options
        lease_entry: dict[str, Any] = {
            "path": str(path),
            "interface": interface,
            "client_mac": client_mac,
            "ssid": data.get("SSID"),
            "ip_address": data.get("IPAddress"),
            "router_ip": data.get("RouterIPAddress"),
            "router_mac": format_bytes_as_mac(data.get("RouterHardwareAddress")),
            "lease_start": lease_start_str,
            "lease_end": lease_end_str,
            "lease_length_seconds": lease_length,
        }

        # Add parsed DHCP options if available
        if packet_info:
            # Add DNS servers as comma-separated string for display
            if "dns_servers" in packet_info:
                lease_entry["dns_servers"] = ", ".join(packet_info["dns_servers"])

            # Add other useful fields
            for key in ["subnet_mask", "domain_name", "dhcp_server_id", "vendor_class"]:
                if key in packet_info:
                    lease_entry[key] = packet_info[key]

        leases.append(lease_entry)
        if opts.collector:
            opts.collector.add(
                "dhcp_lease",
                vfs_path=str(path),
                host_interface=interface,
                host_mac=client_mac,
                timestamp=lease_start_str,
                ssid=data.get("SSID"),
                bssid=None,
                ip_address=data.get("IPAddress"),
                router_ip=data.get("RouterIPAddress"),
                router_mac=format_bytes_as_mac(data.get("RouterHardwareAddress")),
                lease_start=lease_start_str,
                lease_end=lease_end_str,
            )
    return leases


def parse_network_interfaces(paths: list[Path], opts: ParserOptions) -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            entry = {
                "path": str(path),
                "notes": "too_large",
                "top_keys": meta.get("top_keys", []),
            }
            interfaces.append(entry)
            if opts.collector:
                opts.collector.add(
                    "network_interface",
                    vfs_path=str(path),
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("Interfaces")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                entry = {
                    "path": str(path),
                    "bsd_name": item.get("BSD Name"),
                    "type": item.get("SCNetworkInterfaceType"),
                    "user_name": (item.get("SCNetworkInterfaceInfo") or {}).get("UserDefinedName"),
                    "mac_address": format_bytes_as_mac(item.get("IOMACAddress")),
                    "built_in": bool(item.get("IOBuiltin")),
                }
                interfaces.append(entry)
                if opts.collector:
                    opts.collector.add(
                        "network_interface",
                        vfs_path=str(path),
                        host_interface=entry.get("bsd_name"),
                        host_mac=entry.get("mac_address"),
                        notes=_format_notes(
                            [
                                f"type={entry.get('type')}",
                                f"user_name={entry.get('user_name')}",
                                f"built_in={entry.get('built_in')}",
                            ]
                        ),
                    )
    return interfaces


def parse_network_preferences(paths: list[Path], opts: ParserOptions) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            services.append(
                {
                    "path": str(path),
                    "notes": "too_large",
                    "top_keys": meta.get("top_keys", []),
                }
            )
            if opts.collector:
                opts.collector.add(
                    "network_service",
                    vfs_path=str(path),
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue
        network_services = data.get("NetworkServices")
        if not isinstance(network_services, dict):
            continue
        for service_id, service in network_services.items():
            if not isinstance(service, dict):
                continue
            interface = service.get("Interface") or {}
            services.append(
                {
                    "path": str(path),
                    "service_id": service_id,
                    "name": service.get("UserDefinedName"),
                    "hardware": interface.get("Hardware"),
                    "bsd_device": interface.get("DeviceName"),
                    "type": interface.get("Type"),
                    "ipv4": service.get("IPv4"),
                    "proxies": service.get("Proxies"),
                }
            )
            if opts.collector:
                ipv4 = service.get("IPv4") or {}
                ip_address = None
                if isinstance(ipv4, dict):
                    addresses = ipv4.get("Addresses")
                    if isinstance(addresses, list) and addresses:
                        ip_address = addresses[0]
                    elif isinstance(addresses, str):
                        ip_address = addresses
                opts.collector.add(
                    "network_service",
                    vfs_path=str(path),
                    host_interface=(interface.get("DeviceName") if isinstance(interface, dict) else None),
                    ip_address=ip_address,
                    notes=_format_notes(
                        [
                            f"name={service.get('UserDefinedName')}",
                            f"hardware={interface.get('Hardware') if isinstance(interface, dict) else None}",
                            f"type={interface.get('Type') if isinstance(interface, dict) else None}",
                        ]
                    ),
                )
    return services


def parse_eapol(paths: list[Path], opts: ParserOptions) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in paths:
        data, meta = safe_load_plist(path, max_bytes=opts.max_plist_bytes)
        if meta.get("notes") == "too_large":
            profiles.append(
                {
                    "path": str(path),
                    "notes": "too_large",
                    "top_keys": meta.get("top_keys", []),
                }
            )
            if opts.collector:
                opts.collector.add(
                    "eapol_profile",
                    vfs_path=str(path),
                    notes="too_large",
                )
            continue
        if not isinstance(data, dict):
            continue
        system_profiles = data.get("SystemProfiles")
        if isinstance(system_profiles, list):
            for profile in system_profiles:
                if isinstance(profile, dict):
                    entry = {"path": str(path), **profile}
                    profiles.append(entry)
                    if opts.collector:
                        opts.collector.add(
                            "eapol_profile",
                            vfs_path=str(path),
                            notes=_format_notes(
                                [
                                    f"id={profile.get('UniqueIdentifier')}",
                                    f"ssid={profile.get('SSIDString')}",
                                ]
                            ),
                        )
        elif data:
            entry = {"path": str(path), "keys": list(data.keys())}
            profiles.append(entry)
            if opts.collector:
                opts.collector.add(
                    "eapol_profile",
                    vfs_path=str(path),
                    notes="keys_only",
                )
    return profiles


def _format_datetime(value: Any, tzinfo: ZoneInfo | None = None) -> str | None:
    if not isinstance(value, datetime):
        return None
    dt = value
    if tzinfo:
        dt = dt.astimezone(tzinfo) if dt.tzinfo else dt.replace(tzinfo=tzinfo)
        return dt.isoformat()
    if dt.tzinfo:
        dt = dt.astimezone(UTC)
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _parse_datetime_string(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def find_files(root: Path | None, pattern: str) -> list[Path]:
    if not root or not root.exists():
        return []
    return [p for p in root.rglob(pattern) if "provenance" not in p.name.lower()]


def compute_file_provenance(path: Path) -> dict[str, Any]:
    """Compute provenance metadata for a file (SHA256 hash, size, mtime).

    Args:
        path: Path to the file

    Returns:
        Dictionary with provenance fields:
        - sha256: Hex-encoded SHA256 hash of file contents
        - size_bytes: File size in bytes
        - mtime_iso: Last modification time as ISO timestamp
        - path: Original file path as string
    """
    provenance: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        provenance["size_bytes"] = stat.st_size
        provenance["mtime_iso"] = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()

        # Calculate SHA256 hash
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        provenance["sha256"] = sha256.hexdigest()
    except OSError as e:
        provenance["error"] = str(e)
    return provenance


def build_presence_rollups(
    leases: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    window_seconds: int,
    tzinfo: ZoneInfo | None,
) -> list[dict[str, Any]]:
    if not leases or not events:
        return []

    target_events: list[tuple[datetime, dict[str, Any]]] = []
    allowed_sources = PRESENCE_EVENT_SOURCES
    for event in events:
        if allowed_sources and event.get("source") not in allowed_sources:
            continue
        timestamp = event.get("timestamp")
        dt = _normalize_datetime(_parse_datetime_string(timestamp))
        if not dt:
            continue
        target_events.append((dt, event))

    if not target_events:
        return []

    target_events.sort(key=lambda item: item[0])
    event_times = [dt for dt, _ in target_events]
    window = timedelta(seconds=max(window_seconds, 0))

    rollup_map: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}

    for lease in leases:
        lease_start = _normalize_datetime(_parse_datetime_string(lease.get("lease_start")))
        if not lease_start:
            continue
        window_start = lease_start - window
        window_end = lease_start + window
        start_idx = bisect_left(event_times, window_start)
        end_idx = bisect_right(event_times, window_end)
        if start_idx == end_idx:
            continue
        lease_id = lease.get("path") or f"{lease.get('interface')}@{lease.get('ip_address')}"
        for idx in range(start_idx, end_idx):
            event_dt, event = target_events[idx]
            event_local = event_dt if tzinfo is None else event_dt.astimezone(tzinfo)
            date_key = event_local.date().isoformat()
            oui = event.get("oui") or ""
            bssid = (event.get("bssid") or "").lower() or None
            rssi = event.get("rssi")
            for bssid_key in ({bssid} if bssid else set()) | {None}:
                key = (date_key, oui or None, bssid_key)
                rollup = rollup_map.setdefault(
                    key,
                    {
                        "date": date_key,
                        "oui": oui,
                        "bssid": bssid_key or "",
                        "leases": set(),
                        "sightings": 0,
                        "bssid_values": set(),
                        "rssi_values": [],
                    },
                )
                rollup["sightings"] += 1
                if lease_id:
                    rollup["leases"].add(lease_id)
                if bssid:
                    rollup["bssid_values"].add(bssid)
                if isinstance(rssi, int):
                    rollup["rssi_values"].append(rssi)

    rows: list[dict[str, Any]] = []
    for key, rollup in rollup_map.items():
        rssi_values = rollup["rssi_values"]
        rows.append(
            {
                "date": rollup["date"],
                "oui": rollup["oui"],
                "bssid": rollup["bssid"],
                "joins": len(rollup["leases"]),
                "sightings": rollup["sightings"],
                "unique_bssids": len(rollup["bssid_values"]),
                "rssi_min": min(rssi_values) if rssi_values else "",
                "rssi_median": median(rssi_values) if rssi_values else "",
                "rssi_max": max(rssi_values) if rssi_values else "",
            }
        )

    rows.sort(key=lambda row: (row["date"], row["oui"] or "", row["bssid"]))
    return rows


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_presence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "date",
        "oui",
        "bssid",
        "joins",
        "sightings",
        "unique_bssids",
        "rssi_min",
        "rssi_median",
        "rssi_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    ex_root: Path,
    *,
    tzinfo: ZoneInfo | None = None,
    max_plist_mb: int = DEFAULT_MAX_PLIST_MB,
    collector: NormalizedEventCollector | None = None,
) -> dict[str, Any]:
    logs_root = ex_root / "logs"
    databases_root = ex_root / "databases"
    caches_root = ex_root / "caches"

    max_bytes = max(int(max_plist_mb), 1) * 1024 * 1024
    opts = ParserOptions(tzinfo=tzinfo, max_plist_bytes=max_bytes, collector=collector)

    summary: dict[str, Any] = {"exemplar_root": str(ex_root)}

    # Parse known networks FIRST - they have full timestamps for year inference
    # This is more reliable than DHCP leases which may have stale entries
    airport_paths = find_files(logs_root, "com.apple.airport.preferences*.plist")
    airport_paths.extend(find_files(logs_root, "com.apple.wifi.known-networks*.plist"))
    summary["airport_known_networks"] = parse_airport_preferences(airport_paths, opts)

    # Extract reference years from known network timestamps (primary source)
    known_network_timestamps: list[str | None] = []
    for network in summary["airport_known_networks"]:
        # Collect all timestamp fields from known networks
        for key in ["added_at", "joined_by_user_at", "joined_by_system_at", "last_connected"]:
            if ts := network.get(key):
                known_network_timestamps.append(ts)
        # Also check channel history timestamps
        for ch in network.get("channel_history") or []:
            if isinstance(ch, dict) and (ts := ch.get("timestamp")):
                known_network_timestamps.append(ts)

    reference_years = extract_years_from_timestamps(known_network_timestamps)

    # Parse network interfaces early - needed for DHCP interface resolution
    ni_paths = find_files(logs_root, "NetworkInterfaces*.plist")
    summary["network_interfaces"] = parse_network_interfaces(ni_paths, opts)

    # Build MAC → BSD name lookup for DHCP interface resolution
    interface_map: dict[str, str] = {}
    wifi_interfaces: list[str] = []
    for iface in summary["network_interfaces"]:
        mac = iface.get("mac_address")
        bsd = iface.get("bsd_name")
        if mac and bsd:
            interface_map[mac.lower()] = bsd
        # Track WiFi interfaces (IEEE80211 type)
        if iface.get("type") == "IEEE80211" and bsd:
            wifi_interfaces.append(bsd)

    # If exactly one WiFi interface, use it for fallback resolution
    # (if multiple WiFi interfaces exist, we can't reliably determine which one)
    wifi_interface = wifi_interfaces[0] if len(wifi_interfaces) == 1 else None

    # Parse DHCP leases (may still be useful as fallback for year inference)
    dhcp_paths: list[Path] = []
    if logs_root.exists():
        for path in logs_root.rglob("*DHCP*/*"):
            if path.is_file() and not path.name.endswith(".provenance.json"):
                dhcp_paths.append(path)
    summary["dhcp_leases"] = parse_dhcp_leases(
        dhcp_paths, opts, interface_map=interface_map, wifi_interface=wifi_interface
    )

    # If no known network timestamps, fall back to DHCP lease timestamps
    if not reference_years:
        dhcp_timestamps = [lease.get("lease_start") for lease in summary["dhcp_leases"]]
        reference_years = extract_years_from_timestamps(dhcp_timestamps)
        if reference_years:
            logger.debug(f"Using DHCP lease years (fallback): {sorted(reference_years)}")
    else:
        logger.debug(f"Using known network years for WiFi log timestamp inference: {sorted(reference_years)}")

    # Update opts with reference_years if we found any
    if reference_years:
        opts = ParserOptions(
            tzinfo=tzinfo,
            max_plist_bytes=max_bytes,
            collector=collector,
            reference_years=reference_years,
        )

    # Find all wifi logs including rotated logs (wifi.log, wifi.log.0, wifi.log.1, etc.)
    wifi_logs = find_files(logs_root, "wifi*.log*")

    # If multiple WiFi log files, merge them chronologically for deduplication
    if len(wifi_logs) > 1:
        logger.debug(f"Merging {len(wifi_logs)} WiFi log files chronologically...")
        # Create temporary merged log file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, prefix="wifi_merged_") as tmp_file:
            merged_path = Path(tmp_file.name)

        try:
            # Merge all logs into single chronologically-ordered file
            wifi_log_merger.merge_wifi_logs(wifi_logs, merged_path, reference_years=reference_years)
            # Parse the merged log
            summary["wifi_logs"] = parse_wifi_logs([merged_path], opts)
        finally:
            # Clean up temp file
            with contextlib.suppress(Exception):
                merged_path.unlink()
    else:
        # Single file or no files - parse directly
        summary["wifi_logs"] = parse_wifi_logs(wifi_logs, opts)

    tracer_paths = find_files(logs_root, "com.apple.wifi.message-tracer*.plist")
    summary["wifi_message_tracer"] = parse_message_tracer(tracer_paths, opts)

    # Network interfaces already parsed earlier for DHCP interface resolution

    pref_paths = find_files(logs_root, "preferences*.plist")
    summary["network_services"] = parse_network_preferences(pref_paths, opts)

    eapol_paths = find_files(logs_root, "com.apple.eapolclient*.plist")
    summary["eapol_profiles"] = parse_eapol(eapol_paths, opts)

    summary["artefact_counts"] = {
        "wifi_logs": len(wifi_logs),
        "airport_preferences": len(airport_paths),
        "message_tracer": len(tracer_paths),
        "dhcp_leases": len(dhcp_paths),
        "network_interfaces": len(ni_paths),
        "network_preferences": len(pref_paths),
        "eapol": len(eapol_paths),
    }

    # Compute file provenance (SHA256, size, mtime) for all parsed files
    all_files = wifi_logs + airport_paths + tracer_paths + dhcp_paths + ni_paths + pref_paths + eapol_paths
    summary["file_provenance"] = [compute_file_provenance(p) for p in all_files]

    _ = databases_root, caches_root

    return summary


def resolve_exemplar(path: Path) -> Path:
    if (path / "logs").exists():
        return path
    if (path / "exemplar" / "logs").exists():
        return path / "exemplar"
    raise SystemExit(f"Unable to locate logs/ under {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Wi-Fi artefacts from an exemplar directory.")
    parser.add_argument("exemplar", type=Path, help="Path to exemplar root or run directory")
    parser.add_argument("--out", type=Path, help="Optional path to write JSON summary")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--oui-map",
        type=Path,
        help=("Optional CSV mapping OUIs to vendor names (defaults to resources/data/oui.csv if available)"),
    )
    parser.add_argument(
        "--presence-window",
        type=int,
        default=900,
        help="Seconds around each DHCP lease start to correlate presence events (default: 900)",
    )
    parser.add_argument(
        "--max-plist-mb",
        type=int,
        default=DEFAULT_MAX_PLIST_MB,
        help="Maximum plist size to fully parse before treating as too large (default: 50)",
    )
    parser.add_argument(
        "--tz",
        type=str,
        help="Optional IANA timezone (e.g., 'UTC', 'America/Los_Angeles') for timestamp output",
    )
    args = parser.parse_args()

    ex_root = resolve_exemplar(args.exemplar)

    tzinfo: ZoneInfo | None = None
    if args.tz:
        try:
            tzinfo = ZoneInfo(args.tz)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid timezone '{args.tz}': {exc}")

    oui_map_path = args.oui_map
    if not oui_map_path and DEFAULT_OUI_MAP_PATH.exists():
        oui_map_path = DEFAULT_OUI_MAP_PATH
    oui_lookup = enrich_with_oui_vendor(oui_map_path)
    collector = NormalizedEventCollector(oui_lookup)

    summary = build_summary(
        ex_root,
        tzinfo=tzinfo,
        max_plist_mb=args.max_plist_mb,
        collector=collector,
    )

    text = json.dumps(summary, indent=2 if args.pretty else None)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output_path = args.out
    else:
        output_path = Path("wifi_summary.json")
        logger.info(text)
    output_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")

    events_path = output_path.with_name("wifi_events.ndjson")
    write_ndjson(events_path, collector.events)

    presence_rows = build_presence_rollups(
        summary.get("dhcp_leases", []),
        collector.events,
        window_seconds=args.presence_window,
        tzinfo=tzinfo,
    )
    presence_path = output_path.with_name("wifi_presence_daily.csv")
    write_presence_csv(presence_path, presence_rows)

    # Generate HTML report directly (no subprocess needed)
    try:
        from mars.report_modules.wifi_report_generator import wifi_report

        sections = wifi_report.build_summary_sections(summary)
        html = wifi_report.compose_html(summary.get("exemplar_root", "Unknown"), sections)

        html_path = output_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"[info] wrote HTML report to {html_path}")
    except Exception as exc:
        logger.info(f"[warn] Unable to generate report HTML: {exc}")


if __name__ == "__main__":
    main()
