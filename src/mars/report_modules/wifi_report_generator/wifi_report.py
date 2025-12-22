#!/usr/bin/env python3
"""Generate an HTML Wi-Fi/network report from wifi_summary JSON output.

Optionally renders additional artefacts when supplied:
  * wifi_events.ndjson (unified normalized Wi-Fi presence events)
  * wifi_presence_daily.csv (daily rollups derived from DHCP leases)
If the optional files are omitted the report falls to defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mars.utils.debug_logger import logger
from mars.utils.platform_utils import get_logo_data_uri


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_bssid(bssid: str | None) -> str | None:
    if not bssid:
        return None
    text = str(bssid).strip().lower()
    if not text:
        return None
    parts = [p.zfill(2) for p in text.split(":") if p]
    if len(parts) != 6:
        hex_only = re.sub(r"[^0-9a-f]", "", text)
        if len(hex_only) == 12:
            parts = [hex_only[i : i + 2] for i in range(0, 12, 2)]
    return ":".join(parts) if len(parts) == 6 else None


def format_laa_flag(bssid: str | None) -> str:
    normalized = normalize_bssid(bssid)
    if not normalized:
        return ""
    try:
        first_byte = int(normalized.split(":")[0], 16)
    except (ValueError, IndexError):
        return ""
    return "Yes" if (first_byte & 0x02) else "No"


def load_ndjson(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def load_csv(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except Exception:  # noqa: BLE001
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return pd.DataFrame(list(reader)).fillna("")


def to_dataframe(records: Any) -> pd.DataFrame:
    if isinstance(records, list) and records:
        return pd.DataFrame(records)
    return pd.DataFrame()


def paginate_table(table_html: str, table_id: str, page_size: int = 20) -> str:
    """Wrap a table in a paginated container with navigation controls."""
    controls = f"""
    <div class="pagination-controls">
      <div class="pagination-nav">
        <button type="button" class="page-first" onclick="firstPage('{table_id}')" title="First page">⟨⟨</button>
        <button type="button" class="page-prev" onclick="prevPage('{table_id}')" title="Previous page">← Prev</button>
        <span class="page-info">Page 1</span>
        <button type="button" class="page-next" onclick="nextPage('{table_id}')" title="Next page">Next →</button>
        <button type="button" class="page-last" onclick="lastPage('{table_id}')" title="Last page">⟩⟩</button>
      </div>
      <div class="pagination-size">
        <label>Rows per page:</label>
        <select onchange="setPageSize('{table_id}', this.value)">
          <option value="20" {"selected" if page_size == 20 else ""}>20</option>
          <option value="50" {"selected" if page_size == 50 else ""}>50</option>
          <option value="100" {"selected" if page_size == 100 else ""}>100</option>
        </select>
      </div>
    </div>
    """
    return f'<div class="table-wrapper" id="{table_id}" data-page-size="{page_size}">{table_html}{controls}</div>'


def card(title: str, content: str, note: str = "") -> str:
    """Wrap section content in a styled card container."""
    note_html = f'<p class="section-note">{note}</p>' if note else ""
    return f'<div class="card"><h2 class="card-title">{title}</h2>{note_html}{content}</div>'


def compute_recent_lease_map(dhcp_df: pd.DataFrame) -> dict[str, str]:
    if dhcp_df.empty:
        return {}
    leases = dhcp_df.copy()
    leases["lease_start"] = pd.to_datetime(leases["lease_start"], errors="coerce", format="mixed")
    leases = leases.dropna(subset=["lease_start"]).sort_values("lease_start")
    leases["lease_summary"] = leases.apply(
        lambda row: (f"{row.get('interface')} @ {row.get('router_ip')} ({row.get('lease_start')})"),
        axis=1,
    )
    best_by_interface: dict[str, str] = {}
    for _, row in leases.iterrows():
        iface = row.get("interface")
        summary = row.get("lease_summary")
        if iface and summary:
            best_by_interface[iface] = summary
    return best_by_interface


def join_known_with_dhcp(
    known_df: pd.DataFrame,
    recent_leases: dict[str, str],
    dhcp_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join known networks with DHCP lease information.

    Args:
        known_df: DataFrame of known networks
        recent_leases: Dict mapping interface name to lease summary string
        dhcp_df: Optional DataFrame of DHCP leases for SSID-based fallback join

    Returns:
        known_df with recent_lease column added
    """
    known_df = known_df.copy()

    if known_df.empty:
        if "recent_lease" not in known_df.columns:
            known_df["recent_lease"] = "None recorded"
        return known_df

    # Initialize recent_lease column
    known_df["recent_lease"] = "None recorded"

    # First pass: interface-based join
    if "interface" in known_df.columns and recent_leases:
        known_df["recent_lease"] = known_df["interface"].map(recent_leases).fillna("None recorded")

    # Second pass: SSID-based fallback for rows still showing "None recorded"
    # This helps with Big Sur+ format which doesn't have interface in known networks
    if dhcp_df is not None and not dhcp_df.empty and "ssid" in dhcp_df.columns and "ssid" in known_df.columns:
        # Build SSID → lease summary mapping from DHCP leases
        dhcp_with_ssid = dhcp_df[dhcp_df["ssid"].notna()].copy()
        if not dhcp_with_ssid.empty:
            # Parse timestamps and sort to get most recent lease per SSID
            dhcp_with_ssid["lease_start_dt"] = pd.to_datetime(
                dhcp_with_ssid["lease_start"], errors="coerce", format="mixed"
            )
            dhcp_with_ssid = dhcp_with_ssid.dropna(subset=["lease_start_dt"]).sort_values("lease_start_dt")

            # Build SSID → lease summary mapping (last entry wins = most recent)
            ssid_to_lease: dict[str, str] = {}
            for _, row in dhcp_with_ssid.iterrows():
                ssid = row.get("ssid")
                if ssid:
                    # Build summary: interface @ router_ip (timestamp) or just router_ip (timestamp)
                    iface = row.get("interface") or "unknown"
                    router_ip = row.get("router_ip") or "unknown"
                    lease_start = row.get("lease_start") or "unknown"
                    ssid_to_lease[ssid] = f"{iface} @ {router_ip} ({lease_start})"

            # Apply SSID fallback where recent_lease is still "None recorded"
            if ssid_to_lease:
                mask = known_df["recent_lease"] == "None recorded"
                known_df.loc[mask, "recent_lease"] = (
                    known_df.loc[mask, "ssid"].map(ssid_to_lease).fillna("None recorded")
                )

    return known_df


def _format_seamless_ssids(ssid_list: Any) -> str:
    """Format seamless SSID list for display."""
    if not ssid_list or not isinstance(ssid_list, list):
        return ""
    # Filter out empty strings and limit display
    ssids = [s for s in ssid_list if s]
    if not ssids:
        return ""
    if len(ssids) <= 2:
        return "; ".join(ssids)
    return f"{ssids[0]}; {ssids[1]} (+{len(ssids) - 2} more)"


def _format_collocated(group: Any) -> str:
    """Format collocated group for display."""
    if not group:
        return ""
    if isinstance(group, list):
        items = [str(g) for g in group if g]
        if not items:
            return ""
        if len(items) <= 2:
            return "; ".join(items)
        return f"{items[0]}; {items[1]} (+{len(items) - 2} more)"
    return str(group)


def build_known_networks_section(
    df: pd.DataFrame,
    recent_leases: dict[str, str],
    dhcp_df: pd.DataFrame | None = None,
) -> str:
    if df.empty:
        return card("Known Networks", "", "No airport known networks were found.")

    df = join_known_with_dhcp(df.copy(), recent_leases, dhcp_df=dhcp_df)
    if "last_connected" in df.columns:
        df["last_connected"] = pd.to_datetime(df["last_connected"], errors="coerce", format="mixed")

    # Deduplicate: keep most recent entry per SSID+timestamp combo
    # Useful for carved candidates with duplicate fragments
    df = df.drop_duplicates(subset=["ssid", "last_connected"], keep="first")

    # Determine which columns to use based on macOS version
    # Newer variants have add_reason and moving
    # Older versions have passpoint and auto_join
    has_bigsur_fields = "add_reason" in df.columns and "moving" in df.columns and df["add_reason"].notna().any()

    # Sort and limit
    df = df.sort_values("last_connected", ascending=False).head(1000)

    # Format last_connected for display
    df["last_connected_fmt"] = df["last_connected"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("Unknown")  # pyright: ignore[reportAttributeAccessIssue]

    # Build HTML table manually for conditional row styling
    table_html = ['<table class="table table-striped">']
    table_html.append("<thead><tr>")

    # Header row
    headers = ["SSID", "Last Connected", "Security"]
    if has_bigsur_fields:
        # Main row: core fields + timeline + toggle
        # Details row (expandable): Related SSIDs, Co-located, Recent Lease
        headers.extend(
            [
                "Add Reason",
                "Hidden",
                "Moving",
                "Timeline",
                "",  # Toggle button column
            ]
        )
        desc = (
            '<p class="section-note">Top known networks from airport preferences. '
            '"Add Reason" shows Cloud Sync (from other Apple devices) vs WiFi Menu (this device).<br />'
            "Click ▶ to expand details including Related SSIDs, Co-located networks, and Recent Lease.</p>"
        )
    else:
        headers.extend(["Passpoint", "Auto Join", "Hidden"])
        desc = '<p class="section-note">Top known networks from airport preferences.</p>'
        headers.append("Recent Lease")

    for h in headers:
        table_html.append(f"<th>{h}</th>")
    table_html.append("</tr></thead>")

    # Body rows
    table_html.append("<tbody>")
    for _, row in df.iterrows():
        # SSID (used in multiple places)
        ssid = row.get("ssid", "")

        if has_bigsur_fields:
            # Big Sur+ uses expandable rows
            # Determine row class based on add_reason
            add_reason = row.get("add_reason", "")
            cloud_class = " cloud-sync-row" if add_reason == "Cloud Sync" else ""
            table_html.append(f'<tr class="known-main{cloud_class}">')

            # Main row cells: SSID, Last Connected, Security, Add Reason, Hidden, Moving, Timeline, Toggle
            table_html.append(f"<td>{ssid}</td>")

            last_conn = row.get("last_connected_fmt", "Unknown")
            table_html.append(f"<td>{last_conn}</td>")

            security = row.get("security_type", "")
            table_html.append(f"<td>{security}</td>")

            table_html.append(f"<td>{add_reason}</td>")

            hidden = "Yes" if row.get("hidden") else "No"
            table_html.append(f"<td>{hidden}</td>")

            moving = "Yes" if row.get("moving") else "No"
            table_html.append(f"<td>{moving}</td>")

            # Timeline - build data for modal display
            def _clean_val(val: Any) -> str:
                """Clean a value, returning empty string for nan/None."""
                if val is None:
                    return ""
                if isinstance(val, float) and pd.isna(val):
                    return ""
                s = str(val)
                if s.lower() in ("nan", "none", "nat"):
                    return ""
                return s

            joined_user = _clean_val(row.get("joined_by_user_at"))
            joined_system = _clean_val(row.get("joined_by_system_at"))
            last_discovered = _clean_val(row.get("last_discovered_at"))
            disconnect_ts = _clean_val(row.get("last_disconnect_timestamp"))
            disconnect_reason = _clean_val(row.get("last_disconnect_reason"))

            has_timeline = any([joined_user, joined_system, last_discovered, disconnect_ts])
            if has_timeline:
                timeline_data = {
                    "ssid": ssid,
                    "joined_user": joined_user,
                    "joined_system": joined_system,
                    "last_discovered": last_discovered,
                    "disconnect_ts": disconnect_ts,
                    "disconnect_reason": disconnect_reason,
                }
                timeline_json = json.dumps(timeline_data).replace('"', "&quot;")
                timeline_cell = (
                    f'<a href="#" onclick="showTimeline(this, event); return false;" '
                    f'data-timeline="{timeline_json}" style="color: #0066cc;">View</a>'
                )
            else:
                timeline_cell = ""
            table_html.append(f"<td>{timeline_cell}</td>")

            # Toggle button
            table_html.append('<td><button class="details-toggle" onclick="toggleKnownDetails(this)">▶</button></td>')
            table_html.append("</tr>")

            # Details row (hidden by default) with Related SSIDs, Co-located, Recent Lease
            seamless = _format_seamless_ssids(row.get("seamless_ssid_list"))
            collocated = _format_collocated(row.get("collocated_group"))
            recent_lease = row.get("recent_lease", "None recorded")

            table_html.append('<tr class="known-details hidden">')
            table_html.append('<td colspan="8">')
            table_html.append('<div class="detail-grid">')
            table_html.append(f"<div><strong>Related SSIDs</strong><br>{seamless or '—'}</div>")
            table_html.append(f"<div><strong>Co-located</strong><br>{collocated or '—'}</div>")
            table_html.append(f"<div><strong>Recent Lease</strong><br>{recent_lease}</div>")
            table_html.append("</div></td>")
            table_html.append("</tr>")
        else:
            # Legacy format (pre-Big Sur) - no expandable rows
            table_html.append("<tr>")

            table_html.append(f"<td>{ssid}</td>")

            last_conn = row.get("last_connected_fmt", "Unknown")
            table_html.append(f"<td>{last_conn}</td>")

            security = row.get("security_type", "")
            table_html.append(f"<td>{security}</td>")

            # Passpoint
            passpoint = "Yes" if row.get("passpoint") else "No"
            table_html.append(f"<td>{passpoint}</td>")

            # Auto Join
            auto_join = row.get("auto_join")
            auto_join_str = "Yes" if auto_join else ("No" if auto_join is False else "")
            table_html.append(f"<td>{auto_join_str}</td>")

            # Hidden
            hidden = "Yes" if row.get("hidden") else "No"
            table_html.append(f"<td>{hidden}</td>")

            # Recent Lease (in main row for legacy)
            recent_lease = row.get("recent_lease", "None recorded")
            table_html.append(f"<td>{recent_lease}</td>")

            table_html.append("</tr>")

    table_html.append("</tbody></table>")
    table_str = "\n".join(table_html)
    table_str = paginate_table(table_str, "table-known-networks")
    # Extract note text from the desc paragraph
    note = desc.replace('<p class="section-note">', "").replace("</p>", "")
    return card("Known Networks", table_str, note)


def build_bssid_section(
    known_df: pd.DataFrame, events_df: pd.DataFrame
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build BSSID cross-reference section.

    Returns:
        Tuple of (html, log_stats, locations) where:
        - log_stats maps BSSID -> metadata
        - locations is a list of dicts with GPS coordinates for map rendering
    """
    if known_df.empty:
        return (
            card("BSSID Activity", "", "No airport BSSID lists were available to cross-reference."),
            {},
            [],
        )

    rows: list[dict[str, Any]] = []
    log_stats: dict[str, dict[str, Any]] = {}
    # Build location lookup from bss_list (Big Sur+ format with GPS)
    bssid_locations: dict[str, dict[str, Any]] = {}
    for _, row in known_df.iterrows():
        bss_list = row.get("bss_list") or []
        ssid = row.get("ssid", "")
        if isinstance(bss_list, list):
            for bss_entry in bss_list:
                if isinstance(bss_entry, dict):
                    bssid = bss_entry.get("bssid")
                    if bssid:
                        normalized = normalize_bssid(bssid)
                        if normalized and "location" in bss_entry:
                            loc = bss_entry["location"]
                            if isinstance(loc, dict) and loc.get("latitude") and loc.get("longitude"):
                                bssid_locations[normalized] = {
                                    "ssid": ssid,
                                    "bssid": normalized,
                                    "latitude": loc.get("latitude"),
                                    "longitude": loc.get("longitude"),
                                    "accuracy": loc.get("accuracy"),
                                    "timestamp": loc.get("timestamp"),
                                    "map_link_google": loc.get("map_link_google"),
                                }
    if not events_df.empty and "bssid" in events_df.columns:
        temp = events_df.copy()
        temp["bssid"] = temp["bssid"].apply(normalize_bssid)
        temp = temp.dropna(subset=["bssid"])
        if not temp.empty:
            # Parse timestamps for sorting (normalize to UTC)
            temp["timestamp_dt"] = pd.to_datetime(temp["timestamp"], errors="coerce", format="mixed", utc=True)
            # Sort by timestamp so first_seen/last_seen are chronological
            temp = temp.sort_values("timestamp_dt", na_position="last")

            for bssid, group in temp.groupby("bssid", sort=False):
                bssid_str = str(bssid)
                vendor = None
                oui = None
                band_val = None
                channel_val = None
                laa_val = None
                if "vendor" in group.columns:
                    vendor_series = group["vendor"].dropna()
                    if not vendor_series.empty:
                        vendor = str(vendor_series.iloc[0])
                if "oui" in group.columns:
                    oui_series = group["oui"].dropna()
                    if not oui_series.empty:
                        oui = str(oui_series.iloc[0])
                if not oui and bssid_str:
                    oui = ":".join(bssid_str.split(":")[:3])
                if "band" in group.columns:
                    band_series = group["band"].dropna()
                    if not band_series.empty:
                        band_val = str(band_series.iloc[0])
                if "channel" in group.columns:
                    channel_series = group["channel"].dropna()
                    if not channel_series.empty:
                        channel_val = str(channel_series.iloc[0])
                if "laa" in group.columns:
                    laa_series = group["laa"].dropna()
                    if not laa_series.empty:
                        laa_val = str(laa_series.iloc[0])
                if not laa_val:
                    laa_val = format_laa_flag(bssid_str)

                # Get chronological first/last timestamps (already sorted above)
                # Store both raw string and parsed datetime for later comparison
                first_ts_str = group["timestamp"].iloc[0]
                last_ts_str = group["timestamp"].iloc[-1]
                first_ts_dt = group["timestamp_dt"].iloc[0]
                last_ts_dt = group["timestamp_dt"].iloc[-1]

                log_stats[bssid_str] = {
                    "log_count": len(group),
                    "first_seen": first_ts_str,
                    "last_seen": last_ts_str,
                    "first_seen_dt": first_ts_dt,
                    "last_seen_dt": last_ts_dt,
                    "vendor": vendor,
                    "oui": oui,
                    "band": band_val,
                    "channel": channel_val,
                    "laa": laa_val,
                }

    for _, row in known_df.iterrows():
        ssid = row.get("ssid")
        bssid_list = row.get("bssid_list") or []

        # Extract ChannelHistory timestamps (these have full dates unlike WiFi logs)
        ch_timestamps = []
        channel_history = row.get("channel_history") or []
        if isinstance(channel_history, list):
            for ch_entry in channel_history:
                if isinstance(ch_entry, dict):
                    ts = ch_entry.get("timestamp")
                    if ts:
                        ch_timestamps.append(ts)

        # Parse ChannelHistory timestamps (normalize to UTC)
        ch_timestamps_dt = []
        for ts in ch_timestamps:
            parsed = pd.to_datetime(ts, errors="coerce", format="mixed", utc=True)
            if pd.notna(parsed):
                ch_timestamps_dt.append(parsed)

        if isinstance(bssid_list, list):
            for bssid in bssid_list:
                normalized = normalize_bssid(bssid)
                if not normalized:
                    continue
                stats = log_stats.get(normalized)

                # Track SSID(s) associated with this BSSID
                if normalized not in log_stats:
                    log_stats[normalized] = {}
                if "ssids" not in log_stats[normalized]:
                    log_stats[normalized]["ssids"] = []
                if ssid and ssid not in log_stats[normalized]["ssids"]:
                    log_stats[normalized]["ssids"].append(ssid)

                # Merge WiFi log timestamps with ChannelHistory timestamps
                # Use the earliest first_seen and latest last_seen across both sources
                first_seen_str = (stats or {}).get("first_seen")
                last_seen_str = (stats or {}).get("last_seen")
                first_seen_dt = (stats or {}).get("first_seen_dt")
                last_seen_dt = (stats or {}).get("last_seen_dt")

                # Consider ChannelHistory timestamps
                all_timestamps = []
                if pd.notna(first_seen_dt):
                    all_timestamps.append(first_seen_dt)
                if pd.notna(last_seen_dt):
                    all_timestamps.append(last_seen_dt)
                all_timestamps.extend(ch_timestamps_dt)

                # Find earliest and latest across all sources
                if all_timestamps:
                    earliest = min(all_timestamps)
                    latest = max(all_timestamps)

                    # Update first/last seen if ChannelHistory has earlier/later dates
                    if pd.notna(first_seen_dt):
                        if earliest < first_seen_dt:
                            first_seen_str = earliest.strftime("%Y-%m-%d %H:%M:%S")
                            first_seen_dt = earliest
                    else:
                        # No WiFi log data, use ChannelHistory
                        first_seen_str = earliest.strftime("%Y-%m-%d %H:%M:%S")
                        first_seen_dt = earliest

                    if pd.notna(last_seen_dt):
                        if latest > last_seen_dt:
                            last_seen_str = latest.strftime("%Y-%m-%d %H:%M:%S")
                            last_seen_dt = latest
                    else:
                        # No WiFi log data, use ChannelHistory
                        last_seen_str = latest.strftime("%Y-%m-%d %H:%M:%S")
                        last_seen_dt = latest

                # Get location data if available
                loc_data = bssid_locations.get(normalized)
                location_link = ""
                if loc_data and loc_data.get("map_link_google"):
                    accuracy = loc_data.get("accuracy")
                    acc_str = f"±{int(accuracy)}m" if accuracy else ""
                    location_link = (
                        f'<a href="{loc_data["map_link_google"]}" target="_blank" class="map-link">View {acc_str}</a>'
                    )

                rows.append(
                    {
                        "ssid": ssid,
                        "bssid": normalized,
                        "log_count": (stats or {}).get("log_count", 0),
                        "first_seen": first_seen_str,
                        "last_seen": last_seen_str,
                        "vendor": (stats or {}).get("vendor"),
                        "oui": (stats or {}).get("oui") or ":".join(normalized.split(":")[:3]),
                        "band": (stats or {}).get("band"),
                        "channel": (stats or {}).get("channel"),
                        "laa": (stats or {}).get("laa") or format_laa_flag(normalized),
                        "location": location_link,
                    }
                )

    if not rows:
        return (
            card("BSSID Activity", "", "No overlap between airport BSSID entries and parsed Wi-Fi logs."),
            log_stats,
            list(bssid_locations.values()),
        )

    df = pd.DataFrame(rows)

    # For duplicates (same SSID+BSSID in multiple plists), aggregate to keep:
    # - Latest last_seen timestamp (most recent activity)
    # - Earliest first_seen timestamp (earliest known sighting)
    # - Highest log_count
    # - First non-null value for other fields
    if not df.empty and "last_seen" in df.columns:
        # Convert timestamps for comparison (normalize to UTC)
        df["last_seen_dt_temp"] = pd.to_datetime(df["last_seen"], errors="coerce", format="mixed", utc=True)
        df["first_seen_dt_temp"] = pd.to_datetime(df["first_seen"], errors="coerce", format="mixed", utc=True)

        # Sort by timestamp (desc) so groupby.first() will get the latest
        df = df.sort_values("last_seen_dt_temp", ascending=False, na_position="last")

        # Group by SSID+BSSID and aggregate
        agg_funcs = {
            "log_count": "max",  # Highest count
            "first_seen": "first",  # Keep first non-null (after sorting, this is arbitrary)
            "last_seen": "first",  # Keep first (which is latest after sort)
            "last_seen_dt_temp": "first",  # Keep latest
            "first_seen_dt_temp": "last",  # Keep earliest (last after desc sort)
            "vendor": "first",
            "oui": "first",
            "band": "first",
            "channel": "first",
            "laa": "first",
            "location": "first",  # Keep first location link
        }
        # Only aggregate columns that exist
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

        df = df.groupby(["ssid", "bssid"], as_index=False, dropna=False).agg(agg_funcs)

        # Use the temp datetime column for final sorting
        df = df.rename(columns={"last_seen_dt_temp": "last_seen_dt"})
        if "first_seen_dt_temp" in df.columns:
            df = df.drop(columns=["first_seen_dt_temp"])
    else:
        # Fallback to old logic if no last_seen column
        df = df.drop_duplicates(subset=["ssid", "bssid"], keep="first")
        df["last_seen_dt"] = pd.to_datetime(df["last_seen"], errors="coerce", format="mixed", utc=True)

    df = df.sort_values(["last_seen_dt", "log_count"], ascending=[False, False]).head(1000)
    display = df.assign(
        first_seen=lambda d: d["first_seen"].fillna("Unknown"),
        last_seen=lambda d: d["last_seen"].fillna("Unknown"),
    ).drop(columns=["last_seen_dt"])
    for col in ("vendor", "oui", "laa", "band", "channel", "location"):
        if col not in display.columns:
            display[col] = None
    # Fill empty location values
    display["location"] = display["location"].fillna("")

    # Check if we have any location data to show
    has_location_data = (
        "location" in display.columns and display["location"].astype(str).str.strip().replace("", pd.NA).notna().any()
    )

    # Build collapsible BSSID table with main row + hidden details row (like DHCP)
    table_html = ['<table class="table table-striped bssid-table">']

    # Header row - main info only
    table_html.append("<thead>")
    table_html.append("<tr>")
    table_html.append("<th>SSID</th>")
    table_html.append("<th>BSSID</th>")
    table_html.append("<th>Log Count</th>")
    table_html.append("<th>First Seen</th>")
    table_html.append("<th>Last Seen</th>")
    if has_location_data:
        table_html.append("<th>Location</th>")
    table_html.append("<th></th>")  # Details toggle column
    table_html.append("</tr>")
    table_html.append("</thead>")

    # Body rows - main row + hidden details row per BSSID
    table_html.append("<tbody>")
    for _, row in display.iterrows():
        ssid = row.get("ssid", "")
        bssid = row.get("bssid", "")
        log_count = row.get("log_count", 0)
        first_seen = row.get("first_seen", "Unknown")
        last_seen = row.get("last_seen", "Unknown")
        location = row.get("location", "")

        # Main row - key info
        table_html.append('<tr class="bssid-main">')
        table_html.append(f"<td>{ssid}</td>")
        table_html.append(f"<td>{bssid}</td>")
        table_html.append(f"<td>{log_count}</td>")
        table_html.append(f"<td>{first_seen}</td>")
        table_html.append(f"<td>{last_seen}</td>")
        if has_location_data:
            table_html.append(f"<td>{location}</td>")
        table_html.append('<td><button class="details-toggle" onclick="toggleBssidDetails(this)">▶</button></td>')
        table_html.append("</tr>")

        # Details row - hidden by default
        vendor = row.get("vendor", "") or "Unknown"
        oui = row.get("oui", "") or ""
        laa = row.get("laa", "") or ""
        band = row.get("band", "") or ""
        channel = row.get("channel", "") or ""

        colspan = 7 if has_location_data else 6
        table_html.append('<tr class="bssid-details hidden">')
        table_html.append(f'<td colspan="{colspan}">')
        table_html.append('<div class="detail-grid">')
        table_html.append(f"<div><strong>Vendor:</strong> {vendor}</div>")
        table_html.append(f"<div><strong>OUI:</strong> {oui}</div>")
        table_html.append(f"<div><strong>LAA (Randomized):</strong> {laa}</div>")
        if band:
            table_html.append(f"<div><strong>Band:</strong> {band}</div>")
        if channel:
            table_html.append(f"<div><strong>Channel:</strong> {channel}</div>")
        table_html.append("</div>")
        table_html.append("</td>")
        table_html.append("</tr>")

    table_html.append("</tbody>")
    table_html.append("</table>")

    table = "\n".join(table_html)
    table = paginate_table(table, "table-bssid-cross")

    # Adjust description based on whether location data exists
    if has_location_data:
        note = (
            "Airport BSSID entries with log sightings. "
            "Click ▶ to expand vendor/OUI details. "
            '"Location" shows GPS coordinates captured when connecting (Big Sur+).'
        )
    else:
        note = "Airport BSSID entries with log sightings. Click ▶ to expand vendor/OUI details."
    return (
        card("BSSID Cross-References", table, note),
        log_stats,
        list(bssid_locations.values()),
    )


def build_wifi_events_section(events_df: pd.DataFrame) -> str:
    if events_df.empty or "ssid" not in events_df.columns:
        return card("Wi-Fi Log Activity", "", "No Wi-Fi log or normalized event data was available.")

    counts = (
        events_df["ssid"]
        .dropna()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "ssid"})
    )
    counts.columns = ["ssid", "occurrences"]
    counts = counts.head(1000)

    if counts.empty:
        return card("Wi-Fi Log Activity", "", "No SSID references were detected in the parsed events.")

    table = counts.to_html(index=False, classes="table table-striped")
    table = paginate_table(table, "table-wifi-events")
    note = (
        "Most frequently referenced SSIDs across Wi-Fi logs and normalized events. "
        "<strong>Note:</strong> Carved file (candidate) scans may have inflated occurence counts."
    )
    return card("Wi-Fi Log Activity", table, note)


def _first_non_null(series: pd.Series) -> Any:
    for value in series:
        if pd.notna(value) and value != "":
            return value
    return None


def build_scan_burst_section(events_df: pd.DataFrame, log_stats: dict[str, dict[str, Any]]) -> str:
    """Build scan burst section with modal support for full BSSID lists.

    Args:
        events_df: DataFrame of WiFi events
        log_stats: Dict mapping BSSID -> metadata (vendor, OUI, LAA, timestamps, etc.)
    """
    if events_df.empty or "timestamp" not in events_df.columns or "ssid" not in events_df.columns:
        return card("Scan Burst Summary", "", "No event data available to identify scan bursts.")

    required = events_df.dropna(subset=["timestamp", "ssid", "bssid"], how="any").copy()
    if required.empty:
        return card("Scan Burst Summary", "", "No SSID/BSSID combinations with timestamps were available.")

    required["timestamp"] = pd.to_datetime(required["timestamp"], errors="coerce", format="mixed")
    required = required.dropna(subset=["timestamp"])
    if required.empty:
        return card("Scan Burst Summary", "", "Unable to parse timestamps for Wi-Fi events.")

    required["timestamp_second"] = required["timestamp"].dt.floor("s")  # pyright: ignore[reportAttributeAccessIssue]
    required["bssid_norm"] = required["bssid"].apply(normalize_bssid)
    required = required.dropna(subset=["bssid_norm"])
    if required.empty:
        return card("Scan Burst Summary", "", "No normalized BSSID values were available for scan bursts.")
    if "band" not in required.columns:
        required["band"] = ""
    if "channel" not in required.columns:
        required["channel"] = ""

    def _sample_bssids(series: pd.Series) -> str:
        ordered: list[str] = []
        for value in series.dropna():
            text = normalize_bssid(value)
            if text and text not in ordered:
                ordered.append(text)
            if len(ordered) >= 5:
                break
        return "; ".join(ordered)

    def _all_bssids(series: pd.Series) -> str:
        """Capture ALL unique BSSIDs (not just first 5) for modal display."""
        ordered: list[str] = []
        for value in series.dropna():
            text = normalize_bssid(value)
            if text and text not in ordered:
                ordered.append(text)
        return "; ".join(ordered)

    def _randomized_pct(series: pd.Series) -> int:
        values = [normalize_bssid(v) for v in series if pd.notna(v)]
        values = [v for v in values if v]
        if not values:
            return 0
        randomized = sum(1 for v in values if format_laa_flag(v) == "Yes")
        return int(round(100 * randomized / len(values), 0))

    def _vendor_ouis(series: pd.Series) -> str:
        seen: list[str] = []
        for value in series:
            norm = normalize_bssid(value)
            if not norm:
                continue
            oui = ":".join(norm.split(":")[:3])
            if oui not in seen:
                seen.append(oui)
            if len(seen) >= 5:
                break
        return "; ".join(seen)

    grouped = (
        required.groupby(["ssid", "timestamp_second"], dropna=False)
        .agg(
            distinct_bssid_count=("bssid_norm", lambda s: s.dropna().nunique()),
            sample_bssids=("bssid_norm", _sample_bssids),
            all_bssids=("bssid_norm", _all_bssids),
            randomized_pct=("bssid_norm", _randomized_pct),
            vendor_ouis=("bssid_norm", _vendor_ouis),
            any_band=("band", _first_non_null),
            any_channel=("channel", _first_non_null),
        )
        .reset_index()
    )

    bursts = grouped[grouped["distinct_bssid_count"] >= 2]
    if bursts.empty:
        return card("Scan Burst Summary", "", "No scan bursts (multiple BSSIDs per SSID per second) were detected.")

    bursts = bursts.sort_values(["timestamp_second", "distinct_bssid_count"], ascending=[False, False]).head(1000)
    bursts = bursts.assign(timestamp=bursts["timestamp_second"].dt.strftime("%Y-%m-%d %H:%M:%S"))  # pyright: ignore[reportAttributeAccessIssue]

    # Build BSSID details for modal display
    import json

    def build_bssid_details(all_bssids_str: str) -> str:
        """Build JSON array of BSSID details for modal."""
        if not all_bssids_str:
            return "[]"
        bssids = all_bssids_str.split("; ")
        details = []
        for bssid in bssids:
            if not bssid:
                continue
            stats = log_stats.get(bssid, {})
            # Get known SSID(s) for this BSSID
            ssids = stats.get("ssids", [])
            ssid_str = "; ".join(ssids) if ssids else "Unknown"
            details.append(
                {
                    "bssid": bssid,
                    "ssid": ssid_str,
                    "vendor": stats.get("vendor") or "Unknown",
                    "oui": stats.get("oui") or ":".join(bssid.split(":")[:3]),
                    "laa": stats.get("laa") or format_laa_flag(bssid),
                    "first_seen": stats.get("first_seen") or "Unknown",
                    "last_seen": stats.get("last_seen") or "Unknown",
                }
            )
        return json.dumps(details)

    bursts["bssid_details_json"] = bursts["all_bssids"].apply(build_bssid_details)

    display = bursts[
        [
            "timestamp",
            "ssid",
            "distinct_bssid_count",
            "sample_bssids",
            "all_bssids",
            "bssid_details_json",
            "randomized_pct",
            "vendor_ouis",
            "any_band",
            "any_channel",
        ]
    ]
    display = display.fillna("")

    # Build table HTML manually to add data attributes for modal
    table_rows = []
    for _, row in display.iterrows():
        # Escape JSON for HTML attribute
        details_json = row["bssid_details_json"].replace('"', "&quot;").replace("'", "&#39;")
        show_count = int(row["distinct_bssid_count"])

        # Build sample_bssids cell with "Show all" link if >5 BSSIDs
        if show_count > 5:
            sample_cell = f"""{row["sample_bssids"]}<br>
                <a href="#" onclick="showAllBSSIDs(this, event); return false;"
                   style="font-size: 0.9em; color: #0066cc;">
                    Show all {show_count} BSSIDs →
                </a>"""
        else:
            sample_cell = row["sample_bssids"]

        table_rows.append(
            f"""<tr data-bssid-details='{details_json}'>
                <td>{row["timestamp"]}</td>
                <td>{row["ssid"]}</td>
                <td>{show_count}</td>
                <td>{sample_cell}</td>
                <td>{row["randomized_pct"]}</td>
                <td>{row["vendor_ouis"]}</td>
                <td>{row["any_band"]}</td>
                <td>{row["any_channel"]}</td>
            </tr>"""
        )

    table_html = f"""
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>SSID</th>
                    <th>Distinct BSSID Count</th>
                    <th>Sample BSSIDs</th>
                    <th>Randomized %</th>
                    <th>Vendor OUIs</th>
                    <th>Any Band</th>
                    <th>Any Channel</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    """
    table = paginate_table(table_html, "table-scan-burst")
    note = "Grouped sightings where multiple BSSIDs for the same SSID appeared within the same second."
    return card("Scan Burst Summary", table, note)


def build_presence_section(presence_df: pd.DataFrame) -> str:
    if presence_df.empty:
        return card("Daily Presence Rollup", "", "No correlated presence rollup data was provided.")
    df = presence_df.copy()
    expected = [
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
    cols = [col for col in expected if col in df.columns]
    if not cols:
        return card("Daily Presence Rollup", "", "Presence CSV did not match the expected schema.")
    if "sightings" in df.columns:
        df = df.sort_values("date", ascending=False)
    df = df.head(1000)
    table = df[cols].to_html(index=False, classes="table table-striped")
    table = paginate_table(table, "table-presence")
    note = (
        "Aggregated sightings grouped by day/OUI/BSSID "
        "around DHCP lease windows, including join counts, BSSID diversity, and RSSI stats."
    )
    return card("Daily Presence Rollup", table, note)


def build_sightings_near_leases_section(events_df: pd.DataFrame, dhcp_df: pd.DataFrame) -> str:
    if dhcp_df.empty or events_df.empty:
        return card(
            "Sightings Around Leases",
            "",
            "No DHCP lease windows or normalized events were available for correlation.",
        )
    if "timestamp" not in events_df.columns:
        return card("Sightings Around Leases", "", "Events lacked timestamps, preventing lease correlation.")
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], errors="coerce", utc=True, format="mixed")
    ev = ev.dropna(subset=["timestamp"])
    if ev.empty:
        return card(
            "Sightings Around Leases",
            "",
            "No timestamped events were available to compare with lease windows.",
        )
    leases = dhcp_df.copy()
    leases["lease_start"] = pd.to_datetime(leases["lease_start"], errors="coerce", utc=True, format="mixed")
    leases["lease_end"] = pd.to_datetime(leases["lease_end"], errors="coerce", utc=True, format="mixed")
    leases["lease_end"] = leases["lease_end"].fillna(leases["lease_start"])
    leases = leases.dropna(subset=["lease_start"])
    if leases.empty:
        return card("Sightings Around Leases", "", "No valid lease windows remained after parsing the DHCP data.")
    rows: list[dict[str, Any]] = []
    for _, lease in leases.iterrows():
        start = lease["lease_start"]
        end = lease["lease_end"]
        mask = (ev["timestamp"] >= start) & (ev["timestamp"] <= end)
        window_events = ev.loc[mask]
        unique_bssids = 0
        if "bssid" in window_events.columns and not window_events.empty:
            unique_bssids = int(window_events["bssid"].dropna().nunique())
        rows.append(
            {
                "interface": lease.get("interface"),
                "router_ip": lease.get("router_ip"),
                "router_mac": lease.get("router_mac"),
                "lease_start": start,
                "lease_end": end,
                "events_in_window": int(len(window_events)),
                "unique_bssids": unique_bssids,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return card("Sightings Around Leases", "", "No normalized events overlapped any lease window.")

    # Deduplicate: keep first entry per lease_start+router_ip+router_mac combo
    # Don't use 'interface' as it may vary across carved fragments (filename-derived)
    # Use physical lease identifiers instead (time + router details)
    df = df.drop_duplicates(subset=["lease_start", "router_ip", "router_mac"], keep="first")

    # Sort by lease_start descending (newest first)
    df["lease_start"] = pd.to_datetime(df["lease_start"], utc=True, errors="coerce", format="mixed")
    df = df.sort_values("lease_start", ascending=False).head(1000)

    # Format timestamps for display
    for col in ("lease_start", "lease_end"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
    table = df[
        [
            "interface",
            "router_ip",
            "router_mac",
            "lease_start",
            "lease_end",
            "events_in_window",
            "unique_bssids",
        ]
    ].to_html(index=False, classes="table table-striped")
    table = paginate_table(table, "table-sightings-leases")
    note = (
        "Per-lease summary showing how many normalized events and unique BSSIDs appeared during each DHCP lease window."
    )
    return card("Sightings Around Leases", table, note)


def build_dhcp_section(df: pd.DataFrame) -> str:
    if df.empty:
        return card("DHCP Leases", "", "No DHCP lease plists were parsed.")

    df = df.copy()
    df["lease_start"] = pd.to_datetime(df["lease_start"], errors="coerce", format="mixed")
    df["lease_end"] = pd.to_datetime(df["lease_end"], errors="coerce", format="mixed")
    df["lease_end"] = df["lease_end"].fillna(df["lease_start"])

    # Deduplicate: keep first entry per interface+lease_start+IP combo
    df = df.drop_duplicates(subset=["interface", "lease_start", "ip_address"], keep="first")

    # Sort by lease_start descending (newest first)
    df = df.sort_values("lease_start", ascending=False)

    # Add filename column (just the basename, not full path)
    df["filename"] = df["path"].apply(lambda p: Path(p).name if pd.notna(p) else "Unknown")

    # Format timestamps
    df["lease_start_fmt"] = pd.to_datetime(df["lease_start"], errors="coerce", utc=True, format="mixed").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    df["lease_end_fmt"] = pd.to_datetime(df["lease_end"], errors="coerce", utc=True, format="mixed").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Build collapsible DHCP table with main row + hidden details row
    table_html = ['<table class="table table-striped dhcp-table">']

    # Header row - main info only
    table_html.append("<thead>")
    table_html.append("<tr>")
    table_html.append("<th>Interface</th>")
    table_html.append("<th>IP Address</th>")
    table_html.append("<th>SSID</th>")
    table_html.append("<th>Lease Start</th>")
    table_html.append("<th>Lease End</th>")
    table_html.append("<th></th>")  # Details toggle column
    table_html.append("</tr>")
    table_html.append("</thead>")

    # Body rows - main row + hidden details row per lease
    table_html.append("<tbody>")
    for _, lease in df.head(500).iterrows():
        interface = lease.get("interface", "")
        ip_address = lease.get("ip_address", "")
        ssid = lease.get("ssid", "")
        lease_start = lease.get("lease_start_fmt", "")
        lease_end = lease.get("lease_end_fmt", "")

        # Main row - key info
        table_html.append('<tr class="dhcp-main">')
        table_html.append(f"<td>{interface}</td>")
        table_html.append(f"<td>{ip_address}</td>")
        table_html.append(f"<td>{ssid}</td>")
        table_html.append(f"<td>{lease_start}</td>")
        table_html.append(f"<td>{lease_end}</td>")
        table_html.append('<td><button class="details-toggle" onclick="toggleDhcpDetails(this)">▶</button></td>')
        table_html.append("</tr>")

        # Details row - hidden by default
        router_ip = lease.get("router_ip", "")
        router_mac = lease.get("router_mac", "")
        dns_servers = lease.get("dns_servers", "")
        domain_name = lease.get("domain_name", "")
        filename = lease.get("filename", "")

        # Fingerprinting data
        ntp_servers = lease.get("ntp_servers", "")
        if isinstance(ntp_servers, list):
            ntp_servers = "; ".join(ntp_servers)
        vendor_class = lease.get("vendor_class", "")
        dhcp_server_id = lease.get("dhcp_server_id", "")

        table_html.append('<tr class="dhcp-details hidden">')
        table_html.append('<td colspan="6">')
        table_html.append('<div class="detail-grid">')
        table_html.append(f"<div><strong>Router IP:</strong> {router_ip}</div>")
        table_html.append(f"<div><strong>Router MAC:</strong> {router_mac}</div>")
        table_html.append(f"<div><strong>DNS:</strong> {dns_servers}</div>")
        table_html.append(f"<div><strong>Domain:</strong> {domain_name}</div>")
        if ntp_servers:
            table_html.append(f"<div><strong>NTP Servers:</strong> {ntp_servers}</div>")
        if vendor_class:
            table_html.append(f"<div><strong>Vendor Class:</strong> {vendor_class}</div>")
        if dhcp_server_id:
            table_html.append(f"<div><strong>DHCP Server ID:</strong> {dhcp_server_id}</div>")
        table_html.append(f"<div><strong>Lease File:</strong> {filename}</div>")
        table_html.append("</div>")
        table_html.append("</td>")
        table_html.append("</tr>")

    table_html.append("</tbody>")
    table_html.append("</table>")

    table = "\n".join(table_html)
    table = paginate_table(table, "table-dhcp")

    note = (
        "Lease windows parsed from DHCP plists.<br />Click ▶ to expand details including "
        "router info, DNS, and fingerprinting data (NTP servers, vendor class, DHCP server ID)."
    )
    return card("DHCP Leases", table, note)


def build_interface_section(
    interfaces_df: pd.DataFrame,
    services_df: pd.DataFrame,
    recent_leases: dict[str, str],
) -> str:
    if interfaces_df.empty:
        return card("Interfaces & Service Mapping", "", "No network interface configuration data was available.")

    df = interfaces_df.copy().drop_duplicates(subset=["bsd_name", "mac_address"], keep="first")
    if "bsd_name" not in df.columns:
        df["bsd_name"] = None
    df["recent_lease"] = df["bsd_name"].map(recent_leases).fillna("None recorded")
    service_map = {}
    if not services_df.empty and "bsd_device" in services_df.columns:
        service_map = (
            services_df.dropna(subset=["bsd_device"])
            .drop_duplicates("bsd_device")
            .set_index("bsd_device")["name"]
            .to_dict()
        )
    df["service_name"] = df["bsd_name"].map(service_map)

    for col in ("type", "user_name", "service_name", "mac_address", "built_in"):
        if col not in df.columns:
            df[col] = None
    display = df[
        [
            "bsd_name",
            "type",
            "user_name",
            "service_name",
            "mac_address",
            "built_in",
            "recent_lease",
        ]
    ].copy()
    display["built_in"] = display["built_in"].map({True: "Yes", False: "No"})
    object_cols = display.select_dtypes(include="object").columns
    display[object_cols] = display[object_cols].fillna("")
    table = display.to_html(index=False, classes="table table-striped")
    table = paginate_table(table, "table-interfaces")

    note = (
        "Maps parsed hardware interfaces to their service names, "
        "user-defined labels, MAC addresses, and most recent lease activity."
    )
    return card("Interfaces & Service Mapping", table, note)


def build_message_tracer_section(records: list[dict[str, Any]], known_ssids: set[str]) -> str:
    if not records:
        return card("Wi-Fi Message Tracer", "", "No Wi-Fi message tracer plists were parsed.")

    rows: list[dict[str, Any]] = []
    for entry in records:
        for bucket_name in ("association_counts", "roam_counts", "failed_join_counts"):
            bucket = entry.get(bucket_name)
            if isinstance(bucket, list):
                for item in bucket:
                    rows.append(
                        {
                            "source": entry.get("path"),
                            "metric": bucket_name.replace("_", " ").title(),
                            "ssid": item.get("ssid"),
                            "count": item.get("count", 0),
                        }
                    )
    if not rows:
        return card("Wi-Fi Message Tracer", "", "Parsed message tracer plists did not expose SSID metrics.")

    df = pd.DataFrame(rows)
    df["known_network"] = df["ssid"].isin(known_ssids)
    top = df.sort_values("count", ascending=False).head(1000)
    display = top.assign(known_network=lambda d: d["known_network"].map({True: "Yes", False: "No"}))
    table = display[["ssid", "count", "metric", "known_network"]].to_html(index=False, classes="table table-striped")
    table = paginate_table(table, "table-message-tracer")
    note = (
        "Association, roam, and failed-join counts taken from "
        "Wi-Fi message tracer logs, with an indicator for SSIDs already known to the system."
    )
    return card("Wi-Fi Message Tracer Metrics", table, note)


def build_location_map_section(locations: list[dict[str, Any]]) -> str:
    """Build interactive Leaflet map showing network locations.

    Args:
        locations: List of dicts with ssid, bssid, latitude, longitude, accuracy, timestamp

    Returns:
        HTML string with embedded Leaflet map
    """
    if not locations:
        return ""

    # Filter to only locations with valid coordinates
    valid_locs = [loc for loc in locations if loc.get("latitude") and loc.get("longitude")]
    if not valid_locs:
        return ""

    # Serialize locations to JSON for JavaScript
    locations_json = json.dumps(valid_locs)

    map_content = f"""
<!-- Leaflet CSS -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
      crossorigin=""/>

<div id="network-map" style="height: 400px; width: 100%; border-radius: 0.5rem; margin-top: 1rem;"></div>

<!-- Leaflet JS -->
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin=""></script>

<script>
(function() {{
    const locations = {locations_json};

    // Initialize map
    const map = L.map('network-map');

    // Add OpenStreetMap tiles
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }}).addTo(map);

    // Add markers for each location
    const markers = [];
    locations.forEach(loc => {{
        const lat = loc.latitude;
        const lng = loc.longitude;
        const accuracy = loc.accuracy || 0;

        // Create marker
        const marker = L.marker([lat, lng]).addTo(map);

        // Build popup content
        let popup = `<strong>${{loc.ssid || 'Unknown SSID'}}</strong><br>`;
        popup += `BSSID: ${{loc.bssid}}<br>`;
        if (loc.timestamp) {{
            popup += `Time: ${{loc.timestamp}}<br>`;
        }}
        if (accuracy) {{
            popup += `Accuracy: ±${{Math.round(accuracy)}}m<br>`;
        }}
        popup += `<a href="${{loc.map_link_google}}" target="_blank">Open in Google Maps</a>`;
        marker.bindPopup(popup);

        markers.push(marker);

        // Add accuracy circle if significant
        if (accuracy > 10) {{
            L.circle([lat, lng], {{
                radius: accuracy,
                color: '#3388ff',
                fillColor: '#3388ff',
                fillOpacity: 0.1,
                weight: 1
            }}).addTo(map);
        }}
    }});

    // Fit map to show all markers
    if (markers.length > 0) {{
        const group = L.featureGroup(markers);
        map.fitBounds(group.getBounds().pad(0.1));
    }} else {{
        map.setView([0, 0], 2);
    }}
}})();
</script>
"""
    return card(
        "Network Locations Map",
        map_content,
        "GPS coordinates captured when connecting to networks (Big Sur+ only). "
        "Click markers for details. Circles show accuracy radius.",
    )


def build_summary_sections(
    summary: dict[str, Any],
    events_df: pd.DataFrame | None = None,
    presence_df: pd.DataFrame | None = None,
) -> list[str]:
    sections: list[str] = []

    dhcp_df = to_dataframe(summary.get("dhcp_leases"))
    recent_map = compute_recent_lease_map(dhcp_df)

    known_df = to_dataframe(summary.get("airport_known_networks"))
    sections.append(build_known_networks_section(known_df, recent_map, dhcp_df=dhcp_df))

    events_df = events_df if isinstance(events_df, pd.DataFrame) else pd.DataFrame()
    presence_df = presence_df if isinstance(presence_df, pd.DataFrame) else pd.DataFrame()
    legacy_events_df = to_dataframe(summary.get("wifi_logs", {}).get("events"))
    effective_events = events_df if not events_df.empty else legacy_events_df
    sections.append(build_wifi_events_section(effective_events))

    sections.append(build_dhcp_section(dhcp_df))

    if not presence_df.empty:
        sections.append(build_presence_section(presence_df))
    sections.append(build_sightings_near_leases_section(effective_events, dhcp_df))

    interfaces_df = to_dataframe(summary.get("network_interfaces"))
    services_df = to_dataframe(summary.get("network_services"))
    sections.append(build_interface_section(interfaces_df, services_df, recent_map))

    known_bssid_df = join_known_with_dhcp(known_df.copy(), recent_map, dhcp_df=dhcp_df)
    bssid_html, log_stats, gps_locations = build_bssid_section(known_bssid_df, effective_events)
    sections.append(bssid_html)
    sections.append(build_scan_burst_section(effective_events, log_stats))

    # Add embedded map section if we have GPS locations
    if gps_locations:
        sections.append(build_location_map_section(gps_locations))

    known_ssids = set(known_df["ssid"].dropna()) if not known_df.empty and "ssid" in known_df.columns else set()

    tracer_section = build_message_tracer_section(summary.get("wifi_message_tracer") or [], known_ssids)
    sections.append(tracer_section)

    return sections


def compose_html(exemplar: str, sections: list[str]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    logo_data_uri = get_logo_data_uri()
    body = "\n".join(sections)
    return rf"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wi-Fi Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --primary: #2563eb;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --text-secondary: #475569;
      --border: #e2e8f0;
      --border-light: #f1f5f9;
      --link: #2563eb;
      --timeline-user: #16a34a;
      --timeline-system: #2563eb;
      --timeline-disconnect: #dc2626;
    }}
    [data-theme="dark"] {{
      --primary: #60a5fa;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --text-secondary: #cbd5e1;
      --border: #334155;
      --border-light: #1e293b;
      --link: #60a5fa;
      --timeline-user: #4ade80;
      --timeline-system: #60a5fa;
      --timeline-disconnect: #f87171;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 1rem 2rem 2rem 2rem;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ font-size: 1.875rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .meta {{ color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.875rem; }}
    /* Card layout */
    .card {{
      background: var(--card-bg);
      border-radius: 0.75rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .card-title {{
      font-size: 1.125rem;
      font-weight: 600;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--border);
    }}
    .section-note {{ color: var(--text-muted); font-size: 0.875rem; margin: -0.5rem 0 1rem; }}
    /* Table styling */
    .table {{ border-collapse: collapse; width: 100%; font-size: 0.875rem; }}
    .table th, .table td {{ text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); }}
    .table th {{
      background: var(--bg);
      font-weight: 600;
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .table tbody tr:hover {{ background: var(--bg); }}
    .table-striped tbody tr:nth-child(even) {{ background: var(--border-light); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    /* Pagination */
    .table-wrapper {{ margin-bottom: 0; }}
    .hidden-row {{ display: none; }}
    .hidden {{ display: none; }}
    .pagination-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      font-size: 0.875rem;
      flex-wrap: wrap;
    }}
    .pagination-nav {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .pagination-nav button {{
      padding: 0.5rem 0.875rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 0.375rem;
      transition: opacity 0.15s, transform 0.1s;
    }}
    .pagination-nav button:hover:not(:disabled) {{ opacity: 0.9; transform: translateY(-1px); }}
    .pagination-nav button:disabled {{ background: var(--text-muted); opacity: 0.5; cursor: not-allowed; transform: none; }}
    .pagination-nav .page-first,
    .pagination-nav .page-last {{
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 0.625rem;
    }}
    .pagination-nav .page-first:hover:not(:disabled),
    .pagination-nav .page-last:hover:not(:disabled) {{ background: var(--bg); }}
    .pagination-size {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-muted);
    }}
    .pagination-size label {{ font-size: 0.8125rem; }}
    .pagination-size select {{
      padding: 0.375rem 0.5rem;
      font-size: 0.875rem;
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 0.375rem;
    }}
    .page-info {{ color: var(--text-muted); font-size: 0.8125rem; min-width: 150px; text-align: center; }}
    /* DHCP collapsible details */
    .details-toggle {{
      background: none;
      border: none;
      cursor: pointer;
      font-size: 0.75rem;
      color: var(--text-muted);
      padding: 0.25rem 0.5rem;
      transition: transform 0.15s;
    }}
    .details-toggle:hover {{ color: var(--text); }}
    .dhcp-details, .bssid-details, .known-details {{ background: var(--bg); }}
    .dhcp-details td, .bssid-details td, .known-details td {{ padding: 0 !important; border: none !important; }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem;
      padding: 1rem;
      font-size: 0.8125rem;
    }}
    .detail-grid div {{ color: var(--text-secondary); }}
    .detail-grid strong {{ color: var(--text); }}

    /* Modal styles */
    .modal {{
      display: none;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.5);
    }}
    .modal-content {{
      background-color: var(--card-bg);
      margin: 5% auto;
      padding: 1.5rem;
      border: 1px solid var(--border);
      width: 90%;
      max-width: 900px;
      border-radius: 0.75rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      color: var(--text);
    }}
    .modal-close {{
      color: var(--text-muted);
      float: right;
      font-size: 1.5rem;
      font-weight: bold;
      cursor: pointer;
      line-height: 1;
    }}
    .modal-close:hover {{ color: var(--text); }}
    #modalTable {{ margin-top: 1rem; }}
    #modalTable th {{ background: var(--bg); }}

    /* Cloud Sync attribution styling - networks synced from other Apple devices */
    .cloud-sync-row {{ font-style: italic; color: var(--text-muted); }}

    /* Connection timeline styling */
    .timeline-user {{ color: var(--timeline-user); }}
    .timeline-system {{ color: var(--timeline-system); }}
    .timeline-disconnect {{ color: var(--timeline-disconnect); }}

    /* Map and link styling */
    .map-link {{ font-size: 0.8125rem; color: var(--link); }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* Theme toggle button */
    .theme-toggle {{
      position: fixed;
      bottom: 1rem;
      right: 1rem;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      cursor: pointer;
      font-size: 0.875rem;
      color: var(--text);
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      z-index: 100;
      transition: background 0.2s, color 0.2s, border-color 0.2s;
    }}
    .theme-toggle:hover {{ background: var(--bg); }}

    /* Header layout */
    .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }}
    .header-logo {{ height: 80px; }}
  </style>
  <script>
    // Pagination state per table
    const paginationState = {{}};

    function setupPagination() {{
      document.querySelectorAll('.table-wrapper').forEach(wrapper => {{
        const table = wrapper.querySelector('table');
        if (!table) return;
        // Exclude detail rows from pagination count
        const rows = Array.from(table.querySelectorAll('tbody tr:not(.dhcp-details):not(.bssid-details):not(.known-details)'));
        const pageSize = parseInt(wrapper.dataset.pageSize || '20', 10);
        const tableId = wrapper.id;

        // Initialize state
        paginationState[tableId] = {{
          currentPage: 1,
          pageSize: pageSize,
          totalRows: rows.length
        }};

        // Hide controls if few rows
        if (rows.length <= pageSize) {{
          const controls = wrapper.querySelector('.pagination-controls');
          if (controls) controls.style.display = 'none';
          return;
        }}

        // Show first page
        showPage(tableId, 1);
      }});
    }}

    function showPage(tableId, page) {{
      const wrapper = document.getElementById(tableId);
      if (!wrapper) return;

      const state = paginationState[tableId];
      if (!state) return;

      const table = wrapper.querySelector('table');
      // Exclude detail rows from pagination count
      const rows = Array.from(table.querySelectorAll('tbody tr:not(.dhcp-details):not(.bssid-details):not(.known-details)'));
      const totalPages = Math.ceil(state.totalRows / state.pageSize);

      // Clamp page number
      page = Math.max(1, Math.min(page, totalPages));
      state.currentPage = page;

      const startIdx = (page - 1) * state.pageSize;
      const endIdx = startIdx + state.pageSize;

      rows.forEach((row, idx) => {{
        if (idx >= startIdx && idx < endIdx) {{
          row.classList.remove('hidden-row');
          // Also show associated details row if expanded
          const details = row.nextElementSibling;
          const isDetailsRow = details && (details.classList.contains('dhcp-details') || details.classList.contains('bssid-details') || details.classList.contains('known-details'));
          if (isDetailsRow && row.classList.contains('expanded')) {{
            details.classList.remove('hidden-row');
          }}
        }} else {{
          row.classList.add('hidden-row');
          // Hide associated details row
          const details = row.nextElementSibling;
          const isDetailsRow = details && (details.classList.contains('dhcp-details') || details.classList.contains('bssid-details') || details.classList.contains('known-details'));
          if (isDetailsRow) {{
            details.classList.add('hidden-row');
          }}
        }}
      }});

      // Update controls
      updatePaginationControls(tableId);
    }}

    function updatePaginationControls(tableId) {{
      const wrapper = document.getElementById(tableId);
      const state = paginationState[tableId];
      if (!wrapper || !state) return;

      const totalPages = Math.ceil(state.totalRows / state.pageSize);

      const pageInfo = wrapper.querySelector('.page-info');
      if (pageInfo) {{
        pageInfo.textContent = `Page ${{state.currentPage}} of ${{totalPages}} (${{state.totalRows}} rows)`;
      }}

      const prevBtn = wrapper.querySelector('.page-prev');
      const nextBtn = wrapper.querySelector('.page-next');
      const firstBtn = wrapper.querySelector('.page-first');
      const lastBtn = wrapper.querySelector('.page-last');

      if (prevBtn) prevBtn.disabled = state.currentPage <= 1;
      if (nextBtn) nextBtn.disabled = state.currentPage >= totalPages;
      if (firstBtn) firstBtn.disabled = state.currentPage <= 1;
      if (lastBtn) lastBtn.disabled = state.currentPage >= totalPages;
    }}

    function goToPage(tableId, page) {{
      showPage(tableId, page);
    }}

    function prevPage(tableId) {{
      const state = paginationState[tableId];
      if (state) showPage(tableId, state.currentPage - 1);
    }}

    function nextPage(tableId) {{
      const state = paginationState[tableId];
      if (state) showPage(tableId, state.currentPage + 1);
    }}

    function firstPage(tableId) {{
      showPage(tableId, 1);
    }}

    function lastPage(tableId) {{
      const state = paginationState[tableId];
      if (state) {{
        const totalPages = Math.ceil(state.totalRows / state.pageSize);
        showPage(tableId, totalPages);
      }}
    }}

    function setPageSize(tableId, size) {{
      const state = paginationState[tableId];
      if (state) {{
        state.pageSize = parseInt(size, 10);
        showPage(tableId, 1);  // Reset to first page
      }}
    }}

    // DHCP collapsible details toggle
    function toggleDhcpDetails(btn) {{
      const mainRow = btn.closest('tr');
      if (!mainRow) return;
      const detailsRow = mainRow.nextElementSibling;
      if (!detailsRow || !detailsRow.classList.contains('dhcp-details')) return;

      const isExpanded = mainRow.classList.contains('expanded');
      if (isExpanded) {{
        mainRow.classList.remove('expanded');
        detailsRow.classList.add('hidden');
        btn.textContent = '▶';
        btn.classList.remove('expanded');
      }} else {{
        mainRow.classList.add('expanded');
        detailsRow.classList.remove('hidden');
        btn.textContent = '▼';
        btn.classList.add('expanded');
      }}
    }}

    // BSSID collapsible details toggle
    function toggleBssidDetails(btn) {{
      const mainRow = btn.closest('tr');
      if (!mainRow) return;
      const detailsRow = mainRow.nextElementSibling;
      if (!detailsRow || !detailsRow.classList.contains('bssid-details')) return;

      const isExpanded = mainRow.classList.contains('expanded');
      if (isExpanded) {{
        mainRow.classList.remove('expanded');
        detailsRow.classList.add('hidden');
        btn.textContent = '▶';
        btn.classList.remove('expanded');
      }} else {{
        mainRow.classList.add('expanded');
        detailsRow.classList.remove('hidden');
        btn.textContent = '▼';
        btn.classList.add('expanded');
      }}
    }}

    // Known Networks collapsible details toggle
    function toggleKnownDetails(btn) {{
      const mainRow = btn.closest('tr');
      if (!mainRow) return;
      const detailsRow = mainRow.nextElementSibling;
      if (!detailsRow || !detailsRow.classList.contains('known-details')) return;

      const isExpanded = mainRow.classList.contains('expanded');
      if (isExpanded) {{
        mainRow.classList.remove('expanded');
        detailsRow.classList.add('hidden');
        btn.textContent = '▶';
        btn.classList.remove('expanded');
      }} else {{
        mainRow.classList.add('expanded');
        detailsRow.classList.remove('hidden');
        btn.textContent = '▼';
        btn.classList.add('expanded');
      }}
    }}

    // Modal functions for BSSID details
    function showAllBSSIDs(linkElement, event) {{
      event.preventDefault();

      // Find parent row
      const row = linkElement.closest('tr');
      if (!row) return;

      // Get BSSID details from data attribute
      const detailsJson = row.getAttribute('data-bssid-details');
      if (!detailsJson) return;

      try {{
        const details = JSON.parse(detailsJson);

        // Get SSID and timestamp for modal title
        const cells = row.querySelectorAll('td');
        const timestamp = cells[0]?.textContent || 'Unknown';
        const ssid = cells[1]?.textContent || 'Unknown';

        // Set modal title
        document.getElementById('modalTitle').textContent =
          `BSSIDs for "${{ssid}}" at ${{timestamp}}`;

        // Populate modal table
        const tbody = document.getElementById('modalBody');
        tbody.innerHTML = '';

        details.forEach(item => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${{item.bssid}}</td>
            <td>${{item.vendor}}</td>
            <td>${{item.oui}}</td>
            <td>${{item.laa}}</td>
            <td>${{item.first_seen}}</td>
            <td>${{item.last_seen}}</td>
          `;
          tbody.appendChild(tr);
        }});

        // Show modal
        document.getElementById('bssidModal').style.display = 'block';
      }} catch (e) {{
        console.error('Failed to parse BSSID details:', e);
      }}
    }}

    function closeModal() {{
      document.getElementById('bssidModal').style.display = 'none';
      document.getElementById('timelineModal').style.display = 'none';
    }}

    function showTimeline(linkElement, event) {{
      event.preventDefault();
      const timelineJson = linkElement.getAttribute('data-timeline');
      if (!timelineJson) return;

      try {{
        const data = JSON.parse(timelineJson);

        // Set modal title
        document.getElementById('timelineTitle').textContent =
          `Connection Timeline for "${{data.ssid || 'Unknown'}}"`;

        // Build timeline content
        const content = document.getElementById('timelineContent');
        let html = '<dl style="margin: 0;">';

        if (data.joined_user) {{
          html += `<dt class="timeline-user" style="font-weight: bold; margin-top: 0.5rem;">User Connected</dt>`;
          html += `<dd style="margin-left: 1rem;">The user manually selected this network from the WiFi menu.<br><strong>Time:</strong> ${{data.joined_user}}</dd>`;
        }}

        if (data.joined_system) {{
          html += `<dt class="timeline-system" style="font-weight: bold; margin-top: 0.5rem;">System Auto-Connected</dt>`;
          html += `<dd style="margin-left: 1rem;">The system automatically connected to this known network.<br><strong>Time:</strong> ${{data.joined_system}}</dd>`;
        }}

        if (data.last_discovered) {{
          html += `<dt style="font-weight: bold; margin-top: 0.5rem;">Last Discovered</dt>`;
          html += `<dd style="margin-left: 1rem;">The network was detected in range but not connected.<br><strong>Time:</strong> ${{data.last_discovered}}</dd>`;
        }}

        if (data.disconnect_ts) {{
          const reason = data.disconnect_reason ? ` (${{data.disconnect_reason}})` : '';
          html += `<dt class="timeline-disconnect" style="font-weight: bold; margin-top: 0.5rem;">Last Disconnected</dt>`;
          html += `<dd style="margin-left: 1rem;">The connection was terminated${{reason}}.<br><strong>Time:</strong> ${{data.disconnect_ts}}</dd>`;
        }}

        html += '</dl>';
        content.innerHTML = html;

        // Show modal
        document.getElementById('timelineModal').style.display = 'block';
      }} catch (e) {{
        console.error('Failed to parse timeline data:', e);
      }}
    }}

    // Close modal when clicking outside
    window.onclick = function(event) {{
      const bssidModal = document.getElementById('bssidModal');
      const timelineModal = document.getElementById('timelineModal');
      if (event.target === bssidModal || event.target === timelineModal) {{
        closeModal();
      }}
    }}

    document.addEventListener('DOMContentLoaded', setupPagination);

    // Theme toggle functionality - dark mode is default (set in body tag)
    (function() {{
      const saved = localStorage.getItem('mars-theme');
      if (saved === 'light') {{
        document.body.removeAttribute('data-theme');
        const btn = document.querySelector('.theme-toggle');
        if (btn) btn.textContent = '🌙 Dark';
      }}
    }})();
    function toggleTheme() {{
      const body = document.body;
      const btn = document.querySelector('.theme-toggle');
      const isDark = body.getAttribute('data-theme') === 'dark';
      if (isDark) {{
        body.removeAttribute('data-theme');
        btn.textContent = '🌙 Dark';
        localStorage.setItem('mars-theme', 'light');
      }} else {{
        body.setAttribute('data-theme', 'dark');
        btn.textContent = '☀️ Light';
        localStorage.removeItem('mars-theme');  // Dark is default, no need to store
      }}
    }}
    // Update button text on load
    document.addEventListener('DOMContentLoaded', function() {{
      const btn = document.querySelector('.theme-toggle');
      if (document.body.getAttribute('data-theme') === 'dark') {{
        btn.textContent = '☀️ Light';
      }} else {{
        btn.textContent = '🌙 Dark';
      }}
    }});
  </script>
</head>
<body data-theme="dark">
  <button class="theme-toggle" onclick="toggleTheme()">☀️ Light</button>
  <!-- BSSID Modal -->
  <div id="bssidModal" class="modal">
    <div class="modal-content">
      <span class="modal-close" onclick="closeModal()">&times;</span>
      <h2 id="modalTitle">BSSIDs for Scan Burst</h2>
      <table id="modalTable" class="table table-striped">
        <thead>
          <tr>
            <th>BSSID</th>
            <th>Vendor</th>
            <th>OUI</th>
            <th>LAA</th>
            <th>First Seen</th>
            <th>Last Seen</th>
          </tr>
        </thead>
        <tbody id="modalBody"></tbody>
      </table>
    </div>
  </div>

  <!-- Timeline Modal -->
  <div id="timelineModal" class="modal">
    <div class="modal-content" style="max-width: 600px;">
      <span class="modal-close" onclick="closeModal()">&times;</span>
      <h2 id="timelineTitle">Connection Timeline</h2>
      <div id="timelineContent"></div>
    </div>
  </div>

  <div>
    <div
      style="
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.2rem;
      "
    >
      <div>
        <h1>Wi-Fi / Network Activity Report</h1>
        <div class="meta">
          <strong>Top 1,000 Entries | Time: UTC</strong><br />
          <strong>Generated:</strong> {generated}<br />
          <strong>Source:</strong> {exemplar}<br />
        </div>
      </div>
      <div>
        <img src="{logo_data_uri}" alt="WarpedWing Labs Logo" height="100px"/>
      </div>
    </div>
  </div>
  {body}
</body>
</html>
"""


def default_output_path(summary_path: Path) -> Path:
    return summary_path.with_suffix(".html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wi-Fi HTML report from summary JSON.")
    parser.add_argument("summary", type=Path, help="Path to wifi_summary JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination HTML path (default: same name as JSON)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Optional path to wifi_events.ndjson (normalized events)",
    )
    parser.add_argument(
        "--presence",
        type=Path,
        help="Optional path to wifi_presence_daily.csv (daily rollup)",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    events_df = load_ndjson(args.events)
    presence_df = load_csv(args.presence)
    sections = build_summary_sections(summary, events_df, presence_df)
    html = compose_html(summary.get("exemplar_root", "Unknown"), sections)

    output_path = args.output or default_output_path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"[info] wrote report to {output_path}")


if __name__ == "__main__":
    main()
