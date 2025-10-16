#!/usr/bin/env python3

"""
Timestamp Pattern Detection and Interpretation
by WarpedWing Labs

Handles detection and interpretation of various timestamp formats:
- Unix epochs (seconds, milliseconds, microseconds, nanoseconds)
- Apple Cocoa/NSDate (seconds and nanoseconds since 2001)
- WebKit (microseconds since 1601)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

# Timestamp format detection regex
# Matches numbers with 9+ digits (potential timestamps)
# Uses negative lookbehind/lookahead to avoid matching within larger numbers
# Matches complete numbers only (bounded by non-digits or start/end of string)
TIMESTAMP_REGEX = re.compile(rb"(?<![0-9])[0-9]{9,}(?:\.[0-9]+)?(?![0-9])")

# Hexadecimal timestamp regex (e.g., 5b994548.5ad909e0)
# Matches 8 hex chars, optional dot, 8 hex chars
# Uses negative lookbehind/lookahead to avoid matching within larger hex strings
# This prevents matching "21165313" from within "20180321165313"
HEX_TIMESTAMP_REGEX = re.compile(rb"(?<![0-9a-fA-F])[0-9a-fA-F]{8}(?:\.[0-9a-fA-F]{8})?(?![0-9a-fA-F])")

# UUID pattern (RFC 4122 format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)
# Used to exclude UUID segments from timestamp detection
UUID_REGEX = re.compile(rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", re.IGNORECASE)

# Global timestamp validity window (will be overridden by CLI args)
TARGET_START = datetime(2015, 1, 1, tzinfo=UTC)
TARGET_END = datetime(2030, 1, 1, tzinfo=UTC)


def url_decode_bytes(data: bytes) -> tuple[bytes, list[int]]:
    """
    URL-decode bytes while tracking the source offset for each decoded byte.

    This ensures we find timestamps in URL-encoded strings like:
    message:%3CetPan.5b994548.5ad909e0.fe@example.org%3E

    Returns (decoded bytes, offset map). The offset map aligns each byte in the
    decoded output with the index of the corresponding byte in the original
    input (typically the '%' for percent-encoded sequences).
    """
    decoded = bytearray()
    mapping: list[int] = []
    i = 0
    length = len(data)

    while i < length:
        b = data[i]
        if b == 0x25 and i + 2 < length:  # '%'
            hex_chunk = data[i + 1 : i + 3]
            try:
                decoded_byte = int(hex_chunk, 16)
                # Validate byte is in valid range (0-255)
                if not (0 <= decoded_byte <= 255):
                    # Invalid percent-encoding, treat '%' as literal
                    decoded.append(b)
                    mapping.append(i)
                    i += 1
                    continue
            except ValueError:
                # Invalid hex in percent-encoding, treat '%' as literal
                decoded.append(b)
                mapping.append(i)
                i += 1
                continue

            decoded.append(decoded_byte)
            mapping.append(i)
            i += 3
        else:
            decoded.append(b)
            mapping.append(i)
            i += 1

    return bytes(decoded), mapping


def set_timestamp_range(start: datetime, end: datetime):
    """Update the global timestamp validity range"""
    global TARGET_START, TARGET_END
    TARGET_START = start
    TARGET_END = end


def interpret_timestamp_best(value: float | int) -> tuple[str | None, int | None, str]:
    """
    Strict datetime-driven timestamp classifier.

    Attempts to interpret `value` as one of several known epoch formats:
    - Unix (sec, ms, µs, ns)
    - Cocoa / NSDate (sec, ns since 2001-01-01)
    - WebKit (µs since 1601-01-01)

    Returns:
        (kind:str, original_value:int, human_readable:str)
        Returns the ORIGINAL value, not normalized to Unix seconds.
    """
    unix0 = datetime(1970, 1, 1, tzinfo=UTC)
    cocoa0 = datetime(2001, 1, 1, tzinfo=UTC)
    webkit0 = datetime(1601, 1, 1, tzinfo=UTC)

    try:
        val = int(float(value))
    except Exception:
        return None, None, ""

    now = datetime.now(UTC)
    candidates = []

    def add(label: str, dt: datetime):
        """Accept candidate if within configured validity window."""
        if TARGET_START <= dt <= TARGET_END:
            candidates.append((label, dt))

    # --- WebKit (µs since 1601) ---
    # 17-digit numbers are often WebKit timestamps
    if len(str(abs(val))) == 17:
        try:
            dt = webkit0 + timedelta(microseconds=val)
            add("webkit_micro", dt)
        except Exception:
            pass

    # --- Cocoa (since 2001) ---
    for div, label in [(1, "cocoa_sec"), (1e9, "cocoa_nano")]:
        try:
            dt = cocoa0 + timedelta(seconds=val / div)
            add(label, dt)
        except Exception:
            pass

    # --- Unix (since 1970) ---
    for div, label in [
        (1, "unix_sec"),
        (1e3, "unix_milli"),
        (1e6, "unix_micro"),
        (1e9, "unix_nano"),
    ]:
        try:
            dt = unix0 + timedelta(seconds=val / div)
            add(label, dt)
        except Exception:
            pass

    if not candidates:
        return None, None, ""

    # Choose candidate closest to now; tie-break prefers Unix variants
    candidates.sort(
        key=lambda kv: (
            abs(kv[1].timestamp() - now.timestamp()),
            0 if kv[0].startswith("unix") else 1,
        )
    )

    kind, dt = candidates[0]
    return kind, val, dt.strftime("%Y-%m-%d %H:%M:%S GMT")


def interpret_hex_timestamp(hex_str: bytes) -> list[tuple[str, str, str]]:
    """
    Interpret hexadecimal timestamp format (e.g., 5b994548 or 5b994548.5ad909e0).

    These are typically Unix timestamps encoded as hex.
    If there's a dot, we extract BOTH timestamps separately.

    Returns:
        List of (kind:str, hex_value:str, human_readable:str) tuples
        Note: Returns hex_value as string (e.g., "5b994548"), not decimal
    """
    results = []

    try:
        # Split on dot if present
        parts = hex_str.split(b".")

        for part in parts:
            # Skip if not 8 hex digits
            if len(part) != 8:
                continue

            # Parse hex value to decimal for validation
            hex_val = int(part, 16)

            # Try to interpret as unix timestamp
            kind, epoch, human = interpret_timestamp_best(hex_val)

            if kind is not None:
                # Return with special kind indicating hex format
                # Store the original hex string, not the decimal value
                results.append((f"{kind}_hex", part.decode("ascii"), human))

    except Exception:
        pass

    return results


def find_hex_timestamp_candidates(page: bytes) -> list[tuple[int, str, str, str]]:
    """
    Find hexadecimal timestamps in page (e.g., 5b994548 or 5b994548.5ad909e0).

    When pattern like "5b994548.5ad909e0" is found, extracts BOTH timestamps.
    Searches both original page AND URL-decoded version to catch encoded timestamps.

    Excludes UUIDs (e.g., 6E1DD98F-67C5-4222-A810-5BD8C88D8745) to prevent false positives.

    Returns:
        List of (offset, hex_value, kind, human_readable) tuples
        where hex_value is the original hex string (e.g., "5b994548")
    """
    candidates = []
    seen_offsets = set()

    # Search both original and decoded versions
    decoded_page, decoded_map = url_decode_bytes(page)
    pages_to_search = [
        (page, list(range(len(page)))),  # Original page with identity map
        (decoded_page, decoded_map),  # Decoded page with offset mapping
    ]

    for search_page, offset_map in pages_to_search:
        # Build exclusion ranges for UUID segments
        # UUIDs have format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        # We want to exclude all hex segments within UUIDs
        uuid_ranges = set()
        for uuid_match in UUID_REGEX.finditer(search_page):
            uuid_start = uuid_match.start()
            uuid_end = uuid_match.end()
            # Add all byte positions within this UUID to exclusion set
            for pos in range(uuid_start, uuid_end):
                uuid_ranges.add(pos)

        for m in HEX_TIMESTAMP_REGEX.finditer(search_page):
            try:
                hex_str = m.group(0)

                # Skip if this hex string overlaps with a UUID
                hex_start = m.start()
                hex_end = m.end()
                if any(pos in uuid_ranges for pos in range(hex_start, hex_end)):
                    continue

                # Quick filter: skip if all zeros
                if hex_str.replace(b".", b"").replace(b"0", b"") == b"":
                    continue

                # Parse and validate (may return multiple timestamps if dot-separated)
                results = interpret_hex_timestamp(hex_str)

                for kind, hex_val, human in results:
                    # Calculate offset for this specific hex value within the match
                    relative_idx = m.start() + hex_str.find(hex_val.encode())
                    if relative_idx >= len(offset_map):
                        continue
                    offset = offset_map[relative_idx]

                    # Deduplicate by offset
                    if offset in seen_offsets:
                        continue
                    seen_offsets.add(offset)

                    candidates.append((offset, hex_val, kind, human))

            except Exception:
                continue

    return candidates


def find_timestamp_candidates(page: bytes) -> list[tuple[int, float]]:
    """
    Find all numbers in page that could potentially be timestamps.

    Pre-filters by digit count to avoid processing obviously invalid values.
    Valid ranges:
    - 10 digits: unix_sec (e.g., 1609459200)
    - 13 digits: unix_milli or cocoa_sec
    - 16-17 digits: unix_micro, cocoa_nano, webkit_micro
    - 19 digits: unix_nano

    Searches both original page AND URL-decoded version to catch encoded timestamps.

    NOTE: This function does NOT include hex timestamps.
    Use find_hex_timestamp_candidates() separately for those.

    Returns:
        List of (offset, numeric_value) tuples
    """
    candidates = []
    seen_offsets = set()

    # Search both original and decoded versions
    decoded_page, decoded_map = url_decode_bytes(page)
    pages_to_search = [
        (page, list(range(len(page)))),  # Original page
        (decoded_page, decoded_map),  # Decoded page with mapping to original offsets
    ]

    for search_page, offset_map in pages_to_search:
        # Find decimal timestamps
        for m in TIMESTAMP_REGEX.finditer(search_page):
            try:
                raw_str = m.group(0)

                # Quick filter: skip if all zeros or obviously invalid
                if raw_str.replace(b".", b"").replace(b"0", b"") == b"":
                    continue

                raw = float(raw_str)

                # Pre-filter by digit count (ignore decimal for now)
                digit_count = len(str(int(abs(raw))))

                # Only process numbers with plausible timestamp lengths
                # 10=unix_sec, 13=unix_milli/cocoa_sec, 16-17=micro/webkit, 19=nano
                if digit_count not in (10, 13, 16, 17, 19):
                    continue

                offset_idx = m.start()
                if offset_idx >= len(offset_map):
                    continue
                offset = offset_map[offset_idx]

                # Deduplicate by offset
                if offset in seen_offsets:
                    continue
                seen_offsets.add(offset)

                candidates.append((offset, raw))

            except Exception:
                continue

    return candidates


def find_timestamps_with_interpretation(
    page: bytes,
) -> list[tuple[int, float, str, str]]:
    """
    Find and interpret all timestamps in a page.

    Returns:
        List of (offset, epoch_value, kind, human_readable) tuples
    """
    results = []

    candidates = find_timestamp_candidates(page)
    for offset, raw_value in candidates:
        kind, epoch, human = interpret_timestamp_best(raw_value)
        if epoch is not None:
            results.append((offset, epoch, kind, human))

    return results


def find_structured_timestamps(page: bytes) -> list[tuple[int, int, str]]:
    """
    Find timestamps in structured binary formats (8-byte aligned integers).
    These are more likely to be real timestamps than random numbers.

    Returns:
        List of (offset, value, format) tuples where format is 'le64' or 'be64'
    """
    results = []

    # Check 8-byte aligned positions
    for i in range(0, len(page) - 8, 8):
        # Try both little-endian and big-endian
        le_val = int.from_bytes(page[i : i + 8], "little")
        be_val = int.from_bytes(page[i : i + 8], "big")

        # Check if either interpretation yields a valid timestamp
        for val, fmt in [(le_val, "le64"), (be_val, "be64")]:
            kind, epoch, human = interpret_timestamp_best(val)
            if epoch is not None:
                results.append((i, val, fmt))
                break  # Don't double-count same offset

    return results


def detect_timestamp_field_pairs(
    page: bytes, timestamps: list[tuple[int, float, str, str]]
) -> list[tuple[int, int]]:
    """
    Detect common timestamp field pairs (created_at/updated_at, etc.).
    These pairs are strong evidence of real timestamps.

    Returns:
        List of (offset1, offset2) pairs that appear related
    """
    pairs = []

    # Look for timestamps that are:
    # 1. Within 8-32 bytes of each other (typical struct field spacing)
    # 2. Same format type (both unix_sec, both unix_milli, etc.)
    # 3. Second timestamp >= first (updated >= created)

    for i, (off1, val1, kind1, _) in enumerate(timestamps):
        for off2, val2, kind2, _ in timestamps[i + 1 :]:
            # Check proximity
            distance = abs(off2 - off1)
            if distance < 8 or distance > 32:
                continue

            # Check same type
            if kind1 != kind2:
                continue

            # Check temporal ordering makes sense
            if val2 >= val1:  # updated >= created
                pairs.append((off1, off2))

    return pairs


def format_timestamp_summary(kind: str, value: int, human: str, confidence: float) -> str:
    """
    Format a human-readable summary of a timestamp detection.
    """
    conf_label = "HIGH" if confidence >= 0.8 else "MED" if confidence >= 0.6 else "LOW"
    return f"[{conf_label}] {kind}: {value} → {human}"
