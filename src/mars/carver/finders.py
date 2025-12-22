#!/usr/bin/env python3
"""
Data Finders for SQLite Carver
by WarpedWing Labs

Functions for finding timestamps, URLs, text runs, and blob candidates in raw page data.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================================
# Pattern Matching
# ============================================================================

# URL pattern (HTTP/HTTPS URLs)
URL_RE = re.compile(rb"https?://[^\s\0]+", re.IGNORECASE)

# Numeric pattern for timestamp detection
_NUM_RE = re.compile(
    rb"""
    (?:^|(?<=\s)|(?<=\0))      # Start or preceded by whitespace/null
    [-+]?                       # Optional sign
    \d{8,20}                    # 8-20 digits (covers most epoch formats)
    (?:\.\d+)?                  # Optional decimal part
    (?=\s|\0|$|[^\d.])         # Followed by delimiter
    """,
    re.VERBOSE,
)


# ============================================================================
# Finder Functions
# ============================================================================


def find_timestamps(page: bytes, interpret_timestamp_best: Callable):
    """
    Find timestamps in page data.
    Pre-filters by digit count to avoid processing obviously invalid values.

    Valid ranges:
    - 10 digits: unix_sec (e.g., 1609459200)
    - 13 digits: unix_milli or cocoa_sec
    - 16-17 digits: unix_micro, cocoa_nano, webkit_micro
    - 19 digits: unix_nano

    Args:
        page: Raw byte data from page
        interpret_timestamp_best: Function to interpret timestamp values

    Returns:
        List of (offset, epoch_value, kind, human_readable) tuples
    """
    out = []
    for m in _NUM_RE.finditer(page):
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

        except Exception:
            continue

        kind, epoch, human = interpret_timestamp_best(raw)
        if epoch is not None:
            out.append((m.start(), kind, epoch, human))
    return out


def find_urls(page: bytes) -> list[tuple[int, str]]:
    """
    Find HTTP/HTTPS URLs in page data.

    Args:
        page: Raw byte data from page

    Returns:
        List of (offset, url_string) tuples
    """
    out = []
    for m in URL_RE.finditer(page):
        try:
            s = m.group(0).decode("utf-8", errors="replace")
            out.append((m.start(), s))
        except Exception:
            continue
    return out


def find_text_runs(page: bytes, min_len: int = 6) -> list[tuple[int, int, str]]:
    """
    Locate printable text runs in page data.

    Args:
        page: Raw byte data from page
        min_len: Minimum text length to extract

    Returns:
        List of (start_offset, end_offset, text) tuples
    """
    out: list[tuple[int, int, str]] = []
    segments: list[tuple[int, bytes]] = []
    start = 0
    for i, b in enumerate(page):
        if b < 32 and b not in (9, 10, 13):
            if i > start:
                segments.append((start, page[start:i]))
            start = i + 1
    if start < len(page):
        segments.append((start, page[start:]))

    whitespace = b" \t\r\n"

    for off, chunk in segments:
        if not chunk:
            continue

        lead = 0
        while lead < len(chunk) and chunk[lead] in whitespace:
            lead += 1

        trail = 0
        while trail < len(chunk) - lead and chunk[-(trail + 1)] in whitespace:
            trail += 1

        if lead + trail >= len(chunk):
            continue

        core = chunk[lead : len(chunk) - trail]
        try:
            text = core.decode("utf-8", errors="replace")
        except Exception:
            continue

        if len(text) < min_len or not any(c.isalpha() for c in text):
            continue

        start_off = off + lead
        end_off = start_off + len(core)
        out.append((start_off, end_off, text))

    return out


def _read_varint_soft(buf: bytes, pos: int) -> tuple[int | None, int]:
    """
    Read a protobuf-style varint from buffer.

    Args:
        buf: Byte buffer
        pos: Starting position

    Returns:
        Tuple of (value, new_position) or (None, original_position) if invalid
    """
    result = 0
    shift = 0
    i = pos
    while i < len(buf) and shift <= 70:
        b = buf[i]
        result |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            return result, i
        shift += 7
    return None, pos


def find_blob_candidates(page: bytes, min_len: int = 16) -> list[tuple[int, bytes]]:
    """
    Find potential protobuf blob candidates using varint detection.

    Args:
        page: Raw byte data from page
        min_len: Minimum blob length

    Returns:
        List of (offset, blob_data) tuples
    """
    out = []
    i = 0
    while i < len(page):
        key, j = _read_varint_soft(page, i)
        if key is None:
            i += 1
            continue
        if (key & 0x7) != 2:
            i = j
            continue
        ln, k = _read_varint_soft(page, j)
        if ln is None or ln <= 0 or k + ln > len(page):
            i = j
            continue
        blob = page[k : k + ln]
        if len(blob) >= min_len:
            out.append((i, blob))
        i = k + ln
    return out
