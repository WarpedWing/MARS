#!/usr/bin/env python3

"""
Timestamp Classifier V2 - Integrated Classification System
by WarpedWing Labs

Combines Unfurl, time_decode, Snowflake detection, and field analysis
for accurate timestamp vs ID classification.
"""

from __future__ import annotations

import re

from id_detectors import detect_id_pattern

# Import our modules
from timestamp_classifier import (
    REASON_FIELD_NAME_ID,
    REASON_FIELD_NAME_TIMESTAMP,
    REASON_NO_VALID_FORMAT,
    REASON_SNOWFLAKE_STRUCTURE,
    REASON_URL_ID_FIELD,
    REASON_URL_PARAM_TIMESTAMP,
    REASON_VALID_FORMAT_NO_CONTEXT,
    TimestampClassification,
    TimestampFindings,
)
from url_analyzer import URLContext, analyze_urls_in_page

# Try to import time_decode
try:
    from time_decode import timestamp as time_decode_timestamp

    TIME_DECODE_AVAILABLE = True
except ImportError:
    TIME_DECODE_AVAILABLE = False


# Field name patterns
TIMESTAMP_FIELD_KEYWORDS = [
    b"time",  # Catches: time, endTime, startTime, etc.
    b"date",
    b"created",
    b"modified",
    b"updated",
    b"expire",  # Catches: expire, expiry, expires, expiration
    b"last",
    b"timestamp",
    b"started",
    b"ended",
    b"finished",
    b"accessed",
    b"changed",
    b"birth",
    b"mtime",
    b"ctime",
    b"atime",
    b"begin",
    b"ttl",  # Time to live
    b"valid",  # validFrom, validTo, validUntil
    b"duration",
    b"delay",
    b"timeout",
]

ID_FIELD_KEYWORDS = [
    b"id",
    b"uid",
    b"event_id",
    b"notification_id",
    b"user_id",
    b"msg_id",
    b"message_id",
    b"post_id",
    b"thread_id",
    b"session_id",
    b"token",
    b"key",
]

FIELD_PATTERN = re.compile(rb"([a-zA-Z_][a-zA-Z0-9_]{2,30})[\s\x00]*[:=]")


def extract_field_name(page: bytes, offset: int, lookback: int = 50) -> bytes | None:
    """Extract potential field name before a value."""
    start = max(0, offset - lookback)
    context = page[start:offset]

    matches = list(FIELD_PATTERN.finditer(context))
    if matches:
        return matches[-1].group(1).lower()
    return None


def classify_by_field_name(
    field_name: bytes,
) -> tuple[TimestampClassification | None, str | None]:
    """
    Classify based on field name alone.
    Returns (classification, reason) or (None, None) if inconclusive.
    """
    field_str = field_name.decode("utf-8", errors="ignore")

    # Check for timestamp keywords
    if any(kw in field_name for kw in TIMESTAMP_FIELD_KEYWORDS):
        return TimestampClassification.CONFIRMED_TIMESTAMP, REASON_FIELD_NAME_TIMESTAMP

    # Check for ID keywords
    if any(kw in field_name for kw in ID_FIELD_KEYWORDS):
        return TimestampClassification.CONFIRMED_ID, REASON_FIELD_NAME_ID

    return None, None


def format_timestamp_with_time_decode(value: int | float) -> tuple[str | None, str]:
    """
    Try to format timestamp using time_decode.
    Returns (format_type, human_readable).
    """
    if not TIME_DECODE_AVAILABLE:
        # Fallback to simple Unix timestamp
        from datetime import UTC, datetime

        try:
            # Try common formats
            for divisor, fmt_name in [
                (1, "unix_sec"),
                (1000, "unix_milli"),
                (1000000, "unix_micro"),
                (1000000000, "unix_nano"),
            ]:
                try:
                    dt = datetime.fromtimestamp(value / divisor, tz=UTC)
                    if (
                        datetime(2010, 1, 1, tzinfo=UTC)
                        <= dt
                        <= datetime(2035, 1, 1, tzinfo=UTC)
                    ):
                        return fmt_name, dt.strftime("%Y-%m-%d %H:%M:%S GMT")
                except Exception:
                    continue
            return None, "Invalid timestamp"
        except Exception:
            return None, "Parse error"

    # Use time_decode
    try:
        result = time_decode_timestamp.Timestamp(value)
        if result:
            return result.format_name, result.to_string()
    except Exception:
        pass

    return None, "Unknown format"


def classify_timestamp(
    page: bytes,
    offset: int,
    value: int | float,
    url_context: URLContext,
    url_offsets: list[tuple[int, str]],
    all_values: list[int],
) -> TimestampFindings:
    """
    Main classification function.

    Applies all detection methods in priority order:
    1. URL context (highest confidence)
    2. Field name analysis
    3. Snowflake ID detection
    4. time_decode format validation
    5. Ambiguous (valid format, no context)
    """

    # Step 1: Check URL context (HIGHEST PRIORITY)
    is_url_timestamp, source_url = url_context.is_confirmed_timestamp(value)
    if is_url_timestamp:
        fmt_type, human = format_timestamp_with_time_decode(value)
        return TimestampFindings(
            offset=offset,
            value=value,
            classification=TimestampClassification.CONFIRMED_TIMESTAMP,
            format_type=fmt_type or "url_extracted",
            human_readable=human,
            reason=REASON_URL_PARAM_TIMESTAMP,
            source_url=source_url,
            field_name=None,
        )

    is_url_id, source_url = url_context.is_confirmed_id(value)
    if is_url_id:
        return TimestampFindings(
            offset=offset,
            value=value,
            classification=TimestampClassification.CONFIRMED_ID,
            format_type="url_id",
            human_readable="ID extracted from URL",
            reason=REASON_URL_ID_FIELD,
            source_url=source_url,
            field_name=None,
        )

    # Step 2: Check field name
    field_name = extract_field_name(page, offset, lookback=50)
    if field_name:
        field_classification, field_reason = classify_by_field_name(field_name)
        if field_classification:
            if field_classification == TimestampClassification.CONFIRMED_TIMESTAMP:
                fmt_type, human = format_timestamp_with_time_decode(value)
                return TimestampFindings(
                    offset=offset,
                    value=value,
                    classification=field_classification,
                    format_type=fmt_type or "unknown",
                    human_readable=human,
                    reason=field_reason,
                    source_url=None,
                    field_name=field_name.decode("utf-8", errors="ignore"),
                )
            # CONFIRMED_ID
            return TimestampFindings(
                offset=offset,
                value=value,
                classification=field_classification,
                format_type="field_id",
                human_readable="ID from field name",
                reason=field_reason,
                source_url=None,
                field_name=field_name.decode("utf-8", errors="ignore"),
            )

    # Step 3: Check Snowflake ID pattern
    if isinstance(value, int):
        id_result = detect_id_pattern(value, all_values)
        if id_result.is_id and id_result.confidence >= 0.8:
            return TimestampFindings(
                offset=offset,
                value=value,
                classification=TimestampClassification.CONFIRMED_ID,
                format_type=id_result.id_type,
                human_readable=id_result.description,
                reason=REASON_SNOWFLAKE_STRUCTURE,
                source_url=None,
                field_name=None,
            )

    # Step 4: Check nearby URL (might be related)
    nearby_url = url_context.get_url_near_offset(offset, url_offsets, window=200)
    if nearby_url and nearby_url in url_context.url_infos:
        info = url_context.url_infos[nearby_url]
        # If near a URL with IDs, likely this is also an ID
        if info.ids and not info.timestamps:
            return TimestampFindings(
                offset=offset,
                value=value,
                classification=TimestampClassification.LIKELY_ID,
                format_type="near_url_id",
                human_readable="Near URL with IDs",
                reason="Found near URL containing IDs",
                source_url=nearby_url,
                field_name=None,
            )

    # Step 5: Try to validate as timestamp with time_decode
    fmt_type, human = format_timestamp_with_time_decode(value)
    if fmt_type:
        # Valid format, but no context
        return TimestampFindings(
            offset=offset,
            value=value,
            classification=TimestampClassification.AMBIGUOUS,
            format_type=fmt_type,
            human_readable=human,
            reason=REASON_VALID_FORMAT_NO_CONTEXT,
            source_url=None,
            field_name=None,
        )

    # Step 6: No valid format found
    return TimestampFindings(
        offset=offset,
        value=value,
        classification=TimestampClassification.INVALID,
        format_type=None,
        human_readable="Not a valid timestamp",
        reason=REASON_NO_VALID_FORMAT,
        source_url=None,
        field_name=None,
    )


def classify_page_timestamps(
    page: bytes,
    raw_candidates: list[tuple[int, int | float]],
    url_offsets: list[tuple[int, str]],
) -> list[TimestampFindings]:
    """
    Classify all timestamp candidates in a page.

    Args:
        page: Raw page bytes
        raw_candidates: List of (offset, value) tuples
        url_offsets: List of (offset, url_string) tuples

    Returns:
        List of TimestampFindings with classifications
    """
    # Build URL context
    url_context = analyze_urls_in_page(page, url_offsets)

    # Extract all values for sequential detection
    all_values = [int(val) for _, val in raw_candidates if isinstance(val, int)]

    # Classify each candidate
    results = []
    for offset, value in raw_candidates:
        finding = classify_timestamp(
            page=page,
            offset=offset,
            value=value,
            url_context=url_context,
            url_offsets=url_offsets,
            all_values=all_values,
        )
        results.append(finding)

    return results
