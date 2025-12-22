#!/usr/bin/env python3

"""
Timestamp Classifier V2 - Integrated Classification System
by WarpedWing Labs

Combines Unfurl, time_decode, Snowflake detection, and field analysis
for accurate timestamp vs ID classification.
"""

from __future__ import annotations

import re

# Import our modules
from mars.carver.timestamp.types import (
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
from mars.carver.unfurl.analyzer import URLContext, analyze_urls_in_page
from mars.carver.unfurl.id_detectors import detect_id_pattern

# Field name patterns
TIMESTAMP_FIELD_KEYWORDS = [
    b"time",  # Catches: time, endTime, startTime, etc.
    b"date",
    b"creat",  # Catches: created, create_time, creation
    b"modif",  # Catches: modified, modify_time, modification
    b"updat",  # Catches: updated, update_time
    b"expir",  # Catches: expire, expiry, expires, expiration
    b"last",
    b"timestamp",
    b"start",  # Catches: started, start_time
    b"end",  # Catches: ended, end_time
    b"finish",  # Catches: finished, finish_time
    b"access",  # Catches: accessed, access_time
    b"chang",  # Catches: changed, change_time
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
    b"observ",  # Catches: observed, observe_time
    b"receiv",  # Catches: received, receive_time
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

# Special fields that are IDs but may contain embedded timestamps
# These should be logged as BOTH an ID and a potential timestamp
DUAL_NATURE_FIELDS = [
    b"notif_id",  # Facebook notification IDs (often unix_micro)
    b"notification_id",  # May contain timestamps on some platforms
    b"event_id",  # Some platforms embed timestamps in event IDs
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
    # Check for timestamp keywords
    if any(kw in field_name for kw in TIMESTAMP_FIELD_KEYWORDS):
        return TimestampClassification.CONFIRMED_TIMESTAMP, REASON_FIELD_NAME_TIMESTAMP

    # Check for ID keywords
    if any(kw in field_name for kw in ID_FIELD_KEYWORDS):
        return TimestampClassification.CONFIRMED_ID, REASON_FIELD_NAME_ID

    return None, None


def format_timestamp_with_time_decode(value: int | float) -> tuple[str | None, str]:
    """
    Format timestamp by detecting common Unix timestamp formats.
    Returns (format_type, human_readable).
    """
    from datetime import UTC, datetime

    try:
        # Try Chrome/WebKit format first (microseconds since Jan 1, 1601 UTC)
        # These are 17-digit numbers like 13183922580082023
        if 13000000000000000 <= value <= 14000000000000000:
            try:
                WEBKIT_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)
                delta_seconds = value / 1_000_000  # Convert microseconds to seconds
                timestamp = WEBKIT_EPOCH.timestamp() + delta_seconds
                dt = datetime.fromtimestamp(timestamp, tz=UTC)
                if datetime(2000, 1, 1, tzinfo=UTC) <= dt <= datetime(2035, 1, 1, tzinfo=UTC):
                    return "webkit_micro", dt.strftime("%Y-%m-%d %H:%M:%S GMT")
            except Exception:
                pass

        # Try common Unix formats (since Jan 1, 1970 UTC)
        for divisor, fmt_name in [
            (1, "unix_sec"),
            (1000, "unix_milli"),
            (1000000, "unix_micro"),
            (1000000000, "unix_nano"),
        ]:
            try:
                dt = datetime.fromtimestamp(value / divisor, tz=UTC)
                if datetime(2010, 1, 1, tzinfo=UTC) <= dt <= datetime(2035, 1, 1, tzinfo=UTC):
                    return fmt_name, dt.strftime("%Y-%m-%d %H:%M:%S GMT")
            except Exception:
                continue
        return None, "Invalid timestamp"
    except Exception:
        return None, "Parse error"


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

    is_url_id, source_url = url_context.is_confirmed_id(int(value) if isinstance(value, float) else value)
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
                    reason=field_reason or REASON_FIELD_NAME_TIMESTAMP,
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
                reason=field_reason or REASON_FIELD_NAME_ID,
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
        Note: Some values may appear twice if they have dual nature (ID + timestamp)
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

        # Check if this is a dual-nature field (ID that may contain timestamp)
        if finding.classification == TimestampClassification.CONFIRMED_ID:
            # Check field name
            field_name = extract_field_name(page, offset, lookback=50)
            if field_name and field_name in DUAL_NATURE_FIELDS:
                # Also log as a potential timestamp
                fmt_type, human = format_timestamp_with_time_decode(value)
                if fmt_type:  # Only if it's a valid timestamp format
                    timestamp_finding = TimestampFindings(
                        offset=offset,
                        value=value,
                        classification=TimestampClassification.LIKELY_TIMESTAMP,
                        format_type=fmt_type,
                        human_readable=human,
                        reason=f"Dual-nature field '{field_name.decode('utf-8', errors='ignore')}' - ID that may contain embedded timestamp",
                        source_url=finding.source_url,
                        field_name=field_name.decode("utf-8", errors="ignore"),
                    )
                    results.append(timestamp_finding)

            # Also check URL context for dual-nature params
            normalized_value = int(value) if isinstance(value, float) else value
            is_url_id, source_url = url_context.is_confirmed_id(normalized_value)
            if is_url_id and source_url:
                # Check if the URL param name suggests dual nature
                # Extract param name from URL
                try:
                    from urllib.parse import parse_qs, urlparse

                    parsed = urlparse(source_url)
                    params = parse_qs(parsed.query)
                    for param_name, param_values in params.items():
                        if str(value) in [str(v) for v in param_values]:
                            param_name_lower = param_name.lower().encode("utf-8")
                            if param_name_lower in DUAL_NATURE_FIELDS:
                                # Also log as potential timestamp
                                fmt_type, human = format_timestamp_with_time_decode(value)
                                if fmt_type:
                                    timestamp_finding = TimestampFindings(
                                        offset=offset,
                                        value=value,
                                        classification=TimestampClassification.LIKELY_TIMESTAMP,
                                        format_type=fmt_type,
                                        human_readable=human,
                                        reason=f"Dual-nature URL parameter '{param_name}' - ID that may contain embedded timestamp",
                                        source_url=source_url,
                                        field_name=param_name,
                                    )
                                    results.append(timestamp_finding)
                                    break
                except Exception:
                    pass  # If URL parsing fails, skip dual-nature check

    return results


# ---------------- Protobuf Message Scanning (V2) ----------------


def evaluate_value_as_timestamp(
    value: int,
    field_name: str | None = None,
    context_urls: list[str] | None = None,
    all_values: list[int] | None = None,
) -> dict | None:
    """
    Check if a numeric value is a valid timestamp and classify it.

    This is a general-purpose helper that can be used for:
    - Protobuf field values (we don't know what they are yet)
    - Text-extracted numbers (value_text in carved_all)
    - URL-extracted numbers

    Args:
        value: The numeric value to check
        field_name: Optional field name/identifier (e.g., "12" or "created_time")
        context_urls: Optional list of URLs for context
        all_values: Optional list of all values (for sequential ID detection)

    Returns:
        Dictionary with:
            - value: Raw numeric value
            - format: Timestamp format (unix_sec, unix_milli, etc.)
            - human_readable: Formatted timestamp string
            - field_name: Field identifier (if provided)
            - classification: Classification result (ambiguous, likely_timestamp, etc.)
        Returns None if not a valid timestamp format.
    """
    context_urls = context_urls or []
    all_values = all_values or []

    # Validate if this is a valid timestamp
    fmt_type, human = format_timestamp_with_time_decode(value)

    if not fmt_type:
        # Not a valid timestamp format
        return None

    # Build field path for classification
    field_path = field_name if field_name else str(value)

    # Classify the timestamp
    classification = classify_protobuf_timestamp(
        field_path=field_path,
        value=value,
        protobuf_urls=context_urls,
        all_values=all_values,
    )

    return {
        "value": value,
        "format": fmt_type,
        "human_readable": human,
        "field_name": field_name,
        "classification": classification.value,  # Convert enum to string
    }


def extract_numbers_with_field_names(
    message: dict | list | object,
    field_path: str = "",
) -> list[tuple[str, int | float]]:
    """
    Recursively extract all numeric values from a protobuf message along with their field paths.

    This preserves the context of where each number appears in the message structure,
    allowing us to use field names for timestamp vs ID classification.

    Args:
        message: Decoded protobuf message (dict, list, or primitive)
        field_path: Current field path (for recursion tracking)

    Returns:
        List of (field_path, value) tuples
        Examples:
            - ("1", 1234567890)  # Numeric field ID
            - ("user.created_time", 1234567890)  # String field name
            - ("items.0.timestamp", 1234567890)  # Array index notation
    """
    results = []

    if isinstance(message, dict):
        for key, value in message.items():
            # Build field path
            new_path = f"{field_path}.{key}" if field_path else str(key)

            # Recursively extract from nested structures
            results.extend(extract_numbers_with_field_names(value, new_path))

    elif isinstance(message, list):
        for idx, item in enumerate(message):
            # Add array index to path
            new_path = f"{field_path}.{idx}" if field_path else str(idx)
            results.extend(extract_numbers_with_field_names(item, new_path))

    elif isinstance(message, (int, float)):
        # Found a number - return with its field path
        results.append((field_path, message))

    return results


def validate_protobuf_timestamps(
    message: dict,
) -> dict:
    """
    Validate numbers in a protobuf message as timestamps using V2 classifier.

    Scans each field in the protobuf, checking if the value is a valid timestamp,
    and classifies it based on field name context and URL presence.

    Args:
        message: Decoded protobuf message (dict from blackboxprotobuf)

    Returns:
        Dictionary with:
            - timestamp_count: Number of valid timestamps found
            - timestamp_fields: Array of timestamp objects with:
                - value: Raw numeric value
                - format: Timestamp format (unix_sec, unix_milli, etc.)
                - human_readable: Formatted timestamp string
                - field_name: Field identifier (numeric ID or string name)
                - classification: Classification result (ambiguous, likely_timestamp, etc.)
    """
    # Extract all numbers with their field paths
    numbers_with_paths = extract_numbers_with_field_names(message)

    if not numbers_with_paths:
        return {"timestamp_count": 0, "timestamp_fields": []}

    # Extract all strings (to find URLs for context)
    try:
        from mars.carver.protobuf.decoder import extract_strings_from_message

        all_strings = extract_strings_from_message(message)
    except Exception:
        all_strings = []

    # Find URLs in the protobuf
    protobuf_urls = []
    for string in all_strings:
        # Simple URL detection
        if any(string.startswith(proto) for proto in ["http://", "https://", "ftp://"]):
            protobuf_urls.append(string)

    # Collect all integer values for sequential ID detection
    all_values = [int(val) for _, val in numbers_with_paths if isinstance(val, int)]

    timestamp_fields = []

    for field_path, value in numbers_with_paths:
        # Skip non-integer values (timestamps are always integers)
        if not isinstance(value, int):
            continue

        # Extract the leaf field name (last part of the path)
        # e.g., "user.created_time" -> "created_time" or "12" -> "12"
        leaf_name = field_path.split(".")[-1]

        # Use the helper to evaluate this value
        result = evaluate_value_as_timestamp(
            value=value,
            field_name=leaf_name,
            context_urls=protobuf_urls,
            all_values=all_values,
        )

        if result:
            timestamp_fields.append(result)

    return {
        "timestamp_count": len(timestamp_fields),
        "timestamp_fields": timestamp_fields,
    }


def classify_protobuf_timestamp(
    field_path: str,
    value: int,
    protobuf_urls: list[str],
    all_values: list[int],
) -> TimestampClassification:
    """
    Classify a protobuf timestamp based on available context.

    Args:
        field_path: Full field path (e.g., "12" or "user.created_time")
        value: The numeric value
        protobuf_urls: List of URLs found in the protobuf
        all_values: All integer values in the protobuf (for sequential detection)

    Returns:
        TimestampClassification enum value
    """
    # Extract leaf name
    leaf_name = field_path.split(".")[-1]

    # Step 1: Check if field name suggests timestamp (for string field names)
    if not leaf_name.isdigit():
        # We have a string field name - check keywords
        field_bytes = field_path.lower().encode("utf-8")

        # Check for timestamp keywords
        for keyword in TIMESTAMP_FIELD_KEYWORDS:
            if keyword in field_bytes:
                return TimestampClassification.LIKELY_TIMESTAMP

        # Check for ID keywords
        for keyword in ID_FIELD_KEYWORDS:
            if keyword in field_bytes:
                # Field name suggests this is an ID, not a timestamp
                # But we already validated it as a timestamp format
                # This is ambiguous
                return TimestampClassification.AMBIGUOUS

    # Step 2: Check if this value appears in any URLs
    value_str = str(value)
    for url in protobuf_urls:
        if value_str in url:
            # Value appears in URL - check if it's in a timestamp-like param
            try:
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(url)
                params = parse_qs(parsed.query)

                for param_name, param_values in params.items():
                    if value_str in [str(v) for v in param_values]:
                        # Check if param name suggests timestamp
                        param_bytes = param_name.lower().encode("utf-8")
                        for keyword in TIMESTAMP_FIELD_KEYWORDS:
                            if keyword in param_bytes:
                                return TimestampClassification.CONFIRMED_TIMESTAMP

                        # Check if param name suggests ID
                        for keyword in ID_FIELD_KEYWORDS:
                            if keyword in param_bytes:
                                # URL param is an ID, but value is valid timestamp
                                # Could be dual-nature (Snowflake ID)
                                return TimestampClassification.AMBIGUOUS
            except Exception:
                pass

    # Step 3: Check for sequential IDs (Snowflake detection)
    # If there are multiple values close to each other, might be IDs
    if len(all_values) >= 2:
        # Check if this value is part of a sequence
        for other_value in all_values:
            if other_value != value:
                # If values are within 1% of each other, might be sequential IDs
                ratio = abs(value - other_value) / max(value, other_value)
                if ratio < 0.01:  # Within 1%
                    # Likely sequential IDs, not timestamps
                    return TimestampClassification.AMBIGUOUS

    # Step 4: No clear context - classify as AMBIGUOUS
    # It's a valid timestamp format, but we can't determine if it's
    # actually a timestamp or an ID that happens to be in timestamp range
    return TimestampClassification.AMBIGUOUS
