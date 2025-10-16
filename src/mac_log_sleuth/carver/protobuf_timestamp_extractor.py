#!/usr/bin/env python3

"""
Protobuf Timestamp Extractor
by WarpedWing Labs

Extracts likely timestamps from blind protobuf parsing.
Without schemas, we look for timestamp-like patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime


def looks_like_timestamp(value: int | float, field_name: str | None = None) -> tuple[bool, str | None]:
    """
    Check if a value looks like a timestamp.

    Returns: (is_timestamp, human_readable_or_none)
    """
    if not isinstance(value, (int, float)):
        return False, None

    # Field name hints
    if field_name:
        time_keywords = ['time', 'date', 'expire', 'expir', 'ttl', 'valid', 'duration']
        field_lower = field_name.lower()
        has_time_keyword = any(kw in field_lower for kw in time_keywords)
    else:
        has_time_keyword = False

    # Try common timestamp formats
    for divisor, fmt_name in [(1, "unix_sec"), (1000, "unix_milli"),
                               (1000000, "unix_micro"), (1000000000, "unix_nano")]:
        try:
            ts_sec = value / divisor
            dt = datetime.fromtimestamp(ts_sec, tz=UTC)

            # Must be in reasonable range (1990-2040)
            if datetime(1990, 1, 1, tzinfo=UTC) <= dt <= datetime(2040, 1, 1, tzinfo=UTC):
                # If field name suggests timestamp, accept it
                if has_time_keyword:
                    return True, dt.strftime("%Y-%m-%d %H:%M:%S GMT")

                # Otherwise, be more strict (2010-2030)
                if datetime(2010, 1, 1, tzinfo=UTC) <= dt <= datetime(2030, 1, 1, tzinfo=UTC):
                    return True, dt.strftime("%Y-%m-%d %H:%M:%S GMT")
        except (ValueError, OSError, OverflowError):
            continue

    return False, None


def extract_timestamps_from_protobuf(pb_data: dict, parent_key: str = "") -> list[tuple[str, int | float, str]]:
    """
    Recursively extract timestamp-like values from protobuf data.

    Args:
        pb_data: Parsed protobuf dictionary
        parent_key: Parent key for nested structures

    Returns:
        List of (field_path, value, human_readable) tuples
    """
    timestamps = []

    if isinstance(pb_data, dict):
        for key, value in pb_data.items():
            field_path = f"{parent_key}.{key}" if parent_key else key

            # Check if key suggests timestamp
            field_suggests_time = any(kw in key.lower() for kw in
                                     ['time', 'date', 'expire', 'expir', 'ttl', 'valid'])

            if isinstance(value, dict):
                # Check for protobuf number interpretations
                if 'u64' in value:
                    is_ts, human = looks_like_timestamp(value['u64'], key)
                    if is_ts:
                        timestamps.append((f"{field_path}.u64", value['u64'], human))

                if 'u32' in value:
                    is_ts, human = looks_like_timestamp(value['u32'], key)
                    if is_ts:
                        timestamps.append((f"{field_path}.u32", value['u32'], human))

                # Recurse
                timestamps.extend(extract_timestamps_from_protobuf(value, field_path))

            elif isinstance(value, (int, float)):
                is_ts, human = looks_like_timestamp(value, key)
                if is_ts:
                    timestamps.append((field_path, value, human))

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        timestamps.extend(extract_timestamps_from_protobuf(
                            item, f"{field_path}[{i}]"))

    return timestamps


def analyze_protobuf_for_timestamps(json_data: str | dict) -> dict:
    """
    Analyze protobuf JSON data for timestamps.

    Args:
        json_data: Either JSON string or parsed dict

    Returns:
        Dictionary with analysis results
    """
    import json

    if isinstance(json_data, str):
        try:
            pb_data = json.loads(json_data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "timestamps": []}
    else:
        pb_data = json_data

    timestamps = extract_timestamps_from_protobuf(pb_data)

    return {
        "has_timestamps": len(timestamps) > 0,
        "timestamp_count": len(timestamps),
        "timestamps": [
            {
                "field": field,
                "value": value,
                "human_readable": human
            }
            for field, value, human in timestamps
        ],
        "raw_protobuf": pb_data
    }


def should_keep_protobuf(pb_data: dict) -> tuple[bool, str]:
    """
    Determine if a protobuf entry is worth keeping.

    Returns: (keep, reason)
    """
    timestamps = extract_timestamps_from_protobuf(pb_data)

    if timestamps:
        return True, f"Contains {len(timestamps)} timestamp(s)"

    # Check if it has any recognizable structure
    def count_fields(d, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return 0
        count = 0
        if isinstance(d, dict):
            count += len(d)
            for v in d.values():
                if isinstance(v, (dict, list)):
                    count += count_fields(v, depth + 1)
        elif isinstance(d, list):
            for item in d:
                count += count_fields(item, depth + 1)
        return count

    field_count = count_fields(pb_data)

    # Keep if it has significant structure (at least 10 meaningful fields)
    # This filters out small garbage protobufs with generic field names like f1, f2, etc.
    if field_count >= 10:
        # Check if field names are mostly generic (f1, f2, f13, etc.)
        if isinstance(pb_data, dict):
            generic_count = sum(
                1 for k in pb_data.keys() if isinstance(k, str) and k.startswith("f") and k[1:].isdigit()
            )
            # If more than 50% of fields are generic, probably noise unless it's huge
            if generic_count > len(pb_data) * 0.5 and field_count < 20:
                return False, "Generic field names (f1, f2, etc.), likely noise"

        return True, f"Has {field_count} fields"

    # Otherwise, probably noise
    return False, "Too few fields, likely noise"


# Example usage
if __name__ == "__main__":
    # Test with your example
    test_pb = {
        "f13": {"u64": 2308793137325292919, "f64": 2.4688003283849355e-154},
        "f4": [32, 32, 34],
        "f12": {"u32": 1919512696, "f32": 4.623742314505484e+30},
        "f15": {"u64": 3761972665823279650, "f64": 3.379226598219963e-57}
    }

    result = analyze_protobuf_for_timestamps(test_pb)
    print(f"Has timestamps: {result['has_timestamps']}")
    print(f"Timestamps found: {result['timestamp_count']}")
    for ts in result['timestamps']:
        print(f"  {ts['field']}: {ts['value']} -> {ts['human_readable']}")

    # Test with timestamp example
    test_pb2 = {
        "state": 1,
        "endTime": 1592503917395,
        "fileSize": 392223
    }

    result2 = analyze_protobuf_for_timestamps(test_pb2)
    print(f"\nSecond example - Has timestamps: {result2['has_timestamps']}")
    for ts in result2['timestamps']:
        print(f"  {ts['field']}: {ts['value']} -> {ts['human_readable']}")
