#!/usr/bin/env python3

"""
Protobuf Decoder using BlackBoxProtobuf
by WarpedWing Labs

Uses blackboxprotobuf (bbpb) for robust, schema-agnostic protobuf decoding.
This provides much better type inference than manual varint parsing.

Exports:
  - decode_protobuf_bbpb(blob: bytes) -> dict | None
      Decode a blob using blackboxprotobuf. Returns decoded message + typedef.

  - is_likely_protobuf(blob: bytes) -> bool
      Quick heuristic check if blob looks like valid protobuf data.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

try:
    import blackboxprotobuf

    BBPB_AVAILABLE = True
except ImportError:
    BBPB_AVAILABLE = False


def decode_protobuf_bbpb(blob: bytes) -> dict[str, Any] | None:
    """
    Decode protobuf blob using blackboxprotobuf.

    Returns dict with:
        - 'message': Decoded protobuf message (dict)
        - 'typedef': Type definition inferred by bbpb
        - 'json': JSON string of message (using bbpb's json_safe_transform)
        - 'schema': Simplified schema showing field types

    Returns None if decoding fails or bbpb not available.
    """
    if not BBPB_AVAILABLE or not blackboxprotobuf:  # type: ignore
        return None

    if not blob:
        return None

    try:
        # Decode message (we need the dict for extract_strings/numbers)
        message, typedef = blackboxprotobuf.decode_message(blob)  # type: ignore

        # Must have at least 1 field to be useful
        if not isinstance(message, dict) or len(message) == 0:
            return None

        # Use bbpb's protobuf_to_json for better binary/bytes handling
        # Returns (json_string, typedef) tuple according to bbpb docs
        try:
            json_str, _ = blackboxprotobuf.protobuf_to_json(blob, typedef)  # type: ignore
        except Exception:
            # Fallback to manual JSON conversion if protobuf_to_json fails
            json_str = json.dumps(message, indent=2, default=str)

        # Create simplified schema (more human-readable than typedef)
        schema = {}
        if isinstance(typedef, (dict, OrderedDict)):
            for field_num, field_info in typedef.items():
                if isinstance(field_info, (dict, OrderedDict)):
                    field_type = field_info.get("type", "unknown")
                    schema[field_num] = field_type

        return {
            "message": message,
            "typedef": typedef,
            "json": json_str,
            "schema": schema,
        }

    except Exception:
        return None


def extract_strings_from_message(message: dict | list | Any) -> list[str]:
    """
    Recursively extract all string values from decoded protobuf message.

    Useful for finding interesting text content without manually inspecting structure.
    """
    strings = []

    if isinstance(message, dict):
        for value in message.values():
            strings.extend(extract_strings_from_message(value))
    elif isinstance(message, list):
        for item in message:
            strings.extend(extract_strings_from_message(item))
    elif isinstance(message, str):
        # Only keep strings with some length and printable characters
        if len(message) >= 3 and any(c.isalnum() for c in message):
            strings.append(message)
    elif isinstance(message, bytes):
        # Try to decode bytes as UTF-8
        try:
            decoded = message.decode("utf-8", errors="ignore")
            if len(decoded) >= 3 and any(c.isalnum() for c in decoded):
                strings.append(decoded)
        except Exception:
            pass

    return strings


def extract_numbers_from_message(message: dict | list | Any) -> list[int | float]:
    """
    Recursively extract all numeric values from decoded protobuf message.

    Useful for finding potential timestamps without manual inspection.
    """
    numbers = []

    if isinstance(message, dict):
        for value in message.values():
            numbers.extend(extract_numbers_from_message(value))
    elif isinstance(message, list):
        for item in message:
            numbers.extend(extract_numbers_from_message(item))
    elif isinstance(message, (int, float)):
        numbers.append(message)

    return numbers
