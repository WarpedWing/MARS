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


def is_likely_protobuf(blob: bytes, min_fields: int = 2) -> bool:
    """
    Quick heuristic check if blob looks like valid protobuf.

    Args:
        blob: Binary data to check
        min_fields: Minimum number of fields required to consider it protobuf

    Returns:
        True if blob looks like protobuf data
    """
    if not blob or len(blob) < 3:
        return False

    # Quick check: does it start with valid protobuf field tag?
    # Field tag = (field_number << 3) | wire_type
    # Valid wire types: 0, 1, 2, 5 (not 3, 4, 6, 7)
    first_byte = blob[0]
    wire_type = first_byte & 0x07

    if wire_type in (3, 4, 6, 7):
        return False  # Invalid wire type

    # Try to decode with bbpb to see if it's valid
    if not BBPB_AVAILABLE:
        return True  # Can't verify, assume valid

    try:
        message, typedef = blackboxprotobuf.decode_message(blob)
        # Check if we got a reasonable number of fields
        if isinstance(message, dict) and len(message) >= min_fields:
            return True
    except Exception:
        pass

    return False


def decode_protobuf_bbpb(blob: bytes) -> dict[str, Any] | None:
    """
    Decode protobuf blob using blackboxprotobuf.

    Returns dict with:
        - 'message': Decoded protobuf message (dict)
        - 'typedef': Type definition inferred by bbpb
        - 'json': JSON string of message
        - 'schema': Simplified schema showing field types

    Returns None if decoding fails or bbpb not available.
    """
    if not BBPB_AVAILABLE:
        return None

    if not blob:
        return None

    try:
        # Decode message
        message, typedef = blackboxprotobuf.decode_message(blob)

        # Must have at least 1 field to be useful
        if not isinstance(message, dict) or len(message) == 0:
            return None

        # Convert to JSON for storage
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


# Backwards compatibility with old protobuf_extractor.py interface
def maybe_decode_protobuf(blob: bytes, max_depth: int = 4) -> dict | None:
    """
    Backwards-compatible wrapper for existing code.

    Args:
        blob: Binary data to decode
        max_depth: Ignored (bbpb handles depth internally)

    Returns:
        Decoded message dict or None
    """
    result = decode_protobuf_bbpb(blob)
    if result:
        return result["message"]
    return None


def to_json(obj: Any, pretty: bool = True) -> str:
    """
    Backwards-compatible JSON serializer.

    Args:
        obj: Object to serialize (dict, list, etc.)
        pretty: Whether to pretty-print

    Returns:
        JSON string
    """
    if pretty:
        return json.dumps(obj, indent=2, default=str)
    return json.dumps(obj, default=str)
