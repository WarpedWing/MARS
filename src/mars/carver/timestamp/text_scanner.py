#!/usr/bin/env python3
"""
Text Timestamp Scanner
by WarpedWing Labs

Scans text for numeric values that might be timestamps and evaluates them
using field name context (JSON keys, XML keys, etc.) for improved classification.
"""

from __future__ import annotations

import re

from mars.carver.timestamp.classifier import evaluate_value_as_timestamp


def scan_text_for_timestamps(text: str, context_urls: list[str] | None = None) -> dict:
    """
    Scan text for numeric values that might be timestamps and evaluate them.

    Extracts field names from surrounding context (JSON keys, XML keys, etc.)
    to improve classification accuracy.

    Args:
        text: Text to scan (URL, text run, etc.)
        context_urls: Optional list of URLs for context

    Returns:
        Dictionary with:
            - timestamp_count: Number of timestamps found
            - timestamp_fields: JSON-serializable list of timestamp objects
    """
    # First, find all numbers and their positions
    # Match 10+ digit numbers with flexible boundaries to handle quoted strings
    # (?<![.\d]) = not preceded by dot or digit (avoid decimals/longer numbers)
    # (?![.\d]) = not followed by dot or digit
    number_pattern = r"(?<![.\d])\d{10,}(?![.\d])"
    number_matches = [(int(m.group()), m.start()) for m in re.finditer(number_pattern, text)]

    if not number_matches:
        return {"timestamp_count": 0, "timestamp_fields": []}

    matches = []
    all_values = [val for val, _ in number_matches]

    # For each number, extract its field name from context
    for value, pos in number_matches:
        # Get up to 200 chars before this number's position
        context_start = max(0, pos - 200)
        context_before = text[context_start:pos]

        # Extract field name from context
        # Look for patterns like:
        # - JSON: "field_name": 1234567890 or "field_name":"1234567890"
        # - XML/plist: <key>field-name</key><integer>1234567890</integer>
        # - Other: field_name = 1234567890

        field_name = None

        # Try JSON pattern: "field": or 'field': or "field":" or 'field':"
        # The final ["\']? handles numbers stored as strings in JSON
        json_match = re.search(r'["\']([a-zA-Z_\-]+)["\']\s*:\s*["\']?$', context_before)
        if json_match:
            field_name = json_match.group(1)

        # Try XML/plist pattern: <key>field</key> ... <integer>
        # Find ALL <key> tags and take the LAST one (closest to the number)
        elif "<key>" in context_before:
            xml_matches = re.findall(r"<key>([a-zA-Z_\-]+)</key>", context_before)
            if xml_matches:
                field_name = xml_matches[-1]  # Take the last/closest match

        # Try simple assignment pattern: field = or field:
        elif not field_name:
            assign_match = re.search(r"([a-zA-Z_\-]+)\s*[:=]\s*$", context_before)
            if assign_match:
                field_name = assign_match.group(1)

        matches.append((value, field_name))

    if not matches:
        return {"timestamp_count": 0, "timestamp_fields": []}

    # Evaluate each value as a potential timestamp
    timestamp_fields = []
    seen_values = set()  # Deduplicate

    for value, field_name in matches:
        # Skip duplicates
        if value in seen_values:
            continue
        seen_values.add(value)

        result = evaluate_value_as_timestamp(
            value=value,
            field_name=field_name,
            context_urls=context_urls or [],
            all_values=all_values,
        )

        if result:
            timestamp_fields.append(result)

    return {
        "timestamp_count": len(timestamp_fields),
        "timestamp_fields": timestamp_fields,
    }
