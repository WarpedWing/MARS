#!/usr/bin/env python3

"""
ID Pattern Detectors
by WarpedWing Labs

Detects specific ID formats (Snowflake, UUID, etc.) that can be confused with timestamps.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import NamedTuple


class IDDetectionResult(NamedTuple):
    """Result from ID pattern detection"""

    is_id: bool
    id_type: str | None  # 'snowflake', 'uuid', 'sequential', etc.
    confidence: float  # 0.0-1.0
    description: str
    embedded_timestamp: float | None = None  # For Snowflakes


# Timestamp ranges for validation
DISCORD_EPOCH = 1420070400  # 2015-01-01 (Discord/Twitter Snowflake epoch)


def detect_snowflake_id(value: int) -> IDDetectionResult:
    """
    Detect if a value is a Snowflake ID (Twitter, Discord, Instagram).

    Snowflake structure (64-bit):
    - Bits 63-22: Milliseconds since epoch (41 bits)
    - Bits 21-17: Datacenter ID (5 bits)
    - Bits 16-12: Worker ID (5 bits)
    - Bits 11-0: Sequence (12 bits)

    Returns IDDetectionResult with embedded timestamp if valid.
    """
    # Snowflakes are typically 18-19 digits
    if not (18 <= len(str(abs(value))) <= 19):
        return IDDetectionResult(False, None, 0.0, "Wrong digit count")

    try:
        # Extract timestamp (first 41 bits, shifted right 22)
        timestamp_ms = (value >> 22) + (DISCORD_EPOCH * 1000)
        timestamp_sec = timestamp_ms / 1000.0

        # Validate timestamp is reasonable (2015-2030)
        dt = datetime.fromtimestamp(timestamp_sec, tz=UTC)
        if not (datetime(2015, 1, 1, tzinfo=UTC) <= dt <= datetime(2030, 1, 1, tzinfo=UTC)):
            return IDDetectionResult(False, None, 0.0, "Embedded timestamp out of range")

        # Extract other components
        datacenter = (value >> 17) & 0x1F  # 5 bits
        worker = (value >> 12) & 0x1F  # 5 bits
        sequence = value & 0xFFF  # 12 bits

        # Snowflakes should have reasonable component values
        # Datacenter and worker are typically < 32, sequence < 4096
        if datacenter > 31 or worker > 31 or sequence > 4095:
            return IDDetectionResult(False, None, 0.3, "Invalid Snowflake components")

        description = (
            f"Snowflake ID (dc:{datacenter}, worker:{worker}, seq:{sequence}, "
            f"created:{dt.strftime('%Y-%m-%d %H:%M:%S')})"
        )

        return IDDetectionResult(
            is_id=True,
            id_type="snowflake",
            confidence=0.95,
            description=description,
            embedded_timestamp=timestamp_sec,
        )

    except Exception as e:
        return IDDetectionResult(False, None, 0.0, f"Parse error: {e}")


def detect_uuid(value: str | int) -> IDDetectionResult:
    """
    Detect if a value is a UUID.

    UUIDs are 128-bit identifiers, often seen as:
    - Hex strings: "550e8400-e29b-41d4-a716-446655440000"
    - Integers: very large (38-39 digits)
    """
    val_str = str(value)

    # UUID as hex string
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    if uuid_pattern.match(val_str):
        # Check UUID version (4th group, first hex digit)
        version = val_str[14]
        return IDDetectionResult(
            is_id=True,
            id_type=f"uuid_v{version}",
            confidence=1.0,
            description=f"UUID version {version}",
        )

    # UUID as integer (128-bit = 38-39 digits)
    if isinstance(value, int) and 38 <= len(val_str) <= 39:
        return IDDetectionResult(
            is_id=True,
            id_type="uuid_int",
            confidence=0.8,
            description="Possible UUID (integer form)",
        )

    return IDDetectionResult(False, None, 0.0, "Not a UUID")


def detect_sequential_id(value: int, all_values: list[int], tolerance: int = 1000) -> IDDetectionResult:
    """
    Detect if a value is part of a sequential ID series.

    Sequential IDs are incremental (1, 2, 3...) or have small gaps.
    Real timestamps are scattered.

    Args:
        value: Value to check
        all_values: All numeric values found in page
        tolerance: Max gap between sequential IDs
    """
    if len(all_values) < 3:
        return IDDetectionResult(False, None, 0.0, "Not enough values")

    # Find values near this one
    nearby = [v for v in all_values if abs(v - value) <= tolerance * 2]

    if len(nearby) < 3:
        return IDDetectionResult(False, None, 0.0, "No nearby values")

    # Check if they're sequential
    sorted_nearby = sorted(nearby)
    gaps = [sorted_nearby[i + 1] - sorted_nearby[i] for i in range(len(sorted_nearby) - 1)]

    # Sequential IDs have consistent small gaps
    avg_gap = sum(gaps) / len(gaps)
    if avg_gap < tolerance:
        confidence = max(0.3, min(0.9, 1.0 - (avg_gap / tolerance)))
        return IDDetectionResult(
            is_id=True,
            id_type="sequential",
            confidence=confidence,
            description=f"Sequential ID pattern (avg gap: {avg_gap:.0f})",
        )

    return IDDetectionResult(False, None, 0.0, "Not sequential")


def detect_id_pattern(value: int | str, all_values: list[int] | None = None) -> IDDetectionResult:
    """
    Main entry point: Try all ID detection methods.

    Returns the highest-confidence match.
    """
    if not isinstance(value, int):
        # Try UUID detection for strings
        return detect_uuid(value)

    # Try each detector
    results = [
        detect_snowflake_id(value),
        detect_uuid(value),
    ]

    # Add sequential detection if we have context
    if all_values:
        results.append(detect_sequential_id(value, all_values))

    # Return highest confidence match
    best = max(results, key=lambda r: r.confidence)

    if best.confidence >= 0.5:
        return best

    # No match
    return IDDetectionResult(False, None, 0.0, "No ID pattern detected")
