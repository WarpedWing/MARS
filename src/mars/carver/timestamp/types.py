#!/usr/bin/env python3

"""
Timestamp Classification System v2
by WarpedWing Labs

Replaces fuzzy confidence scores with clear categorical classifications.
Uses Unfurl and time_decode for accurate timestamp vs ID detection.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class TimestampClassification(Enum):
    """
    Clear categorical classification for timestamp candidates.
    No fuzzy scores - definitive labels.
    """

    # High confidence - Can trust these
    CONFIRMED_TIMESTAMP = "confirmed_timestamp"  # Field name + format match, or URL param
    CONFIRMED_ID = "confirmed_id"  # Known ID pattern (Snowflake, UUID) or URL ID field

    # Medium confidence - Likely correct
    LIKELY_TIMESTAMP = "likely_timestamp"  # Field name suggests timestamp, valid format
    LIKELY_ID = "likely_id"  # Field name suggests ID, or sequential pattern

    # Low confidence - Need manual review
    AMBIGUOUS = "ambiguous"  # Valid timestamp format, no context
    INVALID = "invalid"  # Doesn't match any known format


class TimestampFindings(NamedTuple):
    """Results from timestamp classification"""

    offset: int  # Byte offset in page
    value: int | float  # Raw numeric value
    classification: TimestampClassification
    format_type: str | None  # e.g., "unix_sec", "snowflake_id", "uuid"
    human_readable: str  # Human-readable timestamp or description
    reason: str  # Why this classification was chosen
    source_url: str | None = None  # URL this value came from (if any)
    field_name: str | None = None  # Extracted field name (if any)


def should_keep_timestamp(classification: TimestampClassification, mode: str = "balanced") -> bool:
    """
    Determine if a timestamp should be kept based on classification and mode.

    Modes:
        strict: Only CONFIRMED_TIMESTAMP
        balanced: CONFIRMED_TIMESTAMP + LIKELY_TIMESTAMP
        permissive: Everything except CONFIRMED_ID
        all: Everything (for debugging)
    """
    if mode == "strict":
        return classification == TimestampClassification.CONFIRMED_TIMESTAMP

    if mode == "balanced":
        return classification in (
            TimestampClassification.CONFIRMED_TIMESTAMP,
            TimestampClassification.LIKELY_TIMESTAMP,
        )

    if mode == "permissive":
        return classification not in (
            TimestampClassification.CONFIRMED_ID,
            TimestampClassification.INVALID,
        )

    if mode == "all":
        return True

    # Default: balanced
    return classification in (
        TimestampClassification.CONFIRMED_TIMESTAMP,
        TimestampClassification.LIKELY_TIMESTAMP,
    )


class ClassificationStats:
    """Track classification distribution for reporting"""

    def __init__(self):
        self.counts = dict.fromkeys(TimestampClassification, 0)
        self.total = 0

    def add(self, classification: TimestampClassification):
        self.counts[classification] += 1
        self.total += 1

    def get_summary(self) -> str:
        """Get human-readable summary"""
        if self.total == 0:
            return "No timestamps found"

        lines = [f"\nTimestamp Classification Summary ({self.total} total):"]
        lines.append("─" * 50)

        # Format with percentages
        lines.append(
            f"[+] Confirmed Timestamps: {self.counts[TimestampClassification.CONFIRMED_TIMESTAMP]:6d} "
            f"({self.counts[TimestampClassification.CONFIRMED_TIMESTAMP] / self.total * 100:5.1f}%)"
        )
        lines.append(
            f"[~] Likely Timestamps:    {self.counts[TimestampClassification.LIKELY_TIMESTAMP]:6d} "
            f"({self.counts[TimestampClassification.LIKELY_TIMESTAMP] / self.total * 100:5.1f}%)"
        )
        lines.append(
            f"[?] Ambiguous:            {self.counts[TimestampClassification.AMBIGUOUS]:6d} "
            f"({self.counts[TimestampClassification.AMBIGUOUS] / self.total * 100:5.1f}%)"
        )
        lines.append(
            f"[x] Likely IDs:           {self.counts[TimestampClassification.LIKELY_ID]:6d} "
            f"({self.counts[TimestampClassification.LIKELY_ID] / self.total * 100:5.1f}%)"
        )
        lines.append(
            f"[-] Confirmed IDs:        {self.counts[TimestampClassification.CONFIRMED_ID]:6d} "
            f"({self.counts[TimestampClassification.CONFIRMED_ID] / self.total * 100:5.1f}%)"
        )
        lines.append(
            f"[!] Invalid:              {self.counts[TimestampClassification.INVALID]:6d} "
            f"({self.counts[TimestampClassification.INVALID] / self.total * 100:5.1f}%)"
        )

        lines.append("─" * 50)
        kept = (
            self.counts[TimestampClassification.CONFIRMED_TIMESTAMP]
            + self.counts[TimestampClassification.LIKELY_TIMESTAMP]
        )
        lines.append(f"Kept (balanced mode): {kept}/{self.total} ({kept / self.total * 100:.1f}%)")

        return "\n".join(lines)


# Classification reasons (for transparency)
REASON_FIELD_NAME_TIMESTAMP = "Field name indicates timestamp"
REASON_FIELD_NAME_ID = "Field name indicates ID"
REASON_URL_PARAM_TIMESTAMP = "URL parameter indicates timestamp"
REASON_URL_ID_FIELD = "URL contains ID field"
REASON_SNOWFLAKE_STRUCTURE = "Matches Snowflake ID structure"
REASON_VALID_FORMAT_NO_CONTEXT = "Valid format, no context"
REASON_NO_VALID_FORMAT = "No valid timestamp format detected"
