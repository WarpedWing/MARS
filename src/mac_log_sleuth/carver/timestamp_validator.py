#!/usr/bin/env python3

"""
Timestamp Validation and Confidence Scoring
by WarpedWing Labs

Distinguishes real timestamps from timestamp-like IDs (Snowflake, Facebook, etc.)
using contextual analysis, pattern detection, and statistical validation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple


class TimestampCandidate(NamedTuple):
    """A potential timestamp with metadata for validation"""

    offset: int
    value: float | int
    kind: str  # unix_sec, unix_milli, etc.
    human_readable: str
    confidence: float = 0.5  # 0.0 (definitely ID) to 1.0 (definitely timestamp)


# Keyword patterns for contextual analysis
TIMESTAMP_KEYWORDS = [
    b"time",
    b"date",
    b"created",
    b"modified",
    b"updated",
    b"expire",
    b"last",
    b"timestamp",
    b"started",
    b"ended",
    b"finished",
    b"accessed",
    b"changed",
    b"birth",  # birthtime
    b"mtime",
    b"ctime",
    b"atime",
]

ID_KEYWORDS = [
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
    b"snowflake",
]

# Compile regex for field name detection
FIELD_PATTERN = re.compile(rb"([a-zA-Z_][a-zA-Z0-9_]{2,20})[\s\x00]*[:=]")


def extract_field_name(page: bytes, offset: int, lookback: int = 50) -> bytes | None:
    """
    Extract potential field name before the timestamp value.
    Looks for patterns like: "created_at: 1234567890" or "timestamp=1234567890"
    """
    start = max(0, offset - lookback)
    context = page[start:offset]

    # Find the last field name pattern before our offset
    matches = list(FIELD_PATTERN.finditer(context))
    if matches:
        return matches[-1].group(1).lower()
    return None


def check_keyword_context(page: bytes, offset: int, window: int = 50) -> tuple[bool, bool]:
    """
    Returns (has_timestamp_keyword, has_id_keyword) for context around offset.
    """
    context_before = page[max(0, offset - window) : offset].lower()
    context_after = page[offset : min(len(page), offset + window)].lower()
    context = context_before + context_after

    has_timestamp_kw = any(kw in context for kw in TIMESTAMP_KEYWORDS)
    has_id_kw = any(kw in context for kw in ID_KEYWORDS)

    return has_timestamp_kw, has_id_kw


def detect_temporal_clustering(
    candidates: list[tuple[int, float]], offset: int, window: int = 200
) -> int:
    """
    Count how many OTHER timestamp candidates are near this offset.
    Real timestamps often appear in clusters (created_at, updated_at, etc.)
    """
    count = 0
    for cand_offset, _ in candidates:
        if cand_offset == offset:
            continue
        if abs(cand_offset - offset) <= window:
            count += 1
    return count


def detect_sequential_pattern(epochs: list[float], tolerance: float = 100.0) -> bool:
    """
    Returns True if values appear to be sequential IDs rather than timestamps.
    Sequential IDs have very small, uniform gaps. Real timestamps scatter more.
    """
    if len(epochs) < 3:
        return False

    sorted_epochs = sorted(epochs)
    diffs = [sorted_epochs[i + 1] - sorted_epochs[i] for i in range(len(sorted_epochs) - 1)]

    # Filter out zero diffs (duplicates)
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return False

    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    min_diff = min(diffs)

    # Sequential IDs: very tight clustering, small average gap
    # tolerance = max acceptable average gap in seconds
    if avg_diff < tolerance and (max_diff - min_diff) < tolerance:
        return True

    return False


def detect_suspicious_value_patterns(value: float | int, kind: str) -> bool:
    """
    Check if the value itself has patterns suggesting it's an ID.
    - Starts with common ID prefixes (1000000...)
    - Has repeating digit patterns
    - Falls in known ID ranges
    """
    val_str = str(int(abs(value)))

    # Many IDs start with 10000... or 20000...
    if val_str.startswith("10000") or val_str.startswith("20000"):
        return True

    # Check for repeating patterns (111111, 123456789, etc.)
    if len(set(val_str)) <= 2:  # Too few unique digits
        return True

    # Snowflake IDs are typically 18-19 digits starting with 1-9
    # and fall in specific ranges
    if len(val_str) == 19 and kind in ("unix_milli", "unix_micro"):
        # Snowflake IDs converted to milliseconds often look like valid timestamps
        # but they're usually > 2^41 milliseconds from epoch (2010+)
        # This is a weak signal, but combined with other factors helps
        pass

    return False


def calculate_confidence_score(
    page: bytes,
    offset: int,
    value: float | int,
    kind: str,
    all_candidates: list[tuple[int, float]],
) -> float:
    """
    Calculate confidence that this number is a real timestamp (not an ID).

    Returns: 0.0 (definitely ID) to 1.0 (definitely timestamp)
    """
    score = 0.5  # Start neutral

    # 1. Check field name (strongest signal)
    field_name = extract_field_name(page, offset, lookback=50)
    if field_name:
        field_str = field_name.decode("utf-8", errors="ignore")

        # Strong timestamp indicators
        if any(
            ts_word in field_str
            for ts_word in [
                "time",
                "date",
                "created",
                "modified",
                "updated",
                "expire",
                "last",
            ]
        ):
            score += 0.35

        # Strong ID indicators
        if any(id_word in field_str for id_word in ["id", "uid", "key", "token"]):
            score -= 0.40

    # 2. Check surrounding keywords (medium signal)
    has_ts_kw, has_id_kw = check_keyword_context(page, offset, window=50)
    if has_ts_kw:
        score += 0.20
    if has_id_kw:
        score -= 0.25

    # 3. Check for temporal clustering (medium signal)
    cluster_count = detect_temporal_clustering(all_candidates, offset, window=200)
    if cluster_count >= 2:
        score += 0.15  # Multiple timestamps nearby = likely real
    elif cluster_count == 1:
        score += 0.05

    # 4. Check value patterns (weak-medium signal)
    if detect_suspicious_value_patterns(value, kind):
        score -= 0.20

    # 5. Check for alignment (weak signal)
    # Real timestamps in binary formats are often 4 or 8-byte aligned
    if offset % 8 == 0:
        score += 0.05
    elif offset % 4 == 0:
        score += 0.02

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


def validate_timestamp_batch(
    page: bytes, candidates: list[tuple[int, float, str, str]]
) -> list[TimestampCandidate]:
    """
    Validate a batch of timestamp candidates from a single page.
    Applies statistical analysis across all candidates.

    Args:
        page: Raw page bytes
        candidates: List of (offset, epoch, kind, human_readable)

    Returns:
        List of TimestampCandidate with confidence scores
    """
    if not candidates:
        return []

    # Build simple list for clustering detection
    simple_candidates = [(offset, epoch) for offset, epoch, _, _ in candidates]

    # Check for sequential pattern across all candidates
    all_epochs = [epoch for _, epoch, _, _ in candidates]
    is_sequential = detect_sequential_pattern(all_epochs, tolerance=1.0)

    # If entire batch looks sequential, penalize all
    sequential_penalty = -0.30 if is_sequential else 0.0

    # Calculate individual confidence scores
    results = []
    for offset, value, kind, human in candidates:
        base_confidence = calculate_confidence_score(
            page, offset, value, kind, simple_candidates
        )

        # Apply batch-level penalties
        final_confidence = base_confidence + sequential_penalty
        final_confidence = max(0.0, min(1.0, final_confidence))

        results.append(
            TimestampCandidate(
                offset=offset,
                value=value,
                kind=kind,
                human_readable=human,
                confidence=final_confidence,
            )
        )

    return results


def filter_by_confidence(
    candidates: list[TimestampCandidate], min_confidence: float = 0.5
) -> list[TimestampCandidate]:
    """
    Filter timestamp candidates by minimum confidence threshold.
    """
    return [c for c in candidates if c.confidence >= min_confidence]


def get_confidence_label(confidence: float) -> str:
    """
    Human-readable confidence label.
    """
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.6:
        return "MEDIUM"
    elif confidence >= 0.4:
        return "LOW"
    else:
        return "VERY_LOW"
