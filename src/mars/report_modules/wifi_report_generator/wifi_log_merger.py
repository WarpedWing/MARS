#!/usr/bin/env python3
"""
WiFi Log Chronological Merger

Merges multiple WiFi log files chronologically with year inference
and deduplication. Handles the fact that macOS WiFi logs lack year
in their timestamps.
"""

from __future__ import annotations

import contextlib
import heapq
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class LogLine:
    """A single log line with parsed timestamp."""

    timestamp: datetime
    original_line: str
    source_file: str
    line_number: int

    def __lt__(self, other):
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


class WiFiTimestampParser:
    """Parse timestamps from WiFi log format."""

    # WiFi log: Thu Jul 23 00:48:51.636 <airportd[128]>
    WIFI_PATTERN = re.compile(
        r"^(?P<dow>[A-Z][a-z]{2})\s+"
        r"(?P<mon>[A-Z][a-z]{2})\s+"
        r"(?P<day>\d{1,2})\s+"
        r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s*"
    )

    MONTH_MAP = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    @classmethod
    def parse_timestamp(cls, line: str, current_year: int | None = None) -> datetime | None:
        """Parse WiFi log timestamp."""
        match = cls.WIFI_PATTERN.match(line)
        if not match:
            return None

        try:
            month = cls.MONTH_MAP[match.group("mon")]
            day = int(match.group("day"))
            time_str = match.group("time")
            hour, minute, sec_ms = time_str.split(":")
            sec, ms = sec_ms.split(".")

            # Use current year if not specified
            year = current_year or datetime.now().year

            return datetime(
                year,
                month,
                day,
                int(hour),
                int(minute),
                int(sec),
                int(ms) * 1000,
                tzinfo=UTC,
            )
        except Exception:
            return None


def infer_year_from_log_content(
    file_paths: list[Path],
    reference_years: set[int] | None = None,
) -> int | None:
    """
    Infer year from log content using day-of-week validation.

    WiFi logs include day of week (e.g., "Thu Jul 23"). We can find which
    year(s) have that date falling on that day of week.

    Strategy:
    1. Parse first few lines from each log to extract (dow, month, day) tuples
    2. For each tuple, check ALL years in range that match the day-of-week
       (day-of-week repeats every 5-6 years, so we must check all matches)
    3. Return year that's consistent across the most dates, preferring recent years

    Year range:
    - If reference_years provided: use min(ref)-1 to max(ref)+1
    - Otherwise: rolling 10-year window ending at current year

    Returns:
        Inferred year, or None if no clear match
    """
    DOW_MAP = {
        "Mon": 0,
        "Tue": 1,
        "Wed": 2,
        "Thu": 3,
        "Fri": 4,
        "Sat": 5,
        "Sun": 6,
    }

    candidate_years = []

    # Track unique dates to validate consistency
    date_to_years: dict[tuple[int, int], set[int]] = {}  # (month, day) -> {years}

    # Sample more deeply from fewer files to get consistent date ranges
    # This is more reliable than shallow sampling across many files
    sample_files = file_paths[: min(len(file_paths), 20)]  # Sample up to 20 files

    for file_path in sample_files:
        try:
            with file_path.open(encoding="utf-8", errors="replace") as f:
                for _ in range(100):  # Sample 100 lines per file
                    line = f.readline()
                    if not line:
                        break

                    # Try to parse timestamp
                    match = WiFiTimestampParser.WIFI_PATTERN.match(line)
                    if not match:
                        continue

                    dow_str = match.group("dow")
                    mon_str = match.group("mon")
                    day = int(match.group("day"))

                    if dow_str not in DOW_MAP or mon_str not in WiFiTimestampParser.MONTH_MAP:
                        continue

                    expected_dow = DOW_MAP[dow_str]
                    month = WiFiTimestampParser.MONTH_MAP[mon_str]
                    date_key = (month, day)

                    # Determine year range to check
                    # If reference years provided, use them (±1 for edge cases)
                    # Otherwise use rolling 10-year window ending at current year
                    current_year = datetime.now().year
                    if reference_years:
                        min_year = min(reference_years) - 1
                        max_year = max(reference_years) + 1
                    else:
                        min_year = current_year - 10
                        max_year = current_year

                    # Check all years in range (don't break early!)
                    # Day-of-week repeats every 5-6 years, so we need to collect ALL
                    # matching years to find the most consistent one later
                    for year in range(min_year, max_year + 1):
                        try:
                            dt = datetime(year, month, day, tzinfo=UTC)
                            actual_dow = dt.weekday()  # 0=Monday, 6=Sunday

                            if actual_dow == expected_dow:
                                candidate_years.append(year)

                                # Track which years match this date
                                if date_key not in date_to_years:
                                    date_to_years[date_key] = set()
                                date_to_years[date_key].add(year)
                                # Don't break - collect ALL matching years
                        except ValueError:
                            # Invalid date (e.g., Feb 30)
                            continue

        except Exception:
            continue

    if candidate_years:
        # Find the year that's consistent across the most dates
        # This handles cases where logs span multiple dates
        if len(date_to_years) > 1:
            # Find year that appears in the most date sets (best overlap)
            year_date_counts: dict[int, int] = {}
            for year_set in date_to_years.values():
                for year in year_set:
                    year_date_counts[year] = year_date_counts.get(year, 0) + 1

            # Prefer years that match more dates (more consistent)
            max_dates = max(year_date_counts.values())
            consistent_years = [y for y, count in year_date_counts.items() if count == max_dates]

            # Note: reference_years constraint already applied in year range selection above

            # Among consistent years, prefer most recent non-future year
            # (carved logs are typically from recent activity, not decades ago)
            if len(consistent_years) == 1:
                chosen_year = consistent_years[0]
            else:
                current_year = datetime.now().year
                valid_years = [y for y in consistent_years if y <= current_year]
                # Prefer most recent year (e.g., 2020 over 2015)
                chosen_year = max(valid_years) if valid_years else min(consistent_years)

            logger.debug(
                f"Inferred year {chosen_year} from day-of-week analysis "
                f"(consistent across {max_dates}/{len(date_to_years)} dates, "
                f"{len(candidate_years)} samples)"
            )
        else:
            # Only one date found - use simple most common year
            year_counts = Counter(candidate_years)
            chosen_year = year_counts.most_common(1)[0][0]
            logger.debug(
                f"Inferred year {chosen_year} from day-of-week analysis ({len(candidate_years)} samples, single date)"
            )

        return chosen_year

    return None


def infer_year_from_files(
    file_paths: list[Path],
    reference_years: set[int] | None = None,
) -> int:
    """
    Infer year from day-of-week validation across distributed file sample.

    Strategy:
    1. Sample files distributed throughout the collection (not just first N)
    2. Find year that's consistent across the most unique dates
    3. For ties, prefer most recent non-future year
    4. Don't rely on file metadata (not preserved in dfVFS extraction)

    Args:
        file_paths: List of file paths to analyze

    Returns:
        Inferred year
    """
    # Sample files distributed throughout the collection
    # This handles cases where different date ranges are in different files
    if len(file_paths) <= 30:
        sample_paths = file_paths
    else:
        # Take every Nth file to get a representative sample
        step = max(1, len(file_paths) // 30)
        sample_indices = list(range(0, len(file_paths), step))[:30]
        sample_paths = [file_paths[i] for i in sample_indices]

    dow_year = infer_year_from_log_content(sample_paths, reference_years)

    if dow_year:
        return dow_year

    # Fallback: use current year minus 5 (reasonable default for recent logs)
    # This is better than using file metadata which isn't reliable for carved files
    current_year = datetime.now().year
    fallback_year = current_year - 5
    logger.debug(f"Day-of-week validation unavailable, using fallback year {fallback_year}")
    return fallback_year


def parse_log_lines(
    line_iterator: Iterator[str],
    source_file: str,
    current_year: int,
) -> Iterator[LogLine]:
    """
    Parse log lines from an iterator, yielding LogLine objects.

    Args:
        line_iterator: Iterator of raw log lines
        source_file: Source filename for tracking
        current_year: Year to use for parsing

    Yields:
        LogLine objects with parsed timestamps
    """
    line_number = 0
    last_timestamp = None

    for line in line_iterator:
        line_number += 1
        line = line.rstrip("\n\r")

        if not line.strip():
            continue

        # Try to parse timestamp
        timestamp = WiFiTimestampParser.parse_timestamp(line, current_year)

        if timestamp:
            last_timestamp = timestamp
        elif last_timestamp:
            # Multi-line log entry - use timestamp from previous line
            timestamp = last_timestamp
        else:
            # No timestamp and no previous - use epoch
            timestamp = datetime(1970, 1, 1, tzinfo=UTC)

        yield LogLine(
            timestamp=timestamp,
            original_line=line,
            source_file=source_file,
            line_number=line_number,
        )


def merge_wifi_logs(
    file_paths: list[Path],
    output_path: Path,
    reference_years: set[int] | None = None,
) -> int:
    """
    Merge multiple WiFi log files chronologically.

    Uses heap-based streaming merge for memory efficiency. Automatically
    infers year from day-of-week analysis with optional reference year hints.

    Args:
        file_paths: List of WiFi log files to merge
        output_path: Output file path for merged log
        reference_years: Optional years from external sources (known networks, DHCP)
                        to constrain year inference

    Returns:
        Total lines written

    Note:
        - Deduplication happens naturally - duplicate lines with same timestamp
          will be adjacent in output
        - Year is inferred from day-of-week analysis, constrained by reference_years
        - Multi-line log entries are kept together with previous timestamp
    """
    if not file_paths:
        logger.debug("No WiFi logs to merge")
        return 0

    # Infer year from day-of-week analysis, constrained by reference years if provided
    inferred_year = infer_year_from_files(file_paths, reference_years)
    logger.debug(f"Using year {inferred_year} for WiFi log timestamps")

    # Open all files and create iterators
    file_iterators = []
    open_files = []

    for file_path in file_paths:
        try:
            f = file_path.open(encoding="utf-8", errors="replace")
            open_files.append(f)
            file_iterators.append(parse_log_lines(f, str(file_path), inferred_year))
        except Exception as e:
            logger.warning(f"Failed to open {file_path.name}: {e}")

    if not file_iterators:
        return 0

    # Initialize heap with first line from each file
    heap = []
    for idx, iterator in enumerate(file_iterators):
        try:
            log_line = next(iterator)
            heapq.heappush(heap, (log_line, idx, iterator))
        except StopIteration:
            pass

    # Write merged output
    total_lines = 0
    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            while heap:
                log_line, idx, iterator = heapq.heappop(heap)

                # Write line
                out_f.write(log_line.original_line)
                if not log_line.original_line.endswith("\n"):
                    out_f.write("\n")

                total_lines += 1

                # Get next line from same file
                try:
                    next_line = next(iterator)
                    heapq.heappush(heap, (next_line, idx, iterator))
                except StopIteration:
                    pass

    finally:
        # Close all open file handles
        for f in open_files:
            with contextlib.suppress(Exception):
                f.close()

    logger.debug(f"Merged {len(file_paths)} WiFi log files → {total_lines:,} lines")
    return total_lines
