"""Comparison types and dataclasses.

Extracted from comparison_calculator.py for better organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

# Timestamp sentinel filtering for timeline charts and metrics
# These represent "null" timestamps, not actual data
MIN_REASONABLE_YEAR = 1970  # Unix epoch
MAX_REASONABLE_YEAR = 2100  # Well beyond current time
SENTINEL_DATES = {"2001-01-01", "1970-01-01"}  # Cocoa/Unix epoch zeros


def _is_sentinel_date(date_str: str) -> bool:
    """Check if a date string represents a timestamp sentinel (null value)."""
    if date_str in SENTINEL_DATES:
        return True
    # Also filter extreme years (Apple distantPast/distantFuture)
    try:
        year = int(date_str[:4])
        return year < MIN_REASONABLE_YEAR or year > MAX_REASONABLE_YEAR
    except (ValueError, IndexError):
        return False


@dataclass
class TableComparison:
    """Comparison results for a single table."""

    name: str
    exemplar_rows: int
    candidate_rows: int  # Carved/intact rows only (via data_source column)
    lf_rows: int  # L&F rows (data_source LIKE 'found_%')
    unique_to_candidate: int  # Total unique recovered (carved + L&F not in exemplar)
    overlap: int  # Rows in both (carved + L&F) and exemplar
    unique_to_exemplar: int  # Lost/unrecoverable (in exemplar but not recovered)

    # Timeline info (if timestamp column detected)
    timestamp_column: str | None = None
    timestamp_format: str | None = None
    exemplar_date_range: tuple[datetime, datetime] | None = None
    candidate_date_range: tuple[datetime, datetime] | None = None
    timeline_extended_days: int = 0  # Days of data before exemplar start
    timeline_extended_after_days: int = 0  # Days of data after exemplar end

    # Per-day date lists for gap visualization (unique dates with data)
    exemplar_dates: list[str] = field(default_factory=list)
    candidate_dates: list[str] = field(default_factory=list)

    # Count of days in candidate not in exemplar (range extension + gap-filling)
    unique_candidate_days: int = 0


@dataclass
class LostAndFoundStats:
    """Statistics for lost_and_found data recovery."""

    table_count: int = 0  # Number of lf_ tables
    total_rows: int = 0  # Total rows across all lf_ tables
    tables: list[str] = field(default_factory=list)  # Table names


@dataclass
class DatabaseComparison:
    """Comparison results for a single database."""

    name: str
    category: str
    exemplar_path: Path | None
    candidate_path: Path | None
    matched: bool
    error: str | None = None
    rebuilt_from_lf: bool = False  # True if only L&F tables (no original structure)

    tables: list[TableComparison] = field(default_factory=list)
    lost_and_found: LostAndFoundStats = field(default_factory=LostAndFoundStats)

    @property
    def total_unique_recovered(self) -> int:
        """Total unique rows recovered across all tables."""
        return sum(t.unique_to_candidate for t in self.tables)

    @property
    def total_overlap(self) -> int:
        """Total overlapping rows across all tables."""
        return sum(t.overlap for t in self.tables)

    @property
    def total_exemplar_rows(self) -> int:
        """Total rows in exemplar across all tables."""
        return sum(t.exemplar_rows for t in self.tables)

    @property
    def total_candidate_rows(self) -> int:
        """Total carved/intact rows in candidate across all tables."""
        return sum(t.candidate_rows for t in self.tables)

    @property
    def total_lf_rows(self) -> int:
        """Total L&F rows across all tables (from data_source column)."""
        return sum(t.lf_rows for t in self.tables)

    @property
    def has_timeline_extension(self) -> bool:
        """True if any table has unique candidate days (range extension or gap-filling)."""
        return any(t.unique_candidate_days > 0 for t in self.tables)

    @property
    def max_timeline_extension_days(self) -> int:
        """Maximum unique candidate days across all tables (includes gap-filling)."""
        if not self.tables:
            return 0
        return max(t.unique_candidate_days for t in self.tables)

    @property
    def total_added_days(self) -> int:
        """Sum of unique candidate days across all tables (for per-DB display)."""
        return sum(t.unique_candidate_days for t in self.tables)


@dataclass
class ComparisonResult:
    """Aggregate comparison results."""

    exemplar_database_count: int
    candidate_matched_count: int
    candidate_with_new_data_count: int
    total_unique_rows_recovered: int
    total_overlap_rows: int

    # Lost and found totals
    total_lf_rows_recovered: int = 0
    databases_with_lf: int = 0
    rebuilt_databases_count: int = 0  # Databases recovered via L&F only (no original structure)

    databases: list[DatabaseComparison] = field(default_factory=list)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    exemplar_scan_dir: str = ""
    candidate_run_dir: str = ""
    generated_at: str = ""

    # Error tracking
    databases_with_errors: int = 0
    path_resolution_failures: int = 0
