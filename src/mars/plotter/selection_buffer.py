#!/usr/bin/env python3
"""
SelectionBuffer - Accumulates series selections across databases for multi-DB plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SeriesSelection:
    """
    A single series selection with full provenance.

    Contains all metadata needed to:
    1. Display the selection in the UI
    2. Fetch data from the correct database
    3. Convert timestamps to epoch using the correct format
    """

    # Source identification
    source_type: str  # "Exemplar" or "Candidate"
    scan_name: str  # Run timestamp, e.g., "MARS_Test_20251122_074612"
    db_path: Path  # Full path to database file
    db_name: str  # Display name (typically db_path.stem)

    # Table/column selection
    table_name: str  # Table containing the data
    column_name: str  # Y-axis column to plot

    # Timestamp metadata (captured at selection time)
    ts_col: str  # Timestamp column name
    ts_format: str | None  # Format hint from rubric (e.g., "mac_absolute_time")

    # Optional label column for hover tooltips
    label_col: str | None = None

    def display_name(self) -> str:
        """Short display name for UI listings."""
        return f"{self.source_type}:{self.db_name} · {self.table_name} · {self.column_name}"

    def short_name(self) -> str:
        """Abbreviated name for chart legends."""
        return f"{self.db_name} · {self.column_name}"


@dataclass
class SelectionBuffer:
    """
    Accumulates series selections across databases.

    Manages the collection of SeriesSelection objects and provides
    utility methods for UI display and data fetching coordination.
    """

    selections: list[SeriesSelection] = field(default_factory=list)
    max_series: int = 5  # Maximum series for a single chart

    def add(self, sel: SeriesSelection) -> bool:
        """Add a selection. Returns False if at max capacity."""
        if len(self.selections) >= self.max_series:
            return False
        self.selections.append(sel)
        return True

    def remove(self, index: int) -> SeriesSelection | None:
        """Remove selection by index. Returns removed item or None."""
        if 0 <= index < len(self.selections):
            return self.selections.pop(index)
        return None

    def clear(self) -> int:
        """Clear all selections. Returns count cleared."""
        count = len(self.selections)
        self.selections.clear()
        return count

    def is_empty(self) -> bool:
        """Check if buffer has no selections."""
        return len(self.selections) == 0

    def is_full(self) -> bool:
        """Check if buffer is at max capacity."""
        return len(self.selections) >= self.max_series

    def count(self) -> int:
        """Return number of selections in buffer."""
        return len(self.selections)

    def get_unique_databases(self) -> list[Path]:
        """Get unique database paths in selection order."""
        seen: set[Path] = set()
        result: list[Path] = []
        for sel in self.selections:
            if sel.db_path not in seen:
                seen.add(sel.db_path)
                result.append(sel.db_path)
        return result

    def get_source_types(self) -> set[str]:
        """Get unique source types in buffer."""
        return {sel.source_type for sel in self.selections}
