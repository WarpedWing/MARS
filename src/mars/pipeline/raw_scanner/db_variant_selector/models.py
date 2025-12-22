#!/usr/bin/env python3
"""Data models for database variant selection.

Contains the core data structures used throughout the variant selection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .db_variant_selector_helpers import TableDef


@dataclass
class DBMeta:
    """Metadata for a database variant.

    Captures schema, statistics, and integrity information for a database file.
    Used to compare and select the best variant among O/C/R/D versions.
    """

    path: Path
    opens: bool
    tables: dict[str, TableDef] = field(default_factory=dict)
    table_names: set[str] = field(default_factory=set)  # raw names as found (normalized)
    effective_table_names: set[str] = field(default_factory=set)  # filtered/normalized names
    columns_by_table: dict[str, set[str]] = field(default_factory=dict)  # normalized columns for filtered tables
    rows_by_table: dict[str, int] = field(default_factory=dict)  # rowcounts per effective table
    total_rows: int = 0
    indices: set[str] = field(default_factory=set)
    application_id: int | None = None
    user_version: int | None = None
    page_size: int | None = None
    encoding: str | None = None
    schema_ok: bool = False
    integrity_ok: bool = False
    ddl_parse_rate: float = 0.0
    file_hash: str | None = None
    notes: list[str] = field(default_factory=list)
    nonempty_tables: set[str] = field(default_factory=set)  # had any row at all
    nonnull_tables: set[str] = field(default_factory=set)  # had any non-NULL cell
    has_lost_and_found: bool = False
    lost_and_found_tables: set[str] = field(default_factory=set)
    byte_scanned: bool = False  # True if tables came from byte scanning (unreliable for exact matching)


@dataclass
class Variant:
    """A database variant with its associated metadata.

    Represents one version of a database file:
    - O: Original (unopened, direct from disk)
    - C: Cloned (SQLite3 .clone operation)
    - R: Recovered (SQLite3 .recover operation)
    - D: Dissected (sqlite_dissect + rebuilt)
    - X: Empty (effectively empty database)
    """

    tag: str  # "O", "C", "R", "D", or "X"
    path: Path
    meta: DBMeta
