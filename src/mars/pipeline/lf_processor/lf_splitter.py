#!/usr/bin/env python3
"""
Lost and Found Table Splitter

Splits a recovered SQLite database's lost_and_found table into multiple tables
based on rootpgno and nfield metadata from sqlite3's .recover command.

The lost_and_found table contains rows that couldn't be attributed to specific
tables during recovery. However, SQLite's recovery metadata provides clues:
- rootpgno: Root page number of the B-tree structure (same value = same table)
- pgno: Physical page number where the row was found
- nfield: Exact number of fields in the row (solves NULL padding ambiguity)
- id: ROWID value (NULL for WITHOUT ROWID tables)

Strategy:
1. Use run-length encoding to detect consecutive homogeneous data runs
   - New fragment starts when ANY of (rootpgno, pgno, nfield, first_non_null_col_type) change
   - Data is only guaranteed homogeneous within a continuous run
2. Infer column types for each fragment using SQLite type affinity rules
3. Merge compatible fragments with matching schemas across different page runs
4. Output new database with split tables: lf_table_1, lf_table_2, etc.

Key insight: Within a single rootpgno, different pgnos can contain completely different
table structures. Both rootpgno AND pgno must remain constant for data to be homogeneous.

Usage:
    python lost_and_found_splitter.py <recovered_db> [--output <split_db>]

Example:
    python lost_and_found_splitter.py f123456.recover.sqlite --output f123456.split.sqlite
"""

import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mars.utils.debug_logger import logger

# Add src/ to path when running as standalone script
# This allows the script to find mars package imports
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent  # Go up to src/ directory
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import after path setup to allow standalone script execution
# ruff: noqa: E402
from mars.pipeline.lf_processor.db_reconstructor import cleanup_wal_files
from mars.pipeline.matcher.rubric_utils import (
    TimestampFormat,
    detect_enhanced_pattern_type,
)


@dataclass
class RowFragment:
    """A row from the lost_and_found table with metadata."""

    rootpgno: int
    pgno: int
    nfield: int
    rowid: int | None
    cells: list[Any]  # c0, c1, c2, ... from lost_and_found


@dataclass
class TableFragment:
    """A group of rows that likely belong to the same table."""

    rootpgno: int
    pgno: int
    nfield: int
    first_col_semantic_type: str  # Semantic type of first non-NULL column for discriminating tables
    rows: list[RowFragment] = field(default_factory=list)
    inferred_types: list[str] = field(default_factory=list)  # SQLite type affinities
    computed_semantic_type: str = field(default="")  # For merge grouping (all rows)
    _row_signatures: set[str] = field(default_factory=set)  # Track signatures during extraction

    @property
    def key(self) -> tuple[int, int, str]:
        """Unique identifier for this fragment."""
        return (self.rootpgno, self.nfield, self.first_col_semantic_type)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def add_row_with_signature(self, row: RowFragment, signature: str) -> None:
        """Add a row and track its semantic signature (for incremental computation)."""
        self.rows.append(row)
        self._row_signatures.add(signature)

    def get_computed_semantic_type(self) -> str:
        """
        Get pre-computed semantic type from tracked signatures.

        Returns the signature if all rows have the same one, otherwise "MIXED".
        This avoids re-iterating over all rows in compute_fragment_semantic_type().
        """
        if len(self._row_signatures) == 1:
            return next(iter(self._row_signatures))
        if len(self._row_signatures) == 0:
            return "NULL"
        return "MIXED"


def infer_sqlite_type(value: Any) -> str:
    """
    Infer SQLite type affinity from a Python value.

    SQLite has 5 type affinities: INTEGER, REAL, TEXT, BLOB, NUMERIC
    See: https://www.sqlite.org/datatype3.html
    """
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, str):
        return "TEXT"
    if isinstance(value, bytes):
        return "BLOB"
    return "BLOB"  # fallback


def infer_semantic_type(value: Any) -> str:
    """
    Infer semantic type from value to prevent mixing incompatible data.

    Uses rubric_utils infrastructure to detect patterns like UUID, email, URL,
    timestamp, path, domain, alphanumeric_id, composite_id, hash_string, etc.

    This ensures tables don't mix semantically different data just because it
    has the same SQLite type (e.g., UUIDs vs bundle IDs, human names vs machine IDs).

    Returns:
        Semantic type string (e.g., "uuid", "email", "url", "timestamp_unix_seconds", etc.)
        Falls back to base SQLite type if no pattern matches
    """
    if value is None:
        return "NULL"

    # For numeric types: check for timestamp patterns using TimestampFormat
    if isinstance(value, (int, float)):
        timestamp_format = TimestampFormat.detect_timestamp_format(value)
        if timestamp_format:
            # Return format name as semantic type (e.g., "unix_seconds", "cocoa_seconds")
            return f"timestamp_{timestamp_format}"
        # Not a timestamp - return basic type
        return "INTEGER" if isinstance(value, int) else "REAL"

    if isinstance(value, bytes):
        return "BLOB"

    # For strings: use detect_enhanced_pattern_type from rubric_utils
    # This includes all basic patterns (url, uuid, email, path, domain) PLUS
    # enhanced patterns (alphanumeric_id, composite_id, compound_uuid, hash_string, etc.)
    if isinstance(value, str):
        s = value.strip()

        # Check for null placeholders first
        if not s or s in ("-", "NULL", "null"):
            return "null_placeholder"

        # Use shared enhanced pattern detection from rubric_utils
        pattern = detect_enhanced_pattern_type(value)
        if pattern:
            return pattern

        # Plain text (human-readable text, names, descriptions, etc.)
        return "TEXT"

    # Fallback for unknown types
    return "BLOB"


def compute_fragment_semantic_type(fragment: TableFragment) -> str:
    """
    Compute semantic type signature from ALL rows in a fragment.

    Returns a consistent signature only if 100% of rows have the same
    first non-NULL column position and semantic type. Otherwise returns "MIXED".

    This is used for fragment merging - only fragments with identical
    semantic type signatures can be merged.

    Optimization: If signatures were tracked during extraction (via add_row_with_signature),
    use the pre-computed result instead of re-iterating all rows.

    Returns:
        "{col_idx}:{semantic_type}" if all rows have same signature
        "MIXED" if rows have different signatures (prevents merging)
        "NULL" if all rows are entirely NULL
    """
    # Use pre-computed signatures if available (optimization)
    if fragment._row_signatures:
        return fragment.get_computed_semantic_type()

    # Fallback: compute by iterating all rows (backward compatibility)
    types_seen: set[str] = set()

    for row in fragment.rows:
        # Find first non-NULL column and its semantic type
        row_signature = "NULL"
        for idx, cell in enumerate(row.cells[: fragment.nfield]):
            if cell is not None:
                semantic_type = infer_semantic_type(cell)
                row_signature = f"{idx}:{semantic_type}"
                break
        types_seen.add(row_signature)

    if len(types_seen) == 1:
        return types_seen.pop()  # All rows have same signature
    return "MIXED"  # Multiple signatures = won't merge with anything


def infer_column_types(rows: list[RowFragment], nfield: int) -> list[str]:
    """
    Infer column types for a fragment by examining all rows.

    Returns list of SQLite type affinities, one per column.
    Uses majority-vote with type precedence: TEXT > REAL > INTEGER > BLOB > NULL
    """
    # Collect types for each column position
    column_types: list[set[str]] = [set() for _ in range(nfield)]

    for row in rows:
        for i, cell in enumerate(row.cells[:nfield]):  # Only use nfield columns
            column_types[i].add(infer_sqlite_type(cell))

    # Resolve type for each column
    resolved_types = []
    for types in column_types:
        types.discard("NULL")  # Ignore NULLs for type inference

        if not types:
            resolved_types.append("BLOB")  # All NULL column
        elif "TEXT" in types:
            resolved_types.append("TEXT")  # TEXT takes precedence (most permissive)
        elif "REAL" in types and "INTEGER" in types:
            resolved_types.append("REAL")  # Mix of REAL and INTEGER = REAL
        elif "REAL" in types:
            resolved_types.append("REAL")
        elif "INTEGER" in types:
            resolved_types.append("INTEGER")
        elif "BLOB" in types:
            resolved_types.append("BLOB")
        else:
            resolved_types.append("BLOB")  # fallback

    return resolved_types


def types_compatible(types1: list[str], types2: list[str]) -> bool:
    """
    Check if two type signatures are compatible for merging.

    Compatible if types match or can be safely widened (INT→REAL).
    TEXT is NOT compatible with INTEGER/REAL - mixing numeric and text data
    in the same column would corrupt type inference and cause mismatches.
    """
    if len(types1) != len(types2):
        return False

    for t1, t2 in zip(types1, types2):
        # Exact match
        if t1 == t2:
            continue
        # Integer can widen to REAL (safe numeric conversion)
        if {t1, t2} == {"INTEGER", "REAL"}:
            continue
        # NULL can merge with anything (no type information)
        if "NULL" in {t1, t2}:
            continue
        # Otherwise incompatible (TEXT vs INTEGER/REAL would corrupt inference)
        return False

    return True


def extract_fragments(db_path: Path) -> list[TableFragment]:
    """
    Extract row fragments from lost_and_found table using run-length encoding.

    Processes rows in order and starts a new fragment whenever ANY of these change:
    - rootpgno (B-tree root page)
    - pgno (physical page number)
    - nfield (number of fields)
    - first_col_semantic_type (position:type of first non-NULL column, e.g., "1:TEXT")

    This ensures homogeneous data within each fragment, as data is only guaranteed
    to be the same type within a consecutive run of identical metadata and semantic type.

    Semantic types prevent mixing incompatible data like UUIDs vs bundle IDs, timestamps
    vs regular integers, URLs vs plain text, etc.

    Args:
        db_path: Path to database with lost_and_found table

    Returns:
        List of TableFragment objects, each containing rows from a homogeneous data run
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row

        # Check if lost_and_found exists
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lost_and_found'")
        if not cursor.fetchone():
            logger.error(f"Error: No lost_and_found table found in {db_path}")
            sys.exit(1)

        # Get column count (c0, c1, c2, ...)
        cursor = con.execute("PRAGMA table_info(lost_and_found)")
        columns = [row["name"] for row in cursor]
        data_columns = [c for c in columns if c.startswith("c")]
        max_columns = len(data_columns)

        # Extract all rows using run-length encoding
        # Rows are processed in order - new fragment starts when ANY property changes
        cursor = con.execute("SELECT * FROM lost_and_found ORDER BY pgno, rootpgno")

        current_fragment: TableFragment | None = None
        fragments: list[TableFragment] = []

        for row in cursor:
            rootpgno = row["rootpgno"]
            nfield = row["nfield"]
            pgno = row["pgno"]
            rowid = row["id"]

            # Extract cell values (c0, c1, ..., c<nfield-1>)
            cells = [row[f"c{i}"] for i in range(max_columns)]

            row_frag = RowFragment(rootpgno=rootpgno, pgno=pgno, nfield=nfield, rowid=rowid, cells=cells)

            # Determine semantic type for discriminating different tables
            # This prevents mixing e.g. UUIDs with bundle IDs, timestamps with ints, etc.
            # Find the first non-NULL column and include its POSITION in the signature
            # Position matters: (TEXT, FLOAT, INT) should NOT merge with (NULL, TEXT, TEXT)
            # even though both have first non-NULL type "TEXT"
            first_col_semantic_type = "NULL"
            for idx, cell in enumerate(cells[:nfield]):  # Only check actual columns
                if cell is not None:
                    semantic_type = infer_semantic_type(cell)
                    first_col_semantic_type = f"{idx}:{semantic_type}"
                    break

            # Check if this row belongs to current fragment
            # Start new fragment ONLY when page metadata changes (rootpgno, pgno, nfield)
            # Do NOT check semantic type here - that caused over-splitting within same pgno
            # All rows from same (rootpgno, pgno, nfield) run belong together regardless of content
            if (
                current_fragment is None
                or rootpgno != current_fragment.rootpgno
                or pgno != current_fragment.pgno
                or nfield != current_fragment.nfield
            ):
                # Start new fragment
                current_fragment = TableFragment(
                    rootpgno=rootpgno,
                    pgno=pgno,
                    nfield=nfield,
                    first_col_semantic_type=first_col_semantic_type,
                )
                fragments.append(current_fragment)

            # Add row to current fragment with signature tracking (optimization)
            # This avoids re-iterating all rows in compute_fragment_semantic_type()
            current_fragment.add_row_with_signature(row_frag, first_col_semantic_type)

    # Infer types for each fragment
    for frag in fragments:
        frag.inferred_types = infer_column_types(frag.rows, frag.nfield)

    return fragments


def merge_compatible_fragments(
    fragments: list[TableFragment],
) -> list[list[TableFragment]]:
    """
    Group fragments for output as lf_tables.

    - Computes semantic type for ALL rows in each fragment
    - Only merges fragments where 100% of rows have SAME semantic signature
    - MIXED fragments (rows with different semantic types) never merge

    This balances table reduction (fewer tables = faster matching) with
    data integrity (no mixing of semantically different data).

    Returns list of groups (each group = list of fragments for one lf_table).
    """
    # Step 1: Compute semantic type for each fragment (checks ALL rows)
    for frag in fragments:
        frag.computed_semantic_type = compute_fragment_semantic_type(frag)

    # Step 2: Group by (nfield, computed_semantic_type)
    # MIXED fragments get unique keys so they never merge with anything
    by_key: dict[tuple[int, str], list[TableFragment]] = defaultdict(list)

    for frag in fragments:
        if frag.computed_semantic_type == "MIXED":
            # Each MIXED fragment gets unique key using object id
            # This ensures MIXED fragments never merge with anything
            key = (frag.nfield, f"MIXED_{id(frag)}")
        else:
            # Homogeneous fragments can merge if same nfield + same semantic type
            key = (frag.nfield, frag.computed_semantic_type)
        by_key[key].append(frag)

    # Step 3: Within each group, further cluster by type compatibility
    # (ensures TEXT and INTEGER columns don't get mixed)
    merged_groups: list[list[TableFragment]] = []

    for (nfield, semantic_type), frags in by_key.items():
        if len(frags) == 1:
            # Only one fragment in group, no clustering needed
            merged_groups.append(frags)
            continue

        # Cluster by compatible inferred_types within the semantic group
        clusters: list[list[TableFragment]] = []

        for frag in frags:
            # Try to add to existing cluster with compatible types
            added = False
            for cluster in clusters:
                if types_compatible(frag.inferred_types, cluster[0].inferred_types):
                    cluster.append(frag)
                    added = True
                    break

            if not added:
                # No compatible cluster found, create new one
                clusters.append([frag])

        merged_groups.extend(clusters)

    return merged_groups


def create_split_database(
    output_path: Path,
    merged_groups: list[list[TableFragment]],
    min_rows: int = 1,
    source_db_path: Path | None = None,
) -> None:
    """
    Create new database with split tables from merged fragment groups.

    Optionally copies all non-lost_and_found tables from source database to support
    Use Case 2 (self-rubric generation from intact tables).

    Args:
        output_path: Path to output database
        merged_groups: List of merged fragment groups
        min_rows: Minimum rows required to create a table (filter noise)
        source_db_path: Optional path to source database (to copy regular tables for CATALOG)
    """
    # Remove existing output
    if output_path.exists():
        output_path.unlink()

    with sqlite3.connect(output_path) as con:
        # Copy all non-lost_and_found tables from source database (for CATALOG self-rubric)
        if source_db_path:
            with sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True) as source_con:
                # Get all tables except lost_and_found and system tables
                cursor = source_con.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'lost_and_found'"
                )
                tables_to_copy = cursor.fetchall()

                copied_count = 0
                for table_name, create_sql in tables_to_copy:
                    if not create_sql:  # Skip tables without CREATE SQL (e.g., sqlite_sequence)
                        continue

                    # Skip system/internal tables
                    if table_name.startswith("sqlite_"):
                        continue  # Skip sqlite_sequence, sqlite_stat1, etc.

                    # Skip FTS virtual table auxiliary tables
                    if any(
                        table_name.endswith(suffix)
                        for suffix in ["_content", "_segdir", "_segments", "_docsize", "_stat"]
                    ):
                        continue

                    # Try to create table structure (may fail for virtual tables)
                    try:
                        con.execute(create_sql)
                    except Exception as e:
                        # Skip tables that can't be created (virtual tables, etc.)
                        logger.debug(f"  Skipped {table_name}: {e}")
                        continue

                    # Copy data
                    cursor = source_con.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()

                    if rows:
                        # Get column count
                        col_count = len(rows[0])
                        placeholders = ", ".join(["?"] * col_count)
                        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

                        for row in rows:
                            con.execute(insert_sql, row)

                        con.commit()
                        copied_count += 1

        con.commit()

        table_num = 1
        stats = {
            "tables_created": 0,
            "total_rows": 0,
            "fragments_merged": 0,
            "rootpgnos_per_table": [],
            "pgnos_per_table": [],
        }

        for group in merged_groups:
            total_rows = sum(frag.row_count for frag in group)

            # Skip tiny fragments (likely noise)
            if total_rows < min_rows:
                continue

            # Determine final schema (widest compatible types)
            nfield = group[0].nfield
            final_types = group[0].inferred_types.copy()

            # Skip fragments with no data columns (metadata only)
            if nfield == 0 or not final_types:
                continue

            for frag in group[1:]:
                for i, (t1, t2) in enumerate(zip(final_types, frag.inferred_types)):
                    # Widen if needed
                    if t1 == t2:
                        continue
                    if {t1, t2} == {"INTEGER", "REAL"}:
                        final_types[i] = "REAL"
                    elif "TEXT" in {t1, t2}:
                        final_types[i] = "TEXT"

            # Create table
            table_name = f"lf_table_{table_num}"
            columns_def = ", ".join(f"c{i} {typ}" for i, typ in enumerate(final_types))
            metadata_def = "rootpgno INTEGER, pgno INTEGER, nfield INTEGER, original_id INTEGER"

            create_sql = f"""
            CREATE TABLE {table_name} (
                {columns_def},
                {metadata_def}
            )
            """
            con.execute(create_sql)

            # Insert rows from all fragments in this group
            placeholders = ", ".join(["?"] * (nfield + 4))  # data + 4 metadata columns
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

            rows_inserted = 0
            rootpgnos_seen = set()
            pgnos_seen = set()

            for frag in group:
                # Track unique page numbers
                rootpgnos_seen.add(frag.rootpgno)
                pgnos_seen.add(frag.pgno)
                for row in frag.rows:
                    # Extract only nfield columns (ignore padding)
                    data_values = row.cells[:nfield]
                    metadata_values = [row.rootpgno, row.pgno, row.nfield, row.rowid]
                    con.execute(insert_sql, data_values + metadata_values)
                    rows_inserted += 1

            con.commit()

            # Statistics
            stats["tables_created"] += 1
            stats["total_rows"] += rows_inserted
            stats["fragments_merged"] += len(group)
            stats["rootpgnos_per_table"].append(len(rootpgnos_seen))
            stats["pgnos_per_table"].append(len(pgnos_seen))

            table_num += 1

        # Switch to DELETE journal mode to avoid WAL file locks on Windows
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        con.execute("PRAGMA journal_mode=DELETE;")

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(output_path)
