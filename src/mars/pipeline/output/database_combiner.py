#!/usr/bin/env python3
"""
Database Combiner for SQLite Databases

Combines multiple SQLite databases into a single merged database with deduplication.

Used in two contexts:
1. Exemplar Processing: Combining multiple instances of the same database type
   (e.g., powerlog archives) during exemplar scan
2. Candidate Processing: Merging metamatch groups (databases with identical schemas
   that don't match any known exemplar) during candidate scan

Handles FTS5 virtual tables specially:
- content= FTS tables: Rebuilds index from external content table
- Regular FTS tables: Copies data directly via INSERT
"""

import bz2
import gzip
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from mars.utils.database_utils import quote_ident
from mars.utils.debug_logger import logger

# FTS5 shadow table suffixes - these are internal tables managed by the FTS5 module
FTS5_SHADOW_SUFFIXES = ("_content", "_data", "_docsize", "_config", "_idx")


def is_fts5_shadow_table(table_name: str, virtual_tables: set[str]) -> bool:
    """
    Check if a table is an FTS5 shadow table.

    FTS5 creates shadow tables named {virtual_table}_{suffix} where suffix is one of:
    content, data, docsize, config, idx

    These tables should not be manually created - they're auto-generated when
    the FTS5 virtual table is created.
    """
    for vt_name in virtual_tables:
        for suffix in FTS5_SHADOW_SUFFIXES:
            if table_name == f"{vt_name}{suffix}":
                return True
    return False


def decompress_file(compressed_path: Path) -> Path:
    """
    Decompress a .gz or .bz2 file to a temporary location.

    Args:
        compressed_path: Path to compressed file

    Returns:
        Path to decompressed file (in temp directory)
    """
    suffix = compressed_path.suffix.lower()

    # Create temp file
    temp_dir = Path(tempfile.mkdtemp(prefix="db_decompress_"))
    output_path = temp_dir / compressed_path.stem  # Remove .gz/.bz2 extension

    if suffix == ".gz":
        with (
            gzip.open(compressed_path, "rb") as gz_file,
            Path.open(output_path, "wb") as out_file,
        ):
            shutil.copyfileobj(gz_file, out_file)
    elif suffix == ".bz2":
        with (
            bz2.open(compressed_path, "rb") as bz_file,
            Path.open(output_path, "wb") as out_file,
        ):
            shutil.copyfileobj(bz_file, out_file)
    else:
        # Not compressed, just copy
        shutil.copy2(compressed_path, output_path)

    return output_path


def strip_constraints(schema_sql: str) -> str:
    """
    Remove PRIMARY KEY and UNIQUE constraints from a CREATE TABLE statement.
    This allows us to insert all rows without constraint violations,
    then deduplicate by content hash afterward.

    IMPORTANT: Also removes AUTOINCREMENT since it requires PRIMARY KEY.
    """
    if not schema_sql:
        return schema_sql

    # Remove AUTOINCREMENT first (it requires PRIMARY KEY, so becomes invalid without it)
    result = re.sub(r"\bAUTOINCREMENT\b", "", schema_sql, flags=re.IGNORECASE)

    # Remove inline PRIMARY KEY (e.g., "id INTEGER PRIMARY KEY")
    result = re.sub(r"\bPRIMARY\s+KEY\b", "", result, flags=re.IGNORECASE)

    # Remove inline UNIQUE (e.g., "name TEXT UNIQUE")
    result = re.sub(r"\bUNIQUE\b", "", result, flags=re.IGNORECASE)

    # Remove table-level PRIMARY KEY constraint (e.g., "PRIMARY KEY (id, name)")
    result = re.sub(r",?\s*PRIMARY\s+KEY\s*\([^)]+\)", "", result, flags=re.IGNORECASE)

    # Remove table-level UNIQUE constraint (e.g., "UNIQUE (col1, col2)")
    result = re.sub(r",?\s*UNIQUE\s*\([^)]+\)", "", result, flags=re.IGNORECASE)

    # Remove CONSTRAINT ... PRIMARY KEY/UNIQUE (named constraints)
    result = re.sub(
        r",?\s*CONSTRAINT\s+\w+\s+(PRIMARY\s+KEY|UNIQUE)\s*\([^)]+\)",
        "",
        result,
        flags=re.IGNORECASE,
    )

    # Clean up any double commas or trailing commas before )
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r",\s*\)", ")", result)

    # Clean up multiple spaces
    result = re.sub(r"  +", " ", result)

    return result


def deduplicate_table_content(con: sqlite3.Connection, table_name: str) -> tuple[int, int]:
    """
    Remove duplicate rows from a table based on content hash of non-PK columns.
    Preserves original schema exactly (types, constraints, etc).

    Args:
        con: Database connection
        table_name: Name of table to deduplicate

    Returns:
        Tuple of (original_count, unique_count)
    """
    from mars.utils.database_utils import deduplicate_table_by_content

    return deduplicate_table_by_content(con, table_name)


def ensure_dest_columns(cursor: sqlite3.Cursor, table_name: str, source_cols_info: list):
    """
    Ensure destination table has all columns from source.
    Adds missing columns with ALTER TABLE if needed.

    Args:
        cursor: Destination database cursor
        table_name: Table name
        source_cols_info: List of (cid, name, type, ...) from PRAGMA table_info
    """
    q_table = quote_ident(table_name)

    # Get current destination columns
    dst_cols_info = cursor.execute(f"PRAGMA table_info({q_table});").fetchall()
    dst_col_names = {c[1] for c in dst_cols_info}  # Column names

    # Add missing columns
    for _, col_name, col_type, *_ in source_cols_info:
        if col_name not in dst_col_names:
            q_col = quote_ident(col_name)
            type_clause = col_type or "TEXT"
            alter_sql = f"ALTER TABLE {q_table} ADD COLUMN {q_col} {type_clause};"
            try:
                cursor.execute(alter_sql)
            except sqlite3.OperationalError as e:
                logger.warning(f"    Warning: Could not add column {col_name}: {e}")


def merge_sqlite_databases(
    source_dbs: list[Path],
    output_db: Path,
    dedup_columns: list[str] | None = None,
    combine_strategy: str = "insert_or_ignore",
) -> dict[str, Any]:
    """
    Merge multiple SQLite databases into a single database.

    Uses INSERT OR IGNORE with deterministic ordering (ORDER BY rowid)
    to merge rows while respecting schema constraints.

    Args:
        source_dbs: List of source database paths (uncompressed)
        output_db: Path to output combined database
        dedup_columns: Not used (kept for API compatibility)
        combine_strategy: Not used (kept for API compatibility)

    Returns:
        Dictionary with merge statistics
    """
    # Ensure output directory exists
    output_db.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output if present
    if output_db.exists():
        output_db.unlink()

    # Track all tables for deduplication
    all_tables: set[str] = set()

    # First pass: identify all FTS5 virtual tables across all sources
    # We need this to detect shadow tables that should be skipped
    fts5_virtual_tables: set[str] = set()
    fts5_schemas: dict[str, str] = {}  # table_name -> CREATE VIRTUAL TABLE sql

    for db_path in source_dbs:
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as scan_conn:
                scan_cur = scan_conn.cursor()
                # Look for FTS5 virtual tables
                for name, sql in scan_cur.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts5%';"
                ).fetchall():
                    fts5_virtual_tables.add(name)
                    if name not in fts5_schemas:
                        fts5_schemas[name] = sql
        except Exception:
            pass  # Continue if we can't scan a source

    if fts5_virtual_tables:
        logger.debug(f"  Found {len(fts5_virtual_tables)} FTS5 virtual table(s)")

    logger.debug(f"  Merging {len(source_dbs)} database(s) into {output_db.name}")

    # Create destination database with context manager
    with sqlite3.connect(str(output_db)) as dst_conn:
        dst_cur = dst_conn.cursor()
        dst_cur.execute("PRAGMA journal_mode=WAL;")  # Better for concurrent access
        dst_cur.execute("PRAGMA synchronous=NORMAL;")
        dst_conn.commit()

        stats = {
            "total_sources": len(source_dbs),
            "tables_merged": 0,
            "total_rows_merged": 0,
            "rows_before_dedup": 0,
            "rows_after_dedup": 0,
            "duplicates_removed": 0,
            "per_db_stats": {},
            "errors": [],
        }

        for db_idx, db_path in enumerate(source_dbs, 1):
            db_stats = {"tables": {}, "total_rows": 0, "errors": []}
            logger.debug(f"    [{db_idx}/{len(source_dbs)}] Processing {db_path.name}")

            try:
                # Open source database read-only with immutable mode to prevent WAL/SHM creation
                with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as src_conn:
                    # Use 'replace' error handling for invalid UTF-8 in TEXT columns
                    src_conn.text_factory = lambda x: x.decode("utf-8", "replace") if isinstance(x, bytes) else x
                    src_cur = src_conn.cursor()

                    # Get all tables from source
                    tables = src_cur.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                    ).fetchall()

                    for table_name, schema_sql in tables:
                        if not schema_sql:  # Skip views or special tables
                            continue

                        # Skip FTS5 shadow tables - they'll be created when we create the virtual table
                        if is_fts5_shadow_table(table_name, fts5_virtual_tables):
                            logger.debug(f"      Skipping FTS5 shadow table: {table_name}")
                            continue

                        # Skip FTS5 virtual tables for now - process them after all regular tables
                        if table_name in fts5_virtual_tables:
                            logger.debug(f"      Deferring FTS5 virtual table: {table_name}")
                            continue

                        q_table = quote_ident(table_name)
                        all_tables.add(table_name)

                        # Try to create table using original schema (preserves constraints)
                        table_created = False
                        try:
                            dst_cur.execute(schema_sql)
                            table_created = True
                        except sqlite3.OperationalError as e:
                            if "already exists" in str(e).lower():
                                table_created = True
                            else:
                                # Schema failed, try stripped version
                                try:
                                    stripped_schema = strip_constraints(schema_sql)
                                    dst_cur.execute(stripped_schema)
                                    table_created = True
                                except sqlite3.OperationalError as e2:
                                    if "already exists" in str(e2).lower():
                                        table_created = True
                                    else:
                                        # Expected for virtual tables with custom modules
                                        logger.debug(f"      Could not create table {table_name}: {e2}")

                        if not table_created:
                            continue

                        # Get column info from source and destination
                        try:
                            src_cols_info = src_cur.execute(f"PRAGMA table_info({q_table});").fetchall()
                            dst_cols_info = dst_cur.execute(f"PRAGMA table_info({q_table});").fetchall()
                        except sqlite3.DatabaseError as e:
                            error_msg = f"Cannot inspect table {table_name}: {e}"
                            db_stats["errors"].append(error_msg)
                            continue

                        # Check if destination table actually exists (has columns)
                        if not dst_cols_info:
                            logger.debug(f"      Table {table_name} has no columns in destination, skipping")
                            continue

                        # Ensure destination has all source columns
                        ensure_dest_columns(dst_cur, table_name, src_cols_info)

                        # Refresh destination columns after potential additions
                        dst_cols_info = dst_cur.execute(f"PRAGMA table_info({q_table});").fetchall()

                        src_col_names = [c[1] for c in src_cols_info]
                        dst_col_names = [c[1] for c in dst_cols_info]
                        common_cols = [c for c in src_col_names if c in dst_col_names]

                        if not common_cols:
                            continue

                        # Exclude INTEGER PRIMARY KEY columns from INSERT to avoid ID collisions
                        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                        # pk=1 means it's a primary key column
                        integer_pk_cols = set()
                        for col_info in dst_cols_info:
                            col_name = col_info[1]
                            col_type = col_info[2]
                            is_pk = col_info[5]
                            # Check if it's an INTEGER PRIMARY KEY (auto-increment)
                            if is_pk == 1 and col_type.upper() in ("INTEGER", "INT"):
                                integer_pk_cols.add(col_name)

                        # Filter out INTEGER PRIMARY KEY columns from the merge
                        # SQLite will auto-assign new IDs, preventing collisions
                        merge_cols = [c for c in common_cols if c not in integer_pk_cols]

                        if not merge_cols:
                            # Only INTEGER PRIMARY KEY columns - skip this table
                            continue

                        # Build INSERT OR IGNORE statement (handles constraint violations)
                        col_list = ", ".join(quote_ident(c) for c in merge_cols)
                        placeholders = ", ".join("?" for _ in merge_cols)
                        insert_sql = f"INSERT OR IGNORE INTO {q_table} ({col_list}) VALUES ({placeholders});"

                        # Copy rows with deterministic ordering (falls back for WITHOUT ROWID tables)
                        # Note: col_list now excludes INTEGER PRIMARY KEY columns
                        try:
                            try:
                                rows = src_cur.execute(f"SELECT {col_list} FROM {q_table} ORDER BY rowid;").fetchall()
                            except sqlite3.OperationalError as e:
                                if "no such column: rowid" in str(e).lower():
                                    # WITHOUT ROWID table - fall back to unordered select
                                    rows = src_cur.execute(f"SELECT {col_list} FROM {q_table};").fetchall()
                                else:
                                    raise

                            for row in rows:
                                dst_cur.execute(insert_sql, row)

                            db_stats["tables"][table_name] = len(rows)
                            db_stats["total_rows"] += len(rows)
                            stats["total_rows_merged"] += len(rows)

                        except sqlite3.DatabaseError as e:
                            error_msg = f"Error merging table {table_name} from {db_path.name}: {e}"
                            db_stats["errors"].append(error_msg)
                            logger.warning(f"      {error_msg}")

                    dst_conn.commit()

                stats["per_db_stats"][str(db_path)] = db_stats
                logger.debug(f"      Added {db_stats['total_rows']:,} rows")

            except Exception as e:
                error_msg = f"Error processing {db_path.name}: {e}"
                stats["errors"].append(error_msg)
                logger.warning(f"      {error_msg}")

        dst_conn.commit()

        # Deduplicate all merged tables
        for table_name in all_tables:
            try:
                before_count, after_count = deduplicate_table_content(dst_conn, table_name)
                stats["rows_before_dedup"] += before_count
                stats["rows_after_dedup"] += after_count
                duplicates = before_count - after_count
                if duplicates > 0:
                    stats["duplicates_removed"] += duplicates
                    logger.debug(
                        f"      Deduplicated {table_name}: {before_count:,} -> {after_count:,} ({duplicates:,} duplicates removed)"
                    )
            except Exception as e:
                logger.debug(f"      Failed to deduplicate {table_name}: {e}")

        dst_conn.commit()

        # Create FTS5 virtual tables after all content tables are populated
        # This ensures the content tables exist for content= FTS tables
        if fts5_schemas:
            logger.debug(f"  Creating {len(fts5_schemas)} FTS5 virtual table(s)")
            for fts_name, fts_sql in fts5_schemas.items():
                try:
                    dst_cur.execute(fts_sql)
                    logger.debug(f"      Created FTS5 table: {fts_name}")
                    all_tables.add(fts_name)

                    # For content= FTS tables, rebuild the index from the content table
                    # This populates the FTS index with data from the already-merged content table
                    q_fts = quote_ident(fts_name)
                    if "content=" in fts_sql.lower() or "content =" in fts_sql.lower():
                        try:
                            dst_cur.execute(f"INSERT INTO {q_fts}({q_fts}) VALUES('rebuild');")
                            logger.debug(f"      Rebuilt FTS index for: {fts_name}")
                        except sqlite3.OperationalError as e:
                            logger.debug(f"      Could not rebuild FTS index for {fts_name}: {e}")
                    else:
                        # For regular FTS5 tables, copy data from source FTS tables
                        # Shadow tables were skipped, so we need to INSERT data directly
                        fts_rows_copied = 0
                        for db_path in source_dbs:
                            try:
                                with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as src_conn:
                                    src_conn.text_factory = (
                                        lambda x: x.decode("utf-8", "replace") if isinstance(x, bytes) else x
                                    )
                                    src_cur = src_conn.cursor()
                                    # Get column names from FTS table
                                    cols_info = src_cur.execute(f"PRAGMA table_info({q_fts})").fetchall()
                                    if cols_info:
                                        col_names = [c[1] for c in cols_info]
                                        col_list = ", ".join(quote_ident(c) for c in col_names)
                                        placeholders = ", ".join("?" for _ in col_names)

                                        rows = src_cur.execute(f"SELECT {col_list} FROM {q_fts}").fetchall()
                                        for row in rows:
                                            try:
                                                dst_cur.execute(
                                                    f"INSERT OR IGNORE INTO {q_fts} ({col_list}) VALUES ({placeholders})",
                                                    row,
                                                )
                                                fts_rows_copied += 1
                                            except sqlite3.Error:
                                                pass  # Skip problematic rows
                            except Exception as e:
                                logger.debug(f"      Could not copy FTS data from {db_path.name} for {fts_name}: {e}")
                        if fts_rows_copied > 0:
                            logger.debug(f"      Copied {fts_rows_copied} rows to FTS table: {fts_name}")
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"      Could not create FTS5 table {fts_name}: {e}")
            dst_conn.commit()

        # Switch from WAL to DELETE mode to ensure all data is in the main file
        # This removes the WAL file and makes the database fully self-contained
        # Subsequent reads with immutable=1 will see the complete schema and data
        dst_cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        dst_cur.execute("PRAGMA journal_mode=DELETE;")

    stats["tables_merged"] = len(all_tables)

    return stats


def combine_database_group(
    primary_db: Path,
    archive_dbs: list[Path],
    output_db: Path,
    combine_strategy: str = "decompress_and_merge",
) -> dict[str, Any]:
    """
    Combine a primary database with its archives.

    Args:
        primary_db: Primary database file (e.g., CurrentPowerlog.PLSQL)
        archive_dbs: List of archive files (compressed or not)
        output_db: Path to output combined database
        combine_strategy: "decompress_and_merge" for SQLite, "decompress_and_concatenate" for text,
            "decompress_only" skips combining (files extracted in place by workspace_manager)

    Returns:
        Dictionary with combine statistics
    """
    temp_dirs = []  # Track temp directories for cleanup

    try:
        # Decompress all archives
        decompressed_dbs = []

        # Add primary database (might not be compressed)
        if primary_db.suffix.lower() in {".gz", ".bz2"}:
            decompressed = decompress_file(primary_db)
            temp_dirs.append(decompressed.parent)
            decompressed_dbs.append(decompressed)
        else:
            decompressed_dbs.append(primary_db)

        # Decompress archives
        for archive in archive_dbs:
            if archive.suffix.lower() in {".gz", ".bz2"}:
                decompressed = decompress_file(archive)
                temp_dirs.append(decompressed.parent)
                decompressed_dbs.append(decompressed)
            else:
                decompressed_dbs.append(archive)

        # Filter to only SQLite databases (check magic bytes, not just extension)
        def is_sqlite_db(path: Path) -> bool:
            """Check if file is a SQLite database by reading magic bytes."""
            try:
                with Path.open(path, "rb") as f:
                    header = f.read(16)
                    return header.startswith(b"SQLite format 3\x00")
            except Exception:
                return False

        sqlite_dbs = [db for db in decompressed_dbs if is_sqlite_db(db)]

        if not sqlite_dbs:
            return {
                "status": "error",
                "message": "No SQLite databases found after decompression",
            }

        # Merge SQLite databases
        if combine_strategy == "decompress_and_merge":
            stats = merge_sqlite_databases(sqlite_dbs, output_db)
            stats["status"] = "success"
            stats["output_db"] = str(output_db)
            return stats
        # Text concatenation (for logs) - not currently used
        return {
            "status": "error",
            "message": f"Strategy '{combine_strategy}' not yet implemented for text files",
        }

    finally:
        # Cleanup temp directories
        for temp_dir in temp_dirs:
            try:  # noqa: SIM105
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
