#!/usr/bin/env python3
"""Utility functions for SQLite database operations."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def quote_identifier(name: str) -> str:
    """
    Quote an SQL identifier (table or column name) to handle special characters.

    Uses double quotes per SQL standard. Escapes any embedded quotes.

    Args:
        name: Identifier to quote

    Returns:
        Quoted identifier safe for SQL
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


# Alias for backward compatibility
quote_ident = quote_identifier


def checkpoint_wal_database(db_path: Path, cleanup_wal: bool = True) -> bool:
    """
    Checkpoint a WAL-mode SQLite database to consolidate data into main file.

    When databases are in WAL mode, recent data may only exist in the .wal file.
    Checkpointing writes this data to the main database file, making it readable
    with immutable=1 mode (which ignores WAL files).

    Args:
        db_path: Path to SQLite database
        cleanup_wal: If True, remove WAL/SHM files after successful checkpoint

    Returns:
        True if checkpoint succeeded, False otherwise
    """
    wal_path = db_path.parent / f"{db_path.name}-wal"
    shm_path = db_path.parent / f"{db_path.name}-shm"

    # Only checkpoint if WAL file exists (SHM alone is not useful)
    if not wal_path.exists():
        # Clean up orphaned SHM file if present
        if shm_path.exists() and cleanup_wal:
            with suppress(OSError):
                shm_path.unlink()
        return True  # Nothing to checkpoint

    try:
        # Delete SHM file before opening to force SQLite to rebuild WAL index.
        # When copying from a live system, the SHM file may be stale/inconsistent
        # with the WAL file, causing SQLite to think there are no pages to checkpoint.
        # Removing it forces SQLite to rebuild the index from the WAL header.
        if shm_path.exists():
            try:
                shm_path.unlink()
                logger.debug(f"Removed stale SHM for fresh checkpoint: {shm_path.name}")
            except OSError as e:
                logger.debug(f"Could not remove SHM file: {e}")

        # Open without immutable to allow WAL reading
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
        try:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
            # Result is (busy, log, checkpointed) - log pages written, checkpointed pages moved
            if result and (result[1] > 0 or result[2] > 0):
                logger.debug(f"WAL checkpoint completed: {db_path.name} (wrote={result[1]}, moved={result[2]})")
            else:
                logger.debug(f"WAL checkpoint: {db_path.name} (no pages to checkpoint)")

            # Switch to DELETE mode to ensure WAL files are removed.
            # This prevents SQLite from recreating WAL/SHM when the database
            # is opened again (e.g., during combination in Phase 4).
            conn.execute("PRAGMA journal_mode=DELETE;")
        finally:
            conn.close()

        # Clean up WAL/SHM files after checkpoint
        if cleanup_wal:
            for aux_path in [wal_path, shm_path]:
                if aux_path.exists():
                    try:
                        aux_path.unlink()
                        logger.debug(f"Removed auxiliary file: {aux_path.name}")
                    except OSError:
                        pass  # Best effort cleanup

        return True

    except Exception as e:
        logger.warning(f"WAL checkpoint failed for {db_path.name}: {e}")
        return False


def copy_database_with_auxiliary_files(src_db: Path, dest_db: Path, checkpoint: bool = True) -> list[Path]:
    """
    Copy SQLite database with all auxiliary files (.shm, .wal, .journal).

    SQLite uses auxiliary files for different journaling modes:
    - .sqlite-shm: Shared memory file for WAL index
    - .sqlite-wal: Write-Ahead Log containing uncommitted transactions
    - .sqlite-journal: Rollback journal containing undo information

    These files are forensically critical as they contain:
    - Recent changes not yet checkpointed to main database
    - Uncommitted transactions
    - Deleted data not yet vacuumed
    - Rollback information for incomplete transactions

    After copying, if checkpoint=True (default), a WAL checkpoint is performed
    on the destination database to consolidate WAL data into the main file.
    This ensures the database is readable with immutable=1 mode.

    Args:
        src_db: Source database path
        dest_db: Destination database path
        checkpoint: If True, perform WAL checkpoint after copy (default True)

    Returns:
        List of successfully copied file paths
    """
    copied_files = []

    # Copy main database file
    try:
        # Use copy instead of copy2 to avoid metadata preservation issues
        # across different filesystems (e.g., mounted DMG to local disk)
        shutil.copy(src_db, dest_db)
        copied_files.append(dest_db)
        logger.debug(f"Copied database: {src_db.name}")
    except Exception as e:
        logger.error(f"Failed to copy database {src_db.name}: {e}")
        return copied_files

    # Copy auxiliary files if they exist
    # SQLite WAL mode uses: DatabaseName.sqlite-shm and DatabaseName.sqlite-wal
    # SQLite rollback journal mode uses: DatabaseName.sqlite-journal
    # (appends to the full filename, not just the stem)
    auxiliary_extensions = ["-shm", "-wal", "-journal"]

    for ext in auxiliary_extensions:
        # Append extension to full filename (e.g., "Cache.sqlite" -> "Cache.sqlite-shm")
        src_aux = src_db.parent / f"{src_db.name}{ext}"
        dest_aux = dest_db.parent / f"{dest_db.name}{ext}"

        if src_aux.exists():
            try:
                shutil.copy(src_aux, dest_aux)
                copied_files.append(dest_aux)
                size_kb = src_aux.stat().st_size / 1024
                logger.debug(f"Copied auxiliary file: {src_aux.name} ({size_kb:.1f}KB)")
            except Exception as e:
                logger.warning(f"Failed to copy {src_aux.name}: {e}")

    # Checkpoint WAL to consolidate data into main database file
    # This is critical for live system scans where data may only be in WAL
    if checkpoint and dest_db.exists():
        wal_path = dest_db.parent / f"{dest_db.name}-wal"
        shm_path = dest_db.parent / f"{dest_db.name}-shm"
        if wal_path.exists() or shm_path.exists():
            logger.debug(f"Checkpointing WAL for {dest_db.name} (wal={wal_path.exists()}, shm={shm_path.exists()})")
        success = checkpoint_wal_database(dest_db, cleanup_wal=True)
        if not success:
            logger.warning(f"WAL checkpoint returned False for {dest_db.name}")

    return copied_files


def copy_database_to_directory(src_db: Path, dest_dir: Path) -> Path:
    """
    Copy SQLite database with auxiliary files to a directory.

    Convenience wrapper for copy_database_with_auxiliary_files that
    automatically constructs the destination path.

    Args:
        src_db: Source database path
        dest_dir: Destination directory

    Returns:
        Path to copied database file
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_db = dest_dir / src_db.name
    copy_database_with_auxiliary_files(src_db, dest_db)
    return dest_db


# ============================================================================
# Database Validation Utilities
# ============================================================================


def is_sqlite_database(db_path: Path) -> bool:
    """
    Check if a file is a valid SQLite database.

    Performs two checks:
    1. Magic bytes check (SQLite format 3 header)
    2. Attempt to open and query the database

    Args:
        db_path: Path to potential database file

    Returns:
        True if file is a valid SQLite database, False otherwise
    """
    try:
        # Check magic bytes first
        with db_path.open("rb") as f:
            header = f.read(16)
            if not header.startswith(b"SQLite format 3\x00"):
                return False

        # Try to open it with immutable mode for read-only filesystems
        # The immutable=1 flag is critical for mounted forensic images
        with readonly_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM sqlite_master")
            cursor.fetchone()
        return True
    except Exception:
        return False


def is_encrypted_database(db_path: Path) -> bool:
    """
    Check if a file is likely encrypted (high entropy, no SQLite header).

    Encrypted files (Apple Data Protection, etc.) will have:
    - No SQLite magic bytes
    - High entropy (uniform byte distribution)
    - Database-like file extension or naming pattern

    This function only checks files that appear to be intended as databases
    (not logs, cache, or other binary files) to avoid false positives.

    Args:
        db_path: Path to potential database file

    Returns:
        True if file appears to be encrypted, False otherwise
    """
    try:
        # Use is_sqlite_database check instead of duplicating logic
        if is_sqlite_database(db_path):
            return False

        # IMPORTANT: Only check files that appear to be databases
        # This prevents flagging logs, cache files, and other binary files as encrypted
        filename_lower = db_path.name.lower()

        # Database file extensions to check
        db_extensions = {
            ".sqlite",
            ".sqlite3",
            ".db",
            ".db3",
            ".sqlitedb",
            ".sdb",
            ".sqlite-db",
            ".sql",
            ".database",
        }

        # Check if file has a database-like extension
        has_db_extension = any(filename_lower.endswith(ext) for ext in db_extensions)

        # Exclusions: Skip files that are clearly not databases
        # Log files
        if any(pattern in filename_lower for pattern in [".log", "log.", "_log"]):
            return False
        # Cache files (unless they have explicit DB extension)
        if "cache" in filename_lower and not has_db_extension:
            return False
        # Preference files (unless they have explicit DB extension)
        if any(pattern in filename_lower for pattern in [".plist", ".preferences", ".prefs"]) and not has_db_extension:
            return False

        # Only proceed with entropy check if file looks like a database
        if not has_db_extension:
            return False

        with db_path.open("rb") as f:
            header = f.read(256)

        # Check for compressed file magic bytes BEFORE entropy check
        # Compressed files have high entropy but aren't encrypted
        if header.startswith(b"\x1f\x8b"):  # Gzip
            return False
        if header.startswith(b"BZ"):  # Bzip2
            return False
        if header.startswith(b"\x50\x4b\x03\x04"):  # ZIP
            return False
        if header.startswith(b"\x50\x4b\x05\x06"):  # Empty ZIP
            return False
        if header.startswith(b"\x50\x4b\x07\x08"):  # Spanned ZIP
            return False

        # Check entropy - encrypted data has ~160 unique bytes in first 256
        unique_bytes = len(set(header))

        # High entropy suggests encryption (vs. corruption or non-DB files)
        # Encrypted (AES/Data Protection): ~155-170 unique bytes (60-65%)
        # Compressed files: Also ~155-170 unique bytes (filtered above)
        # Text files: ~20-40 unique bytes (8-15%)
        # SQLite headers: ~10-20 unique bytes (4-8%)
        # Default threshold of 140 catches encrypted files while avoiding false positives
        return unique_bytes >= 140

    except Exception:
        return False


def is_database_empty_or_null(db_path: Path) -> tuple[bool, str]:
    """
    Check if database is empty or contains only NULL/meaningless data.

    Uses rigorous checks:
    - No tables: empty
    - Only global ignorable tables: empty (e.g., only sqlean_define)
    - All tables have 0 rows: empty
    - All rows are NULL-only: empty
    - At least 1 non-NULL value found: NOT empty

    Args:
        db_path: Path to database

    Returns:
        Tuple of (is_empty, reason)
        - (True, "no_tables"): No tables found
        - (True, "no_rows"): Tables exist but no rows
        - (True, "all_null"): Data exists but only NULL values
        - (False, ""): Database has real data
    """
    from mars.config.schema import GLOBAL_IGNORABLE_TABLES

    try:
        with readonly_connection(db_path) as conn:
            cursor = conn.cursor()

            # Get all tables (excluding sqlite_ system tables)
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            all_tables = [row[0] for row in cursor.fetchall()]

            if not all_tables:
                return True, "no_tables"

            # Filter out globally ignorable tables
            tables = [t for t in all_tables if t not in GLOBAL_IGNORABLE_TABLES]

            if not tables:
                # Only has ignorable tables (e.g., only sqlean_define)
                return True, "no_tables"

            # Check each table for data
            total_rows = 0
            has_non_null = False

            for table in tables:
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                total_rows += row_count

                if row_count == 0:
                    continue

                # Get column names
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]

                if not columns:
                    continue

                # Sample first 100 rows to check for non-NULL values
                # Build query to check if ANY column has non-NULL value
                col_checks = " OR ".join([f"{col} IS NOT NULL" for col in columns])
                query = f"SELECT 1 FROM {table} WHERE {col_checks} LIMIT 1"

                try:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result:
                        has_non_null = True
                        break
                except sqlite3.Error:
                    # Query failed (malformed table?), assume has data
                    has_non_null = True
                    break

            if total_rows == 0:
                return True, "no_rows"

            if not has_non_null:
                return True, "all_null"

            return False, ""

    except Exception:
        # On error, assume NOT empty (don't accidentally discard data)
        return False, ""


def has_effective_tables(db_path: Path) -> bool:
    """
    Check if database has effective tables (non-ignorable, non-lost_and_found).

    Effective tables are:
    - Not in the global ignore list
    - Not system tables (sqlite_*)
    - Not lost_and_found* tables
    - Not matching ignore prefixes/suffixes

    Args:
        db_path: Path to database

    Returns:
        True if database has at least one effective table, False otherwise
    """
    from mars.config.schema import (
        GLOBAL_IGNORABLE_TABLES,
        SchemaComparisonConfig,
    )

    # Load ignore lists from config (single source of truth)
    try:
        _config = SchemaComparisonConfig()
        ignore_tables = _config.ignorable_tables
        ignore_prefixes = _config.ignorable_prefixes
        ignore_suffixes = _config.ignorable_suffixes
    except ImportError:
        # Fallback if config not available
        ignore_tables = GLOBAL_IGNORABLE_TABLES
        ignore_prefixes = {"sqlite_", "sqlean_"}
        ignore_suffixes = {"_content", "_segments", "_segdir", "_docsize", "_stat"}

    try:
        with readonly_connection(db_path) as conn:
            cursor = conn.cursor()

            # Get all tables
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """
            )
            all_tables = [row[0] for row in cursor.fetchall()]

            # Filter tables
            effective_tables = []
            for table_name in all_tables:
                table_lower = table_name.lower()

                # Skip tables matching ignore prefixes
                if any(table_lower.startswith(prefix.lower()) for prefix in ignore_prefixes):
                    continue

                # Skip tables matching ignore suffixes
                if any(table_lower.endswith(suffix.lower()) for suffix in ignore_suffixes):
                    continue

                # Skip ignorable tables
                if table_lower in {t.lower() for t in ignore_tables}:
                    continue

                # Skip lost_and_found tables (salvage tables from sqlite_dissect)
                if table_lower.startswith("lost_and_found"):
                    continue

                effective_tables.append(table_name)

            return len(effective_tables) > 0

    except Exception:
        # On error, assume has effective tables (don't accidentally discard)
        return True


# ============================================================================
# Database Variant Selection Utilities
# ============================================================================


def get_chosen_variant_path(record: dict) -> Path | None:
    """
    Extract the chosen variant path from a database record.

    When databases are processed by db_variant_selector, multiple variants
    may be created (original, clone, recover, dissect_rebuilt). This function
    extracts the path to the chosen variant based on the variant_chosen tag.

    Args:
        record: Database record from sqlite_scan_results.jsonl

    Returns:
        Path to chosen variant database or None if not found

    Examples:
        >>> record = {"variant_chosen": "O", "variant_outputs": {"original": "/path/to/db.sqlite"}}
        >>> get_chosen_variant_path(record)
        Path("/path/to/db.sqlite")
    """
    from pathlib import Path

    variant_outputs = record.get("variant_outputs", {})
    variant_tag = record.get("variant_chosen", "")

    # For compressed files, use decompressed path if available
    if "decompressed" in variant_outputs:
        decompressed_path = variant_outputs["decompressed"]
        if decompressed_path and Path(decompressed_path).exists():
            return Path(decompressed_path)

    # Map variant tag to output key
    variant_key_map = {
        "O": "original",
        "X": "original",
        "C": "clone",
        "R": "recover",
        "D": "dissect_rebuilt",
    }
    variant_key = variant_key_map.get(variant_tag)

    if not variant_key or variant_key not in variant_outputs:
        return None

    chosen_variant_path = variant_outputs[variant_key]
    if not chosen_variant_path or not Path(chosen_variant_path).exists():
        return None

    return Path(chosen_variant_path)


# ============================================================================
# Database Connection Helpers
# ============================================================================


@contextmanager
def readonly_connection(db_path: Path, immutable: bool = True) -> Iterator[sqlite3.Connection]:
    """
    Context manager for read-only database connections.

    Opens database in read-only mode using URI parameter.
    Automatically handles connection cleanup.

    On macOS, Time Machine files may be blocked by Gatekeeper when they
    appear to be "Unix Executable Files". If direct open fails, this
    function copies the file to a temp location and opens from there.

    Args:
        db_path: Path to SQLite database
        immutable: Use immutable flag for source databases (default True).
                  Set to False for newly created/combined databases in WAL mode.

    Yields:
        Read-only SQLite connection

    Example:
        >>> from pathlib import Path
        >>> with readonly_connection(Path("data.db")) as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM table")
        ...     results = cursor.fetchall()
    """
    from pathlib import Path as PathClass  # Avoid type checking import issue

    temp_path: PathClass | None = None
    conn: sqlite3.Connection | None = None

    try:
        # Try direct connection first
        if immutable:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        else:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        # Test that we can actually read from the database file
        # SELECT 1 doesn't read from disk - must query sqlite_master
        # macOS security blocking manifests when actually reading file contents
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()

    except sqlite3.DatabaseError as e:
        err_msg = str(e).lower()
        # macOS blocks files it considers "Unix Executables" with this error
        # Also handle "unable to open database file" which can occur
        if "not a database" in err_msg or "unable to open" in err_msg:
            if conn:
                with suppress(Exception):
                    conn.close()
                conn = None

            # Copy file to temp location with .sqlite extension
            # This bypasses macOS Gatekeeper's file type check
            temp_dir = PathClass(tempfile.mkdtemp(prefix="mars_db_"))
            temp_path = temp_dir / f"{db_path.stem}.sqlite"

            logger.debug(f"macOS blocked {db_path.name}, copying to temp: {temp_path}")
            shutil.copy2(db_path, temp_path)

            # Also copy WAL and SHM files if they exist
            for suffix in ["-wal", "-shm"]:
                wal_path = db_path.parent / f"{db_path.name}{suffix}"
                if wal_path.exists():
                    shutil.copy2(wal_path, temp_path.parent / f"{temp_path.name}{suffix}")

            if immutable:
                conn = sqlite3.connect(f"file:{temp_path}?mode=ro&immutable=1", uri=True)
            else:
                conn = sqlite3.connect(f"file:{temp_path}?mode=ro", uri=True)
        else:
            raise

    try:
        yield conn
    finally:
        if conn:
            conn.close()
        # Clean up temp file if we created one
        if temp_path and temp_path.exists():
            temp_dir = temp_path.parent
            with suppress(Exception):
                shutil.rmtree(temp_dir)


@contextmanager
def writable_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """
    Context manager for writable database connections.

    Opens database in read-write mode with automatic commit/rollback.
    Commits on successful completion, rolls back on exception.

    Args:
        db_path: Path to SQLite database

    Yields:
        Writable SQLite connection

    Example:
        >>> from pathlib import Path
        >>> with writable_connection(Path("data.db")) as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("INSERT INTO table VALUES (?)", (value,))
        ...     # Automatically commits on exit
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def deduplicate_table_by_content(
    con: sqlite3.Connection,
    table_name: str,
    exclude_columns: set[str] | None = None,
) -> tuple[int, int]:
    """
    Remove duplicate rows from a table based on content hash.

    Deduplicates by hashing non-PK column values and keeping first occurrence.
    IMPORTANT: Preserves original schema exactly (types, constraints, etc).

    Args:
        con: Database connection (caller manages commit/close)
        table_name: Name of table to deduplicate
        exclude_columns: Additional columns to exclude from hash (e.g., {"data_source"})

    Returns:
        Tuple of (original_count, unique_count)
    """
    q_table = quote_identifier(table_name)
    exclude_columns = exclude_columns or set()

    # SAVE original schema DDL - critical for schema preservation
    schema_row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        [table_name],
    ).fetchone()
    if not schema_row or not schema_row[0]:
        return 0, 0
    original_schema_ddl = schema_row[0]

    # Get column list and identify PRIMARY KEY columns
    cursor = con.execute(f"PRAGMA table_info({q_table})")
    all_cols_info = cursor.fetchall()

    if not all_cols_info:
        return 0, 0

    # Extract all column names and identify PK columns
    # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
    all_columns = [row[1] for row in all_cols_info]
    pk_columns = {row[1] for row in all_cols_info if row[5] > 0}

    # Common ID column names that are auto-generated and should be excluded
    # These often differ between original and recovered data
    id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt"}

    # Create a map of column names to types for BLOB handling
    col_types = {row[1]: row[2] for row in all_cols_info}

    # Exclude PK columns, common ID columns, and any additional excluded columns from hash
    hash_columns = [
        c for c in all_columns if c not in pk_columns and c.lower() not in id_column_names and c not in exclude_columns
    ]

    if not hash_columns:
        # Table only has PK/excluded columns, nothing to deduplicate by
        return 0, 0

    # Count original rows
    cursor = con.execute(f"SELECT COUNT(*) FROM {q_table}")
    original_count = cursor.fetchone()[0]

    if original_count == 0:
        return 0, 0

    # Create temporary table for unique rows (simplified schema OK for temp)
    temp_table = f"{table_name}_dedup_temp"
    q_temp = quote_identifier(temp_table)

    # Drop temp table if exists from previous failed run
    con.execute(f"DROP TABLE IF EXISTS {q_temp}")
    con.execute(f"CREATE TABLE {q_temp} AS SELECT * FROM {q_table} WHERE 0")

    # Build hash expression for data columns
    # Use hex() for BLOB columns to avoid UTF-8 errors
    hash_parts = []
    for c in hash_columns:
        col_type = col_types.get(c, "").upper()
        if col_type == "BLOB":
            hash_parts.append(f"COALESCE(hex({quote_identifier(c)}), 'NULL')")
        else:
            hash_parts.append(f"COALESCE(CAST({quote_identifier(c)} AS TEXT), 'NULL')")
    hash_expr = " || '|' || ".join(hash_parts)

    # Insert unique rows only (first occurrence wins based on rowid)
    col_list = ", ".join(quote_identifier(c) for c in all_columns)
    dedupe_sql_with_rowid = f"""
    INSERT INTO {q_temp}
    SELECT {col_list} FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY ({hash_expr})
                   ORDER BY ROWID
               ) as rn
        FROM {q_table}
    )
    WHERE rn = 1
    """

    dedupe_sql_no_rowid = f"""
    INSERT INTO {q_temp}
    SELECT {col_list} FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY ({hash_expr})
               ) as rn
        FROM {q_table}
    )
    WHERE rn = 1
    """

    use_python_fallback = False
    try:
        con.execute(dedupe_sql_with_rowid)
    except sqlite3.OperationalError as e:
        if "no such column: rowid" in str(e).lower():
            # WITHOUT ROWID table - try without ORDER BY
            try:
                con.execute(dedupe_sql_no_rowid)
            except sqlite3.OperationalError:
                use_python_fallback = True
        else:
            use_python_fallback = True

    if use_python_fallback:
        # Fallback for older SQLite without window functions
        seen_hashes: set[str] = set()

        # Try with ORDER BY rowid for deterministic deduplication
        try:
            cursor = con.execute(f"SELECT * FROM {q_table} ORDER BY rowid")
        except sqlite3.OperationalError as e:
            if "no such column: rowid" in str(e).lower():
                cursor = con.execute(f"SELECT * FROM {q_table}")
            else:
                raise
        rows = cursor.fetchall()

        quoted_cols = [quote_identifier(c) for c in all_columns]
        insert_placeholders = ", ".join(["?" for _ in all_columns])
        insert_sql = f"INSERT INTO {q_temp} ({', '.join(quoted_cols)}) VALUES ({insert_placeholders})"

        for row in rows:
            # Calculate hash of data columns only
            row_dict = dict(zip(all_columns, row))
            data_values = []
            for c in hash_columns:
                val = row_dict.get(c)
                if val is None:
                    data_values.append("NULL")
                elif isinstance(val, bytes):
                    data_values.append(val.hex())
                else:
                    data_values.append(str(val))
            row_hash = hashlib.md5("|".join(data_values).encode()).hexdigest()

            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                con.execute(insert_sql, row)

    # Count unique rows
    cursor = con.execute(f"SELECT COUNT(*) FROM {q_temp}")
    unique_count = cursor.fetchone()[0]

    # SCHEMA PRESERVATION: Recreate table with original DDL instead of renaming temp
    # Step 1: Drop original table
    con.execute(f"DROP TABLE {q_table}")

    # Step 2: Recreate with ORIGINAL schema (preserves INTEGER vs INT, TIMESTAMP vs NUM, etc)
    con.execute(original_schema_ddl)

    # Step 3: Copy deduplicated data back
    # For tables with INTEGER PRIMARY KEY, exclude PK to let SQLite auto-assign
    if pk_columns and len(pk_columns) == 1:
        pk_col = next(iter(pk_columns))
        pk_type = col_types.get(pk_col, "").upper()
        if pk_type in ("INTEGER", "INT"):
            # Exclude INTEGER PRIMARY KEY - let SQLite auto-assign
            non_pk_cols = [c for c in all_columns if c not in pk_columns]
            if non_pk_cols:
                non_pk_col_list = ", ".join(quote_identifier(c) for c in non_pk_cols)
                con.execute(f"INSERT INTO {q_table} ({non_pk_col_list}) SELECT {non_pk_col_list} FROM {q_temp}")
            else:
                # Only PK column - copy as-is
                con.execute(f"INSERT INTO {q_table} SELECT * FROM {q_temp}")
        else:
            # Non-integer PK - copy all columns
            con.execute(f"INSERT INTO {q_table} SELECT * FROM {q_temp}")
    else:
        # No PK or composite PK - copy all columns
        con.execute(f"INSERT INTO {q_table} SELECT * FROM {q_temp}")

    # Step 4: Drop temp table
    con.execute(f"DROP TABLE {q_temp}")

    return original_count, unique_count
