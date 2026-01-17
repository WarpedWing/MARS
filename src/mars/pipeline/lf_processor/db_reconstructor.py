"""
Database reconstruction from exemplar schema + lost_and_found data.

Reconstructs databases from exemplar PRAGMA schemas and inserts matched LF data
with provenance tracking (data_source column).
"""

import contextlib
import gc
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mars.pipeline.matcher.rubric_utils import (
    TimestampFormat,
    detect_enhanced_pattern_type,
    detect_pattern_type,
)
from mars.utils.database_utils import quote_identifier, readonly_connection
from mars.utils.debug_logger import logger


def _validate_row_against_rubric(
    vals: tuple,
    cols: list[str],
    rubric_metadata: dict | None,
    target_table: str,
) -> tuple[bool, str | None]:
    """
    Validate row data based on rubric roles (data-driven with 100% confidence).

    This is a shared implementation used by both insert_lf_data_into_table() and
    copy_table_data_with_provenance() to avoid code duplication.

    Only validates roles that were confidently set by analyzing actual data.
    Supports both single roles ("timestamp") and multi-roles (["url", "path"]).

    Args:
        vals: Row values to validate
        cols: Column names corresponding to values
        rubric_metadata: Rubric metadata dict with table/column role information
        target_table: Name of target table for rubric lookup

    Returns:
        (is_valid, rejection_reason) tuple
    """
    if not rubric_metadata or "tables" not in rubric_metadata:
        return True, None

    # Navigate to the table's columns in the rubric
    table_rubric = rubric_metadata.get("tables", {}).get(target_table, {})
    columns_list = table_rubric.get("columns", [])

    # Build column metadata lookup (rubric uses list, not dict)
    col_metadata = {c["name"]: c for c in columns_list if "name" in c}

    for col, val in zip(cols, vals):
        if val is None:
            continue

        col_meta = col_metadata.get(col)
        if not col_meta:
            continue

        role = col_meta.get("role")
        if not role:
            continue

        # Handle both single role (string) and multi-role (list)
        roles = [role] if isinstance(role, str) else role

        # Validate timestamp role (both regular and nullable)
        is_timestamp_role = "timestamp" in roles or "nullable_timestamp" in roles
        if is_timestamp_role:
            # Timestamp columns require numeric values (100% confidence)
            if not isinstance(val, (int, float)):
                return (
                    False,
                    f"{col}={val} (non-numeric value in timestamp column, roles={roles})",
                )

            # Zero is a universal "null timestamp" sentinel across all platforms
            # (Unix epoch, Cocoa Jan 1 2001, WebKit year 1601)
            if val == 0:
                pass  # Zero is valid as null timestamp sentinel
            else:
                # Validate against the detected timestamp format (the "lock and key" system)
                expected_format = col_meta.get("timestamp_format")

                # Allow Apple sentinel values (distantPast/distantFuture) to pass
                # These represent null/empty dates in Cocoa databases
                if TimestampFormat.is_apple_timestamp_sentinel(val):
                    pass  # Sentinel values are valid, skip further timestamp checks
                elif expected_format:
                    detected_format = TimestampFormat.detect_timestamp_format(val)
                    if not detected_format:
                        return (
                            False,
                            f"{col}={val} (not a valid timestamp in any known format, expected {expected_format})",
                        )

                    # Compare format names
                    # "unknown" format accepts any valid detected format
                    detected_name = str(detected_format)
                    if expected_format != "unknown" and detected_name != expected_format:
                        return (
                            False,
                            f"{col}={val} (timestamp format mismatch: detected {detected_name}, expected {expected_format})",
                        )
                else:
                    # Fallback: basic range checks
                    if val < 0:
                        return (
                            False,
                            f"{col}={val} (negative timestamp, roles={roles})",
                        )

                    if val > 4102444800000000000:  # Year 2100 in nanoseconds
                        return (
                            False,
                            f"{col}={val} (timestamp too large, roles={roles})",
                        )

        # Validate UUID role
        if "uuid" in roles:
            try:
                # Convert bytes to string if needed
                text = (
                    val.decode("utf-8", errors="ignore")
                    if isinstance(val, bytes)
                    else str(val)
                    if val is not None
                    else ""
                )

                # Check if value is a valid UUID
                if detect_pattern_type(text) != "uuid":
                    return (
                        False,
                        f"{col}={text[:50]} (not a valid UUID, roles={roles})",
                    )
            except Exception:
                return (
                    False,
                    f"{col}=<error decoding> (UUID validation failed, roles={roles})",
                )

        # Validate multi-role text patterns (url, email, path, domain)
        text_pattern_roles = {"url", "email", "path", "domain"}
        detected_pattern_roles = [r for r in roles if r in text_pattern_roles]

        if detected_pattern_roles:
            try:
                # Convert to string if needed
                text_val = str(val) if val is not None else ""

                # Allow empty strings (treat like NULL)
                if text_val == "":
                    continue

                # Try enhanced detection first
                detected = detect_enhanced_pattern_type(text_val)
                if not detected:
                    # Fallback to basic detection
                    detected = detect_pattern_type(text_val)

                # For multi-role columns, the value must match ALL expected patterns
                if detected:
                    # Check if detected pattern is in the expected roles
                    if detected not in detected_pattern_roles:
                        return (
                            False,
                            f"{col}={text_val[:50]} (pattern mismatch: detected {detected}, expected one of {detected_pattern_roles})",
                        )
                else:
                    return (
                        False,
                        f"{col}={text_val[:50]} (no pattern detected, expected one of {detected_pattern_roles})",
                    )
            except Exception:
                return (
                    False,
                    f"{col}=<error> (pattern validation failed, roles={roles})",
                )

        # Validate alphanumeric_id role
        # These must contain BOTH letters and digits (not pure numbers or pure text)
        if "alphanumeric_id" in roles:
            try:
                # Convert to string if needed
                text_val = str(val) if val is not None else ""

                # Allow empty strings (treat like NULL)
                if text_val == "":
                    continue

                text_length = len(text_val)
                uniform_length = None
                string_stats = col_meta.get("string_stats", {})
                if string_stats:
                    max_length = string_stats.get("max_length", None)
                    min_length = string_stats.get("min_length", None)
                    uniform_length = max_length if max_length and (max_length == min_length) else None

                # Check for at least one letter AND at least one digit
                has_letter = bool(re.search(r"[a-zA-Z]", text_val))
                has_digit = bool(re.search(r"\d", text_val))

                if not ((has_letter and has_digit) or (text_length == uniform_length and (has_letter or has_digit))):
                    return (
                        False,
                        f"{col}={text_val[:50]} (alphanumeric_id requires both letters and digits, or uniform length, roles={roles})",
                    )
            except Exception:
                return (
                    False,
                    f"{col}=<error> (alphanumeric_id validation failed, roles={roles})",
                )

        # Validate text_only role - reject non-string values in TEXT columns
        # This prevents integers/floats/bytes from being coerced into text-only columns
        # L&F tables may store values as BLOBs, which come as bytes - also reject those
        # Also reject string values that are purely numeric (e.g., "139" instead of "processOutgoingQueue")
        if "text_only" in roles:
            if isinstance(val, bytes):
                # Try to decode BLOB as UTF-8 to see what's inside
                try:
                    decoded = val.decode("utf-8")
                    # If it decodes to a number string, it's not valid text
                    if decoded.lstrip("-").replace(".", "").isdigit():
                        return (
                            False,
                            f"{col}={decoded} (BLOB contains numeric string in text_only column)",
                        )
                    # Otherwise accept the decoded string value
                    # Note: we can't modify val here, so this is just for validation
                except UnicodeDecodeError:
                    return (
                        False,
                        f"{col}=<binary> (BLOB in text_only column cannot be decoded as text)",
                    )
            elif not isinstance(val, str):
                return (
                    False,
                    f"{col}={val} (text_only column requires string value, got {type(val).__name__})",
                )
            elif val.lstrip("-").replace(".", "").isdigit():
                # Reject string values that are purely numeric
                return (
                    False,
                    f"{col}={val} (text_only column contains numeric string)",
                )

    return True, None


def cleanup_wal_files(db_path: Path) -> None:
    """
    Delete WAL and SHM files if they exist.

    Required for Windows compatibility - even after PRAGMA journal_mode=DELETE,
    Windows may keep -wal and -shm files on disk and locked. Explicitly deleting
    them ensures subsequent operations (like shutil.move) won't fail.

    Args:
        db_path: Path to the SQLite database file
    """
    for suffix in ["-wal", "-shm"]:
        wal_path = Path(str(db_path) + suffix)
        if wal_path.exists():
            with contextlib.suppress(Exception):
                wal_path.unlink()


def validate_column_type_affinity(
    row_vals: tuple,
    source_columns: list[str],
    target_schema_rows: list[tuple],
) -> tuple[bool, str | None]:
    """
    Validate that actual data types match target column type declarations.

    This catches cases where sqlite3's .recover command reconstructs tables with
    correct schema but places data into wrong tables (e.g., TEXT data from a 'files'
    table placed into an INTEGER column of a 'thumbnails' table).

    Args:
        row_vals: Row data values
        source_columns: Column names for the row values
        target_schema_rows: Schema rows from PRAGMA table_info
            (cid, name, type, notnull, dflt_value, pk)

    Returns:
        (is_valid, rejection_reason) - is_valid=False if type mismatch detected
    """
    # Build schema lookup: column_name -> declared_type
    target_schema = {row[1]: (row[2] or "").upper() for row in target_schema_rows}

    for col, val in zip(source_columns, row_vals):
        if val is None:
            continue

        col_type = target_schema.get(col, "")

        # Validate INTEGER columns - reject non-numeric TEXT values
        # SQLite's INTEGER affinity: "INT" appears anywhere in type name
        if "INT" in col_type and isinstance(val, str):
            # TEXT value in INTEGER column is a type mismatch
            # Truncate long values for readability in error message
            val_preview = val[:40] + "..." if len(val) > 40 else val
            return (
                False,
                f"{col}: TEXT value '{val_preview}' in INTEGER column",
            )

        # Validate REAL/FLOAT columns - reject non-numeric TEXT values
        # SQLite's REAL affinity: "REAL", "FLOA", "DOUB" in type name
        if any(t in col_type for t in ("REAL", "FLOA", "DOUB")) and isinstance(val, str):
            # Check if the string is a valid numeric value
            try:
                float(val)
                # It's a numeric string, which is acceptable
            except ValueError:
                val_preview = val[:40] + "..." if len(val) > 40 else val
                return (
                    False,
                    f"{col}: non-numeric TEXT '{val_preview}' in REAL column",
                )

    return True, None


def detect_and_fix_byteswapped_text(text: str) -> tuple[str, bool]:
    """
    Detect if string is misinterpreted UTF-16 ASCII and correct it.

    Some L&F recovered text appears as CJK characters but is actually ASCII that
    was stored as UTF-16LE and misinterpreted by SQLite recovery. Each pair of
    ASCII bytes (char + null) was read as a single CJK codepoint.

    The fix: Split each CJK character back into its two component bytes.

    Examples:
        - `祳瑳浥業牧瑡潩摮` → `systemmigrationd`
        - `潣⹭灡汰⹥敮扴潩摳` → `com.apple.netbiosd`

    Args:
        text: String to check and potentially correct

    Returns:
        (corrected_text, was_swapped) - corrected text and whether correction occurred
    """
    if not text or not isinstance(text, str):
        return text, False

    # Check if string contains high codepoints (CJK or other non-ASCII)
    # that could be misinterpreted UTF-16LE pairs
    has_high_codepoints = any(ord(c) > 0xFF for c in text)
    if not has_high_codepoints:
        return text, False

    # Split each character into its two component bytes (little-endian)
    # Each CJK char's codepoint is split: low byte first, high byte second
    try:
        result = []
        for char in text:
            code = ord(char)
            low_byte = code & 0xFF
            high_byte = (code >> 8) & 0xFF
            result.append(chr(low_byte))
            result.append(chr(high_byte))

        decoded = "".join(result)

        # Check if result is printable ASCII (allowing null bytes which we'll strip)
        # Valid if all chars are either printable ASCII or null
        if all(c == "\x00" or (32 <= ord(c) < 127) or c in "\n\r\t" for c in decoded):
            # Strip null bytes (the high bytes of ASCII chars in UTF-16LE)
            cleaned = decoded.replace("\x00", "")
            if cleaned and len(cleaned) > 0:
                return cleaned, True
    except Exception:
        pass

    return text, False


def fix_byteswapped_row(row: tuple) -> tuple[tuple, int]:
    """
    Apply byte-swap correction to all string values in a row.

    Args:
        row: Tuple of row values

    Returns:
        (corrected_row, num_corrections) - corrected row and count of fixed values
    """
    corrected = []
    num_corrections = 0

    for val in row:
        if isinstance(val, str):
            fixed, was_swapped = detect_and_fix_byteswapped_text(val)
            if was_swapped:
                num_corrections += 1
            corrected.append(fixed)
        else:
            corrected.append(val)

    return tuple(corrected), num_corrections


def get_table_schema(db_path: Path, table_name: str) -> dict[str, Any]:
    """
    Extract full schema for a table including columns, types, constraints, and indices.

    Args:
        db_path: Path to source database
        table_name: Name of table to extract schema from

    Returns:
        {
            "columns": [(name, type, notnull, dflt_value, pk), ...],
            "indices": [{"name": str, "sql": str, "unique": bool}, ...],
            "create_sql": str,  # Original CREATE TABLE statement
        }
    """
    with readonly_connection(db_path) as con:
        # Get column info
        cursor = con.execute(f"PRAGMA table_info({quote_identifier(table_name)})")
        columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

        # Get CREATE TABLE statement
        cursor = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        row = cursor.fetchone()
        create_sql = row[0] if row else None

        # Get indices
        cursor = con.execute(f"PRAGMA index_list({quote_identifier(table_name)})")
        index_list = cursor.fetchall()  # (seq, name, unique, origin, partial)

        indices = []
        for idx_info in index_list:
            idx_name = idx_info[1]
            idx_unique = idx_info[2]

            # Get CREATE INDEX statement
            cursor = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                (idx_name,),
            )
            idx_row = cursor.fetchone()
            idx_sql = idx_row[0] if idx_row else None

            if idx_sql:  # Skip auto-created indices
                indices.append({"name": idx_name, "sql": idx_sql, "unique": idx_unique})

        return {
            "columns": columns,
            "indices": indices,
            "create_sql": create_sql,
        }


def reconstruct_database_with_schema(
    output_db_path: Path,
    table_schemas: dict[str, dict],
    add_data_source_column: bool = True,
) -> None:
    """
    Create a new database with tables matching exemplar schemas.

    Args:
        output_db_path: Path where reconstructed database will be created
        table_schemas: Dict mapping table_name -> schema dict (from get_table_schema)
        add_data_source_column: If True, add 'data_source' TEXT column to each table
    """
    # Remove existing file if present
    if output_db_path.exists():
        output_db_path.unlink()

    with sqlite3.connect(output_db_path) as con:
        # Performance optimizations
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")

        # First pass: identify actual FTS virtual tables by their CREATE SQL
        # This prevents false positives (e.g., "collection_data" is not an FTS shadow table)
        fts_table_names = set()
        for table_name, schema in table_schemas.items():
            create_sql = schema.get("create_sql", "")
            if create_sql and re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE):
                fts_table_names.add(table_name)

        # Build set of actual FTS shadow table names based on detected FTS tables
        # FTS3/FTS4 suffixes: _content, _segdir, _segments, _stat
        # FTS5 suffixes: _content, _data, _idx, _config, _docsize
        fts_auxiliary_suffixes = [
            "_content",
            "_data",  # FTS5
            "_idx",  # FTS5
            "_config",  # FTS5
            "_docsize",
            "_segdir",  # FTS3/FTS4
            "_segments",  # FTS3/FTS4
            "_stat",  # FTS3/FTS4
        ]
        fts_shadow_tables = set()
        for fts_name in fts_table_names:
            for suffix in fts_auxiliary_suffixes:
                fts_shadow_tables.add(f"{fts_name}{suffix}")

        for table_name, schema in table_schemas.items():
            # Skip SQLite internal tables (auto-created/managed by SQLite)
            if table_name in ["sqlite_sequence", "sqlite_stat1", "sqlite_stat4"]:
                continue

            # Skip FTS shadow tables (auto-created by FTS virtual tables)
            # Only skip if this is actually a shadow table for a detected FTS table
            if table_name in fts_shadow_tables:
                continue

            # Use original CREATE TABLE statement if available
            create_sql = schema.get("create_sql")

            if create_sql:
                # Skip FTS virtual tables - create as-is without modification
                if re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE):
                    con.execute(create_sql)
                    continue

                # Remove STRICT keyword to allow BLOB data in TEXT columns
                # STRICT tables enforce type affinity, rejecting BLOBs in TEXT columns
                # For forensic recovery, we need to accept any data type
                create_sql = re.sub(r"\s+STRICT\b", "", create_sql, flags=re.IGNORECASE)

                # If we're adding data_source column, we need to modify the CREATE statement
                if add_data_source_column and "data_source" not in create_sql.lower():
                    # Strategy: Insert data_source BEFORE any table-level constraints
                    # In SQLite, column definitions MUST come before table-level constraints
                    # (PRIMARY KEY, FOREIGN KEY, CHECK, UNIQUE)

                    # Remove trailing ) WITHOUT ROWID or ); or )
                    stripped_sql = create_sql.rstrip()
                    has_semicolon = stripped_sql.endswith(");")
                    has_without_rowid = bool(re.search(r"\)\s*WITHOUT\s+ROWID", stripped_sql, re.IGNORECASE))

                    # Remove trailing bits
                    if has_without_rowid:
                        # Remove ) WITHOUT ROWID or ); WITHOUT ROWID
                        stripped_sql = re.sub(
                            r"\)\s*WITHOUT\s+ROWID\s*;?\s*$",
                            "",
                            stripped_sql,
                            flags=re.IGNORECASE,
                        )
                    elif has_semicolon:
                        stripped_sql = stripped_sql[:-2]
                    elif stripped_sql.endswith(")"):
                        stripped_sql = stripped_sql[:-1]

                    # Look for table-level constraints
                    # These must come AFTER all column definitions
                    constraint_pattern = (
                        r",\s*("
                        r"(?:CONSTRAINT\s+\w+\s+)?"  # Optional: CONSTRAINT name
                        r"(?:PRIMARY\s+KEY|FOREIGN\s+KEY|CHECK|UNIQUE)\s*\("  # Table constraints
                        r")"
                    )

                    match = re.search(constraint_pattern, stripped_sql, re.IGNORECASE)

                    if match:
                        # Insert data_source BEFORE the first table-level constraint
                        insertion_point = match.start()
                        create_sql = (
                            stripped_sql[:insertion_point] + ",\n    data_source TEXT" + stripped_sql[insertion_point:]
                        )
                    else:
                        # No table-level constraints, add at the end
                        create_sql = stripped_sql + ",\n    data_source TEXT"

                    # Re-add closing paren and optional clauses
                    create_sql += "\n)"
                    if has_without_rowid:
                        create_sql += " WITHOUT ROWID"
                    if has_semicolon:
                        create_sql += ";"

                try:
                    con.execute(create_sql)
                except Exception:
                    raise
            else:
                # Fallback: construct CREATE TABLE from column info
                columns_def = []
                pk_cols = []

                for col_info in schema["columns"]:
                    cid, name, col_type, notnull, dflt_value, pk = col_info

                    col_def = f"{name} {col_type}"

                    if notnull:
                        col_def += " NOT NULL"

                    if dflt_value is not None:
                        col_def += f" DEFAULT {dflt_value}"

                    if pk:
                        pk_cols.append(name)

                    columns_def.append(col_def)

                # Add data_source column
                if add_data_source_column:
                    columns_def.append("data_source TEXT")

                # Add PRIMARY KEY constraint if needed
                if pk_cols:
                    pk_constraint = f"PRIMARY KEY ({', '.join(pk_cols)})"
                    columns_def.append(pk_constraint)

                create_statement = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns_def) + "\n)"

                con.execute(create_statement)

            # Create indices
            for idx in schema.get("indices", []):
                idx_sql = idx["sql"]
                if idx_sql:
                    con.execute(idx_sql)

        con.commit()

        # Switch from WAL to DELETE mode to ensure schema is in the main file
        # This removes the WAL file and makes the database fully self-contained
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        con.execute("PRAGMA journal_mode=DELETE;")

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(output_db_path)


def insert_lf_data_into_table(
    target_db: Path,
    target_table: str,
    source_lf_db: Path,
    source_lf_tables: list[str],
    column_mapping: dict[str, str],
    data_source_value: str = "found",
    rubric_metadata: dict | None = None,
) -> int:
    """
    Insert data from LF split tables into a reconstructed table.

    Handles column mapping (c0, c1, c2 -> actual column names) and adds
    data_source provenance.

    Args:
        target_db: Path to reconstructed database
        target_table: Name of table to insert into
        source_lf_db: Path to split database with LF tables
        source_lf_tables: List of LF table names to combine (e.g., ["lf_table_1", "lf_table_5"])
        column_mapping: Dict mapping LF columns to target columns (e.g., {"c0": "id", "c1": "name"})
        data_source_value: Value to use for data_source column ("found", "carved")
        rubric_metadata: Optional pre-loaded rubric metadata dict for validation

    Returns:
        Number of rows inserted
    """
    total_inserted = 0

    # Build INSERT statement
    target_columns = list(column_mapping.values())
    source_columns = list(column_mapping.keys())

    # Add data_source column
    target_columns.append("data_source")

    # Wrapper for module-level validation function (captures rubric_metadata and target_table)
    def validate_row_types(vals: tuple, cols: list[str]) -> tuple[bool, str | None]:
        return _validate_row_against_rubric(vals, cols, rubric_metadata, target_table)

    # Properly quote all identifiers
    quoted_target_cols = [quote_identifier(col) for col in target_columns]
    placeholders = ", ".join(["?" for _ in target_columns])
    # Use INSERT OR IGNORE to skip rows that violate UNIQUE constraints
    insert_sql = f"INSERT OR IGNORE INTO {quote_identifier(target_table)} ({', '.join(quoted_target_cols)}) VALUES ({placeholders})"

    # Create rejected database to preserve all rejected rows
    rejected_db_dir = target_db.parent / "rejected"
    rejected_db_dir.mkdir(parents=True, exist_ok=True)
    rejected_db_path = rejected_db_dir / f"{target_db.stem}_rejected{target_db.suffix}"
    rejected_table_created = False
    rows_rejected = 0
    rejection_reasons = []
    rows_byteswap_corrected = 0

    # Use ExitStack for proper connection cleanup to prevent FD exhaustion
    with contextlib.ExitStack() as stack:
        target_con = stack.enter_context(sqlite3.connect(target_db))
        rejected_con = None  # Opened lazily and added to stack when needed

        # Process each LF table
        with readonly_connection(source_lf_db) as source_con:
            for lf_table in source_lf_tables:
                # Read data from LF table with properly quoted identifiers
                # Try ORDER BY rowid for deterministic processing (falls back for WITHOUT ROWID tables)
                quoted_source_cols = [quote_identifier(col) for col in source_columns]
                base_select_sql = f"SELECT {', '.join(quoted_source_cols)} FROM {quote_identifier(lf_table)}"
                try:
                    cursor = source_con.execute(f"{base_select_sql} ORDER BY rowid")
                except sqlite3.OperationalError as e:
                    if "no such column: rowid" in str(e).lower():
                        # WITHOUT ROWID table - fall back to unordered select
                        cursor = source_con.execute(base_select_sql)
                    else:
                        raise

                # Insert into target table (with validation and data_source value appended)
                for row in cursor:
                    # Reject rows with insufficient meaningful data
                    # Count non-NULL, non-empty, non-zero values
                    # Zero (0) is treated as empty-ish since it's often a placeholder
                    meaningful_count = sum(1 for val in row if val is not None and val != "" and val != 0)

                    # Require at least 2 meaningful values
                    # This rejects near-empty rows (e.g., NULL NULL NULL NULL 0 NULL)
                    # while allowing legitimate data in large tables with many NULLs/flags
                    if meaningful_count < 2:
                        rows_rejected += 1
                        if rows_rejected <= 5:
                            rejection_reasons.append(f"insufficient data ({meaningful_count} meaningful values)")
                        continue

                    # Validate row types before inserting
                    target_cols = list(column_mapping.values())
                    is_valid, rejection_reason = validate_row_types(row, target_cols)
                    if not is_valid:
                        rows_rejected += 1
                        # Collect first 5 rejection reasons for debugging
                        if rows_rejected <= 5:
                            rejection_reasons.append(rejection_reason)

                        # Save rejected row to rejected database (lazy open, added to stack)
                        if rejected_con is None:
                            rejected_con = stack.enter_context(sqlite3.connect(rejected_db_path))

                        # Define rejected_columns (used for both table creation and insertion)
                        rejected_columns = target_columns + ["rejection_reason"]

                        # Create rejected table structure on first rejection
                        if not rejected_table_created:
                            # Copy table schema from target, add rejection_reason column
                            rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]

                            # Get column definitions from target table
                            cursor_schema = target_con.execute(f"PRAGMA table_info({quote_identifier(target_table)})")
                            target_schema = cursor_schema.fetchall()

                            # Build CREATE TABLE statement
                            col_defs = []
                            for col_info in target_schema:
                                col_name = col_info[1]
                                col_type = col_info[2] or "BLOB"
                                col_defs.append(f"{quote_identifier(col_name)} {col_type}")

                            # Add rejection_reason column
                            col_defs.append(f"{quote_identifier('rejection_reason')} TEXT")

                            create_sql = (
                                f"CREATE TABLE IF NOT EXISTS {quote_identifier(target_table)} ({', '.join(col_defs)})"
                            )
                            rejected_con.execute(create_sql)
                            rejected_table_created = True

                        # Insert rejected row with rejection reason
                        row_with_source_and_reason = list(row) + [
                            data_source_value,
                            rejection_reason,
                        ]
                        rejected_placeholders = ", ".join(["?" for _ in rejected_columns])
                        rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]
                        rejected_insert_sql = f"INSERT OR IGNORE INTO {quote_identifier(target_table)} ({', '.join(rejected_quoted_cols)}) VALUES ({rejected_placeholders})"
                        rejected_con.execute(rejected_insert_sql, row_with_source_and_reason)

                        continue  # Skip this malformed row from main database

                    # Apply byte-swap correction for CJK text that's actually ASCII
                    corrected_row, num_corrections = fix_byteswapped_row(row)
                    if num_corrections > 0:
                        rows_byteswap_corrected += 1
                        row = corrected_row

                    # Append data_source value to row data
                    row_with_source = list(row) + [data_source_value]
                    cursor_result = target_con.execute(insert_sql, row_with_source)
                    # Only count as inserted if row was actually added (not skipped by OR IGNORE)
                    total_inserted += cursor_result.rowcount

        # Commit and finalize before ExitStack closes connections
        target_con.commit()

        # Switch to DELETE mode to ensure data is in the main file
        target_con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        target_con.execute("PRAGMA journal_mode=DELETE;")

        # Commit rejected database if used
        if rejected_con:
            rejected_con.commit()
    # ExitStack automatically closes all connections here

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(target_db)

    # Log byte-swap corrections
    if rows_byteswap_corrected > 0:
        logger.debug(f"  L&F byte-swap corrections: {rows_byteswap_corrected} rows corrected in {target_table}")

    # Per-table gc.collect() removed - now handled per-source-database in caller
    # to reduce overhead while still preventing handle exhaustion
    # gc.collect()

    return total_inserted


def copy_table_data_with_provenance(
    target_db: Path,
    target_table: str,
    source_db: Path,
    source_table: str,
    data_source_value: str = "intact",
    exemplar_schemas_dir: Path | None = None,
    rubric_metadata: dict | None = None,
) -> int:
    """
    Copy data from source table to target table, adding data_source column.

    Args:
        target_db: Path to target database
        target_table: Name of table to insert into
        source_db: Path to source database
        source_table: Name of table to copy from
        data_source_value: Value to use for data_source column (default: "intact")
        exemplar_schemas_dir: Optional path to exemplar schemas directory for rubric validation
        rubric_metadata: Optional pre-loaded rubric metadata dict (avoids loading on every call)

    Returns:
        Number of rows copied
    """
    # Get column lists from source table first (before opening target connection)
    with readonly_connection(source_db) as source_con:
        cursor = source_con.execute(f"PRAGMA table_info({quote_identifier(source_table)})")
        source_columns_all = [row[1] for row in cursor.fetchall() if row[1] != "data_source"]

    # Use ExitStack for proper connection cleanup to prevent FD exhaustion
    with contextlib.ExitStack() as stack:
        target_con = stack.enter_context(sqlite3.connect(target_db))
        rejected_con = None  # Opened lazily and added to stack when needed

        cursor = target_con.execute(f"PRAGMA table_info({quote_identifier(target_table)})")
        target_schema_rows = cursor.fetchall()
        target_columns_all = [row[1] for row in target_schema_rows if row[1] != "data_source"]

        # Only copy columns that exist in BOTH source and target (intersection)
        # This handles schema mismatches where source has different columns than target
        common_columns = [col for col in source_columns_all if col in target_columns_all]

        if not common_columns:
            # No common columns, can't copy any data
            # ExitStack will close target_con automatically
            return 0

        # Exclude INTEGER PRIMARY KEY columns to avoid ID collisions
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        integer_pk_cols = set()
        for col_info in target_schema_rows:
            col_name = col_info[1]
            col_type = col_info[2]
            is_pk = col_info[5]
            # Single-column INTEGER PRIMARY KEY is auto-increment in SQLite
            if is_pk == 1 and col_type.upper() in ("INTEGER", "INT"):
                # Check if it's the only PK column (not part of composite PK)
                pk_count = sum(1 for c in target_schema_rows if c[5] > 0)
                if pk_count == 1:
                    integer_pk_cols.add(col_name)

        # Filter out INTEGER PRIMARY KEY from copy - SQLite will auto-assign new IDs
        source_columns = [c for c in common_columns if c not in integer_pk_cols]

        if not source_columns:
            # Only INTEGER PRIMARY KEY columns - skip
            # ExitStack will close target_con automatically
            return 0

        # Load rubric metadata for data-driven validation
        # Roles are set only when 100% of sampled values match the pattern,
        # If rubric was passed as parameter, use it; otherwise load from disk

        if rubric_metadata is None:
            # No rubric passed - try loading from disk if exemplar_schemas_dir provided
            if exemplar_schemas_dir:
                rubric_metadata = {}
                for suffix in ["", "_combined"]:
                    # exemplar_schemas_dir is already schemas/{exemplar_name}/
                    # rubrics are at schemas/{exemplar_name}/{exemplar_name}.rubric.json
                    # Try finding the exemplar's main rubric
                    exemplar_name = exemplar_schemas_dir.name
                    rubric_path = exemplar_schemas_dir / f"{exemplar_name}{suffix}.rubric.json"
                    if rubric_path.exists():
                        try:
                            with Path.open(rubric_path) as f:
                                rubric_metadata = json.load(f)
                            break
                        except Exception as e:
                            logger.debug(f"[DEBUG] Failed to load rubric {rubric_path}: {e}")
            else:
                rubric_metadata = {}

        # Wrapper for module-level validation function (captures rubric_metadata and target_table)
        def validate_row_types(vals: tuple, cols: list[str]) -> tuple[bool, str | None]:
            return _validate_row_against_rubric(vals, cols, rubric_metadata, target_table)

        # Build INSERT statement with properly quoted identifiers
        target_columns = source_columns + ["data_source"]
        quoted_target_cols = [quote_identifier(col) for col in target_columns]
        placeholders = ", ".join(["?" for _ in target_columns])
        insert_sql = f"INSERT OR IGNORE INTO {quote_identifier(target_table)} ({', '.join(quoted_target_cols)}) VALUES ({placeholders})"

        # Read data from source table with properly quoted identifiers
        # Try ORDER BY rowid for deterministic processing order (falls back for WITHOUT ROWID tables)
        quoted_source_cols = [quote_identifier(col) for col in source_columns]
        base_select_sql = f"SELECT {', '.join(quoted_source_cols)} FROM {quote_identifier(source_table)}"

        # Fetch all rows first to handle STRICT table errors
        # STRICT tables may raise errors when reading BLOB data from TEXT columns
        rows_copied = 0
        with readonly_connection(source_db) as source_con:
            rows = None
            # Try with ORDER BY rowid first (deterministic for regular tables)
            try:
                cursor = source_con.execute(f"{base_select_sql} ORDER BY rowid")
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "no such column: rowid" in str(e).lower():
                    # WITHOUT ROWID table - fall back to unordered select
                    try:
                        cursor = source_con.execute(base_select_sql)
                        rows = cursor.fetchall()
                    except Exception:
                        rows = None
                # Other errors - try fallback below
            except Exception:
                rows = None

            # If rows is still None, try rowid-by-rowid fallback for STRICT tables
            if rows is None:
                rows = []
                try:
                    max_rowid = source_con.execute(
                        f"SELECT MAX(rowid) FROM {quote_identifier(source_table)}"
                    ).fetchone()[0]
                    if max_rowid:
                        for rowid in range(1, max_rowid + 1):
                            try:
                                row = source_con.execute(f"{base_select_sql} WHERE rowid = ?", (rowid,)).fetchone()
                                if row:
                                    rows.append(row)
                            except Exception:
                                # Skip rows that can't be read (STRICT violations)
                                continue
                except Exception:
                    # WITHOUT ROWID or other error - try without rowid filtering
                    try:
                        cursor = source_con.execute(base_select_sql)
                        rows = cursor.fetchall()
                    except Exception:
                        rows = []

        # Create rejected database to preserve all rejected rows
        # Database path: target_db.parent/rejected/{name}_rejected.sqlite
        # Example: catalog/Quarantine Events_admin/rejected/Quarantine Events_admin_rejected.sqlite
        rejected_db_dir = target_db.parent / "rejected"
        rejected_db_dir.mkdir(parents=True, exist_ok=True)
        rejected_db_path = rejected_db_dir / f"{target_db.stem}_rejected{target_db.suffix}"
        rejected_table_created = False

        # Insert rows into target with rubric-based validation
        rows_rejected = 0
        rows_exception = 0  # Track rows lost to exceptions
        rows_byteswap_corrected = 0  # Track rows with byte-swapped text corrections
        rejection_reasons = []
        for row in rows:
            try:
                # Reject completely NULL rows (no actual data)
                if all(val is None for val in row):
                    rows_rejected += 1
                    if rows_rejected <= 5:
                        rejection_reasons.append("all-NULL row (no data)")
                    continue

                # Validate type affinity first (catch TEXT in INTEGER columns, etc.)
                # This catches cases where .recover puts data in wrong tables
                is_valid, rejection_reason = validate_column_type_affinity(row, source_columns, target_schema_rows)
                if not is_valid:
                    rows_rejected += 1
                    if rows_rejected <= 5:
                        rejection_reasons.append(rejection_reason)

                    # Save rejected row to rejected database (lazy open, added to stack)
                    if rejected_con is None:
                        rejected_con = stack.enter_context(sqlite3.connect(rejected_db_path))

                    rejected_columns = target_columns + ["rejection_reason"]

                    if not rejected_table_created:
                        rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]
                        cursor = target_con.execute(f"PRAGMA table_info({quote_identifier(target_table)})")
                        target_schema = cursor.fetchall()
                        col_defs = []
                        for col_info in target_schema:
                            col_name = col_info[1]
                            col_type = col_info[2] or "BLOB"
                            col_defs.append(f"{quote_identifier(col_name)} {col_type}")
                        col_defs.append(f"{quote_identifier('rejection_reason')} TEXT")
                        create_sql = (
                            f"CREATE TABLE IF NOT EXISTS {quote_identifier(target_table)} ({', '.join(col_defs)})"
                        )
                        rejected_con.execute(create_sql)
                        rejected_table_created = True

                    row_with_source_and_reason = list(row) + [
                        data_source_value,
                        rejection_reason,
                    ]
                    rejected_placeholders = ", ".join(["?" for _ in rejected_columns])
                    rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]
                    rejected_insert_sql = f"INSERT OR IGNORE INTO {quote_identifier(target_table)} ({', '.join(rejected_quoted_cols)}) VALUES ({rejected_placeholders})"
                    rejected_con.execute(rejected_insert_sql, row_with_source_and_reason)

                    continue

                # Validate semantic roles (timestamps, UUIDs, etc.)
                is_valid, rejection_reason = validate_row_types(row, source_columns)
                if not is_valid:
                    rows_rejected += 1
                    # Collect first 5 rejection reasons for debugging
                    if rows_rejected <= 5:
                        rejection_reasons.append(rejection_reason)

                    # Save rejected row to rejected database (lazy open, added to stack)
                    if rejected_con is None:
                        rejected_con = stack.enter_context(sqlite3.connect(rejected_db_path))

                    # Define rejected_columns (used for both table creation and insertion)
                    rejected_columns = target_columns + ["rejection_reason"]

                    # Create rejected table structure on first rejection (includes rejection_reason column)
                    if not rejected_table_created:
                        # Copy table schema from target, add rejection_reason column
                        rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]

                        # Get column definitions from target table
                        cursor = target_con.execute(f"PRAGMA table_info({quote_identifier(target_table)})")
                        target_schema = cursor.fetchall()

                        # Build CREATE TABLE statement
                        col_defs = []
                        for col_info in target_schema:
                            col_name = col_info[1]
                            col_type = col_info[2] or "BLOB"
                            col_defs.append(f"{quote_identifier(col_name)} {col_type}")

                        # Add rejection_reason column
                        col_defs.append(f"{quote_identifier('rejection_reason')} TEXT")

                        create_sql = (
                            f"CREATE TABLE IF NOT EXISTS {quote_identifier(target_table)} ({', '.join(col_defs)})"
                        )
                        rejected_con.execute(create_sql)
                        rejected_table_created = True

                    # Insert rejected row with rejection reason
                    row_with_source_and_reason = list(row) + [
                        data_source_value,
                        rejection_reason,
                    ]
                    rejected_placeholders = ", ".join(["?" for _ in rejected_columns])
                    rejected_quoted_cols = [quote_identifier(col) for col in rejected_columns]
                    rejected_insert_sql = f"INSERT OR IGNORE INTO {quote_identifier(target_table)} ({', '.join(rejected_quoted_cols)}) VALUES ({rejected_placeholders})"
                    rejected_con.execute(rejected_insert_sql, row_with_source_and_reason)

                    continue  # Skip this malformed row from main database

                # Apply byte-swap correction for CJK text that's actually ASCII
                corrected_row, num_corrections = fix_byteswapped_row(row)
                if num_corrections > 0:
                    rows_byteswap_corrected += 1
                    row = corrected_row

                row_with_source = list(row) + [data_source_value]
                cursor_result = target_con.execute(insert_sql, row_with_source)
                rows_copied += cursor_result.rowcount
            except Exception as e:
                # Skip rows that can't be inserted (constraint violations, etc.)
                rows_exception += 1
                if rows_exception <= 5:
                    logger.debug(f"[DEBUG] Exception processing row in {target_table}: {e}")
                continue

        if rows_exception > 0:
            logger.debug(f"[DEBUG] Table {target_table}: {rows_exception} rows lost to exceptions")
        if rows_byteswap_corrected > 0:
            logger.debug(f"    ℹ {target_table}: corrected byte-swapped text in {rows_byteswap_corrected} rows")

        # Commit and finalize before ExitStack closes connections
        target_con.commit()

        # Switch to DELETE mode to ensure data is in the main file
        target_con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        target_con.execute("PRAGMA journal_mode=DELETE;")

        # Commit rejected database if any rejections occurred
        if rejected_con:
            rejected_con.commit()
    # ExitStack automatically closes all connections here

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(target_db)

    # Force garbage collection to release SQLite connections
    # gc.collect()

    return rows_copied


def deduplicate_table(db_path: Path, table_name: str) -> int:
    """
    Remove duplicate rows from a table based on content hash.

    Preserves the first occurrence of each unique row.
    IMPORTANT: Preserves original schema exactly (types, constraints, etc).

    Args:
        db_path: Path to database
        table_name: Name of table to deduplicate

    Returns:
        Number of duplicate rows removed
    """
    from mars.utils.database_utils import deduplicate_table_by_content

    with sqlite3.connect(db_path) as con:
        # Exclude data_source from hash - it's provenance metadata, not actual data
        original_count, unique_count = deduplicate_table_by_content(con, table_name, exclude_columns={"data_source"})

        con.commit()

        # Switch to DELETE mode to ensure changes are in the main file
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        con.execute("PRAGMA journal_mode=DELETE;")

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(db_path)

    # Force garbage collection to release SQLite connections
    gc.collect()

    return original_count - unique_count


def create_manifest_file(
    output_path: Path,
    output_type: str,
    output_name: str,
    source_databases: list[dict],
    combined_stats: dict,
    base_name: str | None = None,
    username: str | None = None,
    profiles: list[str] | None = None,
) -> bool:
    """
    Create a manifest.json file documenting the output database.

    Args:
        output_path: Path where manifest.json should be created (e.g., catalog/Powerlog/Powerlog_manifest.json)
        output_type: Type of output: "catalog", "metamatch", or "found_data"
        output_name: Name of the output (e.g., "Powerlog", "itemtable_d57c0edb")
        source_databases: List of dicts with source DB info:
            - db_name: str
            - intact_rows: int
            - lf_rows: int
            - remnant_tables: int (optional)
            - remnant_rows: int (optional)
        combined_stats: Dict with combined output statistics:
            - total_intact_rows: int
            - total_lf_rows: int
            - total_remnant_tables: int
            - total_remnant_rows: int
            - table_stats: list[dict] with table-level stats
        base_name: Catalog base name (e.g., "Chrome Cookies" for "Chrome Cookies_user")
            Used by comparison calculator to group variants by base name.
        username: Username for export path resolution (e.g., "username" for user-scoped DBs,
            "_multi" for shared tables). Used to resolve Users/* wildcards in canonical paths.
        profiles: List of Chrome/browser profiles this database matches (e.g., ["Default"],
            ["Profile 1"]). Used to resolve profile wildcards in canonical paths.

    Returns:
        True if successful, False otherwise
    """
    try:
        manifest = {
            "output_type": output_type,
            "output_name": output_name,
            "base_name": base_name or output_name,
            "username": username,
            "profiles": profiles or [],
            "created": datetime.now().isoformat(),
            "source_databases": source_databases,
            "combined_output": combined_stats,
        }

        with output_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"Error creating manifest: {e}")
        return False
