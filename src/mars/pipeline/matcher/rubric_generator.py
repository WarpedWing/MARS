#!/usr/bin/env python3
"""
Unified Rubric Generation Module

This module provides shared logic for generating rubrics from SQLite databases,
used by both:
1. Exemplar scanning (clean databases from intact systems)
2. Self-match rubrics (damaged/recovered databases during recovery)

By consolidating the logic, all improvements (UUID detection, timestamp format,
signature patterns, FK inference, example confidence) benefit both systems.
"""

from __future__ import annotations

import contextlib
import sqlite3
import statistics
from collections import Counter
from typing import Any

from mars.pipeline.matcher.rubric_utils import (
    TimestampFormat,
    detect_enhanced_pattern_type,
    detect_pattern_type,
    detect_programming_case_role,
)
from mars.utils.database_utils import quote_ident

# Minimum non-null sample size required before assigning semantic roles
# Small samples (< 10 rows) give meaningless 100% confidence
# Fall back to database column datatypes if sample size is insufficient
MIN_ROLE_SAMPLE_SIZE = 5


# =============================================================================
# Single-Pass Column Value Analyzer (Performance Optimization)
# =============================================================================
# Instead of iterating column values 5-7 times for different analyses,
# we collect all metrics in a single pass through the data.


class ColumnValueAnalysis:
    """Results from single-pass column value analysis."""

    __slots__ = (
        "null_count",
        "total_count",
        "empty_string_count",
        "non_null_values",
        "numeric_values",
        "text_values",
        "text_lengths",
        "has_numeric_strings",
        "bool_values",
    )

    def __init__(self) -> None:
        self.null_count: int = 0
        self.total_count: int = 0
        self.empty_string_count: int = 0
        self.non_null_values: list[Any] = []
        self.numeric_values: list[float] = []
        self.text_values: list[str] = []
        self.text_lengths: list[int] = []
        self.has_numeric_strings: bool = False
        self.bool_values: set[bool] | None = None  # None means mixed/non-bool

    @property
    def null_likelihood(self) -> float:
        """Calculate NULL likelihood ratio."""
        return self.null_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def empty_percentage(self) -> float:
        """Calculate empty string percentage."""
        return self.empty_string_count / self.total_count if self.total_count > 0 else 0.0


def analyze_column_values(values: list[Any]) -> ColumnValueAnalysis:
    """
    Analyze column values in a single pass, collecting all metrics at once.

    This replaces multiple iterations:
    - NULL counting
    - Empty string counting
    - Non-null value filtering
    - Numeric value extraction
    - Text value extraction
    - Text length calculation
    - Numeric string detection
    - Boolean detection

    Args:
        values: List of column values (may include None)

    Returns:
        ColumnValueAnalysis with all computed metrics
    """
    result = ColumnValueAnalysis()
    result.total_count = len(values)

    # Track if all non-null values are bools (for bool detection)
    all_bools = True
    bool_set: set[bool] = set()

    for val in values:
        if val is None:
            result.null_count += 1
            continue

        result.non_null_values.append(val)

        # Empty string check (for TEXT NOT NULL columns)
        if val == "":
            result.empty_string_count += 1

        # Type-specific collection
        if isinstance(val, bool):
            # Must check bool BEFORE int (bool is subclass of int in Python)
            bool_set.add(val)
        elif isinstance(val, (int, float)):
            all_bools = False
            result.numeric_values.append(float(val))
        elif isinstance(val, str):
            all_bools = False
            result.text_values.append(val)
            result.text_lengths.append(len(val))

            # Check for numeric strings (e.g., "163" stored as TEXT)
            if not result.has_numeric_strings:
                stripped = val.lstrip("-")
                if stripped.replace(".", "").isdigit() and stripped:
                    result.has_numeric_strings = True
        else:
            # Other types (bytes, etc.)
            all_bools = False

    # Only set bool_values if ALL non-null values were actual bools
    if all_bools and bool_set:
        result.bool_values = bool_set

    return result


def detect_fts_table(conn: sqlite3.Connection, table_name: str) -> tuple[bool, str | None]:
    """
    Check if table is FTS3/4 virtual table.

    FTS (Full-Text Search) virtual tables are extremely permissive - they accept
    ANY data type (integers, floats, BLOBs) and transparently convert to TEXT.
    This causes massive false positives during lost_and_found matching.

    Args:
        conn: SQLite connection
        table_name: Table name to check

    Returns:
        Tuple of (is_fts: bool, fts_version: str | None)
        Examples: (True, "FTS4"), (True, "FTS3"), (False, None)
    """
    try:
        cur = conn.cursor()
        sql = cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()

        if sql and sql[0]:
            sql_upper = sql[0].upper()
            if "VIRTUAL TABLE" in sql_upper:
                if "FTS4" in sql_upper:
                    return True, "FTS4"
                if "FTS3" in sql_upper:
                    return True, "FTS3"

        return False, None
    except Exception:
        # If we can't determine, assume not FTS
        return False, None


def can_query_table(conn: sqlite3.Connection, table_name: str) -> tuple[bool, str | None]:
    """
    Test if a table can be successfully queried.

    Some tables (e.g., FTS virtual tables with custom tokenizers) exist in the schema
    but fail when queried because SQLite doesn't have the required module/tokenizer.

    Args:
        conn: SQLite connection
        table_name: Table name to check

    Returns:
        Tuple of (is_queryable: bool, error_message: str | None)
        Examples: (True, None), (False, "unknown tokenizer: ab_cf_tokenizer")
    """
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {quote_ident(table_name)} LIMIT 1")
        cur.fetchone()
        return True, None
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        # Known safe-to-skip errors
        if "no such module" in error_msg or "unknown tokenizer" in error_msg:
            return False, str(e)
        # Unknown error - re-raise
        raise
    except sqlite3.DatabaseError as e:
        return False, str(e)


def sample_table_for_stats(conn: sqlite3.Connection, table: str, limit: int = 10000) -> list[tuple]:
    """
    Sample rows for STATISTICS and ENUM DETECTION.

    Uses RANDOM sampling (not robust sampling) to ensure we capture
    the full distribution of values, including all enum members.

    Args:
        conn: Database connection
        table: Table name
        limit: Maximum rows to sample (default 10,000 for good statistics)

    Returns:
        List of row tuples
    """
    qname = quote_ident(table)
    cur = conn.cursor()
    try:
        # Get row count
        row_count = cur.execute(f"SELECT COUNT(*) FROM {qname}").fetchone()[0]

        if row_count > limit:
            # Use RANDOM() for unbiased sampling
            return cur.execute(f"SELECT * FROM {qname} ORDER BY RANDOM() LIMIT {limit};").fetchall()
        # Small table - take all rows
        return cur.execute(f"SELECT * FROM {qname};").fetchall()
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        # Virtual table with unavailable module or custom tokenizer
        if "no such module" in error_msg or "unknown tokenizer" in error_msg:
            return []
        # UTF-8 decoding errors (corrupted data in database)
        if "could not decode" in error_msg or "utf-8" in error_msg:
            # print(
            #    f"[WARNING] Cannot read table {table} due to encoding error."
            # )
            return []
        raise
    except sqlite3.DatabaseError:
        return []


def infer_type_from_examples(examples: list) -> str | None:
    """
    Infer actual SQLite type from sample data.

    This is crucial for BLOB columns where sqlite3 .recover uses BLOB as a
    wildcard when it can't determine the type, but the actual data may be
    consistently INTEGER, TEXT, etc.

    Args:
        examples: List of sample values from a column

    Returns:
        SQLite type name if 100% consistent, None otherwise

    Rules:
        - Requires 100% consistency among non-NULL values
        - Returns "INTEGER" for all ints
        - Returns "REAL" for any float (even if mixed with ints)
        - Returns "TEXT" if all are text/str
        - Returns "BLOB" if all are bytes
        - Returns None for mixed types or empty samples
    """
    if not examples:
        return None

    # Filter out NULLs - we only care about actual data types
    non_null_examples = [ex for ex in examples if ex is not None]

    if not non_null_examples:
        return None

    # Determine Python types present
    type_counts = {}
    for value in non_null_examples:
        if isinstance(value, bool):
            # In SQLite, bool is stored as INTEGER (0 or 1)
            py_type = "int"
        elif isinstance(value, int):
            py_type = "int"
        elif isinstance(value, float):
            py_type = "float"
        elif isinstance(value, str):
            py_type = "str"
        elif isinstance(value, bytes):
            py_type = "bytes"
        else:
            py_type = "mixed"

        type_counts[py_type] = type_counts.get(py_type, 0) + 1

    # Check for 100% consistency (or int+float which maps to REAL)
    if len(type_counts) == 1:
        # All same type
        py_type = list(type_counts.keys())[0]

        if py_type == "int":
            return "INTEGER"
        if py_type == "float":
            return "REAL"
        if py_type == "str":
            return "TEXT"
        if py_type == "bytes":
            return "BLOB"
        return None

    # Special case: mix of int and float = REAL affinity
    if set(type_counts.keys()) == {"int", "float"}:
        return "REAL"

    # Mixed types - no inference possible
    return None


def infer_role_from_column_name(col_name: str) -> str | None:
    """
    Infer semantic role from column name using heuristics.

    Args:
        col_name: Column name to analyze

    Returns:
        Semantic role ("url", "uuid", "email", "path", "domain") or None
    """
    col_lower = col_name.lower()

    # URL heuristics
    if any(pattern in col_lower for pattern in ["url", "link", "href"]):
        return "url"

    # UUID heuristics
    if any(pattern in col_lower for pattern in ["uuid", "guid", "uniqueid"]):
        return "uuid"

    # Email heuristics
    if any(pattern in col_lower for pattern in ["email", "mail_address", "emailaddress"]):
        return "email"

    # Path heuristics
    if any(pattern in col_lower for pattern in ["path", "filepath", "file_path", "pathname"]):
        return "path"

    # Domain heuristics
    if any(pattern in col_lower for pattern in ["domain", "hostname", "host_name"]):
        return "domain"

    return None


def detect_uuid_role(
    col_values: list,
    col_type: str,
    min_sample_size: int = MIN_ROLE_SAMPLE_SIZE,
) -> bool:
    """
    Detect if column contains UUIDs.

    Requires 100% confidence to avoid rejecting legitimate data during validation.
    If only 80% are UUIDs, we can't safely validate as UUID-only during recovery.

    Args:
        col_values: Sample values from column (uses all available, up to 10,000)
        col_type: Column type (BLOB, TEXT, VARCHAR, etc.)
        min_sample_size: Minimum sample size for role detection (default from config)

    Returns:
        True if 100% of non-NULL samples are UUIDs AND sample size >= min_sample_size
    """
    if col_type not in ("BLOB", "TEXT") and (not col_type or "CHAR" not in col_type):
        return False

    uuid_count = 0
    total_checked = 0

    # Check ALL available values (up to 10,000 from stats sample)
    for val in col_values:
        try:
            # Convert bytes to string if needed
            text = val.decode("utf-8", errors="ignore") if isinstance(val, bytes) else str(val)
            if detect_pattern_type(text) == "uuid":
                uuid_count += 1
            total_checked += 1
        except (AttributeError, UnicodeDecodeError):
            total_checked += 1
            continue

    # Require 100% confidence AND minimum sample size
    # Small samples give meaningless 100% confidence (e.g., 1/1 = 100%)
    # Otherwise we'd reject legitimate data during validation
    return uuid_count == total_checked and total_checked >= min_sample_size


def detect_timestamp_role(
    col_values: list,
    col_type: str,
    is_pk: bool,
    has_pk_in_name: bool,
    min_sample_size: int = MIN_ROLE_SAMPLE_SIZE,
) -> tuple[bool, str | None, bool]:
    """
    Detect if column contains timestamps and identify format.

    Requires 100% confidence to avoid rejecting legitimate data during validation.
    If only 80% are timestamps, we can't safely validate as timestamp-only during recovery.

    Also detects "nullable_timestamp" columns where zeros represent null/absent timestamps.
    This is common in macOS databases (Network Usage, etc.) where 0 means "no timestamp".

    Args:
        col_values: Sample values from column (uses all available, up to 10,000)
        col_type: Column type (INTEGER, REAL, FLOAT)
        is_pk: Is this a primary key column?
        has_pk_in_name: Does column name contain "PK"?
        min_sample_size: Minimum sample size for role detection (default from config)

    Returns:
        (is_timestamp, format_name, is_nullable) tuple where:
        - is_timestamp: True if timestamps detected AND sample size >= min_sample_size
        - format_name: Most common timestamp format, or None if not a timestamp
        - is_nullable: True if column has zeros alongside valid timestamps (0 = null/absent)
    """
    if not col_type or not ("INT" in col_type or "REAL" in col_type or "FLOAT" in col_type):
        return False, None, False

    # Skip primary keys and columns with "PK" in name
    if is_pk or has_pk_in_name:
        return False, None, False

    timestamp_detections = []
    total_checked = 0
    zero_count = 0

    # Check ALL available values (up to 10,000 from stats sample)
    for val in col_values:
        if isinstance(val, (int, float)):
            total_checked += 1
            # Track zeros separately - they represent null/absent timestamps
            if val == 0:
                zero_count += 1
                continue
            # Also skip Apple sentinel values for format detection
            if TimestampFormat.is_apple_timestamp_sentinel(val):
                # Sentinels are valid but don't count toward format detection
                continue
            ts_format = TimestampFormat.detect_timestamp_format(val)
            if ts_format:
                timestamp_detections.append(ts_format)

    if total_checked == 0:
        return False, None, False

    # Calculate non-zero, non-sentinel values that should be timestamps
    non_special_count = total_checked - zero_count
    # Subtract sentinel count (we skipped them in the loop)
    sentinel_count = sum(
        1 for val in col_values if isinstance(val, (int, float)) and TimestampFormat.is_apple_timestamp_sentinel(val)
    )
    non_special_count -= sentinel_count

    # Need at least some non-zero values to detect timestamp format
    if non_special_count == 0:
        return False, None, False

    # Check if all non-zero, non-sentinel values are valid timestamps
    all_timestamps_valid = len(timestamp_detections) == non_special_count

    # ALL non-special values must be valid timestamps (100% confidence required)
    if not all_timestamps_valid:
        return False, None, False

    # Require minimum number of timestamp values (configurable, default 1)
    # This allows users to require more evidence before treating column as timestamp
    if len(timestamp_detections) < min_sample_size:
        return False, None, False

    # Require ALL timestamps to be in the SAME format
    # Mixed formats indicate this is NOT a timestamp column (likely enums/bitfields)
    # Example: Chrome History "transition" has values in both cocoa_seconds and unix_seconds ranges
    format_counts = Counter(timestamp_detections)
    if len(format_counts) > 1:
        # Multiple different timestamp formats detected
        # This is not a real timestamp column
        return False, None, False

    # All timestamps are the same format
    most_common_format = format_counts.most_common(1)[0][0]
    # Handle both enum objects and strings
    format_name = most_common_format.name if hasattr(most_common_format, "name") else str(most_common_format)

    # Determine if this is a nullable timestamp (has zeros representing null)
    is_nullable = zero_count > 0

    return True, format_name, is_nullable


def detect_multi_role_from_patterns(
    col_values: list,
    effective_type: str,
    min_sample_size: int = MIN_ROLE_SAMPLE_SIZE,
) -> list[str] | str | None:
    """
    Detect semantic roles from text patterns with 100% confidence requirement.

    Checks ALL available values (up to 10,000) and returns roles only if 100%
    of non-NULL values match specific patterns. Supports multi-role columns
    where values consistently match multiple patterns (e.g., url+path, email+domain).

    Args:
        col_values: Sample values from column (uses all available, up to 10,000)
        effective_type: Column type (TEXT, VARCHAR, etc.)
        min_sample_size: Minimum sample size for role detection (default from config)

    Returns:
        - List of roles if multiple patterns detected with 100% confidence AND sample size >= min_sample_size
        - Single role string if one pattern detected with 100% confidence AND sample size >= min_sample_size
        - None if no consistent pattern, mixed patterns, or insufficient sample size
    """
    if effective_type not in ("TEXT", "VARCHAR") and (not effective_type or "CHAR" not in effective_type):
        return None

    # Track all patterns detected across ALL values
    pattern_counts: dict[str, int] = {}
    total_checked = 0

    # Check ALL available values (up to 10,000 from stats sample)
    for val in col_values:
        if val is None or val == "":
            continue  # Skip NULL/empty

        total_checked += 1

        # Try enhanced pattern detection first
        pattern = detect_enhanced_pattern_type(val)
        if not pattern:
            # Fallback to basic pattern detection
            pattern = detect_pattern_type(str(val))

        if pattern:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    if total_checked == 0:
        return None

    # Require minimum sample size for meaningful confidence
    # Small samples give meaningless 100% confidence (e.g., 1/1 = 100%)
    if total_checked < min_sample_size:
        return None

    # Find patterns that appear in 100% of values
    # Some patterns naturally co-occur (url is also path, email contains domain)
    perfect_patterns = [p for p, count in pattern_counts.items() if count == total_checked]

    if not perfect_patterns:
        # No 100% pattern match - try threshold-based programming_case detection
        # This allows columns with mix of snake_case/single words to be detected
        # (e.g., ['user_id', 'name', 'created_at'] where 'name' is single word)
        return detect_programming_case_role(col_values)

    # If only one pattern, return it as single role
    if len(perfect_patterns) == 1:
        return perfect_patterns[0]

    # Multiple patterns with 100% confidence - return as multi-role
    # Sort for consistency (alphabetically)
    return sorted(perfect_patterns)


def detect_signature_pattern(numeric_values: list[float], col_type: str) -> tuple[bool, list[int] | None, bool]:
    """
    Detect signature patterns (hashed values with consistent string length).

    Example: ZUUIDHASH with massive stdev but all values are 19-20 chars long.
    Example: expires_utc with 0 (no expiry) or 17-digit webkit values.

    Zeros are treated as null markers, not signature variants. A column with
    values [0, 0, 65000000000000000, 65000000000000000] gets signature_lengths=[17]
    with is_nullable=True, not signature_lengths=[1, 17].

    Args:
        numeric_values: Numeric sample values
        col_type: Column type

    Returns:
        (is_signature, signature_lengths, is_nullable) tuple where:
        - is_signature: True if signature pattern detected
        - signature_lengths: List of string lengths (excluding zeros)
        - is_nullable: True if zeros exist (representing null/absent values)
    """
    if not numeric_values or len(numeric_values) < 5:
        return False, None, False

    is_integer = "INT" in col_type and "REAL" not in col_type and "FLOAT" not in col_type

    if not is_integer:
        return False, None, False

    # Separate zeros (null markers) from actual signature values
    non_zero_values = [v for v in numeric_values if v != 0]
    zero_count = len(numeric_values) - len(non_zero_values)

    # Need at least 5 non-zero values for signature detection
    if len(non_zero_values) < 5:
        return False, None, False

    # Calculate statistics on non-zero values only
    stdev = statistics.stdev(non_zero_values) if len(non_zero_values) >= 10 else 0
    value_range = max(non_zero_values) - min(non_zero_values)

    # Extreme variance indicates hashed or distributed values
    if stdev > 1e15 or value_range > 1e18:
        # Check string length consistency on NON-ZERO values only
        str_lengths = [len(str(int(v))) for v in non_zero_values]
        unique_lengths = set(str_lengths)

        # If all non-zero values have similar string length (within 2 chars)
        if len(unique_lengths) <= 2:
            is_nullable = zero_count > 0
            return True, sorted(unique_lengths), is_nullable

    return False, None, False


def infer_foreign_keys(
    conn: sqlite3.Connection,
    table_name: str,
    table_info: list[tuple],
    all_tables: list[str],
    check_data: bool = True,
) -> dict[str, dict]:
    """
    Infer foreign key relationships using multi-factor heuristics.

    Combines:
    - Index hints (+3): Index name contains both child & parent table
    - Name matching (+2): Column name matches table name pattern
    - Type compatibility (+1): Types are compatible
    - Data validation (+3 if ≥80%, +1 if ≥50%): JOIN check

    Args:
        conn: Database connection
        table_name: Current table name
        table_info: PRAGMA table_info results
        all_tables: List of all table names in database
        check_data: Whether to perform expensive data validation (default: True)

    Returns:
        dict: {column_name: {"table": parent_table, "score": score, "ratio": match_ratio}}
    """
    cur = conn.cursor()
    foreign_keys = {}

    # Get indices for this table
    indices = {}
    try:
        # PRAGMA index_list format: (seq, name, unique, origin, partial)
        for seq, idx_name, unique, *_ in cur.execute(f"PRAGMA index_list({quote_ident(table_name)})"):
            # Use separate cursor for nested query to avoid interrupting iteration
            cur2 = conn.cursor()
            idx_cols = [r[2] for r in cur2.execute(f"PRAGMA index_info({idx_name})")]
            indices[idx_name] = idx_cols
    except sqlite3.Error:
        indices = {}

    # Get candidate keys for each table
    candidate_keys = {}
    for parent_table in all_tables:
        try:
            parent_info = cur.execute(f"PRAGMA table_info({quote_ident(parent_table)})").fetchall()
            # Find PK columns
            pk_cols = [row[1] for row in parent_info if row[5]]  # pk flag
            if pk_cols:
                candidate_keys[parent_table] = pk_cols[0]  # Use first PK
            else:
                candidate_keys[parent_table] = "rowid"  # Fallback
        except sqlite3.Error:
            continue

    # Check each non-PK column
    for cid, col_name, col_type, notnull, dflt, pk in table_info:
        if pk:  # Skip primary keys
            continue
        if col_name.upper() in (
            "Z_PK",
            "Z_ENT",
            "Z_OPT",
        ):  # Skip CoreData metadata
            continue

        best_match = None
        best_score = 0

        for parent_table in all_tables:
            if parent_table == table_name:  # Skip self-references for now
                continue

            parent_pk = candidate_keys.get(parent_table)
            if not parent_pk:
                continue

            score = 0

            # 1. Index hint (+3): Index name contains both tables
            for idx_name, idx_cols in indices.items():
                if col_name in idx_cols and parent_table.lower() in idx_name.lower():
                    score += 3
                    break

            # 2. Name matching (+2)
            col_lower = col_name.lower()
            parent_lower = parent_table.lower()
            if col_lower == parent_lower or col_lower == f"{parent_lower}_id" or col_lower == f"z{parent_lower}":
                score += 2

            # 3. Type compatibility (+1)
            # Simple heuristic: assume compatible for now
            score += 1

            # 4. Data validation (expensive, only if score already promising)
            if check_data and score >= 3:
                try:
                    # Check match ratio
                    query = f"""
                    SELECT
                      SUM(CASE WHEN B.{quote_ident(parent_pk)} IS NOT NULL THEN 1 ELSE 0 END)*1.0 /
                      NULLIF(SUM(CASE WHEN A.{quote_ident(col_name)} IS NOT NULL THEN 1 ELSE 0 END), 0)
                    FROM {quote_ident(table_name)} A
                    LEFT JOIN {quote_ident(parent_table)} B
                      ON A.{quote_ident(col_name)} = B.{quote_ident(parent_pk)}
                    """
                    result = cur.execute(query).fetchone()
                    ratio = result[0] if result and result[0] is not None else 0.0

                    if ratio >= 0.8:
                        score += 3  # High confidence FK
                    elif ratio >= 0.5:
                        score += 1  # Possible FK

                    # Keep best match with data validation
                    if (
                        score >= 4
                        and ratio > 0
                        and (
                            score > best_score
                            or (score == best_score and ratio > (best_match.get("ratio", 0) if best_match else 0))
                        )
                    ):
                        best_match = {
                            "table": parent_table,
                            "score": score,
                            "ratio": round(ratio, 3),
                        }
                        best_score = score

                except sqlite3.Error:
                    pass  # Query failed, skip this candidate
            else:
                # No data validation - use metadata score only
                # Require score >= 4 (e.g., index hint + name match + type = 3+2+1 = 6)
                if score >= 4 and score > best_score:
                    best_match = {
                        "table": parent_table,
                        "score": score,
                        "ratio": None,  # No data validation performed
                    }
                    best_score = score

        if best_match:
            foreign_keys[col_name] = best_match

    return foreign_keys


def generate_table_rubric(
    conn: sqlite3.Connection,
    table_name: str,
    all_tables: list[str] | None = None,
    stats_sample_size: int = 10000,
    infer_fks: bool = True,
    check_fk_data: bool = True,
    min_timestamp_rows: int = 1,
) -> dict[str, Any]:
    """
    Generate a complete rubric for a single table.

    This is the unified rubric generation function used by both:
    - Exemplar scanning (clean databases)
    - Self-match rubrics (recovered databases)

    Args:
        conn: Database connection
        table_name: Name of table to analyze
        all_tables: List of all tables (for FK inference), or None to auto-detect
        stats_sample_size: Number of rows to sample for statistics (default: 10,000)
        infer_fks: Whether to infer foreign keys (default: True)
        check_fk_data: Whether to validate FKs with data (expensive, default: True)
        min_timestamp_rows: Minimum timestamp values to assign role (default: 1)

    Returns:
        Dictionary with rubric metadata for this table
    """
    cur = conn.cursor()

    # Early check: Skip tables that can't be queried (e.g., FTS with custom tokenizers)
    queryable, query_error = can_query_table(conn, table_name)
    if not queryable:
        return {
            "columns": [],
            "notes": [f"Skipped: {query_error}"],
        }

    # Detect if this is an FTS3/4 virtual table
    # FTS tables accept ANY data type and convert to TEXT, causing false positives
    is_fts_table, fts_version = detect_fts_table(conn, table_name)

    # Get table info (may fail for virtual tables with unavailable modules)
    try:
        table_info = cur.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    except sqlite3.OperationalError as e:
        # Virtual table with unavailable module - skip it
        if "no such module" in str(e).lower():
            return {
                "columns": [],
                "notes": [f"Skipped: Virtual table with unavailable module ({e})"],
            }
        raise

    if not table_info:
        return {"columns": []}

    # Get all tables if not provided (needed for FK inference)
    if all_tables is None and infer_fks:
        all_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # Sample rows for statistics (random sampling for unbiased distribution)
    stats_rows = sample_table_for_stats(conn, table_name, limit=stats_sample_size)

    # Analyze each column
    columns = []

    # Determine PK type for table-level metadata
    # This helps distinguish INTEGER PRIMARY KEY (rowid alias) from composite PKs
    pk_cols = [(row[1], row[2].upper() if row[2] else "") for row in table_info if row[5] > 0]
    # pk_cols is list of (name, type) for PK columns

    if len(pk_cols) == 1 and pk_cols[0][1] in ("INTEGER", "INT"):
        pk_type = "integer_pk"  # Single INTEGER PRIMARY KEY (rowid alias)
    elif len(pk_cols) > 1:
        pk_type = "composite_pk"  # Multiple columns in PK
    elif len(pk_cols) == 0:
        pk_type = "no_pk"  # No explicit PK (uses implicit rowid)
    else:
        pk_type = "unknown"

    for cid, col_name, col_type, notnull, dflt, pk in table_info:
        col_meta = {
            "name": col_name,
            "type": col_type,
            "primary_key": bool(pk),
            "not_null": bool(notnull),
            "is_fts": is_fts_table,  # Flag FTS columns for strict validation
        }

        # Get ALL values for this column from stats sample
        # (including NULLs for likelihood calculation)
        all_col_values = [row[cid] for row in stats_rows if cid < len(row)]

        # Single-pass analysis of column values (optimization: replaces 5-7 iterations)
        analysis = analyze_column_values(all_col_values)

        # Store NULL likelihood if there are some NULLs
        if analysis.null_likelihood > 0:
            col_meta["null_likelihood"] = analysis.null_likelihood

        # --- Empty String Detection (for TEXT NOT NULL columns) ---
        # Some TEXT columns are NOT NULL but contain many empty strings ('')
        # Example: Chrome Cookies 'value' column is NOT NULL but 95% empty strings
        # We need to track this separately from NULL likelihood for accurate validation
        # Only store if empty strings are present (threshold: 1% or more)
        if (
            bool(notnull)
            and col_type
            and ("TEXT" in col_type.upper() or "CHAR" in col_type.upper() or col_type.upper() == "")
            and analysis.empty_percentage >= 0.01
        ):
            col_meta["can_be_empty"] = True
            col_meta["empty_percentage"] = analysis.empty_percentage

        # Use pre-computed non-NULL values for further analysis
        col_values = analysis.non_null_values

        if not col_values:
            # No data - use column name heuristic for timestamp detection
            # This helps validate L&F data when exemplar table is empty
            col_name_lower = col_name.lower()
            if "timestamp" in col_name_lower and col_type in (
                "REAL",
                "INTEGER",
                "INT",
            ):
                col_meta["role"] = "timestamp"
                col_meta["timestamp_format"] = "unknown"  # Accept any valid format

            columns.append(col_meta)
            continue

        # --- Type Inference ---
        # Infer actual type from data (crucial for BLOB columns that may contain
        # TEXT, INTEGER, etc. due to SQLite's dynamic typing or recovery tools)
        inferred_type = infer_type_from_examples(col_values)

        if inferred_type and inferred_type != col_type:
            # Type mismatch detected - store both types
            col_meta["declared_type"] = col_type
            col_meta["inferred_type"] = inferred_type

            # Use inferred type for all subsequent analysis
            effective_type = inferred_type
        else:
            # Types match or couldn't infer - use declared type
            effective_type = col_type

        # --- Semantic Role Detection ---
        # Priority order:
        # 1. Timestamp detection (most specific, numeric-based)
        # 2. UUID detection (high confidence pattern)
        # 3. Column name heuristics
        # 4. Pattern inference from examples (for URL, email, path, domain)

        # Timestamp format detection (check first, as timestamps are numeric)
        is_pk = bool(pk)
        has_pk_in_name = "PK" in col_name.upper()
        is_timestamp, timestamp_format, is_nullable = detect_timestamp_role(
            col_values, effective_type, is_pk, has_pk_in_name, min_timestamp_rows
        )
        if is_timestamp:
            # Use nullable_timestamp role if column has zeros (0 = null/absent timestamp)
            col_meta["role"] = "nullable_timestamp" if is_nullable else "timestamp"
            col_meta["timestamp_format"] = timestamp_format
        # UUID detection (use effective_type for accurate detection)
        elif detect_uuid_role(col_values, effective_type):
            col_meta["role"] = "uuid"
        # Pattern detection from actual data (requires 100% confidence)
        # NOTE: We do NOT use column name heuristics to avoid false positives.
        # Example: ZABSOLUTETRIGGER has type=TIMESTAMP but contains enum data (-781139055)
        else:
            detected_roles = detect_multi_role_from_patterns(col_values, effective_type)
            if detected_roles:
                col_meta["role"] = detected_roles

        # --- Numeric Statistics ---
        # Note: We intentionally do NOT skip primary keys here anymore.
        # Composite PKs (like message_id in PRIMARY KEY(conversation_id, message_id))
        # benefit from range validation. Autoincrement PKs will get harmless stats.

        if (
            not has_pk_in_name  # Only skip if column name contains "PK"
            and effective_type
            and ("INT" in effective_type or "REAL" in effective_type or "FLOAT" in effective_type)
        ):
            # Use pre-computed numeric values from single-pass analysis
            numeric_values = analysis.numeric_values

            if len(numeric_values) >= 2:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                mean_val = statistics.mean(numeric_values)
                value_range = max_val - min_val
                unique_count = len(set(numeric_values))

                stats = {
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "range": value_range,
                    "unique_count": unique_count,
                    "count": len(numeric_values),
                }

                # Calculate stdev
                if len(numeric_values) >= 10:
                    with contextlib.suppress(statistics.StatisticsError):
                        stats["stdev"] = statistics.stdev(numeric_values)

                # Enum detection (use effective_type for accuracy)
                is_integer = "INT" in effective_type and "REAL" not in effective_type and "FLOAT" not in effective_type
                if is_integer and value_range <= 20 and unique_count <= 20 and len(numeric_values) >= 5:
                    stats["enum_detected"] = True
                    stats["enum_values"] = sorted(set(numeric_values))
                    col_meta["role"] = "enum"
                    col_meta["enum_values"] = stats["enum_values"]

                # Signature pattern detection (use effective_type for accuracy)
                # Don't overwrite timestamp roles - timestamps with zeros can look
                # like signatures (e.g., expires_utc with 0 = no expiry)
                is_signature, signature_lengths, sig_nullable = detect_signature_pattern(numeric_values, effective_type)
                existing_role = col_meta.get("role")
                if is_signature and existing_role not in (
                    "timestamp",
                    "nullable_timestamp",
                ):
                    stats["signature_pattern"] = True
                    stats["signature_lengths"] = signature_lengths
                    # Use nullable_signature for columns with zeros (null markers)
                    if sig_nullable:
                        col_meta["role"] = "nullable_signature"
                        col_meta["signature_nullable"] = True
                    else:
                        col_meta["role"] = "signature"
                    col_meta["signature_lengths"] = signature_lengths

                col_meta["stats"] = stats

        # --- String Length Statistics (for TEXT columns) ---
        # Used to reject suspiciously short values in long-string columns
        # (e.g., reject "admin" in metadata columns with avg 100 chars)
        # Use pre-computed text values and lengths from single-pass analysis
        if (
            effective_type in ("TEXT", "VARCHAR") or (effective_type and "CHAR" in effective_type.upper())
        ) and analysis.text_values:
            col_meta["string_stats"] = {
                "min_length": min(analysis.text_lengths),
                "max_length": max(analysis.text_lengths),
                "avg_length": sum(analysis.text_lengths) / len(analysis.text_lengths),
            }

            # Assign text_only role for TEXT columns with meaningful string data
            # This rejects integers/floats that get coerced into TEXT columns
            # BUT: Skip if exemplar contains purely numeric strings (e.g., "163" is valid)
            # Use pre-computed has_numeric_strings from single-pass analysis
            if not col_meta.get("role") and effective_type == "TEXT" and not analysis.has_numeric_strings:
                col_meta["role"] = "text_only"

        # --- Boolean Detection (True/False only, not 0/1) ---
        # Detect columns with only True/False values (literal booleans)
        # Note: Don't use 0/1 as that could be any limited dataset
        # IMPORTANT: Check actual Python type, not just value (0==False, 1==True in Python)
        # Use pre-computed bool_values from single-pass analysis
        if effective_type in ("INTEGER", "REAL", "TEXT") and analysis.bool_values is not None:
            col_meta["role"] = "bool"
            col_meta["bool_values"] = sorted(analysis.bool_values)

        columns.append(col_meta)

    # --- Foreign Key Inference ---

    if infer_fks and all_tables:
        foreign_keys = infer_foreign_keys(conn, table_name, table_info, all_tables, check_data=check_fk_data)

        # Add FK information to column metadata
        for col in columns:
            col_name = col["name"]
            if col_name in foreign_keys:
                fk_info = foreign_keys[col_name]
                col["foreign_key"] = True
                col["references_table"] = fk_info["table"]
                col["fk_confidence"] = {
                    "score": fk_info["score"],
                    "ratio": fk_info.get("ratio", 0.0),
                }

    return {
        "columns": columns,
        "pk_type": pk_type,
        "pk_columns": [col[0] for col in pk_cols],
    }
