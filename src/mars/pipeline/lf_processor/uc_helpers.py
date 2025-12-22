#!/usr/bin/env python3
"""
Lost & Found Use Case Helpers

Shared utility functions for LF database reconstruction across all use cases.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from mars.utils.database_utils import readonly_connection

if TYPE_CHECKING:
    from pathlib import Path


def is_fts_table(table_name: str, table_schemas: dict) -> bool:
    """
    Check if table is an FTS virtual table, auxiliary table, or SQLite internal table.

    Args:
        table_name: Name of the table to check
        table_schemas: Dict mapping table names to schema info

    Returns:
        True if table is FTS-related or SQLite internal, False otherwise
    """
    # Skip SQLite internal tables
    if table_name in ["sqlite_sequence", "sqlite_stat1", "sqlite_stat4"]:
        return True

    # Skip FTS auxiliary tables
    fts_suffixes = ["_content", "_segdir", "_segments", "_docsize", "_stat"]
    if any(table_name.endswith(suffix) for suffix in fts_suffixes):
        return True

    # Skip FTS virtual tables
    schema = table_schemas.get(table_name, {})
    create_sql = schema.get("create_sql", "")
    if create_sql:  # Only search if create_sql is not None/empty
        return bool(re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE))

    return False


def determine_match_label(exact_matches: list, metamatch: dict) -> str | None:
    """
    Extract match label from exact_matches or metamatch metadata.

    Prioritizes strong matches (tables_equal+columns_equal, hash) over weak matches.

    Args:
        exact_matches: List of exact match dicts with match type and label
        metamatch: Metamatch dict with group_label

    Returns:
        Match label string or None if no label found
    """
    match_label = None

    # Try exact matches first (prioritize strong matches)
    if exact_matches:
        for match in exact_matches:
            match_type = match.get("match", "")
            if match_type in ["tables_equal+columns_equal", "hash"]:
                match_label = match.get("label")
                break
        # Fallback to first match if no strong match found
        if not match_label and exact_matches:
            match_label = exact_matches[0].get("label")

    # Fall back to metamatch label
    if not match_label and metamatch:
        match_label = metamatch.get("group_label")

    return match_label


def sanitize_filename(label: str) -> str:
    """
    Sanitize a label for use in filenames.

    Args:
        label: Label to sanitize

    Returns:
        Sanitized label safe for filesystem use
    """
    return label.replace(" ", "_").replace("/", "_")


def get_non_lf_tables(split_db: Path) -> list[str]:
    """
    Get list of non-LF tables (regular tables, not lf_table_*).

    Args:
        split_db: Path to split database

    Returns:
        List of table names that are not LF tables
    """
    with readonly_connection(split_db) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'lf_table_%'")
        tables = [row[0] for row in cursor.fetchall()]
    return tables


def create_shortened_name_hash(table_name: str, max_length: int = 20) -> str:
    """
    Create a shortened table name with hash for folder naming.

    Args:
        table_name: Original table name
        max_length: Maximum length for shortened name

    Returns:
        String like "shortened_tablename_abc123" for use in folder names
    """
    # Shorten table name
    shortened = table_name[:max_length] if len(table_name) > max_length else table_name

    # Generate short hash (first 6 chars of MD5)
    hash_val = hashlib.md5(table_name.encode()).hexdigest()[:6]

    return f"{shortened}_{hash_val}"
