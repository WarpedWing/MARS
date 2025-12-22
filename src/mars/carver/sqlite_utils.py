#!/usr/bin/env python3
"""
SQLite Utilities for Carver
by WarpedWing Labs

Helper functions for reading SQLite database headers and schema information.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def read_sqlite_header_info(fp: Path) -> tuple[int, str, int]:
    """
    Extract database header information from SQLite file.

    Reads the first 100 bytes to extract:
    - Page size (in bytes)
    - Text encoding (UTF-8, UTF-16le, UTF-16be)
    - Number of pages

    Args:
        fp: Path to SQLite database file

    Returns:
        Tuple of (page_size, encoding, num_pages)

    Raises:
        ValueError: If not a valid SQLite database
    """
    with fp.open("rb") as f:
        data = f.read(100)
    if len(data) < 100 or data[:16] != b"SQLite format 3\0":
        raise ValueError("Not a valid SQLite header")
    page_size = int.from_bytes(data[16:18], "big") or 4096
    if page_size == 1:  # 65536 sentinel
        page_size = 65536
    enc_code = int.from_bytes(data[56:60], "big")
    encoding = {1: "UTF-8", 2: "UTF-16le", 3: "UTF-16be"}.get(enc_code, "UTF-8")
    fsize = fp.stat().st_size
    pages = fsize // page_size if page_size else 0
    return page_size, encoding, int(pages)


def check_schema_has_blobs(fp: Path) -> bool:
    """
    Check if the database schema contains any BLOB columns.

    This helps avoid false positive protobuf detection in databases
    that never stored binary data.

    Args:
        fp: Path to SQLite database file

    Returns:
        True if any table has a BLOB column, False otherwise
    """
    try:
        with sqlite3.connect(str(fp)) as conn:
            cursor = conn.cursor()

            # Get all table schemas
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")

            for (sql,) in cursor.fetchall():
                if sql:
                    # Case-insensitive search for BLOB column type
                    sql_upper = sql.upper()
                    if " BLOB" in sql_upper or "\tBLOB" in sql_upper or "(BLOB" in sql_upper:
                        return True

            return False

    except Exception:
        # If we can't read the schema (corrupted DB, etc.), assume no blobs
        return False
