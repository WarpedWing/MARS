#!/usr/bin/env python3
"""
Output Writers for SQLite Carver
by WarpedWing Labs

Functions for writing carved data to SQLite databases, CSV, and JSONL formats.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def analyze_text_quality(text: str) -> dict[str, Any]:
    """
    Analyze text quality for carved strings.

    Classifies text based on presence of Unicode replacement characters (�)
    which indicate corrupted or binary data that couldn't be decoded.

    Args:
        text: Text string to analyze

    Returns:
        Dictionary with:
        - quality: 'good' (no �), 'mixed' (some �), 'poor' (mostly �)
        - cleaned: Text with � characters removed
        - stats: Character counts
    """
    if not text:
        return {
            "quality": "good",
            "cleaned": "",
            "stats": {"printable": 0, "replacement": 0, "total": 0},
        }

    replacement_char = "\ufffd"  # �
    blank = "\u0020"
    total_chars = len(text)
    replacement_count = text.count(replacement_char)

    # Create cleaned version (� removed)
    cleaned = text.replace(replacement_char, blank).strip()

    # Quality classification
    if replacement_count == 0:
        quality = "good"
    elif replacement_count / total_chars > 0.5:
        # More than 50% replacement chars = garbage
        quality = "poor"
    elif len(cleaned) < 10:
        # After cleaning, very little useful text remains = garbage
        quality = "poor"
    else:
        # Has some replacement chars but also useful content
        quality = "mixed"

    return {
        "quality": quality,
        "cleaned": cleaned,
        "stats": {
            "printable": len(cleaned),
            "replacement": replacement_count,
            "total": total_chars,
        },
    }


def open_out_db(path: Path) -> sqlite3.Connection:
    """
    Create and initialize output database for carved artifacts.

    Creates two tables:
    - carved_all: All carved artifacts (timestamps, URLs, text, blobs)
    - carved_protobufs: Decoded protobuf messages

    Args:
        path: Path to output database

    Returns:
        Open SQLite connection
    """
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE carved_all (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_no INTEGER,
            page_offset INTEGER,
            abs_offset INTEGER,
            cluster_id INTEGER,
            kind TEXT,                -- 'url' | 'text' | 'blob'
            value_text TEXT,          -- Original text (never modified)
            value_text_clean TEXT,    -- Cleaned text (� replacement chars removed)
            text_quality TEXT,        -- For text: 'good' | 'mixed' | 'poor' | NULL (for non-text)
            timestamp_count INTEGER,  -- Number of timestamps found in value_text
            timestamp_fields TEXT     -- JSON array of timestamp objects (value, format, human_readable, classification)
        );
    """
    )
    c.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_all
        ON carved_all(kind, value_text, page_no);
    """
    )
    c.execute(
        """
        CREATE TABLE carved_protobufs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_abs_offset INTEGER,
            page_no INTEGER,
            abs_offset INTEGER,
            json_pretty TEXT,               -- Full decoded message as JSON (search this for strings/numbers)
            schema TEXT,                    -- Inferred field types (simplified)
            field_count INTEGER,            -- Number of top-level fields
            timestamp_count INTEGER,        -- Number of timestamps found
            timestamp_fields TEXT,          -- JSON array of timestamp field names and values
            decoder_used TEXT               -- Which decoder was used (blackboxprotobuf, legacy, etc.)
        );
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_protobufs_page
        ON carved_protobufs(page_no);
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_protobufs_offset
        ON carved_protobufs(abs_offset);
    """
    )
    conn.commit()
    c.execute("PRAGMA journal_mode=DELETE;")
    c.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()
    return conn


def write_csv(path: Path, header: list[str], rows: list[list[Any]]):
    """
    Write CSV file with header and rows.

    Args:
        path: Path to CSV file
        header: Column names
        rows: Data rows
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def append_csv_batch(path: Path, rows: list[list[Any]]):
    """
    Append rows to existing CSV file (no header).

    Args:
        path: Path to CSV file
        rows: Data rows to append
    """
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def append_jsonl(path: Path, obj: dict[str, Any]):
    """
    Append JSON object to JSONL (JSON Lines) file.

    Args:
        path: Path to JSONL file
        obj: Dictionary to append as JSON line
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
