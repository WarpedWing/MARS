#!/usr/bin/env python3
"""
Generate a schema CSV and portable rubric JSON from any SQLite database.

This tool is used by the exemplar scanner to generate rubrics from clean
databases on intact systems. The rubrics are then used for database recovery
and classification.

Outputs:
  Schemas/<NAME>.schema.csv            # Table,Column,Type
  Rubrics/<NAME>.rubric.json           # Complete rubric with metadata

Features:
- Database-agnostic: Works with ANY SQLite database type
- Analyzes actual sample data to detect timestamp formats
- Detects semantic patterns: URLs, UUIDs, emails, file paths, domains
- Tracks NULL likelihood for confidence scoring during matching
- Foreign key inference using multi-factor heuristics
- Statistical analysis for numeric columns (min/max/mean/stdev)
- Enum detection for status/flag columns

Usage:
  python generate_sqlite_schema_rubric.py \
      --db /path/to/database.sqlite \
      --name MyDatabase_2025_10_29
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger


# -------- Helpers --------
def fetch_tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Returns list of (name, sql) for tables in sqlite_master, including internal ones."""
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    return cur.fetchall()


def fetch_columns(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """
    Returns PRAGMA table_info(table) rows:
      cid, name, type, notnull, dflt_value, pk

    Returns empty list for virtual tables with missing modules.
    """
    cur = conn.cursor()
    qname = '"' + table.replace('"', '""') + '"'
    try:
        return cur.execute(f"PRAGMA table_info({qname});").fetchall()
    except Exception:
        # Virtual tables with missing modules (e.g., echo_document_map) will fail
        # Return empty list so they're skipped gracefully
        return []


def safe_type(sqlite_affinity: str | None) -> str:
    """Keep exact string (including blank for sqlite_sequence's 'seq')."""
    return sqlite_affinity or ""


# Note: detect_fts_table is now imported from rubric_generator to avoid duplication


def dump_schema_csv(conn: sqlite3.Connection, tables: list[tuple], output_path: Path):
    """
    Write schema CSV file for a database.

    Args:
        conn: SQLite connection
        tables: List of (table_name, sql) tuples from fetch_tables()
        output_path: Path to write CSV file
    """
    with output_path.open("w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Table", "Column", "Type"])
        for tname, _sql in tables:
            cols = fetch_columns(conn, tname)
            if not cols:
                continue
            for _cid, col_name, col_type, _notnull, _dflt, _pk in cols:
                writer.writerow([tname, col_name, safe_type(col_type)])


def generate_rubric(
    conn: sqlite3.Connection,
    tables: list[tuple],
    rubric_name: str = "database",
    min_timestamp_rows: int = 1,
) -> dict:
    """
    Generate rubric dictionary for a database.

    Uses unified rubric generator for consistency with self-match rubrics.

    Args:
        conn: SQLite connection
        tables: List of (table_name, sql) tuples from fetch_tables()
        rubric_name: Name to use in metadata
        min_timestamp_rows: Minimum timestamp values to assign role (default: 1)

    Returns:
        Rubric dictionary ready to be serialized to JSON
    """
    from mars.pipeline.matcher.rubric_generator import generate_table_rubric

    rubric = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rubric_name": rubric_name,
        "notes": [
            "This rubric is portable across hosts; ranges are hints, not constraints.",
            "Timestamp formats are detected from actual sample data, not just column names.",
            "Supports: Unix (sec/ms/µs/ns), Cocoa (sec/ns since 2001), WebKit (µs since 1601), hex timestamps.",
            "Semantic pattern detection: Columns with consistent patterns marked with role (uuid, url, email, path, domain).",
            "NULL likelihood tracking: Stores percentage of NULL values per column for confidence scoring during matching.",
            "Signature pattern detection: Hashed values with extreme variance and consistent string length.",
            "Example confidence scoring: Values weighted by sample size, frequency, and data type.",
            "ID columns and PK-named columns are excluded from statistical ranges.",
            "Statistical ranges (min/max/mean/range/unique_count/stdev) calculated for numeric columns using up to 10K sample rows.",
            "Enum/status fields auto-detected: INTEGER columns with ≤20 unique values and range ≤20 marked with role='enum'.",
        ],
        "tables": {},
    }

    # Get all table names for FK inference
    all_table_names = [tname for tname, _ in tables]

    skipped_tables = []
    for tname, _sql in tables:
        # Use unified rubric generator (consolidates all improvements)
        table_rubric = generate_table_rubric(
            conn=conn,
            table_name=tname,
            all_tables=all_table_names,
            stats_sample_size=10000,
            infer_fks=True,
            check_fk_data=True,  # Exemplar DBs are clean, can afford data validation
            min_timestamp_rows=min_timestamp_rows,
        )

        # Check if table was skipped (virtual table with unavailable module)
        if "notes" in table_rubric and any("Skipped:" in note for note in table_rubric["notes"]):
            skipped_tables.append((tname, table_rubric["notes"][0]))

        # Store in unified format
        # Unified: {"columns": [{name, type, ...}, ...]}
        # The rubric matcher handles both legacy and unified formats
        rubric["tables"][tname] = table_rubric

    # Add metadata about skipped tables
    if skipped_tables:
        rubric["skipped_tables"] = [{"table": t, "reason": r} for t, r in skipped_tables]

    return rubric


# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Generate schema CSV and rubric JSON from any SQLite database.")
    ap.add_argument("--db", required=True, help="Path to SQLite database (.sqlite/.db)")
    ap.add_argument(
        "--name",
        help="Base name for outputs (no extension). If omitted, derived from DB filename.",
    )
    ap.add_argument("--schemas-dir", default="Schemas", help="Directory to write schema CSV")
    ap.add_argument("--rubrics-dir", default="Rubrics", help="Directory to write rubric JSON")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs if present")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        logger.error(f"✗ DB not found: {db_path}")
        sys.exit(1)

    # Use a filesystem-friendly UTC timestamp when --name is omitted
    base_name = args.name or f"{db_path.stem}.{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    schemas_dir = Path(args.schemas_dir)
    rubrics_dir = Path(args.rubrics_dir)
    schemas_dir.mkdir(parents=True, exist_ok=True)
    rubrics_dir.mkdir(parents=True, exist_ok=True)

    schema_csv_path = schemas_dir / f"{base_name}.schema.csv"
    rubric_json_path = rubrics_dir / f"{base_name}.rubric.json"

    if not args.force:
        for p in (schema_csv_path, rubric_json_path):
            if p.exists():
                logger.error(f"✗ Output exists (use --force to overwrite): {p}")
                sys.exit(2)

    with readonly_connection(db_path) as conn:
        tables = fetch_tables(conn)

        # 1) Generate schema CSV using reusable function
        dump_schema_csv(conn, tables, schema_csv_path)

        # 2) Generate rubric JSON using reusable function
        rubric = generate_rubric(conn, tables, rubric_name=base_name)
        # Add source_db to metadata
        rubric["source_db"] = str(db_path.resolve())

        with rubric_json_path.open("w", encoding="utf-8") as jf:
            json.dump(rubric, jf, indent=2, ensure_ascii=False)

        logger.info(f" Schema CSV:  {schema_csv_path}")
        logger.info(f" Rubric JSON: {rubric_json_path}")


if __name__ == "__main__":
    main()
