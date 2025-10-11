#!/usr/bin/env python3
"""
Generate a Powerlog schema CSV and a portable rubric JSON from a SQLite DB.

Outputs:
  Schemas/<NAME>.powerlog.schema.csv            # Table,Column,Type
  Rubrics/<NAME>.powerlog.rubric.json           # type/roles/epoch rules/hints

Rules baked in:
- Columns whose names contain 'timestamp' or 'date' (case-insensitive) are Unix epoch.
  min = 2010-01-01 (1262304000), max = 2100-01-01 (4102444800).
- Columns named exactly 'ID' are treated as ID/PK-like (no fake stats).
- Empty tables carry a "no sample data" note; no fake numeric ranges.
- Known soft hints (not constraints) e.g. Battery Level in [0, 100].

Usage:
  python generate_powerlog_schema_and_rubric.py \
      --db /path/to/Powerlog.PLSQL \
      --name MyMacMini_2025_10_06
"""
import argparse
import csv
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# -------- Constants --------
EPOCH_MIN_2010 = 1262304000  # 2010-01-01 00:00:00 UTC
EPOCH_MAX_2100 = 4102444800  # 2100-01-01 00:00:00 UTC

TIMESTAMP_NAME_RE = re.compile(r"(timestamp|date)", re.IGNORECASE)

# Optional, non-binding hints
# (We only add these when column name & table match.)
SOFT_HINTS = {
    # table_prefix : { column_name : {"range":[min,max]} }
    "PLBatteryAgent": {"Level": {"range": [0, 100]}}
}


# -------- Helpers --------
def is_epoch_column(col_name: str) -> bool:
    return bool(TIMESTAMP_NAME_RE.search(col_name or ""))


def soft_hints_for(table: str, col: str):
    for prefix, colmap in SOFT_HINTS.items():
        if table.startswith(prefix) and col in colmap:
            return colmap[col]
    return None


def safe_type(sqlite_affinity: str | None) -> str:
    # Keep exact string (including blank for sqlite_sequence's 'seq')
    return sqlite_affinity or ""


def fetch_tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Returns list of (name, sql) for tables in sqlite_master, including internal ones."""
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    return cur.fetchall()


def fetch_columns(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """
    Returns PRAGMA table_info(table) rows:
      cid, name, type, notnull, dflt_value, pk
    """
    cur = conn.cursor()
    qname = '"' + table.replace('"', '""') + '"'
    return cur.execute(f"PRAGMA table_info({qname});").fetchall()


def sample_table_info(conn: sqlite3.Connection, table: str, limit: int = 200):
    """Sample up to 'limit' rows for examples only (never constraints)."""
    qname = '"' + table.replace('"', '""') + '"'
    cur = conn.cursor()
    try:
        return cur.execute(f"SELECT * FROM {qname} LIMIT {limit};").fetchall()
    except sqlite3.DatabaseError:
        return []


def infer_examples(rows, col_index: int, max_examples: int = 5):
    """Grab up to a few non-null examples for rubric doc."""
    ex = []
    for r in rows:
        if col_index < len(r):
            v = r[col_index]
            if v is not None:
                ex.append(v)
                if len(ex) >= max_examples:
                    break
    return ex


# -------- Main --------
def main():
    ap = argparse.ArgumentParser(
        description="Generate Powerlog schema CSV and rubric JSON."
    )
    ap.add_argument(
        "--db", required=True, help="Path to Powerlog SQLite DB (.PLSQL/.sqlite)"
    )
    ap.add_argument(
        "--name",
        help="Base name for outputs (no extension). If omitted, derived from DB filename.",
    )
    ap.add_argument(
        "--schemas-dir", default="Schemas", help="Directory to write schema CSV"
    )
    ap.add_argument(
        "--rubrics-dir", default="Rubrics", help="Directory to write rubric JSON"
    )
    ap.add_argument(
        "--force", action="store_true", help="Overwrite existing outputs if present"
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"✗ DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Use a filesystem-friendly UTC timestamp when --name is omitted
    base_name = (
        args.name or f"{db_path.stem}.{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )
    schemas_dir = Path(args.schemas_dir)
    rubrics_dir = Path(args.rubrics_dir)
    schemas_dir.mkdir(parents=True, exist_ok=True)
    rubrics_dir.mkdir(parents=True, exist_ok=True)

    schema_csv_path = schemas_dir / f"{base_name}.powerlog.schema.csv"
    rubric_json_path = rubrics_dir / f"{base_name}.powerlog.rubric.json"

    if not args.force:
        for p in (schema_csv_path, rubric_json_path):
            if p.exists():
                print(
                    f"✗ Output exists (use --force to overwrite): {p}", file=sys.stderr
                )
                sys.exit(2)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        tables = fetch_tables(conn)

        # 1) Schema CSV
        with schema_csv_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["Table", "Column", "Type"])
            for tname, _sql in tables:
                cols = fetch_columns(conn, tname)
                if not cols:
                    continue
                for _cid, col_name, col_type, _notnull, _dflt, _pk in cols:
                    writer.writerow([tname, col_name, safe_type(col_type)])

        # 2) Rubric JSON
        rubric = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "source_db": str(db_path.resolve()),
            "epoch_bounds": {"min": EPOCH_MIN_2010, "max": EPOCH_MAX_2100},
            "notes": [
                "This rubric is portable across hosts; ranges are hints, not constraints.",
                "Any column with 'timestamp' or 'date' in its name is Unix epoch.",
                "ID columns are treated as identifiers (no statistical ranges).",
            ],
            "tables": {},
        }

        for tname, _sql in tables:
            cols = fetch_columns(conn, tname)
            if not cols:
                continue

            # examples only (never constraints)
            sample_rows = sample_table_info(conn, tname, limit=200)
            sample_count = len(sample_rows)
            non_null_counts = [0] * len(cols)
            if sample_count:
                for row in sample_rows:
                    for idx in range(min(len(row), len(cols))):
                        if row[idx] is not None:
                            non_null_counts[idx] += 1

            columns_list = []
            types_list = []
            columns_dict = {}

            for idx, (cid, col_name, col_type, notnull, dflt, pk) in enumerate(cols):
                columns_list.append(col_name)
                types_list.append(safe_type(col_type) or "TEXT")

                declared_notnull = bool(notnull)
                observed_non_null = non_null_counts[idx] if sample_count else 0
                observed_fill_ratio = (
                    observed_non_null / sample_count if sample_count else None
                )
                observed_notnull = bool(
                    sample_count and observed_non_null == sample_count
                )
                effective_notnull = declared_notnull or observed_notnull

                cinfo = {
                    "type": safe_type(col_type),
                    "notnull": effective_notnull,
                    "declared_notnull": declared_notnull,
                    "observed_non_null": observed_non_null if sample_count else 0,
                    "observed_total": sample_count,
                    "primary_key": bool(pk),
                }
                if observed_fill_ratio is not None:
                    cinfo["observed_fill_ratio"] = round(observed_fill_ratio, 5)

                if col_name == "ID":
                    cinfo["role"] = "id"

                if is_epoch_column(col_name):
                    cinfo["role"] = "epoch_timestamp"
                    cinfo["epoch_min"] = EPOCH_MIN_2010
                    cinfo["epoch_max"] = EPOCH_MAX_2100

                hint = soft_hints_for(tname, col_name)
                if hint:
                    cinfo["hints"] = hint

                examples = infer_examples(sample_rows, idx)
                if examples:
                    clean = []
                    for v in examples:
                        if isinstance(v, bytes):
                            try:
                                clean.append(v.decode("utf-8", errors="replace"))
                            except Exception:
                                clean.append(repr(v))
                        else:
                            clean.append(v)
                    cinfo["examples"] = clean
                else:
                    cinfo["note"] = cinfo.get("note", "no sample data")

                columns_dict[col_name] = cinfo

            rubric["tables"][tname] = {
                "columns": columns_dict,  # per-column metadata dict
                "column_list": columns_list,  # flat list for consumers
                "types": types_list,  # flat types aligned with column_list
            }

        with rubric_json_path.open("w", encoding="utf-8") as jf:
            json.dump(rubric, jf, indent=2, ensure_ascii=False)

        print(f"✅ Schema CSV:  {schema_csv_path}")
        print(f"✅ Rubric JSON: {rubric_json_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
