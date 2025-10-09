#!/usr/bin/env python3
"""
schema_manager.py — Dynamic Powerlog schema loader

Reads schema CSVs from ./Schemas/ to drive Powerlog classification/merging.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import defaultdict

SCHEMA_DIR = Path(__file__).parent / "Schemas"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "schema"


def _load_schema_file(path: Path) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            table = row["Table"].strip()
            col = row["Column"].strip()
            ctype = row.get("Type", "TEXT").strip()
            if table and col:
                tables[table][col] = ctype
    return {table: dict(cols) for table, cols in tables.items()}


def load_schema_catalog() -> dict[str, dict]:
    """
    Load each schema CSV into a catalog keyed by slug:
    {
        slug: {
            "label": original stem,
            "path": Path,
            "tables": {table: {column: type}},
            "tables_lower": {lowercased table names},
        }
    }
    """
    catalog: dict[str, dict] = {}
    for f in sorted(SCHEMA_DIR.glob("*.schema.csv")):
        try:
            tables = _load_schema_file(f)
        except Exception as e:
            print(f"⚠️  Could not load schema file {f.name}: {e}")
            continue
        slug = _slugify(f.stem)
        catalog[slug] = {
            "label": f.stem,
            "path": f,
            "tables": tables,
            "tables_lower": {t.lower() for t in tables.keys()},
        }
    return catalog


def load_all_schemas() -> dict[str, dict[str, str]]:
    """Merge every schema file into a {table: {column: type}} structure."""
    schema = defaultdict(dict)
    for entry in load_schema_catalog().values():
        for table, cols in entry["tables"].items():
            schema[table].update(cols)
    return dict(schema)


def compare_schema(db_schema, known_schemas):
    """
    Compare a database's live schema (from PRAGMA table_info)
    against known schemas. Returns a list of (table, column, type)
    for new or unseen columns. Comparison is case-insensitive and
    normalizes whitespace for table and column names.
    """
    new_cols = []

    def normalize(name: str) -> str:
        return " ".join(name.strip().split())

    known_schemas_norm = {
        normalize(k).lower(): {normalize(c).lower(): v for c, v in cols.items()}
        for k, cols in known_schemas.items()
    }

    for table, columns in db_schema.items():
        known_cols = known_schemas_norm.get(normalize(table).lower(), {})
        for col, ctype in columns.items():
            if normalize(col).lower() not in known_cols:
                new_cols.append((table, col, ctype))
    return new_cols


def save_missing_schema_report(missing, out_path):
    """Save a CSV report of unknown columns for analyst review."""
    if not missing:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Table", "Column", "Type"])
        writer.writerows(missing)
    print(f"🆕 New columns logged to {out_path}")


if __name__ == "__main__":
    catalog = load_schema_catalog()
    schemas = load_all_schemas()
    print(
        f"Loaded {len(schemas)} tables across {len(catalog)} schema files "
        f"({', '.join(sorted(catalog.keys()))})"
    )
