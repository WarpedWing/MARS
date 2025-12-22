#!/usr/bin/env python3
"""Generate exemplar hash lookup table.

Scans schemas directory and creates a JSON mapping:
    {schema_hash: [database_name, ...]}

The schema hash is computed from both table names AND column names,
providing precise schema matching. Only databases with identical
table and column structures will share a hash.

This allows O(1) lookups instead of loading and comparing full rubrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mars.utils.debug_logger import logger

# Add parent directory to path for db_variant_selector imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_rubric(rubric_path: Path) -> dict[str, Any]:
    """Load a rubric JSON file."""
    with rubric_path.open() as f:
        return json.load(f)


def is_ignorable_table(
    name: str,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
) -> bool:
    """Check if a table should be ignored during schema comparison.

    This MUST match the logic in db_variant_selector.is_ignorable_table()
    to ensure hash computation is consistent.
    """
    SALVAGE_TABLE_CANON = {
        "lost_and_found",
        "lostandfound",
        "_lost_and_found",
        "lost_found",
        "lostfound",
    }

    n = name.lower()
    if n in ignore_tables:
        return True
    if n in SALVAGE_TABLE_CANON:  # exclude salvage from matching/probes
        return True
    if any(n.startswith(p) for p in ignore_prefixes):
        return True
    if any(n.endswith(s) for s in ignore_suffixes):
        return True
    return bool(n.startswith("z_") and n.endswith("_cache"))


def extract_effective_tables(
    rubric: dict[str, Any],
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
) -> set[str]:
    """Extract effective table names from a rubric.

    Args:
        rubric: Rubric data
        ignore_tables: Tables to ignore (sqlite_* etc)
        ignore_prefixes: Table name prefixes to ignore
        ignore_suffixes: Table name suffixes to ignore

    Returns:
        Set of effective table names
    """
    tables = rubric.get("tables", {})
    effective = set()

    for table_name in tables:
        if not is_ignorable_table(table_name, ignore_tables, ignore_prefixes, ignore_suffixes):
            effective.add(table_name)

    return effective


def compute_schema_hash(
    rubric: dict[str, Any],
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
) -> str:
    """Compute a hash of table names AND column names.

    This creates a signature based on both table names and their column names,
    providing precise schema matching. Databases with identical tables but
    different columns will have different hashes.

    Args:
        rubric: Rubric data
        ignore_tables: Tables to ignore
        ignore_prefixes: Table name prefixes to ignore
        ignore_suffixes: Table name suffixes to ignore

    Returns:
        MD5 hash of the full schema signature (tables + columns)
    """
    import hashlib

    tables = rubric.get("tables", {})
    table_signatures = []

    # Process each table in sorted order for consistent hashing
    for table_name in sorted(tables):
        if is_ignorable_table(table_name, ignore_tables, ignore_prefixes, ignore_suffixes):
            continue

        # Get column names for this table, sorted and lowercased
        # Rubric format: columns is a LIST of dicts with "name" key
        # e.g., [{"name": "id", "type": "INTEGER"}, {"name": "url", "type": "TEXT"}]
        table_def = tables[table_name]
        columns = table_def.get("columns", [])
        if isinstance(columns, list):
            col_names = sorted(c["name"].lower() for c in columns if isinstance(c, dict) and "name" in c)
        else:
            # Fallback for legacy dict format (if any)
            col_names = sorted(c.lower() for c in columns)

        # Format: tablename|col1,col2,col3
        table_sig = f"{table_name.lower()}|{','.join(col_names)}"
        table_signatures.append(table_sig)

    # Join all table signatures with newlines and hash
    schema_signature = "\n".join(table_signatures)
    return hashlib.md5(schema_signature.encode("utf-8")).hexdigest()


def generate_hash_lookup(
    schemas_dir: Path,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    """Generate hash lookup table from schemas directory.

    Args:
        schemas_dir: Path to schemas directory (e.g., exemplar/databases/schemas/)
        ignore_tables: Tables to ignore during hash computation
        ignore_prefixes: Table name prefixes to ignore
        ignore_suffixes: Table name suffixes to ignore
        output_path: Optional path to save JSON (defaults to schemas_dir/exemplar_hash_lookup.json)

    Returns:
        Dictionary mapping schema hash (tables + columns) to list of database names
        (multiple exemplars may share identical schemas)
    """
    hash_lookup: dict[str, list[str]] = {}

    if output_path is None:
        output_path = schemas_dir / "exemplar_hash_lookup.json"

    # Scan schemas directory
    for rubric_dir in sorted(schemas_dir.iterdir()):
        if not rubric_dir.is_dir():
            continue

        db_name = rubric_dir.name

        # Look for combined rubric first, then regular
        combined_rubric = rubric_dir / f"{db_name}_combined.rubric.json"
        regular_rubric = rubric_dir / f"{db_name}.rubric.json"

        rubric_path = combined_rubric if combined_rubric.exists() else regular_rubric

        if not rubric_path.exists():
            logger.warning(f"No rubric found for {db_name}")
            continue

        try:
            # Load rubric
            rubric = load_rubric(rubric_path)

            # Extract effective tables (for counting only)
            effective_tables = extract_effective_tables(rubric, ignore_tables, ignore_prefixes, ignore_suffixes)

            if not effective_tables:
                logger.warning(f"No effective tables for {db_name}")
                continue

            # Compute hash of full schema (tables + columns)
            schema_hash = compute_schema_hash(rubric, ignore_tables, ignore_prefixes, ignore_suffixes)

            # Add to lookup (multiple exemplars may share identical schema)
            if schema_hash in hash_lookup:
                hash_lookup[schema_hash].append(db_name)
                logger.info(
                    f"{db_name}: Identical schema to {hash_lookup[schema_hash][0]}, "
                    f"adding as duplicate ({len(hash_lookup[schema_hash])} total)"
                )
            else:
                hash_lookup[schema_hash] = [db_name]
                logger.info(f"{db_name}: {schema_hash[:12]}... ({len(effective_tables)} tables)")

        except Exception as e:
            logger.error(f"Error processing {db_name}: {e}")
            continue

    # Save to JSON
    total_exemplars = sum(len(names) for names in hash_lookup.values())
    logger.info(f"Saving {len(hash_lookup)} unique table hashes ({total_exemplars} exemplars) to {output_path}")
    with output_path.open("w") as f:
        json.dump(hash_lookup, f, indent=2, sort_keys=True)

    return hash_lookup
