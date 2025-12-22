#!/usr/bin/env python3
"""Schema matching functions for database variant selection.

This module provides schema comparison and matching logic:
- Hash-based fast lookup for exact schema matches
- Table name matching
- Column structure matching
- Exemplar database matching

The matching strategy uses a two-tier approach:
1. Fast O(1) hash lookup when available (optional optimization)
2. Full table/column comparison fallback
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger

from .db_variant_selector_helpers import exemplar_label

if TYPE_CHECKING:
    from pathlib import Path

    from .models import DBMeta


def load_hash_lookup(schemas_dir: Path) -> dict[str, list[str]]:
    """Load exemplar hash lookup table from schemas directory.

    Reads the pre-generated hash lookup file that maps schema hashes to database names.
    This enables O(1) schema matching instead of loading all rubrics.

    Args:
        schemas_dir: Path to schemas directory (e.g., exemplar/databases/schemas/)

    Returns:
        Dictionary mapping schema hash (tables + columns) to list of database names
        (multiple exemplars may share identical schemas)
    """
    lookup_path = schemas_dir / "exemplar_hash_lookup.json"

    if not lookup_path.exists():
        return {}

    try:
        with lookup_path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Hash lookup file is corrupted!")
        logger.error(f"  File: {lookup_path}")
        logger.error(f"  Error: {e}")
        logger.error(f"  Absolute path: {lookup_path.resolve()}")
        logger.info("  Attempting to delete corrupted file...")
        try:
            lookup_path.unlink()
            logger.info("  Deleted corrupted file")
            logger.info("  File will be regenerated on next exemplar scan")
        except Exception as del_err:
            logger.error(f"  Could not delete file: {del_err}")
        logger.warning("  Continuing without hash lookup (will use slower rubric matching)...")
        return {}
    except FileNotFoundError:
        logger.warning(f"Hash lookup file not found: {lookup_path}")
        logger.warning("  This exemplar scan may be incomplete or corrupted")
        logger.warning(f"  Absolute path: {lookup_path.resolve()}")
        logger.warning("  Continuing without hash lookup...")
        return {}
    except Exception as e:
        logger.error("Failed to load hash lookup!")
        logger.error(f"  File: {lookup_path}")
        logger.error(f"  Error: {e}")
        logger.error(f"  Absolute path: {lookup_path.resolve()}")
        logger.warning("  Continuing without hash lookup...")
        return {}


def compute_case_schema_hash(case: DBMeta) -> str:
    """Compute schema hash for a case database (tables + columns).

    This must match the format used by generate_hash_lookup.py.
    Uses both table names and column names for precise schema matching.

    Args:
        case: Case database metadata

    Returns:
        MD5 hash of the full schema signature (tables + columns)
    """
    # Get effective table names (already filtered) in sorted order
    table_names = sorted(case.effective_table_names or set())
    table_signatures = []

    for table_name in table_names:
        # Get column names for this table, sorted and lowercased
        col_names = sorted(c.lower() for c in case.columns_by_table.get(table_name, set()))

        # Format: tablename|col1,col2,col3
        table_sig = f"{table_name.lower()}|{','.join(col_names)}"
        table_signatures.append(table_sig)

    # Join all table signatures with newlines and hash
    schema_signature = "\n".join(table_signatures)
    return hashlib.md5(schema_signature.encode("utf-8")).hexdigest()


def fast_hash_match(case: DBMeta, hash_lookup: dict[str, list[str]]) -> list[str]:
    """Fast O(1) hash lookup for case database.

    Computes schema hash (tables + columns) and checks against pre-generated lookup table.
    Returns list of candidate database names with identical schema structure.

    Args:
        case: Case database metadata
        hash_lookup: Hash lookup table (hash -> [db_name, ...])

    Returns:
        List of candidate database names (empty if no match)
    """
    if not hash_lookup:
        return []

    # Compute hash of tables + columns
    effective_tables = case.effective_table_names or set()
    if not effective_tables:
        return []

    schema_hash = compute_case_schema_hash(case)

    # O(1) lookup - returns list of candidates with identical schema
    return hash_lookup.get(schema_hash, [])


def exact_table_match(case: DBMeta, ex: DBMeta) -> bool:
    """Check if case and exemplar have exact table name match.

    Args:
        case: Case database metadata
        ex: Exemplar database metadata

    Returns:
        True if both have identical non-empty sets of table names
    """
    a = set(case.effective_table_names or set())
    b = set(ex.effective_table_names or set())
    return a == b and len(a) > 0


def exact_column_match(case: DBMeta, ex: DBMeta) -> bool:
    """Check if case and exemplar have exact column structure match.

    Compares both table names AND column names for each table.

    Args:
        case: Case database metadata
        ex: Exemplar database metadata

    Returns:
        True if all tables and their columns match exactly
    """
    if set(case.columns_by_table.keys()) != set(ex.columns_by_table.keys()):
        return False
    return all(case.columns_by_table[t] == ex.columns_by_table[t] for t in case.columns_by_table)


def compute_exact_matches(
    case: DBMeta,
    exemplars: list[DBMeta],
    exemplars_root: Path,
    hash_lookup: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Compute exact schema matches between case and exemplars.

    Uses two-tier matching strategy:
    1. If hash lookup succeeds: narrow to candidate exemplars only (fast path)
    2. If hash lookup fails/missing: check all exemplars (fallback)
    3. Check table names first (required)
    4. Check columns to determine match type:
       - "tables_equal+columns_equal": Full schema match (tables + columns)
       - "tables": Partial match (table names only, columns differ)

    Args:
        case: Case database metadata
        exemplars: List of exemplar database metadata
        exemplars_root: Root directory of exemplars (for labeling)
        hash_lookup: Optional hash lookup table for fast filtering (gracefully degrades if None)

    Returns:
        List of match records with label, count, sample_path, match type
        Sorted by label name for consistent output
    """
    # Fast path: use hash lookup to filter candidates (optional optimization)
    # If hash matches, we only check exemplars with matching labels
    candidate_db_names: set[str] = set()
    if hash_lookup:
        candidate_db_names = set(fast_hash_match(case, hash_lookup))
        # Note: If no hash match, candidate_db_names is empty and we check ALL exemplars

    # Optimization: Pre-filter exemplars when hash lookup succeeded
    # Build label -> exemplars lookup to avoid computing label for every exemplar
    if candidate_db_names:
        label_to_exemplars: dict[str, list[DBMeta]] = {}
        for ex in exemplars:
            label = exemplar_label(ex.path, exemplars_root)
            label_to_exemplars.setdefault(label, []).append(ex)

        # Only check exemplars that match candidate labels
        filtered_exemplars = []
        for label in candidate_db_names:
            filtered_exemplars.extend(label_to_exemplars.get(label, []))
    else:
        filtered_exemplars = exemplars

    matches_by_label: dict[str, dict] = {}
    for ex in filtered_exemplars:
        # Table-names-only matching (consistent for both hash and non-hash paths)
        if not exact_table_match(case, ex):
            continue

        # Check if columns match too
        columns_match = exact_column_match(case, ex)
        match_type = "tables_equal+columns_equal" if columns_match else "tables"

        label = exemplar_label(ex.path, exemplars_root)
        entry = matches_by_label.get(label)
        if entry is None:
            matches_by_label[label] = {
                "label": label,
                "count": 1,
                "sample_path": ex.path.as_posix(),
                "match": match_type,
            }
        else:
            entry["count"] += 1
            # If this exemplar has columns match, upgrade the match type
            if columns_match:
                entry["match"] = "tables_equal+columns_equal"

    return [matches_by_label[k] for k in sorted(matches_by_label)]
