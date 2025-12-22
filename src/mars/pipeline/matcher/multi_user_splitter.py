"""
Multi-User Rubric Splitter

When multiple users have the same database type (e.g., Mail Envelope Index_user,
Mail Envelope Index_username), tables with identical schemas across all instances
are marked as "shared" and a "_multi" rubric is created. This prevents:
1. Data duplication (same carved data matched to multiple users)
2. Misattribution (email attributed to wrong user)

Tables unique to one user or with schema differences stay unmarked in user rubrics.

KEY: User rubrics keep ALL tables (for database-level hash matching to work).
Shared tables are marked with "shared_with" for routing during LF processing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mars.config.schema import GLOBAL_IGNORABLE_TABLES
from mars.utils.debug_logger import logger


def load_catalog_db_names(catalog_path: Path) -> set[str]:
    """
    Load database base names from catalog YAML.

    Args:
        catalog_path: Path to artifact_recovery_catalog.yaml

    Returns:
        Set of database base names (e.g., "Mail Envelope Index", "Chrome History")
    """
    import yaml

    with Path.open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    db_names = set()
    for category in catalog.values():
        if isinstance(category, list):
            for entry in category:
                if isinstance(entry, dict) and "name" in entry:
                    db_names.add(entry["name"])

    return db_names


def group_multi_user_databases(rubric_dir: Path, catalog_db_names: set[str]) -> dict[str, list[Path]]:
    """
    Group rubric folders by database base name.

    Identifies user-scoped databases by matching folder names against catalog.
    E.g., "Mail Envelope Index_user" and "Mail Envelope Index_username"
    both have base name "Mail Envelope Index".

    Args:
        rubric_dir: Directory containing rubric folders (schemas/)
        catalog_db_names: Set of known database names from catalog

    Returns:
        Dict mapping base name to list of rubric folder paths.
        Only includes groups with 2+ instances (multi-user).
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    for folder in rubric_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name

        # Find matching catalog name (longest match wins)
        matched_base = None
        for db_name in catalog_db_names:
            if folder_name.startswith(db_name + "_") and (matched_base is None or len(db_name) > len(matched_base)):
                matched_base = db_name

        if matched_base:
            groups[matched_base].append(folder)

    # Only return groups with multiple instances
    return {base: paths for base, paths in groups.items() if len(paths) >= 2}


def compare_table_schemas(table1: dict[str, Any], table2: dict[str, Any]) -> bool:
    """
    Check if two table schemas are identical.

    Compares column names and types only (not roles or stats).

    Args:
        table1: First table schema dict with "columns" list
        table2: Second table schema dict with "columns" list

    Returns:
        True if schemas match exactly
    """
    cols1 = [(c["name"], c["type"]) for c in table1.get("columns", [])]
    cols2 = [(c["name"], c["type"]) for c in table2.get("columns", [])]
    return cols1 == cols2


def create_multi_rubric(
    rubric_paths: list[Path],
    base_name: str,
) -> tuple[dict[str, Any] | None, dict[Path, dict[str, Any]], set[str]]:
    """
    Create a _multi rubric with tables shared across all user instances.

    User rubrics are NOT modified to remove tables - they keep all tables
    for database-level hash matching. Instead, shared tables are marked
    with "shared_with" for routing during LF processing.

    Args:
        rubric_paths: List of rubric folder paths for same database type
        base_name: Base database name (e.g., "Mail Envelope Index")

    Returns:
        Tuple of:
            - multi_rubric: Rubric with shared tables (None if no shared tables)
            - modified_rubrics: Dict mapping original path to rubric with shared tables MARKED
            - shared_table_names: Set of table names that are shared
    """
    if len(rubric_paths) < 2:
        return None, {}, set()

    # Load all rubrics
    rubrics: dict[Path, dict[str, Any]] = {}
    for folder in rubric_paths:
        rubric_files = list(folder.glob("*.rubric.json"))
        if not rubric_files:
            continue
        # Use first rubric file found (should only be one per folder)
        rubric_path = rubric_files[0]
        with Path.open(rubric_path) as f:
            rubrics[folder] = json.load(f)

    if len(rubrics) < 2:
        return None, {}, set()

    # Get table names from each rubric
    all_tables: dict[Path, dict[str, Any]] = {}
    for folder, rubric in rubrics.items():
        all_tables[folder] = rubric.get("tables", {})

    # Find tables that exist in ALL rubrics
    first_tables = next(iter(all_tables.values()))
    common_table_names = set(first_tables.keys())
    for tables in all_tables.values():
        common_table_names &= set(tables.keys())

    # Filter out globally ignorable tables (meta, sqlite_sequence, etc.)
    # These are useless for forensic purposes - no point creating _multi for them
    ignorable_lower = {t.lower() for t in GLOBAL_IGNORABLE_TABLES}
    common_table_names = {t for t in common_table_names if t.lower() not in ignorable_lower}

    if not common_table_names:
        # No useful tables to share - don't create _multi rubric
        return None, {}, set()

    # Filter to tables with identical schemas across ALL rubrics
    shared_tables: dict[str, Any] = {}
    shared_table_names: set[str] = set()
    for table_name in common_table_names:
        # Get the table from first rubric as reference
        first_folder = next(iter(all_tables.keys()))
        ref_table = all_tables[first_folder][table_name]

        # Check if all other rubrics have identical schema
        all_match = True
        for folder, tables in all_tables.items():
            if folder == first_folder:
                continue
            if not compare_table_schemas(ref_table, tables[table_name]):
                all_match = False
                break

        if all_match:
            shared_tables[table_name] = ref_table
            shared_table_names.add(table_name)

    if not shared_tables:
        return None, {}, set()

    # Create multi rubric (copy structure from first rubric, replace tables)
    first_folder = next(iter(rubrics.keys()))
    multi_rubric = rubrics[first_folder].copy()
    multi_rubric["tables"] = shared_tables
    multi_rubric["rubric_name"] = f"{base_name}_multi"

    # Create modified rubrics with shared tables MARKED (not removed)
    # This preserves all tables for database-level hash matching
    multi_name = f"{base_name}_multi"
    modified_rubrics: dict[Path, dict[str, Any]] = {}
    for folder, rubric in rubrics.items():
        import copy

        modified = copy.deepcopy(rubric)
        for table_name in shared_table_names:
            if table_name in modified.get("tables", {}):
                # Mark table as shared - routing will use this during LF processing
                modified["tables"][table_name]["shared_with"] = multi_name
        modified_rubrics[folder] = modified

    logger.debug(
        f"Created _multi rubric with {len(shared_tables)} shared tables, marked in {len(modified_rubrics)} user rubrics"
    )

    return multi_rubric, modified_rubrics, shared_table_names


def split_and_write_rubrics(rubric_dir: Path, catalog_path: Path, dry_run: bool = False) -> dict[str, int]:
    """
    Main entry point: create _multi rubrics and mark shared tables in user rubrics.

    User rubrics keep ALL tables (for database-level hash matching).
    Shared tables are marked with "shared_with" for routing during LF processing.

    Args:
        rubric_dir: Directory containing rubric folders (schemas/)
        catalog_path: Path to artifact_recovery_catalog.yaml
        dry_run: If True, don't write files, just return counts

    Returns:
        Dict with counts: {"multi_rubrics_created": N, "tables_marked": M}
    """
    catalog_db_names = load_catalog_db_names(catalog_path)
    groups = group_multi_user_databases(rubric_dir, catalog_db_names)

    stats = {"multi_rubrics_created": 0, "tables_marked": 0, "user_rubrics_updated": 0}

    for base_name, paths in groups.items():
        logger.debug(f"Processing multi-user group: {base_name} ({len(paths)} instances)")

        multi_rubric, modified_rubrics, shared_table_names = create_multi_rubric(paths, base_name)

        if multi_rubric is None:
            logger.debug(f"  No shared tables found for {base_name}")
            continue

        shared_count = len(shared_table_names)
        stats["tables_marked"] += shared_count
        stats["multi_rubrics_created"] += 1

        if dry_run:
            logger.debug(f"  [DRY RUN] Would create {base_name}_multi with {shared_count} tables")
            continue

        # Create _multi folder and write rubric
        multi_folder = rubric_dir / f"{base_name}_multi"
        multi_folder.mkdir(exist_ok=True)
        multi_rubric_path = multi_folder / f"{base_name}_multi.rubric.json"

        with Path.open(multi_rubric_path, "w") as f:
            json.dump(multi_rubric, f, indent=2)

        logger.debug(f"  Created {multi_rubric_path.name} with {shared_count} tables")

        # Update user rubrics (mark shared tables, keep all tables)
        for folder, modified in modified_rubrics.items():
            rubric_files = list(folder.glob("*.rubric.json"))
            if rubric_files:
                with Path.open(rubric_files[0], "w") as f:
                    json.dump(modified, f, indent=2)
                marked_count = sum(1 for t in modified.get("tables", {}).values() if t.get("shared_with"))
                logger.debug(f"  Updated {folder.name}: {marked_count} tables marked as shared")
                stats["user_rubrics_updated"] += 1

    return stats
