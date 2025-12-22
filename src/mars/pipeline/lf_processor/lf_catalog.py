#!/usr/bin/env python3
"""
CATALOG: Exact Matches Processor

Handles Lost & Found reconstruction for databases with exact catalog matches.

Process:
1. Match LF tables against catalog exemplar rubrics
2. Reconstruct databases using exemplar schemas
3. Combine intact data + matched LF data from all matching databases
4. Create remnant databases for unmatched LF tables

Output: databases/catalog/{exemplar_name}/{exemplar_name}.sqlite
"""

from __future__ import annotations

import contextlib
import gc
import json
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.common.catalog_manager import CatalogManager
from mars.pipeline.lf_processor.lf_matcher import (
    match_lf_tables_to_exemplars,
)
from mars.pipeline.lf_processor.lf_reconstruction import (
    reconstruct_exemplar_database,
)
from mars.utils.cleanup_utilities import cleanup_sqlite_directory
from mars.utils.database_utils import get_chosen_variant_path
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from rich.progress import TaskID

    from mars.utils.progress_utils import ProgressContextType


# Pattern to strip username and version suffix from exemplar names
# Matches: _username or _username_vN at end of string
_USERNAME_VERSION_PATTERN = re.compile(r"_[a-zA-Z][a-zA-Z0-9_]*(?:_v\d+)?$")


def _count_exact_matches(record: dict) -> int:
    """
    Count how many exact matches (tables_equal+columns_equal or hash) a record has.

    Args:
        record: Database record with exact_matches list

    Returns:
        Number of exact matches with full schema match (not just table names)
    """
    exact_matches = record.get("exact_matches", [])
    return sum(1 for m in exact_matches if m.get("match") in ("tables_equal+columns_equal", "hash"))


def _has_unique_exact_match(db_entries: list, exemplar_name: str) -> bool:
    """
    Check if all databases in db_entries have a UNIQUE exact match to this exemplar.

    A unique exact match means:
    1. The record has exactly ONE tables_equal+columns_equal (or hash) match
    2. That match is for the specified exemplar_name

    If any database matches multiple versions exactly (e.g., v1 and v3 have identical
    schemas), this returns False because we can't uniquely attribute the data.

    Args:
        db_entries: List of database entries with record data
        exemplar_name: Name of the exemplar to check for unique match

    Returns:
        True if all entries have a unique exact match to this exemplar
    """
    for entry in db_entries:
        record = entry.get("record", {})
        exact_count = _count_exact_matches(record)

        # If multiple versions match exactly, not unique
        if exact_count != 1:
            return False

        # Check if the single exact match is for this exemplar
        exact_matches = record.get("exact_matches", [])
        is_match = False
        for match in exact_matches:
            if match.get("label") == exemplar_name and match.get("match") in ("tables_equal+columns_equal", "hash"):
                is_match = True
                break

        if not is_match:
            return False

    return True


def match_catalog_databases(
    catalog_groups: dict,
    prep_lookup: dict,
    exemplar_db_dir: Path | None,
) -> dict:
    """
    Match LF tables from catalog databases against their exemplar rubrics.

    Args:
        catalog_groups: Dict mapping exemplar_name -> [database records]
        prep_lookup: Dict mapping db_name -> prepared database info
        exemplar_db_dir: Path to exemplar databases directory

    Returns:
        Dict mapping exemplar_name -> [db_entries with match_results]
    """
    catalog_tracking = {}

    # Load catalog manager for per-DB ignorable tables lookup
    # Check if catalog exists at the expected location, otherwise use default (None)
    catalog_yaml = None
    if exemplar_db_dir:
        candidate_yaml = exemplar_db_dir.parent.parent / "artifact_recovery_catalog.yaml"
        if candidate_yaml.exists():
            catalog_yaml = candidate_yaml
    catalog_mgr = CatalogManager(catalog_yaml)

    for exemplar_name, databases in catalog_groups.items():
        # Find exemplar rubric
        exemplar_rubric_path = None
        candidate_paths = []

        if exemplar_db_dir and exemplar_db_dir.exists():
            schemas_dir = exemplar_db_dir / "schemas"

            # Try exact match first, then with _combined suffix
            candidate_paths = [
                schemas_dir / exemplar_name / f"{exemplar_name}.rubric.json",
                schemas_dir / exemplar_name / f"{exemplar_name}_combined.rubric.json",
            ]

            for candidate in candidate_paths:
                if candidate.exists():
                    exemplar_rubric_path = candidate
                    break

        if not exemplar_rubric_path:
            if not candidate_paths:
                logger.warning(f"    ⚠ Exemplar rubric not found for {exemplar_name} (no exemplar dir)")
            continue

        # Load exemplar rubric
        with exemplar_rubric_path.open() as f:
            exemplar_rubric = json.load(f)

        # Prepare rubric for matching
        exemplar_rubrics = [{"name": exemplar_name, "rubric": exemplar_rubric}]

        # Look up per-DB ignorable tables (e.g., 'settings' for Kext Policy Database)
        # LF data shouldn't match to ignorable tables
        per_db_ignorable = catalog_mgr.get_ignorable_tables_for_rubric(exemplar_name)
        per_db_ignorable_tables = {exemplar_name: set(per_db_ignorable)} if per_db_ignorable else None

        # Track matches for this exemplar
        catalog_tracking[exemplar_name] = []

        # Match LF tables from each database in this catalog group
        for db_record in databases:
            db_name = Path(db_record["case_path"]).stem

            # Find prepared database info
            db_info = prep_lookup.get(db_name)

            # Get source database path (chosen variant)
            source_db = db_info["source_db"] if db_info else get_chosen_variant_path(db_record)

            # Handle databases with and without LF tables
            if db_info and db_info["split_db"]:
                # Database has LF tables - match them
                split_db = db_info["split_db"]
                match_results = match_lf_tables_to_exemplars(
                    split_db,
                    exemplar_rubrics,
                    exact_match_name=exemplar_name,
                    nearest_exemplar_names=None,
                    per_db_ignorable_tables=per_db_ignorable_tables,
                )
            else:
                # Database has no LF tables - only intact data to copy
                split_db = None
                match_results = None

            # Track results (no output database creation yet)
            # Include ALL catalog matches, even without LF tables (they have intact data)
            catalog_tracking[exemplar_name].append(
                {
                    "db_name": db_name,
                    "split_db": split_db,
                    "source_db": source_db,
                    "match_results": match_results,
                    "record": db_record,  # Keep full record for access to exact_matches
                }
            )

    logger.debug(f"Catalog: {len(catalog_tracking)} catalog group(s) matched")

    return catalog_tracking


def create_catalog_outputs(
    catalog_tracking: dict,
    databases_dir: Path,
    exemplar_db_dir: Path | None,
    progress_context: ProgressContextType | None = None,
    phase_sub_task: TaskID | None = None,
    consumed_lf_tables: dict[str, set[str]] | None = None,
) -> dict:
    """
    Create output databases for all catalog matches.

    Also creates _multi catalogs for shared tables when multiple user-specific
    exemplars share tables (e.g., Mail Envelope Index_username1 and _username2
    share tables that go into Mail Envelope Index_multi).

    Args:
        catalog_tracking: Dict mapping exemplar_name -> [db_entries with matches]
        databases_dir: Base output directory
        exemplar_db_dir: Path to exemplar databases directory
        progress_context: Optional LFProgressContext for hierarchical progress tracking
        phase_sub_task: Optional TaskID for the phase-level sub-task (managed by orchestrator)
        consumed_lf_tables: Optional dict tracking which LF tables have been consumed.
            Keys are db_name, values are sets of lf_table names. Modified in place
            as LF tables are inserted into output databases.

    Returns:
        Dict with processing statistics
    """
    total_created = 0
    total_groups = len(catalog_tracking)
    current_idx = 0

    # Track _multi groups: maps multi_name -> list of (exemplar_name, db_entries)
    multi_groups: dict[str, list[tuple[str, list]]] = {}

    for exemplar_name, db_entries in catalog_tracking.items():
        current_idx += 1

        # Update phase sub-task with current catalog being processed
        if progress_context and phase_sub_task:
            progress_context.update_sub(
                phase_sub_task,
                current_idx,
                description=f"Catalog {current_idx}/{total_groups}: {exemplar_name}",
            )
        if not db_entries:
            continue

        # Find exemplar rubric
        exemplar_rubric_path = None
        if exemplar_db_dir and exemplar_db_dir.exists():
            schemas_dir = exemplar_db_dir / "schemas"
            candidate_paths = [
                schemas_dir / exemplar_name / f"{exemplar_name}.rubric.json",
                schemas_dir / exemplar_name / f"{exemplar_name}_combined.rubric.json",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    exemplar_rubric_path = candidate
                    break

        if not exemplar_rubric_path:
            logger.warning("      ⚠ Rubric not found, skipping")
            continue

        # Load rubric
        with exemplar_rubric_path.open() as f:
            exemplar_rubric = json.load(f)

        # Try to read base_name, username, and profiles from exemplar provenance file
        # Check both regular and combined provenance files
        base_name = None
        username = None
        profiles: list[str] = []
        if exemplar_db_dir and exemplar_db_dir.exists():
            catalog_dir_exemplar = exemplar_db_dir / "catalog" / exemplar_name
            provenance_candidates = [
                catalog_dir_exemplar / f"{exemplar_name}.provenance.json",
                catalog_dir_exemplar / f"{exemplar_name}.combined.provenance.json",
            ]
            for provenance_path in provenance_candidates:
                if provenance_path.exists():
                    try:
                        with provenance_path.open() as f:
                            prov = json.load(f)
                        base_name = prov.get("base_name") or prov.get("name")

                        # Extract profiles list from provenance
                        profiles = prov.get("profiles", [])

                        # Extract real username from source path (not from exemplar_name suffix)
                        # e.g., "/Users/admin/Library/..." → "admin"
                        source_dbs = prov.get("source_databases", [])
                        if source_dbs:
                            original_source = source_dbs[0].get("original_source", "")
                            user_match = re.search(r"/Users/([^/]+)/", original_source)
                            if user_match:
                                username = user_match.group(1)

                        break
                    except Exception:
                        pass

        # For versioned/combined DBs, the provenance "name" may include username+version
        # e.g., "CFURL Cache Database_username_v1" should become "CFURL Cache Database"
        # If base_name equals exemplar_name, try to extract the true base name
        if base_name and base_name == exemplar_name:
            match = _USERNAME_VERSION_PATTERN.search(base_name)
            if match:
                base_name = base_name[: match.start()]

        # Fallback: Extract username from exemplar_name suffix if not found in provenance
        # e.g., "Chrome Cookies_username" with base_name "Chrome Cookies" → "username"
        if not username and base_name and exemplar_name.startswith(base_name + "_"):
            username = exemplar_name[len(base_name) + 1 :]
            # Strip version suffix from username fallback (e.g., "admin_v2" → "admin")
            version_match = re.search(r"_v\d+(_empty)?$", username)
            if version_match:
                username = username[: version_match.start()]

        # Check if this exemplar has shared tables (part of a _multi group)
        multi_name = _get_multi_name_from_rubric(exemplar_rubric)

        # Check if databases have a UNIQUE exact match to this exemplar
        # If so, keep all data here (no split to _multi)
        # If multiple versions match exactly, route shared tables to _multi
        has_unique_match = _has_unique_exact_match(db_entries, exemplar_name)

        if multi_name and not has_unique_match:
            # Only add to _multi groups if NOT a unique exact match
            # Databases with unique exact matches keep ALL their data
            if multi_name not in multi_groups:
                multi_groups[multi_name] = []
            multi_groups[multi_name].append((exemplar_name, db_entries))

        # Create output directory
        catalog_dir = databases_dir / "catalog" / exemplar_name

        # Reconstruct database (progress is tracked at the orchestrator level)
        # skip_shared_tables=False for unique exact matches - keep all data here
        result = reconstruct_exemplar_database(
            exemplar_name=exemplar_name,
            db_entries=db_entries,
            exemplar_rubric=exemplar_rubric,
            output_dir=catalog_dir,
            output_type="carved",
            exemplar_db_dir=exemplar_db_dir,
            progress_context=progress_context,
            base_name=base_name,
            username=username,
            profiles=profiles,  # Pass profiles for export path resolution
            consumed_lf_tables=consumed_lf_tables,
            skip_shared_tables=not has_unique_match,  # Keep all tables for unique matches
        )

        if result.get("success"):
            total_rows = result.get("total_intact_rows", 0) + result.get("total_lf_rows", 0)

            # Check if database is effectively empty (only ignorable table data)
            is_effectively_empty = False
            if total_rows > 0 and catalog_dir.exists():
                db_files = list(catalog_dir.glob("*.sqlite"))
                if db_files:
                    from mars.pipeline.common.catalog_manager import (
                        CatalogManager,
                    )

                    catalog_yaml = (
                        exemplar_db_dir.parent.parent / "artifact_recovery_catalog.yaml" if exemplar_db_dir else None
                    )
                    catalog_mgr = CatalogManager(catalog_yaml)
                    # Look up per-DB ignorable tables (e.g., 'settings' for Kext Policy Database)
                    per_db_ignorable = catalog_mgr.get_ignorable_tables_for_rubric(exemplar_name)
                    is_effectively_empty = catalog_mgr.is_database_empty(
                        db_files[0], schema_ignorable_tables=per_db_ignorable
                    )
                    # Force garbage collection to release SQLite connection (Windows)
                    gc.collect()

            if total_rows == 0 or is_effectively_empty:
                # Move to empty/ instead of deleting - keeps LF fragments for inspection
                empty_dir = databases_dir / "empty"
                empty_dir.mkdir(parents=True, exist_ok=True)
                target = empty_dir / exemplar_name
                if catalog_dir.exists():
                    if target.exists():
                        with contextlib.suppress(Exception):
                            cleanup_sqlite_directory(target)
                    # Retry move with gc.collect() (Windows file handle release)
                    for attempt in range(3):
                        try:
                            shutil.move(str(catalog_dir), str(target))
                            break
                        except PermissionError:
                            if attempt < 2:
                                gc.collect()
                                time.sleep(0.1 * (attempt + 1))
                            else:
                                raise
                logger.debug(f"      Moved empty catalog to empty/: {exemplar_name}")
            else:
                total_created += 1

    # Create _multi catalogs for shared tables
    multi_created = 0
    if multi_groups and exemplar_db_dir:
        multi_created = _create_multi_catalogs(
            multi_groups=multi_groups,
            databases_dir=databases_dir,
            exemplar_db_dir=exemplar_db_dir,
            consumed_lf_tables=consumed_lf_tables,
        )
        total_created += multi_created

    logger.debug(f"Catalog: {total_created} catalog database(s) created ({multi_created} _multi)")

    return {"total_created": total_created, "multi_created": multi_created}


def _get_multi_name_from_rubric(rubric: dict) -> str | None:
    """
    Get the _multi rubric name if this rubric has shared tables.

    Returns the first "shared_with" value found, or None if no shared tables.
    """
    tables = rubric.get("tables", {})
    if not isinstance(tables, dict):
        return None

    for table_info in tables.values():
        if isinstance(table_info, dict) and table_info.get("shared_with"):
            return table_info["shared_with"]

    return None


def _create_multi_catalogs(
    multi_groups: dict[str, list[tuple[str, list]]],
    databases_dir: Path,
    exemplar_db_dir: Path,
    consumed_lf_tables: dict[str, set[str]] | None = None,
) -> int:
    """
    Create _multi catalogs from aggregated db_entries across user exemplars.

    Args:
        multi_groups: Dict mapping multi_name -> [(exemplar_name, db_entries), ...]
        databases_dir: Base output directory
        exemplar_db_dir: Path to exemplar databases directory
        consumed_lf_tables: Optional dict tracking which LF tables have been consumed.
            Modified in place as LF tables are inserted.

    Returns:
        Number of _multi catalogs created
    """
    created = 0
    schemas_dir = exemplar_db_dir / "schemas"

    # Load catalog manager for per-DB ignorable tables lookup
    # Check if catalog exists at the expected location, otherwise use default (None)
    catalog_yaml = None
    candidate_yaml = exemplar_db_dir.parent.parent / "artifact_recovery_catalog.yaml"
    if candidate_yaml.exists():
        catalog_yaml = candidate_yaml
    catalog_mgr = CatalogManager(catalog_yaml)

    for multi_name, user_groups in multi_groups.items():
        # Find _multi rubric
        multi_rubric_path = schemas_dir / multi_name / f"{multi_name}.rubric.json"
        if not multi_rubric_path.exists():
            logger.warning(f"      ⚠ _multi rubric not found: {multi_name}")
            continue

        with multi_rubric_path.open() as f:
            multi_rubric = json.load(f)

        # For _multi databases, base_name is the multi_name without "_multi" suffix
        # e.g., "Mail Envelope Index_multi" -> "Mail Envelope Index"
        base_name = multi_name.removesuffix("_multi") if multi_name.endswith("_multi") else None

        # Aggregate db_entries from user exemplars, deduplicating by split_db
        # Same source database may appear in multiple user exemplars; we only want
        # to process each LF source once to prevent data going to multiple tables
        #
        # Also filter out entries with unique exact matches - those should stay
        # in their user-specific catalog, not be duplicated to _multi
        seen_split_dbs: set[str] = set()
        unique_db_entries = []
        user_names = []
        for user_exemplar, db_entries in user_groups:
            user_names.append(user_exemplar)
            for entry in db_entries:
                # Skip entries with unique exact matches (they keep all data in user catalog)
                record = entry.get("record", {})
                exact_count = _count_exact_matches(record)
                if exact_count == 1:
                    # This database uniquely matches one version - skip for _multi
                    continue

                split_db = entry.get("split_db")
                if split_db:
                    split_db_key = str(split_db)
                    if split_db_key in seen_split_dbs:
                        continue  # Skip duplicate source
                    seen_split_dbs.add(split_db_key)
                unique_db_entries.append(entry)

        if not unique_db_entries:
            continue

        # Re-match LF tables against _multi rubric for proper best-match scoring
        # User exemplar match_results may have matched same LF table to different
        # tables based on their own rubric's statistics. Re-matching ensures
        # each LF table goes to its BEST match in the _multi rubric.
        multi_rubrics = [{"name": multi_name, "rubric": multi_rubric}]

        # Look up per-DB ignorable tables for _multi rubric
        per_db_ignorable = catalog_mgr.get_ignorable_tables_for_rubric(multi_name)
        per_db_ignorable_tables = {multi_name: set(per_db_ignorable)} if per_db_ignorable else None

        for entry in unique_db_entries:
            split_db = entry.get("split_db")
            if split_db and split_db.exists():
                entry["match_results"] = match_lf_tables_to_exemplars(
                    split_db, multi_rubrics, per_db_ignorable_tables=per_db_ignorable_tables
                )

        logger.debug(f"      Creating {multi_name} from {len(unique_db_entries)} source(s) ({', '.join(user_names)})")

        # Create _multi catalog output directory
        multi_dir = databases_dir / "catalog" / multi_name

        # Reconstruct _multi database with shared tables ONLY
        # skip_shared_tables=False because we WANT the shared tables here
        # No need for contributing_exemplar_names since we re-matched against _multi
        result = reconstruct_exemplar_database(
            exemplar_name=multi_name,
            db_entries=unique_db_entries,
            exemplar_rubric=multi_rubric,
            output_dir=multi_dir,
            output_type="carved",
            exemplar_db_dir=exemplar_db_dir,
            progress_context=None,
            skip_shared_tables=False,  # Include ALL tables from _multi rubric
            base_name=base_name,
            username="_multi",  # Explicit _multi username for export paths
            consumed_lf_tables=consumed_lf_tables,
        )

        if result.get("success"):
            total_rows = result.get("total_intact_rows", 0) + result.get("total_lf_rows", 0)
            if total_rows == 0:
                # Remove empty _multi catalog - no usable data
                if multi_dir.exists():
                    with contextlib.suppress(Exception):
                        cleanup_sqlite_directory(multi_dir)
                logger.debug(f"      Removed empty _multi catalog: {multi_name}")
            else:
                created += 1
                logger.debug(
                    f"      Created {multi_name}: {result.get('total_intact_rows', 0)} "
                    f"intact + {result.get('total_lf_rows', 0)} LF rows"
                )

    return created
