#!/usr/bin/env python3
"""
NEAREST: Best-Fit Exemplar Processor

Handles Lost & Found reconstruction for individual databases matched to nearest exemplars.

Process:
1. Match LF tables against nearest exemplar rubrics
2. Reconstruct databases using nearest exemplar schemas
3. Combine intact data + matched LF data
4. Create remnant databases for unmatched LF tables

Output: databases/found_data/{exemplar_name}/{exemplar_name}.sqlite
"""

from __future__ import annotations

import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.common.catalog_manager import CatalogManager
from mars.pipeline.lf_processor.lf_matcher import (
    match_lf_tables_to_exemplars,
)
from mars.pipeline.lf_processor.lf_reconstruction import (
    reconstruct_exemplar_database,
)
from mars.utils.database_utils import get_chosen_variant_path
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from rich.progress import TaskID

    from mars.utils.progress_utils import ProgressContextType


# Pattern to strip username and version suffix from exemplar names
# Matches: _username or _username_vN at end of string
_USERNAME_VERSION_PATTERN = re.compile(r"_[a-zA-Z][a-zA-Z0-9_]*(?:_v\d+)?$")


def _extract_exemplar_name(exemplar_path: Path) -> str:
    """Extract exemplar name from path, handling nested profile structures.

    Handles both flat and nested catalog structures:
    - Flat: catalog/{ExemplarName}/{ExemplarName}.db → ExemplarName
    - Nested: catalog/{ExemplarName}/{ProfileName}/{file}.db → ExemplarName

    Looks for 'catalog' in the path and returns the directory immediately after it.
    """
    parts = exemplar_path.parts
    try:
        catalog_idx = parts.index("catalog")
        # Return the directory name immediately after "catalog"
        if catalog_idx + 1 < len(parts):
            return parts[catalog_idx + 1]
    except ValueError:
        pass
    # Fallback to parent name (original behavior)
    return exemplar_path.parent.name


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


def match_nearest_exemplar_databases(
    individual_databases: list[dict],
    prep_lookup: dict,
    exemplar_db_dir: Path | None,
) -> dict:
    """
    Match LF tables from individual databases against nearest exemplar rubrics.

    Args:
        individual_databases: List of database records without catalog/metamatch
        prep_lookup: Dict mapping db_name -> prepared database info
        exemplar_db_dir: Path to exemplar databases directory

    Returns:
        Dict mapping exemplar_name -> [db_entries with match_results]
    """
    nearest_tracking = defaultdict(list)

    # Load catalog manager for per-DB ignorable tables lookup
    # Check if catalog exists at the expected location, otherwise use default (None)
    catalog_yaml = None
    if exemplar_db_dir:
        candidate_yaml = exemplar_db_dir.parent.parent / "artifact_recovery_catalog.yaml"
        if candidate_yaml.exists():
            catalog_yaml = candidate_yaml
    catalog_mgr = CatalogManager(catalog_yaml)

    for db_record in individual_databases:
        db_name = Path(db_record["case_path"]).stem

        # Skip databases marked as empty with variant_chosen="X"
        # UNLESS they have LF tables with recoverable data
        is_empty = db_record.get("decision", {}).get("empty", False)
        variant_chosen = db_record.get("variant_chosen", "")
        has_lf = db_record.get("meta_snapshot", {}).get("has_lost_and_found", False)
        if is_empty and variant_chosen == "X" and not has_lf:
            continue

        # Get prepared database info
        db_info = prep_lookup.get(db_name)

        # Get source database path (chosen variant)
        source_db = db_info["source_db"] if db_info else get_chosen_variant_path(db_record)

        # Handle databases with and without LF tables
        split_db = db_info["split_db"] if db_info and db_info["split_db"] else None

        # Get nearest exemplars from record.
        # These are close matches (e.g., tables only) or strong hints from byte carving.
        nearest_exemplars = db_record.get("nearest_exemplars", [])

        # Match LF tables against ALL nearest exemplars
        # (corrupt DBs can have fragments from multiple DB types)
        for exemplar_info in nearest_exemplars:
            exemplar_path = exemplar_info.get("exemplar")
            if not exemplar_path:
                continue

            # Extract exemplar name from path
            # Handles both flat and nested structures:
            # - Flat: .../catalog/{ExemplarName}/{ExemplarName}.db
            # - Nested: .../catalog/{ExemplarName}/{ProfileName}/{file}.db
            exemplar_path_obj = Path(exemplar_path)
            exemplar_name = _extract_exemplar_name(exemplar_path_obj)

            # Find rubric for this exemplar
            exemplar_rubric_path = None
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
                logger.debug(f"    ⚠ Rubric not found for {exemplar_name}, skipping")
                continue

            # Load rubric
            with exemplar_rubric_path.open() as f:
                exemplar_rubric = json.load(f)

            # Prepare rubric for matching
            exemplar_rubrics = [{"name": exemplar_name, "rubric": exemplar_rubric}]

            # Look up per-DB ignorable tables (e.g., 'settings' for Kext Policy Database)
            per_db_ignorable = catalog_mgr.get_ignorable_tables_for_rubric(exemplar_name)
            per_db_ignorable_tables = {exemplar_name: set(per_db_ignorable)} if per_db_ignorable else None

            # Match LF tables against this exemplar (skip if no LF tables)
            if split_db:
                match_results = match_lf_tables_to_exemplars(
                    split_db,
                    exemplar_rubrics,
                    exact_match_name=None,
                    nearest_exemplar_names=[exemplar_name],
                    per_db_ignorable_tables=per_db_ignorable_tables,
                )
            else:
                match_results = None

            # Track results (grouped by exemplar)
            # Note: Same database can contribute to multiple exemplars
            # Include ALL nearest matches, even without LF tables (they have intact data)
            nearest_tracking[exemplar_name].append(
                {
                    "db_name": db_name,
                    "split_db": split_db,
                    "source_db": source_db,
                    "match_results": match_results,
                    "record": db_record,  # Keep full record
                    "exemplar_info": exemplar_info,  # Keep exemplar path
                }
            )

        # Force garbage collection after each database to release any unreferenced
        # connections and prevent file descriptor exhaustion (ERRNO 24)
        gc.collect()

    # Summary
    total_matches = sum(len(dbs) for dbs in nearest_tracking.values())
    logger.debug(
        f"Nearest: {len(individual_databases)} individual database(s) processed, "
        f"{total_matches} exemplar match(es) tracked"
    )

    return dict(nearest_tracking)


def create_nearest_outputs(
    nearest_tracking: dict,
    databases_dir: Path,
    exemplar_db_dir: Path | None,
    progress_context: ProgressContextType | None = None,
    phase_sub_task: TaskID | None = None,
    consumed_lf_tables: dict[str, set[str]] | None = None,
) -> dict:
    """
    Create output databases for all nearest exemplar matches.

    Args:
        nearest_tracking: Dict mapping exemplar_name -> [db_entries with matches]
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
    total_groups = len(nearest_tracking)
    current_idx = 0

    # Track _multi groups: maps multi_name -> list of (exemplar_name, db_entries)
    multi_groups: dict[str, list[tuple[str, list]]] = {}

    for exemplar_name, db_entries in nearest_tracking.items():
        current_idx += 1

        # Update phase sub-task with current nearest being processed
        if progress_context and phase_sub_task:
            progress_context.update_sub(
                phase_sub_task,
                current_idx,
                description=f"Nearest {current_idx}/{total_groups}: {exemplar_name}",
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
        # Fall back to "name" field if "base_name" doesn't exist
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

                        # Get profiles list from provenance
                        profiles = prov.get("profiles", [])

                        # Extract real username from original source path
                        # e.g., "/Users/admin/Library/..." -> "admin"
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
        # Strip version suffix if present: "username_v2" -> "username"
        if not username and base_name and exemplar_name.startswith(base_name + "_"):
            suffix = exemplar_name[len(base_name) + 1 :]
            # Remove version suffix (e.g., "_v1", "_v2") to get real username
            version_match = re.search(r"_v\d+$", suffix)
            username = suffix[: version_match.start()] if version_match else suffix

        # Check if this exemplar has shared tables (part of a _multi group)
        multi_name = _get_multi_name_from_rubric(exemplar_rubric)
        if multi_name:
            if multi_name not in multi_groups:
                multi_groups[multi_name] = []
            multi_groups[multi_name].append((exemplar_name, db_entries))

        # Create output directory
        found_data_dir = databases_dir / "found_data" / exemplar_name

        # Reconstruct database (progress is tracked at the orchestrator level)
        result = reconstruct_exemplar_database(
            exemplar_name=exemplar_name,
            db_entries=db_entries,
            exemplar_rubric=exemplar_rubric,
            output_dir=found_data_dir,
            output_type="found_data",
            exemplar_db_dir=exemplar_db_dir,
            progress_context=progress_context,
            base_name=base_name,
            username=username,
            profiles=profiles,
            consumed_lf_tables=consumed_lf_tables,
        )

        if result.get("success"):
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

    logger.debug(f"Nearest: {total_created} found_data database(s) created ({multi_created} _multi)")

    return {"total_created": total_created, "multi_created": multi_created}


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
        seen_split_dbs: set[str] = set()
        unique_db_entries = []
        user_names = []
        for user_exemplar, db_entries in user_groups:
            user_names.append(user_exemplar)
            for entry in db_entries:
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

        # Create _multi catalog output directory (in found_data/ for NEAREST)
        multi_dir = databases_dir / "found_data" / multi_name

        # Reconstruct _multi database with shared tables ONLY
        # skip_shared_tables=False because we WANT the shared tables here
        # No need for contributing_exemplar_names since we re-matched against _multi
        result = reconstruct_exemplar_database(
            exemplar_name=multi_name,
            db_entries=unique_db_entries,
            exemplar_rubric=multi_rubric,
            output_dir=multi_dir,
            output_type="found_data",
            exemplar_db_dir=exemplar_db_dir,
            progress_context=None,
            skip_shared_tables=False,  # Include ALL tables from _multi rubric
            base_name=base_name,
            username="_multi",  # Explicit _multi username for export paths
            consumed_lf_tables=consumed_lf_tables,
        )

        if result.get("success"):
            created += 1
            logger.debug(
                f"      Created {multi_name}: {result.get('total_intact_rows', 0)} "
                f"intact + {result.get('total_lf_rows', 0)} LF rows"
            )

    return created
