#!/usr/bin/env python3
"""
Lost & Found Database Reconstruction Orchestrator

Orchestrates the complete Lost & Found processing pipeline for recovered SQLite databases.
This orchestrator coordinates 4 processing modes by delegating to specialized processors:

- MERGE: Metamatch groups - databases with similar schemas combined via superrubric
- CATALOG: Catalog matches - databases with exact schema matches to known exemplars
- NEAREST: Individual/Nearest - databases matched to nearest exemplar
- ORPHAN: Unmatched lost_and_found tables preserved for review

The orchestrator handles:
- Phase 1: Preparing split databases from lost_and_found tables
- Phase 2: Grouping databases by match type (catalog/metamatch/individual)
- Phase 3: MERGE - Metamatch groups
- Phase 4: CATALOG - Exact matches
- Phase 5: NEAREST - Best-fit matches
- Phase 6: ORPHAN - Unmatched tables
- Phase 7: Reclassifying found_data results
- Cleanup: Removing temporary directories

Extracted from candidate_processor.py, then refactored to delegate to processors.
"""

from __future__ import annotations

import gc
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import Progress

from mars.utils.cleanup_utilities import cleanup_sqlite_directory
from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger
from mars.utils.progress_utils import (
    LFProgressContext,
    _NoOpProgressContext,
    create_standard_progress,
)

if TYPE_CHECKING:
    from rich.console import Console

    from mars.config import MARSConfig, ProjectPaths


def _retry_move(src: Path, dst: Path, max_attempts: int = 3) -> None:
    """
    Move a directory with retry for Windows file handle release.

    On Windows, file handles may not be released immediately after closing
    SQLite connections. This function retries the move operation with
    gc.collect() and small delays to allow handle release.

    Args:
        src: Source path to move
        dst: Destination path
        max_attempts: Maximum number of attempts (default 3)

    Raises:
        PermissionError: If all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            shutil.move(str(src), str(dst))
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                gc.collect()
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


class LFOrchestrator:
    """Orchestrates Lost & Found database reconstruction across all use cases."""

    def __init__(
        self,
        paths: ProjectPaths,
        exemplar_db_dir: Path | None = None,
        config: MARSConfig | None = None,
    ):
        """
        Initialize the Lost & Found orchestrator.

        Args:
            paths: Project paths configuration
            exemplar_db_dir: Path to exemplar databases directory (optional)
            config: Configuration object (optional)
        """
        self.paths = paths
        self.exemplar_db_dir = exemplar_db_dir
        self.config = config

    def process_lost_and_found_tables(self, richConsole: Console | None):
        """
        Process lost_and_found tables from recovered databases.

        This method orchestrates the complete LF processing pipeline:
        1. Prepares split databases from lost_and_found tables
        2. Groups databases by match type
        3. Delegates to UC processors for reconstruction
        4. Reclassifies results based on recovery success
        5. Cleans up temporary files
        """
        logger.debug("Processing lost_and_found tables...")

        # Check if results file exists
        results_path = self.paths.db_selected_variants / "sqlite_scan_results.jsonl"
        if not results_path.exists():
            return

        # Read all databases that need processing (use spinner for quick operation)
        databases_with_lf = []

        # Use console status spinner if available
        from contextlib import nullcontext

        status_context = (
            richConsole.status("[cyan]Loading database results...[/cyan]") if richConsole else nullcontext()
        )

        with status_context, Path.open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    has_lf = record.get("meta_snapshot", {}).get("has_lost_and_found", False)
                    has_match = record.get("decision", {}).get("matched", False)
                    has_metamatch = bool(record.get("metamatch", {}).get("group_id"))

                    # Skip empty databases unless they have LF tables
                    is_empty = record.get("decision", {}).get("empty", False)
                    variant_chosen = record.get("variant_chosen", "")
                    if is_empty and variant_chosen == "X" and not has_lf:
                        continue

                    # Include if database needs processing:
                    # - has_lf: needs LF table recovery
                    # - has_match: will contribute intact data to catalog
                    # - has_metamatch: will be combined in MERGE (any group size, including singletons)
                    # Note: nearest_exemplars is just a naming hint, not a routing decision.
                    if has_lf or has_match or has_metamatch:
                        databases_with_lf.append(record)
                except json.JSONDecodeError:
                    continue

        if not databases_with_lf:
            return

        lf_count = sum(1 for r in databases_with_lf if r.get("meta_snapshot", {}).get("has_lost_and_found"))
        logger.debug(f"Found {len(databases_with_lf)} database(s) to process ({lf_count} with LF tables)")

        # Delegate to helper method that coordinates all phases with unified progress
        self._process_all_phases_with_progress(databases_with_lf, lf_count, richConsole)

    def _process_all_phases_with_progress(self, databases_with_lf: list, lf_count: int, richConsole: Console | None):
        """
        Process all 7 phases of LF reconstruction with unified progress tracking.

        Args:
            databases_with_lf: List of database records to process
            lf_count: Number of databases with actual LF tables
        """
        # Import required modules
        from mars.pipeline.lf_processor.lf_splitter import (
            create_split_database,
            extract_fragments,
            merge_compatible_fragments,
        )
        from mars.pipeline.raw_scanner.self_rubric_generator import (
            generate_self_rubric_for_catalog,
        )
        from mars.utils.database_utils import get_chosen_variant_path

        # Create unified progress for all 7 phases
        with create_standard_progress(
            label="",
            header_title="Lost & Found Processing",
            header_subtitle=f"{len(databases_with_lf)} databases ({lf_count} with LF tables)",
            show_time="elapsed",
            console=richConsole,
            config=self.config,
        ) as progress:
            # Main task tracks the 7 phases
            task = progress.add_task("[cyan]Phase 1/7: Preparing split databases...", total=7)

            # Create hierarchical progress context with 7 phases
            # Use isinstance check for proper type narrowing
            if isinstance(progress, Progress):
                ctx: LFProgressContext | _NoOpProgressContext = LFProgressContext(progress, task, total_phases=7)
            else:
                ctx = _NoOpProgressContext(console=richConsole)

            # ==================================================================
            # PHASE 1: Prepare all split databases and collect metadata
            # ==================================================================
            logger.debug("Phase 1: Creating split databases...")
            ctx.update_main("Phase 1/7: Preparing split databases...", phase=0)

            lf_temp_dir = self.paths.db_selected_variants / "lf_processing"
            lf_temp_dir.mkdir(parents=True, exist_ok=True)

            prepared_databases = []

            # Create sub-task for Phase 1 splitting
            phase1_sub = ctx.create_sub_task("Splitting databases", total=len(databases_with_lf))

            for idx, record in enumerate(databases_with_lf, 1):
                case_path = record.get("case_path", "")
                db_name = Path(case_path).stem if case_path else f"unknown_{idx}"

                ctx.update_sub(phase1_sub, idx, description=f"Splitting {db_name}")

                chosen_variant_path = get_chosen_variant_path(record)
                if not chosen_variant_path:
                    continue

                has_lf = record.get("meta_snapshot", {}).get("has_lost_and_found", False)
                if not has_lf:
                    # Database has no LF tables - will only contribute intact data
                    prepared_databases.append(
                        {
                            "record": record,
                            "db_name": db_name,
                            "split_db": None,
                            "source_db": chosen_variant_path,
                            "self_rubric": None,
                        }
                    )
                    continue

                # Look for extracted lost_and_found database
                lf_extracted_db = chosen_variant_path.parent / f"{db_name}_lost_and_found.sqlite"

                if not lf_extracted_db.exists():
                    prepared_databases.append(
                        {
                            "record": record,
                            "db_name": db_name,
                            "split_db": None,
                            "source_db": chosen_variant_path,
                            "self_rubric": None,
                        }
                    )
                    continue

                # Create split database
                split_db_path = lf_temp_dir / f"{db_name}.split.sqlite"

                try:
                    if not split_db_path.exists():
                        # Verify lost_and_found table exists
                        try:
                            with sqlite3.connect(f"file:{lf_extracted_db}?mode=ro", uri=True) as test_con:
                                cursor = test_con.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' AND name='lost_and_found'"
                                )
                                has_lf = cursor.fetchone() is not None

                            if not has_lf:
                                continue
                        except Exception:
                            continue

                        # Extract and merge fragments
                        fragments = extract_fragments(lf_extracted_db)
                        if not fragments:
                            continue

                        merged_groups = merge_compatible_fragments(fragments)

                        # Create split database with ONLY LF tables
                        create_split_database(
                            output_path=split_db_path,
                            merged_groups=merged_groups,
                            min_rows=1,
                            source_db_path=None,
                        )

                    prepared_databases.append(
                        {
                            "record": record,
                            "db_name": db_name,
                            "split_db": split_db_path,
                            "source_db": chosen_variant_path,
                            "self_rubric": None,
                        }
                    )

                except Exception as e:
                    logger.error(f"  [{idx}/{len(databases_with_lf)}] {db_name}: Error preparing database: {e}")
                    continue

            # Phase 1 complete
            ctx.remove_sub(phase1_sub)
            logger.debug(f"Phase 1 complete: {len(prepared_databases)} database(s) prepared")
            ctx.update_main("Phase 1/7: Complete", phase=1)
            gc.collect()  # Release file handles between phases

            if not prepared_databases:
                return

            # ==================================================================
            # PHASE 2: Group databases by catalog match / metamatch / individual
            # ==================================================================
            ctx.update_main("Phase 2/7: Grouping databases...", phase=1)
            logger.debug("Phase 2: Grouping databases...")

            records_to_group = [db["record"] for db in prepared_databases]
            groups = self._group_databases_for_lf_processing(records_to_group)

            prep_lookup = {db["db_name"]: db for db in prepared_databases}

            logger.debug(f"  Catalog groups: {len(groups['catalog_groups'])}")
            logger.debug(f"  Metamatch groups: {len(groups['metamatch_groups'])}")
            logger.debug(f"  Individual databases: {len(groups['individual'])}")

            # Generate self-rubrics for individual databases
            individual_rubric_count = 0
            if groups["individual"]:
                phase2_sub = ctx.create_sub_task("Generating self-rubrics", total=len(groups["individual"]))
                for idx, record in enumerate(groups["individual"], 1):
                    db_name = Path(record["case_path"]).stem
                    db_info = prep_lookup.get(db_name)
                    if db_info and db_info["source_db"]:
                        self_rubric = generate_self_rubric_for_catalog(
                            db_info["source_db"],
                            db_name,
                            self.paths.db_selected_variants.parent,
                        )
                        db_info["self_rubric"] = self_rubric
                        if self_rubric:
                            individual_rubric_count += 1

                    ctx.update_sub(phase2_sub, idx, description=f"Self-rubric for {db_name}")
                ctx.remove_sub(phase2_sub)

            # ==================================================================
            # PHASE 3: MERGE - Early combination and superrubric generation
            # ==================================================================
            ctx.update_main("Phase 3/7: Merge (metamatch groups)...", phase=2)
            logger.debug("Phase 3: Merge (metamatch groups)...")

            from mars.pipeline.lf_processor.lf_merge import (
                process_metamatch_group,
            )

            databases_dir = self.paths.db_selected_variants.parent
            lf_temp_dir = databases_dir / "lf_temp"
            lf_temp_dir.mkdir(parents=True, exist_ok=True)

            metamatch_results = {}
            total_groups = len(groups["metamatch_groups"])

            if total_groups > 0:
                phase3_sub = ctx.create_sub_task(f"Processing 0/{total_groups} groups", total=total_groups)

                for idx, (group_id, group_info) in enumerate(groups["metamatch_groups"].items()):
                    group_label = group_info.get("group_label", group_id[:8])
                    ctx.update_sub(phase3_sub, idx, description=f"Group {idx + 1}/{total_groups}: {group_label}")

                    try:
                        result = process_metamatch_group(
                            group_id=group_id,
                            group_info=group_info,
                            prep_lookup=prep_lookup,
                            databases_dir=databases_dir,
                            lf_temp_dir=lf_temp_dir,
                            exemplar_db_dir=self.exemplar_db_dir,
                            progress_context=ctx,
                        )
                        metamatch_results[group_id] = result

                        if not result.get("success"):
                            logger.debug(f"    ✗ Failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        metamatch_results[group_id] = {
                            "success": False,
                            "error": str(e),
                        }

                    # Release file handles after each group to prevent FD exhaustion
                    gc.collect()

                ctx.update_sub(phase3_sub, total_groups)
                ctx.remove_sub(phase3_sub)

            logger.debug(f"Phase 3 complete: {len(metamatch_results)} metamatch group(s) processed")
            ctx.update_main("Phase 3/7: Complete", phase=3)
            gc.collect()  # Release file handles between phases

            # ==================================================================
            # Initialize consumed_lf_tables tracking from MERGE results
            # ==================================================================
            # This prevents the same LF table from being inserted into multiple
            # output databases. LF tables consumed in MERGE are marked here, and
            # CATALOG/NEAREST will check and update this dict as they process.
            from mars.pipeline.lf_processor.lf_orphan import collect_matched_lf_tables

            # Collect LF tables consumed by MERGE phase
            consumed_lf_tables: dict[str, set[str]] = {}
            merge_matched = collect_matched_lf_tables(
                metamatch_results=metamatch_results,
                catalog_tracking={},  # CATALOG hasn't run yet
                nearest_tracking={},  # NEAREST hasn't run yet
            )
            # Convert to mutable sets for tracking
            for db_name, matched_tables in merge_matched.items():
                consumed_lf_tables[db_name] = set(matched_tables)

            # ==================================================================
            # PHASE 4: CATALOG - Match and reconstruct
            # ==================================================================
            ctx.update_main("Phase 4/7: Catalog (exact matches)...", phase=3)
            logger.debug("Phase 4: Catalog (exact matches)...")

            from mars.pipeline.lf_processor.lf_catalog import (
                create_catalog_outputs,
                match_catalog_databases,
            )

            # Match LF tables against catalog rubrics
            catalog_tracking = match_catalog_databases(
                catalog_groups=groups["catalog_groups"],
                prep_lookup=prep_lookup,
                exemplar_db_dir=self.exemplar_db_dir,
            )

            # Create catalog output databases with progress tracking
            total_catalog_groups = len(catalog_tracking)
            if total_catalog_groups > 0:
                phase4_sub = ctx.create_sub_task(
                    f"Processing 0/{total_catalog_groups} catalogs", total=total_catalog_groups
                )

                create_catalog_outputs(
                    catalog_tracking=catalog_tracking,
                    databases_dir=databases_dir,
                    exemplar_db_dir=self.exemplar_db_dir,
                    progress_context=ctx,
                    phase_sub_task=phase4_sub,
                    consumed_lf_tables=consumed_lf_tables,
                )

                ctx.remove_sub(phase4_sub)
            else:
                create_catalog_outputs(
                    catalog_tracking=catalog_tracking,
                    databases_dir=databases_dir,
                    exemplar_db_dir=self.exemplar_db_dir,
                    consumed_lf_tables=consumed_lf_tables,
                )

            ctx.update_main("Phase 4/7: Complete", phase=4)
            gc.collect()  # Release file handles between phases

            # ==================================================================
            # PHASE 5: NEAREST - Match and reconstruct
            # ==================================================================
            ctx.update_main("Phase 5/7: Nearest (best-fit matches)...", phase=4)
            logger.debug("Phase 5: Nearest (best-fit matches)...")

            from mars.pipeline.lf_processor.lf_nearest import (
                create_nearest_outputs,
                match_nearest_exemplar_databases,
            )

            # Match LF tables against nearest exemplar rubrics
            nearest_tracking = match_nearest_exemplar_databases(
                individual_databases=groups["individual"],
                prep_lookup=prep_lookup,
                exemplar_db_dir=self.exemplar_db_dir,
            )

            # Create found_data output databases with progress tracking
            total_nearest_groups = len(nearest_tracking)
            if total_nearest_groups > 0:
                phase5_sub = ctx.create_sub_task(
                    f"Processing 0/{total_nearest_groups} nearest", total=total_nearest_groups
                )

                create_nearest_outputs(
                    nearest_tracking=nearest_tracking,
                    databases_dir=databases_dir,
                    exemplar_db_dir=self.exemplar_db_dir,
                    progress_context=ctx,
                    phase_sub_task=phase5_sub,
                    consumed_lf_tables=consumed_lf_tables,
                )

                ctx.remove_sub(phase5_sub)
            else:
                create_nearest_outputs(
                    nearest_tracking=nearest_tracking,
                    databases_dir=databases_dir,
                    exemplar_db_dir=self.exemplar_db_dir,
                    consumed_lf_tables=consumed_lf_tables,
                )

            ctx.update_main("Phase 5/7: Complete", phase=5)
            gc.collect()  # Release file handles between phases

            # ==================================================================
            # PHASE 6: ORPHAN - Unmatched LF tables
            # ==================================================================
            ctx.update_main("Phase 6/7: Orphan (unmatched tables)...", phase=5)
            logger.debug("Phase 6: Orphan (unmatched tables)...")

            from mars.pipeline.lf_processor.lf_orphan import (
                collect_matched_lf_tables,
                process_orphan_tables,
            )

            # Collect all matched LF tables from Merge/Catalog/Nearest
            matched_lf_by_db = collect_matched_lf_tables(
                metamatch_results=metamatch_results,
                catalog_tracking=catalog_tracking,
                nearest_tracking=nearest_tracking,
            )

            # Process orphan tables with progress tracking
            found_data_dir = databases_dir / "found_data"
            found_data_dir.mkdir(parents=True, exist_ok=True)

            total_dbs_to_check = len(prep_lookup)
            if total_dbs_to_check > 0:
                phase6_sub = ctx.create_sub_task(f"Checking 0/{total_dbs_to_check} databases", total=total_dbs_to_check)

                process_orphan_tables(
                    prep_lookup=prep_lookup,
                    matched_lf_by_db=matched_lf_by_db,
                    found_data_dir=found_data_dir,
                    progress_context=ctx,
                    phase_sub_task=phase6_sub,
                )

                ctx.remove_sub(phase6_sub)
            else:
                process_orphan_tables(
                    prep_lookup=prep_lookup,
                    matched_lf_by_db=matched_lf_by_db,
                    found_data_dir=found_data_dir,
                )

            ctx.update_main("Phase 6/7: Complete", phase=6)
            gc.collect()  # Release file handles between phases

            # ==================================================================
            # PHASE 7: Reclassify found_data based on recovery results
            # ==================================================================
            ctx.update_main("Phase 7/7: Reclassifying found_data...", phase=6)
            logger.debug("Phase 7: Reclassifying found_data databases...")

            found_data_dir = databases_dir / "found_data"
            catalog_dir = databases_dir / "catalog"
            empty_dir = databases_dir / "empty"
            empty_dir.mkdir(parents=True, exist_ok=True)

            reclassified_to_catalog = 0
            reclassified_to_empty = 0

            merged_count = 0

            if found_data_dir.exists():
                for folder in found_data_dir.iterdir():
                    if not folder.is_dir():
                        continue

                    # Skip orphan folders (no manifest file)
                    manifest_path = folder / f"{folder.name}_manifest.json"
                    if not manifest_path.exists():
                        continue

                    # Read manifest to check total_lf_rows
                    try:
                        with manifest_path.open() as f:
                            manifest = json.load(f)

                        # total_lf_rows is nested under combined_output
                        combined_output = manifest.get("combined_output", {})
                        total_lf_rows = combined_output.get("total_lf_rows", 0)

                        if total_lf_rows > 0:
                            # Check if catalog/{exemplar_name} already exists
                            existing_catalog = catalog_dir / folder.name
                            if existing_catalog.exists():
                                # Merge into existing catalog entry
                                self._merge_catalog_entry(folder, existing_catalog)
                                # Remove source folder after merge (Windows-compatible cleanup)
                                cleanup_sqlite_directory(folder)
                                merged_count += 1
                            else:
                                # No existing entry - just move (no _found suffix)
                                _retry_move(folder, existing_catalog)
                                reclassified_to_catalog += 1
                        else:
                            # Move to empty (matched exemplar but no recoverable data)
                            target_path = empty_dir / folder.name

                            # Handle collision
                            if target_path.exists():
                                target_path = empty_dir / f"{folder.name}_lf"

                            _retry_move(folder, target_path)
                            reclassified_to_empty += 1

                    except Exception as e:
                        logger.warning(f"  ⚠ Failed to reclassify {folder.name}: {e}")

            # Scan catalog/ for databases that are effectively empty
            # (may have been placed there but only have ignorable table data)
            if catalog_dir.exists():
                from mars.pipeline.common.catalog_manager import (
                    CatalogManager,
                )

                catalog_yaml = (
                    self.exemplar_db_dir.parent / "artifact_recovery_catalog.yaml" if self.exemplar_db_dir else None
                )
                catalog_mgr = CatalogManager(catalog_yaml)

                for folder in list(catalog_dir.iterdir()):
                    if not folder.is_dir():
                        continue

                    # Find the sqlite file
                    db_files = list(folder.glob("*.sqlite"))
                    if not db_files:
                        continue

                    # Check if database is effectively empty
                    try:
                        # Look up per-DB ignorable tables (e.g., 'settings' for Kext Policy Database)
                        per_db_ignorable = catalog_mgr.get_ignorable_tables_for_rubric(folder.name)
                        if catalog_mgr.is_database_empty(db_files[0], schema_ignorable_tables=per_db_ignorable):
                            # Force garbage collection to release SQLite connection (Windows)
                            gc.collect()
                            target_path = empty_dir / folder.name
                            if target_path.exists():
                                cleanup_sqlite_directory(target_path)
                            _retry_move(folder, target_path)
                            reclassified_to_empty += 1
                            logger.debug(f"  Moved empty database to empty/: {folder.name}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Failed to check emptiness of {folder.name}: {e}")

            logger.debug(
                f"Phase 7 complete: {reclassified_to_catalog} moved to catalog, "
                f"{merged_count} merged with existing, {reclassified_to_empty} moved to empty"
            )
            ctx.update_main("Phase 7/7: Complete", phase=7)

            # ==================================================================
            # CLEANUP: Remove incomplete outputs and temporary directories
            # ==================================================================
            ctx.update_main("Cleaning up temporary files...")

            # CRITICAL: Clear split_db references from prep_lookup BEFORE cleanup
            # This releases Python's references to Path objects pointing to temporary files
            for db_info in prep_lookup.values():
                db_info["split_db"] = None  # Clear reference to split database
            prep_lookup.clear()  # Clear entire lookup to release all references

            # Force garbage collection to release any SQLite connections
            gc.collect()
            time.sleep(0.1)  # Small delay for OS handle release

            # Clean up incomplete metamatch folders (no manifest = failed/interrupted)
            metamatches_dir = databases_dir / "metamatches"
            incomplete_removed = 0
            if metamatches_dir.exists():
                for folder in list(metamatches_dir.iterdir()):
                    if not folder.is_dir():
                        continue
                    manifest = folder / f"{folder.name}_manifest.json"
                    if not manifest.exists():
                        try:
                            cleanup_sqlite_directory(folder)
                            incomplete_removed += 1
                        except Exception:
                            pass
                if incomplete_removed > 0:
                    logger.debug(f"  Cleaned up {incomplete_removed} incomplete metamatch folder(s)")

            # Remove lf_temp directory (used by metamatch processing)
            lf_temp_dir = databases_dir / "lf_temp"
            if lf_temp_dir.exists():
                try:
                    cleanup_sqlite_directory(lf_temp_dir)
                    logger.debug("  Cleaned up lf_temp directory")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to remove lf_temp: {e}")

            # Remove lf_processing directory (used by split database processing)
            lf_processing_dir = self.paths.db_selected_variants / "lf_processing"
            if lf_processing_dir.exists():
                try:
                    cleanup_sqlite_directory(lf_processing_dir)
                    logger.debug("  Cleaned up lf_processing directory")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to remove lf_processing: {e}")

    def _merge_catalog_entry(self, source_folder: Path, target_folder: Path) -> None:
        """
        Merge a found_data entry into an existing catalog entry.

        This combines L&F-only data with existing carved data when both
        match the same exemplar schema.

        Args:
            source_folder: Path to found_data/{exemplar}/ folder to merge FROM
            target_folder: Path to catalog/{exemplar}/ folder to merge INTO
        """
        exemplar_name = target_folder.name

        # 1. Find database files
        source_db_files = list(source_folder.glob("*.sqlite"))
        target_db_files = list(target_folder.glob("*.sqlite"))

        if source_db_files and target_db_files:
            source_db = source_db_files[0]
            target_db = target_db_files[0]

            try:
                # Get tables from source database
                with readonly_connection(source_db) as src_con:
                    cursor = src_con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                    source_tables = [row[0] for row in cursor.fetchall()]

                # Copy rows from each table (with deduplication via data_source)
                with sqlite3.connect(target_db) as target_con:
                    rows_merged = 0

                    for table_name in source_tables:
                        try:
                            # Get columns from source
                            with readonly_connection(source_db) as src_con:
                                cursor = src_con.execute(f"PRAGMA table_info({self._quote_identifier(table_name)})")
                                source_cols = [row[1] for row in cursor.fetchall()]

                            # Check if table exists in target
                            cursor = target_con.execute(f"PRAGMA table_info({self._quote_identifier(table_name)})")
                            target_cols = [row[1] for row in cursor.fetchall()]

                            if not target_cols:
                                continue  # Table doesn't exist in target

                            # Find common columns
                            common_cols = [c for c in source_cols if c in target_cols]
                            if not common_cols:
                                continue

                            # Copy data (try ORDER BY rowid for deterministic merge, falls back for WITHOUT ROWID)
                            cols_quoted = ", ".join(self._quote_identifier(c) for c in common_cols)
                            with readonly_connection(source_db) as src_con:
                                try:
                                    cursor = src_con.execute(
                                        f"SELECT {cols_quoted} FROM {self._quote_identifier(table_name)} ORDER BY rowid"
                                    )
                                except sqlite3.OperationalError as e:
                                    if "no such column: rowid" in str(e).lower():
                                        cursor = src_con.execute(
                                            f"SELECT {cols_quoted} FROM {self._quote_identifier(table_name)}"
                                        )
                                    else:
                                        raise
                                rows = cursor.fetchall()

                            if rows:
                                placeholders = ", ".join("?" * len(common_cols))
                                target_con.executemany(
                                    f"INSERT OR IGNORE INTO {self._quote_identifier(table_name)} "
                                    f"({cols_quoted}) VALUES ({placeholders})",
                                    rows,
                                )
                                rows_merged += len(rows)

                        except sqlite3.Error as e:
                            logger.debug(f"    Skipping table {table_name}: {e}")

                    target_con.commit()
                logger.debug(f"    Merged {rows_merged} rows into {exemplar_name}")

            except Exception as e:
                logger.warning(f"    ⚠ Failed to merge databases: {e}")

        # 2. Merge manifest files
        source_manifest = source_folder / f"{exemplar_name}_manifest.json"
        target_manifest = target_folder / f"{exemplar_name}_manifest.json"

        if source_manifest.exists() and target_manifest.exists():
            try:
                with source_manifest.open() as f:
                    src_mf = json.load(f)
                with target_manifest.open() as f:
                    tgt_mf = json.load(f)

                # Merge source_databases lists
                tgt_sources = tgt_mf.get("source_databases", [])
                src_sources = src_mf.get("source_databases", [])
                tgt_sources.extend(src_sources)
                tgt_mf["source_databases"] = tgt_sources

                # Recalculate combined_output stats
                tgt_combined = tgt_mf.get("combined_output", {})
                src_combined = src_mf.get("combined_output", {})

                for key in [
                    "total_intact_rows",
                    "total_lf_rows",
                    "total_remnant_tables",
                    "total_remnant_rows",
                    "duplicates_removed",
                ]:
                    tgt_combined[key] = tgt_combined.get(key, 0) + src_combined.get(key, 0)

                tgt_mf["combined_output"] = tgt_combined

                # Write updated manifest
                with target_manifest.open("w") as f:
                    json.dump(tgt_mf, f, indent=2)

            except Exception as e:
                logger.warning(f"    ⚠ Failed to merge manifests: {e}")

        # 3. Merge rejected subfolders
        source_lf = source_folder / "rejected"
        target_lf = target_folder / "rejected"

        if source_lf.exists():
            target_lf.mkdir(parents=True, exist_ok=True)
            for item in source_lf.iterdir():
                dest = target_lf / item.name
                if not dest.exists():
                    if item.is_file():
                        shutil.copy2(str(item), str(dest))
                    elif item.is_dir():
                        shutil.copytree(str(item), str(dest))

    def _quote_identifier(self, identifier: str) -> str:
        """Quote a SQL identifier to handle special characters."""
        return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'

    def _group_databases_for_lf_processing(self, databases_with_lf: list[dict]) -> dict:
        """
        Group databases with lost_and_found tables for processing.

        Groups databases by:
        1. exact_match (for CATALOG processing)
        2. metamatch.group_id (for MERGE processing)
        3. Individual processing (no match, for NEAREST/ORPHAN)

        Args:
            databases_with_lf: List of database records with lost_and_found tables

        Returns:
            Dict with:
                - "catalog_groups": dict mapping exemplar_name -> [records]
                - "metamatch_groups": dict mapping group_id -> {group_info, records}
                - "individual": [records] without catalog or metamatch
        """
        from collections import defaultdict

        catalog_groups = defaultdict(list)
        metamatch_groups = defaultdict(lambda: {"databases": []})
        individual = []

        for record in databases_with_lf:
            # Check for exact catalog match first (highest priority)
            decision = record.get("decision", {})
            is_matched = decision.get("matched", False)

            if is_matched:
                exact_matches = record.get("exact_matches", [])
                if exact_matches:
                    # Only group as catalog if it's a FULL schema match
                    full_match = None
                    for match in exact_matches:
                        match_type = match.get("match", "")
                        if match_type in ["tables_equal+columns_equal", "hash"]:
                            full_match = match
                            break

                    if full_match:
                        exact_match_label = full_match.get("label")
                        if exact_match_label:
                            catalog_groups[exact_match_label].append(record)
                            continue

            # Check for metamatch group - includes ALL sizes (even singletons)
            # Metamatch preserves original schema, so data is safe
            metamatch = record.get("metamatch", {})
            group_id = metamatch.get("group_id")
            group_size = metamatch.get("group_size", 0)

            if group_id:
                # Store group metadata
                if "group_label" not in metamatch_groups[group_id]:
                    metamatch_groups[group_id]["group_label"] = metamatch.get(
                        "group_label", f"metamatch_{group_id[:8]}"
                    )
                    metamatch_groups[group_id]["first_table"] = metamatch.get("first_table", "unknown")
                    metamatch_groups[group_id]["group_size"] = group_size

                metamatch_groups[group_id]["databases"].append(record)
                continue

            # No catalog match or metamatch - only route to individual (NEAREST) if has LF tables
            # NEAREST uses nearest_exemplar matching which requires compatible schemas.
            # Databases without LF have nothing to recover and stay in selected_variants.
            has_lf = record.get("meta_snapshot", {}).get("has_lost_and_found", False)
            if has_lf:
                individual.append(record)

        # Note: We intentionally do NOT route metamatch groups to NEAREST based on
        # nearest_exemplars. MERGE (metamatch) preserves original schemas and uses
        # a self-generated superrubric for LF recovery. Routing to NEAREST risks data
        # loss when schemas don't match the nearest exemplar.

        return {
            "catalog_groups": dict(catalog_groups),
            "metamatch_groups": dict(metamatch_groups),
            "individual": individual,
        }
