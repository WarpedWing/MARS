#!/usr/bin/env python3
"""
Database Carving Orchestration

Handles forensic carving of empty/failed databases (variant X) and renaming
carved output with match hints from rubric matching.

Extracted from processor.py to improve modularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mars.config.schema import CarverConfig
from mars.utils.debug_logger import logger
from mars.utils.file_utils import read_jsonl

if TYPE_CHECKING:
    from rich.console import Console

    from mars.config import MARSConfig, ProjectPaths


class DatabaseCarver:
    """Orchestrates database carving operations."""

    def __init__(
        self,
        paths: ProjectPaths,
        stats: dict,
        config: MARSConfig | None = None,
        source_type: str = "candidate",
    ):
        """
        Initialize the database carver.

        Args:
            paths: Project paths configuration
            stats: Statistics dictionary to update
            config: Configuration object (optional)
            source_type: Type of source being scanned ("candidate" or "time_machine")
        """
        self.paths = paths
        self.stats = stats
        self.config = config
        self.source_type = source_type

    def rename_carved_databases_with_match_hints(
        self,
        carved_dir: Path,
        match_hints_lookup: dict,
        console: Console | None = None,
    ):
        """
        Rename carved database files to include match hints (exact_match/metamatch labels).

        Args:
            carved_dir: Directory containing carved database output folders
            match_hints_lookup: Dict mapping source db path -> {exact_matches, metamatch}
            console: Optional Rich console for status display
        """
        if not carved_dir.exists():
            return

        logger.debug(
            f"\n  Checking carved databases for match hints ({len(match_hints_lookup)} databases with hints)..."
        )

        renamed_count = 0

        # Iterate through carved output directories
        for carved_folder in carved_dir.iterdir():
            if not carved_folder.is_dir():
                continue

            # Find .sqlite files in the folder (may be in subfolder)
            sqlite_files = list(carved_folder.rglob("*.sqlite"))

            for sqlite_file in sqlite_files:
                # Extract base filename from filename (e.g., "f123456789_carved.sqlite" -> "f123456789")
                # Pattern: {f_name}_{timestamp}_carved.sqlite or similar
                filename_parts = sqlite_file.stem.split("_")
                if not filename_parts:
                    continue

                # First part should be the base filename
                f_name = filename_parts[0]

                # Look for match hints - try multiple matching strategies
                match_hints = None
                for source_path_str, hints in match_hints_lookup.items():
                    source_path = Path(source_path_str)
                    # Try multiple matching strategies:
                    # 1. Exact match on filename stem
                    if source_path.stem == f_name:
                        match_hints = hints
                        break
                    # 2. Check if filename is in the full path
                    if f_name in str(source_path):
                        match_hints = hints
                        break
                    # 3. Check parent folder name (might be f123_hash format)
                    if source_path.parent.name.startswith(f_name):
                        match_hints = hints
                        break

                if not match_hints:
                    logger.debug(f"  No match hints found for carved DB: {sqlite_file.name} (f-name: {f_name})")
                    continue

                # Extract match label (same logic as orphans/remnants)
                exact_matches = match_hints.get("exact_matches", [])
                metamatch = match_hints.get("metamatch", {})

                match_label = None
                if exact_matches:
                    for match in exact_matches:
                        match_type = match.get("match", "")
                        if match_type in ["tables_equal+columns_equal", "hash"]:
                            match_label = match.get("label")
                            break
                    if not match_label and exact_matches:
                        match_label = exact_matches[0].get("label")
                if not match_label and metamatch:
                    match_label = metamatch.get("group_label")

                if not match_label:
                    logger.debug(f"  No usable match label for {sqlite_file.name} (has hints but no label)")
                    continue

                # Create new filename with match label
                # Format: {MatchLabel}_{original_filename}
                safe_label = match_label.replace(" ", "_").replace("/", "_")
                new_filename = f"{safe_label}_{sqlite_file.name}"
                new_path = sqlite_file.parent / new_filename

                # Rename file if name would change
                if sqlite_file != new_path and not new_path.exists():
                    try:
                        sqlite_file.rename(new_path)
                        renamed_count += 1
                        logger.debug(f"  Renamed carved DB: {sqlite_file.name} → {new_filename}")

                        # Also rename the folder to include match label
                        # Format: {MatchLabel}_{original_folder_name}
                        new_folder_name = f"{safe_label}_{carved_folder.name}"
                        new_folder_path = carved_folder.parent / new_folder_name

                        if carved_folder != new_folder_path and not new_folder_path.exists():
                            try:
                                carved_folder.rename(new_folder_path)
                                logger.debug(f"  Renamed carved folder: {carved_folder.name} → {new_folder_name}")
                                # Update reference to use new folder path for remaining files
                                carved_folder = new_folder_path
                            except Exception as e:
                                logger.debug(f"  Warning: Could not rename folder {carved_folder.name}: {e}")

                        # Break after first successful rename to avoid processing same folder multiple times
                        break

                    except Exception as e:
                        logger.debug(f"  Warning: Could not rename {sqlite_file.name}: {e}")

        if renamed_count > 0:
            logger.debug(f"  ✓ Renamed {renamed_count} carved database(s) with match hints")

    def carve_databases(self):
        """
        Carve databases marked as variant_chosen='X' (empty/failed validation).

        This is the final step in the SQLite processing pipeline:
        - Identifies databases that failed validation (variant X)
        - Uses forensic carving to extract data from deleted/corrupted pages
        - Outputs carved data to databases/carved/ directory

        Output structure:
            databases/carved/
                {filename}_carved/
                    sqlite/
                        {filename}_carved.sqlite
                    {filename}_carved_protobufs.jsonl

        Note: Carving is skipped for time_machine sources since TM databases
        are intact copies, not carved fragments that need recovery.
        """
        # Skip carving for Time Machine scans - databases are intact
        if self.source_type == "time_machine":
            logger.debug("Skipping carving for Time Machine scan (databases are intact)")
            return

        from mars.carver.batch_carver import batch_carve_databases

        logger.debug("")
        logger.separator("=", 80)
        logger.debug("STEP 5: Carving Empty/Failed Databases")
        logger.separator("=", 80)

        # Find databases to carve (variant_chosen = "X")
        databases_to_carve = []
        match_hints_lookup = {}  # Map db_path -> {exact_matches, metamatch} for naming

        if not self.paths.db_selected_variants.exists():
            logger.debug("  No selected_variants directory found, skipping carving")
            return

        # Load database records from JSONL file (one JSON object per line)
        results_jsonl = self.paths.db_selected_variants / "sqlite_scan_results.jsonl"
        if not results_jsonl.exists():
            logger.debug("  No sqlite_scan_results.jsonl found, skipping carving")
            return

        # Parse JSONL file line by line
        try:
            for record in read_jsonl(results_jsonl, filter_type="case"):
                variant_chosen = record.get("variant_chosen", "")

                # Check if this database should be carved (variant X = empty/failed)
                if variant_chosen == "X":
                    # Get the chosen variant path (original database)
                    variant_outputs = record.get("variant_outputs", {})
                    original_path = variant_outputs.get("original")

                    if original_path:
                        db_path = Path(original_path)
                        if db_path.exists():
                            databases_to_carve.append(db_path)

                            # Store match hints for later renaming
                            match_hints_lookup[str(db_path)] = {
                                "exact_matches": record.get("exact_matches", []),
                                "metamatch": record.get("metamatch", {}),
                            }

        except Exception as e:
            logger.warning(f"  Warning: Could not load sqlite_scan_results.jsonl: {e}")
            return

        if not databases_to_carve:
            logger.debug("  No databases require carving")
            return

        logger.debug(f"\n  Found {len(databases_to_carve)} corrupt database(s) to carve:")
        # Create carved output directory
        self.paths.db_carved.mkdir(parents=True, exist_ok=True)

        # Create Rich console for progress display
        from rich.console import Console

        console = Console()

        try:
            # Use CarverConfig settings instead of hardcoded values
            carver_cfg = self.config.carver if self.config else CarverConfig()
            results = batch_carve_databases(
                databases=databases_to_carve,
                output_dir=self.paths.db_carved,
                ts_start=carver_cfg.ts_start,
                ts_end=carver_cfg.ts_end,
                filter_mode=carver_cfg.filter_mode,
                enable_protobuf=carver_cfg.decode_protobuf,
                enable_csv=carver_cfg.csv_export,
                enable_parallel=carver_cfg.parallel_processing,
                parallel_threshold=carver_cfg.parallel_threshold,
                console=console,
                config=self.config,
            )

            # Update stats
            self.stats["carved_databases"] = results["success"]
            self.stats["carved_failed"] = results["failed"]

            logger.debug(f"\n  ✓ Carving complete: {results['success']}/{results['total']} succeeded")

            if results["failed"] > 0:
                logger.debug(f"    ⚠ {results['failed']} database(s) failed to carve")

            # Rename carved databases to include match hints (exact_matches/metamatch labels)
            # This may take time for many files, so show a spinner
            with console.status("[bold cyan]Organizing carved databases with match hints..."):
                self.rename_carved_databases_with_match_hints(self.paths.db_carved, match_hints_lookup, console)

        except Exception as e:
            logger.debug(f"  ✗ Carving failed: {e}")
            import traceback

            logger.debug(f"    Traceback: {traceback.format_exc()}")
