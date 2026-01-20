#!/usr/bin/env python3
"""
Raw File Processor
Orchestrates processing of recovered/carved files from PhotoRec, Scalpel, or other
file carving tools. Delegates to specialized scripts for each file type.

Usage:
    python processor.py --input-dir ./Recovered_Files --output-dir ./Processed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.lf_processor.lf_orchestrator import LFOrchestrator
from mars.pipeline.raw_scanner.database_carver import DatabaseCarver
from mars.pipeline.raw_scanner.file_categorizer import FileCategorizer
from mars.pipeline.raw_scanner.stats_reporter import CategorizationReporter
from mars.report_modules.report_module_manager import ReportModuleManager
from mars.utils.cleanup_utilities import CleanupUtilities, cleanup_sqlite_directory
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    import rich.console

    from mars.config import MARSConfig, ProjectPaths


class RawFileProcessor:
    """Orchestrator for processing recovered/carved files.

    This class delegates to specialized scripts rather than duplicating their logic:
    - SQLite processing → db_variant_selector.py (finding, fixing, classifying, moving)
    - JSONLZ4 processing → ff_jsonlz4_salvage.py (parsing, classifying, exporting CSVs)
    - Text logs → log merger
    - Plists → plist processor
    - Keychains → keychain processor
    """

    def __init__(
        self,
        input_dir: Path,
        config: MARSConfig | None = None,
        paths: ProjectPaths | None = None,
        exemplar_db_dir: Path | None = None,
        source_type: str = "candidate",
        extraction_manifest: Path | None = None,
    ):
        """
        Initialize raw file processor.

        Args:
            input_dir: Root directory containing recovered files (carved files to scan)
            config: Configuration object (optional, creates default if None)
            paths: ProjectPaths for output structure
            exemplar_db_dir: Path to exemplar scan output (for db_variant_selector)
            source_type: Type of source being scanned. Options:
                - "candidate": Carved/recovered files (default)
                - "time_machine": Files extracted from Time Machine backup
            extraction_manifest: Optional path to extraction manifest from TM scan.
                When provided, file categorization trusts the manifest's classification.
        """
        self.input_dir = Path(input_dir)
        self.exemplar_db_dir = Path(exemplar_db_dir) if exemplar_db_dir else None
        self.source_type = source_type
        self.extraction_manifest = Path(extraction_manifest) if extraction_manifest else None

        # Handle config with backward compatibility
        if config is None:
            from mars.config import MARSConfig

            config = MARSConfig()

        self.config = config

        # Validate paths
        if paths is None:
            raise ValueError("paths parameter is required")

        self.paths = paths

        # Create output directories
        self.paths.create_candidate_dirs()

        # Initialize report module manager
        self.module_manager = ReportModuleManager(
            config=self.config,
            paths=self.paths,
            console=None,  # processor doesn't have console by default
        )

        # Initialize statistics (must come before reporter and database_carver)
        self.stats = {
            # File categorization stats
            "total_files": 0,
            "classified": 0,
            "unknown": 0,
            "archives": 0,
            "wifi_plists_kept": 0,
            "asl_logs_kept": 0,
            "jsonlz4_kept": 0,
            "plists_skipped": 0,
            "plists_skipped_bytes": 0,
            "text_logs": 0,
            "sqlite_dbs": 0,
            "other": 0,
            # Database processing stats
            "sqlite_dbs_found": 0,
            "sqlite_dbs_matched": 0,
            "sqlite_dbs_matched_nonempty": 0,
            "sqlite_dbs_unmatched": 0,
            "jsonlz4_found": 0,
            "jsonlz4_processed": 0,
            "metamatch_unique_schemas": 0,
            "metamatch_singletons": 0,
            "metamatch_multi_member": 0,
            "metamatch_largest_group": 0,
            "carved_databases": 0,
            "carved_failed": 0,
        }

        # Initialize file categorizer and reporter
        self.categorizer = FileCategorizer(self)
        self.reporter = CategorizationReporter(self.stats)
        self.database_carver = DatabaseCarver(self.paths, self.stats, self.config, source_type=self.source_type)

        # Initialize cleanup utilities
        # Note: self.paths.root points to the candidate scan directory
        # (e.g., .../candidates/20251119_040458), so go up 2 levels to project root
        project_root = self.paths.root.parent.parent
        self.cleanup_utilities = CleanupUtilities(
            output_dir=project_root,
        )

        # Initialize file storage for categorized files
        self.text_logs = {}  # {LogType: [RecoveredFile, ...]}
        self.sqlite_dbs = {}  # {schema_name: [RecoveredFile, ...]}
        self.other_files = {}  # {LogType: [RecoveredFile, ...]}
        self.archives = []  # [RecoveredFile, ...]
        self.errors = []  # [{file, operation, error, timestamp}, ...]

        # Minimum confidence for fingerprinting
        self.min_confidence = 0.6

    def _load_catalog(self):
        """Load database catalog from YAML file.

        Returns:
            dict: Catalog dictionary

        Note:
            Cached after first load as self.catalog
        """
        if hasattr(self, "catalog"):
            return self.catalog

        import yaml

        # Find catalog file
        catalog_path = Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"

        if not catalog_path.exists():
            logger.warning(f"Warning: Catalog not found at {catalog_path}")
            self.catalog = {}
            return self.catalog

        with catalog_path.open() as f:
            self.catalog = yaml.safe_load(f)

        return self.catalog

    def get_sqlite_paths(self) -> list[Path]:
        """
        Get all SQLite database paths from file_categorizer.

        Returns paths to actual SQLite files (decompressed if from archives).
        """
        paths = []
        for category_files in self.sqlite_dbs.values():
            for recovered_file in category_files:
                # Use decompressed_path if available (from archives), otherwise source_path
                db_path = recovered_file.decompressed_path or recovered_file.source_path
                paths.append(db_path)
        return paths

    def process_sqlite_databases(self, richConsole: rich.console.Console | None = None):
        """
        Process SQLite databases using db_variant_selector.py.

        This delegates all SQLite processing to db_variant_selector which:
        - Receives pre-scanned SQLite list from file_categorizer (no re-scanning)
        - Fixes corrupted databases
        - Classifies by schema/rubric matching
        - Moves matched databases to paths.databases/
        """
        # Silent operation - message shown in TUI by app.py

        if not self.exemplar_db_dir:
            logger.warning("No exemplar directory provided, skipping SQLite processing")
            return

        # Import db_variant_selector
        from mars.pipeline.raw_scanner.db_variant_selector import (
            db_variant_selector,
        )

        # Set up paths
        catalog_dir = self.exemplar_db_dir / "catalog"
        schemas_dir = self.exemplar_db_dir / "schemas"
        results_path = self.paths.db_selected_variants / "sqlite_scan_results.jsonl"

        # Get pre-scanned SQLite paths from file_categorizer
        sqlite_paths = self.get_sqlite_paths()

        try:
            # Call db_variant_selector with direct parameters (no CLI)
            # Pass richConsole directly - results go to JSONL file, not stdout
            db_variant_selector.main(
                cases_dir=self.input_dir,
                exemplars_dir=catalog_dir,
                variants_dir=self.paths.db_selected_variants,
                results_path=results_path,
                config=self.config,
                schemas_dir=schemas_dir,
                use_catalog=True,
                use_rubrics=True,
                emit_nearest=True,
                richConsole=richConsole,
                sqlite_paths=sqlite_paths,
                extraction_manifest_path=self.extraction_manifest,
            )

            # Parse results
            logger.debug(f"Looking for results at: {results_path}")
            if results_path.exists():
                logger.debug("Results file exists, parsing...")
                with results_path.open() as f:
                    for line in f:
                        record = json.loads(line)
                        if record.get("type") == "header":
                            self.stats["sqlite_dbs_found"] = record.get("cases_scanned", 0)
                            logger.debug(f"Found header: cases_scanned={self.stats['sqlite_dbs_found']}")
                        elif record.get("type") == "case":
                            decision = record.get("decision", {})
                            if decision.get("matched"):
                                self.stats["sqlite_dbs_matched"] += 1
                                # Track matched databases with actual data (non-empty)
                                if not decision.get("empty"):
                                    self.stats["sqlite_dbs_matched_nonempty"] += 1
                            else:
                                self.stats["sqlite_dbs_unmatched"] += 1
            else:
                logger.warning(f"Results file does not exist: {results_path}")

            logger.debug(
                f"SQLite processing complete: {self.stats['sqlite_dbs_found']} found, "
                f"{self.stats['sqlite_dbs_matched']} matched"
            )

        except Exception as e:
            logger.error(f"Error processing SQLite databases: {e}")

    def cleanup_database_residue(self, aggressive: bool = False):
        """
        Clean up database residue: extract lost_and_found tables and remove unnecessary variants.

        This delegates to residue_processor.py which:
        - Extracts lost_and_found tables into separate databases
        - Cleans up unnecessary variant files (cloned, recovered)
        - Optionally does aggressive cleanup (removes even chosen variants for empty/unmatched)

        Args:
            aggressive: If True, delete even chosen variants for empty/unmatched databases
        """

        # Check if results file exists
        results_path = self.paths.db_selected_variants / "sqlite_scan_results.jsonl"
        if not results_path.exists():
            return

        # Import residue_processor
        from mars.pipeline.raw_scanner.db_variant_selector import (
            residue_processor,
        )

        # Call residue processor directly (not through CLI to avoid verbose output)
        residue_output = self.paths.db_selected_variants / "residue"
        residue_output.mkdir(parents=True, exist_ok=True)

        try:
            # Run residue processing silently
            residue_processor.process_results(
                results_path=results_path,
                output_root=residue_output,
                aggressive_cleanup=aggressive,
            )
        except Exception as e:
            logger.error(f"Error during residue cleanup: {e}")

    def run_metamatching(self):
        """
        Run metamatching on unclassified databases.

        Groups unmatched databases by exact schema hash and adds metamatch
        metadata to the results file. Does NOT move files - metadata only.

        Every unmatched database gets a label, even singletons (group_size: 1).
        """
        logger.debug("Running metamatching on unclassified databases...")

        # Check if results file exists
        results_path = self.paths.db_selected_variants / "sqlite_scan_results.jsonl"
        if not results_path.exists():
            return

        # Import metamatch_processor
        from mars.pipeline.raw_scanner.db_variant_selector import (
            metamatch_processor,
        )

        try:
            # Run metamatching (reads and rewrites JSONL with annotations)
            stats = metamatch_processor.process_results(results_path)

            # Update stats
            self.stats["metamatch_unique_schemas"] = stats.get("unique_schemas", 0)
            self.stats["metamatch_singletons"] = stats.get("singleton_count", 0)
            self.stats["metamatch_multi_member"] = stats.get("multi_member_count", 0)
            self.stats["metamatch_largest_group"] = stats.get("largest_group", 0)

            logger.debug(
                f"Metamatching complete: {self.stats['metamatch_unique_schemas']} unique schemas "
                f"({self.stats['metamatch_multi_member']} with 2+ members, "
                f"{self.stats['metamatch_singletons']} singletons)"
            )

        except Exception as e:
            logger.error(f"Error during metamatching: {e}")

    def cleanup_temporary_folders(self, richConsole: rich.console.Console | None = None):
        """
        Clean up temporary folders created during processing.

        Removes:
        - _temp/ directory (used for archive decompression)
        - Empty rejected/ folders (created preemptively but unused)

        Args:
            richConsole: Optional rich console for spinner display
        """

        def _do_cleanup():
            logger.debug("Starting temporary folder cleanup")

            # Remove _temp directory if it exists
            temp_dir = self.paths.root / "_temp"
            if temp_dir.exists():
                try:
                    cleanup_sqlite_directory(temp_dir)
                    logger.debug("Cleaned up _temp directory")
                except Exception as e:
                    logger.warning(f"Could not remove _temp directory: {e}")
            else:
                logger.debug("No _temp directory found")

            # Remove empty rejected folders
            logger.debug("Checking for empty rejected folders")
            self.cleanup_utilities.cleanup_empty_rejected_folders()
            logger.debug("Temporary folder cleanup completed")

        # Use spinner if console available and progress bars enabled
        show_progress = richConsole is not None and self.config.ui.show_progress_bars

        if show_progress and richConsole is not None:
            with richConsole.status("[bold cyan]Cleaning up temporary files...[/bold cyan]"):
                _do_cleanup()
        else:
            _do_cleanup()

    def process_all(self, richConsole: rich.console.Console | None = None, free_scan: bool = False):
        """
        Run all processing steps.

        This is the main entry point that orchestrates all file processing:
        0. File categorization (scan, classify, selective collection)
        1. SQLite databases (db_variant_selector.py)
        2. Metamatching (metamatch_processor.py)
        3. Database residue cleanup (residue_processor.py)
        3.5. Lost and found processing (lf_processor)
        3.75. Forensic carving of empty/failed databases (carver)
        4. Report modules (JSONLZ4, WiFi reports, etc.)

        Args:
            richConsole: Optional rich console for TUI display
            free_scan: If True, runs only free scan report modules
        """
        # Silent processing - stats shown in TUI

        # Step 0: Scan and classify files (WiFi plists, ASL logs, JSONLZ4, text logs, SQLite)
        self.categorizer.scan_and_classify(richConsole)

        # Step 1: Process SQLite databases
        self.process_sqlite_databases(richConsole)

        # Step 2: Run metamatching on unclassified databases
        self.run_metamatching()

        # Step 3: Clean up database residue (extract lost_and_found, remove variants)
        if richConsole:
            with richConsole.status("[bold cyan]Cleaning up unused files...[/bold cyan]"):
                self.cleanup_database_residue(aggressive=False)
        else:
            self.cleanup_database_residue(aggressive=False)

        # Step 3.5: Process lost_and_found tables (split and reconstruct)
        lf_orchestrator = LFOrchestrator(
            paths=self.paths,
            exemplar_db_dir=self.exemplar_db_dir,
            config=self.config,
        )
        lf_orchestrator.process_lost_and_found_tables(richConsole)

        # Step 3.75: Carve empty/failed databases (variant X)
        self.database_carver.carve_databases()

        # Step 4: Run report modules
        logger.debug("\n[STEP 4] Running report modules...")
        self._load_catalog()  # Load catalog for module target resolution

        # Update module manager console if running from TUI
        if richConsole:
            self.module_manager.console = richConsole

        module_type = "free" if free_scan else "candidate"

        self.module_manager.run_modules(
            scan_type=module_type,
            source_root=self.paths.root,
            reports_dir=self.paths.reports,
            catalog=self.catalog,
        )

        # Step 5: Clean up temporary folders
        self.cleanup_temporary_folders(richConsole)

        # Processing complete - stats available in self.stats for TUI

    def print_stats(self):
        """Get processing statistics (for TUI display).

        Note: Stats are stored in self.stats dict and shown in TUI.
        This method is kept for backward compatibility but does not print.
        """
        pass


# ============================================================================
# Main entry point
# ============================================================================


def main():
    """Main entry point with unified CLI parser."""
    # Import unified parser
    try:
        from mars.cli.args import (
            args_to_config,
            create_processor_parser,
        )
        from mars.config import ProjectPaths
    except ImportError:
        logger.error("Could not import required modules")
        sys.exit(1)

    parser = create_processor_parser()
    args = parser.parse_args()

    # Load config from args
    config = args_to_config(args)

    # Create paths
    if hasattr(args, "output_dir") and args.output_dir:
        output_dir = Path(args.output_dir)
        paths = ProjectPaths.from_existing(output_dir, config.output)
    else:
        logger.error("--output-dir is required")
        sys.exit(1)

    # Create processor
    processor = RawFileProcessor(
        input_dir=Path(args.input_dir),
        paths=paths,
        config=config,
        exemplar_db_dir=(
            Path(args.exemplar_db_dir) if hasattr(args, "exemplar_db_dir") and args.exemplar_db_dir else None
        ),
    )

    # Run processing
    processor.process_all()


if __name__ == "__main__":
    main()
