#!/usr/bin/env python3

"""
Exemplar Scanner - Phase 1 of MARS Pipeline
by WarpedWing Labs

Scans a reference macOS system (exemplar) to locate and catalog databases.
Generates schemas and rubrics from found databases to create "ground truth"
fingerprints for later recovery operations.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel

# Add src to path for direct execution
if __name__ == "__main__" or __name__.endswith(".__main__"):
    src_dir = Path(__file__).resolve().parent.parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import yaml

# Extracted scanner utilities
from mars.pipeline.exemplar_scanner.cleanup_manager import CleanupManager
from mars.pipeline.exemplar_scanner.database_processor import (
    DatabaseProcessor,
)
from mars.pipeline.exemplar_scanner.dfvfs_manager import (
    DFVFSWorkspaceManager,
)
from mars.pipeline.exemplar_scanner.directory_indexer import (
    build_user_directory_index,
)
from mars.pipeline.exemplar_scanner.exemplar_cataloger import (
    ExemplarCataloger,
)
from mars.pipeline.exemplar_scanner.manifest_processor import (
    group_manifest_by_target,
)
from mars.pipeline.exemplar_scanner.schema_generator import (
    generate_hash_lookup,
)
from mars.pipeline.exemplar_scanner.workspace_manager import (
    decompress_workspace_archives,
)
from mars.pipeline.matcher.multi_user_splitter import (
    split_and_write_rubrics,
)

# Import existing tools
from mars.pipeline.mount_utils.e01_mounter import E01Mounter
from mars.pipeline.output.structure import OutputStructure
from mars.report_modules.report_module_manager import ReportModuleManager
from mars.utils.debug_logger import logger
from mars.utils.progress_utils import create_multithreaded_progress

if TYPE_CHECKING:
    from mars.config import MARSConfig
    from mars.pipeline.mount_utils.dfvfs_exporter import ExportRecord

DEFAULT_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"


class ExemplarScanner:
    """
    Scans a reference macOS system to locate databases and generate schemas/rubrics.
    """

    def __init__(
        self,
        source_path: Path,
        output_structure: OutputStructure,
        catalog_path: Path | None = None,
        config: MARSConfig | None = None,
        console=None,
        is_forensic_image: bool = False,
        partition_labels: list[str] | None = None,
    ):
        """
        Initialize the exemplar scanner.

        Args:
            source_path: Root path of the exemplar system (e.g., /Volumes/MacintoshHD)
            output_structure: OutputStructure instance for organized output
            catalog_path: Path to database catalog YAML (defaults to built-in catalog)
            config: Configuration object (recommended). If None, uses defaults.
            console: Optional Rich Console for TUI integration (for progress display)
            is_forensic_image: True if source is E01/DD/raw image (uses dfVFS), False for mounted local directory
            partition_labels: List of partition labels to scan (for disk images). None = all partitions.
        """
        # Import config here to avoid circular imports
        if config is None:
            from mars.config import MARSConfig

            config = MARSConfig()

        self.config = config
        self.source_path = source_path
        self.output = output_structure
        self.catalog_path = catalog_path or DEFAULT_CATALOG_PATH
        self._dfvfs_manifest: dict[Path, ExportRecord] = {}
        self._dfvfs_workspace: Path | None = None
        self._original_source_path: Path | None = None
        self.debug_messages: list[str] = []
        self.console = console  # Store console for TUI progress display
        self.is_forensic_image = is_forensic_image  # Store whether to use dfVFS
        self.partition_labels = partition_labels  # Filter to specific partitions

        # Use config values
        self.max_workers = self.config.scanner.max_workers

        # Load database catalog
        self.catalog = self._load_catalog()

        # Initialize report module manager
        self.module_manager = ReportModuleManager(
            config=self.config,
            paths=self.output.paths,
            console=self.console,
        )

        # Tracking (thread-safe lists)
        self.found_databases: list[dict[str, Any]] = []
        self.failed_databases: list[dict[str, Any]] = []
        self.generated_schemas: list[Path] = []  # List of all schema files (for reference)
        self.generated_rubrics: list[Path] = []  # List of all rubric files (for reference)

        # Lock for metadata access during multi-threaded operations
        import threading

        self._metadata_lock = threading.Lock()
        self.unique_schema_dirs: set[Path] = set()  # Set of unique schema directories
        self.combined_databases: list[dict[str, Any]] = []  # Combined database tracking

        # Initialize cleanup manager
        self.cleanup_manager = CleanupManager(paths=self.output.paths)

        # Initialize exemplar cataloger (Phase 3-4 cataloging)
        self.cataloger = ExemplarCataloger(
            output=self.output,
            config=self.config,
            cleanup_manager=self.cleanup_manager,
            unique_schema_dirs=self.unique_schema_dirs,
            generated_schemas=self.generated_schemas,
            generated_rubrics=self.generated_rubrics,
            combined_databases=self.combined_databases,
        )

        # Initialize database processor
        self.database_processor = DatabaseProcessor(
            output=self.output,
            source_path=self.source_path,
            dfvfs_manifest=self._dfvfs_manifest,
            metadata_lock=self._metadata_lock,
            found_databases=self.found_databases,
            failed_databases=self.failed_databases,
        )

        logger.debug("[init] ExemplarScanner initialized")

    def _load_catalog(self) -> dict[str, Any]:
        """Load the database catalog YAML file."""
        if not self.catalog_path.exists():
            logger.warning(f"Catalog not found at {self.catalog_path}, using empty catalog")
            return {}

        with Path.open(self.catalog_path) as f:
            return yaml.safe_load(f)

    def _should_ignore_file(self, path: Path) -> bool:
        """
        Check if file should be ignored (macOS metadata and system files).

        Uses centralized config for ignore list.

        Args:
            path: Path to check

        Returns:
            True if file should be ignored, False otherwise
        """
        return self.config.should_ignore_file(path)

    def _normalize_db_definition(self, db_def: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize database definition to handle both old and new catalog formats.

        Returns normalized definition with:
            - scope: "system" or "user" (default: "system")
            - has_archives: bool (default: False)
            - primary: dict with path and glob_pattern
            - archives: list of archive locations (empty if no archives)
            - combine_strategy: how to combine archives (if applicable)
            - ... other fields from original definition
        """
        normalized = dict(db_def)  # Copy

        # Set defaults
        normalized.setdefault("scope", "system")
        normalized.setdefault("has_archives", False)

        # Handle new format (has "primary" key)
        if "primary" in db_def:
            # Already in new format
            normalized.setdefault("archives", [])
        else:
            # Old format: convert to new format
            normalized["primary"] = {
                "glob_pattern": db_def.get("glob_pattern", ""),
            }
            normalized["archives"] = []

        # Infer scope from glob_pattern if not specified
        if normalized["scope"] == "system":
            # Check if pattern contains Users/* wildcard (macOS/Linux or Windows)
            pattern = normalized["primary"]["glob_pattern"]
            if (
                pattern.startswith("Users/*/")
                or "/Users/*/" in pattern
                or "\\Users\\" in pattern
                or pattern.startswith("Users\\")
            ):
                normalized["scope"] = "user"

        return normalized

    def scan(
        self,
        categories: list[str] | None = None,  # Deprecated: use config.exemplar.enabled_catalog_groups
        custom_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Scan the exemplar system for databases.

        Args:
            categories: DEPRECATED - Use config.exemplar.enabled_catalog_groups instead.
                Group filtering is now configured via settings UI and persisted per-project.
            custom_paths: Additional custom paths to scan (glob patterns)

        Returns:
            Summary dictionary with scan results
        """
        # Note: categories parameter is deprecated, filtering now uses config.exemplar.enabled_catalog_groups
        _ = categories  # Suppress unused parameter warning
        # Silent processing - stats shown in TUI

        # Phase 1: Collect and normalize all database definitions to process
        db_jobs: list[tuple[dict[str, Any], str]] = []

        # Get enabled catalog groups from config (empty = all groups)
        enabled_groups = self.config.exemplar.enabled_catalog_groups

        # Sort for deterministic catalog iteration order
        for group_name, databases in sorted(self.catalog.items()):
            # Skip metadata entries
            if group_name in ("catalog_metadata", "skip_databases"):
                continue

            if not isinstance(databases, list):
                continue

            # Apply group filter from config settings
            # Empty list means all groups enabled
            if enabled_groups and group_name not in enabled_groups:
                logger.debug(f"[scan] Skipping group '{group_name}' (not in enabled_groups)")
                continue

            for db_def in databases:
                # Normalize to handle both old and new catalog formats
                normalized_def = self._normalize_db_definition(db_def)
                db_jobs.append((normalized_def, group_name))

        # Custom paths
        custom_jobs: list[str] = custom_paths or []

        # Check if source is a local directory or forensic image
        # Use the is_forensic_image flag from initialization (set by TUI)
        if not self.is_forensic_image:
            # Local directory - use direct filesystem access (fast)
            logger.debug(
                "[scan] Using direct filesystem access (local directory)",
            )
        else:
            # Forensic image - use dfVFS export (slower but supports E01/raw images)
            # Create dfVFS workspace manager
            dfvfs_manager = DFVFSWorkspaceManager(
                source_path=self.source_path,
                catalog_path=self.catalog_path,
                partition_labels=self.partition_labels,
                console=self.console,
                config=self.config,
            )

            # Setup workspace and export files
            workspace_dir, export_manifest = dfvfs_manager.setup_workspace(db_jobs, custom_jobs)

            # Update the manifest dict in-place so DatabaseProcessor sees the new entries
            # (reassigning would break the reference that DatabaseProcessor holds)
            self._dfvfs_manifest.clear()
            self._dfvfs_manifest.update(export_manifest)
            self._dfvfs_workspace = workspace_dir
            self._original_source_path = self.source_path
            self.source_path = workspace_dir
            # Update database_processor's source_path to point to workspace
            self.database_processor.source_path = workspace_dir

        try:
            return self._run_scan_inner(db_jobs, custom_jobs)
        finally:
            if self._original_source_path is not None:
                self.source_path = self._original_source_path
                # Restore database_processor's source_path
                self.database_processor.source_path = self._original_source_path
                self._original_source_path = None
            if self._dfvfs_workspace and self._dfvfs_workspace.exists():
                shutil.rmtree(self._dfvfs_workspace, ignore_errors=True)
            self._dfvfs_workspace = None
            self._dfvfs_manifest = {}

    def _run_scan_inner(
        self,
        db_jobs: list[tuple[dict[str, Any], str]],
        custom_jobs: list[str],
    ) -> dict[str, Any]:
        """Execute catalog scanning against the exported workspace."""

        logger.debug(
            "[PHASE 1] Discovering database files (optimized search)...",
        )
        all_db_files: list[tuple[Path, dict[str, Any], str]] = []

        seen_real_paths: set[Path] = set()
        queued_paths: set[Path] = set()

        manifest_available = bool(self._dfvfs_manifest)
        manifest_entries_by_target: dict[str, list[tuple[Path, ExportRecord]]] = {}
        if manifest_available:
            manifest_entries_by_target = group_manifest_by_target(self._dfvfs_manifest)
            logger.debug(
                f"[dfvfs] Manifest contains {len(self._dfvfs_manifest)} exported file(s)",
            )

        # SQLite auxiliary file suffixes - these are handled as companion files, not separate databases
        auxiliary_suffixes = ("-wal", "-shm", "-journal")

        # macOS firmlink prefix to normalize - both paths point to same physical file
        firmlink_prefix = "/System/Volumes/Data"

        def normalize_firmlink_path(path: str) -> str:
            """Normalize macOS firmlink path to canonical form for deduplication."""
            if path.startswith(firmlink_prefix):
                return path[len(firmlink_prefix) :]
            return path

        # Track normalized paths for firmlink deduplication
        seen_normalized_paths: set[str] = set()

        # Only build user directory index for non-dfVFS (direct filesystem access)
        user_dirs = []
        if not manifest_available:
            logger.debug("  [INIT] Building user directory index...")
            user_dirs = build_user_directory_index(self.source_path)
            if user_dirs:
                logger.debug(f"    -> Found {len(user_dirs)} user directories")
            else:
                logger.debug("    -> Users directory not found")

        import time

        for idx, (db_def, category) in enumerate(db_jobs):
            db_name = db_def.get("name", "Unknown")
            primary = db_def.get("primary", {})
            glob_pattern = primary.get("glob_pattern", "")

            if not glob_pattern:
                logger.debug(
                    f"  [{idx + 1}/{len(db_jobs)}] {db_name} - SKIPPED (no pattern)",
                )
                continue

            logger.debug(f"  [{idx + 1}/{len(db_jobs)}] {db_name}")
            # if self.console:
            #     self.console.print(
            #         f"[dim]Searching for {db_name} ({idx + 1}/{len(db_jobs)})...[/dim]"
            #     )

            handled_manifest = False
            if manifest_entries_by_target:
                manifest_matches = manifest_entries_by_target.pop(db_name, None)
                if manifest_matches:
                    # Filter out auxiliary files (WAL/SHM/journal) - they are handled as companion files
                    db_matches = [
                        (p, r)
                        for p, r in manifest_matches
                        if not any(r.virtual_path.endswith(suffix) for suffix in auxiliary_suffixes)
                    ]
                    skipped_aux = len(manifest_matches) - len(db_matches)
                    if skipped_aux > 0:
                        logger.debug(f"    [dfvfs] Skipped {skipped_aux} auxiliary file(s) (WAL/SHM/journal)")

                    if db_matches:
                        logger.debug(
                            f"    [dfvfs] Using {len(db_matches)} exported match(es)",
                        )
                        for export_path, record in db_matches:
                            if export_path in queued_paths:
                                continue
                            # Check for firmlink duplicates using normalized path
                            normalized_vpath = normalize_firmlink_path(record.virtual_path)
                            if normalized_vpath in seen_normalized_paths:
                                logger.debug(f"      -> skipped (firmlink duplicate): {record.virtual_path}")
                                continue
                            seen_normalized_paths.add(normalized_vpath)
                            queued_paths.add(export_path)
                            try:
                                real_path = export_path.resolve()
                            except (OSError, RuntimeError):
                                real_path = export_path
                            seen_real_paths.add(real_path)
                            all_db_files.append((export_path, db_def, category))
                            logger.debug(f"      -> queued {record.virtual_path}")
                        handled_manifest = True
                    elif manifest_matches:
                        # All matches were auxiliary files - treat as handled but with no databases
                        handled_manifest = True

            if handled_manifest:
                continue

            # If using dfVFS and file not in manifest, skip it (don't search workspace)
            if manifest_available:
                logger.debug(
                    "    [dfvfs] Not in exported manifest - skipping (not in source image)",
                )
                continue

            # Only do glob search for non-dfVFS (direct filesystem access)
            try:
                start = time.time()
                found_paths = []

                logger.debug(
                    f"    [GLOB] Searching glob='{glob_pattern}'",
                )

                # Try direct path optimization for patterns without ** (recursive glob)
                if glob_pattern and "**" not in glob_pattern:
                    relative = Path(glob_pattern.lstrip("/"))
                    direct_candidates = [
                        self.source_path / relative,
                        self.source_path / "Library" / relative,
                        self.source_path / "private" / "var" / relative,
                    ]

                    if db_def.get("scope") == "user" or "Users/*/" in glob_pattern:
                        logger.debug(f"    [USER] scope=user, user_dirs={len(user_dirs)}")
                        for user_dir in user_dirs:
                            # Include leading slash in replace to avoid absolute path issue
                            user_relative_path = glob_pattern.replace("/Users/*/", "")
                            user_candidate = user_dir / user_relative_path
                            logger.debug(f"    [USER] candidate: {user_candidate}")
                            direct_candidates.append(user_candidate)

                            containers_dir = user_dir / "Library" / "Containers"
                            if containers_dir.exists():
                                try:
                                    # Sort for deterministic order
                                    for container in sorted(containers_dir.iterdir()):
                                        if container.is_dir():
                                            container_path = container / "Data" / user_relative_path
                                            direct_candidates.append(container_path)
                                except (PermissionError, OSError):
                                    pass

                    for candidate_path in direct_candidates:
                        try:
                            # Check for any glob wildcard characters: *, ?, [
                            has_glob_chars = any(c in candidate_path.name for c in "*?[")
                            if has_glob_chars:
                                # Log glob search details
                                parent_exists = candidate_path.parent.exists()
                                logger.debug(
                                    f"      [GLOB] pattern={candidate_path.name!r} "
                                    f"parent={candidate_path.parent} exists={parent_exists}"
                                )
                                if parent_exists:
                                    # Sort for deterministic order
                                    matches = sorted(candidate_path.parent.glob(candidate_path.name))
                                    logger.debug(f"      [GLOB] -> {len(matches)} matches")
                                    found_paths.extend(matches)
                            elif candidate_path.exists() and candidate_path.is_file():
                                found_paths.append(candidate_path)
                        except (PermissionError, OSError) as e:
                            logger.debug(f"      [GLOB] error: {e}")
                            continue

                    if found_paths:
                        elapsed = time.time() - start
                        logger.debug(
                            f"    -> Found {len(found_paths)} matches via direct path in {elapsed:.1f}s",
                        )

                        if glob_pattern and "/" in glob_pattern:
                            parts = glob_pattern.split("/")
                            has_middle_wildcard = any("*" in part for part in parts[1:-1] if part != "**")

                            if has_middle_wildcard:
                                parent_dirs = sorted({p.parent.parent for p in found_paths})
                                initial_count = len(found_paths)
                                for parent_dir in parent_dirs:
                                    try:
                                        # Sort for deterministic order
                                        matches = sorted(parent_dir.glob(parts[-1]))
                                        found_paths.extend(matches)
                                    except (PermissionError, OSError):
                                        continue
                                added = len(found_paths) - initial_count
                                if added > 0 and not self.console:
                                    logger.debug(
                                        f"    -> Expanded glob search added {added} more matches",
                                    )

                if not found_paths and glob_pattern:
                    try:
                        # Path.glob requires relative patterns, strip leading slash
                        relative_pattern = glob_pattern.lstrip("/")
                        # Sort for deterministic order
                        matched = sorted(self.source_path.glob(relative_pattern))
                        found_paths = [p for p in matched if p.is_file()]
                    except (PermissionError, OSError, NotImplementedError) as exc:
                        logger.debug(f"    [glob] fallback error: {exc}")
                        found_paths = []

                if found_paths:
                    elapsed = time.time() - start
                    logger.debug(
                        f"    -> Found {len(found_paths)} matches via glob in {elapsed:.1f}s",
                    )

                # Deduplicate and sort for deterministic order
                found_paths = sorted(dict.fromkeys(found_paths))
                elapsed = time.time() - start

                if not found_paths:
                    logger.debug(f"    -> NO MATCHES found for {db_name} (glob={glob_pattern!r})")

                skipped_symlinks = 0
                unique_files_added = 0

                for db_path in found_paths:
                    if not db_path.is_file():
                        continue
                    try:
                        real_path = db_path.resolve()
                        if real_path in seen_real_paths:
                            skipped_symlinks += 1
                            continue
                        seen_real_paths.add(real_path)
                        all_db_files.append((db_path, db_def, category))
                        unique_files_added += 1
                    except (OSError, RuntimeError):
                        continue

                if unique_files_added > 0:
                    logger.debug(
                        f"    -> Found {unique_files_added} unique file(s) in {elapsed:.1f}s",
                    )
                    if skipped_symlinks > 0 and not self.console:
                        logger.debug(
                            f"    -> Skipped {skipped_symlinks} duplicate symlink(s)",
                        )
                else:
                    logger.debug(f"    -> No matches found ({elapsed:.1f}s)")

            except Exception as exc:
                logger.debug(f"    -> [ERROR] {exc}")

        for custom_pattern in custom_jobs:
            manifest_matches = []
            if manifest_entries_by_target:
                manifest_matches = manifest_entries_by_target.pop(custom_pattern, [])
                if not manifest_matches:
                    manifest_matches = manifest_entries_by_target.pop("custom", [])

            if manifest_matches:
                # Filter out auxiliary files (WAL/SHM/journal)
                db_matches = [
                    (p, r)
                    for p, r in manifest_matches
                    if not any(r.virtual_path.endswith(suffix) for suffix in auxiliary_suffixes)
                ]
                if db_matches:
                    logger.debug(
                        f"[CUSTOM] Using {len(db_matches)} exported match(es) for {custom_pattern}",
                    )
                    for export_path, record in db_matches:
                        if export_path in queued_paths:
                            continue
                        # Check for firmlink duplicates
                        normalized_vpath = normalize_firmlink_path(record.virtual_path)
                        if normalized_vpath in seen_normalized_paths:
                            continue
                        seen_normalized_paths.add(normalized_vpath)
                        queued_paths.add(export_path)
                        try:
                            real_path = export_path.resolve()
                        except (OSError, RuntimeError):
                            real_path = export_path
                        seen_real_paths.add(real_path)
                        db_def = {
                            "name": export_path.name,
                            "path": record.virtual_path,  # Use virtual path from dfVFS
                            "description": f"Custom pattern match: {record.virtual_path}",
                            "priority": "custom",
                            "category": "custom",
                        }
                        all_db_files.append((export_path, db_def, "custom"))
                continue

            # If using dfVFS and custom pattern not in manifest, skip it
            if manifest_available:
                logger.debug(
                    f"[CUSTOM] Pattern '{custom_pattern}' not in manifest - skipping",
                )
                continue

            # Only glob search for non-dfVFS (direct filesystem access)
            logger.debug(f"[CUSTOM] Scanning custom pattern: {custom_pattern}")
            # Sort for deterministic order
            found_paths = sorted(self.source_path.glob(custom_pattern))
            if not found_paths:
                logger.debug("  -> Not found")
                continue
            for db_path in found_paths:
                if not db_path.is_file():
                    continue
                logger.debug(f"  -> {db_path.relative_to(self.source_path)}")
                db_def = {
                    "name": db_path.name,
                    "path": str(db_path.relative_to(self.source_path)),
                    "description": f"Custom database: {db_path.name}",
                    "priority": "custom",
                    "category": "custom",
                }
                all_db_files.append((db_path, db_def, "custom"))

        if manifest_entries_by_target:
            leftover = sum(len(v) for v in manifest_entries_by_target.values())
            logger.debug(
                f"[dfvfs] {leftover} exported file(s) unclaimed by catalog; treating as custom",
            )
            # Sort for deterministic iteration order
            for target_name, entries in sorted(manifest_entries_by_target.items()):
                # Filter out auxiliary files (WAL/SHM/journal)
                db_entries = [
                    (p, r)
                    for p, r in entries
                    if not any(r.virtual_path.endswith(suffix) for suffix in auxiliary_suffixes)
                ]
                for export_path, record in db_entries:
                    if export_path in queued_paths:
                        continue
                    # Check for firmlink duplicates
                    normalized_vpath = normalize_firmlink_path(record.virtual_path)
                    if normalized_vpath in seen_normalized_paths:
                        continue
                    seen_normalized_paths.add(normalized_vpath)
                    queued_paths.add(export_path)
                    try:
                        real_path = export_path.resolve()
                    except (OSError, RuntimeError):
                        real_path = export_path
                    seen_real_paths.add(real_path)
                    db_def = {
                        "name": export_path.name,
                        "path": record.virtual_path,  # Use virtual path from dfVFS
                        "description": f"dfVFS export ({target_name})",
                        "priority": "custom",
                        "category": "custom",
                    }
                    all_db_files.append((export_path, db_def, "custom"))
                manifest_entries_by_target.pop(target_name, None)

        total_files = len(all_db_files)
        logger.debug(f"  [FOUND] {total_files} database files to process")

        if total_files == 0:
            logger.debug("[WARNING] No databases found after processing exported workspace")
            return self._generate_summary()

        logger.debug(
            f"[PHASE 2] Processing databases (parallel workers: {self.max_workers})...",
        )

        # Use provided console if available (for TUI), otherwise create new Progress
        with create_multithreaded_progress(
            "Processing Files",
            header_subtitle="Categorizing...",
            show_time="elapsed",
            console=self.console,
            config=self.config,
        ) as progress:
            task = progress.add_task("[cyan]Processing files...", total=total_files)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_database_safe, db_path, db_def, category): (db_path, db_def)
                    for db_path, db_def, category in all_db_files
                }
                for future in as_completed(futures):
                    db_path, db_def = futures[future]
                    db_name = db_def.get("name", db_path.name)
                    try:
                        # Add timeout to prevent hanging on problematic databases
                        # 60 seconds should be enough for any legitimate database
                        future.result(timeout=60)
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )
                    except TimeoutError:
                        # Log timeout errors (debug only - don't break progress bar layout)
                        logger.debug(f"Skipped {db_name}: Processing timeout (>60s)")
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )
                    except Exception as e:
                        # Log errors (debug only - race conditions produce misleading messages)
                        logger.debug(f"Skipped {db_name}: {e}")
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )

        # Database processing phase - use Live panel if progress bars enabled
        show_progress_bars = self.config.ui.show_progress_bars if self.config else True

        def _run_database_processing(panel_group=None):
            """Run database processing steps, optionally updating panel_group."""
            if panel_group:
                panel_group.renderables.append(f" • Combining {len(db_jobs)} database files with archives...")

            # Combine DBs with primary and archives (e.g., Powerlog)
            self.cataloger.combine_databases(db_jobs, panel_group)

            if panel_group:
                panel_group.renderables.append(" • Cataloging databases...")

            # Combine DBs with matching schemas
            self.cataloger.auto_combine_multiple_files(panel_group)

            # Split multi-user rubrics: shared tables → _multi, unique → user rubric
            # This prevents data duplication when carved data matches multiple users
            # Must run BEFORE hash lookup so _multi rubrics get indexed
            multi_stats = split_and_write_rubrics(
                self.output.paths.exemplar_schemas,
                self.catalog_path,
            )
            if multi_stats["multi_rubrics_created"] > 0:
                logger.debug(
                    f"Created {multi_stats['multi_rubrics_created']} _multi rubrics, "
                    f"marked {multi_stats['tables_marked']} shared tables"
                )
                # Add _multi directories to schema count
                for folder in self.output.paths.exemplar_schemas.iterdir():
                    if folder.is_dir() and folder.name.endswith("_multi"):
                        self.unique_schema_dirs.add(folder)

            # Hash table names for fast matching (includes _multi rubrics)
            generate_hash_lookup(self.output.paths.exemplar_schemas)

            # Write consolidated provenance files for logs, caches, keychains
            self.output.write_consolidated_provenance_files()

            # Clean up duplicate log files from folders with preserve_structure enabled
            # Dynamically find folders from catalog
            folders_to_clean = set()
            for db_def, _ in db_jobs:
                if db_def.get("preserve_structure") and db_def.get("file_type") == "log":
                    folder_name = db_def.get("name")
                    if folder_name:
                        folders_to_clean.add(folder_name)

            for folder_name in folders_to_clean:
                removed = self.output.cleanup_duplicate_log_files(folder_name)
                if removed > 0:
                    logger.debug(
                        f"[CLEANUP] Removed {removed} duplicate file(s) from '{folder_name}' root folder",
                    )

            # Create .logarchive from Unified Logs + UUID Text for external log parsers
            archive_path = self.output.create_logarchive()
            if archive_path:
                logger.debug(f"[LOGARCHIVE] Created {archive_path.name}")

        if show_progress_bars and self.console:
            # Use Live panel for visual feedback
            panel_group = Group("[bold dark_sea_green4]Database Processing:[/bold dark_sea_green4]\n")
            processing_panel = Panel(
                panel_group,
                title="Cataloging and Processing Databases",
                padding=(1, 1),
                style="bold deep_sky_blue1",
                border_style="deep_sky_blue3",
            )
            with Live(
                processing_panel,
                console=self.console,
                transient=True,
                refresh_per_second=10,
            ):
                _run_database_processing(panel_group)
        else:
            # Debug mode or no console - run without Live panel
            _run_database_processing(None)

        logger.debug("[PHASE 6] Running report modules...")

        # Run all active report modules for exemplar scan
        self.module_manager.run_modules(
            scan_type="exemplar",
            source_root=self.output.paths.exemplar,
            reports_dir=self.output.paths.reports,
            catalog=self.catalog,
        )

        # Save metadata after all processing completes (single write, no race conditions)
        self.output.save_metadata()

        return self._generate_summary()

    def _generate_summary(self) -> dict[str, Any]:
        """Generate scan summary and save to file."""
        # Count unique schema directories (not total schema files written)
        unique_schemas = len(self.unique_schema_dirs)

        # Count encrypted databases
        encrypted_databases = [db for db in self.found_databases if db.get("is_encrypted", False)]
        encrypted_count = len(encrypted_databases)

        summary = {
            "scan_completed": datetime.now(UTC).isoformat(),
            "source_path": str(self.source_path),
            "total_found": len(self.found_databases),
            "total_failed": len(self.failed_databases),
            "total_encrypted": encrypted_count,
            "schemas_generated": unique_schemas,  # Unique schema types, not total files
            "rubrics_generated": unique_schemas,  # Same count (1 schema = 1 rubric per type)
            "total_schema_files": len(self.generated_schemas),  # Total files written (for reference)
            "combined_databases": len(self.combined_databases),
            "found_databases": self.found_databases,
            "failed_databases": self.failed_databases,
            "encrypted_databases": encrypted_databases,
            "combined_databases_details": self.combined_databases,
        }

        # Save summary
        summary_path = self.output.get_report_path("exemplar_scan_summary")
        with Path.open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Scan summary saved to file
        return summary

    # Thread-safe wrapper for processing a database.
    def _process_database_safe(self, db_path: Path, db_def: dict[str, Any], category: str):
        self.database_processor.process_database_safe(db_path, db_def, category)

    # Scan for a database based on catalog definition.
    def _scan_database_definition(self, db_def: dict[str, Any], category: str):
        self.database_processor.scan_database_definition(db_def, category)

    # Scan a custom path pattern not in the catalog.
    def _scan_custom_path(self, glob_pattern: str):
        self.database_processor.scan_custom_path(glob_pattern)

    # Process a found database: copy, hash, generate schema/rubric.
    def _process_database(self, db_path: Path, db_def: dict[str, Any], category: str):
        self.database_processor.process_database(db_path, db_def, category)

    # Decompress .gz and .bz2 files in dfVFS workspace after export.
    def _decompress_workspace_archives(self) -> None:
        if not self._dfvfs_workspace or not self._dfvfs_workspace.exists():
            return

        # Silent operation - no debug output in TUI mode
        new_manifest_entries = decompress_workspace_archives(
            self._dfvfs_workspace,
            self._dfvfs_manifest,
            debug_callback=None,
        )

        # Update manifest with new entries
        if new_manifest_entries:
            self._dfvfs_manifest.update(new_manifest_entries)


def main():
    """Main entry point for exemplar scanner."""
    from mars.cli.args import (
        args_to_config,
        create_scanner_parser,
        handle_config_commands,
    )

    # Parse arguments using unified parser
    parser = create_scanner_parser()
    args = parser.parse_args()

    # Convert args to config
    config = args_to_config(args)

    # Handle config management commands (--show-config, --save-config)
    if handle_config_commands(args, config):
        sys.exit(0)

    # Validate that either --source or --e01-image is provided
    if not args.source and not args.e01_image:
        logger.error("Must specify either --source or --e01-image")
        parser.print_help()
        sys.exit(1)

    if args.source and args.e01_image:
        logger.error("Cannot specify both --source and --e01-image")
        sys.exit(1)

    # Validate source path if provided
    if args.source:
        if not args.source.exists():
            logger.error(f"Source path does not exist: {args.source}")
            sys.exit(1)

        if not args.source.is_dir():
            logger.error(f"Source path is not a directory: {args.source}")
            sys.exit(1)

    # Validate E01 image if provided
    if args.e01_image and not args.e01_image.exists():
        logger.error(f"E01 image not found: {args.e01_image}")
        sys.exit(1)

    # Create output structure
    # Use args.case_name if provided, otherwise default to "Exemplar"
    case_name = getattr(args, "case_name", None) or "Exemplar"
    output_dir = getattr(args, "output_dir", None)
    output = OutputStructure(base_output_dir=output_dir, case_name=case_name)
    output.create(workflow="exemplar")

    # Handle E01 mounting if needed
    raw_mount_point = None
    fs_mount_point = None
    disk_device = None
    mounter = None
    source_path = args.source

    if args.e01_image:
        logger.info(f"Mounting E01 image: {args.e01_image}")
        mounter = E01Mounter()

        try:
            raw_mount_point, fs_mount_point, disk_device = mounter.mount_e01(args.e01_image)

            if not fs_mount_point:
                logger.error("Failed to mount filesystem from E01 image")
                sys.exit(1)

            source_path = fs_mount_point
            logger.info(f"E01 mounted at: {fs_mount_point}")

        except Exception as e:
            logger.error(f"Failed to mount E01 image: {e}")
            # Cleanup any partial mounts
            if raw_mount_point and raw_mount_point.exists():
                subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)

    # Create scanner with config
    scanner = ExemplarScanner(
        source_path=source_path,
        output_structure=output,
        catalog_path=args.catalog,
        config=config,
    )

    # Run scan with automatic cleanup
    try:
        summary = scanner.scan(
            categories=args.categories,
            custom_paths=args.custom_paths,
        )

        logger.info("Exemplar scan completed successfully")
        logger.info(f"Output directory: {output.root}")
        logger.info(
            f"Summary: {len(summary['found_databases'])} databases, "
            f"{summary['schemas_generated']} schemas, "
            f"{summary['rubrics_generated']} rubrics"
        )

    except KeyboardInterrupt:
        logger.warning("Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup mounted E01 image
        if mounter and (fs_mount_point or disk_device or raw_mount_point):
            logger.info("Unmounting E01 image...")
            mounter.cleanup_mount(
                fs_mount_point=fs_mount_point,
                disk_device=disk_device,
                raw_mount_point=raw_mount_point,
                graceful=True,
            )


if __name__ == "__main__":
    main()
