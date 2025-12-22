#!/usr/bin/env python3
"""
dfVFS Workspace Manager for Exemplar Scanner

Manages dfVFS export workspace for forensic image processing.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from mars.pipeline.exemplar_scanner.workspace_manager import (
    decompress_workspace_archives,
)
from mars.pipeline.mount_utils.dfvfs_exporter import DFVFSExporter

if TYPE_CHECKING:
    from collections.abc import Callable

    from mars.pipeline.mount_utils.dfvfs_exporter import ExportRecord


class DFVFSWorkspaceManager:
    """Manages dfVFS export workspace for forensic image processing."""

    def __init__(
        self,
        source_path: Path,
        catalog_path: Path,
        partition_labels: list[str] | None = None,
        debug_callback: Callable[[str], None] | None = None,
        console: Console | None = None,
        config=None,
    ):
        """
        Initialize dfVFS workspace manager.

        Args:
            source_path: Path to forensic image file
            catalog_path: Path to database catalog YAML
            partition_labels: Optional partition labels to filter
            debug_callback: Optional callback for debug messages
            console: Optional rich console for output
            config: Configuration object (optional)
        """
        self.source_path = source_path
        self.catalog_path = catalog_path
        self.partition_labels = partition_labels
        self.debug = debug_callback or (lambda msg: None)
        self.console = console
        self.config = config

        self.manifest: dict[Path, ExportRecord] = {}
        self.workspace: Path | None = None

    def _decompress_archives(self) -> None:
        """Decompress .gz and .bz2 files in workspace after export."""
        if not self.workspace or not self.workspace.exists():
            return

        new_manifest_entries = decompress_workspace_archives(
            self.workspace,
            self.manifest,
            debug_callback=None,  # Silent operation
        )

        # Update manifest with new entries
        if new_manifest_entries:
            self.manifest.update(new_manifest_entries)

    def cleanup(self) -> None:
        """Clean up temporary workspace directory."""
        if self.workspace and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace = None
        self.manifest = {}

    def setup_workspace(
        self, db_jobs: list[tuple[dict, str]], custom_jobs: list[str]
    ) -> tuple[Path, dict[Path, ExportRecord]]:
        """
        Create dfVFS export workspace and export files.

        Args:
            db_jobs: List of (db_def, category) tuples from catalog
            custom_jobs: List of custom glob patterns

        Returns:
            Tuple of (workspace_path, export_manifest)
        """
        # Create temporary workspace
        workspace_dir = Path(tempfile.mkdtemp(prefix="mars_dfvfs_export_"))

        # Get target names from catalog jobs
        target_names = {db_def.get("name", "Unknown") for db_def, _ in db_jobs}

        # Export catalog targets
        if not self.console:
            self.console = Console()

        with self.console.status("[bold dark_sea_green4]Exporting catalog files...[/bold dark_sea_green4]"):
            exporter = DFVFSExporter(
                self.source_path,
                console=self.console,
                partition_labels=self.partition_labels,
                config=self.config,
            )

        export_manifest = exporter.export_catalog(
            workspace=workspace_dir,
            target_names=target_names,
            catalog_path=self.catalog_path,
        )

        # Export custom patterns
        if custom_jobs:
            custom_manifest = exporter.export_patterns(workspace_dir, custom_jobs)
            export_manifest.update(custom_manifest)

        self.manifest = export_manifest
        self.workspace = workspace_dir

        self.debug(f"[dfvfs] Exported {len(export_manifest)} files to workspace: {workspace_dir}")

        # Report exported targets
        exported_targets = {record.target_name for record in export_manifest.values()}
        missing_targets = target_names - exported_targets

        if missing_targets:
            from mars.utils.debug_logger import logger

            logger.debug(f"[dfvfs] Missing targets: {', '.join(sorted(missing_targets))}")
            self.debug(f"[dfvfs] Missing targets: {', '.join(sorted(missing_targets))}")

        if not export_manifest:
            self.debug("[dfvfs] No catalog matches were exported. Please verify the catalog patterns.")

        # Decompress archived files in workspace
        self._decompress_archives()

        return workspace_dir, export_manifest
