#!/usr/bin/env python3
"""
Manifest Processor for Exemplar Scanner

Processes dfVFS manifest entries for efficient database lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mars.pipeline.exemplar_scanner.workspace_manager import ExportRecord


def group_manifest_by_target(
    manifest: dict[Path, ExportRecord],
) -> dict[str, list[tuple[Path, ExportRecord]]]:
    """
    Group manifest entries by target database name for fast lookup.

    Args:
        manifest: dfVFS export manifest mapping export paths to records

    Returns:
        Dictionary mapping target names to list of (export_path, record) tuples,
        sorted for deterministic iteration
    """
    entries_by_target: dict[str, list[tuple[Path, ExportRecord]]] = {}

    # Sort for deterministic iteration order
    for export_path, record in sorted(manifest.items()):
        entries_by_target.setdefault(record.target_name, []).append((export_path, record))

    return entries_by_target
