#!/usr/bin/env python3
"""Database discovery utilities for exemplar scanner.

Handles file discovery based on catalog definitions and custom glob patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def scan_database_definition(
    db_def: dict[str, Any],
    category: str,
    source_path: Path,
    process_callback: Callable[[Path, dict[str, Any], str], None],
    failed_databases: list[dict[str, Any]],
) -> None:
    """Scan for a database based on catalog definition.

    Args:
        db_def: Database definition from catalog
        category: Category name
        source_path: Root path to scan
        process_callback: Callback to process found databases
        failed_databases: List to append failures to
    """
    db_name = db_def.get("name", "Unknown")
    glob_pattern = db_def.get("glob_pattern")

    if not glob_pattern:
        # Skip - no pattern defined
        return

    # Search for databases matching the pattern
    # Sort for deterministic order
    found_paths = sorted(source_path.glob(glob_pattern))

    if not found_paths:
        # Not found - this is normal for optional databases
        return

    for db_path in found_paths:
        if not db_path.is_file():
            continue

        try:
            process_callback(db_path, db_def, category)
        except Exception as e:
            failed_databases.append(
                {
                    "name": db_name,
                    "path": str(db_path),
                    "category": category,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )


def scan_custom_path(
    glob_pattern: str,
    source_path: Path,
    process_callback: Callable[[Path, dict[str, Any], str], None],
    failed_databases: list[dict[str, Any]],
) -> None:
    """Scan a custom path pattern not in the catalog.

    Args:
        glob_pattern: Glob pattern to search for
        source_path: Root path to scan
        process_callback: Callback to process found databases
        failed_databases: List to append failures to
    """
    # Sort for deterministic order
    found_paths = sorted(source_path.glob(glob_pattern))

    if not found_paths:
        # Not found - this is normal
        return

    for db_path in found_paths:
        if not db_path.is_file():
            continue

        db_def = {
            "name": db_path.name,
            "path": str(db_path.relative_to(source_path)),
            "description": f"Custom database: {db_path.name}",
            "priority": "custom",
            "category": "custom",
        }

        try:
            process_callback(db_path, db_def, "custom")
        except Exception as e:
            failed_databases.append(
                {
                    "name": db_path.name,
                    "path": str(db_path),
                    "category": "custom",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
