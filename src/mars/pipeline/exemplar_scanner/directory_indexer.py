#!/usr/bin/env python3
"""
Directory Indexer for Exemplar Scanner

Builds index of user directories for database discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def build_user_directory_index(source_path: Path) -> list[Path]:
    """
    Build index of user home directories for database search.

    Finds all user directories under /Users/ (excluding Shared and Guest).
    Only needed for direct filesystem access (not dfVFS exports).

    Args:
        source_path: Root path of filesystem to scan

    Returns:
        List of user directory paths, sorted for deterministic order
    """
    users_root = source_path / "Users"
    if not users_root.exists():
        return []

    # Sort for deterministic order
    user_dirs = sorted([d for d in users_root.glob("*") if d.is_dir() and d.name not in ("Shared", "Guest")])

    return user_dirs
