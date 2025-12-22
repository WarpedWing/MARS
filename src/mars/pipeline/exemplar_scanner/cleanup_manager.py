#!/usr/bin/env python3
"""
Cleanup Manager for Exemplar Scanner

Handles cleanup operations after database processing, such as removing empty
folders from the originals directory after cataloging.

Extracted from exemplar_processor.py to improve modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mars.config import ProjectPaths


class CleanupManager:
    """Manages cleanup operations for the exemplar scanner."""

    def __init__(self, paths: ProjectPaths):
        """
        Initialize the cleanup manager.

        Args:
            paths: Project paths configuration (provides access to output directories)
        """
        self.paths = paths

    def _is_empty_tree(self, folder: Path) -> bool:
        """
        Check if a folder contains only empty directories (recursively).

        Returns True if the folder is empty or contains only empty subdirectories.
        Returns False if any file exists anywhere in the tree.

        Args:
            folder: Directory to check

        Returns:
            True if folder tree contains no files
        """
        if not folder.is_dir():
            return False

        for item in folder.iterdir():
            if item.is_file():
                return False
            if item.is_dir() and not self._is_empty_tree(item):
                return False

        return True

    def _remove_empty_tree(self, folder: Path) -> bool:
        """
        Remove a folder and all its empty subdirectories.

        Args:
            folder: Directory to remove

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Remove subdirectories first (depth-first)
            for item in folder.iterdir():
                if item.is_dir():
                    self._remove_empty_tree(item)
            folder.rmdir()
            return True
        except OSError:
            return False

    def cleanup_empty_originals_folders(self) -> int:
        """
        Remove empty folders from databases/originals/ after cataloging.

        After Phase 4, single databases have been moved to catalog/, leaving
        empty folders in originals/. This cleans them up, including nested
        empty profile folders (e.g., Chrome_user/Default/).

        Returns:
            Number of top-level folders removed
        """
        if not self.paths.exemplar_originals.exists():
            return 0

        removed_count = 0

        # Sort for deterministic order
        for db_folder in sorted(self.paths.exemplar_originals.iterdir()):
            if not db_folder.is_dir():
                continue

            # Check if folder tree is completely empty (no files anywhere)
            if self._is_empty_tree(db_folder) and self._remove_empty_tree(db_folder):
                removed_count += 1

        return removed_count
