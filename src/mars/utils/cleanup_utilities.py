#!/usr/bin/env python3
"""Cleanup utilities for temporary folders.

This module handles cleanup operations for the exemplar/candidate pipeline.
"""

from __future__ import annotations

import contextlib
import gc
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mars.utils.debug_logger import logger
from mars.utils.platform_utils import is_windows


def _cleanup_wal_files(db_path: Path) -> None:
    """Delete WAL and SHM files if they exist (Windows compatibility)."""
    for suffix in ["-wal", "-shm"]:
        wal_path = Path(str(db_path) + suffix)
        if wal_path.exists():
            with contextlib.suppress(Exception):
                wal_path.unlink()


def _release_sqlite_handle(db_file: Path) -> bool:
    """
    Release SQLite handle by opening and immediately closing connection.

    For deletion purposes, we don't need full WAL checkpoint - just need to
    ensure any stale handles are released. This is much faster than checkpoint.

    Returns True if successful, False otherwise.
    """
    try:
        # Short timeout - if locked by another process, skip it
        conn = sqlite3.connect(str(db_file), timeout=0.5)
        conn.close()
        del conn
        _cleanup_wal_files(db_file)
        return True
    except Exception:
        return False


def _cleanup_sqlite_directory_defensive(directory: Path, max_attempts: int = 5) -> None:
    """
    Defensive cleanup for Windows - handles SQLite WAL files and locked handles.

    Optimized for large directories (1500+ files):
    1. Try direct deletion first (often works without WAL cleanup)
    2. gc.collect() + second attempt before expensive WAL processing
    3. Parallel WAL cleanup only if needed
    4. Shorter retry delays
    """
    # First attempt: try direct deletion without any preparation
    # This often succeeds and is the fastest path
    # No logging - this is optimistic, failure is expected
    try:
        shutil.rmtree(directory)
        return
    except PermissionError:
        pass  # Expected - need further cleanup

    # Force garbage collection to release any Python-held connections
    gc.collect()

    # Second attempt after gc - often enough on its own
    try:
        shutil.rmtree(directory)
        return
    except PermissionError:
        pass  # Need WAL cleanup

    # Only process SQLite files that actually have WAL files
    db_files_with_wal = [db for db in directory.rglob("*.sqlite") if Path(str(db) + "-wal").exists()]

    if db_files_with_wal:
        logger.debug(f"[cleanup] Processing {len(db_files_with_wal)} SQLite file(s) with WAL in parallel")

        # Use thread pool for parallel WAL cleanup (I/O-bound)
        # Limit workers to avoid overwhelming the system
        max_workers = min(16, len(db_files_with_wal))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_release_sqlite_handle, db): db for db in db_files_with_wal}
            for future in as_completed(futures):
                pass  # Just wait for completion

        gc.collect()
        time.sleep(0.05)  # Brief delay for OS handle release

    # Retry rmtree with shorter delays
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(directory)
            return
        except PermissionError as e:
            last_error = e
            if attempt < max_attempts - 1:
                gc.collect()
                # Shorter delays: 0.05s, 0.1s, 0.15s, 0.2s
                time.sleep(0.05 * (attempt + 1))

    # If we get here, all attempts failed
    if last_error:
        raise last_error


def cleanup_sqlite_directory(directory: Path, max_attempts: int = 5) -> None:
    """
    Clean up SQLite WAL files and remove a directory.

    On macOS/Linux: Uses fast path (direct shutil.rmtree) since file locks
    are rarely an issue. Falls back to defensive cleanup if needed.

    On Windows: Uses defensive cleanup to handle SQLite WAL files and
    ensure connections are properly released before deletion.

    Args:
        directory: Directory to clean up and remove
        max_attempts: Maximum retry attempts for removal (default 5)

    Raises:
        PermissionError: If removal fails after all cleanup attempts
    """
    if not directory.exists():
        return

    if is_windows():
        # Windows needs defensive cleanup for SQLite file handles
        _cleanup_sqlite_directory_defensive(directory, max_attempts)
    else:
        # Fast path for macOS/Linux - try direct deletion first
        try:
            shutil.rmtree(directory)
            return
        except PermissionError:
            # Rare case: file locked on non-Windows, fall back to defensive cleanup
            logger.debug(f"[cleanup] Fast path failed for {directory}, using defensive cleanup")
            _cleanup_sqlite_directory_defensive(directory, max_attempts)


class CleanupUtilities:
    """Cleanup utilities for temporary folders and empty directories."""

    def __init__(
        self,
        output_dir: Path,
    ):
        """
        Initialize cleanup utilities.

        Args:
            output_dir: Base output directory (project root)
        """
        self.output_dir = output_dir

    def cleanup_empty_rejected_folders(self):
        """
        Remove empty rejected folders from catalog and candidate databases.

        Rejected folders are created preemptively to store rejected rows during
        validation, but if no rows are rejected, the folders remain empty and should
        be removed to avoid clutter.

        Searches recursively in:
        - candidates/*/databases/catalog/*/rejected/
        - candidates/*/databases/empty/*/rejected/
        - candidates/*/databases/metamatches/*/rejected/
        - exemplar/databases/catalog/*/rejected/
        """
        removed_count = 0

        # Search both candidate and catalog directories
        search_bases = []

        # Add candidate directories (catalog, empty, and metamatches)
        candidates_dir = self.output_dir / "candidates"
        if candidates_dir.exists():
            for candidate_folder in candidates_dir.iterdir():
                if candidate_folder.is_dir():
                    # Check catalog, empty, and metamatches subdirectories
                    for subdir in ["catalog", "empty", "metamatches"]:
                        db_path = candidate_folder / "databases" / subdir
                        if db_path.exists():
                            search_bases.append(db_path)

        # Add exemplar catalog directory
        exemplar_catalog = self.output_dir / "exemplar" / "databases" / "catalog"
        if exemplar_catalog.exists():
            search_bases.append(exemplar_catalog)

        # Scan each base directory for rejected folders
        for base_dir in search_bases:
            try:
                # Find all rejected directories
                for rejected_folder in base_dir.glob("*/rejected"):
                    if not rejected_folder.is_dir():
                        continue

                    # Check if folder is empty (ignore .DS_Store)
                    try:
                        remaining_files = [
                            f for f in rejected_folder.iterdir() if f.is_file() and f.name != ".DS_Store"
                        ]
                        # Check for subdirectories
                        subdirs = [d for d in rejected_folder.iterdir() if d.is_dir()]

                        if not remaining_files and not subdirs:
                            # Empty rejected folder - remove .DS_Store first, then folder
                            for f in rejected_folder.iterdir():
                                if f.name == ".DS_Store":
                                    f.unlink()
                            rejected_folder.rmdir()
                            removed_count += 1
                            logger.debug(
                                f"  [cyan][→][/cyan] Removed empty rejected: {rejected_folder.relative_to(self.output_dir)}"
                            )
                    except Exception as e:
                        logger.warning(f"Warning: Could not remove {rejected_folder.relative_to(self.output_dir)}: {e}")

            except Exception as e:
                logger.warning(f"Warning: Cleanup failed for {base_dir.relative_to(self.output_dir)}: {e}")

        # Always log completion with count (both to log and print for visibility)
        if removed_count > 0:
            msg = f"Cleanup: Removed {removed_count} empty rejected folder(s)"
            logger.debug(msg)
        else:
            msg = "Cleanup: No empty rejected folders found"
            logger.debug(msg)
