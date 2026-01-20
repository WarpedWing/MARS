#!/usr/bin/env python3

"""
Output Directory Structure Management
by WarpedWing Labs

Manages the organized output directory structure for MARS forensic cases.
Ensures original filenames are preserved and provenance is tracked.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mars.config import MARSConfig, ProjectPaths
from mars.utils.database_utils import copy_database_with_auxiliary_files
from mars.utils.debug_logger import logger
from mars.utils.file_utils import (
    MKDIR_KWARGS,
    FileTimestamps,
    compute_md5_hash,
    get_file_timestamps,
)


class OutputStructure:
    """
    Manages the output directory structure for a forensic case.

    This class wraps ProjectPaths and provides backward compatibility
    for existing code while using the centralized path configuration.
    """

    def __init__(
        self,
        base_output_dir: Path | None = None,
        case_name: str = "Case",
        config: MARSConfig | None = None,
        paths: ProjectPaths | None = None,
        case_number: str | None = None,
        examiner_name: str | None = None,
        description: str | None = None,
    ):
        """
        Initialize output structure.

        Args:
            base_output_dir: Parent directory for output (defaults to ./MARS_Output)
            case_name: Name of the forensic case
            config: Configuration object (optional)
            paths: Pre-created ProjectPaths (optional - if provided, other args are ignored)
            case_number: Case/incident number for forensic documentation
            examiner_name: Name of the examiner conducting analysis
            description: Description of the exemplar scan
        """
        if paths is not None:
            # Use provided paths directly
            self.paths = paths
            self.root = paths.root
            # Extract case_name and timestamp from root directory name
            folder_name = self.root.name
            if folder_name.startswith("MARS_"):
                parts = folder_name.replace("MARS_", "", 1).rsplit("_", 2)
                if len(parts) >= 3:
                    self.case_name = parts[0]
                    self.timestamp = f"{parts[1]}_{parts[2]}"
                else:
                    self.case_name = case_name
                    self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            else:
                self.case_name = case_name
                self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        else:
            # Create new paths
            if config is None:
                config = MARSConfig()

            if base_output_dir is None:
                base_output_dir = Path.cwd() / "MARS_Output"

            self.case_name = case_name
            self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

            # Create ProjectPaths using config
            self.paths = ProjectPaths.create(
                base_dir=base_output_dir,
                output_config=config.output,
                case_name=case_name,
            )
            self.root = self.paths.root

        # Store case info
        self.case_number = case_number
        self.examiner_name = examiner_name
        self.description = description

        # Case metadata
        self.metadata = {
            "case_name": self.case_name,
            "case_number": self.case_number,
            "examiner_name": self.examiner_name,
            "description": self.description,
            "created": datetime.now(UTC).isoformat(),
            "version": "2.0",
            "databases_processed": [],
            "archives_extracted": [],
            "files_recovered": [],
            "files_failed": [],
        }

        # Lock for thread-safe metadata access
        import threading

        self._metadata_lock = threading.Lock()

    @classmethod
    def from_existing(cls, root_dir: Path) -> OutputStructure:
        """
        Load OutputStructure from an existing directory.

        This is useful for loading an existing exemplar scan to add candidate runs.

        Args:
            root_dir: Path to existing output root directory

        Returns:
            OutputStructure instance pointing to existing directory
        """
        # Load config to get directory names
        config = MARSConfig()

        # Reconstruct ProjectPaths from existing directory
        from mars.config.paths import ProjectPaths

        paths = ProjectPaths.from_existing(root_dir, config.output)

        # Create instance with the reconstructed paths
        instance = cls.__new__(cls)
        instance.paths = paths
        instance.root = root_dir

        # Extract case name and timestamp from folder name
        folder_name = root_dir.name
        # Expected format: MARS_{case_name}_{timestamp}
        if folder_name.startswith("MARS_"):
            parts = folder_name.replace("MARS_", "", 1).rsplit("_", 2)
            if len(parts) >= 3:
                instance.case_name = parts[0]
                instance.timestamp = f"{parts[1]}_{parts[2]}"
            elif len(parts) == 2:
                instance.case_name = parts[0]
                instance.timestamp = parts[1]
            else:
                instance.case_name = folder_name
                instance.timestamp = "unknown"
        else:
            instance.case_name = folder_name
            instance.timestamp = "unknown"

        # Initialize case info attributes (will be populated from metadata if available)
        instance.case_number = None
        instance.examiner_name = None
        instance.description = None

        # Try to load existing metadata
        metadata_path = instance.root / "case_metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    instance.metadata = json.load(f)
                # Extract case info from loaded metadata
                instance.case_number = instance.metadata.get("case_number")
                instance.examiner_name = instance.metadata.get("examiner_name")
                instance.description = instance.metadata.get("description")
            except json.JSONDecodeError as e:
                # Handle corrupted metadata file (multiple JSON objects written)
                logger.info(
                    f"Warning: case_metadata.json is corrupted (multiple objects): {e}",
                    file=sys.stderr,
                )
                logger.info(
                    "  Attempting to repair by using first valid JSON object...",
                    file=sys.stderr,
                )

                try:
                    with metadata_path.open() as f:
                        content = f.read()

                    # Extract first valid JSON object
                    decoder = json.JSONDecoder()
                    first_obj, end_idx = decoder.raw_decode(content)

                    # Save repaired version
                    backup_path = metadata_path.with_suffix(".json.backup")
                    metadata_path.rename(backup_path)
                    logger.info(
                        f"  Backed up corrupted file to: {backup_path.name}",
                        file=sys.stderr,
                    )

                    with metadata_path.open("w") as f:
                        json.dump(first_obj, f, indent=2)
                    logger.error("  ✓ Repaired metadata file")

                    instance.metadata = first_obj
                except Exception as repair_error:
                    logger.info(
                        f"  ✗ Could not repair metadata: {repair_error}",
                        file=sys.stderr,
                    )
                    logger.error("  Creating new metadata...")
                    instance.metadata = {
                        "case_name": instance.case_name,
                        "created": datetime.now(UTC).isoformat(),
                        "version": "2.0",
                        "databases_processed": [],
                        "archives_extracted": [],
                        "files_recovered": [],
                        "_repaired": True,
                    }
        else:
            instance.metadata = {
                "case_name": instance.case_name,
                "created": datetime.now(UTC).isoformat(),
                "version": "2.0",
                "databases_processed": [],
                "archives_extracted": [],
                "files_recovered": [],
                "files_failed": [],
            }

        # Lock for thread-safe metadata access
        import threading

        instance._metadata_lock = threading.Lock()

        return instance

    def create(self, workflow: str | None = None) -> Path:
        """
        Create the directory structure for the specified workflow.

        Args:
            workflow: Workflow type - "exemplar", "candidate", or None for all directories
                     - "exemplar": Only creates dirs needed for exemplar scanning
                     - "candidate": Only creates dirs needed for candidate processing
                     - None: Creates all directories (legacy behavior)

        Returns:
            Path to root directory
        """
        # Create directories based on workflow
        if workflow == "exemplar":
            self.paths.create_exemplar_dirs()
        elif workflow == "candidate":
            self.paths.create_candidate_dirs()
        else:
            # Default: create all directories (backward compatibility)
            self.paths.create_all()

        # Create initial case metadata file
        self.save_metadata()

        return self.root

    def cleanup_duplicate_log_files(self, folder_name: str) -> int:
        """
        Remove duplicate files from log folder root if they exist in subdirectories.

        When using preserve_structure with multiple glob patterns, files may be
        copied multiple times. This cleanup removes files from the root folder
        if an identical file exists in a subdirectory.

        Args:
            folder_name: Name of the log folder to clean up (e.g., "Unified Log (All Diagnostics)")

        Returns:
            Number of duplicate files removed
        """
        log_folder = self.paths.exemplar_logs / folder_name
        if not log_folder.exists():
            return 0

        # Get all files directly in the root folder (not in subdirectories)
        root_files = [f for f in log_folder.iterdir() if f.is_file()]

        # Build a map of filenames to their paths in subdirectories
        subdir_files: dict[str, list[Path]] = {}
        for item in log_folder.rglob("*"):
            if item.is_file() and item.parent != log_folder:
                # File is in a subdirectory, not root
                subdir_files.setdefault(item.name, []).append(item)

        removed_count = 0
        for root_file in root_files:
            # Check if this filename exists in subdirectories
            if root_file.name in subdir_files:
                root_md5 = compute_md5_hash(root_file)
                # Check if any subdirectory file has matching content
                for subdir_file in subdir_files[root_file.name]:
                    try:
                        subdir_md5 = compute_md5_hash(subdir_file)
                        if root_md5 == subdir_md5:
                            # Duplicate found - remove from root
                            root_file.unlink()
                            removed_count += 1
                            break
                    except Exception:
                        # Skip files that can't be read
                        continue

        return removed_count

    def create_logarchive(
        self,
        unified_log_folder: str = "Unified Log (All Diagnostics)",
        uuid_text_folder: str = "UUID Text",
    ) -> Path | None:
        """
        Create a .logarchive bundle from Unified Logs and UUID Text folders.

        A .logarchive is a directory structure that macOS log tools (like `log show`)
        can read directly. It requires the tracev3 files from diagnostics plus the
        UUID text files for symbol resolution.

        This function:
        1. Moves UUID Text contents into Unified Log folder
        2. Removes the empty UUID Text folder
        3. Renames the combined folder to {case_name}.logarchive

        Args:
            unified_log_folder: Name of the Unified Log folder
            uuid_text_folder: Name of the UUID Text folder

        Returns:
            Path to the created .logarchive, or None if creation failed
        """
        unified_path = self.paths.exemplar_logs / unified_log_folder
        uuid_path = self.paths.exemplar_logs / uuid_text_folder

        # Check if Unified Log folder exists
        if not unified_path.exists():
            logger.debug(f"Unified Log folder not found: {unified_path}")
            return None

        # Move UUID Text contents into Unified Log folder if it exists
        if uuid_path.exists():
            try:
                # Move each item from UUID Text into Unified Log root
                for item in uuid_path.iterdir():
                    # Skip macOS system files
                    if item.name in (".DS_Store", ".localized"):
                        item.unlink()
                        continue

                    dest = unified_path / item.name
                    if dest.exists():
                        # Skip if destination already exists (e.g., system files)
                        if item.is_file():
                            item.unlink()  # Delete the duplicate
                        continue
                    shutil.move(str(item), str(dest))

                # Remove UUID Text folder (use rmtree in case of leftover system files)
                shutil.rmtree(uuid_path, ignore_errors=True)
                logger.debug("Merged UUID Text into Unified Log folder")
            except Exception as e:
                logger.warning(f"Failed to merge UUID Text folder: {e}")
                # Continue anyway - the logarchive may still be partially useful

        # Rename to .logarchive
        archive_name = f"{self.case_name}.logarchive"
        archive_path = self.paths.exemplar_logs / archive_name

        try:
            # Remove existing archive if present (from previous run)
            if archive_path.exists():
                shutil.rmtree(archive_path)

            unified_path.rename(archive_path)
            logger.debug(f"Created logarchive: {archive_path.name}")
            return archive_path
        except Exception as e:
            logger.warning(f"Failed to create logarchive: {e}")
            return None

    def save_metadata(self) -> Path:
        """Save case metadata to JSON file using atomic write.

        Uses temp file + rename for atomic writes to prevent corruption.
        """
        metadata_path = self.root / "case_metadata.json"

        # Ensure parent directory exists
        metadata_path.parent.mkdir(**MKDIR_KWARGS)

        # Write to temp file first for atomic write
        temp_path = metadata_path.with_suffix(".json.tmp")

        try:
            with temp_path.open("w") as f:
                json.dump(self.metadata, f, indent=2)

            # Atomic rename
            temp_path.replace(metadata_path)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

        return metadata_path

    def create_candidate_run(
        self,
        timestamp: str | None = None,
        exemplar_description: str | None = None,
        candidate_description: str | None = None,
    ) -> Path:
        """
        Create timestamped candidate processing run directory.

        This creates a subdirectory within candidates/ for a specific processing run,
        allowing multiple test runs without manual cleanup.

        Args:
            timestamp: Optional timestamp string (defaults to current time in YYYYMMDD_HHMMSS format)
            exemplar_description: Description of the exemplar scan being matched against
            candidate_description: Description of this candidate scan

        Returns:
            Path to the run directory: candidates/{timestamp}/

        Example:
            >>> output = OutputStructure()
            >>> run_dir = output.create_candidate_run()
            >>> # Creates: candidates/20251115_103045/
            >>> results = run_dir / "sqlite_scan_results.jsonl"
        """
        if timestamp is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        run_dir = self.paths.candidates / timestamp
        run_dir.mkdir(**MKDIR_KWARGS)

        # Save run metadata with case info
        run_metadata = {
            "timestamp": timestamp,
            "created": datetime.now(UTC).isoformat(),
            "case_name": self.case_name,
            "case_number": self.case_number,
            "examiner_name": self.examiner_name,
            "exemplar_description": exemplar_description,
            "candidate_description": candidate_description,
            "run_type": "candidate_scan",
        }
        metadata_path = run_dir / "metadata.json"
        with Path.open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)

        return run_dir

    def get_original_db_dir(self, original_filename: str) -> Path:
        """
        Get directory for storing an original database file.
        Preserves original filename in directory name.

        Args:
            original_filename: Original name of the database file (with or without extension)

        Returns:
            Path to directory: exemplar/databases/originals/{original_filename}/
        """
        # Strip extension for directory name, but keep it for reference
        base_name = Path(original_filename).stem
        db_dir = self.paths.exemplar_originals / base_name
        db_dir.mkdir(**MKDIR_KWARGS)
        return db_dir

    def get_schema_dir(self, original_db_name: str) -> Path:
        """
        Get directory for schema and rubric files.
        Preserves original database name in directory name.

        Args:
            original_db_name: Original name of the database

        Returns:
            Path to directory: exemplar/databases/schemas/{original_db_name}/
        """
        base_name = Path(original_db_name).stem
        schema_dir = self.paths.exemplar_schemas / base_name
        schema_dir.mkdir(**MKDIR_KWARGS)
        return schema_dir

    def _handle_filename_collision(
        self,
        source_path: Path,
        dest_path: Path,
        original_filename: str,
    ) -> Path:
        """
        Handle filename collision by checking MD5 and creating unique name if needed.

        Args:
            source_path: Path to source file
            dest_path: Proposed destination path
            original_filename: Original filename for collision resolution

        Returns:
            Final destination path (either original dest_path or unique path if collision)
        """
        if not dest_path.exists():
            return dest_path

        # Check if it's the same file (by comparing MD5)
        existing_md5 = compute_md5_hash(dest_path)
        new_md5 = compute_md5_hash(source_path)

        if existing_md5 == new_md5:
            # Same file, use existing path
            return dest_path

        # Different file with same name - create unique filename
        path_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:8]
        stem = Path(original_filename).stem
        suffix = Path(original_filename).suffix
        unique_filename = f"{stem}_{path_hash}{suffix}"
        dest_path_unique = dest_path.parent / unique_filename

        return dest_path_unique

    def copy_to_originals(
        self,
        source_path: Path,
        original_filename: str | None = None,
        folder_name: str | None = None,
        subfolder: str | None = None,
        profile_subfolder: str | None = None,
    ) -> Path:
        """
        Copy a database to the originals directory, preserving its original filename.

        Args:
            source_path: Path to the source database file
            original_filename: Optional override for original filename (defaults to source_path.name)
            folder_name: Optional folder name to use (defaults to original_filename stem)
                        Use catalog name to avoid collisions (e.g., "Safari_History" vs "Chrome_History")
            subfolder: Optional subfolder within the database folder (e.g., "Archives", "Quarantine")
            profile_subfolder: Optional browser profile subfolder (e.g., "Default", "Profile 1")
                              Creates nested structure: folder_name/profile_subfolder/

        Returns:
            Path to copied file
        """
        if original_filename is None:
            original_filename = source_path.name

        if folder_name is None:
            folder_name = Path(original_filename).stem

        dest_dir = self.get_original_db_dir(folder_name)

        # Add profile subfolder if specified (for multi-profile browsers)
        if profile_subfolder:
            dest_dir = dest_dir / profile_subfolder
            dest_dir.mkdir(**MKDIR_KWARGS)

        # Add subfolder if specified (e.g., for archive files)
        if subfolder:
            dest_dir = dest_dir / subfolder
            dest_dir.mkdir(**MKDIR_KWARGS)

        dest_path = dest_dir / original_filename

        # Handle filename collisions
        original_dest = dest_path
        dest_path = self._handle_filename_collision(source_path, dest_path, original_filename)

        # Skip if same file already copied
        if dest_path == original_dest and dest_path.exists():
            return dest_path

        copy_database_with_auxiliary_files(source_path, dest_path)

        # Update metadata (thread-safe)
        with self._metadata_lock:
            self.metadata["databases_processed"].append(
                {
                    "original_filename": original_filename,
                    "source_path": str(source_path),
                    "copied_to": str(dest_path),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        # Note: Metadata saved at end of scan to avoid race conditions

        return dest_path

    def copy_to_encrypted(
        self,
        source_path: Path,
        original_filename: str | None = None,
        folder_name: str | None = None,
        subfolder: str | None = None,
        profile_subfolder: str | None = None,  # Accepted but not used for encrypted
    ) -> Path:
        """
        Copy an encrypted database to the encrypted directory.

        Args:
            source_path: Path to the encrypted database file
            original_filename: Optional override for original filename (defaults to source_path.name)
            folder_name: Optional folder name to use (defaults to original_filename stem)
            subfolder: Optional subfolder within the encrypted folder
            profile_subfolder: Accepted for API compatibility but not used (encrypted DBs not cataloged)

        Returns:
            Path to copied file
        """
        # Note: profile_subfolder is intentionally not used for encrypted databases
        # since we can't generate schemas or combine encrypted databases
        if original_filename is None:
            original_filename = source_path.name

        if folder_name is None:
            folder_name = Path(original_filename).stem

        # Create encrypted directory structure similar to originals
        dest_dir = self.paths.exemplar_encrypted / folder_name

        # Add subfolder if specified
        if subfolder:
            dest_dir = dest_dir / subfolder

        dest_dir.mkdir(**MKDIR_KWARGS)
        dest_path = dest_dir / original_filename

        # For encrypted files, skip if already exists (no need for duplicates
        # since we can't generate schemas or combine encrypted databases)
        if dest_path.exists():
            logger.debug(f"Skipping duplicate encrypted file: {original_filename}")
            return dest_path

        copy_database_with_auxiliary_files(source_path, dest_path)

        # Update metadata (thread-safe)
        with self._metadata_lock:
            self.metadata["databases_processed"].append(
                {
                    "original_filename": original_filename,
                    "source_path": str(source_path),
                    "copied_to": str(dest_path),
                    "encrypted": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        # Note: Metadata saved at end of scan to avoid race conditions

        return dest_path

    def copy_to_logs(
        self,
        source_path: Path,
        original_filename: str | None = None,
        folder_name: str | None = None,
        subfolder: str | None = None,
        profile_subfolder: str | None = None,  # Accepted for API compatibility
        preserve_structure: bool = False,
        base_dir: str | None = None,
        virtual_path: str | None = None,
        file_timestamps: FileTimestamps | None = None,
    ) -> Path:
        """
        Copy a non-SQLite file to the logs directory.

        Similar to copy_to_originals but for non-database files (logs, plist, etc.)
        that were cataloged but are not SQLite databases.

        Args:
            source_path: Path to the source file (workspace path for dfVFS exports)
            original_filename: Optional override for original filename
            folder_name: Optional folder name to use (defaults to original_filename stem)
            subfolder: Optional subfolder within the log folder
            preserve_structure: If True, preserve directory structure
            base_dir: Base directory string (e.g., "private/var/db/diagnostics")
            virtual_path: Original path in source image (e.g., "/private/var/db/diagnostics/Special/file.txt")
            file_timestamps: Optional original file timestamps (if not provided, extracted from source_path)

        Returns:
            Path to copied file
        """
        from pathlib import PurePosixPath

        if original_filename is None:
            original_filename = source_path.name

        if folder_name is None:
            folder_name = Path(original_filename).stem

        # Create logs directory structure
        if preserve_structure and base_dir:
            # Determine the path to use for structure calculation
            # Priority: virtual_path (dfVFS exports) > source_path (directory scans)
            path_for_structure = virtual_path if virtual_path else str(source_path)

            # Example: base_dir="private/var/db/diagnostics"
            #          path_for_structure="/private/var/db/diagnostics/Special/file.txt"
            #          Result: logs/Unified Log (All Diagnostics)/Special/file.txt
            try:
                # Normalize base_dir for searching
                base_normalized = base_dir.strip("/")

                # Find base_dir within the path and extract the relative portion
                # This handles both:
                # - virtual_path: "/private/var/db/diagnostics/Special/file.txt"
                # - source_path:  "/Volumes/MacHD/private/var/db/diagnostics/Special/file.txt"
                path_str = path_for_structure.replace("\\", "/")  # Normalize separators
                base_idx = path_str.find(base_normalized)

                if base_idx != -1:
                    # Extract everything after base_dir
                    after_base = path_str[base_idx + len(base_normalized) :].lstrip("/")
                    if after_base:
                        relative_path = PurePosixPath(after_base)
                        dest_dir = self.paths.exemplar_logs / folder_name / relative_path.parent
                    else:
                        # File is directly in base_dir, no subdirectory
                        dest_dir = self.paths.exemplar_logs / folder_name
                else:
                    # base_dir not found in path, fall back to flat structure
                    dest_dir = self.paths.exemplar_logs / folder_name
                    if subfolder:
                        dest_dir = dest_dir / subfolder
            except Exception:
                # Any error, fall back to flat structure
                dest_dir = self.paths.exemplar_logs / folder_name
                if subfolder:
                    dest_dir = dest_dir / subfolder
        else:
            # Flat structure (existing behavior)
            dest_dir = self.paths.exemplar_logs / folder_name
            # Add subfolder if specified
            if subfolder:
                dest_dir = dest_dir / subfolder

        dest_dir.mkdir(**MKDIR_KWARGS)
        dest_path = dest_dir / original_filename

        # Handle filename collisions
        original_dest = dest_path
        dest_path = self._handle_filename_collision(source_path, dest_path, original_filename)

        # Skip if same file already copied
        if dest_path == original_dest and dest_path.exists():
            return dest_path

        # Get original file timestamps before copy (if not provided)
        if file_timestamps is None:
            file_timestamps = get_file_timestamps(source_path)

        # For non-database files, use simple copy (no auxiliary files)
        shutil.copy2(source_path, dest_path)

        # Update metadata (different list for logs) - thread-safe
        with self._metadata_lock:
            if "logs_processed" not in self.metadata:
                self.metadata["logs_processed"] = []

            self.metadata["logs_processed"].append(
                {
                    "original_filename": original_filename,
                    "source_path": str(source_path),
                    "copied_to": str(dest_path),
                    # Original file timestamps (from source, not MARS processing time)
                    **file_timestamps.to_dict(),
                    # MARS processing timestamp
                    "processed_at": datetime.now(UTC).isoformat(),
                }
            )
        # Note: Metadata saved at end of scan to avoid race conditions

        return dest_path

    def copy_to_caches(
        self,
        source_path: Path,
        original_filename: str | None = None,
        folder_name: str | None = None,
        subfolder: str | None = None,
        profile_subfolder: str | None = None,  # Accepted for API compatibility
        file_timestamps: FileTimestamps | None = None,
    ) -> Path:
        """
        Copy a cache file to the caches directory.

        Similar to copy_to_logs but for cache files (Firefox cache2, Safari cache, etc.).
        Caches are kept separate from logs for organizational clarity.

        Args:
            source_path: Path to the source cache file
            original_filename: Optional override for original filename
            folder_name: Optional folder name to use (defaults to original_filename stem)
            subfolder: Optional subfolder within the caches folder
            file_timestamps: Optional original file timestamps (if not provided, extracted from source_path)

        Returns:
            Path to copied cache file
        """
        if original_filename is None:
            original_filename = source_path.name

        if folder_name is None:
            folder_name = Path(original_filename).stem

        # Create caches directory structure
        dest_dir = self.paths.exemplar_caches / folder_name

        # Add subfolder if specified
        if subfolder:
            dest_dir = dest_dir / subfolder

        dest_dir.mkdir(**MKDIR_KWARGS)
        dest_path = dest_dir / original_filename

        # Handle filename collisions
        original_dest = dest_path
        dest_path = self._handle_filename_collision(source_path, dest_path, original_filename)

        # Skip if same file already copied
        if dest_path == original_dest and dest_path.exists():
            return dest_path

        # Get original file timestamps before copy (if not provided)
        if file_timestamps is None:
            file_timestamps = get_file_timestamps(source_path)

        # For cache files, use simple copy (no auxiliary files)
        shutil.copy2(source_path, dest_path)

        # Update metadata (separate list for caches) - thread-safe
        with self._metadata_lock:
            if "caches_processed" not in self.metadata:
                self.metadata["caches_processed"] = []

            self.metadata["caches_processed"].append(
                {
                    "original_filename": original_filename,
                    "source_path": str(source_path),
                    "copied_to": str(dest_path),
                    # Original file timestamps (from source, not MARS processing time)
                    **file_timestamps.to_dict(),
                    # MARS processing timestamp
                    "processed_at": datetime.now(UTC).isoformat(),
                }
            )
        # Note: Metadata saved at end of scan to avoid race conditions

        return dest_path

    def copy_to_keychains(
        self,
        source_path: Path,
        original_filename: str | None = None,
        folder_name: str | None = None,
        subfolder: str | None = None,
        profile_subfolder: str | None = None,  # Accepted for API compatibility
        file_timestamps: FileTimestamps | None = None,
    ) -> Path:
        """
        Copy a keychain file to the keychains directory.

        Similar to copy_to_logs but for keychain files (.keychain, .keychain-db).
        Keychains are kept separate from databases and logs for organizational clarity.

        Args:
            source_path: Path to the source keychain file
            original_filename: Optional override for original filename
            folder_name: Optional folder name to use (defaults to original_filename stem)
            subfolder: Optional subfolder within the keychains folder
            file_timestamps: Optional original file timestamps (if not provided, extracted from source_path)

        Returns:
            Path to copied keychain file
        """
        if original_filename is None:
            original_filename = source_path.name

        if folder_name is None:
            folder_name = Path(original_filename).stem

        # Create keychains directory structure
        dest_dir = self.paths.exemplar_keychains / folder_name

        # Add subfolder if specified
        if subfolder:
            dest_dir = dest_dir / subfolder

        dest_dir.mkdir(**MKDIR_KWARGS)
        dest_path = dest_dir / original_filename

        # Handle filename collisions
        original_dest = dest_path
        dest_path = self._handle_filename_collision(source_path, dest_path, original_filename)

        # Skip if same file already copied
        if dest_path == original_dest and dest_path.exists():
            return dest_path

        # Get original file timestamps before copy (if not provided)
        if file_timestamps is None:
            file_timestamps = get_file_timestamps(source_path)

        # For keychain files, use simple copy (no auxiliary files)
        shutil.copy2(source_path, dest_path)

        # Update metadata (separate list for keychains) - thread-safe
        with self._metadata_lock:
            if "keychains_processed" not in self.metadata:
                self.metadata["keychains_processed"] = []

            self.metadata["keychains_processed"].append(
                {
                    "original_filename": original_filename,
                    "source_path": str(source_path),
                    "copied_to": str(dest_path),
                    # Original file timestamps (from source, not MARS processing time)
                    **file_timestamps.to_dict(),
                    # MARS processing timestamp
                    "processed_at": datetime.now(UTC).isoformat(),
                }
            )
        # Note: Metadata saved at end of scan to avoid race conditions

        return dest_path

    def create_provenance_file(self, db_path: Path, original_filename: str, provenance_data: dict[str, Any]) -> Path:
        """
        Create a provenance JSON file for a database.
        Preserves original filename in the provenance filename.

        Args:
            db_path: Path to the database file
            original_filename: Original name of the database
            provenance_data: Dictionary containing provenance information

        Returns:
            Path to provenance file
        """
        base_name = Path(original_filename).stem
        provenance_path = db_path.parent / f"{base_name}.provenance.json"

        # Add standard metadata
        provenance_data["original_filename"] = original_filename
        provenance_data["created"] = datetime.now(UTC).isoformat()
        provenance_data["case_name"] = self.case_name

        with Path.open(provenance_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        return provenance_path

    def get_log_path(self, log_name: str) -> Path:
        """
        Get path for a log file.

        Args:
            log_name: Name of the log (e.g., 'carver', 'recovery', 'classification')

        Returns:
            Path to log file: exemplar/logs/{log_name}.log
        """
        return self.paths.exemplar_logs / f"{log_name}.log"

    def get_report_path(self, report_name: str) -> Path:
        """
        Get path for a text report.

        Args:
            report_name: Name of the report (e.g., 'summary', 'recovery_log')

        Returns:
            Path to report file: reports/{report_name}.txt
        """
        return self.paths.reports / f"{report_name}.txt"

    def create_index_html(self, content: str) -> Path:
        """
        Create the main case report HTML file.

        Args:
            content: HTML content for the index page

        Returns:
            Path to index.html
        """
        index_path = self.root / "index.html"
        with Path.open(index_path, "w") as f:
            f.write(content)
        return index_path

    def write_consolidated_provenance_files(self) -> None:
        """
        Write consolidated provenance JSON files for logs, caches, and keychains.

        Groups files by folder and creates one _provenance.json per folder
        containing all files in that folder. Follows the DB combining pipeline pattern.

        This should be called at the end of the scan to finalize provenance tracking.
        """
        # Process logs
        if "logs_processed" in self.metadata and self.metadata["logs_processed"]:
            self._write_provenance_by_folder(self.metadata["logs_processed"], "logs")

        # Process caches
        if "caches_processed" in self.metadata and self.metadata["caches_processed"]:
            self._write_provenance_by_folder(self.metadata["caches_processed"], "caches")

        # Process keychains
        if "keychains_processed" in self.metadata and self.metadata["keychains_processed"]:
            self._write_provenance_by_folder(self.metadata["keychains_processed"], "keychains")

    def _write_provenance_by_folder(self, file_list: list[dict[str, Any]], artifact_type: str) -> None:
        """
        Write consolidated provenance files grouped by folder.

        Args:
            file_list: List of file metadata dictionaries
            artifact_type: Type of artifact (logs, caches, keychains)
        """
        from collections import defaultdict

        # Group files by parent folder
        by_folder: dict[Path, list[dict[str, Any]]] = defaultdict(list)

        for file_data in file_list:
            copied_to = Path(file_data["copied_to"])
            parent_folder = copied_to.parent
            by_folder[parent_folder].append(file_data)

        # Write one provenance file per folder
        for folder, files in by_folder.items():
            # Use folder name for provenance file: "Firefox_Cache_user_provenance.json"
            folder_name = folder.name
            provenance_path = folder / f"{folder_name}_provenance.json"

            provenance_data = {
                "artifact_type": artifact_type,
                "folder": str(folder),
                "case_name": self.case_name,
                "created": datetime.now(UTC).isoformat(),
                "file_count": len(files),
                "files": files,
            }

            with Path.open(provenance_path, "w") as f:
                json.dump(provenance_data, f, indent=2)

            # Optional: Log the creation
            # logger.info(f"  Created {provenance_path.relative_to(self.root)}")


def _find_logarchive_root(folder: Path, expected_subdir: str) -> Path | None:
    """
    Find the actual root containing logarchive data.

    For TM extractions, the structure may be nested like:
    folder/private/var/db/diagnostics/{Persist,Signpost,Special,timesync}

    This finds the deepest folder containing the expected_subdir.

    Args:
        folder: Starting folder to search
        expected_subdir: Name of expected subdirectory (e.g., "diagnostics", "uuidtext")

    Returns:
        Path to the folder containing the expected_subdir, or None if not found
    """
    # Direct match at folder level
    if (folder / expected_subdir).exists():
        return folder

    # Search nested structures (TM extraction may have private/var/db/... prefix)
    for subdir in folder.rglob(expected_subdir):
        if subdir.is_dir():
            return subdir.parent

    return None


def create_logarchive_from_logs(
    logs_dir: Path,
    archive_name: str = "system",
    unified_log_folder: str = "Unified Log (All Diagnostics)",
    uuid_text_folder: str = "UUID Text",
) -> Path | None:
    """
    Create a .logarchive bundle from Unified Logs and UUID Text folders.

    This is a standalone utility function that can create a logarchive from any
    logs directory (works for both exemplar and candidate scans).

    A .logarchive is a directory structure that macOS log tools (like `log show`)
    can read directly. It requires the tracev3 files from diagnostics plus the
    UUID text files for symbol resolution.

    This function:
    1. Finds the actual diagnostics and uuidtext directories (handling nested TM structures)
    2. Creates a flat logarchive with all content at the root level
    3. Removes the source folders after successful creation

    Args:
        logs_dir: Base logs directory (e.g., candidate_paths.logs)
        archive_name: Name for the logarchive (without .logarchive extension)
        unified_log_folder: Name of the Unified Log folder
        uuid_text_folder: Name of the UUID Text folder

    Returns:
        Path to the created .logarchive, or None if creation failed
    """
    unified_path = logs_dir / unified_log_folder
    uuid_path = logs_dir / uuid_text_folder

    # Check if Unified Log folder exists
    if not unified_path.exists():
        logger.debug(f"Unified Log folder not found: {unified_path}")
        return None

    archive_full_name = f"{archive_name}.logarchive"
    archive_path = logs_dir / archive_full_name

    try:
        # Remove existing archive if present (from previous run)
        if archive_path.exists():
            shutil.rmtree(archive_path)

        # Create the logarchive directory
        archive_path.mkdir(parents=True, exist_ok=True)

        # Find the actual diagnostics directory (may be nested from TM extraction)
        diagnostics_root = _find_logarchive_root(unified_path, "diagnostics")

        if diagnostics_root and (diagnostics_root / "diagnostics").exists():
            # Move contents of diagnostics/ to archive root (Persist, Signpost, Special, timesync)
            diagnostics_dir = diagnostics_root / "diagnostics"
            for item in diagnostics_dir.iterdir():
                if item.name in (".DS_Store", ".localized"):
                    continue
                dest = archive_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            logger.debug("Moved diagnostics content to logarchive")
        else:
            # Exemplar-style structure: contents already at unified_path root
            for item in unified_path.iterdir():
                if item.name in (".DS_Store", ".localized"):
                    continue
                dest = archive_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            logger.debug("Moved unified log content to logarchive (exemplar style)")

        # Handle UUID Text folder if it exists
        if uuid_path.exists():
            # Find the actual uuidtext directory (may be nested from TM extraction)
            uuidtext_root = _find_logarchive_root(uuid_path, "uuidtext")

            if uuidtext_root and (uuidtext_root / "uuidtext").exists():
                # Move contents of uuidtext/ to archive root (hex folders like 0A, 0B, etc.)
                uuidtext_dir = uuidtext_root / "uuidtext"
                for item in uuidtext_dir.iterdir():
                    if item.name in (".DS_Store", ".localized"):
                        continue
                    dest = archive_path / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
                logger.debug("Moved uuidtext content to logarchive")
            else:
                # Exemplar-style structure: contents already at uuid_path root
                for item in uuid_path.iterdir():
                    if item.name in (".DS_Store", ".localized"):
                        continue
                    dest = archive_path / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
                logger.debug("Moved UUID text content to logarchive (exemplar style)")

            # Remove UUID Text folder
            shutil.rmtree(uuid_path, ignore_errors=True)

        # Remove the original Unified Log folder
        shutil.rmtree(unified_path, ignore_errors=True)

        logger.debug(f"Created logarchive: {archive_path.name}")
        return archive_path

    except Exception as e:
        logger.warning(f"Failed to create logarchive: {e}")
        return None
