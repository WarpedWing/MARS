#!/usr/bin/env python3
"""
File Type Classification and Categorization

Handles:
- Scanning carved output directories
- Archive decompression triage
- Fingerprinting and classifying files
- Organizing by artifact type
"""

import multiprocessing
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
)

if TYPE_CHECKING:
    from mars.pipeline.raw_scanner.candidate_orchestrator import (
        RawFileProcessor,
    )

# Import fingerprinter
from mars.pipeline.fingerprinter.text_fingerprinter import (
    LogType,
    detect_specific_wifi_plist_type,
    identify_log_type,
)
from mars.pipeline.raw_scanner.catalog_wifi_mapper import (
    CatalogWifiMapper,
)
from mars.utils.compression_utils import (
    decompress_file_with_recovery,
    get_compression_type,
    read_compressed_with_recovery,
)
from mars.utils.database_utils import is_encrypted_database
from mars.utils.debug_logger import logger
from mars.utils.file_utils import compute_md5_hash

# SQLite magic bytes
SQLITE_MAGIC = b"SQLite format 3\x00"

# Progress update throttling constants (reduce lock contention on Progress object)
_PROGRESS_WORKER_INTERVAL = 10  # Update worker bar every N files
_PROGRESS_OVERALL_INTERVAL = 25  # Update overall bar + timer every N files


class RecoveredFile:
    """Metadata for a recovered/classified file."""

    def __init__(
        self,
        source_path: Path,
        file_type: LogType,
        confidence: float,
        size: int,
        md5: str | None = None,
        first_timestamp: str | None = None,
        last_timestamp: str | None = None,
        decompressed_path: Path | None = None,
        reasons: list[str] | None = None,
        # Original file timestamps (from source, not MARS processing time)
        file_created: datetime | None = None,
        file_modified: datetime | None = None,
        file_accessed: datetime | None = None,
    ):
        self.source_path = source_path
        self.file_type = file_type
        self.confidence = confidence
        self.size = size
        self.md5 = md5
        self.first_timestamp = first_timestamp
        self.last_timestamp = last_timestamp
        self.decompressed_path = decompressed_path
        self.reasons = reasons or []
        # Original file timestamps
        self.file_created = file_created
        self.file_modified = file_modified
        self.file_accessed = file_accessed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_path": str(self.source_path),
            "file_type": self.file_type.value,
            "confidence": self.confidence,
            "size": self.size,
            "md5": self.md5,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "decompressed_path": (str(self.decompressed_path) if self.decompressed_path else None),
            "reasons": self.reasons,
            # Original file timestamps (from source, not MARS processing time)
            "file_created": self.file_created.isoformat() if self.file_created else None,
            "file_modified": self.file_modified.isoformat() if self.file_modified else None,
            "file_accessed": self.file_accessed.isoformat() if self.file_accessed else None,
        }


class FileCategorizer:
    """Categorize and classify recovered files."""

    def __init__(self, processor: "RawFileProcessor"):
        """
        Initialize file categorizer.

        Args:
            processor: Parent RawFileProcessor instance
        """
        self.processor = processor

        # Initialize catalog-based WiFi folder mapper
        catalog_path = Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"
        self.wifi_mapper = CatalogWifiMapper(catalog_path)

        # Load extraction manifest if available (for Time Machine scans)
        self.extraction_manifest: dict[str, dict] | None = None
        if processor.extraction_manifest and processor.extraction_manifest.exists():
            from mars.pipeline.raw_scanner.tm_extractor import load_extraction_manifest

            self.extraction_manifest = load_extraction_manifest(processor.extraction_manifest)
            if self.extraction_manifest:
                logger.debug(f"Loaded extraction manifest with {len(self.extraction_manifest)} entries")

    def _should_ignore_file(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored.

        For Time Machine scans with extraction manifest, only filter macOS metadata
        files, not extensions (to include message attachments like images/videos).

        For regular candidate scans, use full config-based filtering.
        """
        name = file_path.name

        # Always filter macOS metadata files
        if name in self.processor.config.scanner.ignore_files:
            return True
        if any(name.startswith(prefix) for prefix in self.processor.config.scanner.ignore_prefixes):
            return True

        # For TM scans with manifest, DON'T filter by extension
        # This allows message attachments (images, videos) to be processed
        if self.processor.source_type == "time_machine" and self.extraction_manifest:
            return False

        # For regular candidate scans, filter by extension
        return file_path.suffix.lower() in self.processor.config.scanner.ignore_extensions

    def _classify_from_manifest(
        self,
        file_path: Path,
        size: int,
        file_created: datetime,
        file_modified: datetime,
        file_accessed: datetime,
    ) -> RecoveredFile | None:
        """
        Classify a file using the extraction manifest.

        For Time Machine scans, the manifest contains ARC-based classification
        that we trust instead of fingerprinting.

        Args:
            file_path: Path to file
            size: File size in bytes
            file_created: File creation timestamp
            file_modified: File modification timestamp
            file_accessed: File access timestamp

        Returns:
            RecoveredFile if file found in manifest, None otherwise
        """
        if not self.extraction_manifest:
            return None

        # Look up file in manifest by relative path
        try:
            rel_path = str(file_path.relative_to(self.processor.input_dir))
        except ValueError:
            return None

        manifest_entry = self.extraction_manifest.get(rel_path)
        if not manifest_entry:
            return None

        artifact_type = manifest_entry.get("artifact_type", "")
        artifact_name = manifest_entry.get("artifact_name", "")
        backup_id = manifest_entry.get("backup_id", "")

        # Map artifact_type to LogType
        if artifact_type == "database":
            log_type = LogType.SQLITE
        elif artifact_type == "log":
            log_type = LogType.TM_LOG
        elif artifact_type == "cache":
            log_type = LogType.TM_CACHE
        elif artifact_type == "keychain":
            log_type = LogType.TM_KEYCHAIN
        else:
            return None  # Unknown type, fall back to fingerprinting

        # Compute MD5 for databases (needed for matching)
        md5 = None
        if log_type == LogType.SQLITE:
            md5 = compute_md5_hash(file_path)

        return RecoveredFile(
            source_path=file_path,
            file_type=log_type,
            confidence=1.0,
            size=size,
            md5=md5,
            reasons=[f"TM manifest: {artifact_name} from {backup_id}"],
            file_created=file_created,
            file_modified=file_modified,
            file_accessed=file_accessed,
        )

    def _write_tm_provenance(self, source_path: Path, dest_path: Path) -> None:
        """
        Write provenance data for a Time Machine file.

        Looks up the file in the extraction manifest and writes a JSON
        provenance file alongside the copied file.

        Args:
            source_path: Original file path in tm_extracted
            dest_path: Destination path where file was copied
        """
        if not self.extraction_manifest:
            return

        try:
            rel_path = str(source_path.relative_to(self.processor.input_dir))
            manifest_entry = self.extraction_manifest.get(rel_path)
            if not manifest_entry:
                return

            provenance_path = dest_path.parent / f"{dest_path.stem}_provenance.json"
            provenance_data = {
                "artifact_name": manifest_entry.get("artifact_name"),
                "artifact_type": manifest_entry.get("artifact_type"),
                "backup_id": manifest_entry.get("backup_id"),
                "backup_date": manifest_entry.get("backup_date"),
                "original_path": manifest_entry.get("original_path"),
                "source_path": manifest_entry.get("source_path"),
                "file_timestamps": manifest_entry.get("file_timestamps"),
                "copied_to": str(dest_path),
            }
            import json

            with provenance_path.open("w") as f:
                json.dump(provenance_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write provenance for {source_path.name}: {e}")

    def iter_input_files(self) -> list[Path]:
        """
        Find all files recursively in the input directory.

        No assumptions about folder structure - works with any organization:
        - Type-organized (sqlite/, gz/, etc.)
        - Raw PhotoRec output (recup_dir.*)
        - Custom user folder structure
        - Mixed/nested directories

        Returns:
            List of file paths to process
        """
        # Recursively find ALL files, regardless of folder structure
        # Filter out macOS metadata and system files using config
        # For TM scans, skip extension filtering to include attachments (images, videos)
        all_files = [f for f in self.processor.input_dir.rglob("*") if f.is_file() and not self._should_ignore_file(f)]

        if not all_files:
            logger.warning(f"No files found in {self.processor.input_dir}")
            return []

        return all_files

    def classify_file(self, file_path: Path) -> RecoveredFile | None:
        """
        Classify a single file using fingerprinting.

        For Time Machine scans with extraction manifest, trusts the ARC catalog's
        classification instead of fingerprinting.

        Args:
            file_path: Path to file to classify

        Returns:
            RecoveredFile object or None if classification failed
        """
        try:
            # Get file stats once for size and timestamps
            file_stat = file_path.stat()
            size = file_stat.st_size

            # Capture original file timestamps (from source, not MARS processing time)
            birth_ts = getattr(file_stat, "st_birthtime", file_stat.st_ctime)
            file_created = datetime.fromtimestamp(birth_ts, UTC)
            file_modified = datetime.fromtimestamp(file_stat.st_mtime, UTC)
            file_accessed = datetime.fromtimestamp(file_stat.st_atime, UTC)

            # Check extraction manifest first (for Time Machine scans)
            if self.extraction_manifest:
                manifest_result = self._classify_from_manifest(
                    file_path, size, file_created, file_modified, file_accessed
                )
                if manifest_result is not None:
                    return manifest_result

            # Check file extension to determine type
            file_lower = file_path.name.lower()
            compression_type = None

            # Fast-path: .sqlite files - trust extension, skip header check
            if file_lower.endswith(".sqlite"):
                md5 = compute_md5_hash(file_path)
                return RecoveredFile(
                    source_path=file_path,
                    file_type=LogType.SQLITE,
                    confidence=1.0,
                    size=size,
                    md5=md5,
                    reasons=["SQLite extension detected"],
                    file_created=file_created,
                    file_modified=file_modified,
                    file_accessed=file_accessed,
                )

            # Fast-path: .jsonlz4 files - trust extension, skip header check
            if file_lower.endswith(".jsonlz4"):
                md5 = compute_md5_hash(file_path)
                return RecoveredFile(
                    source_path=file_path,
                    file_type=LogType.JSONLZ4,
                    confidence=1.0,
                    size=size,
                    md5=md5,
                    reasons=["JSONLZ4 extension detected"],
                    file_created=file_created,
                    file_modified=file_modified,
                    file_accessed=file_accessed,
                )

            # Skip .html.gz files (too numerous and not useful)
            if file_lower.endswith(".html.gz"):
                return None

            # Detect compression type from extension
            compression_type = get_compression_type(file_path)

            # Decompress compressed archives inline
            if compression_type:
                # Skip very large compressed files (>500MB)
                if size > 500 * 1024 * 1024:
                    return None

                # Decompress with compression type hint to skip magic byte validation
                decompressed_header = read_compressed_with_recovery(
                    file_path,
                    size=8192,
                    try_recovery=True,
                    compression_type=compression_type,
                )

                if not decompressed_header:
                    # Failed to decompress even with recovery
                    self.processor.stats["archives_failed"] = self.processor.stats.get("archives_failed", 0) + 1
                    return None

                # Classify and handle the decompressed content
                return self._handle_compressed_file(
                    file_path, decompressed_header, compression_type, size, file_created, file_modified, file_accessed
                )

            # Read file header once for all checks
            with Path.open(file_path, "rb") as f:
                header = f.read(8192)

            # Fallback: Check magic bytes for gzip/bzip2 (corrupted extension or no extension)
            if header[:2] == b"\x1f\x8b":
                compression_type = "gzip"
            elif header[:2] == b"BZ":
                compression_type = "bzip2"

            # If magic bytes indicate compression, decompress inline
            if compression_type:
                decompressed_header = read_compressed_with_recovery(file_path, size=8192, try_recovery=True)

                if not decompressed_header:
                    self.processor.stats["archives_failed"] = self.processor.stats.get("archives_failed", 0) + 1
                    return None

                return self._handle_compressed_file(
                    file_path, decompressed_header, compression_type, size, file_created, file_modified, file_accessed
                )

            # Check for SQLite magic
            if header.startswith(SQLITE_MAGIC):
                md5 = compute_md5_hash(file_path)
                return RecoveredFile(
                    source_path=file_path,
                    file_type=LogType.SQLITE,
                    confidence=1.0,
                    size=size,
                    md5=md5,
                    reasons=["SQLite magic bytes detected"],
                    file_created=file_created,
                    file_modified=file_modified,
                    file_accessed=file_accessed,
                )

            # Try text log fingerprinting (pass pre-read header to avoid re-reading)
            result = identify_log_type(file_path, min_confidence=self.processor.min_confidence, header=header)

            if result.log_type == LogType.UNKNOWN:
                return None

            # Compute hash if high confidence
            md5 = compute_md5_hash(file_path) if result.confidence >= 0.8 else None

            return RecoveredFile(
                source_path=file_path,
                file_type=result.log_type,
                confidence=result.confidence,
                size=size,
                md5=md5,
                first_timestamp=result.first_timestamp,
                last_timestamp=result.last_timestamp,
                reasons=result.reasons,
                file_created=file_created,
                file_modified=file_modified,
                file_accessed=file_accessed,
            )

        except Exception as e:
            logger.debug(f"Classification failed for {file_path.name}: {e}")
            self.processor.errors.append(
                {
                    "file": str(file_path),
                    "operation": "classify",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            return None

    def _handle_compressed_file(
        self,
        file_path: Path,
        decompressed_header: bytes,
        compression_type: str,
        original_size: int,
        file_created: datetime,
        file_modified: datetime,
        file_accessed: datetime,
    ) -> RecoveredFile | None:
        """
        Handle a compressed file by classifying its decompressed contents.

        For useful file types (SQLite, logs, WiFi plists, etc.), decompresses
        the full file to temp, copies to appropriate destination, and cleans up.

        Args:
            file_path: Path to compressed file
            decompressed_header: First 8KB of decompressed content
            compression_type: "gzip" or "bzip2"
            original_size: Size of compressed file
            file_created: Original file creation timestamp
            file_modified: Original file modification timestamp
            file_accessed: Original file access timestamp

        Returns:
            RecoveredFile for the decompressed content, or None if not useful
        """
        # Check what type of file is inside the archive
        if decompressed_header.startswith(SQLITE_MAGIC):
            # SQLite database inside archive
            result_type = LogType.SQLITE
            confidence = 1.0
            reasons = [f"SQLite database in {compression_type} archive"]
        else:
            # Try text log fingerprinting on decompressed content
            # Create a BytesIO or temp file for identify_log_type
            temp_path = None
            try:
                # Write decompressed header to temp file for fingerprinting
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".tmp", delete=False) as tmp:
                    tmp.write(decompressed_header)
                    temp_path = Path(tmp.name)

                result = identify_log_type(temp_path, min_confidence=self.processor.min_confidence)
                result_type = result.log_type
                confidence = result.confidence
                reasons = result.reasons + [f"Found in {compression_type} archive"]

            finally:
                # Clean up temp file
                if temp_path and temp_path.exists():
                    temp_path.unlink()

            # Skip if unknown or not useful
            if result_type == LogType.UNKNOWN:
                return None

        # Check if this is a useful type worth decompressing fully
        useful_types = {
            LogType.SQLITE,
            LogType.PLIST_WIFI,
            LogType.WIFI_LOG,
            LogType.SYSTEM_LOG,
            LogType.INSTALL_LOG,
            LogType.JSONLZ4,
            LogType.KEYCHAIN,
            LogType.ASL_BINARY,
        }

        if result_type not in useful_types:
            # Not a useful type - skip
            return None

        # Decompress full file to temp location
        decompressed_path = None
        try:
            # Determine appropriate extension
            if result_type == LogType.SQLITE:
                ext = ".sqlite"
            elif result_type in (
                LogType.WIFI_LOG,
                LogType.SYSTEM_LOG,
                LogType.INSTALL_LOG,
            ):
                ext = ".log"
            elif result_type in (
                LogType.PLIST_WIFI,
                LogType.PLIST_XML,
                LogType.PLIST_BINARY,
            ):
                ext = ".plist"
            elif result_type == LogType.JSONLZ4:
                ext = ".jsonlz4"
            elif result_type == LogType.KEYCHAIN:
                ext = ".keychain"
            elif result_type == LogType.ASL_BINARY:
                ext = ".asl"
            else:
                ext = ".dat"

            # Create temp file for decompressed content
            with tempfile.NamedTemporaryFile(mode="wb", suffix=ext, delete=False) as tmp:
                decompressed_path = Path(tmp.name)

            # Decompress fully with recovery fallback
            success = decompress_file_with_recovery(
                file_path,
                decompressed_path,
                compression_type=compression_type,
                try_recovery=True,
            )

            if not success:
                # Failed to decompress even with recovery
                self.processor.stats["archives_failed"] = self.processor.stats.get("archives_failed", 0) + 1
                return None

            # Get size of decompressed file
            decompressed_size = decompressed_path.stat().st_size

            # Copy to appropriate destination based on type
            dest_path = None
            stat_key = None

            if result_type == LogType.SQLITE:
                # Remove .gz or .bz2 extension, keep base name
                base_name = file_path.stem  # e.g., "database.sqlite" from "database.sqlite.gz"
                # Check if database is encrypted
                if is_encrypted_database(decompressed_path):
                    # Route encrypted database to encrypted folder
                    encrypted_dir = self.processor.paths.db_encrypted
                    encrypted_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = encrypted_dir / base_name
                    counter = 1
                    while dest_path.exists():
                        stem = Path(base_name).stem
                        suffix = Path(base_name).suffix or ".db"
                        dest_path = encrypted_dir / f"{stem}_{counter:03d}{suffix}"
                        counter += 1
                    shutil.copy2(decompressed_path, dest_path)
                    stat_key = "sqlite_encrypted_from_archives"
                    logger.debug(f"Encrypted database from archive routed to: {dest_path}")
                else:
                    dest_path = self.processor.paths.temp / base_name
                    shutil.copy2(decompressed_path, dest_path)
                    stat_key = "sqlite_from_archives"

            elif result_type == LogType.PLIST_WIFI:
                # Validate plist and organize into exemplar folder structure
                try:
                    with decompressed_path.open("rb") as f:
                        header = f.read(8192)

                    # Validate plist structure by magic bytes only - don't use plistlib.load()
                    # because it fails on valid binary plists with UID/Set objects or unusual encodings
                    is_valid_plist = (
                        header.startswith(b"bplist")  # Binary plist
                        or (header.startswith(b"<?xml") and b"<plist" in header[:512])  # XML plist
                        or header.startswith(b"<!DOCTYPE plist")  # DOCTYPE plist
                    )

                    if not is_valid_plist:
                        raise ValueError("Not a valid plist format")

                    # Detect specific WiFi plist type
                    specific_type, confidence = detect_specific_wifi_plist_type(header)

                    # Map to exemplar folder structure using catalog
                    folder_info = self.wifi_mapper.get_folder_info(specific_type) if specific_type else None

                    if folder_info:
                        folder_name, base_filename = folder_info
                        dest_dir = self.processor.paths.logs / folder_name
                    else:
                        # Unknown WiFi plist type (not in catalog)
                        dest_dir = self.processor.paths.logs / "WiFi (Unknown)"
                        base_filename = file_path.stem

                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Generate unique filename with sequential counter
                    dest_path = dest_dir / f"{base_filename}.plist"
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{base_filename}_{counter:03d}.plist"
                        counter += 1

                    shutil.copy2(decompressed_path, dest_path)
                    stat_key = "wifi_plists_from_archives"

                except Exception as e:
                    # Plist is corrupt or cannot be parsed
                    logger.info(
                        f"WiFi plist from archive {file_path.name} failed validation: {e}",
                    )
                    corrupt_dir = self.processor.paths.logs / "WiFi (Corrupt)"
                    corrupt_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = corrupt_dir / f"{file_path.stem}_corrupt.plist"
                    counter = 1
                    while dest_path.exists():
                        dest_path = corrupt_dir / f"{file_path.stem}_corrupt_{counter:03d}.plist"
                        counter += 1
                    shutil.copy2(decompressed_path, dest_path)
                    stat_key = "wifi_plists_corrupt"

            elif result_type == LogType.WIFI_LOG:
                wifi_dir = self.processor.paths.logs / "WiFi Log"
                wifi_dir.mkdir(parents=True, exist_ok=True)
                # Use semantic naming with sequential counter
                dest_path = wifi_dir / "wifi.log"
                counter = 1
                while dest_path.exists():
                    dest_path = wifi_dir / f"wifi_{counter:03d}.log"
                    counter += 1
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "wifi_logs_from_archives"

            elif result_type == LogType.SYSTEM_LOG:
                system_dir = self.processor.paths.logs / "System Log"
                system_dir.mkdir(parents=True, exist_ok=True)
                base_name = file_path.stem
                dest_path = system_dir / base_name
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "system_logs_from_archives"

            elif result_type == LogType.INSTALL_LOG:
                install_dir = self.processor.paths.logs / "Install Log"
                install_dir.mkdir(parents=True, exist_ok=True)
                base_name = file_path.stem
                dest_path = install_dir / base_name
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "install_logs_from_archives"

            elif result_type == LogType.ASL_BINARY:
                asl_dir = self.processor.paths.logs / "Apple System Log (ASL)"
                asl_dir.mkdir(parents=True, exist_ok=True)
                base_name = file_path.stem
                dest_path = asl_dir / base_name
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "asl_logs_from_archives"

            elif result_type == LogType.JSONLZ4:
                firefox_lz4_dir = self.processor.paths.caches / "Firefox LZ4"
                firefox_lz4_dir.mkdir(parents=True, exist_ok=True)
                base_name = file_path.stem
                dest_path = firefox_lz4_dir / base_name
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "jsonlz4_from_archives"

            elif result_type == LogType.KEYCHAIN:
                base_name = file_path.stem
                dest_path = self.processor.paths.keychains / base_name
                shutil.copy2(decompressed_path, dest_path)
                stat_key = "keychains_from_archives"

            # Update stats
            if stat_key:
                self.processor.stats[stat_key] = self.processor.stats.get(stat_key, 0) + 1
            self.processor.stats["archives_processed"] = self.processor.stats.get("archives_processed", 0) + 1

            # For encrypted databases, don't return RecoveredFile (already handled)
            if stat_key and "encrypted" in stat_key:
                return None

            # Return RecoveredFile for the decompressed content
            return RecoveredFile(
                source_path=file_path,
                file_type=result_type,
                confidence=confidence,
                size=decompressed_size,
                md5=None,
                decompressed_path=dest_path,
                reasons=reasons,
                file_created=file_created,
                file_modified=file_modified,
                file_accessed=file_accessed,
            )

        except Exception as e:
            logger.debug(f"Decompression failed for {file_path.name}: {e}")
            self.processor.errors.append(
                {
                    "file": str(file_path),
                    "operation": f"decompress_{compression_type}",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            return None

        finally:
            # Clean up temp decompressed file
            if decompressed_path and decompressed_path.exists():
                decompressed_path.unlink(missing_ok=True)

    # ==========================================================================
    # ========================= BATCH PROCESSING ==============================
    # ==========================================================================
    def process_file_batch(
        self,
        file_batch: list[Path],
        progress: Progress | None = None,
        task_id: TaskID | None = None,
        overall_task: TaskID | None = None,
        start_time: float | None = None,
    ) -> dict[str, Any]:
        """
        Process a batch of files in a worker thread.

        Args:
            file_batch: List of file paths to process
            progress: Rich Progress instance for updating progress
            task_id: Task ID for this batch's progress bar
            overall_task: Task ID for overall progress bar
            start_time: Start time for elapsed time calculation

        Returns:
            Dictionary containing batch results and stats
        """
        batch_stats = {
            "total_files": 0,
            "classified": 0,
            "unknown": 0,
            "sqlite_dbs": 0,
            "text_logs": 0,
            "wifi_plists_kept": 0,
            "wifi_plists_corrupt": 0,
            "plists_skipped": 0,
            "plists_skipped_bytes": 0,
            "asl_logs_kept": 0,
            "jsonlz4_kept": 0,
            "keychains_kept": 0,
        }

        batch_results = {
            "sqlite_dbs": [],
            "text_logs": {},
            "other_files": {},
        }

        for file_path in file_batch:
            batch_stats["total_files"] += 1
            file_count = batch_stats["total_files"]

            classified_file = self.classify_file(file_path)

            # Update progress bars (throttled to reduce lock contention)
            if progress:
                # Worker progress: update every _PROGRESS_WORKER_INTERVAL files
                if task_id is not None and file_count % _PROGRESS_WORKER_INTERVAL == 0:
                    progress.update(task_id, advance=_PROGRESS_WORKER_INTERVAL)

                # Overall progress + timer: update every _PROGRESS_OVERALL_INTERVAL files
                if overall_task is not None and file_count % _PROGRESS_OVERALL_INTERVAL == 0:
                    progress.update(overall_task, advance=_PROGRESS_OVERALL_INTERVAL)

                    # Update elapsed time display
                    if start_time is not None:
                        elapsed = time.time() - start_time
                        minutes, seconds = divmod(int(elapsed), 60)
                        hours, minutes = divmod(minutes, 60)
                        time_str = (
                            f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"
                        )
                        progress.update(
                            overall_task,
                            description=f"[cyan]Progress [dim]({time_str})[/dim]",
                        )

            if not classified_file:
                batch_stats["unknown"] += 1
                continue

            batch_stats["classified"] += 1

            # Handle files based on classification
            if classified_file.decompressed_path is not None:
                # File from archive - already processed
                if classified_file.file_type == LogType.SQLITE:
                    batch_results["sqlite_dbs"].append(classified_file)
                    batch_stats["sqlite_dbs"] += 1

                elif classified_file.file_type == LogType.PLIST_WIFI:
                    batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)

                elif classified_file.file_type in (
                    LogType.WIFI_LOG,
                    LogType.SYSTEM_LOG,
                    LogType.INSTALL_LOG,
                    LogType.ASL_BINARY,
                ):
                    batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                    # Track ASL logs separately
                    if classified_file.file_type == LogType.ASL_BINARY:
                        batch_stats["asl_logs_kept"] += 1

                elif classified_file.file_type in (LogType.JSONLZ4, LogType.KEYCHAIN):
                    # Skip keychains for candidate scans (exemplar_db_dir is set for candidates)
                    if classified_file.file_type == LogType.KEYCHAIN and self.processor.exemplar_db_dir is not None:
                        continue

                    batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                    # Track JSONLZ4 and keychains separately
                    if classified_file.file_type == LogType.JSONLZ4:
                        batch_stats["jsonlz4_kept"] += 1
                    elif classified_file.file_type == LogType.KEYCHAIN:
                        batch_stats["keychains_kept"] += 1

                continue

            # Handle direct (non-archive) files
            self._process_direct_file(
                file_path,
                classified_file,
                batch_stats,
                batch_results,
            )

        # Final update to catch any remaining files not covered by throttling
        if progress:
            file_count = batch_stats["total_files"]
            # Update worker progress for any remaining files
            if task_id is not None:
                remainder_worker = file_count % _PROGRESS_WORKER_INTERVAL
                if remainder_worker > 0:
                    progress.update(task_id, advance=remainder_worker)

            # Update overall progress for any remaining files
            if overall_task is not None:
                remainder_overall = file_count % _PROGRESS_OVERALL_INTERVAL
                if remainder_overall > 0:
                    progress.update(overall_task, advance=remainder_overall)

        return {"stats": batch_stats, "results": batch_results}

    def _process_direct_file(
        self,
        file_path: Path,
        classified_file: RecoveredFile,
        batch_stats: dict,
        batch_results: dict,
    ):
        """Process a single non-archive file (thread-safe helper)."""
        # Handle SQLite files
        if classified_file.file_type == LogType.SQLITE:
            try:
                # Check if database is encrypted
                if is_encrypted_database(file_path):
                    # Route encrypted database to encrypted folder with provenance
                    encrypted_dir = self.processor.paths.db_encrypted

                    # Try to get artifact info from extraction manifest for TM scans
                    artifact_name = None
                    manifest_entry = None
                    if self.extraction_manifest:
                        try:
                            rel_path = str(file_path.relative_to(self.processor.input_dir))
                            manifest_entry = self.extraction_manifest.get(rel_path)
                            if manifest_entry:
                                artifact_name = manifest_entry.get("artifact_name")
                        except ValueError:
                            pass

                    # Create semantic subfolder if we have artifact name
                    if artifact_name:
                        encrypted_dir = encrypted_dir / artifact_name
                    encrypted_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = encrypted_dir / file_path.name
                    counter = 1
                    while dest_path.exists():
                        dest_path = encrypted_dir / f"{file_path.stem}_{counter:03d}{file_path.suffix}"
                        counter += 1
                    shutil.copy2(file_path, dest_path)

                    # Write provenance for TM encrypted databases
                    if manifest_entry:
                        provenance_path = dest_path.with_suffix(".provenance.json")
                        provenance_data = {
                            "encrypted": True,
                            "artifact_name": artifact_name,
                            "backup_id": manifest_entry.get("backup_id"),
                            "backup_date": manifest_entry.get("backup_date"),
                            "original_path": manifest_entry.get("original_path"),
                            "source_path": manifest_entry.get("source_path"),
                            "file_timestamps": manifest_entry.get("file_timestamps"),
                            "copied_to": str(dest_path),
                        }
                        with provenance_path.open("w") as f:
                            import json

                            json.dump(provenance_data, f, indent=2)

                    batch_stats["sqlite_encrypted"] = batch_stats.get("sqlite_encrypted", 0) + 1
                    logger.debug(f"Encrypted database routed to: {dest_path}")
                else:
                    # Normal database - copy to temp for processing
                    dest_path = self.processor.paths.temp / file_path.name
                    shutil.copy2(file_path, dest_path)
                    batch_results["sqlite_dbs"].append(classified_file)
                    batch_stats["sqlite_dbs"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy SQLite file {file_path.name}: {e}")
            return

        # Handle WiFi plists with validation and semantic naming
        if classified_file.file_type == LogType.PLIST_WIFI:
            try:
                with file_path.open("rb") as f:
                    header = f.read(8192)

                # Validate plist structure by magic bytes only - don't use plistlib.load()
                # because it fails on valid binary plists with UID/Set objects or unusual encodings
                is_valid_plist = (
                    header.startswith(b"bplist")  # Binary plist
                    or (header.startswith(b"<?xml") and b"<plist" in header[:512])  # XML plist
                    or header.startswith(b"<!DOCTYPE plist")  # DOCTYPE plist
                )

                if not is_valid_plist:
                    raise ValueError("Not a valid plist format")

                specific_type, _ = detect_specific_wifi_plist_type(header)

                # Map to exemplar folder structure using catalog
                folder_info = self.wifi_mapper.get_folder_info(specific_type) if specific_type else None

                if folder_info:
                    folder_name, base_filename = folder_info
                    dest_dir = self.processor.paths.logs / folder_name
                else:
                    # Unknown WiFi plist type (not in catalog)
                    dest_dir = self.processor.paths.logs / "WiFi (Unknown)"
                    base_filename = file_path.stem

                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / f"{base_filename}.plist"
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{base_filename}_{counter:03d}.plist"
                    counter += 1

                shutil.copy2(file_path, dest_path)
                batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["wifi_plists_kept"] += 1

            except Exception:
                # Corrupt plist
                corrupt_dir = self.processor.paths.logs / "WiFi (Corrupt)"
                corrupt_dir.mkdir(parents=True, exist_ok=True)
                dest_path = corrupt_dir / f"{file_path.stem}_corrupt.plist"
                counter = 1
                while dest_path.exists():
                    dest_path = corrupt_dir / f"{file_path.stem}_corrupt_{counter:03d}.plist"
                    counter += 1
                try:
                    shutil.copy2(file_path, dest_path)
                    batch_stats["wifi_plists_corrupt"] += 1
                except Exception as e:
                    logger.debug(f"Failed to copy corrupt WiFi plist {file_path.name}: {e}")
            return

        # Handle WiFi logs
        if classified_file.file_type == LogType.WIFI_LOG:
            try:
                wifi_dir = self.processor.paths.logs / "WiFi Log"
                wifi_dir.mkdir(parents=True, exist_ok=True)
                dest_path = wifi_dir / "wifi.log"
                counter = 1
                while dest_path.exists():
                    dest_path = wifi_dir / f"wifi_{counter:03d}.log"
                    counter += 1
                shutil.copy2(file_path, dest_path)
                batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["text_logs"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy WiFi log {file_path.name}: {e}")
            return

        # Handle System logs
        if classified_file.file_type == LogType.SYSTEM_LOG:
            try:
                system_dir = self.processor.paths.logs / "System Log"
                system_dir.mkdir(parents=True, exist_ok=True)
                dest_path = system_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["text_logs"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy system log {file_path.name}: {e}")
            return

        # Handle Install logs
        if classified_file.file_type == LogType.INSTALL_LOG:
            try:
                install_dir = self.processor.paths.logs / "Install Log"
                install_dir.mkdir(parents=True, exist_ok=True)
                dest_path = install_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["text_logs"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy install log {file_path.name}: {e}")
            return

        # Handle ASL logs
        if classified_file.file_type == LogType.ASL_BINARY:
            try:
                asl_dir = self.processor.paths.logs / "Apple System Log (ASL)"
                asl_dir.mkdir(parents=True, exist_ok=True)
                dest_path = asl_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["text_logs"] += 1
                batch_stats["asl_logs_kept"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy ASL log {file_path.name}: {e}")
            return

        # Handle JSONLZ4 files (Firefox compressed JSON)
        if classified_file.file_type == LogType.JSONLZ4:
            try:
                firefox_lz4_dir = self.processor.paths.caches / "Firefox LZ4"
                firefox_lz4_dir.mkdir(parents=True, exist_ok=True)
                dest_path = firefox_lz4_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["jsonlz4_kept"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy JSONLZ4 file {file_path.name}: {e}")
            return

        # Handle Keychain files (exemplar scans only)
        if classified_file.file_type == LogType.KEYCHAIN:
            # Skip keychains for candidate scans (exemplar_db_dir is set for candidates)
            if self.processor.exemplar_db_dir is not None:
                return

            try:
                keychain_dir = self.processor.paths.keychains
                keychain_dir.mkdir(parents=True, exist_ok=True)
                dest_path = keychain_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["keychains_kept"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy keychain {file_path.name}: {e}")
            return

        # Handle Time Machine log files (classified by ARC catalog)
        if classified_file.file_type == LogType.TM_LOG:
            try:
                # Preserve directory structure from TM extraction
                rel_path = file_path.relative_to(self.processor.input_dir)
                # Remove the "logs/" prefix if present since we're putting in logs dir
                parts = rel_path.parts
                if parts and parts[0] == "logs":
                    rel_path = Path(*parts[1:]) if len(parts) > 1 else Path(file_path.name)
                dest_path = self.processor.paths.logs / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                # Write provenance data for TM files
                self._write_tm_provenance(file_path, dest_path)
                batch_results["text_logs"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["text_logs"] += 1
            except Exception as e:
                logger.debug(f"Failed to copy TM log {file_path.name}: {e}")
            return

        # Handle Time Machine cache files (classified by ARC catalog)
        if classified_file.file_type == LogType.TM_CACHE:
            try:
                # Preserve directory structure from TM extraction
                rel_path = file_path.relative_to(self.processor.input_dir)
                # Remove the "caches/" prefix if present since we're putting in caches dir
                parts = rel_path.parts
                if parts and parts[0] == "caches":
                    rel_path = Path(*parts[1:]) if len(parts) > 1 else Path(file_path.name)
                dest_path = self.processor.paths.caches / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                # Write provenance data for TM files
                self._write_tm_provenance(file_path, dest_path)
                batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["caches_kept"] = batch_stats.get("caches_kept", 0) + 1
            except Exception as e:
                logger.debug(f"Failed to copy TM cache {file_path.name}: {e}")
            return

        # Handle Time Machine keychain files (classified by ARC catalog)
        if classified_file.file_type == LogType.TM_KEYCHAIN:
            try:
                # Preserve directory structure from TM extraction
                rel_path = file_path.relative_to(self.processor.input_dir)
                # Remove the "keychains/" prefix if present since we're putting in keychains dir
                parts = rel_path.parts
                if parts and parts[0] == "keychains":
                    rel_path = Path(*parts[1:]) if len(parts) > 1 else Path(file_path.name)
                dest_path = self.processor.paths.keychains / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                # Write provenance data for TM files
                self._write_tm_provenance(file_path, dest_path)
                batch_results["other_files"].setdefault(classified_file.file_type, []).append(classified_file)
                batch_stats["keychains_kept"] = batch_stats.get("keychains_kept", 0) + 1
            except Exception as e:
                logger.debug(f"Failed to copy TM keychain {file_path.name}: {e}")
            return

        # Skip non-WiFi plists (we already handled PLIST_WIFI above)
        if classified_file.file_type in (LogType.PLIST_XML, LogType.PLIST_BINARY):
            batch_stats["plists_skipped"] += 1
            batch_stats["plists_skipped_bytes"] += classified_file.size
            return

    # ==========================================================================
    # =========================== SCAN AND CLASSIFY ===========================
    # ==========================================================================
    def scan_and_classify(self, console: Console | None):
        """
        Phase 1: Scan all files and classify them (multithreaded).

        This is the entry point for processing.
        """

        if not console:
            console = Console()  # Console yourself

        # Discover files with a spinner and running count
        logger.debug("Scanning input directory...")
        files = []
        skipped_count = 0
        total_seen = 0
        with console.status(
            "[bold dark_sea_green4]Discovering files...[/bold dark_sea_green4]", spinner="arrow3"
        ) as status:
            for file_path in self.processor.input_dir.rglob("*"):
                if file_path.is_file():
                    total_seen += 1
                    if self._should_ignore_file(file_path):
                        skipped_count += 1
                    else:
                        files.append(file_path)
                    # Update status every 25 files for better responsiveness on slow storage
                    if total_seen % 25 == 0:
                        status.update(
                            f"[bold dark_sea_green4]Discovering files...[/bold dark_sea_green4] [cyan]{len(files):,} found ({skipped_count:,} skipped)[/]"
                        )
            # Final update with total count
            if files or skipped_count > 0:
                status.update(f"[bold cyan]Discovered {len(files):,} file(s) to process ({skipped_count:,} skipped)")

        total = len(files)

        if len(files) == 0:
            logger.warning("No files found to process")
            return

        # ================== PROCESSING FILES ======================
        # Shuffle files to distribute slow files (archives, large files) evenly across batches
        # This prevents one worker from getting stuck with all the slow files
        import random

        random.shuffle(files)

        # Determine number of worker threads (use CPU count or max 8)
        num_workers = min(multiprocessing.cpu_count(), 8)
        batch_size = max(100, total // (num_workers * 4))  # At least 100 files per batch

        # Split files into batches
        batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

        # Check if progress bars should be shown
        show_progress = self.processor.config.ui.show_progress_bars

        # Track start time for elapsed time display
        import time

        start_time = time.time()

        # Create Rich progress bars with multiple tasks (if enabled)
        # Note: We manually add elapsed time to overall task description
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(),
                MofNCompleteColumn(),
            )

            # Dictionary to track worker tasks
            worker_tasks = {}
            task_lock = threading.Lock()

            # Create overall progress task
            overall_task = progress.add_task("[cyan]Progress", total=total)
        else:
            # No progress bars - use placeholders
            progress = None
            worker_tasks = {}
            task_lock = threading.Lock()
            overall_task = None

        def process_batch_with_progress(_batch_idx: int, batch: list[Path]):
            """Process a batch and update its dedicated progress bar."""
            if progress is None:
                # No progress bars - just process the batch
                return self.process_file_batch(
                    batch,
                    progress=None,
                    task_id=None,
                    overall_task=None,
                    start_time=start_time,
                )

            thread_name = threading.current_thread().name

            # Create or reset progress bar for this worker thread
            with task_lock:
                if thread_name not in worker_tasks:
                    # First batch for this worker - create new task
                    worker_tasks[thread_name] = progress.add_task(
                        f"[green]Worker {len(worker_tasks) + 1}",
                        total=len(batch),
                        visible=True,
                    )
                else:
                    # Worker is reusing thread - reset task for new batch
                    task_id = worker_tasks[thread_name]
                    progress.reset(
                        task_id,
                        total=len(batch),
                        completed=0,
                        visible=True,
                        description=f"[green]Worker {list(worker_tasks.keys()).index(thread_name) + 1}",
                    )

            task_id = worker_tasks[thread_name]

            # Process the batch (overall progress and timer update per-file inside)
            result = self.process_file_batch(
                batch,
                progress=progress,
                task_id=task_id,
                overall_task=overall_task,
                start_time=start_time,
            )

            # Hide this worker's bar (will be reset if worker gets another batch)
            progress.update(task_id, visible=False)

            return result

        # Process batches in parallel with live progress display
        batch_results = []

        if show_progress:
            # Wrap progress in Panel
            assert progress is not None  # Type checker hint
            panel = Panel(
                progress,
                title="[bold deep_sky_blue1]File Categorization[/bold deep_sky_blue1]",
                border_style="deep_sky_blue3",
                padding=(1, 1),
            )

            # Note: Only enter Live context, not progress context, to avoid double rendering
            with (
                Live(panel, refresh_per_second=10, transient=True),
                ThreadPoolExecutor(max_workers=num_workers) as executor,
            ):
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_batch_with_progress, i, batch): i for i, batch in enumerate(batches)
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_num = future_to_batch[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as exc:
                        logger.error(f"Batch {batch_num} generated an exception: {exc}")
        else:
            # No progress bars - print plain text progress
            console.print(f"[dim]→[/dim] File Categorization: {total:,} files in {len(batches)} batches")

            completed_batches = 0
            total_batches = len(batches)
            last_pct_printed = -1

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_batch_with_progress, i, batch): i for i, batch in enumerate(batches)
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_num = future_to_batch[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        completed_batches += 1

                        # Print milestone updates (every 25%)
                        progress_pct = (completed_batches / total_batches) * 100
                        if progress_pct >= last_pct_printed + 10 or completed_batches == total_batches:
                            if completed_batches == total_batches:
                                console.print(
                                    f"[dim]  └─[/dim] Completed: {completed_batches}/{total_batches} batches ({total:,} files)"
                                )
                            else:
                                console.print(
                                    f"[dim]  ├─[/dim] Progress: {completed_batches}/{total_batches} batches ({progress_pct:.0f}%)"
                                )
                            last_pct_printed = progress_pct
                    except Exception as exc:
                        logger.error(f"Batch {batch_num} generated an exception: {exc}")

        # Merge all batch results into processor collections
        for batch_result in batch_results:
            stats = batch_result["stats"]
            results = batch_result["results"]

            # Merge stats
            for key, value in stats.items():
                self.processor.stats[key] = self.processor.stats.get(key, 0) + value

            # Merge SQLite databases
            for sqlite_file in results["sqlite_dbs"]:
                self.processor.sqlite_dbs.setdefault("from_archives", []).append(sqlite_file)

            # Merge text logs
            for log_type, files_list in results["text_logs"].items():
                self.processor.text_logs.setdefault(log_type, []).extend(files_list)

            # Merge other files
            for file_type, files_list in results["other_files"].items():
                self.processor.other_files.setdefault(file_type, []).extend(files_list)

        logger.debug("Classification complete")
        self.processor.reporter.print_stats(console=console, config=self.processor.config)
