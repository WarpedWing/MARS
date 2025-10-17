#!/usr/bin/env python3
"""
PhotoRec Output Processor for Mac Log Sleuth
by WarpedWing Labs

Processes PhotoRec carved output to identify, recover, and combine macOS forensic artifacts.

Workflow:
1. Scan all PhotoRec output directories (recup_dir.*)
2. Classify each file using fingerprinting
3. Decompress archives (with recovery fallback for corrupted files)
4. Group by artifact type (wifi logs, Safari history, etc.)
5. Combine fragments chronologically (text logs) or by schema (SQLite DBs)
6. Generate comprehensive recovery report

Usage:
    python photorec_processor.py --photorec-dir ./PhotoRec_Output --output-dir ./Recovered

Dependencies:
    - text_log_fingerprinter.py
    - gzrecover, bzip2recover (optional, for corrupted archives)
    - sqlite3 CLI with .recover command
    - sqlite_dissect (optional, for advanced SQLite recovery)
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import heapq
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

# Try to import fingerprinter
try:
    from mac_log_sleuth.text_log_fingerprinter import (
        LogFingerprint,
        LogType,
        identify_log_type,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from text_log_fingerprinter import LogFingerprint, LogType, identify_log_type

# Try to import output structure
try:
    from mac_log_sleuth.pipeline.output_structure import OutputStructure
except ImportError:
    try:
        from output_structure import OutputStructure
    except ImportError:
        OutputStructure = None


# ============================================================================
# Constants
# ============================================================================

SQLITE_MAGIC = b"SQLite format 3\x00"
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
MAX_MEMORY_LOG_SIZE = 100 * 1024 * 1024  # 100MB - load into memory
MAX_MEMORY_DB_SIZE = 500 * 1024 * 1024  # 500MB - load into memory


# ============================================================================
# Data Classes
# ============================================================================


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
            "decompressed_path": str(self.decompressed_path) if self.decompressed_path else None,
            "reasons": self.reasons,
        }


@dataclass
class SQLiteSchema:
    """Schema fingerprint for a SQLite database."""

    tables: list[str]  # Table names sorted alphabetically
    table_schemas: dict[str, str]  # Table name -> CREATE TABLE statement
    row_counts: dict[str, int]  # Table name -> row count
    size_bytes: int
    schema_hash: str  # MD5 hash of sorted table names for quick matching

    def matches(self, other: "SQLiteSchema", tolerance: float = 0.8) -> bool:
        """
        Check if two schemas match (for identifying fragments).

        Args:
            other: Another schema to compare against
            tolerance: Minimum fraction of matching tables (0.0-1.0)

        Returns:
            True if schemas match within tolerance
        """
        if not self.tables or not other.tables:
            return False

        # Quick check: schema hash
        if self.schema_hash == other.schema_hash:
            return True

        # Detailed check: table overlap
        common_tables = set(self.tables) & set(other.tables)
        max_tables = max(len(self.tables), len(other.tables))
        overlap = len(common_tables) / max_tables if max_tables > 0 else 0

        return overlap >= tolerance


# ============================================================================
# Timestamp Parsing & Log Line Sorting
# ============================================================================


@dataclass
class LogLine:
    """A single log line with parsed timestamp."""
    timestamp: datetime
    original_line: str
    source_file: str
    line_number: int

    def __lt__(self, other):
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


class TimestampParser:
    """Parse timestamps from various macOS log formats."""

    # WiFi log: Thu Jul 23 00:48:51.636 <airportd[128]>
    WIFI_PATTERN = re.compile(
        r'^(?P<dow>[A-Z][a-z]{2})\s+'
        r'(?P<mon>[A-Z][a-z]{2})\s+'
        r'(?P<day>\d{1,2})\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s*'
    )

    # System/Install log: Nov 23 17:45:15 hostname process[pid]:
    SYSLOG_PATTERN = re.compile(
        r'^(?P<mon>[A-Z][a-z]{2})\s+'
        r'(?P<day>\d{1,2})\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
    )

    MONTH_MAP = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    }

    @classmethod
    def parse_wifi_timestamp(cls, line: str, current_year: int = None) -> datetime | None:
        """Parse WiFi log timestamp."""
        match = cls.WIFI_PATTERN.match(line)
        if not match:
            return None

        try:
            month = cls.MONTH_MAP[match.group('mon')]
            day = int(match.group('day'))
            time_str = match.group('time')
            hour, minute, sec_ms = time_str.split(':')
            sec, ms = sec_ms.split('.')

            # Use current year if not specified
            year = current_year or datetime.now().year

            return datetime(
                year, month, day,
                int(hour), int(minute), int(sec), int(ms) * 1000,
                tzinfo=UTC
            )
        except Exception:
            return None

    @classmethod
    def parse_syslog_timestamp(cls, line: str, current_year: int = None) -> datetime | None:
        """Parse system/install log timestamp."""
        match = cls.SYSLOG_PATTERN.match(line)
        if not match:
            return None

        try:
            month = cls.MONTH_MAP[match.group('mon')]
            day = int(match.group('day'))
            time_str = match.group('time')
            hour, minute, sec = time_str.split(':')

            year = current_year or datetime.now().year

            return datetime(
                year, month, day,
                int(hour), int(minute), int(sec),
                tzinfo=UTC
            )
        except Exception:
            return None

    @classmethod
    def parse_line(cls, line: str, log_type: LogType, current_year: int = None) -> datetime | None:
        """Parse timestamp from line based on log type."""
        if log_type == LogType.WIFI_LOG:
            return cls.parse_wifi_timestamp(line, current_year)
        elif log_type in (LogType.SYSTEM_LOG, LogType.INSTALL_LOG):
            return cls.parse_syslog_timestamp(line, current_year)
        return None


# ============================================================================
# PhotoRec Processor
# ============================================================================


class PhotoRecProcessor:
    """Main processor for PhotoRec carved output."""

    def __init__(
        self,
        photorec_dir: Path,
        output_dir: Path,
        min_confidence: float = 0.6,
        verbose: bool = False,
    ):
        """
        Initialize PhotoRec processor.

        Args:
            photorec_dir: Root directory containing recup_dir.* folders
            output_dir: Output directory for recovered artifacts
            min_confidence: Minimum confidence threshold for classification
            verbose: Enable verbose output
        """
        self.photorec_dir = photorec_dir
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.verbose = verbose

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "_temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.corrupted_dir = self.output_dir / "corrupted_archives"
        self.corrupted_dir.mkdir(exist_ok=True)
        self.decompressed_dir = self.output_dir / "decompressed"
        self.decompressed_dir.mkdir(exist_ok=True)

        # Buffers for grouping files by type
        self.text_logs: dict[LogType, list[RecoveredFile]] = defaultdict(list)
        self.sqlite_dbs: dict[str, list[RecoveredFile]] = defaultdict(list)
        self.other_files: dict[LogType, list[RecoveredFile]] = defaultdict(list)

        # Tracking
        self.stats = {
            "total_files": 0,
            "classified": 0,
            "unknown": 0,
            "corrupted_archives": 0,
            "text_logs": 0,
            "sqlite_dbs": 0,
            "other": 0,
        }

        # Error tracking
        self.errors: list[dict[str, Any]] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with optional verbosity control."""
        prefix = {
            "INFO": "ℹ",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍",
        }.get(level, "•")

        if level == "DEBUG" and not self.verbose:
            return

        print(f"{prefix} {message}")

    def compute_md5(self, file_path: Path) -> str:
        """Compute MD5 hash of file."""
        md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception:
            return ""

    def extract_sqlite_schema(self, db_path: Path, max_size_mb: int = 50) -> SQLiteSchema | None:
        """
        Extract schema fingerprint from SQLite database.

        Args:
            db_path: Path to SQLite database
            max_size_mb: Skip databases larger than this (likely library databases)

        Returns:
            SQLiteSchema object or None if extraction failed or database too large
        """
        try:
            size_bytes = db_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            # Skip large databases (likely system libraries, not forensic artifacts)
            if size_mb > max_size_mb:
                self.log(f"Skipping large database {db_path.name} ({size_mb:.1f}MB)", "DEBUG")
                return None

            # Open database read-only
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Get all table names (excluding internal SQLite tables)
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

            if not tables:
                conn.close()
                return None

            # Get CREATE TABLE statements for each table
            table_schemas = {}
            row_counts = {}
            for table in tables:
                # Get schema
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
                result = cursor.fetchone()
                if result:
                    table_schemas[table] = result[0] or ""

                # Get row count (with timeout protection)
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
                    row_counts[table] = cursor.fetchone()[0]
                except Exception:
                    row_counts[table] = -1  # Error getting count

            conn.close()

            # Create schema hash from sorted table names
            schema_str = "|".join(sorted(tables))
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()

            return SQLiteSchema(
                tables=tables,
                table_schemas=table_schemas,
                row_counts=row_counts,
                size_bytes=size_bytes,
                schema_hash=schema_hash,
            )

        except Exception as e:
            self.log(f"Failed to extract schema from {db_path.name}: {e}", "DEBUG")
            return None

    def iter_photorec_files(self) -> list[Path]:
        """
        Find all files in PhotoRec output directories.

        Supports two organizational schemes:
        1. Type-organized (after PhotoRec refinery): sqlite/, gz/, bz2/, etc.
        2. Raw PhotoRec output: recup_dir.1/, recup_dir.2/, etc.

        Returns:
            List of file paths to process
        """
        files = []

        # Relevant file type folders (from PhotoRec refinery app)
        relevant_folders = [
            "sqlite", "gz", "bz2", "txt", "log",
            "plist", "json", "jsonlz4", "asl"
        ]

        # Try type-organized folders first
        for folder_name in relevant_folders:
            folder_path = self.photorec_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                # Recursively find all files (handles subdirectories like sqlite/1/, sqlite/2/, etc.)
                dir_files = [f for f in folder_path.rglob("*") if f.is_file()]
                files.extend(dir_files)
                self.log(f"  {folder_name}/: {len(dir_files)} files", "DEBUG")

        # If no type folders found, try raw recup_dir.* format
        if not files:
            recup_dirs = sorted(self.photorec_dir.glob("recup_dir.*"))
            if recup_dirs:
                self.log(f"Found {len(recup_dirs)} recup_dir.* directories (raw PhotoRec format)")
                for recup_dir in recup_dirs:
                    if not recup_dir.is_dir():
                        continue
                    dir_files = [f for f in recup_dir.iterdir() if f.is_file()]
                    files.extend(dir_files)
                    self.log(f"  {recup_dir.name}: {len(dir_files)} files", "DEBUG")

        if not files:
            self.log(f"No files found in {self.photorec_dir}", "WARNING")
            self.log("Expected either: type folders (sqlite/, gz/, etc.) or recup_dir.* folders", "INFO")

        return files

    def classify_file(self, file_path: Path) -> RecoveredFile | None:
        """
        Classify a single file using fingerprinting.

        Args:
            file_path: Path to file to classify

        Returns:
            RecoveredFile object or None if classification failed
        """
        try:
            size = file_path.stat().st_size

            # Check for SQLite first (very common and specific magic)
            with open(file_path, "rb") as f:
                header = f.read(16)

            if header.startswith(SQLITE_MAGIC):
                # SQLite database - special handling
                md5 = self.compute_md5(file_path)
                return RecoveredFile(
                    source_path=file_path,
                    file_type=LogType.SQLITE,
                    confidence=1.0,
                    size=size,
                    md5=md5,
                    reasons=["SQLite magic bytes detected"],
                )

            # Try text log fingerprinting
            result = identify_log_type(file_path, min_confidence=self.min_confidence)

            if result.log_type == LogType.UNKNOWN:
                return None

            # Compute hash if high confidence
            md5 = self.compute_md5(file_path) if result.confidence >= 0.8 else None

            return RecoveredFile(
                source_path=file_path,
                file_type=result.log_type,
                confidence=result.confidence,
                size=size,
                md5=md5,
                first_timestamp=result.first_timestamp,
                last_timestamp=result.last_timestamp,
                reasons=result.reasons,
            )

        except Exception as e:
            self.errors.append(
                {
                    "file": str(file_path),
                    "operation": "classify",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            return None

    def decompress_archive(self, file_path: Path) -> Path | None:
        """
        Decompress gzip or bzip2 archive.

        Args:
            file_path: Path to compressed file

        Returns:
            Path to decompressed file, or None if decompression failed
        """
        try:
            # Read header to detect compression type
            with open(file_path, "rb") as f:
                magic = f.read(2)

            # Determine output filename
            output_name = file_path.stem if file_path.suffix in ('.gz', '.bz2') else f"{file_path.name}.decompressed"
            output_path = self.decompressed_dir / output_name

            # Try decompression
            if magic == b'\x1f\x8b':  # gzip
                try:
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    return output_path
                except Exception as e:
                    self.log(f"gzip decompression failed for {file_path.name}: {e}", "DEBUG")
                    # Try recovery
                    return self._try_gzrecover(file_path, output_path)

            elif magic == b'BZ':  # bzip2
                try:
                    with bz2.open(file_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    return output_path
                except Exception as e:
                    self.log(f"bzip2 decompression failed for {file_path.name}: {e}", "DEBUG")
                    # Try recovery
                    return self._try_bzip2recover(file_path, output_path)

        except Exception as e:
            self.errors.append({
                "file": str(file_path),
                "operation": "decompress",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            })

        return None

    def _try_gzrecover(self, file_path: Path, output_path: Path) -> Path | None:
        """Try to recover corrupted gzip file using gzrecover."""
        gzrecover = shutil.which("gzrecover")
        if not gzrecover:
            self.log("gzrecover not found in PATH, cannot recover", "DEBUG")
            return None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # gzrecover outputs to current directory
                result = subprocess.run(
                    [gzrecover, str(file_path)],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=30,
                    text=True
                )

                # Look for recovered file
                recovered = list(tmpdir_path.glob("*"))
                if recovered:
                    shutil.copy(recovered[0], output_path)
                    self.log(f"Recovered {file_path.name} with gzrecover", "SUCCESS")
                    return output_path

        except subprocess.TimeoutExpired:
            self.log(f"gzrecover timeout on {file_path.name}", "WARNING")
        except Exception as e:
            self.log(f"gzrecover failed: {e}", "DEBUG")

        return None

    def _try_bzip2recover(self, file_path: Path, output_path: Path) -> Path | None:
        """Try to recover corrupted bzip2 file using bzip2recover."""
        bzip2recover = shutil.which("bzip2recover")
        if not bzip2recover:
            self.log("bzip2recover not found in PATH, cannot recover", "DEBUG")
            return None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Copy to temp (bzip2recover works in place)
                temp_file = tmpdir_path / file_path.name
                shutil.copy(file_path, temp_file)

                # Run recovery
                result = subprocess.run(
                    [bzip2recover, str(temp_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=30,
                    text=True
                )

                # Look for recovered files (rec*.bz2)
                recovered = list(tmpdir_path.glob("rec*.bz2"))
                if recovered:
                    # Try to decompress recovered file
                    with bz2.open(recovered[0], 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    self.log(f"Recovered {file_path.name} with bzip2recover", "SUCCESS")
                    return output_path

        except subprocess.TimeoutExpired:
            self.log(f"bzip2recover timeout on {file_path.name}", "WARNING")
        except Exception as e:
            self.log(f"bzip2recover failed: {e}", "DEBUG")

        return None

    def process_archives(self):
        """
        Phase 2: Decompress and recover archives.

        Processes all gz/bz2 files found during classification.
        """
        self.log("=" * 70)
        self.log("Phase 2: Decompressing and Recovering Archives")
        self.log("=" * 70)

        # Collect all files that need decompression
        archives_to_process = []

        for recovered_list in [self.text_logs, self.sqlite_dbs, self.other_files]:
            for files in (recovered_list.values() if isinstance(recovered_list, dict) else []):
                for recovered in files:
                    suffix = recovered.source_path.suffix.lower()
                    if suffix in ('.gz', '.bz2'):
                        archives_to_process.append(recovered)

        if not archives_to_process:
            self.log("No archives to process")
            return

        self.log(f"Processing {len(archives_to_process)} compressed files...")

        success_count = 0
        recovered_count = 0
        failed_count = 0

        for idx, recovered in enumerate(archives_to_process, 1):
            if idx % 50 == 0 or idx == len(archives_to_process):
                percent = (idx / len(archives_to_process)) * 100
                print(f"\r  Progress: {idx}/{len(archives_to_process)} ({percent:.1f}%)", end="", flush=True)

            decompressed_path = self.decompress_archive(recovered.source_path)

            if decompressed_path:
                # Update recovered file to point to decompressed version
                recovered.decompressed_path = decompressed_path
                success_count += 1

                # Re-classify decompressed content
                reclassified = self.classify_file(decompressed_path)
                if reclassified and reclassified.confidence > recovered.confidence:
                    # Better classification after decompression
                    recovered.file_type = reclassified.file_type
                    recovered.confidence = reclassified.confidence
                    recovered.first_timestamp = reclassified.first_timestamp
                    recovered.last_timestamp = reclassified.last_timestamp
                    recovered_count += 1
            else:
                # Move to corrupted folder
                try:
                    corrupted_path = self.corrupted_dir / recovered.source_path.name
                    shutil.copy(recovered.source_path, corrupted_path)
                    failed_count += 1
                except Exception:
                    pass

        print()  # Newline after progress
        self.log(f"Decompression complete:")
        self.log(f"  ✓ Success: {success_count}")
        self.log(f"  🔧 Recovered: {recovered_count}")
        self.log(f"  ❌ Failed: {failed_count} (moved to corrupted_archives/)")

    def merge_text_logs(self):
        """
        Phase 3: Merge text logs chronologically.

        Uses streaming merge for memory efficiency.

        Note: macOS logs don't include year in timestamps. We infer year from:
        1. File metadata (modification time)
        2. Context from exemplar filesystem
        3. Fallback to current year

        This is a known limitation - year ambiguity cannot be fully resolved
        without external context.
        """
        self.log("=" * 70)
        self.log("Phase 3: Merging Text Logs Chronologically")
        self.log("=" * 70)

        if not self.text_logs:
            self.log("No text logs to merge")
            return

        combined_dir = self.output_dir / "combined_logs"
        combined_dir.mkdir(exist_ok=True)

        for log_type, files in self.text_logs.items():
            self.log(f"Merging {len(files)} {log_type.value} files...")

            # Infer year from file metadata
            inferred_year = self._infer_year_from_files(files)
            self.log(f"  Using year: {inferred_year} (inferred from file metadata)", "DEBUG")

            # Use streaming merge with heap
            output_path = combined_dir / f"combined_{log_type.value}.log"
            total_lines = self._merge_log_files_streaming(files, output_path, log_type, inferred_year)

            self.log(f"  ✓ Created {output_path.name} with {total_lines:,} lines", "SUCCESS")

    def _infer_year_from_files(self, files: list[RecoveredFile]) -> int:
        """
        Infer year from file metadata.

        Strategy (in order of preference):
        1. Gzip header MTIME field (embedded creation timestamp)
        2. File modification time
        3. Use most common year across all files
        4. Fallback to current year

        Note: This is heuristic - year cannot be definitively determined
        from macOS logs without external context.
        """
        years = []

        for recovered in files:
            file_path = recovered.source_path  # Use original file for gzip header

            # Try to extract timestamp from gzip header (most reliable)
            if file_path.suffix.lower() == '.gz':
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        if header[:2] == b'\x1f\x8b':  # Gzip magic
                            # MTIME is bytes 4-7 (little-endian uint32)
                            mtime = int.from_bytes(header[4:8], 'little')
                            if mtime > 0:  # 0 means no timestamp
                                year = datetime.fromtimestamp(mtime, tz=UTC).year
                                years.append(year)
                                self.log(f"  Found gzip timestamp: {year} from {file_path.name}", "DEBUG")
                                continue
                except Exception:
                    pass

            # Fallback to file modification time
            try:
                decompressed = recovered.decompressed_path or file_path
                mtime = decompressed.stat().st_mtime
                year = datetime.fromtimestamp(mtime, tz=UTC).year
                years.append(year)
            except Exception:
                continue

        if years:
            # Use most common year
            from collections import Counter
            most_common_year = Counter(years).most_common(1)[0][0]
            return most_common_year

        # Fallback
        return datetime.now().year

    def _merge_log_files_streaming(
        self,
        files: list[RecoveredFile],
        output_path: Path,
        log_type: LogType,
        current_year: int
    ) -> int:
        """
        Merge multiple log files chronologically using streaming approach.

        Uses heap-based merge for memory efficiency (doesn't load all files into memory).

        Args:
            files: List of RecoveredFile objects to merge
            output_path: Output file path
            log_type: Type of log (for timestamp parsing)
            current_year: Year to use for timestamp parsing

        Returns:
            Total lines written
        """
        # Open all files and create iterators
        file_iterators = []
        open_files = []

        for recovered in files:
            # Use decompressed path if available, otherwise source
            file_path = recovered.decompressed_path or recovered.source_path

            try:
                if file_path.stat().st_size > MAX_MEMORY_LOG_SIZE:
                    # Large file - use streaming
                    f = open(file_path, 'r', encoding='utf-8', errors='replace')
                    open_files.append(f)
                    file_iterators.append(self._parse_log_lines(f, str(file_path), log_type, current_year))
                else:
                    # Small file - load into memory
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                    file_iterators.append(self._parse_log_lines(iter(lines), str(file_path), log_type, current_year))

            except Exception as e:
                self.log(f"Failed to open {file_path.name}: {e}", "WARNING")
                self.errors.append({
                    "file": str(file_path),
                    "operation": "merge_logs",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                })

        if not file_iterators:
            return 0

        # Heap-based merge
        heap = []
        total_lines = 0

        # Initialize heap with first line from each file
        for idx, iterator in enumerate(file_iterators):
            try:
                log_line = next(iterator)
                heapq.heappush(heap, (log_line, idx, iterator))
            except StopIteration:
                pass

        # Write merged output
        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                while heap:
                    log_line, idx, iterator = heapq.heappop(heap)

                    # Write line
                    out_f.write(log_line.original_line)
                    if not log_line.original_line.endswith('\n'):
                        out_f.write('\n')

                    total_lines += 1

                    # Get next line from same file
                    try:
                        next_line = next(iterator)
                        heapq.heappush(heap, (next_line, idx, iterator))
                    except StopIteration:
                        pass

        finally:
            # Close all open file handles
            for f in open_files:
                try:
                    f.close()
                except Exception:
                    pass

        return total_lines

    def _parse_log_lines(
        self,
        line_iterator: Iterator[str],
        source_file: str,
        log_type: LogType,
        current_year: int
    ) -> Iterator[LogLine]:
        """
        Parse log lines from an iterator, yielding LogLine objects.

        Args:
            line_iterator: Iterator of raw log lines
            source_file: Source filename for tracking
            log_type: Log type for timestamp parsing
            current_year: Year to use for parsing

        Yields:
            LogLine objects with parsed timestamps
        """
        line_number = 0
        last_timestamp = None

        for line in line_iterator:
            line_number += 1
            line = line.rstrip('\n\r')

            if not line.strip():
                continue

            # Try to parse timestamp
            timestamp = TimestampParser.parse_line(line, log_type, current_year)

            if timestamp:
                last_timestamp = timestamp
            elif last_timestamp:
                # Multi-line log entry - use timestamp from previous line
                timestamp = last_timestamp
            else:
                # No timestamp and no previous - skip or use epoch
                timestamp = datetime(1970, 1, 1, tzinfo=UTC)

            yield LogLine(
                timestamp=timestamp,
                original_line=line,
                source_file=source_file,
                line_number=line_number,
            )

    def group_sqlite_databases(self):
        """
        Phase 4: Group SQLite databases by schema fingerprint.

        Extracts schema from each database and groups fragments with matching schemas.
        Filters out large library databases (>50MB) as they're typically not forensic artifacts.
        """
        self.log("=" * 70)
        self.log("Phase 4: Grouping SQLite Databases by Schema")
        self.log("=" * 70)

        if "uncategorized" not in self.sqlite_dbs or not self.sqlite_dbs["uncategorized"]:
            self.log("No SQLite databases to process")
            return

        uncategorized = self.sqlite_dbs["uncategorized"]
        self.log(f"Processing {len(uncategorized)} SQLite databases...")

        # Extract schemas and group by schema hash
        schema_groups: dict[str, list[tuple[RecoveredFile, SQLiteSchema]]] = defaultdict(list)
        skipped_large = 0
        skipped_error = 0

        for recovered in uncategorized:
            schema = self.extract_sqlite_schema(recovered.source_path, max_size_mb=50)

            if not schema:
                size_mb = recovered.size / (1024 * 1024)
                if size_mb > 50:
                    skipped_large += 1
                else:
                    skipped_error += 1
                continue

            # Group by schema hash
            schema_groups[schema.schema_hash].append((recovered, schema))

        # Report findings
        self.log(f"\nFound {len(schema_groups)} unique database schemas")
        self.log(f"Skipped {skipped_large} large databases (>50MB, likely system libraries)")
        if skipped_error > 0:
            self.log(f"Skipped {skipped_error} databases (errors or empty)")

        # Print detailed schema information
        for idx, (schema_hash, group) in enumerate(sorted(schema_groups.items(), key=lambda x: -len(x[1])), 1):
            fragment_count = len(group)
            first_recovered, first_schema = group[0]

            # Get representative info from first database in group
            tables_str = ", ".join(first_schema.tables[:3])
            if len(first_schema.tables) > 3:
                tables_str += f", ... ({len(first_schema.tables)} total)"

            total_size = sum(r.size for r, _ in group)
            total_size_mb = total_size / (1024 * 1024)

            self.log(f"\n  Schema {idx}: {fragment_count} fragment(s), {total_size_mb:.1f}MB total")
            self.log(f"    Tables: {tables_str}")
            self.log(f"    Hash: {schema_hash[:16]}...")

            # Store grouped databases with meaningful key
            group_key = f"schema_{schema_hash[:8]}_{first_schema.tables[0] if first_schema.tables else 'unknown'}"
            self.sqlite_dbs[group_key] = [r for r, _ in group]

            # Show fragment details in verbose mode
            if self.verbose and fragment_count > 1:
                for recovered, schema in group:
                    total_rows = sum(c for c in schema.row_counts.values() if c > 0)
                    size_mb = recovered.size / (1024 * 1024)
                    self.log(f"      - {recovered.source_path.name}: {size_mb:.1f}MB, {total_rows:,} rows", "DEBUG")

        # Remove uncategorized now that we've grouped them
        del self.sqlite_dbs["uncategorized"]

    def scan_and_classify(self):
        """
        Phase 1: Scan all PhotoRec files and classify them.

        This is the entry point for processing.
        """
        self.log("=" * 70)
        self.log("Phase 1: Scanning and Classifying Files")
        self.log("=" * 70)

        files = self.iter_photorec_files()
        total = len(files)

        if total == 0:
            self.log("No files found to process", "WARNING")
            return

        self.log(f"Processing {total} files...")

        for idx, file_path in enumerate(files, 1):
            self.stats["total_files"] += 1

            if idx % 100 == 0 or idx == total:
                percent = (idx / total) * 100
                print(f"\r  Progress: {idx}/{total} ({percent:.1f}%)", end="", flush=True)

            recovered = self.classify_file(file_path)

            if not recovered:
                self.stats["unknown"] += 1
                continue

            self.stats["classified"] += 1

            # Sort into appropriate bucket
            if recovered.file_type in (
                LogType.WIFI_LOG,
                LogType.SYSTEM_LOG,
                LogType.INSTALL_LOG,
            ):
                self.text_logs[recovered.file_type].append(recovered)
                self.stats["text_logs"] += 1

            elif recovered.file_type == LogType.SQLITE:
                # SQLite DB - will categorize by schema later
                self.sqlite_dbs["uncategorized"].append(recovered)
                self.stats["sqlite_dbs"] += 1

            else:
                # Other artifacts (plists, JSON, ASL, etc.)
                self.other_files[recovered.file_type].append(recovered)
                self.stats["other"] += 1

        print()  # Newline after progress
        self.log("Classification complete")
        self.print_stats()

    def print_stats(self):
        """Print processing statistics."""
        print()
        self.log(f"Total files scanned: {self.stats['total_files']}")
        self.log(f"  ✓ Classified: {self.stats['classified']}")
        self.log(f"  ? Unknown: {self.stats['unknown']}")
        print()
        self.log(f"Recovered artifacts:")
        self.log(f"  Text logs: {self.stats['text_logs']}")
        self.log(f"  SQLite DBs: {self.stats['sqlite_dbs']}")
        self.log(f"  Other: {self.stats['other']}")

        if self.text_logs:
            print()
            self.log("Text log breakdown:")
            for log_type, files in sorted(self.text_logs.items()):
                self.log(f"  {log_type.value}: {len(files)} files")

        if self.other_files:
            print()
            self.log("Other artifacts breakdown:")
            for file_type, files in sorted(self.other_files.items()):
                self.log(f"  {file_type.value}: {len(files)} files")

    def save_report(self):
        """Save comprehensive processing report."""
        report_path = self.output_dir / "photorec_processing_report.json"

        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "photorec_dir": str(self.photorec_dir),
            "output_dir": str(self.output_dir),
            "min_confidence": self.min_confidence,
            "statistics": self.stats,
            "text_logs": {
                log_type.value: [f.to_dict() for f in files]
                for log_type, files in self.text_logs.items()
            },
            "sqlite_dbs": {
                category: [f.to_dict() for f in files]
                for category, files in self.sqlite_dbs.items()
            },
            "other_files": {
                file_type.value: [f.to_dict() for f in files]
                for file_type, files in self.other_files.items()
            },
            "errors": self.errors,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"Processing report saved to {report_path}", "SUCCESS")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PhotoRec carved output for macOS forensic artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PhotoRec output
  python photorec_processor.py --photorec-dir ./PhotoRec_Output --output-dir ./Recovered

  # Lower confidence threshold for more aggressive recovery
  python photorec_processor.py --photorec-dir ./PhotoRec_Output --output-dir ./Recovered --min-confidence 0.5

  # Verbose mode
  python photorec_processor.py --photorec-dir ./PhotoRec_Output --output-dir ./Recovered --verbose
        """,
    )

    parser.add_argument(
        "--photorec-dir",
        type=Path,
        required=True,
        help="PhotoRec output directory (containing recup_dir.* folders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for recovered artifacts",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for classification (default: 0.6)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.photorec_dir.exists():
        print(f"❌ PhotoRec directory not found: {args.photorec_dir}", file=sys.stderr)
        sys.exit(1)

    # Create processor
    processor = PhotoRecProcessor(
        photorec_dir=args.photorec_dir,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        verbose=args.verbose,
    )

    # Run phase 1: scan and classify
    processor.scan_and_classify()

    # Run phase 2: decompress and recover archives
    processor.process_archives()

    # Run phase 3: merge text logs chronologically
    processor.merge_text_logs()

    # Run phase 4: group SQLite databases by schema
    processor.group_sqlite_databases()

    # Save report
    processor.save_report()

    print()
    processor.log("Processing complete!", "SUCCESS")


if __name__ == "__main__":
    main()
