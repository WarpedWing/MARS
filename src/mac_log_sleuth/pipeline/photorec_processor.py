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
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    def iter_photorec_files(self) -> list[Path]:
        """
        Find all files in PhotoRec output directories.

        Returns:
            List of file paths to process
        """
        files = []

        # Look for recup_dir.* folders
        recup_dirs = sorted(self.photorec_dir.glob("recup_dir.*"))

        if not recup_dirs:
            self.log(f"No recup_dir.* folders found in {self.photorec_dir}", "WARNING")
            return files

        self.log(f"Found {len(recup_dirs)} PhotoRec output directories")

        for recup_dir in recup_dirs:
            if not recup_dir.is_dir():
                continue

            dir_files = [f for f in recup_dir.iterdir() if f.is_file()]
            files.extend(dir_files)
            self.log(f"  {recup_dir.name}: {len(dir_files)} files", "DEBUG")

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
                    file_type=LogType.UNKNOWN,  # Will categorize by schema later
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

            elif recovered.file_type == LogType.UNKNOWN and recovered.reasons and "SQLite" in recovered.reasons[0]:
                # SQLite DB - will categorize by schema
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

    # Save report
    processor.save_report()

    print()
    processor.log("Processing complete!", "SUCCESS")


if __name__ == "__main__":
    main()
