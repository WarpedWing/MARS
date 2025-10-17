#!/usr/bin/env python3

"""
Output Directory Structure Management
by WarpedWing Labs

Manages the organized output directory structure for Mac Log Sleuth forensic cases.
Ensures original filenames are preserved and provenance is tracked.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OutputStructure:
    """
    Manages the output directory structure for a forensic case.

    Directory layout:
        MacLogSleuth_CaseName_YYYYMMDD_HHMMSS/
        ├── index.html                          # Main case report
        ├── case_metadata.json                  # Case metadata and timeline
        ├── reports/                            # Text reports and logs
        │   ├── summary.txt
        │   ├── recovery_log.txt
        │   └── errors.txt
        ├── plots/                              # Plotly visualizations
        │   ├── timeline.html
        │   ├── data_addition.html
        │   └── field_distribution.html
        ├── databases/                          # All database files
        │   ├── combined/                       # Merged databases (only place without original names)
        │   │   ├── TCC_Combined.sqlite
        │   │   ├── Safari_History_Combined.sqlite
        │   │   └── Messages_Combined.sqlite
        │   ├── originals/                      # Unmodified source databases
        │   │   └── {original_filename}/        # Organized by original filename
        │   │       ├── {original_filename}.db
        │   │       └── {original_filename}.provenance.json
        │   ├── recovered/                      # Recovered/repaired databases
        │   │   ├── intact/                     # Databases that opened successfully
        │   │   │   └── {original_filename}_intact/
        │   │   ├── repaired/                   # sqlite3 .recover or sqlite_dissect output
        │   │   │   └── {original_filename}_repaired/
        │   │   └── carved/                     # Carver output
        │   │       └── {original_filename}_carved/
        │   └── failed/                         # Databases that couldn't be recovered
        │       └── {original_filename}_failed/
        │           ├── {original_filename}.db
        │           └── {original_filename}.failure.json
        ├── compressed/                         # Extracted compressed archives
        │   └── {original_archive_name}/
        │       ├── extraction_metadata.json
        │       └── extracted_files/
        ├── exports/                            # CSV/JSON exports
        │   └── {original_db_name}/
        │       ├── {table_name}.csv
        │       └── {table_name}.json
        ├── schemas/                            # Database schemas and rubrics
        │   └── {original_db_name}/
        │       ├── {original_db_name}_schema.sql
        │       └── {original_db_name}_rubric.json
        ├── working/                            # Temporary processing files
        │   └── {original_filename}/
        └── logs/                               # Processing logs
            ├── carver.log
            ├── recovery.log
            └── classification.log

    Key principle: All files (except combined databases) preserve their original filename
    in either the directory name or the file name itself.
    """

    def __init__(self, base_output_dir: Path | None = None, case_name: str = "Case"):
        """
        Initialize output structure.

        Args:
            base_output_dir: Parent directory for output (defaults to ./MacLogSleuth_Output)
            case_name: Name of the forensic case
        """
        if base_output_dir is None:
            base_output_dir = Path.cwd() / "MacLogSleuth_Output"

        # Create timestamped case directory
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.case_name = case_name
        self.timestamp = timestamp
        self.root = base_output_dir / f"MacLogSleuth_{case_name}_{timestamp}"

        # Define subdirectories
        self.reports_dir = self.root / "reports"
        self.plots_dir = self.root / "plots"
        self.databases_dir = self.root / "databases"
        self.combined_dir = self.databases_dir / "combined"
        self.originals_dir = self.databases_dir / "originals"
        self.recovered_dir = self.databases_dir / "recovered"
        self.intact_dir = self.recovered_dir / "intact"
        self.repaired_dir = self.recovered_dir / "repaired"
        self.carved_dir = self.recovered_dir / "carved"
        self.failed_dir = self.databases_dir / "failed"
        self.compressed_dir = self.root / "compressed"
        self.exports_dir = self.root / "exports"
        self.schemas_dir = self.root / "schemas"
        self.working_dir = self.root / "working"
        self.logs_dir = self.root / "logs"

        # Case metadata
        self.metadata = {
            "case_name": case_name,
            "created": datetime.now(UTC).isoformat(),
            "version": "2.0",
            "databases_processed": [],
            "archives_extracted": [],
            "files_recovered": [],
            "files_failed": [],
        }

    def create(self) -> Path:
        """
        Create the complete directory structure.

        Returns:
            Path to root directory
        """
        # Create all directories
        directories = [
            self.root,
            self.reports_dir,
            self.plots_dir,
            self.databases_dir,
            self.combined_dir,
            self.originals_dir,
            self.intact_dir,
            self.repaired_dir,
            self.carved_dir,
            self.failed_dir,
            self.compressed_dir,
            self.exports_dir,
            self.schemas_dir,
            self.working_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create initial case metadata file
        self.save_metadata()

        return self.root

    def save_metadata(self) -> Path:
        """Save case metadata to JSON file."""
        metadata_path = self.root / "case_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return metadata_path

    def get_original_db_dir(self, original_filename: str) -> Path:
        """
        Get directory for storing an original database file.
        Preserves original filename in directory name.

        Args:
            original_filename: Original name of the database file (with or without extension)

        Returns:
            Path to directory: originals/{original_filename}/
        """
        # Strip extension for directory name, but keep it for reference
        base_name = Path(original_filename).stem
        db_dir = self.originals_dir / base_name
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir

    def get_recovered_db_dir(
        self, original_filename: str, recovery_type: str = "intact"
    ) -> Path:
        """
        Get directory for storing a recovered database file.
        Preserves original filename in directory name.

        Args:
            original_filename: Original name of the database file
            recovery_type: Type of recovery ('intact', 'repaired', 'carved')

        Returns:
            Path to directory: recovered/{recovery_type}/{original_filename}_{recovery_type}/
        """
        base_name = Path(original_filename).stem

        if recovery_type == "intact":
            parent_dir = self.intact_dir
        elif recovery_type == "repaired":
            parent_dir = self.repaired_dir
        elif recovery_type == "carved":
            parent_dir = self.carved_dir
        else:
            raise ValueError(f"Invalid recovery_type: {recovery_type}")

        db_dir = parent_dir / f"{base_name}_{recovery_type}"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir

    def get_failed_db_dir(self, original_filename: str) -> Path:
        """
        Get directory for storing a failed database file.
        Preserves original filename in directory name.

        Args:
            original_filename: Original name of the database file

        Returns:
            Path to directory: failed/{original_filename}_failed/
        """
        base_name = Path(original_filename).stem
        db_dir = self.failed_dir / f"{base_name}_failed"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir

    def get_compressed_extraction_dir(self, archive_filename: str) -> Path:
        """
        Get directory for extracting a compressed archive.
        Preserves original archive filename in directory name.

        Args:
            archive_filename: Original name of the archive file

        Returns:
            Path to directory: compressed/{archive_filename}/
        """
        base_name = Path(archive_filename).stem
        extract_dir = self.compressed_dir / base_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        return extract_dir

    def get_export_dir(self, original_db_name: str) -> Path:
        """
        Get directory for CSV/JSON exports from a database.
        Preserves original database name in directory name.

        Args:
            original_db_name: Original name of the database

        Returns:
            Path to directory: exports/{original_db_name}/
        """
        base_name = Path(original_db_name).stem
        export_dir = self.exports_dir / base_name
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def get_schema_dir(self, original_db_name: str) -> Path:
        """
        Get directory for schema and rubric files.
        Preserves original database name in directory name.

        Args:
            original_db_name: Original name of the database

        Returns:
            Path to directory: schemas/{original_db_name}/
        """
        base_name = Path(original_db_name).stem
        schema_dir = self.schemas_dir / base_name
        schema_dir.mkdir(parents=True, exist_ok=True)
        return schema_dir

    def get_working_dir(self, original_filename: str) -> Path:
        """
        Get working directory for temporary processing files.
        Preserves original filename in directory name.

        Args:
            original_filename: Original name of the file being processed

        Returns:
            Path to directory: working/{original_filename}/
        """
        base_name = Path(original_filename).stem
        work_dir = self.working_dir / base_name
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def copy_to_originals(
        self, source_path: Path, original_filename: str | None = None
    ) -> Path:
        """
        Copy a database to the originals directory, preserving its original filename.

        Args:
            source_path: Path to the source database file
            original_filename: Optional override for original filename (defaults to source_path.name)

        Returns:
            Path to copied file
        """
        if original_filename is None:
            original_filename = source_path.name

        dest_dir = self.get_original_db_dir(original_filename)
        dest_path = dest_dir / original_filename

        shutil.copy2(source_path, dest_path)

        # Update metadata
        self.metadata["databases_processed"].append(
            {
                "original_filename": original_filename,
                "source_path": str(source_path),
                "copied_to": str(dest_path),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.save_metadata()

        return dest_path

    def copy_to_recovered(
        self,
        source_path: Path,
        original_filename: str,
        recovery_type: str = "intact",
        new_filename: str | None = None,
    ) -> Path:
        """
        Copy a recovered database to the recovered directory.
        Preserves original filename in directory structure.

        Args:
            source_path: Path to the recovered database file
            original_filename: Original name of the source database
            recovery_type: Type of recovery ('intact', 'repaired', 'carved')
            new_filename: Optional new filename (defaults to original_filename)

        Returns:
            Path to copied file
        """
        if new_filename is None:
            new_filename = original_filename

        dest_dir = self.get_recovered_db_dir(original_filename, recovery_type)
        dest_path = dest_dir / new_filename

        shutil.copy2(source_path, dest_path)

        # Update metadata
        self.metadata["files_recovered"].append(
            {
                "original_filename": original_filename,
                "recovery_type": recovery_type,
                "source_path": str(source_path),
                "copied_to": str(dest_path),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.save_metadata()

        return dest_path

    def create_provenance_file(
        self, db_path: Path, original_filename: str, provenance_data: dict[str, Any]
    ) -> Path:
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

        with open(provenance_path, "w") as f:
            json.dump(provenance_data, f, indent=2)

        return provenance_path

    def create_failure_report(
        self, original_filename: str, error_info: dict[str, Any]
    ) -> Path:
        """
        Create a failure report for a database that couldn't be recovered.
        Preserves original filename in the failure report.

        Args:
            original_filename: Original name of the failed database
            error_info: Dictionary containing error information

        Returns:
            Path to failure report
        """
        dest_dir = self.get_failed_db_dir(original_filename)
        base_name = Path(original_filename).stem
        failure_path = dest_dir / f"{base_name}.failure.json"

        # Add standard metadata
        error_info["original_filename"] = original_filename
        error_info["failed_at"] = datetime.now(UTC).isoformat()
        error_info["case_name"] = self.case_name

        with open(failure_path, "w") as f:
            json.dump(error_info, f, indent=2)

        # Update metadata
        self.metadata["files_failed"].append(
            {
                "original_filename": original_filename,
                "failure_report": str(failure_path),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.save_metadata()

        return failure_path

    def get_combined_db_path(self, db_category: str) -> Path:
        """
        Get path for a combined database file.
        This is the ONLY place where original filenames are not preserved.

        Args:
            db_category: Category name (e.g., 'TCC', 'Safari_History', 'Messages')

        Returns:
            Path to combined database: combined/{db_category}_Combined.sqlite
        """
        return self.combined_dir / f"{db_category}_Combined.sqlite"

    def get_log_path(self, log_name: str) -> Path:
        """
        Get path for a log file.

        Args:
            log_name: Name of the log (e.g., 'carver', 'recovery', 'classification')

        Returns:
            Path to log file: logs/{log_name}.log
        """
        return self.logs_dir / f"{log_name}.log"

    def get_plot_path(self, plot_name: str) -> Path:
        """
        Get path for a Plotly visualization.

        Args:
            plot_name: Name of the plot (e.g., 'timeline', 'data_addition')

        Returns:
            Path to plot file: plots/{plot_name}.html
        """
        return self.plots_dir / f"{plot_name}.html"

    def get_report_path(self, report_name: str) -> Path:
        """
        Get path for a text report.

        Args:
            report_name: Name of the report (e.g., 'summary', 'recovery_log')

        Returns:
            Path to report file: reports/{report_name}.txt
        """
        return self.reports_dir / f"{report_name}.txt"

    def create_index_html(self, content: str) -> Path:
        """
        Create the main case report HTML file.

        Args:
            content: HTML content for the index page

        Returns:
            Path to index.html
        """
        index_path = self.root / "index.html"
        with open(index_path, "w") as f:
            f.write(content)
        return index_path


def example_usage():
    """Example usage of OutputStructure class."""
    # Create output structure for a case
    output = OutputStructure(case_name="MacBookPro_2023")
    output.create()

    # Copy original database (preserves filename)
    source_db = Path("/Users/alice/Library/Safari/History.db")
    original_copy = output.copy_to_originals(source_db)
    print(f"Original copied to: {original_copy}")
    # Output: MacLogSleuth_MacBookPro_2023_20250116_143022/databases/originals/History/History.db

    # Create provenance file (preserves filename)
    provenance = output.create_provenance_file(
        db_path=original_copy,
        original_filename="History.db",
        provenance_data={
            "source_system": "MacBook Pro 2023",
            "collection_date": "2025-01-16",
            "md5": "abc123...",
        },
    )
    print(f"Provenance file: {provenance}")
    # Output: .../originals/History/History.provenance.json

    # Copy recovered database (preserves original filename in path)
    recovered_db = Path("/tmp/History_repaired.db")
    recovered_copy = output.copy_to_recovered(
        source_path=recovered_db,
        original_filename="History.db",
        recovery_type="repaired",
        new_filename="History_Repaired.db",
    )
    print(f"Recovered copied to: {recovered_copy}")
    # Output: .../recovered/repaired/History_repaired/History_Repaired.db

    # Create combined database (ONLY place without original filename)
    combined_path = output.get_combined_db_path("Safari_History")
    print(f"Combined database: {combined_path}")
    # Output: .../combined/Safari_History_Combined.sqlite

    # Create failure report (preserves original filename)
    failure_report = output.create_failure_report(
        original_filename="Messages.db",
        error_info={
            "error": "Database header corrupted beyond repair",
            "attempted_methods": ["sqlite3 .recover", "sqlite_dissect", "carver"],
        },
    )
    print(f"Failure report: {failure_report}")
    # Output: .../failed/Messages_failed/Messages.failure.json


if __name__ == "__main__":
    example_usage()
