#!/usr/bin/env python3
"""
Post-Scan HTML Report Generator.

Generates self-contained HTML reports after exemplar, candidate, and free scans
with summary statistics, folder explanations, and clickable links to help users
navigate MARS's output structure.
"""

from __future__ import annotations

import re
import webbrowser
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger
from mars.utils.platform_utils import get_logo_data_uri

if TYPE_CHECKING:
    from rich.console import Console


# Folder explanations for the report
FOLDER_EXPLANATIONS: dict[str, dict[str, str]] = {
    # Exemplar folders
    "catalog": {
        "title": "Catalog",
        "desc": "Databases matched by table & column structure with the macOS catalog.",
        "scan_types": ["exemplar"],
    },
    "originals": {
        "title": "Originals",
        "desc": "Individual database files that were combined and moved to catalog.",
        "scan_types": ["exemplar"],
    },
    "encrypted": {
        "title": "Encrypted",
        "desc": "Databases that cannot be opened for rubric creation due to suspected encryption.",
        "scan_types": ["exemplar"],
    },
    "schemas": {
        "title": "Schemas",
        "desc": "Database schemas and rubrics. Exemplar: catalog schemas. Candidate: metamatch rubrics.",
        "scan_types": ["exemplar", "candidate"],
    },
    "logs": {
        "title": "Logs",
        "desc": "Text log files (system.log, wifi.log, unified logs, etc.) Also contains mock logarchive.",
        "scan_types": ["exemplar", "candidate"],
    },
    "caches": {
        "title": "Caches",
        "desc": "Cache files (Firefox cache, JSONLZ4, etc.).",
        "scan_types": ["exemplar", "candidate"],
    },
    "keychains": {
        "title": "Keychains",
        "desc": "macOS keychain files (.keychain, .keychain-db).",
        "scan_types": ["exemplar", "candidate"],
    },
    # Candidate folders
    "selected_variants": {
        "title": "Selected Variants",
        "desc": "Raw database files optimized for data recovery if possible.",
        "scan_types": ["candidate"],
    },
    "catalog_candidate": {
        "title": "Catalog",
        "desc": "Databases matched by table & column structure with exemplar.",
        "scan_types": ["candidate"],
    },
    "empty": {
        "title": "Empty",
        "desc": "Candidate databases that matched an exemplar but contain no usable data.",
        "scan_types": ["candidate"],
    },
    "metamatches": {
        "title": "Metamatches",
        "desc": "Databases that don't match an exemplar but contain data, combined with like databases. Folder names include a table name to help identification.",
        "scan_types": ["candidate"],
    },
    "found_data": {
        "title": "Found Data",
        "desc": "Tables salvaged from non-corrupt databases that couldn't be auto-categorized. May contain database hints.",
        "scan_types": ["candidate"],
    },
    "carved": {
        "title": "Carved",
        "desc": "Byte-carved databases too corrupt to salvage. Visual inspection may find artifacts. May contain database hints.",
        "scan_types": ["candidate"],
    },
    "lost_and_found": {
        "title": "Lost and Found",
        "desc": "Rows rejected by rubric, or remnant data associated with the database that couldn't be auto-classified.",
        "scan_types": ["candidate"],
    },
    # Shared
    "reports": {
        "title": "Reports",
        "desc": "Analysis reports, visualizations, and module outputs.",
        "scan_types": ["exemplar", "candidate"],
    },
}  # pyright: ignore[reportAssignmentType]


def _sanitize_filename(text: str, max_length: int = 30) -> str:
    """
    Sanitize text for use in filename.

    Args:
        text: Text to sanitize
        max_length: Maximum length of result

    Returns:
        Sanitized string: lowercase, spaces to underscores, special chars removed
    """
    if not text:
        return ""
    # Lowercase and replace spaces/special chars
    sanitized = re.sub(r"[^a-z0-9_-]", "_", text.lower())
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Truncate
    return sanitized[:max_length]


def _get_version() -> str:
    """Get MARS version string."""
    try:
        from importlib.metadata import version

        return version("mars")
    except Exception:
        return "dev"


class ScanReportGenerator:
    """Generate post-scan HTML reports."""

    def __init__(self, project_root: Path, project_name: str):
        """
        Initialize report generator.

        Args:
            project_root: Root directory of the project
            project_name: Name of the project
        """
        self.project_root = project_root
        self.project_name = project_name
        self.reports_dir = project_root / "output"

    def generate_exemplar_report(
        self,
        results: dict,
        output_dir: Path,
        description: str = "",
        console: Console | None = None,
    ) -> Path | None:
        """
        Generate HTML report for an exemplar scan.

        Args:
            results: Results dictionary from ExemplarScanner.scan()
            output_dir: The exemplar output directory (e.g., MARS_{name}_{timestamp})
            description: Optional scan description
            console: Optional Rich console for prompting

        Returns:
            Path to generated report, or None if failed
        """
        try:
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Build filename with description
            desc_part = f"_{_sanitize_filename(description)}" if description else ""
            filename = f"exemplar_scan{desc_part}.html"
            report_path = reports_dir / filename

            # Extract stats from results
            found_dbs = results.get("found_databases", [])
            type_counts = {"database": 0, "log": 0, "cache": 0, "keychain": 0}
            for db in found_dbs:
                ft = db.get("file_type", "database")
                type_counts[ft] = type_counts.get(ft, 0) + 1

            # Encrypted count excludes keychains (they have their own count/folder)
            encrypted_count = sum(
                1 for db in found_dbs if db.get("is_encrypted", False) and db.get("file_type") != "keychain"
            )
            sqlite_count = sum(1 for db in found_dbs if db.get("is_sqlite", False))

            stats = {
                "total_found": results.get("total_found", 0),
                "schemas_generated": results.get("schemas_generated", 0),
                "rubrics_generated": results.get("rubrics_generated", 0),
                "total_failed": results.get("total_failed", 0),
                "sqlite_count": sqlite_count,
                "encrypted_count": encrypted_count,
                "log_count": type_counts.get("log", 0),
                "cache_count": type_counts.get("cache", 0),
                "keychain_count": type_counts.get("keychain", 0),
                "combined_databases": results.get("combined_databases", 0),
            }

            # Build relative folder paths for exemplar scan
            folders = self._get_exemplar_folders(output_dir)

            # Find module reports in reports subdirectories
            module_reports = self._find_module_reports(reports_dir)

            # Generate HTML
            html = self._build_exemplar_html(
                stats=stats,
                description=description,
                output_dir=output_dir,
                folders=folders,
                module_reports=module_reports,
            )

            report_path.write_text(html, encoding="utf-8")
            logger.info("\n[bold blue]Exemplar scan report generated.[/bold blue]")
            logger.debug(f"Exemplar scan report generated: {report_path}")

            # Update index page
            self._update_index_page(output_dir.parent)

            # Prompt to open in browser
            self._prompt_open_browser(report_path, console)

            return report_path

        except Exception as e:
            logger.error(f"Failed to generate exemplar report: {e}")
            return None

    def generate_candidate_report(
        self,
        stats: dict,
        processor_stats: dict,
        run_dir: Path,
        output_dir: Path,
        description: str = "",
        console: Console | None = None,
    ) -> Path | None:
        """
        Generate HTML report for a candidate scan.

        Args:
            stats: Database matching statistics
            processor_stats: Full processor statistics
            run_dir: The candidate run directory (e.g., candidates/20251120_123456)
            output_dir: The exemplar output directory root
            description: Optional scan description
            console: Optional Rich console for prompting

        Returns:
            Path to generated report, or None if failed
        """
        try:
            # Reports go in the candidate run's reports folder
            reports_dir = run_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Build filename with timestamp and description
            timestamp = run_dir.name  # e.g., "20251120_123456"
            desc_part = f"_{_sanitize_filename(description)}" if description else ""
            filename = f"candidate_{timestamp}{desc_part}.html"
            report_path = reports_dir / filename

            # Build relative folder paths for candidate scan
            folders = self._get_candidate_folders(run_dir)

            # Find module reports
            module_reports = self._find_module_reports(reports_dir)

            # Find exemplar report if it exists
            exemplar_report = self._find_exemplar_report(output_dir)

            # Generate HTML
            html = self._build_candidate_html(
                stats=stats,
                processor_stats=processor_stats,
                description=description,
                run_dir=run_dir,
                folders=folders,
                module_reports=module_reports,
                exemplar_report_path=exemplar_report,
            )

            report_path.write_text(html, encoding="utf-8")
            logger.info("\n[bold blue]Candidate scan report generated.[/bold blue]")
            logger.debug(f"Candidate scan report generated: {report_path}")

            # Update index page
            self._update_index_page(output_dir.parent)

            # Prompt to open in browser
            self._prompt_open_browser(report_path, console)

            return report_path

        except Exception as e:
            logger.error(f"Failed to generate candidate report: {e}")
            return None

    def generate_free_report(
        self,
        scan_type: str,
        stats: dict,
        output_dir: Path,
        description: str = "",
        console: Console | None = None,
    ) -> Path | None:
        """
        Generate HTML report for a free scan (exemplar, candidate, or salvage).

        Args:
            scan_type: Type of free scan ("free_exemplar", "free_candidate", "salvage")
            stats: Statistics dictionary
            output_dir: The output directory for the scan
            description: Optional scan description
            console: Optional Rich console for prompting

        Returns:
            Path to generated report, or None if failed
        """
        try:
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Build filename
            desc_part = f"_{_sanitize_filename(description)}" if description else ""
            filename = f"{scan_type}{desc_part}.html"
            report_path = reports_dir / filename

            # Find module reports
            module_reports = self._find_module_reports(reports_dir)

            # Generate HTML based on scan type
            html = self._build_free_html(
                scan_type=scan_type,
                stats=stats,
                description=description,
                output_dir=output_dir,
                module_reports=module_reports,
            )

            report_path.write_text(html, encoding="utf-8")
            logger.info(f"Free scan report generated: {report_path}")

            # Update index page
            self._update_index_page(output_dir.parent)

            # Prompt to open in browser
            self._prompt_open_browser(report_path, console)

            return report_path

        except Exception as e:
            logger.error(f"Failed to generate free scan report: {e}")
            return None

    def _get_exemplar_folders(self, output_dir: Path) -> list[dict]:
        """Get folder information for exemplar scan."""
        exemplar_dir = output_dir / "exemplar"
        db_dir = exemplar_dir / "databases"

        folders = []
        folder_checks = [
            ("catalog", db_dir / "catalog"),
            ("originals", db_dir / "originals"),
            ("encrypted", db_dir / "encrypted"),
            ("schemas", db_dir / "schemas"),
            ("logs", exemplar_dir / "logs"),
            ("caches", exemplar_dir / "caches"),
            ("keychains", exemplar_dir / "keychains"),
            ("reports", output_dir / "reports"),
        ]

        for key, path in folder_checks:
            if path.exists() and any(path.iterdir()):
                info = FOLDER_EXPLANATIONS.get(key, {"title": key.title(), "desc": ""})
                rel_path = path.relative_to(output_dir)
                # Prepend ../ because report is inside reports/ subdirectory
                folders.append(
                    {
                        "name": info["title"],
                        "path": f"../{rel_path}",
                        "desc": info["desc"],
                        "exists": True,
                    }
                )

        return folders

    def _get_candidate_folders(self, run_dir: Path, report_base_dir: Path | None = None) -> list[dict]:
        """Get folder information for candidate scan.

        Args:
            run_dir: The candidate run directory containing databases/
            report_base_dir: Base directory where report is saved (for relative path calc).
                           If None, assumes report is in run_dir/reports/
        """
        db_dir = run_dir / "databases"

        folders = []
        folder_checks = [
            ("catalog_candidate", db_dir / "catalog"),
            ("selected_variants", db_dir / "selected_variants"),
            ("metamatches", db_dir / "metamatches"),
            ("found_data", db_dir / "found_data"),
            ("carved", db_dir / "carved"),
            ("empty", db_dir / "empty"),
            ("schemas", db_dir / "schemas"),
            ("logs", run_dir / "logs"),
            ("caches", run_dir / "caches"),
            ("keychains", run_dir / "keychains"),
            ("reports", run_dir / "reports"),
        ]

        # Determine base for relative path calculation
        # Report is saved in {report_base_dir}/reports/ or {run_dir}/reports/
        path_base = report_base_dir if report_base_dir is not None else run_dir

        for key, path in folder_checks:
            if path.exists() and any(path.iterdir()):
                info = FOLDER_EXPLANATIONS.get(key, {"title": key.title(), "desc": ""})
                rel_path = path.relative_to(path_base)
                # Prepend ../ because report is inside reports/ subdirectory
                folders.append(
                    {
                        "name": info["title"],
                        "path": f"../{rel_path}",
                        "desc": info["desc"],
                        "exists": True,
                    }
                )

        return folders

    def _find_module_reports(self, reports_dir: Path) -> list[dict]:
        """Find module reports in subdirectories."""
        module_reports = []
        if not reports_dir.exists():
            return module_reports

        # Scan for subdirectories with reports
        for subdir in reports_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith(("_", ".")):
                # Find report files in this subdirectory
                report_files = []
                for ext in [".html", ".csv", ".json", ".ndjson", ".jsonl"]:
                    report_files.extend(subdir.glob(f"*{ext}"))

                if report_files:
                    module_reports.append(
                        {
                            "name": subdir.name.replace("_", " ").title(),
                            "path": subdir.name,
                            "files": [
                                {"name": f.name, "path": f"{subdir.name}/{f.name}"}
                                for f in sorted(report_files, key=lambda x: x.name)
                            ],
                        }
                    )

        return sorted(module_reports, key=lambda x: x["name"])

    def _find_exemplar_report(self, output_dir: Path) -> Path | None:
        """Find the exemplar report for linking from candidate reports.

        Args:
            output_dir: The exemplar output directory (e.g., output/MARS_...)

        Returns:
            Path to exemplar report if found, None otherwise
        """
        # Exemplar reports are in output_dir/reports/, not output_dir/exemplar/reports/
        exemplar_reports_dir = output_dir / "reports"
        if not exemplar_reports_dir.exists():
            return None

        # Find most recent exemplar report (by modification time)
        reports = list(exemplar_reports_dir.glob("exemplar_*.html"))
        if not reports:
            return None

        # Return the most recently modified report
        return max(reports, key=lambda p: p.stat().st_mtime)

    def _update_index_page(self, output_parent: Path) -> Path | None:
        """
        Update or create the index page linking all project scans.

        Args:
            output_parent: The output directory containing all scan folders

        Returns:
            Path to index.html or None if failed
        """
        try:
            # Collect all scan reports
            exemplar_reports = []
            candidate_reports = []

            for scan_dir in output_parent.iterdir():
                if not scan_dir.is_dir() or not scan_dir.name.startswith("MARS_"):
                    continue

                reports_dir = scan_dir / "reports"
                if reports_dir.exists():
                    # Find exemplar reports
                    for report in reports_dir.glob("exemplar_scan*.html"):
                        exemplar_reports.append(
                            {
                                "name": scan_dir.name,
                                "path": f"{scan_dir.name}/reports/{report.name}",
                                "filename": report.name,
                            }
                        )

                    # Find free scan reports
                    for report in reports_dir.glob("free_*.html"):
                        exemplar_reports.append(
                            {
                                "name": scan_dir.name,
                                "path": f"{scan_dir.name}/reports/{report.name}",
                                "filename": report.name,
                            }
                        )

                # Find candidate reports in candidates subdirectory
                candidates_dir = scan_dir / "candidates"
                if candidates_dir.exists():
                    for run_dir in candidates_dir.iterdir():
                        if run_dir.is_dir():
                            run_reports = run_dir / "reports"
                            if run_reports.exists():
                                for report in run_reports.glob("candidate_*.html"):
                                    candidate_reports.append(
                                        {
                                            "name": f"{scan_dir.name} / {run_dir.name}",
                                            "path": f"{scan_dir.name}/candidates/{run_dir.name}/reports/{report.name}",
                                            "filename": report.name,
                                        }
                                    )

            # Generate index HTML
            index_path = output_parent / "index.html"
            html = self._build_index_html(
                exemplar_reports=exemplar_reports,
                candidate_reports=candidate_reports,
            )

            index_path.write_text(html, encoding="utf-8")
            logger.debug(f"Index page updated: {index_path}")
            return index_path

        except Exception as e:
            logger.error(f"Failed to update index page: {e}")
            return None

    def _prompt_open_browser(self, report_path: Path, console: Console | None) -> None:
        """Prompt user to open report in browser."""
        if console is None:
            return

        try:
            from rich.prompt import Confirm

            if Confirm.ask("\n[cyan]Open report in browser?[/cyan]", default=True):
                import subprocess
                import sys

                if sys.platform == "darwin":
                    # Use 'open' command on macOS to respect default browser
                    subprocess.run(["open", str(report_path)], check=False)
                else:
                    webbrowser.open(f"file://{report_path}")
            return
        except Exception:
            pass  # Fail silently if no terminal

    def _build_exemplar_html(
        self,
        stats: dict,
        description: str,
        output_dir: Path,
        folders: list[dict],
        module_reports: list[dict],
    ) -> str:
        """Build HTML for exemplar scan report."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        desc_html = f'<p class="description">{description}</p>' if description else ""

        # Build folder links
        folder_html = self._build_folder_section(folders)

        # Build module reports section
        module_html = self._build_module_reports_section(module_reports)

        # Build stats cards
        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats["total_found"]}</div>
                <div class="stat-label">Total Artifacts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{stats["schemas_generated"]}</div>
                <div class="stat-label">Schemas Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["rubrics_generated"]}</div>
                <div class="stat-label">Rubrics Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warning">{stats["total_failed"]}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>

        <div class="card">
            <h3 class="card-title">Artifact Breakdown</h3>
            <table>
                <tr><td>SQLite Databases</td><td class="num">{stats["sqlite_count"]}</td></tr>
                <tr><td>Encrypted Databases</td><td class="num">{stats["encrypted_count"]}</td></tr>
                <tr><td>Log Files</td><td class="num">{stats["log_count"]}</td></tr>
                <tr><td>Cache Files</td><td class="num">{stats["cache_count"]}</td></tr>
                <tr><td>Keychain Files</td><td class="num">{stats["keychain_count"]}</td></tr>
                <tr><td>Combined Databases</td><td class="num">{stats["combined_databases"]}</td></tr>
            </table>
        </div>
        """

        return self._build_base_html(
            title="MARS: Exemplar Scan Report",
            scan_type="Exemplar Scan",
            description_html=desc_html,
            stats_html=stats_html,
            folder_html=folder_html,
            module_html=module_html,
            timestamp=timestamp,
            output_dir=output_dir,
        )

    def _build_candidate_html(
        self,
        stats: dict,
        processor_stats: dict,
        description: str,
        run_dir: Path,
        folders: list[dict],
        module_reports: list[dict],
        exemplar_report_path: Path | None = None,
    ) -> str:
        """Build HTML for candidate scan report."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build description with optional exemplar link
        desc_parts = []
        # Show candidate description first
        if description:
            desc_parts.append(f'<p class="description">{description}</p>')
        # Add exemplar link - "Exemplar:" is the clickable link
        if exemplar_report_path and exemplar_report_path.exists():
            desc_parts.append(f'<p><a class="exemplar-link" href="{exemplar_report_path}">View Exemplar Report</a></p>')
        desc_html = "\n            ".join(desc_parts)

        # Build folder links
        folder_html = self._build_folder_section(folders)

        # Build module reports section
        module_html = self._build_module_reports_section(module_reports)

        # Build stats cards
        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get("total_found", 0)}</div>
                <div class="stat-label">SQLite DBs Found</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{stats.get("matched_nonempty", 0)}</div>
                <div class="stat-label">Matched with Data</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("matched_total", 0)}</div>
                <div class="stat-label">Total Matched</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warning">{stats.get("unmatched", 0)}</div>
                <div class="stat-label">Unmatched</div>
            </div>
        </div>

        <div class="card">
            <h3 class="card-title">Artifacts Collected</h3>
            <table>
                <tr><td>Total Files Scanned</td><td class="num">{processor_stats.get("total_files", 0):,}</td></tr>
                <tr><td>Files Classified</td><td class="num">{processor_stats.get("classified", 0):,}</td></tr>
                <tr><td>SQLite Databases</td><td class="num">{processor_stats.get("sqlite_dbs", 0)}</td></tr>
                <tr><td>Text Logs</td><td class="num">{processor_stats.get("text_logs", 0)}</td></tr>
                <tr><td>WiFi Plists</td><td class="num">{processor_stats.get("wifi_plists_kept", 0)}</td></tr>
                <tr><td>ASL Logs</td><td class="num">{processor_stats.get("asl_logs_kept", 0)}</td></tr>
                <tr><td>JSONLZ4 Files</td><td class="num">{processor_stats.get("jsonlz4_kept", 0)}</td></tr>
                <tr><td>Ignored Files</td><td class="num">{processor_stats.get("unknown", 0):,}</td></tr>
            </table>
        </div>
        """

        return self._build_base_html(
            title="MARS: Candidate Scan Report",
            scan_type="Candidate Scan",
            description_html=desc_html,
            stats_html=stats_html,
            folder_html=folder_html,
            module_html=module_html,
            timestamp=timestamp,
            output_dir=run_dir,
        )

    def _build_free_html(
        self,
        scan_type: str,
        stats: dict,
        description: str,
        output_dir: Path,
        module_reports: list[dict],
    ) -> str:
        """Build HTML for free scan report."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        desc_html = f'<p class="description">{description}</p>' if description else ""

        # Build folder links based on scan type
        if scan_type == "free_exemplar":
            folders = self._get_exemplar_folders(output_dir)
            scan_title = "Free Exemplar Scan"
        else:
            # For free candidate and salvage, check candidates subfolder
            candidates_dir = output_dir / "candidates"
            if candidates_dir.exists():
                # Find most recent run
                runs = sorted(candidates_dir.iterdir(), reverse=True)
                # Pass output_dir as report_base_dir since report is at output_dir/reports/
                folders = self._get_candidate_folders(runs[0], report_base_dir=output_dir) if runs else []
            else:
                folders = []
            scan_title = "Free Candidate Scan" if scan_type == "free_candidate" else "Salvage Operation"

        folder_html = self._build_folder_section(folders)
        module_html = self._build_module_reports_section(module_reports)

        # Build stats based on scan type
        if scan_type == "free_exemplar":
            stats_html = f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get("total_files", 0)}</div>
                    <div class="stat-label">Files Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value success">{stats.get("successful", 0)}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value warning">{stats.get("failed", 0)}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            """
        else:
            stats_html = f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get("total_files", 0)}</div>
                    <div class="stat-label">Files Scanned</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("sqlite_dbs_found", 0)}</div>
                    <div class="stat-label">SQLite DBs Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value success">{stats.get("sqlite_dbs_matched", 0)}</div>
                    <div class="stat-label">Matched</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("metamatch_unique_schemas", 0)}</div>
                    <div class="stat-label">Metamatch Groups</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get("carved_databases", 0)}</div>
                    <div class="stat-label">Carved DBs</div>
                </div>
            </div>
            """

        return self._build_base_html(
            title=f"{scan_title} Report",
            scan_type=scan_title,
            description_html=desc_html,
            stats_html=stats_html,
            folder_html=folder_html,
            module_html=module_html,
            timestamp=timestamp,
            output_dir=output_dir,
        )

    def _build_folder_section(self, folders: list[dict]) -> str:
        """Build the folder guide HTML section."""
        if not folders:
            return ""

        rows = []
        for folder in folders:
            rows.append(
                f"""
            <tr>
                <td><a href="{folder["path"]}" class="folder-link">{folder["name"]}</a></td>
                <td class="desc">{folder["desc"]}</td>
            </tr>
            """
            )

        return f"""
        <div class="card">
            <h3 class="card-title">Output Folders</h3>
            <table class="folder-table">
                <thead>
                    <tr><th>Folder</th><th>Description</th></tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _build_module_reports_section(self, module_reports: list[dict]) -> str:
        """Build the module reports HTML section."""
        if not module_reports:
            return ""

        sections = []
        for module in module_reports:
            file_links = []
            for f in module["files"]:
                file_links.append(f'<a href="{f["path"]}" class="report-link">{f["name"]}</a>')

            sections.append(
                f"""
            <div class="module-section">
                <h4>{module["name"]}</h4>
                <div class="report-links">
                    {" ".join(file_links)}
                </div>
            </div>
            """
            )

        return f"""
        <div class="card">
            <h3 class="card-title">Module Reports</h3>
            {"".join(sections)}
        </div>
        """

    def _build_index_html(
        self,
        exemplar_reports: list[dict],
        candidate_reports: list[dict],
    ) -> str:
        """Build the index page HTML."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        version = _get_version()
        logo_data_uri = get_logo_data_uri()

        # Build exemplar section
        exemplar_rows = []
        for report in exemplar_reports:
            exemplar_rows.append(
                f"""
            <tr>
                <td><a href="{report["path"]}">{report["filename"]}</a></td>
                <td class="desc">{report["name"]}</td>
            </tr>
            """
            )

        exemplar_html = (
            f"""
        <div class="card">
            <h3 class="card-title">Exemplar Scans</h3>
            <table>
                <tbody>
                    {"".join(exemplar_rows) if exemplar_rows else '<tr><td colspan="2" class="empty">No exemplar scans found</td></tr>'}
                </tbody>
            </table>
        </div>
        """
            if exemplar_rows
            else ""
        )

        # Build candidate section
        candidate_rows = []
        for report in candidate_reports:
            candidate_rows.append(
                f"""
            <tr>
                <td><a href="{report["path"]}">{report["filename"]}</a></td>
                <td class="desc">{report["name"]}</td>
            </tr>
            """
            )

        candidate_html = (
            f"""
        <div class="card">
            <h3 class="card-title">Candidate Scans</h3>
            <table>
                <tbody>
                    {"".join(candidate_rows) if candidate_rows else '<tr><td colspan="2" class="empty">No candidate scans found</td></tr>'}
                </tbody>
            </table>
        </div>
        """
            if candidate_rows
            else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name} - MARS Reports</title>
    {self._get_css()}
</head>
<body>
    {self._get_theme_toggle_script()}
    <div class="container">
      <header>
        <div
          style="
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.2rem;
          "
        >
          <div style="padding-top: 1.2rem">
            <h1>{self.project_name}</h1>
            <p class="subtitle">MARS Report Index</p>
          </div>
          <div>
            <img src="{logo_data_uri}" alt="WarpedWing Labs Logo" height="100px" />
          </div>
        </div>
      </header>

        {exemplar_html}
        {candidate_html}

        <footer>
            <p>Generated: {timestamp}</p>
            <p>MARS v{version}</p>
        </footer>
    </div>
</body>
</html>"""

    def _build_base_html(
        self,
        title: str,
        scan_type: str,
        description_html: str,
        stats_html: str,
        folder_html: str,
        module_html: str,
        timestamp: str,
        output_dir: Path,
    ) -> str:
        """Build the base HTML template."""
        version = _get_version()
        logo_data_uri = get_logo_data_uri()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {self.project_name}</title>
    {self._get_css()}
</head>
<body>
    {self._get_theme_toggle_script()}
    <div class="container">
        <header>
        <div
          style="
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.2rem;
          "
        >
          <div style="padding-top: 1.2rem">
            <h1>{title}</h1>
            <p class="subtitle">{self.project_name}</p>
            {description_html}
          </div>
          <div>
            <img src="{logo_data_uri}" alt="WarpedWing Labs Logo" height="100px" />
          </div>
        </div>
      </header>

        {stats_html}
        {folder_html}
        {module_html}

        <footer>
            <p>Output: {output_dir}</p>
            <p>Generated: {timestamp}</p>
            <p>MARS v{version}</p>
        </footer>
    </div>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get CSS styles for reports."""
        return """<style>
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #ca8a04;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --link: #2563eb;
        }
        [data-theme="dark"] {
            --primary: #60a5fa;
            --success: #4ade80;
            --warning: #fbbf24;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
            --link: #60a5fa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        header { margin-bottom: 2rem; }
        h1 {
            font-size: 1.875rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: var(--text);
        }
        .subtitle { color: var(--text-muted); margin-bottom: 0.5rem; }
        .description {
            color: var(--text-muted);
            font-style: italic;
            margin-top: 0.5rem;
        }
        .card {
            background: var(--card-bg);
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .card-title {
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        th, td {
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .num { text-align: right; font-variant-numeric: tabular-nums; }
        .desc { color: var(--text-muted); font-size: 0.85rem; }
        .empty { color: var(--text-muted); text-align: center; font-style: italic; }
        a { color: var(--link); text-decoration: none; }
        a:hover { text-decoration: underline; }
        .folder-link { font-weight: 500; }
        .report-links { display: flex; flex-wrap: wrap; gap: 0.75rem; }
        .report-link {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: var(--bg);
            border-radius: 0.375rem;
            font-size: 0.875rem;
        }
        .exemplar-link {
            display: inline-block;
            padding-top: 0.25rem;
            background: var(--accent);
            color: white;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .exemplar-link:hover { text-decoration: none; opacity: 0.9; }
        .module-section { margin-bottom: 1rem; }
        .module-section h4 {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text);
        }
        footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.875rem;
        }
        .theme-toggle {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.875rem;
            color: var(--text);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            z-index: 100;
        }
        .theme-toggle:hover { background: var(--bg); }
    </style>"""

    def _get_theme_toggle_script(self) -> str:
        """Get JavaScript for theme toggle."""
        return """<button class="theme-toggle" onclick="toggleTheme()">Light</button>
    <script>
        (function() {
            // Dark mode is default - only switch to light if explicitly saved
            const saved = localStorage.getItem('mars-theme');
            if (saved === 'light') {
                document.body.removeAttribute('data-theme');
                document.querySelector('.theme-toggle').textContent = 'Dark';
            } else {
                // Default to dark mode
                document.body.setAttribute('data-theme', 'dark');
            }
        })();
        function toggleTheme() {
            const body = document.body;
            const btn = document.querySelector('.theme-toggle');
            const isDark = body.getAttribute('data-theme') === 'dark';
            if (isDark) {
                body.removeAttribute('data-theme');
                btn.textContent = 'Dark';
                localStorage.setItem('mars-theme', 'light');
            } else {
                body.setAttribute('data-theme', 'dark');
                btn.textContent = 'Light';
                localStorage.removeItem('mars-theme');  // Dark is default, no need to store
            }
        }
    </script>"""
