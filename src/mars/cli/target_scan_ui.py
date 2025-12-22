#!/usr/bin/env python3
"""
Target Scan UI - Candidate/carved file scanning interface.

Handles scanning of carved/recovered files against exemplar databases.
Extracted from app.py to improve modularity.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mars.config import ConfigLoader

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"


class TargetScanUI:
    """UI for scanning carved/recovered files against exemplars."""

    def __init__(self, console: Console, project: MARSProject):
        """
        Initialize Target Scan UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project

    def run_scan(
        self,
        target_path: Path,
        exemplar_scan: dict | None = None,
        imported_package_path: Path | None = None,
        imported_package_name: str | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Run raw scanner on target directory.

        Args:
            target_path: Path to directory containing carved files
            exemplar_scan: Selected exemplar scan dictionary (for project exemplar)
            imported_package_path: Path to imported exemplar package (alternative to exemplar_scan)
            imported_package_name: Display name of imported package
            show_header_callback: Optional callback to display project header
        """
        from mars.pipeline.output.structure import OutputStructure

        # Must have either exemplar_scan or imported_package_path
        if exemplar_scan is None and imported_package_path is None:
            self.console.print("[bold red]Error: No exemplar source specified.[/bold red]")
            return

        if show_header_callback:
            show_header_callback()

        # Show brief confirmation prompt
        if imported_package_path:
            exemplar_display = f"Imported: {imported_package_name or imported_package_path.name}"
        elif exemplar_scan is not None:
            exemplar_display = exemplar_scan["output_dir"]
        else:
            exemplar_display = "Unknown"

        scan_target_panel = Panel(
            f"[{BDSB1}]Scan Carved Files[/{BDSB1}]\nTarget: {target_path}\nExemplar: {exemplar_display}",
            border_style="light_goldenrod2",
            title=f"[{BDSB1}]Confirm Candidate Scan[/{BDSB1}]",
            padding=(1, 1),
        )
        self.console.print(scan_target_panel)

        # Prompt for optional description
        description = (
            Prompt.ask(
                "\n[cyan]Enter optional description (or press Enter to skip)[/cyan]",
                default="",
            ).strip()
            or None
        )

        if not Confirm.ask("\n[bold cyan]Start scan?[/bold cyan]", default=True):
            return

        if show_header_callback:
            show_header_callback()
        self.console.print(
            Panel(
                f"[{BDSB1}]Loading...[/{BDSB1}]",
                border_style=f"{DSB3}",
            )
        )
        scan_id = None
        start_time = time.time()

        try:
            # Process recovered files using RawFileProcessor
            from mars.config import ProjectPaths
            from mars.pipeline.raw_scanner.candidate_orchestrator import (
                RawFileProcessor,
            )

            # Determine exemplar source and set up paths accordingly
            if imported_package_path:
                # Using imported exemplar package
                # Create output structure for imported package scans
                # Output goes to project output dir with "Imported" as case name
                output = OutputStructure(
                    base_output_dir=self.project.project_dir / "output",
                    case_name=f"Imported_{imported_package_name or 'Package'}",
                    case_number=self.project.config.get("case_number"),
                    examiner_name=self.project.config.get("examiner_name"),
                )
                exemplar_output_dir = output.root

                # Create the directory structure
                output.create(workflow="candidate")

                # Create candidates run directory with descriptions
                candidates_run_dir = output.create_candidate_run(
                    exemplar_description=f"Imported package: {imported_package_name}",
                    candidate_description=description,
                )

                # Start tracking recovery scan in database
                # Use -1 as sentinel for imported exemplar (no project exemplar scan)
                scan_id = self.project.start_recovery_scan(
                    target_path=target_path,
                    exemplar_scan_id=-1,  # Sentinel for imported exemplar
                    output_dir=candidates_run_dir,
                    description=description or f"Imported: {imported_package_name}",
                )

                # For imported packages, use the package's catalog directory
                # Imported packages have: schemas/, catalog/
                exemplar_dbs_dir = imported_package_path

            else:
                # Using project exemplar scan
                assert exemplar_scan is not None  # Checked at function start
                exemplar_output_dir = self.project.project_dir / "output" / exemplar_scan["output_dir"]

                output = OutputStructure.from_existing(exemplar_output_dir)

                # Create candidates run directory with descriptions
                candidates_run_dir = output.create_candidate_run(
                    exemplar_description=exemplar_scan.get("description"),
                    candidate_description=description,
                )

                # Start tracking recovery scan in database
                scan_id = self.project.start_recovery_scan(
                    target_path=target_path,
                    exemplar_scan_id=exemplar_scan["id"],
                    output_dir=candidates_run_dir,
                    description=description,
                )

                # Set up exemplar path for db_variant_selector
                exemplar_dbs_dir = exemplar_output_dir / "exemplar" / "databases"

            run_timestamp = candidates_run_dir.name
            candidate_paths = ProjectPaths.create_candidate_run(
                candidates_root=output.paths.candidates,
                run_name=run_timestamp,
                output_config=None,
            )

            # Load config for processing
            config = ConfigLoader.load(project_dir=self.project.project_dir)

            # Create processor and run all processing steps
            processor = RawFileProcessor(
                input_dir=target_path,
                paths=candidate_paths,
                config=config,
                exemplar_db_dir=exemplar_dbs_dir,
            )

            # Clear screen and show header
            if show_header_callback:
                show_header_callback()

            # Create header panel
            header_table = Table(show_header=False, box=None, padding=(0, 1))
            header_table.add_column(style="bold cyan", width=12)
            header_table.add_column(style="deep_sky_blue4")
            header_table.add_row("Target:", str(target_path))
            header_table.add_row("Exemplar:", exemplar_display)

            header = Panel(
                header_table,
                title=f"[{BDSB1}]Scanning Candidate Files[/{BDSB1}]",
                border_style=f"{DSB3}",
            )
            self.console.print(header)

            # Run all processing (SQLite + JSONLZ4 + future processors)
            # Progress bars will display automatically during processing
            processor.process_all(self.console)

            # Extract stats from processor
            stats = {
                "total_found": processor.stats["sqlite_dbs_found"],
                "matched_total": processor.stats["sqlite_dbs_matched"],
                "matched_nonempty": processor.stats["sqlite_dbs_matched_nonempty"],
                "unmatched": processor.stats["sqlite_dbs_unmatched"],
            }

            # Calculate duration
            duration = time.time() - start_time

            # Show results and generate report
            self._display_results(
                stats,
                processor.stats,
                run_dir=candidates_run_dir,
                output_dir=exemplar_output_dir,
                description=description,
                show_header_callback=show_header_callback,
            )

            # Complete scan tracking with results
            if scan_id:
                self.project.complete_recovery_scan(
                    scan_id=scan_id,
                    databases_found=stats.get("total_found", 0),
                    databases_matched=stats.get("matched_nonempty", 0),
                    databases_unmatched=stats.get("unmatched", 0),
                    duration_seconds=duration,
                )

            self.project.log_operation(
                operation="target_scan",
                status="success",
                details=f"Scanned {stats.get('total_found', 0)} databases from {target_path}",
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()

            # Show error in TUI
            self.console.print(f"\n[bold red]Error during scan:[/bold red] {e}")
            self.console.print(f"\nTrace: {error_details}")

            # Mark scan as failed
            if scan_id:
                self.project.fail_recovery_scan(scan_id, str(e))

            self.project.log_operation(
                operation="target_scan",
                status="error",
                error=str(e),
            )

    def _display_results(
        self,
        stats: dict,
        processor_stats: dict,
        run_dir: Path | None = None,
        output_dir: Path | None = None,
        description: str | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Display scan results and generate HTML report.

        Args:
            stats: Database matching statistics
            processor_stats: Full processor statistics
            run_dir: Candidate run directory (for report output)
            output_dir: Exemplar output directory
            description: Optional scan description
            show_header_callback: Optional callback to display project header
        """
        if show_header_callback:
            show_header_callback()
        self.console.print(
            Panel(
                "[bold dark_sea_green4]Candidate Scan Complete[/bold dark_sea_green4]",
                border_style="green",
            )
        )

        # Primary database matching summary
        summary = Table(show_header=True, header_style=f"{BDSB1}")
        summary.add_column("Database Matching", style=f"{DSB1}")
        summary.add_column("Count", justify="right", style="green")

        summary.add_row("SQLite Databases Found", str(stats.get("total_found", 0)))
        summary.add_row("Matched to Exemplar", str(stats.get("matched_total", 0)))
        summary.add_row("Matched with Data", str(stats.get("matched_nonempty", 0)))
        summary.add_row("Unmatched Databases", str(stats.get("unmatched", 0)))

        self.console.print(summary)

        # File categorization breakdown
        self.console.print("\n[bold]Artifacts Collected:[/bold]")
        artifact_table = Table(show_header=False, box=None, padding=(0, 2))
        artifact_table.add_column("Label1", style="dim", width=20)
        artifact_table.add_column("Value1", style="cyan", width=8, justify="right")
        artifact_table.add_column("Label2", style="dim", width=20)
        artifact_table.add_column("Value2", style="cyan", width=8, justify="right")

        artifact_table.add_row(
            "Total Files Scanned:",
            f"{processor_stats.get('total_files', 0):,}",
            "Files Classified:",
            f"{processor_stats.get('classified', 0):,}",
        )
        artifact_table.add_row(
            "SQLite Databases:",
            str(processor_stats.get("sqlite_dbs", 0)),
            "Text Logs:",
            str(processor_stats.get("text_logs", 0)),
        )
        artifact_table.add_row(
            "WiFi Plists:",
            str(processor_stats.get("wifi_plists_kept", 0)),
            "ASL Logs:",
            str(processor_stats.get("asl_logs_kept", 0)),
        )
        artifact_table.add_row(
            "JSONLZ4 Files:",
            str(processor_stats.get("jsonlz4_kept", 0)),
            "Ignored Files:",
            f"{processor_stats.get('unknown', 0):,}",
        )

        self.console.print(artifact_table)

        # Disk space saved (if plists were skipped)
        plists_skipped = processor_stats.get("plists_skipped", 0)
        if plists_skipped > 0:
            saved_gb = processor_stats.get("plists_skipped_bytes", 0) / (1024**3)
            self.console.print(
                f"\n[dim]Space saved: {saved_gb:.1f} GB ({plists_skipped:,} non-WiFi plists skipped)[/dim]"
            )

        # Generate HTML report
        if run_dir and output_dir:
            from mars.cli.scan_report_generator import ScanReportGenerator

            report_gen = ScanReportGenerator(
                project_root=self.project.project_dir,
                project_name=self.project.config.get("project_name", "Unknown"),
            )
            report_gen.generate_candidate_report(
                stats=stats,
                processor_stats=processor_stats,
                run_dir=run_dir,
                output_dir=output_dir,
                description=description or "",
                console=self.console,
            )
