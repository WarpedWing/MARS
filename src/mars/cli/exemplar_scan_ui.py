#!/usr/bin/env python3
"""
Exemplar Scan UI - Exemplar scanning and partition selection interface.

Handles exemplar scanning from directories, disk images, and archives.
Extracted from app.py to improve modularity.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from mars.config import ConfigLoader
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from mars.pipeline.mount_utils.dfvfs_glob_utils import TargetFS
    from mars.pipeline.project.manager import MARSProject

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"
BDSB3 = "bold deep_sky_blue3"


class ExemplarScanUI:
    """UI for exemplar scanning operations."""

    def __init__(self, console: Console, project: MARSProject):
        """
        Initialize Exemplar Scan UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project

    def select_partitions(
        self,
        image_path: Path,
        show_header_callback: Callable | None = None,
    ) -> list[str] | None:
        """
        Show available partitions in disk image and let user select which to scan.

        Args:
            image_path: Path to disk image
            show_header_callback: Optional callback to display project header

        Returns:
            List of partition labels to scan, or None if cancelled
        """

        if show_header_callback:
            show_header_callback()

        # Scan for available filesystems
        targets = self._scan_for_targets(image_path)
        if not targets:
            return None

        # Show available partitions
        partition_table = self._generate_partition_table(targets)

        if show_header_callback:
            show_header_callback()

        dim_example_text = "\n\n[dim]Enter partition number(s) to scan (comma-separated), or 'a' for all.\nOr, enter 'b' to return.[/dim]"
        self.console.print(
            Panel(
                Group(
                    f"[{BDSB1}]Select Partition(s) to Scan\n[/{BDSB1}]",
                    partition_table,
                    dim_example_text,
                ),
                border_style=f"{DSB3}",
                padding=(1, 1),
                title=f"[{BDSB1}]Image Partitions[/{BDSB1}]",
            )
        )

        # Let user select partition(s) - loop until valid input
        while True:
            choice = (
                Prompt.ask(
                    "\n[bold cyan]Select partition(s)[/bold cyan]",
                )
                .lower()
                .strip()
            )

            if choice == "a":
                # Scan all partitions
                return [target.label for target in targets if target.label]

            if choice == "b":
                return None  # Return to main menu

            # Parse selection (single number or comma-separated)
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                selected_labels = []
                has_invalid = False

                for idx in indices:
                    if 1 <= idx <= len(targets):
                        if targets[idx - 1].label:
                            selected_labels.append(targets[idx - 1].label)
                    else:
                        self.console.print(f"[yellow]Invalid partition number: {idx}[/yellow]")
                        has_invalid = True

                if has_invalid:
                    continue  # Retry on invalid indices

                if selected_labels:
                    return selected_labels

                self.console.print("[red]No valid partitions selected[/red]")
                continue

            except ValueError:
                self.console.print(
                    "[red]Invalid input. Enter a number, comma-separated numbers, 'a' for all, or 'b' to go back.[/red]"
                )
                continue

    def _scan_for_targets(self, image_path: Path) -> list[TargetFS] | None:
        """Scan disk image for available filesystems."""
        from mars.pipeline.mount_utils import dfvfs_glob_utils

        with self.console.status(f"[{BDSB1}]Loading targets...[/{BDSB1}]"):
            try:
                targets = dfvfs_glob_utils.scan_sources(str(image_path), {"apfs", "hfs", "tsk", "ntfs"})
            except Exception as e:
                self.console.print(f"[red]Error scanning image: {e}[/red]")
                Prompt.ask("\nPress Enter to continue")
                return None

            if not targets:
                self.console.print("[red]No partitions found in image[/red]")
                Prompt.ask("\nPress Enter to continue")
                return None
            return targets

    def _generate_partition_table(self, targets: list[TargetFS]) -> Table:
        """Generate a table showing available partitions."""
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("#", style="bold sky_blue2", width=4)
        table.add_column("Label", style="dim")
        table.add_column("Name", min_width=30)
        table.add_column("Type")
        table.add_column("Size (GB)", justify="right")

        for idx, target in enumerate(targets, 1):
            # Format size in GB
            size_str = "Unknown"
            if target.size_bytes:
                size_gb = target.size_bytes / (1024**3)
                size_str = f"{size_gb:.1f}"

            # Get volume name - highlight "- Data" volumes (contain user data on macOS 10.15+)
            name_str = Text(target.volume_name, style=BDSB1) if target.volume_name else Text("(no name)", style="dim")
            if "- Data" in name_str.plain:
                name_str = Text(f"★ {name_str}", style="bold dark_sea_green4")

            table.add_row(
                str(idx),
                target.label or "(no label)",
                name_str,
                target.type_indicator.upper(),
                size_str,
            )
        return table

    def run_scan(
        self,
        source_path: Path,
        *,
        is_image: bool = False,
        partition_labels: list[str] | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Run exemplar scanner on source.

        Args:
            source_path: Root directory to scan
            is_image: Whether source is a disk image (E01, etc.)
            partition_labels: List of partition labels to scan (for disk images)
            show_header_callback: Optional callback to display project header
        """
        from mars.pipeline.exemplar_scanner.exemplar_orchestrator import (
            ExemplarScanner,
        )
        from mars.pipeline.output.structure import OutputStructure

        if show_header_callback:
            show_header_callback()

        # Show scan info
        info_table = Table(show_header=False, box=None)
        info_table.add_column(style="bold", width=15)
        info_table.add_column()

        source_label = "Disk Image" if is_image else "Source"
        info_table.add_row(f"{source_label}:", str(source_path))
        info_table.add_row("Project:", self.project.config["project_name"])

        scan_target_panel = Panel(
            info_table,
            border_style="light_goldenrod2",
            title=f"[{BDSB1}]Confirm Exemplar Scan[/{BDSB1}]",
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

        scan_id = None
        start_time = time.time()

        try:
            # Load config for processing
            config = ConfigLoader.load(project_dir=self.project.project_dir)

            # Create output structure with case info
            output_dir = self.project.project_dir / "output"
            case_name = self.project.config["project_name"].replace(" ", "_")
            output = OutputStructure(
                output_dir,
                case_name=case_name,
                case_number=self.project.config.get("case_number"),
                examiner_name=self.project.config.get("examiner_name"),
                description=description,
            )
            output.create(workflow="exemplar")

            # Start tracking scan in database
            scan_id = self.project.start_exemplar_scan(source_path, output.root, description=description)

            # Create scanner
            scanner = ExemplarScanner(
                source_path=source_path,
                output_structure=output,
                config=config,
                console=self.console,
                is_forensic_image=is_image,
                partition_labels=partition_labels,
            )

            # Start Exemplar Scanner
            results = scanner.scan()

            # Calculate duration
            duration = time.time() - start_time

            # Show results and generate report
            self._show_scan_results(
                scanner,
                results,
                output_dir=output.root,
                description=description,
                show_header_callback=show_header_callback,
            )

            # Complete scan tracking with results
            if scan_id:
                # Get hash entries count if available
                hash_entries = 0
                hash_file = output.paths.exemplar_schemas / "exemplar_hash_lookup.json"
                if hash_file.exists():
                    try:
                        import json

                        with hash_file.open() as f:
                            hash_data = json.load(f)
                            hash_entries = len(hash_data)
                    except Exception:
                        pass

                self.project.complete_exemplar_scan(
                    scan_id=scan_id,
                    databases_found=results.get("total_found", 0),
                    schemas_generated=results.get("schemas_generated", 0),
                    rubrics_generated=results.get("rubrics_generated", 0),
                    hash_entries=hash_entries,
                    duration_seconds=duration,
                )

            self.project.log_operation(
                operation="exemplar_scan",
                status="success",
                details=f"Scanned {len(scanner.found_databases)} databases from {source_path}",
            )

        except Exception as e:
            self.console.print(f"\n[bold red]Error during scan:[/bold red] {e}")

            # Mark scan as failed
            if scan_id:
                self.project.fail_exemplar_scan(scan_id, str(e))

            self.project.log_operation(
                operation="exemplar_scan",
                status="error",
                error=str(e),
            )

    def _show_scan_results(
        self,
        scanner,
        results: dict,
        output_dir: Path | None = None,
        description: str | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Display scan results summary and generate HTML report.

        Args:
            scanner: ExemplarScanner instance
            results: Results dictionary from scan()
            output_dir: Output directory for report generation
            description: Optional scan description
            show_header_callback: Optional callback to display project header
        """
        if show_header_callback:
            show_header_callback()

        # Analyze found_databases for detailed breakdown
        found_dbs = results.get("found_databases", [])

        # Count by file_type
        type_counts = {"database": 0, "log": 0, "cache": 0, "keychain": 0}
        for db in found_dbs:
            ft = db.get("file_type", "database")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        # Count encrypted and SQLite databases
        encrypted_count = sum(1 for db in found_dbs if db.get("is_encrypted", False))
        sqlite_count = sum(1 for db in found_dbs if db.get("is_sqlite", False))

        # Primary summary table
        summary = Table(show_header=False, show_edge=False, box=None)
        summary.add_column(style=f"{DSB1}")
        summary.add_column(width=5, justify="right", style="green")
        summary.add_column(style=f"{DSB1}")
        summary.add_column(width=5, justify="right", style="green")
        summary.add_column(style=f"{DSB1}")
        summary.add_column(width=5, justify="right", style="green")

        summary.add_row(
            "Total Artifacts Found:",
            str(results.get("total_found", 0)),
            "  SQLite Databases",
            str(sqlite_count),
            "  Encrypted:",
            str(encrypted_count),
        )
        summary.add_row(
            "Schemas Generated:",
            str(results.get("schemas_generated", 0)),
            "  Log Files:",
            str(type_counts.get("log", 0)),
            "  Cache Files:",
            str(type_counts.get("cache", 0)),
        )
        summary.add_row(
            "Rubrics Generated:",
            str(results.get("rubrics_generated", 0)),
            "  Keychain Files:",
            str(type_counts.get("keychain", 0)),
            "  Combined DBs:",
            str(results.get("combined_databases", 0)),
        )

        scan_summary_group = Group(Text("Exemplar Scan Complete\n", style="bold dark_sea_green4"), summary)

        self.console.print(
            Panel(
                scan_summary_group,
                border_style="green",
            )
        )

        # Show failed databases with details (debug only - race conditions produce misleading errors)
        if scanner.failed_databases:
            logger.debug(f"Failed databases ({len(scanner.failed_databases)}):")
            for i, fail_info in enumerate(scanner.failed_databases, 1):
                db_name = fail_info.get("name", "Unknown")
                error = fail_info.get("error", "Unknown error")
                path = fail_info.get("path", "")
                logger.debug(f"  {i}. {db_name}")
                logger.debug(f"     Path: {path}")
                logger.debug(f"     Error: {error}")

        # Output location
        self.console.print("\n[bold]Output saved to:[/bold]")
        self.console.print(f"[dim]{self.project.project_dir / 'output'}[/dim]")

        # Generate HTML report
        if output_dir:
            from mars.cli.scan_report_generator import ScanReportGenerator

            report_gen = ScanReportGenerator(
                project_root=self.project.project_dir,
                project_name=self.project.config.get("project_name", "Unknown"),
            )
            report_gen.generate_exemplar_report(
                results=results,
                output_dir=output_dir,
                description=description or "",
                console=self.console,
            )
