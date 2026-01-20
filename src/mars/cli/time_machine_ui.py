#!/usr/bin/env python3
"""
Time Machine Scan UI - Interface for scanning Time Machine backups.

Handles:
- Time Machine volume detection and validation
- Backup enumeration and selection
- Artifact extraction from selected backups
- Running candidate pipeline on extracted files
"""

from __future__ import annotations

import time
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mars.config import ConfigLoader
from mars.pipeline.raw_scanner.tm_extractor import (
    get_extraction_summary,
    load_artifact_catalog,
    write_extraction_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject
    from mars.utils.time_machine_utils import TimeMachineBackup

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"


class TimeMachineScanUI:
    """UI for scanning Time Machine backups."""

    def __init__(self, console: Console, project: MARSProject, show_current_project_menu: Callable):
        """
        Initialize Time Machine Scan UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project
        self.show_current_project_menu = show_current_project_menu

    def select_tm_volume(self) -> Path | None:
        """
        Browse for and select backup_manifest.plist from a Time Machine volume.

        Returns:
            Path to TM volume root, or None if cancelled/invalid
        """
        from mars.cli.explorer import browse_for_file

        # Browse for backup_manifest.plist file directly
        self.show_current_project_menu()
        manifest_path = browse_for_file(
            start_path=Path("/Volumes"),
            file_filter=".plist",
            title="Select Time Machine Backup Manifest",
            explanation="Navigate to your Time Machine volume and select backup_manifest.plist",
        )

        if manifest_path is None:
            return None

        # Validate correct file was selected
        if manifest_path.name != "backup_manifest.plist":
            self.console.print(
                f"[bold red]Error:[/bold red] Please select 'backup_manifest.plist', not '{manifest_path.name}'."
            )
            return None

        # Return the volume root (parent of manifest)
        return manifest_path.parent

    def select_backups(
        self,
        backups: list[TimeMachineBackup],
    ) -> list[TimeMachineBackup] | None:
        """
        Display backup selection UI.

        Args:
            backups: List of available backups (sorted newest first)

        Returns:
            List of selected backups, or None if cancelled
        """
        self.show_current_project_menu()
        if not backups:
            self.console.print("[bold red]No backups found in the selected volume.[/bold red]")
            return None

        # Display available backups
        table = Table(
            title="[bold deep_sky_blue1]Available Time Machine Backups[/bold deep_sky_blue1]",
            show_header=True,
            header_style="bold",
            border_style="deep_sky_blue3",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Date", style="cyan")
        table.add_column("Files", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Path", style="dim")

        for idx, backup in enumerate(backups, 1):
            date_str = backup.backup_date.strftime("%Y-%m-%d %H:%M")
            files_str = f"{backup.file_count:,}" if backup.file_count else "?"
            size_str = self._format_bytes(backup.logical_size) if backup.logical_size else "?"
            path_str = backup.backup_path.name

            table.add_row(str(idx), date_str, files_str, size_str, path_str)

        self.console.print(table)
        self.console.print()

        # Selection prompt
        self.console.print("[bold]Selection options:[/bold]")
        self.console.print("  [cyan]a[/cyan] - Select all backups")
        self.console.print("  [cyan]1[/cyan] - Select specific backup by number")
        self.console.print("  [cyan]1,3,5[/cyan] - Select multiple backups by number")
        self.console.print("  [cyan]1-3[/cyan] - Select range of backups")
        self.console.print("  [cyan]q[/cyan] - Cancel")
        self.console.print()

        selection = Prompt.ask(
            "[bold cyan]Select backups to scan[/bold cyan]",
            default="a" if len(backups) <= 3 else "1",
        )

        if selection.lower() == "q":
            return None

        if selection.lower() == "a":
            return backups

        # Parse selection
        try:
            selected_indices = self._parse_selection(selection, len(backups))
            return [backups[i] for i in selected_indices]
        except ValueError as e:
            self.console.print(f"[bold red]Invalid selection:[/bold red] {e}")
            return None

    def _parse_selection(self, selection: str, max_count: int) -> list[int]:
        """Parse selection string into list of 0-based indices."""
        indices: set[int] = set()

        parts = selection.replace(" ", "").split(",")
        for part in parts:
            if "-" in part:
                # Range
                start_str, end_str = part.split("-", 1)
                start = int(start_str) - 1
                end = int(end_str) - 1
                if start < 0 or end >= max_count or start > end:
                    raise ValueError(f"Invalid range: {part}")
                indices.update(range(start, end + 1))
            else:
                # Single number
                idx = int(part) - 1
                if idx < 0 or idx >= max_count:
                    raise ValueError(f"Invalid number: {part}")
                indices.add(idx)

        return sorted(indices)

    def _format_bytes(self, size_bytes: int) -> str:
        """Format byte count as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024  # type: ignore[assignment]
        return f"{size_bytes:.1f} PB"

    def run_scan(
        self,
        tm_volume: Path,
        selected_backups: list[TimeMachineBackup],
        exemplar_scan: dict,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Execute Time Machine backup scan workflow.

        Args:
            tm_volume: Path to Time Machine volume root
            selected_backups: List of backups to scan
            exemplar_scan: Selected exemplar scan dictionary
            show_header_callback: Optional callback to display project header
        """
        from mars.config import ProjectPaths
        from mars.pipeline.output.structure import OutputStructure, create_logarchive_from_logs
        from mars.pipeline.raw_scanner.candidate_orchestrator import RawFileProcessor

        if show_header_callback:
            show_header_callback()

        # Show confirmation
        backup_dates = ", ".join(b.backup_date.strftime("%Y-%m-%d") for b in selected_backups[:3])
        if len(selected_backups) > 3:
            backup_dates += f" (+{len(selected_backups) - 3} more)"

        exemplar_display = exemplar_scan.get("description") or exemplar_scan["output_dir"]

        confirm_panel = Panel(
            f"[{BDSB1}]Time Machine Scan[/{BDSB1}]\n"
            f"Volume: {tm_volume}\n"
            f"Backups: {backup_dates}\n"
            f"Exemplar: {exemplar_display}",
            border_style="light_goldenrod2",
            title=f"[{BDSB1}]Confirm Time Machine Scan[/{BDSB1}]",
            padding=(1, 1),
        )
        self.console.print(confirm_panel)

        # Prompt for description
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
            # Set up output structure
            exemplar_output_dir = self.project.project_dir / "output" / exemplar_scan["output_dir"]
            output = OutputStructure.from_existing(exemplar_output_dir)

            # Create candidates run directory
            candidates_run_dir = output.create_candidate_run(
                exemplar_description=exemplar_scan.get("description"),
                candidate_description=description or f"Time Machine scan: {len(selected_backups)} backup(s)",
            )

            # Create extraction output directory within the candidate run
            extraction_dir = candidates_run_dir / "tm_extracted"
            extraction_dir.mkdir(parents=True, exist_ok=True)

            # Start tracking scan
            scan_id = self.project.start_recovery_scan(
                target_path=tm_volume,
                exemplar_scan_id=exemplar_scan["id"],
                output_dir=candidates_run_dir,
                description=description or f"Time Machine: {len(selected_backups)} backup(s)",
            )

            # Phase 1: Extract artifacts from TM backups
            self.console.print(
                Panel(
                    f"[{BDSB1}]Phase 1: Extracting Artifacts[/{BDSB1}]",
                    border_style=f"{DSB3}",
                )
            )

            catalog = load_artifact_catalog()
            extraction_result = self._extract_with_progress(
                selected_backups,
                extraction_dir,
                catalog,
            )

            summary = get_extraction_summary(extraction_result)

            # Write extraction manifest for the candidate pipeline
            manifest_path = write_extraction_manifest(extraction_result)

            self.console.print(
                f"\n[green]Extraction complete:[/green] "
                f"{summary['total_extracted']} artifacts, "
                f"{summary['skipped_duplicates']} duplicates skipped"
            )

            if summary["extraction_errors"]:
                self.console.print(f"[yellow]Extraction errors: {summary['extraction_errors']}[/yellow]")
                # Show details of permission errors
                permission_errors = [(p, e) for p, e in extraction_result.extraction_errors if "Permission denied" in e]
                if permission_errors:
                    self.console.print(
                        f"[yellow]  └─ {len(permission_errors)} files require elevated permissions (run with sudo)[/yellow]"
                    )
                    for path, _error in permission_errors[:3]:
                        self.console.print(f"[dim]     • {path.name}[/dim]")
                    if len(permission_errors) > 3:
                        self.console.print(f"[dim]     • ... and {len(permission_errors) - 3} more[/dim]")

            # Phase 2: Run candidate pipeline on extracted files
            self.console.print(
                Panel(
                    f"[{BDSB1}]Phase 2: Processing Candidates[/{BDSB1}]",
                    border_style=f"{DSB3}",
                )
            )

            run_timestamp = candidates_run_dir.name
            candidate_paths = ProjectPaths.create_candidate_run(
                candidates_root=output.paths.candidates,
                run_name=run_timestamp,
                output_config=None,
            )

            # Load config
            config = ConfigLoader.load(project_dir=self.project.project_dir)

            # Set up exemplar path
            exemplar_dbs_dir = exemplar_output_dir / "exemplar" / "databases"

            # Create processor - use extraction_dir as input
            processor = RawFileProcessor(
                input_dir=extraction_dir,
                paths=candidate_paths,
                config=config,
                exemplar_db_dir=exemplar_dbs_dir,
                source_type="time_machine",
                extraction_manifest=manifest_path,
            )

            # Run all processing
            processor.process_all(self.console)

            # Create logarchive from Unified Logs + UUID Text
            logarchive_path = create_logarchive_from_logs(
                logs_dir=candidate_paths.logs,
                archive_name="time_machine",
            )
            if logarchive_path:
                self.console.print(f"[green]Created logarchive:[/green] {logarchive_path.name}")

            # Extract stats
            stats = {
                "total_found": processor.stats["sqlite_dbs_found"],
                "matched_total": processor.stats["sqlite_dbs_matched"],
                "matched_nonempty": processor.stats["sqlite_dbs_matched_nonempty"],
                "unmatched": processor.stats["sqlite_dbs_unmatched"],
            }

            duration = time.time() - start_time

            # Display results
            self._display_results(
                extraction_summary=summary,
                candidate_stats=stats,
                processor_stats=processor.stats,
                run_dir=candidates_run_dir,
                duration=duration,
                show_header_callback=show_header_callback,
            )

            # Complete scan tracking
            if scan_id:
                self.project.complete_recovery_scan(
                    scan_id=scan_id,
                    databases_found=stats.get("total_found", 0),
                    databases_matched=stats.get("matched_nonempty", 0),
                    databases_unmatched=stats.get("unmatched", 0),
                    duration_seconds=duration,
                )

            self.project.log_operation(
                operation="time_machine_scan",
                status="success",
                details=f"Scanned {len(selected_backups)} backup(s), extracted {summary['total_extracted']} artifacts",
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()

            self.console.print(f"\n[bold red]Error during scan:[/bold red] {e}")
            self.console.print(f"\nTrace: {error_details}")

            if scan_id:
                self.project.fail_recovery_scan(scan_id, str(e))

            self.project.log_operation(
                operation="time_machine_scan",
                status="error",
                error=str(e),
            )

    def _extract_with_progress(
        self,
        backups: list[TimeMachineBackup],
        output_dir: Path,
        catalog: dict,
    ):
        """Extract artifacts with two-level progress display.

        Shows both:
        - Backup level: "Processing backup 1/3: 2026-01-19 13:46"
        - Artifact level: "Scanning: Safari History (45/156)"
        """
        from mars.pipeline.raw_scanner.tm_extractor import (
            ExtractionResult,
            extract_artifacts_from_backup,
            iter_catalog_entries,
        )

        # Count total catalog entries for artifact progress
        total_entries = len(list(iter_catalog_entries(catalog)))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            # Outer task: backup level
            backup_task = progress.add_task(
                f"Processing {len(backups)} backup(s)...",
                total=len(backups),
            )
            # Inner task: artifact level (within each backup)
            artifact_task = progress.add_task(
                "Initializing...",
                total=total_entries,
                visible=True,
            )

            # Shared hash dict for cross-backup deduplication
            seen_hashes: dict[str, Path] = {}
            combined = ExtractionResult(output_dir=output_dir)
            total_backups = len(backups)

            for idx, backup in enumerate(backups):
                backup_label = backup.backup_date.strftime("%Y-%m-%d %H:%M")
                progress.update(
                    backup_task,
                    completed=idx,
                    description=f"Backup {idx + 1}/{total_backups}: {backup_label}",
                )
                # Reset artifact progress for each backup
                progress.update(artifact_task, completed=0)

                def update_artifact_progress(current: int, total: int, message: str) -> None:
                    progress.update(
                        artifact_task,
                        completed=current,
                        total=total,
                        description=message,
                    )

                result = extract_artifacts_from_backup(
                    backup=backup,
                    output_dir=output_dir,
                    catalog=catalog,
                    progress_callback=update_artifact_progress,
                    seen_hashes=seen_hashes,
                )

                combined.artifacts.extend(result.artifacts)
                combined.skipped_duplicates += result.skipped_duplicates
                combined.extraction_errors.extend(result.extraction_errors)

            # Mark both tasks complete
            progress.update(backup_task, completed=len(backups), description="All backups extracted")
            progress.update(artifact_task, completed=total_entries, description="Extraction complete")

        return combined

    def _display_results(
        self,
        extraction_summary: dict,
        candidate_stats: dict,
        processor_stats: dict,
        run_dir: Path,
        duration: float,
        show_header_callback: Callable | None = None,
    ) -> None:
        """Display scan results."""
        if show_header_callback:
            show_header_callback()

        # Results table
        results_table = Table(show_header=False, box=None, padding=(0, 1))
        results_table.add_column(style="bold cyan", width=24)
        results_table.add_column(style="white")

        results_table.add_row("Artifacts Extracted:", str(extraction_summary["total_extracted"]))
        results_table.add_row("Duplicates Skipped:", str(extraction_summary["skipped_duplicates"]))
        results_table.add_row("", "")
        results_table.add_row("SQLite DBs Found:", str(candidate_stats.get("total_found", 0)))
        results_table.add_row("Matched (non-empty):", str(candidate_stats.get("matched_nonempty", 0)))
        results_table.add_row("Unmatched:", str(candidate_stats.get("unmatched", 0)))
        results_table.add_row("", "")
        results_table.add_row("Duration:", f"{duration:.1f} seconds")
        results_table.add_row("Output:", str(run_dir))

        self.console.print(
            Panel(
                results_table,
                title=f"[{BDSB1}]Time Machine Scan Complete[/{BDSB1}]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Show breakdown by type
        by_type = extraction_summary.get("by_type", {})
        if by_type:
            type_parts = [f"{t}: {c}" for t, c in sorted(by_type.items())]
            self.console.print(f"\n[dim]Artifacts by type: {', '.join(type_parts)}[/dim]")

        # Generate HTML report
        if run_dir:
            from mars.cli.scan_report_generator import ScanReportGenerator

            # Build combined stats for report
            # Keys must match what _build_candidate_html expects:
            # total_found, matched_total, matched_nonempty, unmatched
            stats = {
                "total_found": candidate_stats.get("total_found", 0),
                "matched_total": candidate_stats.get("matched_total", 0),
                "matched_nonempty": candidate_stats.get("matched_nonempty", 0),
                "unmatched": candidate_stats.get("unmatched", 0),
                "tm_artifacts_extracted": extraction_summary.get("total_extracted", 0),
                "tm_duplicates_skipped": extraction_summary.get("skipped_duplicates", 0),
                "tm_by_type": extraction_summary.get("by_type", {}),
            }

            report_gen = ScanReportGenerator(
                project_root=self.project.project_dir,
                project_name=self.project.config.get("project_name", "Unknown"),
            )
            report_gen.generate_candidate_report(
                stats=stats,
                processor_stats=processor_stats,
                run_dir=run_dir,
                output_dir=run_dir,
                description="Time Machine Backup Scan",
                console=self.console,
            )
