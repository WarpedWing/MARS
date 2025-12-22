#!/usr/bin/env python3
"""
Export UI - Export data for external forensic tools.

Provides TUI interface for packaging MARS output (exemplar, candidate, or combined)
into formats compatible with mac_apt, APOLLO, plaso, and other forensic tools.

Also handles exemplar package export/import for sharing sanitized reference
exemplars without private data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mars.cli.exemplar_packager import ExemplarPackager
from mars.cli.export_packager import (
    ExportMethod,
    ExportPackager,
    ExportResult,
    ExportSource,
    ExportStructure,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"


class ExportUI:
    """UI for exporting data to external forensic tools."""

    def __init__(self, console: Console, project: MARSProject):
        """Initialize Export UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project

    def show_menu(
        self,
        exemplar_scan: dict | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """Display export menu.

        Args:
            exemplar_scan: Selected exemplar scan dict (optional)
            show_header_callback: Optional callback to display project header
        """
        while True:
            if show_header_callback:
                show_header_callback()

            self.console.print(
                Panel(
                    "[bold light_goldenrod3]Export Data for External Tools[/bold light_goldenrod3]\n\n"
                    "Package databases with canonical macOS filenames for use with\n"
                    "mac_apt, APOLLO, plaso, and other forensic analysis tools.",
                    border_style="light_goldenrod3",
                )
            )

            # Get output directory
            output_dir = self._get_output_dir(exemplar_scan)
            if not output_dir:
                self.console.print("[yellow]No scan output directory found. Run a scan first.[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                return

            # Check what's available
            has_exemplar = (output_dir / "exemplar" / "databases" / "catalog").exists()
            has_candidate = self._has_candidate_catalog(output_dir)

            if not has_exemplar and not has_candidate:
                self.console.print(
                    "[yellow]No database catalogs found. Run an exemplar or candidate scan first.[/yellow]"
                )
                Prompt.ask("\nPress Enter to continue")
                return

            # Show options
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style=f"{BDSB1}", width=4)
            table.add_column()
            table.add_column(style="dim")

            if has_exemplar:
                table.add_row(
                    "1.",
                    "[bold orange_red1]Export Exemplar[/bold orange_red1]",
                    "Databases from exemplar scan",
                )
            else:
                table.add_row(
                    "[dim]1.[/dim]",
                    "[dim]Export Exemplar[/dim]",
                    "[dim]No exemplar scan[/dim]",
                )

            if has_candidate:
                table.add_row(
                    "2.",
                    "[bold orange_red1]Export Candidate[/bold orange_red1]",
                    "Databases from candidate scan",
                )
            else:
                table.add_row(
                    "[dim]2.[/dim]",
                    "[dim]Export Candidate[/dim]",
                    "[dim]No candidate scan[/dim]",
                )

            if has_exemplar and has_candidate:
                table.add_row(
                    "3.",
                    "[bold orange_red1]Export Combined[/bold orange_red1]",
                    "Merged & deduplicated databases",
                )
            else:
                table.add_row(
                    "[dim]3.[/dim]",
                    "[dim]Export Combined[/dim]",
                    "[dim]Requires both exemplar and candidate[/dim]",
                )

            # Exemplar Package export option
            table.add_row("", "")
            if has_exemplar:
                table.add_row(
                    "4.",
                    "[bold orange_red1]Export Exemplar Package[/bold orange_red1]",
                    "Create shareable package (no private data)",
                )
            else:
                table.add_row(
                    "[dim]4.[/dim]",
                    "[dim]Export Exemplar Package[/dim]",
                    "[dim]Requires exemplar scan[/dim]",
                )

            table.add_row("", "")
            table.add_row("", "[dim](B)ack to main menu[/dim]")

            self.console.print(table)

            # Get choice
            valid = ["b"]
            if has_exemplar:
                valid.extend(["1", "4"])
            if has_candidate:
                valid.append("2")
            if has_exemplar and has_candidate:
                valid.append("3")

            sorted_choices = sorted(valid, key=lambda x: (x, x.isdigit()))

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=sorted_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                return

            # Handle exemplar package export
            if choice == "4":
                self._export_exemplar_package(output_dir, show_header_callback)
                continue

            # Map choice to source for database export
            source_map = {
                "1": ExportSource.EXEMPLAR,
                "2": ExportSource.CANDIDATE,
                "3": ExportSource.COMBINED,
            }
            source = source_map[choice]

            # Get export options and run
            result = self._run_export_wizard(output_dir, source, show_header_callback, exemplar_scan)
            if result:
                if show_header_callback:
                    show_header_callback()
                self._show_export_results(result)

    def _get_output_dir(self, exemplar_scan: dict | None) -> Path | None:
        """Get the output directory for exports.

        Args:
            exemplar_scan: Optional exemplar scan dict

        Returns:
            Path to output directory or None
        """
        if exemplar_scan and "output_dir" in exemplar_scan:
            return self.project.project_dir / "output" / exemplar_scan["output_dir"]

        # Try to find most recent output
        output_root = self.project.project_dir / "output"
        if output_root.exists():
            output_dirs = sorted(
                [d for d in output_root.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )
            if output_dirs:
                return output_dirs[0]

        return None

    def _has_candidate_catalog(self, output_dir: Path) -> bool:
        """Check if candidate catalog exists.

        Args:
            output_dir: Output directory

        Returns:
            True if candidate catalog exists
        """
        candidates_dir = output_dir / "candidates"
        if not candidates_dir.exists():
            return False

        # Find latest run directory
        run_dirs = sorted(
            [d for d in candidates_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            return False

        catalog_dir = run_dirs[0] / "databases" / "catalog"
        return catalog_dir.exists() and any(catalog_dir.iterdir())

    def _get_candidate_descriptions(self) -> dict[str, str]:
        """Build a lookup of candidate run directory names to descriptions.

        Queries the recovery_scans table to get descriptions for each
        candidate run directory.

        Returns:
            Dict mapping directory name (timestamp) to description
        """
        desc_lookup: dict[str, str] = {}
        try:
            recovery_scans = self.project.get_recovery_scans(active_only=False)
            for scan in recovery_scans:
                # output_dir is like "candidates/20251125_194722"
                output_dir = scan.get("output_dir", "")
                if output_dir.startswith("candidates/"):
                    dir_name = output_dir.split("/")[-1]
                    desc = scan.get("description") or ""
                    if desc:
                        desc_lookup[dir_name] = desc
        except Exception:
            # If database access fails, just return empty lookup
            pass
        return desc_lookup

    def _get_candidate_runs(self, output_dir: Path) -> list[Path]:
        """Get list of candidate run directories with catalogs.

        Args:
            output_dir: Output directory

        Returns:
            List of candidate run directories (sorted newest first)
        """
        candidates_dir = output_dir / "candidates"
        if not candidates_dir.exists():
            return []

        run_dirs = []
        for d in candidates_dir.iterdir():
            if d.is_dir():
                catalog_dir = d / "databases" / "catalog"
                if catalog_dir.exists() and any(catalog_dir.iterdir()):
                    run_dirs.append(d)

        return sorted(run_dirs, key=lambda d: d.name, reverse=True)

    def _select_candidate_run(
        self,
        output_dir: Path,
        show_header_callback: Callable | None = None,
        for_combined: bool = False,
    ) -> Path | None:
        """Show selection menu for candidate runs.

        Args:
            output_dir: Output directory
            show_header_callback: Optional header callback
            for_combined: If True, selecting for combined export (shows combined DB info)

        Returns:
            Selected candidate run directory or None if cancelled
        """
        run_dirs = self._get_candidate_runs(output_dir)

        if not run_dirs:
            self.console.print("[yellow]No candidate scans found.[/yellow]")
            return None

        if len(run_dirs) == 1:
            # Only one run, use it automatically
            return run_dirs[0]

        # Multiple runs - show selection menu
        if show_header_callback:
            show_header_callback()

        title = "Select Candidate Scan for Combined Export:" if for_combined else "Select Candidate Scan:"

        self.console.print(
            Panel(
                title,
                border_style=DSB3,
                style="bold",
            )
        )

        if for_combined:
            # Count combined DBs in exemplar catalog
            exemplar_catalog = output_dir / "exemplar" / "databases" / "catalog"
            combined_count = 0
            if exemplar_catalog.exists():
                combined_count = sum(1 for f in exemplar_catalog.rglob("*.combined.db"))
            self.console.print(f"[dim]Combined databases available: {combined_count}[/dim]\n")

        # Get descriptions for candidate runs
        desc_lookup = self._get_candidate_descriptions()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()
        table.add_column(style="dim")

        for i, run_dir in enumerate(run_dirs, 1):
            # Count databases in catalog
            catalog_dir = run_dir / "databases" / "catalog"
            db_count = sum(1 for d in catalog_dir.iterdir() if d.is_dir())
            # Format timestamp as readable date/time (YYYYMMDD_HHMMSS -> Nov 27, 2025 5:46 PM)
            ts = run_dir.name
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                formatted = dt.strftime("%b %d, %Y %-I:%M %p")
            except ValueError:
                formatted = ts
            # Show "most recent" indicator for first item
            label = f"[green]{formatted}[/green]"
            if i == 1:
                label += " [cyan](most recent)[/cyan]"
            # Get description if available
            desc = desc_lookup.get(ts, "")
            info = f"{db_count} databases"
            if desc:
                info = f"{desc} ({db_count} dbs)"
            table.add_row(f"{i}.", label, info)

        table.add_row("", "")
        table.add_row("", "[dim](B)ack[/dim]")
        table.add_row("", "")
        self.console.print(table)

        valid_choices = [str(i) for i in range(1, len(run_dirs) + 1)] + ["b"]
        choice = Prompt.ask(
            "[cyan]Select candidate scan[/cyan]",
            choices=valid_choices,
            default="1",
        ).lower()

        if choice == "b":
            return None

        return run_dirs[int(choice) - 1]

    def _run_export_wizard(
        self,
        output_dir: Path,
        source: ExportSource,
        show_header_callback: Callable | None = None,
        exemplar_scan: dict | None = None,
    ) -> ExportResult | None:
        """Run the export wizard to collect options and perform export.

        Args:
            output_dir: Output directory
            source: Export source type
            show_header_callback: Optional header callback
            exemplar_scan: Optional exemplar scan dict for case info

        Returns:
            ExportResult or None if cancelled
        """
        # For candidate and combined exports, select which candidate run to use
        candidate_run = None
        if source in (ExportSource.CANDIDATE, ExportSource.COMBINED):
            candidate_run = self._select_candidate_run(
                output_dir,
                show_header_callback,
                for_combined=(source == ExportSource.COMBINED),
            )
            if candidate_run is None:
                return None

        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                f"[bold]Export {source.value.title()} Databases[/bold]",
                border_style=DSB3,
            )
        )

        # Show selected candidate run
        if candidate_run:
            try:
                cand_dt = datetime.strptime(candidate_run.name, "%Y%m%d_%H%M%S")
                cand_display = cand_dt.strftime("%b %d, %Y %-I:%M %p")
            except ValueError:
                cand_display = candidate_run.name
            self.console.print(f"[dim]Candidate scan: {cand_display}[/dim]\n")

        # Structure selection
        self.console.print("\n[bold]Directory Structure:[/bold]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()
        table.add_column(style="dim")

        table.add_row(
            "1.",
            "[green]Flat[/green]",
            "Minimal folders",
        )
        table.add_row(
            "2.",
            "[green]Full Paths[/green]",
            "Full macOS directory structure",
        )
        table.add_row("", "")
        table.add_row("", "[dim](B)ack[/dim]")
        table.add_row("", "")
        self.console.print(table)

        structure_choice = Prompt.ask(
            "[cyan]Select structure[/cyan]",
            choices=["1", "2", "b"],
            default="1",
        ).lower()

        if structure_choice == "b":
            return None

        structure = ExportStructure.FLAT if structure_choice == "1" else ExportStructure.FULL_PATH

        # Track data source option (Combined exports only)
        track_data_source = False
        if source == ExportSource.COMBINED:
            self.console.print("\n[bold]Track Data Source:[/bold]")
            self.console.print("[dim]Add a 'data_source' column to track where each row came from[/dim]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style=f"{BDSB1}", width=4)
            table.add_column()
            table.add_column(style="dim")

            table.add_row(
                "y",
                "[green]Yes[/green]",
                "Tag rows as 'exemplar', 'carved', or 'found'",
            )
            table.add_row(
                "n",
                "[green]No[/green]",
                "No provenance tracking",
            )
            table.add_row("", "")
            table.add_row("", "[dim](B)ack[/dim]")
            table.add_row("", "")
            self.console.print(table)

            track_choice = Prompt.ask(
                "[cyan]Track row provenance?[/cyan]",
                choices=["y", "n", "b"],
                default="y",
            ).lower()

            if track_choice == "b":
                return None

            track_data_source = track_choice == "y"

        # Method selection (skip for Combined - always copy)
        if source == ExportSource.COMBINED:
            method = ExportMethod.COPY
        else:
            self.console.print("\n[bold]Export Method:[/bold]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style=f"{BDSB1}", width=4)
            table.add_column()
            table.add_column(style="dim")

            table.add_row(
                "1.",
                "[green]Copy[/green]",
                "Full copies (portable, can be zipped)",
            )
            table.add_row(
                "2.",
                "[green]Symlinks[/green]",
                "Symbolic links (saves space, local only)",
            )
            table.add_row("", "")
            table.add_row("", "[dim](B)ack[/dim]")
            table.add_row("", "")
            self.console.print(table)

            method_choice = Prompt.ask(
                "[cyan]Select method[/cyan]",
                choices=["1", "2", "b"],
                default="1",
            ).lower()

            if method_choice == "b":
                return None

            method = ExportMethod.COPY if method_choice == "1" else ExportMethod.SYMLINK

        # Include logs option
        self.console.print("\n[bold]Include Logs and Keychains:[/bold]")
        self.console.print("[dim]Export WiFi logs, System logs, ASL logs, keychains, etc.[/dim]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()
        table.add_column(style="dim")

        table.add_row(
            "y",
            "[green]Yes[/green]",
            "Include logs/ and keychains/ folders",
        )
        table.add_row(
            "n",
            "[green]No[/green]",
            "Databases only",
        )
        table.add_row("", "")
        table.add_row("", "[dim](B)ack[/dim]")
        table.add_row("", "")
        self.console.print(table)

        include_logs_choice = Prompt.ask(
            "[cyan]Include logs?[/cyan]",
            choices=["y", "n", "b"],
            default="n",
        ).lower()

        if include_logs_choice == "b":
            return None

        include_logs = include_logs_choice == "y"

        # Output location (include timestamp for uniqueness)
        if show_header_callback:
            show_header_callback()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_export_dir = output_dir / "export" / f"{source.value}_{timestamp}"
        self.console.print("\n[bold]Export Location:[/bold]")
        self.console.print(f"Default: {default_export_dir}")

        custom = Confirm.ask(
            "Use custom location?",
            default=False,
        )

        if custom:
            export_path_str = Prompt.ask("Enter export path")
            export_dir = Path(export_path_str).expanduser().resolve()
        else:
            export_dir = default_export_dir

        # Confirm
        self.console.print("\n[bold]Export Summary:[/bold]")
        self.console.print(f"  Source: [green]{source.value}[/green]")
        if candidate_run:
            # Format the candidate run timestamp nicely
            try:
                cand_dt = datetime.strptime(candidate_run.name, "%Y%m%d_%H%M%S")
                cand_formatted = cand_dt.strftime("%b %d, %Y %-I:%M %p")
            except ValueError:
                cand_formatted = candidate_run.name
            self.console.print(f"  Candidate scan: [green]{cand_formatted}[/green]")
        self.console.print(f"  Structure: [green]{structure.value}[/green]")
        if source != ExportSource.COMBINED:
            self.console.print(f"  Method: [green]{method.value}[/green]")
        if source == ExportSource.COMBINED:
            tracking_label = "Yes (exemplar/recovered)" if track_data_source else "No"
            self.console.print(f"  Track data source: [green]{tracking_label}[/green]")
        include_logs_label = "Yes" if include_logs else "No"
        self.console.print(f"  Include logs: [green]{include_logs_label}[/green]")
        self.console.print(f"  Export to: [cyan]{export_dir}[/cyan]")

        if not Confirm.ask("\nProceed with export?", default=True):
            return None

        # Build case_info from project config and scan info
        case_info = {
            "case_name": self.project.config.get("project_name"),
            "case_number": self.project.config.get("case_number"),
            "examiner_name": self.project.config.get("examiner_name"),
            "exemplar_description": exemplar_scan.get("description") if exemplar_scan else None,
            "candidate_description": None,  # Will be populated from candidate metadata if needed
        }

        # Try to get candidate description from metadata.json
        if candidate_run:
            candidate_metadata_path = candidate_run / "metadata.json"
            if candidate_metadata_path.exists():
                import json

                try:
                    with candidate_metadata_path.open() as f:
                        cand_meta = json.load(f)
                        case_info["candidate_description"] = cand_meta.get("candidate_description")
                except Exception:
                    pass

        # Perform export
        self.console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task_desc = "Exporting databases and logs..." if include_logs else "Exporting databases..."
            task = progress.add_task(task_desc, total=None)

            packager = ExportPackager(output_dir, case_info=case_info)
            result = packager.export_databases(
                export_dir=export_dir,
                source=source,
                structure=structure,
                method=method,
                candidate_run=candidate_run,
                track_data_source=track_data_source,
                include_logs=include_logs,
            )

            progress.update(task, completed=True)

        return result

    def _show_export_results(self, result: ExportResult) -> None:
        """Display export results.

        Args:
            result: ExportResult from export operation
        """

        if result.success:
            # Build summary lines
            summary_lines = [
                "[bold dark_sea_green4]Export Complete[/bold dark_sea_green4]\n",
                f"Databases: [cyan]{len(result.exported_files)}[/cyan]",
            ]
            if result.exported_logs or result.exported_keychains:
                summary_lines.append(
                    f"Logs: [cyan]{len(result.exported_logs)}[/cyan] | "
                    f"Keychains: [cyan]{len(result.exported_keychains)}[/cyan]"
                )
            summary_lines.append(f"Total size: [cyan]{self._format_size(result.total_size)}[/cyan]")
            summary_lines.append(f"Location: [cyan]{result.export_dir}[/cyan]")

            self.console.print(
                Panel(
                    "\n".join(summary_lines),
                    border_style="green",
                )
            )
        else:
            self.console.print(
                Panel(
                    "[bold red]Export Failed[/bold red]\n\nNo databases were exported. Check errors below.",
                    border_style="red",
                )
            )

        if result.errors:
            self.console.print(f"\n[red]Errors: {len(result.errors)}[/red]")
            for name, error in result.errors[:5]:
                self.console.print(f"  - {name}: {error}")

        self.console.print(f"\n[dim]Manifest written to: {result.export_dir}/_manifest.json[/dim]")
        Prompt.ask("\nPress Enter to continue")

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string like "1.5 MB"
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes = int(size_bytes / 1024)
        return f"{size_bytes:.1f} TB"

    # =========================================================================
    # Exemplar Package Export/Import
    # =========================================================================

    def _export_exemplar_package(
        self,
        output_dir: Path,
        show_header_callback: Callable | None = None,
    ) -> None:
        """Export exemplar as shareable package.

        Args:
            output_dir: Output directory containing exemplar
            show_header_callback: Optional header callback
        """
        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                "[bold light_goldenrod3]Export Exemplar Package[/bold light_goldenrod3]\n\n"
                "Create a shareable package containing:\n"
                "  - Database rubrics (matching signatures)\n"
                "  - Schema definitions\n"
                "  - Empty database shells (schema only, no data)\n\n"
                "[dim]Private data (usernames, file paths) will be sanitized.[/dim]",
                border_style="light_goldenrod3",
            )
        )

        # Get exemplar directory
        exemplar_dir = output_dir / "exemplar"
        if not exemplar_dir.exists():
            self.console.print("[red]Exemplar directory not found.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Get package name
        default_name = "Reference_Exemplar"
        package_name = Prompt.ask(
            "\n[cyan]Package name[/cyan]",
            default=default_name,
        ).strip()

        if not package_name:
            self.console.print("[yellow]Cancelled.[/yellow]")
            return

        # Sanitize name (replace spaces with underscores)
        package_name = package_name.replace(" ", "_")

        # Get description
        description = Prompt.ask(
            "[cyan]Description (optional)[/cyan]",
            default="",
        ).strip()

        # Get OS info
        self.console.print("\n[bold]OS Information (optional):[/bold]")
        os_name = Prompt.ask("[cyan]OS Name[/cyan]", default="macOS").strip()
        os_version = Prompt.ask("[cyan]OS Version[/cyan]", default="").strip()

        os_info = {}
        if os_name:
            os_info["name"] = os_name
        if os_version:
            os_info["version"] = os_version

        # Output location - use project-level exports directory
        default_output = self.project.project_dir / "exports"
        self.console.print(f"\n[bold]Output location:[/bold] {default_output}")

        custom = Confirm.ask("Use custom location?", default=False)
        if custom:
            export_path_str = Prompt.ask("Enter export path")
            output_path = Path(export_path_str).expanduser().resolve()
        else:
            output_path = default_output

        output_path.mkdir(parents=True, exist_ok=True)

        # Confirm
        self.console.print("\n[bold]Export Summary:[/bold]")
        self.console.print(f"  Package name: [green]{package_name}[/green]")
        if description:
            self.console.print(f"  Description: [green]{description}[/green]")
        if os_info:
            self.console.print(f"  OS: [green]{os_info.get('name', '')} {os_info.get('version', '')}[/green]")
        self.console.print(f"  Output: [cyan]{output_path}/Exemplar_{package_name}[/cyan]")

        if not Confirm.ask("\nProceed with export?", default=True):
            return

        # Perform export
        self.console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Exporting exemplar package...", total=None)

            packager = ExemplarPackager(exemplar_dir)
            result = packager.export(
                output_path=output_path,
                exemplar_name=package_name,
                description=description,
                os_info=os_info if os_info else None,
            )

            progress.update(task, completed=True)

        # Show results
        if result.success:
            self.console.print(
                Panel(
                    f"[bold dark_sea_green4]Export Complete[/bold dark_sea_green4]\n\n"
                    f"Package: [cyan]{result.output_path}[/cyan]\n"
                    f"Databases: [cyan]{result.database_count}[/cyan]",
                    border_style="green",
                )
            )
        else:
            self.console.print(
                Panel(
                    "[bold red]Export Failed[/bold red]",
                    border_style="red",
                )
            )
            for error in result.errors:
                self.console.print(f"  [red]{error}[/red]")

        if result.warnings:
            self.console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.warnings[:10]:
                self.console.print(f"  - {warning}")

        Prompt.ask("\nPress Enter to continue")
