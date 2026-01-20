#!/usr/bin/env python3
"""
Comparison UI - Data recovery comparison report interface.

Handles comparison report generation between exemplar and candidate scans.
Extracted from app.py to improve modularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject

# Rich Colors
BDSB1 = "bold deep_sky_blue1"


class ComparisonUI:
    """UI for data recovery comparison reports."""

    def __init__(self, console: Console, project: MARSProject):
        """
        Initialize Comparison UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project

    def generate_report(
        self,
        exemplar_scan: dict,
        exemplar_output_dir: Path,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Generate data recovery comparison report.

        Args:
            exemplar_scan: Selected exemplar scan dict
            exemplar_output_dir: Path to exemplar output directory
            show_header_callback: Optional callback to display project header
        """
        from mars.pipeline.comparison import (
            ComparisonCalculator,
            generate_report,
        )

        candidates_root = exemplar_output_dir / "candidates"

        # Let user select candidate run
        candidate_runs = sorted(
            [d for d in candidates_root.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        if not candidate_runs:
            self.console.print("[yellow]No candidate scans found.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Get recovery scans from database to look up descriptions (always needed)
        recovery_scans = self.project.get_recovery_scans(exemplar_scan_id=exemplar_scan["id"])
        # Build lookup: output_dir basename -> description
        desc_lookup = {}
        for scan in recovery_scans:
            # output_dir is like "candidates/20251125_042025"
            run_name = Path(scan["output_dir"]).name
            desc_lookup[run_name] = scan.get("description") or ""

        if len(candidate_runs) == 1:
            selected_run = candidate_runs[0]
        else:
            if show_header_callback:
                show_header_callback()
            self.console.print(
                Panel(
                    "[bold cyan]Select Candidate Run for Comparison[/bold cyan]",
                    border_style="cyan",
                )
            )

            run_table = Table(show_header=True, header_style=f"{BDSB1}")
            run_table.add_column("#", style="bold", width=4)
            run_table.add_column("Timestamp", style="cyan")
            run_table.add_column("Description", style="italic")

            for idx, run_dir in enumerate(candidate_runs, start=1):
                desc = desc_lookup.get(run_dir.name, "")
                if len(desc) > 35:
                    desc = desc[:32] + "..."
                run_table.add_row(str(idx), run_dir.name, desc)

            self.console.print(run_table)

            run_choice = Prompt.ask(
                "\n[bold cyan]Select run[/bold cyan]",
                choices=[str(i) for i in range(1, len(candidate_runs) + 1)] + ["b"],
            )
            if run_choice == "b":
                return
            selected_run = candidate_runs[int(run_choice) - 1]

        # Show progress
        self.console.print(f"\n[bold]Comparing:[/bold] {exemplar_scan['output_dir']} vs {selected_run.name}")
        self.console.print("[dim]Calculating deduplication and timeline coverage...[/dim]\n")

        try:
            # Set up paths
            exemplar_db_dir = exemplar_output_dir / "exemplar" / "databases"
            candidate_matched_dir = selected_run / "matched"
            candidate_results_jsonl = selected_run / "databases" / "selected_variants" / "sqlite_scan_results.jsonl"

            # Calculate comparison
            calc = ComparisonCalculator(
                exemplar_db_dir=exemplar_db_dir,
                candidate_matched_dir=candidate_matched_dir,
                candidate_results_jsonl=candidate_results_jsonl,
            )
            result = calc.calculate()

            # Find existing scan reports for linking
            # Note: exemplar reports are in output_dir/reports/, not output_dir/exemplar/reports/
            exemplar_reports_dir = exemplar_output_dir / "reports"
            candidate_reports_dir = selected_run / "reports"

            # Find most recent exemplar report
            exemplar_report_path = None
            if exemplar_reports_dir.exists():
                exemplar_reports = list(exemplar_reports_dir.glob("exemplar_*.html"))
                if exemplar_reports:
                    exemplar_report_path = max(exemplar_reports, key=lambda p: p.stat().st_mtime)

            # Find most recent candidate report
            candidate_report_path = None
            if candidate_reports_dir.exists():
                candidate_reports = list(candidate_reports_dir.glob("candidate_*.html"))
                if candidate_reports:
                    candidate_report_path = max(candidate_reports, key=lambda p: p.stat().st_mtime)

            # Get descriptions
            exemplar_description = exemplar_scan.get("description", "")
            candidate_description = desc_lookup.get(selected_run.name, "")

            # Generate HTML report in comparison_reports subfolder
            # Each report gets a unique filename based on candidate run timestamp
            reports_dir = exemplar_output_dir / "reports" / "comparison_reports"
            report_filename = f"comparison_{selected_run.name}.html"
            html_path = generate_report(
                result,
                output_dir=reports_dir,
                filename=report_filename,
                exemplar_description=exemplar_description,
                candidate_description=candidate_description,
                exemplar_report_path=exemplar_report_path,
                candidate_report_path=candidate_report_path,
            )

            # Show summary in CLI
            if show_header_callback:
                show_header_callback()
            self.console.print(
                Panel(
                    "[bold dark_sea_green4]Comparison Report Generated[/bold dark_sea_green4]",
                    border_style="green",
                )
            )

            # Summary stats
            summary_table = Table(show_header=False, box=None, padding=(0, 2))
            summary_table.add_column(style="dim", width=24)
            summary_table.add_column(style="cyan", justify="right")

            summary_table.add_row(
                "Databases Matched:",
                f"{result.candidate_matched_count} / {result.exemplar_database_count}",
            )
            summary_table.add_row(
                "Unique Rows Recovered:",
                f"{result.total_unique_rows_recovered:,}",
            )
            summary_table.add_row(
                "Databases with New Data:",
                str(result.candidate_with_new_data_count),
            )

            # Count timeline extensions
            timeline_ext = sum(1 for db in result.databases if db.has_timeline_extension)
            if timeline_ext > 0:
                summary_table.add_row("Timeline Extensions:", str(timeline_ext))

            self.console.print(summary_table)

            self.console.print(f"\n[bold]Full report:[/bold] {html_path}")

            # Offer to open report
            if Confirm.ask("\nOpen report in browser?", default=True):
                import subprocess
                import sys

                if sys.platform == "darwin":
                    # Use 'open' command on macOS to respect default browser
                    subprocess.run(["open", str(html_path)], check=False)
                else:
                    import webbrowser

                    webbrowser.open(f"file://{html_path}")

        except Exception as e:
            self.console.print(f"[bold red]Error generating report:[/bold red] {e}")
            import traceback

            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
