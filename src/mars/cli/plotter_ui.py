#!/usr/bin/env python3
"""
Plotter UI - Chart and timeline visualization interface.

Handles database visualization and chart plotting functionality.
Extracted from app.py to improve modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mars.plotter.selection_buffer import SelectionBuffer, SeriesSelection
from mars.plotter.sqlite_plotter import QuitToMainMenu, series_picker_mode

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"


class PlotterUI:
    """UI for chart plotting and database visualization."""

    def __init__(self, console: Console, project: MARSProject):
        """
        Initialize Plotter UI.

        Args:
            console: Rich console instance
            project: Current MARS project
        """
        self.console = console
        self.project = project

    def _select_plottable_database(
        self,
        db_root: Path,
        source_type: str = "Exemplar",
        show_header_callback: Callable | None = None,
    ) -> Path | None:
        """
        Select a database with pre-filtering for plottability.

        Shows databases in catalog with plottability status and row counts.
        Filters out databases that cannot be plotted.

        Args:
            db_root: Root directory containing database folders (catalog)
            source_type: "Exemplar" or "Candidate" for back navigation text
            show_header_callback: Optional callback to display project header

        Returns:
            Path to selected database or None if cancelled
        """
        from mars.plotter.db import check_database_plottable

        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                f"[{BDSB1}]Select Database for Visualization[/{BDSB1}]",
                border_style=f"{DSB3}",
            )
        )

        # Scan all databases in catalog
        self.console.print("[dim]Scanning databases for plottable data...[/dim]\n")

        db_entries: list[tuple[str, Path, dict]] = []

        for entry_dir in sorted(db_root.iterdir()):
            if not entry_dir.is_dir() or entry_dir.name.startswith("."):
                continue

            # Find database files in this directory
            db_files = list(entry_dir.glob("*.sqlite")) + list(entry_dir.glob("*.db"))
            if not db_files:
                continue

            db_path = db_files[0]
            info = check_database_plottable(db_path)
            db_entries.append((entry_dir.name, db_path, info))

        if not db_entries:
            self.console.print("[yellow]No databases found in catalog.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return None

        # Separate plottable and non-plottable
        plottable = [(n, p, i) for n, p, i in db_entries if i["plottable"]]
        non_plottable = [(n, p, i) for n, p, i in db_entries if not i["plottable"]]

        if not plottable:
            self.console.print("[yellow]No databases with plottable data found.[/yellow]\n")
            self.console.print("[dim]Reasons:[/dim]")
            for name, _path, info in non_plottable[:10]:
                self.console.print(f"  [dim]{name}: {info['reason']}[/dim]")
            if len(non_plottable) > 10:
                self.console.print(f"  [dim]...and {len(non_plottable) - 10} more[/dim]")
            Prompt.ask("\nPress Enter to continue")
            return None

        # Build selection table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Database", style="bold")
        table.add_column("Tables", justify="right")
        table.add_column("Rows", justify="right")

        for i, (name, _path, info) in enumerate(plottable, 1):
            table.add_row(
                str(i),
                name,
                str(info["tables_with_data"]),
                f"{info['total_rows']:,}",
            )

        self.console.print(f"[bold]Plottable databases ({len(plottable)}):[/bold]")
        self.console.print(table)

        if non_plottable:
            self.console.print(f"\n[dim]{len(non_plottable)} database(s) excluded (no plottable data)[/dim]")

        back_target = "run selection" if source_type == "Candidate" else "source selection"
        self.console.print(f"\n[dim](b) Back to {back_target}[/dim]")

        # Prompt for selection
        valid_choices = [str(i) for i in range(1, len(plottable) + 1)] + ["b"]
        choice = Prompt.ask(
            "\n[bold cyan]Select database[/bold cyan]",
            choices=valid_choices,
            show_default=False,
        )

        if choice.lower() == "b":
            return None

        idx = int(choice) - 1
        return plottable[idx][1]

    def _show_buffer_menu(
        self,
        buffer: SelectionBuffer,
        exemplar_output_dir: Path | None,
        show_header_callback: Callable | None = None,
    ) -> str:
        """
        Show buffer status and menu options.

        Args:
            buffer: Current SelectionBuffer with accumulated selections
            exemplar_output_dir: Base output directory for paths (None if not yet selected)
            show_header_callback: Optional callback to display project header

        Returns:
            User choice: "add", "plot", "clear", or "back"
        """
        if show_header_callback:
            show_header_callback()

        # Build buffer status display
        if buffer.is_empty():
            status_text = "[dim]No series selected yet[/dim]"
        else:
            lines = [f"[bold]{buffer.count()}/{buffer.max_series} series selected:[/bold]\n"]
            for i, sel in enumerate(buffer.selections, 1):
                lines.append(f"  {i}. {sel.display_name()}")
            status_text = "\n".join(lines)

        self.console.print(
            Panel(
                f"[{BDSB1}]Selection Buffer[/{BDSB1}]\n\n{status_text}",
                border_style=f"{DSB3}",
            )
        )

        # Build menu options
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()

        if not buffer.is_full():
            table.add_row("1.", "[bold dark_sea_green4]Add series from database[/bold dark_sea_green4]")
        else:
            table.add_row(
                "[dim]1.[/dim]",
                "[dim]Add series (buffer full - 5/5)[/dim]",
            )

        if not buffer.is_empty():
            table.add_row("2.", "[bold dark_sea_green4]Plot selected series[/bold dark_sea_green4]")
            table.add_row("3.", "Clear all selections")
        else:
            table.add_row("[dim]2.[/dim]", "[dim]Plot (no series selected)[/dim]")
            table.add_row("[dim]3.[/dim]", "[dim]Clear (nothing to clear)[/dim]")

        table.add_row(None, "(B)ack to main menu")

        self.console.print("\n[bold]Options:[/bold]")
        self.console.print(table)

        # Build valid choices based on state
        valid_choices = ["b"]
        if not buffer.is_full():
            valid_choices.append("1")
        if not buffer.is_empty():
            valid_choices.extend(["2", "3"])

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=valid_choices,
            show_default=False,
        ).lower()

        if choice == "1":
            return "add"
        if choice == "2":
            return "plot"
        if choice == "3":
            return "clear"
        return "back"

    def _prompt_chart_options(self, series_count: int) -> dict | None:
        """
        Prompt user for chart configuration options.

        Args:
            series_count: Number of series to plot (affects available chart types)

        Returns:
            Dict with chart options, or None if user cancelled
        """
        # Chart type selection
        self.console.print("\n[bold]Chart Type:[/bold]")
        chart_table = Table(show_header=False, box=None, padding=(0, 2))
        chart_table.add_column(style="cyan", width=4)
        chart_table.add_column()

        if series_count == 1:
            chart_table.add_row("1.", "Line")
            chart_table.add_row("2.", "Scatter")
            chart_table.add_row("3.", "Bar")
            chart_table.add_row(None, "(B)ack")
            self.console.print(chart_table)
            chart_choice = Prompt.ask(
                "[bold cyan]Select chart type[/bold cyan]",
                choices=["1", "2", "3", "b"],
                show_default=False,
            ).lower()
            if chart_choice == "b":
                return None
            style_map = {"1": "line", "2": "scatter", "3": "bar"}
            style = style_map[chart_choice]
        else:
            chart_table.add_row("1.", "Overlay Line (up to 2 series)")
            chart_table.add_row("2.", "Stacked Line (multiple series)")
            chart_table.add_row("3.", "Scatter (up to 2 series)")
            chart_table.add_row(None, "(B)ack")
            self.console.print(chart_table)

            valid = ["2", "b"]  # Stacked always available
            if series_count <= 2:
                valid.extend(["1", "3"])

            chart_choice = Prompt.ask(
                "[bold cyan]Select chart type[/bold cyan]",
                choices=valid,
                show_default=False,
            ).lower()
            if chart_choice == "b":
                return None
            style_map = {"1": "line", "2": "stacked", "3": "scatter"}
            style = style_map[chart_choice]

        # Timezone selection
        self.console.print("\n[bold]Timezone:[/bold]")
        tz_table = Table(show_header=False, box=None, padding=(0, 2))
        tz_table.add_column(style="cyan", width=4)
        tz_table.add_column()
        tz_table.add_row("1.", "Local time")
        tz_table.add_row("2.", "UTC")
        tz_table.add_row("3.", "Custom offset")
        self.console.print(tz_table)

        tz_choice = Prompt.ask(
            "[bold cyan]Select timezone[/bold cyan]",
            choices=["1", "2", "3"],
            default="1",
            show_default=False,
        )
        tz_map = {"1": "local", "2": "utc", "3": "offset"}
        tzmode = tz_map[tz_choice]

        offset_minutes = 0
        if tzmode == "offset":
            offset_str = Prompt.ask(
                "[cyan]Enter offset in minutes (e.g., -300 for UTC-5)[/cyan]",
                default="0",
            )
            try:
                offset_minutes = int(offset_str)
            except ValueError:
                offset_minutes = 0

        # Dark mode
        dark_mode = (
            Prompt.ask(
                "\n[bold cyan]Use dark theme?[/bold cyan]",
                choices=["y", "n"],
                default="n",
            ).lower()
            == "y"
        )

        # Rolling mean smoothing (only for line/scatter, not bar)
        smooth_window = 1
        if style in ("line", "stacked", "scatter"):
            apply_smoothing = (
                Prompt.ask(
                    "\n[bold cyan]Apply rolling mean smoothing?[/bold cyan]",
                    choices=["y", "n"],
                    default="n",
                ).lower()
                == "y"
            )
            if apply_smoothing:
                self.console.print("[dim]Window size (points to average, e.g., 3, 5, 10). 1 = none[/dim]")
                while True:
                    window_str = Prompt.ask("[cyan]Window size[/cyan]", default="5")
                    if window_str.isdigit() and int(window_str) >= 1:
                        smooth_window = int(window_str)
                        break
                    self.console.print("[yellow]Enter a positive integer (e.g., 3, 5, 10)[/yellow]")

        return {
            "style": style,
            "tzmode": tzmode,
            "offset_minutes": offset_minutes,
            "dark_mode": dark_mode,
            "step_for_binary": True,
            "smooth_window": smooth_window,
        }

    def _render_from_buffer(
        self,
        buffer: SelectionBuffer,
        plots_dir: Path,
        show_header_callback: Callable | None = None,
    ) -> bool:
        """
        Render a chart from the current buffer selections.

        Args:
            buffer: SelectionBuffer with series to plot
            plots_dir: Output directory for plots
            show_header_callback: Optional callback to display project header

        Returns:
            True if chart was rendered, False if user cancelled
        """
        from datetime import datetime

        from mars.plotter.db import build_multi_db_dataset
        from mars.plotter.rendering import (
            _inject_page_background,
            create_overlay_figure,
            render_stacked_lines,
        )

        if buffer.is_empty():
            self.console.print("[yellow]No series to plot.[/yellow]")
            return False

        if show_header_callback:
            show_header_callback()

        # Show buffer summary
        self.console.print(
            Panel(
                f"[{BDSB1}]Chart Configuration[/{BDSB1}]\n\n"
                f"[dim]Plotting {buffer.count()} series from {len(buffer.get_unique_databases())} database(s)[/dim]",
                border_style=f"{DSB3}",
            )
        )

        # Prompt for chart options
        options = self._prompt_chart_options(buffer.count())
        if options is None:
            return False  # User cancelled

        try:
            # Fetch and align data from all selections
            xs, series_dict, labels_dict = build_multi_db_dataset(buffer.selections, epoch_window=None)

            if not xs:
                self.console.print("[yellow]No data to plot (empty result).[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                return False

            # Build chart title
            source_types = buffer.get_source_types()
            title = self._compute_chart_title_multi_db(buffer.selections, source_types)

            # Ensure output directory exists
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename base
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_names = "_".join(sorted({sel.db_name for sel in buffer.selections})[:3])
            if len(buffer.get_unique_databases()) > 3:
                db_names += "_etc"

            # Create figure based on style
            if options["style"] == "stacked":
                # Build panel data for stacked rendering
                panels = []
                for label, ys in series_dict.items():
                    panel_labels = labels_dict.get(label, [])
                    panels.append(
                        {
                            "label": label,
                            "xs": xs,
                            "ys": ys,
                            "hover_labels": panel_labels if panel_labels else None,
                        }
                    )
                # render_stacked_lines writes file directly and returns path
                out_file = render_stacked_lines(
                    outdir=plots_dir,
                    base_title=f"multi_{db_names}_{timestamp}",
                    tzmode=options["tzmode"],
                    offset_minutes=options["offset_minutes"],
                    panels=panels,
                    step_for_binary=options["step_for_binary"],
                    smooth_window=options["smooth_window"],
                    dark_mode=options["dark_mode"],
                )
            else:
                # Overlay figure (line, scatter, bar)
                fig = create_overlay_figure(
                    xs,
                    series_dict,
                    style=options["style"],
                    tzmode=options["tzmode"],
                    offset_minutes=options["offset_minutes"],
                    step_for_binary=options["step_for_binary"],
                    title=title,
                    smooth_window=options["smooth_window"],
                )

                # Add hover labels to traces if available
                for i, (label, _) in enumerate(series_dict.items()):
                    hover_labels = labels_dict.get(label, [])
                    if hover_labels and i < len(fig.data):  # type: ignore[arg-type]
                        clean_labels = [lbl if lbl else "" for lbl in hover_labels]
                        fig.data[i].hovertext = clean_labels
                        fig.data[i].hovertemplate = "%{hovertext}<br>%{x}<br>%{y}<extra></extra>"

                # Apply dark mode if requested
                if options["dark_mode"]:
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                    )

                out_file = plots_dir / f"multi_{db_names}_{timestamp}.html"
                fig.write_html(str(out_file), include_plotlyjs=True)

                # Inject dark mode page background if enabled
                if options["dark_mode"]:
                    _inject_page_background(out_file, dark_mode=True)

            self.console.print(f"\n[bold dark_sea_green4]Chart saved:[/bold dark_sea_green4] {out_file}")

            # Open in browser
            import webbrowser

            webbrowser.open(f"file://{out_file}")

        except Exception as e:
            self.console.print(f"\n[bold red]Error rendering chart:[/bold red] {e}")

        # Ask if user wants to make another chart
        another = Prompt.ask(
            "\n[bold cyan]Make another chart with same selections?[/bold cyan]",
            choices=["y", "n"],
            default="n",
        ).lower()

        if another == "y":
            # Recursive call to render again with same buffer
            return self._render_from_buffer(buffer, plots_dir, show_header_callback)

        return True

    def _compute_chart_title_multi_db(
        self,
        selections: list[SeriesSelection],
        source_types: set[str],
    ) -> str:
        """
        Compute a descriptive chart title for multi-database plots.

        Args:
            selections: List of SeriesSelection objects
            source_types: Set of source types ("Exemplar", "Candidate")

        Returns:
            Chart title string
        """
        project_name = self.project.config.get("project_name", "MARS")

        # Summarize sources
        if source_types == {"Exemplar"}:
            source_desc = "Exemplar"
        elif source_types == {"Candidate"}:
            source_desc = "Candidate"
        else:
            source_desc = "Exemplar + Candidate"

        # Summarize databases
        unique_dbs = list({sel.db_name for sel in selections})
        if len(unique_dbs) == 1:
            db_desc = unique_dbs[0]
        elif len(unique_dbs) <= 3:
            db_desc = ", ".join(unique_dbs)
        else:
            db_desc = f"{len(unique_dbs)} databases"

        return f"{project_name} | {source_desc} | {db_desc}"

    def show_menu_with_buffer(
        self,
        exemplar_scan: dict | None = None,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Display chart plotter menu with SelectionBuffer workflow.

        This is the new entry point that uses the buffer-based multi-database
        plotting workflow. Goes directly to Selection Buffer, deferring exemplar
        selection to when the user clicks "Add Buffer".

        Args:
            exemplar_scan: Optional pre-selected exemplar scan dict (for backwards compat)
            show_header_callback: Optional callback to display project header
        """
        # Track exemplar context - selected on first "Add Buffer" if not provided
        current_exemplar: dict | None = exemplar_scan
        exemplar_output_dir: Path | None = None
        plots_dir: Path | None = None

        if current_exemplar:
            output_dir_name = str(current_exemplar["output_dir"])
            exemplar_output_dir = self.project.project_dir / "output" / output_dir_name
            plots_dir = exemplar_output_dir / "reports" / "plots"

        # Create fresh buffer for this session
        buffer = SelectionBuffer()

        # Main loop: buffer menu -> add series -> buffer menu -> plot -> exit
        while True:
            action = self._show_buffer_menu(buffer, exemplar_output_dir, show_header_callback)

            if action == "back":
                return  # Exit to main menu

            if action == "clear":
                count = buffer.clear()
                self.console.print(f"[dim]Cleared {count} selection(s).[/dim]")
                continue

            if action == "plot":
                if plots_dir is None:
                    self.console.print("[yellow]No databases selected yet.[/yellow]")
                    continue
                rendered = self._render_from_buffer(buffer, plots_dir, show_header_callback)
                if rendered:
                    # After plotting, clear buffer and return to main menu
                    buffer.clear()
                    return
                # User cancelled from chart options - go back to buffer menu
                continue

            if action == "add":
                # Select exemplar if not yet selected
                if current_exemplar is None:
                    current_exemplar = self._select_exemplar_for_plotter(show_header_callback)
                    if current_exemplar is None:
                        continue  # User cancelled - back to buffer menu
                    output_dir_name = str(current_exemplar["output_dir"])
                    exemplar_output_dir = self.project.project_dir / "output" / output_dir_name
                    plots_dir = exemplar_output_dir / "reports" / "plots"

                # Enter the add-series flow (exemplar_output_dir is guaranteed non-None here)
                assert exemplar_output_dir is not None
                try:
                    self._add_series_to_buffer(
                        buffer,
                        exemplar_output_dir,
                        show_header_callback,
                    )
                except QuitToMainMenu:
                    return

    def _select_exemplar_for_plotter(
        self,
        show_header_callback: Callable | None = None,
    ) -> dict | None:
        """
        Select an exemplar scan for the plotter session.

        Auto-selects if only one valid scan exists. Shows selection menu
        if multiple scans exist.

        Args:
            show_header_callback: Optional callback to display project header

        Returns:
            Selected exemplar scan dict, or None if cancelled/none available
        """
        scans = self.project.get_exemplar_scans(active_only=True)

        if not scans:
            self.console.print("[yellow]No exemplar scans found. Run an exemplar scan first.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return None

        # Validate that output directories still exist
        valid_scans = []
        for scan in scans:
            output_dir = self.project.project_dir / "output" / scan["output_dir"]
            if output_dir.exists():
                valid_scans.append(scan)

        if not valid_scans:
            self.console.print("[yellow]No valid exemplar scans found (output directories missing).[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return None

        # Auto-select if only one valid scan
        if len(valid_scans) == 1:
            return valid_scans[0]

        # Multiple scans - show selection menu
        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                f"[{BDSB1}]Select Exemplar Scan[/{BDSB1}]\n\n"
                "[dim]Choose which exemplar's databases to plot from.[/dim]",
                border_style=f"{DSB3}",
            )
        )

        table = Table(show_header=True, header_style=f"{BDSB1}")
        table.add_column("#", style="bold", width=4)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Description", style="dim")

        for idx, scan in enumerate(valid_scans, start=1):
            timestamp = scan["timestamp"][:19].replace("T", " ") + "Z"
            desc = scan.get("description") or ""
            if len(desc) > 40:
                desc = desc[:37] + "..."
            table.add_row(str(idx), timestamp, desc)

        self.console.print(table)
        self.console.print("\n[dim](b) Back to buffer menu[/dim]")

        valid_choices = [str(i) for i in range(1, len(valid_scans) + 1)] + ["b"]
        choice = Prompt.ask(
            "\n[bold cyan]Select exemplar[/bold cyan]",
            choices=valid_choices,
            show_default=False,
        )

        if choice.lower() == "b":
            return None

        return valid_scans[int(choice) - 1]

    def _add_series_to_buffer(
        self,
        buffer: SelectionBuffer,
        exemplar_output_dir: Path,
        show_header_callback: Callable | None = None,
    ) -> None:
        """
        Flow for adding series to the buffer.

        Guides user through source -> database -> series selection,
        then adds results to the buffer.

        Args:
            buffer: SelectionBuffer to add selections to
            exemplar_output_dir: Base output directory
            show_header_callback: Optional callback to display project header
        """
        # Check if candidates exist
        candidates_root = exemplar_output_dir / "candidates"
        has_candidates = candidates_root.exists() and list(candidates_root.iterdir())

        # Show source selection
        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                f"[{BDSB1}]Add Series to Buffer[/{BDSB1}]\n\n"
                f"[dim]Current: {buffer.count()}/{buffer.max_series} series[/dim]",
                border_style=f"{DSB3}",
            )
        )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()

        table.add_row("1.", "[bold dark_sea_green4]Exemplar databases[/bold dark_sea_green4]")
        if has_candidates:
            table.add_row("2.", "[bold dark_sea_green4]Candidate scan databases[/bold dark_sea_green4]")
        else:
            table.add_row(
                "[dim]2.[/dim]",
                "[dim]Candidate scan databases (no candidates available)[/dim]",
            )
        table.add_row(None, "(B)ack to buffer menu")

        self.console.print("[bold]Select database source:[/bold]")
        self.console.print(table)

        valid_choices = ["1", "b"]
        if has_candidates:
            valid_choices = ["1", "2", "b"]

        source_choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=valid_choices,
            show_default=False,
        ).lower()

        if source_choice == "b":
            return  # Back to buffer menu

        # Determine source type and paths
        if source_choice == "1":
            source_type = "Exemplar"
            db_root = exemplar_output_dir / "exemplar" / "databases" / "catalog"
            scan_name = exemplar_output_dir.name

            if not db_root.exists():
                self.console.print(f"[yellow]Database directory not found: {db_root}[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                return
        else:
            source_type = "Candidate"
            # Select candidate run first
            candidate_run = self._select_candidate_run(candidates_root, show_header_callback)
            if candidate_run is None:
                return  # User backed out

            db_root = candidate_run / "databases" / "catalog"
            scan_name = candidate_run.name

            if not db_root.exists():
                self.console.print(f"[yellow]Database directory not found: {db_root}[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                return

        # Select database
        db_path = self._select_plottable_database(
            db_root,
            source_type=source_type,
            show_header_callback=show_header_callback,
        )

        if db_path is None:
            return  # User backed out

        # Run series picker
        if show_header_callback:
            show_header_callback()

        selections = series_picker_mode(db_path, source_type, scan_name)

        if selections is None or not selections:
            self.console.print("[dim]No series selected.[/dim]")
            return

        # Add to buffer
        added = 0
        for sel in selections:
            if buffer.add(sel):
                added += 1
            else:
                self.console.print(f"[yellow]Buffer full - could not add: {sel.short_name()}[/yellow]")
                break

        if added > 0:
            self.console.print(f"[green]Added {added} series to buffer.[/green]")

    def _select_candidate_run(
        self,
        candidates_root: Path,
        show_header_callback: Callable | None = None,
    ) -> Path | None:
        """
        Select a candidate run from available runs.

        Args:
            candidates_root: Path to candidates directory
            show_header_callback: Optional callback to display project header

        Returns:
            Path to selected candidate run directory, or None if cancelled
        """
        candidate_runs = sorted(
            [d for d in candidates_root.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        if not candidate_runs:
            self.console.print("[yellow]No candidate runs found.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return None

        # If only one run, return it directly
        if len(candidate_runs) == 1:
            return candidate_runs[0]

        # Build description lookup from recovery_scans table
        desc_lookup = self._get_candidate_descriptions()

        # Show run selection
        if show_header_callback:
            show_header_callback()

        self.console.print(
            Panel(
                "[bold cyan]Select Candidate Run[/bold cyan]",
                border_style="cyan",
            )
        )

        run_table = Table(show_header=True, header_style=f"{BDSB1}")
        run_table.add_column("#", style="bold", width=4)
        run_table.add_column("Timestamp", style="cyan")
        run_table.add_column("Description", style="dim")

        for idx, run_dir in enumerate(candidate_runs, start=1):
            desc = desc_lookup.get(run_dir.name, "")
            # Truncate long descriptions
            if len(desc) > 40:
                desc = desc[:37] + "..."
            run_table.add_row(str(idx), run_dir.name, desc)

        self.console.print(run_table)
        self.console.print("\n[dim](b) Back[/dim]")

        run_choices = [str(i) for i in range(1, len(candidate_runs) + 1)] + ["b"]
        run_choice = Prompt.ask(
            "\n[bold cyan]Select candidate run[/bold cyan]",
            choices=run_choices,
            show_default=False,
        )

        if run_choice.lower() == "b":
            return None

        return candidate_runs[int(run_choice) - 1]

    def _get_candidate_descriptions(self) -> dict[str, str]:
        """
        Build a lookup of candidate run directory names to descriptions.

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
