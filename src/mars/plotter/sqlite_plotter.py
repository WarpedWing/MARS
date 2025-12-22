#!/usr/bin/env python3

"""
sqlite_plotter.py

Interactive Plotly-based visualizer for SQLite databases.
Uses Rich for terminal UI.
"""

from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from mars.plotter.db import (
    detect_tables_with_timestamp,
    filter_columns_with_non_null,
    list_graphable_columns,
    list_label_columns,
    load_rubric_for_database,
)
from mars.plotter.selection_buffer import SeriesSelection
from mars.plotter.tui import (
    cerr,
    chead,
    cinfo,
    console,
    cwarn,
    prompt_input,
)
from mars.utils.database_utils import readonly_connection


class QuitToMainMenu(Exception):
    """Raised when user wants to quit the plotter and return to main menu."""

    pass


def prompt_series_selection(
    tnames: list[str],
    graphable_map: dict[str, list[str]],
    max_series: int = 5,
    row_count_map: dict[str, int] | None = None,
    ts_cols_map: dict[str, list[tuple[str, str | None]]] | None = None,
    ts_col_map: dict[str, str] | None = None,
    ts_format_map: dict[str, str | None] | None = None,
) -> list[tuple[str, str]] | None:
    """Interactive selector for up to `max_series` (table, column) pairs.

    Returns None if user chooses to go back from table selection.

    If a table has multiple timestamp columns and ts_cols_map is provided,
    prompts user to select which timestamp column to use. Updates ts_col_map
    and ts_format_map with the selection.
    """
    row_count_map = row_count_map or {}
    ts_cols_map = ts_cols_map or {}
    selections: list[tuple[str, str]] = []
    current_table: str | None = None
    tables_with_ts_selected: set[str] = set()  # Track tables where user already chose timestamp

    console.print(
        Panel(
            f"Select up to [bold]{max_series}[/bold] series to plot.\n"
            "[dim]'b' = back to database selection | 'q' = quit to main menu | Enter = finish[/dim]",
            title="[bold cyan]Series Selection[/bold cyan]",
            border_style="cyan",
        )
    )

    while len(selections) < max_series:
        if current_table is None:
            # Show tables menu
            console.print("\n[bold]Tables with graphable columns:[/bold]")
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="cyan", width=4)
            table.add_column()
            table.add_column(style="dim", justify="right")

            for i, t in enumerate(tnames, 1):
                cols = graphable_map.get(t, [])
                row_count = row_count_map.get(t, 0)
                row_info = f"{row_count:,} rows" if row_count > 0 else "empty"
                ts_count = len(ts_cols_map.get(t, []))
                ts_info = f" | {ts_count} timestamps" if ts_count > 1 else ""

                if cols:
                    table.add_row(
                        f"{i}.",
                        t,
                        f"[cyan]{len(cols)} cols[/cyan] | {row_info}{ts_info}",
                    )
                else:
                    table.add_row(
                        f"[dim]{i}.[/dim]",
                        f"[dim]{t}[/dim]",
                        f"[dim]no numeric cols | {row_info}[/dim]",
                    )

            table.add_row("", "[dim](b) Back to database selection[/dim]", "")
            console.print(table)

            tbl_input = prompt_input()
            if tbl_input.lower() == "q":
                raise QuitToMainMenu()
            if tbl_input.lower() == "b":
                if selections:
                    cwarn("Selections cleared.")
                return None
            if tbl_input == "" and selections:
                break

            selected_table: str | None = None
            if tbl_input.isdigit():
                idx = int(tbl_input)
                if 1 <= idx <= len(tnames):
                    selected_table = tnames[idx - 1]
            if selected_table is None:
                for t in tnames:
                    if tbl_input.lower() == t.lower():
                        selected_table = t
                        break
            if not selected_table or not graphable_map.get(selected_table):
                cwarn("Please enter a valid table index/name with graphable columns.")
                continue

            # Prompt for timestamp column if multiple exist and not already selected
            ts_cols = ts_cols_map.get(selected_table, [])
            if (
                len(ts_cols) > 1
                and selected_table not in tables_with_ts_selected
                and ts_col_map is not None
                and ts_format_map is not None
            ):
                ts_choice = prompt_timestamp_column(selected_table, ts_cols)
                if ts_choice is None:
                    # User chose to go back
                    continue
                ts_col_map[selected_table] = ts_choice[0]
                ts_format_map[selected_table] = ts_choice[1]
                tables_with_ts_selected.add(selected_table)
                cinfo(f"Using [cyan]{ts_choice[0]}[/cyan] as timestamp column")

            current_table = selected_table

        # Column selection for current table
        available = [c for c in graphable_map[current_table] if (current_table, c) not in selections]
        if not available:
            cwarn("No additional columns available in this table; choose another.")
            current_table = None
            continue

        console.print(f"\n[bold]Columns in [cyan]{current_table}[/cyan]:[/bold]")
        col_table = Table(show_header=False, box=None, padding=(0, 1))
        col_table.add_column(style="cyan", width=4)
        col_table.add_column()

        for i, c in enumerate(available, 1):
            col_table.add_row(f"{i}.", c)
        col_table.add_row("", "[dim](b) Back to tables | Enter = finish[/dim]")
        console.print(col_table)

        col_input = prompt_input()
        if col_input.lower() == "q":
            raise QuitToMainMenu()
        if col_input == "":
            if selections:
                break
            cwarn("Select at least one column before finishing.")
            continue
        if col_input.lower() == "b":
            current_table = None
            continue

        column: str | None = None
        if col_input.isdigit():
            idx = int(col_input)
            if 1 <= idx <= len(available):
                column = available[idx - 1]
        else:
            for c in available:
                if col_input.lower() == c.lower():
                    column = c
                    break
        if not column:
            cwarn("Please pick a valid column.")
            continue

        selections.append((current_table, column))

        # Show current selections
        console.print("\n[bold dark_sea_green4]Current selection:[/bold dark_sea_green4]")
        for i, (t, c) in enumerate(selections, 1):
            console.print(f"  {i}. [cyan]{t}[/cyan] . {c}")

        if len(selections) >= max_series:
            cinfo(f"Reached the limit of {max_series} series.")
            break

    return selections


def prompt_timestamp_column(
    table_name: str,
    timestamp_cols: list[tuple[str, str | None]],
) -> tuple[str, str | None] | None:
    """Prompt user to select a timestamp column when multiple exist.

    Returns (column_name, format) or None if user wants to go back.
    """
    if len(timestamp_cols) == 1:
        return timestamp_cols[0]

    console.print(f"\n[bold]Table [cyan]{table_name}[/cyan] has multiple timestamp columns:[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", width=4)
    table.add_column()
    table.add_column(style="dim")

    for i, (col_name, fmt) in enumerate(timestamp_cols, 1):
        fmt_display = fmt if fmt else "auto-detect"
        table.add_row(f"{i}.", col_name, f"({fmt_display})")

    table.add_row("", "[dim](b) Back[/dim]", "")
    console.print(table)

    while True:
        choice = prompt_input("Select timestamp column")
        if choice.lower() == "b":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(timestamp_cols):
                return timestamp_cols[idx - 1]
        # Allow exact match by name
        for col_name, fmt in timestamp_cols:
            if choice.lower() == col_name.lower():
                return (col_name, fmt)
        cwarn("Invalid selection. Try again.")


def prompt_label_column(
    table_name: str,
    label_cols: list[str],
) -> str | None:
    """Prompt user to select a TEXT column for hover labels.

    Returns column name or None if user skips/declines.
    """
    if not label_cols:
        return None

    console.print(f"\n[bold]Available label columns in [cyan]{table_name}[/cyan]:[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", width=4)
    table.add_column()

    for i, col_name in enumerate(label_cols, 1):
        table.add_row(f"{i}.", col_name)

    table.add_row("", "[dim](Enter) Skip labels[/dim]")
    console.print(table)

    while True:
        choice = prompt_input("Select label column for hover tooltips")
        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(label_cols):
                return label_cols[idx - 1]
        # Allow exact match by name
        for col_name in label_cols:
            if choice.lower() == col_name.lower():
                return col_name
        cwarn("Invalid selection. Press Enter to skip or pick a valid column.")


def series_picker_mode(
    db_path: Path,
    source_type: str,
    scan_name: str,
) -> list[SeriesSelection] | None:
    """Interactive series picker - returns selections without rendering.

    This is used by the multi-database plotter UI to collect series selections
    from a single database. The selections are returned to the caller for
    accumulation into a SelectionBuffer.

    Args:
        db_path: Path to SQLite database
        source_type: "Exemplar" or "Candidate"
        scan_name: Scan timestamp for display (e.g., "MARS_Test_20251122_074612")

    Returns:
        List of SeriesSelection objects, or None if user cancelled
    """
    chead(f"Series Picker\n[dim]{source_type}: {scan_name}[/dim]\n[dim]{db_path.name}[/dim]")

    # Try to load rubric for semantic column detection
    rubric = load_rubric_for_database(db_path)
    if not rubric:
        cwarn("No rubric found - using column name matching for timestamps")

    with readonly_connection(db_path) as conn:
        tables = detect_tables_with_timestamp(conn, rubric)
        if not tables:
            cerr("No tables with timestamp columns found.")
            return None

        # Filter out empty tables
        tables = [(t, cols, ts_cols, cnt) for t, cols, ts_cols, cnt in tables if cnt > 0]
        if not tables:
            cerr("All tables with timestamps are empty.")
            return None

        tnames = [t for (t, _, _, _) in tables]
        graphable_map: dict[str, list[str]] = {}
        ts_cols_map: dict[str, list[tuple[str, str | None]]] = {}
        ts_col_map: dict[str, str] = {}
        ts_format_map: dict[str, str | None] = {}
        row_count_map: dict[str, int] = {}
        label_cols_map: dict[str, list[str]] = {}

        for tname, col_pairs, timestamp_cols, row_count in tables:
            cols = list_graphable_columns(col_pairs)
            cols = filter_columns_with_non_null(conn, tname, cols)
            graphable_map[tname] = cols
            ts_cols_map[tname] = timestamp_cols
            ts_col_map[tname] = timestamp_cols[0][0]
            ts_format_map[tname] = timestamp_cols[0][1]
            row_count_map[tname] = row_count
            label_cols_map[tname] = list_label_columns(col_pairs)

        if not any(graphable_map.get(t) for t in tnames):
            cerr("Found tables with timestamps but none have numeric columns to plot.")
            return None

        # Prompt for series selection (returns list of (table, column) tuples)
        selections = prompt_series_selection(
            tnames,
            graphable_map,
            max_series=5,
            row_count_map=row_count_map,
            ts_cols_map=ts_cols_map,
            ts_col_map=ts_col_map,
            ts_format_map=ts_format_map,
        )
        if selections is None:
            cinfo("Returning to previous menu.")
            return None
        if not selections:
            cerr("No series selected.")
            return None

        # Optionally prompt for label column (shared across all selections from same table)
        # For simplicity in v1, prompt once if all selections are from same table
        label_col: str | None = None
        unique_tables = list({tbl for tbl, _ in selections})
        if len(unique_tables) == 1:
            table_name = unique_tables[0]
            available_labels = label_cols_map.get(table_name, [])
            if available_labels:
                label_col = prompt_label_column(table_name, available_labels)

        # Convert (table, column) tuples to SeriesSelection objects
        result: list[SeriesSelection] = []
        for table_name, column_name in selections:
            sel = SeriesSelection(
                source_type=source_type,
                scan_name=scan_name,
                db_path=db_path,
                db_name=db_path.stem,
                table_name=table_name,
                column_name=column_name,
                ts_col=ts_col_map[table_name],
                ts_format=ts_format_map.get(table_name),
                label_col=label_col if table_name in unique_tables else None,
            )
            result.append(sel)

        return result
