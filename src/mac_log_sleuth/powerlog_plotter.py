#!/usr/bin/env python3

"""
powerlog_plotter.py

Interactive Plotly-based visualizer for Powerlog SQLite databases.
"""

import argparse
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

# Add the parent directory of this file to the Python path.
# This allows the script to be run directly, as well as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objs as go

from mac_log_sleuth.plotter.db import (
    build_union_dataset,
    detect_tables_with_timestamp,
    fetch_series,
    filter_columns_with_non_null,
    list_graphable_columns,
)
from mac_log_sleuth.plotter.rendering import (
    create_overlay_figure,
    render_plot_html,
    render_stacked_lines,
)
from mac_log_sleuth.plotter.tui import C, cerr, cgood, chead, cinfo, cwarn
from mac_log_sleuth.plotter.utils import to_epoch, try_parse_datetime


def prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        s = input("> ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        # allow exact match by name
        for opt in options:
            if s.lower() == opt.lower():
                return opt
        cwarn("Invalid selection. Try again.")


def prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    """Ask a strict yes/no question.
    Accepts only y/yes or n/no. If default is provided, empty input chooses it.
    """
    while True:
        suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
        s = input(f"{prompt}{suffix}: ").strip().lower()
        if not s and default is not None:
            return bool(default)
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        cwarn("Please answer 'y' or 'n'.")


def prompt_choice_with_disabled(
    prompt: str,
    options: list[str],
    disabled: dict[str, str] | set[str] | None = None,
) -> str:
    """Prompt for a choice but disallow any in ``disabled``.

    ``disabled`` may be a set of option labels or a dict mapping label to a reason.
    Matching is case-insensitive, and disabled options are shown in gray with the reason.
    """

    if disabled is None:
        disabled_map: dict[str, str] = {}
    elif isinstance(disabled, set):
        disabled_map = dict.fromkeys(disabled, "(disabled)")
    else:
        disabled_map = dict(disabled)

    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            note = disabled_map.get(opt, "")
            if opt in disabled_map:
                extra = f" {note}" if note else ""
                print(f"  {i}. {C.FG.GRAY}{opt}{extra}{C.RESET}")
            else:
                print(f"  {i}. {opt}")

        s = input("> ").strip()
        choice: str | None = None
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                choice = options[idx - 1]
        else:
            for opt in options:
                if s.lower() == opt.lower():
                    choice = opt
                    break
        if not choice:
            cwarn("Invalid selection. Try again.")
            continue
        if choice in disabled_map:
            reason = disabled_map[choice]
            cwarn(f"That option is disabled {reason if reason else ''}.")
            continue
        return choice


def prompt_optional_time(label: str) -> float | None:
    print(f"{label} (Format: 'YYYY-MM-DD [HH:MM[:SS]]' or epoch seconds)")
    s = input("> ").strip()
    if not s:
        return None
    dt = try_parse_datetime(s)
    if dt is None:
        cwarn("Unrecognized time; try again.")
        return prompt_optional_time(label)
    return to_epoch(dt)


def prompt_series_selection(
    tnames: list[str],
    graphable_map: dict[str, list[str]],
    max_series: int = 5,
) -> list[tuple[str, str]]:
    """Interactive selector for up to `max_series` (table, column) pairs.

    After the first pick, stays on the same table. Type 'b' to choose a new table,
    press Enter to finish (once at least one series selected), or 'q' to quit.
    """

    selections: list[tuple[str, str]] = []
    current_table: str | None = None
    print(
        "Select up to "
        f"{max_series} series. Type 'q' to quit at any prompt, 'b' to choose a new table,"
        " and press Enter to finish once you have what you need."
    )

    while len(selections) < max_series:
        if current_table is None:
            # Prompt for table
            print("\nTables with graphable columns:")
            for i, t in enumerate(tnames, 1):
                cols = graphable_map.get(t, [])
                if cols:
                    print(f"  {i}. {t}")
                else:
                    print(f"  {i}. {C.FG.GRAY}{t} (no numeric columns){C.RESET}")
            tbl_input = input("> ").strip()
            if tbl_input.lower() == "q":
                sys.exit(0)
            if tbl_input == "" and selections:
                break
            table: str | None = None
            if tbl_input.isdigit():
                idx = int(tbl_input)
                if 1 <= idx <= len(tnames):
                    table = tnames[idx - 1]
            if table is None:
                for t in tnames:
                    if tbl_input.lower() == t.lower():
                        table = t
                        break
            if not table or not graphable_map.get(table):
                cwarn("Please enter a valid table index/name.")
                continue
            current_table = table

        # Column selection for current table
        available = [
            c
            for c in graphable_map[current_table]
            if (current_table, c) not in selections
        ]
        if not available:
            cwarn(
                "No additional columns available in this table; choose another table."
            )
            current_table = None
            continue

        print(f"\nColumns in {current_table} (Enter=finish, 'b'=back, 'q'=quit):")
        for i, c in enumerate(available, 1):
            print(f"  {i}. {c}")
        col_input = input("> ").strip()
        if col_input.lower() == "q":
            sys.exit(0)
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
        print("Current selection:")
        for i, (t, c) in enumerate(selections, 1):
            print(f"  {i}. {t} · {c}")
        if len(selections) >= max_series:
            cinfo(f"Reached the current limit of {max_series} series.")
            break

    return selections


def prompt_chart_type_for_selection(count: int) -> str:
    """Offer chart types with contextual disabling based on series count."""

    if count <= 1:
        return prompt_choice_with_disabled(
            "\nChart type:", ["line", "scatter", "bar"]
        )

    labels = ["overlay line", "stacked line", "scatter", "bar"]
    disabled: dict[str, str] = {}
    if count > 1:
        disabled["bar"] = "(1 datatype max)"
    if count > 2:
        disabled["overlay line"] = "(2 datatypes max)"
        disabled["scatter"] = "(2 datatypes max)"

    choice = prompt_choice_with_disabled("\nChart type:", labels, disabled)
    mapping = {
        "overlay line": "line-overlay",
        "stacked line": "line-stacked",
        "scatter": "scatter",
        "bar": "bar",
    }
    return mapping[choice]


def prompt_epoch_window() -> tuple[float | None, float | None]:
    """Prompt for optional start/end times."""
    epoch_window: tuple[float | None, float | None] = (None, None)
    if prompt_yes_no("Do you want to restrict to a time window?", default=False):
        cinfo("(Times are optional; press Enter to skip.)")
        start_epoch = prompt_optional_time("Start time")
        end_epoch = prompt_optional_time("End time")
        if (
            (start_epoch is not None)
            and (end_epoch is not None)
            and (end_epoch < start_epoch)
        ):
            cwarn("End is before start; swapping.")
            start_epoch, end_epoch = end_epoch, start_epoch
        epoch_window = (start_epoch, end_epoch)
    return epoch_window


def maybe_export_png(
    html_path: Path, fig_callback: Callable[[], go.Figure]
) -> Path | None:
    """
    Optionally export PNG via kaleido if user wants and it's installed.
    fig_callback() must return a fresh figure (because we didn't keep it).
    """
    if not prompt_yes_no("Export PNG image as well?", default=False):
        return None
    try:
        import plotly.io as pio

        fig = fig_callback()
        png_path = html_path.with_suffix(".png")
        # Render at a wider width for clearer bars in PNG
        pio.write_image(fig, str(png_path), width=1200, height=600, scale=2)
        return png_path
    except ModuleNotFoundError:
        cwarn("Kaleido is not installed. To enable PNG export:")
        print("  pip install kaleido --upgrade")
    except Exception as e:
        cwarn(f"PNG export failed: {e}")
    return None


def repl(db_path: Path, outdir: Path):
    chead("Powerlog Plotter (interactive)")
    cinfo(f"Database: {db_path}")
    cinfo("Series selection: type 'q' to quit, 'b' to change tables, Enter to finish.")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = detect_tables_with_timestamp(conn)
        if not tables:
            cerr("No tables with 'timestamp' and data found.")
            return

        tnames = [t for (t, _) in tables]
        graphable_map: dict[str, list[str]] = {}
        ts_col_map: dict[str, str] = {}
        for t, col_pairs in tables:
            cols = list_graphable_columns(col_pairs)
            cols = filter_columns_with_non_null(conn, t, cols)
            graphable_map[t] = cols
            ts_col_map[t] = next(
                (c for c, _ in col_pairs if c.lower() == "timestamp"), "timestamp"
            )

        if not any(graphable_map.get(t) for t in tnames):
            cerr("Found tables with 'timestamp' but none have numeric columns to plot.")
            return

        selections = prompt_series_selection(tnames, graphable_map, max_series=5)
        if not selections:
            cerr("No series selected.")
            return

        epoch_window = prompt_epoch_window()

        while True:
            chart_key = prompt_chart_type_for_selection(len(selections))

            tzmode = prompt_choice("\nTimestamp display:", ["LOCAL", "UTC", "OFFSET"])
            offset_minutes = 0
            if tzmode == "OFFSET":
                print("Enter offset in minutes (e.g., -300 for UTC-5, 120 for UTC+2):")
                while True:
                    s = input("> ").strip()
                    if s and (s.lstrip("+-").isdigit()):
                        offset_minutes = int(s)
                        break
                    cwarn(
                        "Please enter an integer number of minutes (e.g., -300, 0, 60)."
                    )

            smooth_window = 1
            if chart_key in (
                "line-overlay",
                "line-stacked",
                "scatter",
            ) and prompt_yes_no("Apply rolling-mean smoothing?", default=False):
                print("Window size (number of points, e.g., 3, 5, 10). 1 = none")
                while True:
                    s = input("> ").strip()
                    if s.isdigit() and int(s) >= 1:
                        smooth_window = int(s)
                        break
                    cwarn("Enter a positive integer (e.g., 3, 5, 10).")

            step_for_binary = True
            title = Path(db_path).stem

            allow_png = True
            fig_factory: Callable[[], go.Figure] | None = None

            if chart_key == "line-stacked":
                panels = []
                for tbl, col in selections:
                    xs_s, ys_s = fetch_series(
                        conn, tbl, ts_col_map[tbl], [col], epoch_window
                    )
                    if not xs_s:
                        continue
                    panels.append(
                        {"label": f"{tbl} · {col}", "xs": xs_s, "ys": list(ys_s[col])}
                    )
                if not panels:
                    cerr("No data points matched the selection/time window.")
                    return
                html_path = render_stacked_lines(
                    outdir=outdir,
                    base_title=title,
                    tzmode=tzmode,
                    offset_minutes=offset_minutes,
                    panels=panels,
                    step_for_binary=step_for_binary,
                    smooth_window=smooth_window,
                    x_range=epoch_window,
                )
                allow_png = False
            else:
                xs_union, ys_union = build_union_dataset(
                    conn, selections, ts_col_map, epoch_window
                )
                if not xs_union:
                    cerr("No data points matched the selection/time window.")
                    return
                style = (
                    "line"
                    if chart_key in ("line-overlay", "line")
                    else ("scatter" if chart_key == "scatter" else "bar")
                )
                labels = list(ys_union.keys())
                table_title = selections[0][0] if len(selections) == 1 else "Combined"
                y2_col = labels[1] if len(labels) >= 2 else None
                x_limits = epoch_window
                html_path = render_plot_html(
                    outdir=outdir,
                    base_title=title,
                    table=table_title,
                    tzmode=tzmode,
                    offset_minutes=offset_minutes,
                    xs=xs_union,
                    ys=ys_union,
                    style=style,
                    y2_col=y2_col,
                    bar_mode="group",
                    smooth_window=smooth_window,
                    step_for_binary=step_for_binary,
                    line_layout="overlay",
                    x_range=x_limits,
                )

                def fig_factory(
                    xs=xs_union,
                    ys=ys_union,
                    st=style,
                    table_name=table_title,
                    x_rng=x_limits,
                ):
                    from mac_log_sleuth.plotter.rendering import compute_chart_title

                    chart_title_png, _, _ = compute_chart_title(
                        title, xs, x_rng, tzmode, offset_minutes
                    )
                    return create_overlay_figure(
                        xs,
                        ys,
                        st,
                        tzmode,
                        offset_minutes,
                        step_for_binary,
                        chart_title_png,
                        x_range=x_rng,
                    )

            try:
                cgood(f"HTML written: {html_path.resolve()}")
            except Exception:
                cgood(f"HTML written: {html_path}")

            if allow_png and fig_factory is not None:
                png_path = maybe_export_png(html_path, fig_factory)
                if png_path:
                    try:
                        cgood(f"PNG written:  {png_path.resolve()}")
                    except Exception:
                        cgood(f"PNG written:  {png_path}")
            elif not allow_png:
                cinfo("PNG export disabled for stacked line layout (HTML only).")

            if not prompt_yes_no("Create another graph?", default=False):
                return

            if not prompt_yes_no("Reuse the same series selection?", default=True):
                selections = prompt_series_selection(
                    tnames, graphable_map, max_series=5
                )
                if not selections:
                    cerr("No series selected.")
                    return

            if not prompt_yes_no("Reuse the same time window?", default=True):
                epoch_window = prompt_epoch_window()

    finally:
        conn.close()


# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Interactive Plotly visualizer for Powerlog SQLite databases."
    )
    ap.add_argument(
        "--db", required=True, help="Path to Powerlog SQLite (e.g., .PLSQL)"
    )
    ap.add_argument("--outdir", default="Plots", help="Output directory for charts")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        cerr(f"DB not found: {db_path}")
        sys.exit(1)

    # Respect user paths like ~/Desktop/output and make absolute for clarity
    outdir = Path(args.outdir).expanduser()
    repl(db_path, outdir)


if __name__ == "__main__":
    main()
