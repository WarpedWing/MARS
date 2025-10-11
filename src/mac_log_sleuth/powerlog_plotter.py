#!/usr/bin/env python3

"""
powerlog_plotter.py

Interactive Plotly-based visualizer for Powerlog SQLite databases.

Features
- Lists only tables that have a 'timestamp' column AND at least one row.
- Lets user pick 1 or 2 numeric Y columns; 'timestamp' is always the X axis.
- Optional time window (flexible parsing) or full time range.
- Human-readable X axis (local, UTC, or fixed offset).
- Chart types: line, scatter, bar (dual-axis supported; bar overlays the second series).
- Writes HTML to an output directory; optionally exports PNG via Kaleido if available.

Usage
  python powerlog_plotter_v2.py --db /path/to/Powerlog.PLSQL \
    [--outdir ./Plots]

Notes
- If Kaleido isn't installed and you choose PNG export, you'll get a friendly hint:
    pip install kaleido --upgrade

- This script never modifies the source DB.
"""

import argparse
import sqlite3
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Plotly (HTML always; PNG optional via kaleido if installed)
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# ---------------- ANSI ----------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    IT = "\033[3m"
    U = "\033[4m"

    FG = type(
        "FG",
        (),
        {
            "GRAY": "\033[90m",
            "RED": "\033[31m",
            "GREEN": "\033[32m",
            "YELLOW": "\033[33m",
            "BLUE": "\033[34m",
            "MAGENTA": "\033[35m",
            "CYAN": "\033[36m",
            "WHITE": "\033[37m",
        },
    )


def cinfo(msg):
    print(f"{C.FG.CYAN}{msg}{C.RESET}")


def cgood(msg):
    print(f"{C.FG.GREEN}{msg}{C.RESET}")


def cwarn(msg):
    print(f"{C.FG.YELLOW}{msg}{C.RESET}")


def cerr(msg):
    print(f"{C.FG.RED}{msg}{C.RESET}")


def chead(msg):
    print(f"{C.BOLD}{msg}{C.RESET}")


# ------------- Helpers -------------
EPOCH_BOUNDS_DEFAULT = (1262304000.0, 4102444800.0)  # 2010-01-01 .. 2100-01-01


@dataclass
class ProfileEpochBounds:
    lo: float = EPOCH_BOUNDS_DEFAULT[0]
    hi: float = EPOCH_BOUNDS_DEFAULT[1]


def is_floatable(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def try_parse_datetime(s: str) -> datetime | None:
    """
    Accept flexible inputs:
      - Raw epoch seconds (int/float)
      - YYYY-MM-DD
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - YYYY/MM/DD ...
      - ISO-like 'YYYY-MM-DDTHH:MM[:SS]'
    Returns naive datetime in local time.
    """
    s = (s or "").strip()
    if not s:
        return None
    # epoch?
    if is_floatable(s):
        try:
            v = float(s)
            # heuristics: treat large numbers as epoch seconds
            if 1000000000 <= v <= 9999999999:
                return datetime.fromtimestamp(v)
        except Exception:
            pass

    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def to_epoch(dt: datetime) -> float:
    # Treat naive as local time
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.astimezone(UTC).timestamp()


def human_time_from_epoch(
    epoch_s: float,
    tzmode: str,
    offset_minutes: int = 0,
) -> datetime:
    if tzmode == "UTC":
        return datetime.fromtimestamp(epoch_s, tz=UTC)
    if tzmode == "OFFSET":
        return datetime.fromtimestamp(
            epoch_s, tz=timezone(timedelta(minutes=offset_minutes)))
    # Local
    return datetime.fromtimestamp(epoch_s)


def detect_tables_with_timestamp(
    conn: sqlite3.Connection,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Return list of (table_name, columns) where 'timestamp' column exists
    and there is at least one row present.
    """
    cur = conn.cursor()
    out = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (tname,) in cur.fetchall():
        # pragma for columns
        cols = cur.execute(
            f'PRAGMA table_info("{tname.replace('"', '""')}");'
        ).fetchall()
        if not cols:
            continue
        # columns: cid, name, type, notnull, dflt, pk
        has_ts = any("timestamp" in (r[1] or "").lower() for r in cols)
        if not has_ts:
            continue
        # at least one row?
        try:
            cur.execute(f'SELECT 1 FROM "{tname.replace('"', '""')}" LIMIT 1;')
            if cur.fetchone() is None:
                continue
        except sqlite3.DatabaseError:
            continue
        # Build (name, type)
        col_pairs = [(r[1], (r[2] or "").upper()) for r in cols]
        out.append((tname, col_pairs))
    return out


def list_graphable_columns(col_pairs: list[tuple[str, str]]) -> list[str]:
    """
    Return column names that are numeric-like (INTEGER or REAL).
    Excludes 'timestamp' and obvious ID columns.
    """
    out = []
    for name, ctype in col_pairs:
        # skip timestamp (X-axis), and avoid plotting surrogate IDs
        nlow = (name or "").lower()
        if nlow == "timestamp" or nlow == "id" or nlow == "rowid":
            continue
        t = (ctype or "").upper()
        if t.startswith("INT") or t.startswith("REAL"):
            out.append(name)
    return out


def _nice_bucket_seconds(cand: float) -> float:
    """Choose a human-friendly bucket size >= candidate seconds."""
    NICE = [
        1,
        2,
        5,
        10,
        15,
        30,
        60,
        120,
        300,
        600,
        900,
        1800,
        3600,
        7200,
        14400,
        21600,
        43200,
        86400,
    ]
    for n in NICE:
        if cand <= n:
            return float(n)
    return float(NICE[-1])


def bin_time_series(
    xs: list[float],
    ys: dict[str, list[float | None]],
    target_bars: int = 120,
) -> tuple[list[float], dict[str, list[float | None]], float]:
    """
    Downsample into time buckets for bar charts to keep bars visible.
    Returns (bucket_times_epoch, binned_series, suggested_bar_width_ms).
    """
    if not xs:
        return xs, ys, 0.0
    if len(xs) < 2:
        return xs, ys, 60_000.0
    start, end = xs[0], xs[-1]
    span = max(1.0, end - start)
    cand = span / max(1, target_bars)
    bucket_s = _nice_bucket_seconds(cand)
    if bucket_s <= 0:
        bucket_s = 60.0

    # Prepare accumulators per bucket and per series
    sums: dict[int, dict[str, float]] = {}
    counts: dict[int, dict[str, int]] = {}

    cols = list(ys.keys())
    for i, ts in enumerate(xs):
        bidx = int((ts - start) // bucket_s)
        srow = sums.setdefault(bidx, {})
        crow = counts.setdefault(bidx, {})
        for c in cols:
            v = ys[c][i] if i < len(ys[c]) else None
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            srow[c] = srow.get(c, 0.0) + fv
            crow[c] = crow.get(c, 0) + 1

    # Build outputs ordered by bucket index
    bidxs = sorted(sums.keys())
    bx: list[float] = [start + idx * bucket_s for idx in bidxs]
    by: dict[str, list[float | None]] = {c: [] for c in cols}
    for idx in bidxs:
        srow = sums.get(idx, {})
        crow = counts.get(idx, {})
        for c in cols:
            cnt = crow.get(c, 0)
            if cnt <= 0:
                by[c].append(None)
            else:
                by[c].append(srow.get(c, 0.0) / cnt)

    width_ms = bucket_s * 1000.0 * 0.8
    return bx, by, width_ms


def filter_columns_with_non_null(
    conn: sqlite3.Connection,
    table: str,
    cols: list[str],
) -> list[str]:
    """
    Keep only columns that have at least one non-NULL value in the table.
    """
    cur = conn.cursor()
    qtable = table.replace('"', '""')
    kept: list[str] = []
    for col in cols:
        qcol = col.replace('"', '""')
        try:
            cur.execute(f'SELECT 1 FROM "{qtable}" WHERE "{qcol}" IS NOT NULL LIMIT 1;')
            if cur.fetchone() is not None:
                kept.append(col)
        except sqlite3.DatabaseError:
            # if any issue, be conservative and drop the column from choices
            continue
    return kept


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


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_label(label: str, max_len: int = 30) -> str:
    if len(label) <= max_len:
        return label
    return "..." + label[-max_len:]


def compute_chart_title(
    base_title: str,
    xs: list[float],
    x_range: tuple[float | None, float | None] | None,
    tzmode: str,
    offset_minutes: int,
) -> tuple[str, float | None, float | None]:
    data_min = min(xs) if xs else None
    data_max = max(xs) if xs else None
    start_epoch = x_range[0] if (x_range and x_range[0] is not None) else data_min
    end_epoch = x_range[1] if (x_range and x_range[1] is not None) else data_max
    if start_epoch is None and end_epoch is None:
        return base_title, None, None
    if start_epoch is None:
        start_epoch = end_epoch
    if end_epoch is None:
        end_epoch = start_epoch
    if start_epoch > end_epoch:
        start_epoch, end_epoch = end_epoch, start_epoch
    start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
    end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    if start_str == end_str:
        title = f"{base_title}: {start_str}"
    else:
        title = f"{base_title}: {start_str} to {end_str}"
    return title, start_epoch, end_epoch


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
        choice = prompt_choice_with_disabled(
            "\nChart type:", ["line", "scatter", "bar"]
        )
        return choice

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


def build_union_dataset(
    conn: sqlite3.Connection,
    selections: list[tuple[str, str]],
    ts_col_map: dict[str, str],
    epoch_window: tuple[float | None, float | None] | None,
) -> tuple[list[float], dict[str, list[float | None]]]:
    """Fetch each (table, column) and align by global timestamps."""
    series_data: list[tuple[str, list[float], list[float | None]]] = []
    all_ts: set[float] = set()
    for table, column in selections:
        xs, ys = fetch_series(conn, table, ts_col_map[table], [column], epoch_window)
        if not xs:
            continue
        label = f"{table} · {column}"
        values = ys.get(column, [])
        series_data.append((label, xs, values))
        all_ts.update(xs)
    union_x = sorted(all_ts)
    if not union_x:
        return [], {{}}
    union_series: dict[str, list[float | None]] = {}
    for label, xs, values in series_data:
        lookup = dict(zip(xs, values))
        union_series[label] = [lookup.get(x) for x in union_x]
    return union_x, union_series


def create_overlay_figure(
    xs: list[float],
    ys: dict[str, list[float | None]],
    style: str,
    tzmode: str,
    offset_minutes: int,
    step_for_binary: bool,
    title: str,
    x_range: tuple[float | None, float | None] | None = None,
) -> go.Figure:
    labels = list(ys.keys())
    x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs]
    fig = go.Figure()

    if not labels:
        return fig

    primary = labels[0]
    primary_data = ys[primary]
    if style == "line":
        shape1 = (
            "hv" if (step_for_binary and is_binary_series(primary_data)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=primary_data,
                mode="lines+markers",
                name=primary,
                yaxis="y1",
                line_shape=shape1,
            )
        )
    elif style == "scatter":
        fig.add_trace(
            go.Scatter(x=x_dt, y=primary_data, mode="markers", name=primary, yaxis="y1")
        )
    elif style == "bar":
        fig.add_trace(
            go.Bar(x=x_dt, y=primary_data, name=primary, yaxis="y1", opacity=0.75)
        )

    if len(labels) >= 2:
        secondary = labels[1]
        secondary_data = ys[secondary]
        if style == "line":
            shape2 = (
                "hv"
                if (step_for_binary and is_binary_series(secondary_data))
                else "linear"
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=secondary_data,
                    mode="lines+markers",
                    name=secondary,
                    yaxis="y2",
                    line_shape=shape2,
                )
            )
        elif style == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=secondary_data,
                    mode="markers",
                    name=secondary,
                    yaxis="y2",
                )
            )
        elif style == "bar":
            fig.add_trace(
                go.Bar(
                    x=x_dt,
                    y=secondary_data,
                    name=secondary,
                    yaxis="y1",
                    opacity=0.55,
                )
            )

    fig.update_layout(
        title=title,
        xaxis={
            "title": "Time",
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": labels[0],
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
        height=600,
    )

    if len(labels) >= 2 and style != "bar":
        fig.update_layout(
            yaxis2={
                "title": labels[1],
                "overlaying": "y",
                "side": "right",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
    elif len(labels) >= 2 and style == "bar":
        fig.update_layout(
            yaxis2={
                "title": labels[1],
                "overlaying": "y",
                "side": "right",
                "matches": "y",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
        fig.update_layout(barmode="group")

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    fig.update_layout(
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "text": f"X-axis: {tzlabel}. Data plotted as acquired from Powerlog.",
                "showarrow": False,
                "font": {"size": 11},
            }
        ]
    )

    return fig


def is_binary_series(values: Iterable[float | None]) -> bool:
    """True if non-null values are a subset of {0,1}."""
    found = set()
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            return False
        if fv not in (0.0, 1.0):
            return False
        found.add(int(fv))
    return True


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return list(values)
    out: list[float | None] = []
    acc: list[float | None] = []
    s = 0.0
    c = 0
    for v in values:
        acc.append(v)
        if v is not None:
            s += float(v)
            c += 1
        if len(acc) > window:
            old = acc.pop(0)
            if old is not None:
                s -= float(old)
                c -= 1
        out.append((s / c) if c > 0 else None)
    return out


def fetch_series(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    y_cols: list[str],
    epoch_window: tuple[float | None, float | None] | None = None,
) -> tuple[list[float], dict[str, list[float | None]]]:
    """
    Fetch X (epoch timestamps) and Y series (dict per selected column).
    Applies optional [start,end] epoch window filtering at SQL level.
    """
    cur = conn.cursor()
    qtable = table.replace('"', '""')
    q_ts = ts_col.replace('"', '""')
    select_cols = [q_ts] + [c.replace('"', '""') for c in y_cols]
    sel = ", ".join(f'"{c}"' for c in select_cols)
    sql = f'SELECT {sel} FROM "{qtable}"'
    params: list[Any] = []
    if epoch_window and any(epoch_window):
        clauses = []
        if epoch_window[0] is not None:
            clauses.append(f'"{q_ts}" >= ?')
            params.append(epoch_window[0])
        if epoch_window[1] is not None:
            clauses.append(f'"{q_ts}" <= ?')
            params.append(epoch_window[1])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
    sql += f' ORDER BY "{q_ts}" ASC'
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.DatabaseError as e:
        cerr(f"Query failed: {e}")
        return [], {c: [] for c in y_cols}

    xs: list[float] = []
    ys: dict[str, list[float | None]] = {c: [] for c in y_cols}
    for r in rows:
        # r[0] = timestamp
        ts = r[0]
        if ts is None:
            continue
        try:
            tsf = float(ts)
        except Exception:
            continue
        xs.append(tsf)
        for idx, col in enumerate(y_cols, start=1):
            v = r[idx]
            if v is None:
                ys[col].append(None)
            else:
                # numeric-only series
                try:
                    ys[col].append(float(v))
                except Exception:
                    ys[col].append(None)
    return xs, ys


def render_stacked_lines(
    outdir: Path,
    base_title: str,
    tzmode: str,
    offset_minutes: int,
    panels: list[dict],  # each: {label:str, xs:list[float], ys:list[float|None]}
    step_for_binary: bool = True,
    smooth_window: int = 1,
    x_range: tuple[float | None, float | None] | None = None,
) -> Path:
    """
    Render stacked line panels (1 per series). Each panel is independent and
    shares the time axis. Returns HTML path. PNG export handled by caller.
    """
    if not panels:
        raise ValueError("No panels to plot.")

    rows = len(panels)
    subplot_titles = [truncate_label(p.get("label", "")) for p in panels]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
    )
    all_xs: list[float] = []
    for i, panel in enumerate(panels, start=1):
        xs = panel.get("xs", [])
        ys = panel.get("ys", [])
        label = panel.get("label", "")
        all_xs.extend(xs)
        is_bool = is_binary_series(ys)
        yplot = (
            ys if (is_bool or smooth_window <= 1) else rolling_mean(list(ys), smooth_window)
        )
        x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs]
        shape = "hv" if (step_for_binary and is_bool) else "linear"
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=yplot,
                mode="lines+markers",
                name=label,  # Full label for legend
                line_shape=shape,
            ),
            row=i,
            col=1,
        )

    chart_title, start_epoch, end_epoch = compute_chart_title(
        base_title, all_xs, x_range, tzmode, offset_minutes
    )
    fig.update_layout(
        title=chart_title,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
        height=max(600, 350 + 220 * (rows - 1)),
    )

    if start_epoch is not None and end_epoch is not None:
        start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
        end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
        for row in range(1, rows + 1):
            fig.update_xaxes(range=[start_dt, end_dt], row=row, col=1)

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    annotations = list(fig.layout.annotations)
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.28,
            "text": f"X-axis: {tzlabel}. Data plotted as acquired from Powerlog.",
            "showarrow": False,
            "font": {"size": 11},
        }
    )
    fig.update_layout(annotations=annotations)

    outdir = ensure_outdir(outdir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    html_path = outdir / f"stacked.{rows}.{ts}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    return html_path


def render_plot_html(
    outdir: Path,
    base_title: str,
    table: str,
    tzmode: str,
    offset_minutes: int,
    xs: list[float],
    ys: dict[str, list[float | None]],
    style: str,
    y2_col: str | None = None,
    *,
    bar_mode: str = "group",  # "group" or "overlay"
    smooth_window: int = 1,
    step_for_binary: bool = True,
    line_layout: str = "overlay",  # "overlay" or "stacked"
    x_range: tuple[float | None, float | None] | None = None,
) -> Path:
    """
    Build Plotly figure and save as HTML. Returns the HTML path.
    """
    # For bar charts, bin densely-sampled data so bars are visible
    xs_used, ys_used = xs, ys
    width_ms = 0.0
    if style == "bar":
        xs_used, ys_used, width_ms = bin_time_series(xs, ys, target_bars=120)
    # Detect boolean series on original data (pre-smoothing)
    binary_map = {k: is_binary_series(ys[k]) for k in ys}
    # Convert X to human-readable datetimes in chosen tz
    x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs_used]

    fig = go.Figure()

    # Optional smoothing for non-bar styles (skip boolean series)
    if style != "bar" and smooth_window and smooth_window > 1:
        ys_used = {
            k: (v if binary_map.get(k, False) else rolling_mean(v, smooth_window))
            for k, v in ys_used.items()
        }

    series = list(ys_used.keys())
    if not series:
        raise ValueError("No Y series to plot.")

    # width_ms is suggested by bin_time_series when style == "bar"
    chart_title, start_epoch, end_epoch = compute_chart_title(
        base_title, xs_used, x_range, tzmode, offset_minutes
    )

    # This is dead code as of the current REPL logic, but kept for potential future use.
    if style == "line" and y2_col and (line_layout.lower().startswith("stack")):
        # Build stacked subplots sharing X
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(truncate_label(series[0]), truncate_label(y2_col)),
        )
        shape1 = (
            "hv" if (step_for_binary and binary_map.get(series[0], False)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[series[0]],
                mode="lines+markers",
                name=truncate_label(series[0]),
                line_shape=shape1,
            ),
            row=1,
            col=1,
        )
        shape2 = (
            "hv" if (step_for_binary and binary_map.get(y2_col, False)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[y2_col],
                mode="lines+markers",
                name=truncate_label(y2_col),
                line_shape=shape2,
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            title=chart_title,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
            },
            margin={"l": 60, "r": 60, "t": 70, "b": 100},
            height=700,
        )
        if start_epoch is not None and end_epoch is not None:
            start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
            end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
            fig.update_xaxes(range=[start_dt, end_dt], row=1, col=1)
            fig.update_xaxes(range=[start_dt, end_dt], row=2, col=1)

        panel_titles = {truncate_label(series[0]), truncate_label(y2_col)}
        for ann in fig.layout.annotations:
            if ann.text in panel_titles:
                ann.update(
                    x=0.5, xanchor="center", y=1.08, yanchor="bottom", font={"size": 14}
                )

        tzlabel = {
            "LOCAL": "Local time",
            "UTC": "UTC",
            "OFFSET": f"UTC{offset_minutes:+d}m",
        }.get(tzmode, "Local time")
        annotations = list(fig.layout.annotations)
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "text": (
                    f"X-axis: human-readable {tzlabel}. "
                    f"Series: {truncate_label(series[0])} (top), {truncate_label(y2_col)} (bottom). "
                    "Data plotted as acquired from Powerlog."
                ),
                "showarrow": False,
                "font": {"size": 11},
            }
        )
        fig.update_layout(annotations=annotations)
        # Write and return (reuse writer below)
        outdir = ensure_outdir(outdir)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")

        def safe_name(s: str | None) -> str:
            s = (s or "").strip()
            safe = []
            for ch in s:
                if ch.isalnum() or ch in ("_", "-", ".", "+"):
                    safe.append(ch)
                else:
                    safe.append("_")
            out = "".join(safe)
            return out or "series"

        fname = f"{safe_name(table)}.{safe_name(series[0])}-{safe_name(y2_col)}.line-stacked.{ts}.html"
        html_path = outdir / fname
        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        except Exception:
            from plotly.offline import plot as plotly_plot  # type: ignore

            plotly_plot(
                fig, filename=str(html_path), auto_open=True, include_plotlyjs="cdn"
            )
        if not html_path.exists():
            cwarn(f"HTML not found after write attempt: {html_path}")
        return html_path

    # Primary series
    y1 = series[0]
    if style == "line":
        shape1 = "hv" if (step_for_binary and binary_map.get(y1, False)) else "linear"
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[y1],
                mode="lines+markers",
                name=y1,
                yaxis="y1",
                line_shape=shape1,
            )
        )
    elif style == "scatter":
        fig.add_trace(
            go.Scatter(x=x_dt, y=ys_used[y1], mode="markers", name=y1, yaxis="y1")
        )
    elif style == "bar":
        fig.add_trace(
            go.Bar(
                x=x_dt,
                y=ys_used[y1],
                name=y1,
                yaxis="y1",
                width=width_ms if width_ms > 0 else None,
                opacity=0.75,
            )
        )
    else:
        raise ValueError("Unknown style.")

    # Optional secondary series on y2 (overlay)
    y2_on_y2_axis = (
        y2_col
        and y2_col in ys_used
        and (style != "bar" or bar_mode.lower().startswith("over"))
    )

    if y2_col and y2_col in ys_used:
        y2_axis = "y2" if y2_on_y2_axis else "y1"
        if style == "line":
            shape2 = (
                "hv"
                if (step_for_binary and binary_map.get(y2_col, False))
                else "linear"
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=ys_used[y2_col],
                    mode="lines+markers",
                    name=y2_col,
                    yaxis=y2_axis,
                    line_shape=shape2,
                )
            )
        elif style == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=ys_used[y2_col],
                    mode="markers",
                    name=y2_col,
                    yaxis=y2_axis,
                )
            )
        elif style == "bar":
            fig.add_trace(
                go.Bar(
                    x=x_dt,
                    y=ys_used[y2_col],
                    name=y2_col,
                    yaxis=y2_axis,
                    width=width_ms if width_ms > 0 else None,
                    opacity=0.55,
                )
            )
        else:
            pass

    # Layout
    fig.update_layout(
        title=chart_title,
        xaxis={
            "title": "Time",
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": series[0],
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
    )

    if y2_on_y2_axis:
        fig.update_layout(
            yaxis2={
                "title": y2_col,
                "overlaying": "y",
                "side": "right",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
    elif (y2_col in series) and style == "bar":
        fig.update_layout(
            yaxis2={
                "title": y2_col,
                "overlaying": "y",
                "side": "right",
                "matches": "y",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            },
            barmode="group",
        )
    elif style == "bar":
        fig.update_layout(
            barmode=("overlay" if bar_mode.lower().startswith("over") else "group")
        )

    if start_epoch is not None and end_epoch is not None:
        start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
        end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
        fig.update_xaxes(range=[start_dt, end_dt])

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    foot = (
        f"X-axis: human-readable {tzlabel}. "
        f"Series plotted: {', '.join(series)}. Data plotted as acquired from Powerlog."
    )
    annotations = list(fig.layout.annotations)
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.28,
            "text": foot,
            "showarrow": False,
            "font": {"size": 11},
        }
    )
    fig.update_layout(annotations=annotations, height=600)

    def safe_name(s: str | None) -> str:
        s = (s or "").strip()
        # replace any path-separator or problematic chars
        safe = []
        for ch in s:
            if ch.isalnum() or ch in ("_", "-", ".", "+"):
                safe.append(ch)
            else:
                safe.append("_")
        out = "".join(safe)
        return out or "series"

    outdir = ensure_outdir(outdir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"{safe_name(table)}.{safe_name(series[0])}{('-' + safe_name(y2_col)) if y2_col else ''}.{style}.{ts}.html"
    html_path = outdir / fname

    try:
        # Prefer the explicit writer to ensure file creation across Plotly versions
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        # Fallback to offline.plot if available
        try:
            from plotly.offline import plot as plotly_plot  # type: ignore

            plotly_plot(
                fig, filename=str(html_path), auto_open=False, include_plotlyjs="cdn"
            )
        except Exception as e:
            cerr(f"Failed to write HTML: {e}")
            raise

    if not html_path.exists():
        cwarn(f"HTML not found after write attempt: {html_path}")
    return html_path


def maybe_export_png(html_path: Path, fig_callback) -> Path | None:
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


# ------------- REPL -------------
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