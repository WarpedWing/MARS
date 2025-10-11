#!/usr/bin/env python3

"""
powerlog_plotter_v2.py

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
from collections.abc import Iterable
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
    epoch_s: float, tzmode: str, offset_minutes: int = 0
) -> datetime:
    if tzmode == "UTC":
        return datetime.fromtimestamp(epoch_s, tz=UTC)
    if tzmode == "OFFSET":
        return datetime.fromtimestamp(
            epoch_s, tz=timezone(timedelta(minutes=offset_minutes))
        )
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
    xs: list[float], ys: dict[str, list[float | None]], target_bars: int = 120
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
    conn: sqlite3.Connection, table: str, cols: list[str]
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
    prompt: str, options: list[str], disabled: set[str]
) -> str:
    """Prompt for a choice but disallow any in disabled.
    Matching is case-insensitive, and disabled options are shown in gray.
    """
    disabled_lower = {d.lower() for d in disabled}
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            if opt.lower() in disabled_lower:
                print(f"  {i}. {C.FG.GRAY}{opt} (disabled){C.RESET}")
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
        if choice.lower() in disabled_lower:
            cwarn("That option is disabled. Choose another.")
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
    scatter_trend: bool = False,
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

    # Special case: stacked line layout (two panels)
    if style == "line" and y2_col and (line_layout.lower().startswith("stack")):
        # Build stacked subplots sharing X
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(series[0], y2_col),
        )
        shape1 = (
            "hv" if (step_for_binary and binary_map.get(series[0], False)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[series[0]],
                mode="lines+markers",
                name=series[0],
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
                name=y2_col,
                line_shape=shape2,
            ),
            row=2,
            col=1,
        )
        # Layout
        title = f"{base_title} · {table} — {series[0]} vs {y2_col}"
        fig.update_layout(
            title=title,
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
        # Footer annotation
        tzlabel = {
            "LOCAL": "Local time",
            "UTC": "UTC",
            "OFFSET": f"UTC{offset_minutes:+d}m",
        }.get(tzmode, "Local time")
        foot = (
            f"X-axis: human-readable {tzlabel}. "
            f"Series: {series[0]} (top), {y2_col} (bottom). "
            "Data plotted as acquired from Powerlog."
        )
        fig.update_layout(
            annotations=[
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.28,
                    "text": foot,
                    "showarrow": False,
                    "font": {"size": 11},
                }
            ]
        )
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
                fig, filename=str(html_path), auto_open=False, include_plotlyjs="cdn"
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
        if scatter_trend:
            # Simple least-squares fit on epoch seconds vs value
            pts = [
                (xs_used[i], float(ys_used[y1][i]))
                for i in range(min(len(xs_used), len(ys_used[y1])))
                if ys_used[y1][i] is not None
            ]
            if len(pts) >= 2:
                xs_, ys_ = zip(*pts)
                n = len(xs_)
                sx = sum(xs_)
                sy = sum(ys_)
                sxx = sum(x * x for x in xs_)
                sxy = sum(x * y for x, y in pts)
                denom = n * sxx - sx * sx
                if denom != 0:
                    a = (n * sxy - sx * sy) / denom
                    b = (sy - a * sx) / n
                    yhat = [a * x + b for x in xs_used]
                    fig.add_trace(
                        go.Scatter(
                            x=x_dt,
                            y=yhat,
                            mode="lines",
                            name=f"{y1} trend",
                            yaxis="y1",
                            line={"dash": "dash", "color": "#555"},
                            showlegend=True,
                        )
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
    # Title: DB · Table — Y1[ vs Y2]
    title = f"{base_title} · {table} — {series[0]}"
    if y2_col and (y2_col in series):
        title += f" vs {y2_col}"
    fig.update_layout(
        title=title,
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

    # Add secondary axis label/scale
    if y2_on_y2_axis:
        # Right axis actually used (overlay mode or non-bar)
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
        # Grouped bars: show a matched right axis for labeling/ticks
        fig.update_layout(
            yaxis2={
                "title": y2_col,
                "overlaying": "y",
                "side": "right",
                "matches": "y",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )

    if style == "bar":
        # group = side-by-side
        fig.update_layout(
            barmode=("overlay" if bar_mode.lower().startswith("over") else "group")
        )

    # Footer in annotations
    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    foot = (
        f"X-axis: human-readable {tzlabel}. "
        f"Y-axis1: {series[0]}{'; Y-axis2: ' + y2_col if (y2_col and y2_col in series) else ''}. "
        "Data plotted as acquired from Powerlog."
    )
    fig.update_layout(
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "text": foot,
                "showarrow": False,
                "font": {"size": 11},
            }
        ],
        height=600,
    )

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

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Step 1: list available tables
        tables = detect_tables_with_timestamp(conn)
        if not tables:
            cerr("No tables with 'timestamp' and data found.")
            return

        # Precompute graphable columns per table (exclude all-null)
        tnames = [t for (t, _) in tables]
        graphable_map: dict[str, list[str]] = {}
        for t, col_pairs in tables:
            cols = list_graphable_columns(col_pairs)
            cols = filter_columns_with_non_null(conn, t, cols)
            graphable_map[t] = cols

        print("\nAvailable tables with timestamp & data:")
        for i, t in enumerate(tnames, 1):
            cols = graphable_map.get(t, [])
            if cols:
                print(f"  {i}. {t}")
            else:
                print(f"  {i}. {C.FG.GRAY}{t} (no numeric columns){C.RESET}")

        if not any(graphable_map.get(t) for t in tnames):
            cerr("Found tables with 'timestamp' but none have numeric columns to plot.")
            return

        # Pick a graphable table; keep prompting if not
        table: str | None = None
        while True:
            sel = input("\nSelect table (number or exact name): ").strip()
            table = None
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(tnames):
                    table = tnames[idx - 1]
            if table is None:
                for t in tnames:
                    if sel.lower() == t.lower():
                        table = t
                        break
            if not table:
                cerr("Invalid table selection. Try again.")
                continue
            if not graphable_map.get(table):
                cwarn(
                    "That table has no numeric columns to plot. Choose another table."
                )
                continue
            break

        # Columns
        col_pairs = next(cols for (t, cols) in tables if t == table)
        ts_col = next((c for c, ty in col_pairs if c.lower() == "timestamp"), None)
        if not ts_col:
            cerr("Internal error: no timestamp column found.")
            return

        graphable = graphable_map[table]
        if not graphable:
            cerr("No numeric columns available to graph in this table.")
            # Shouldn't happen due to earlier checks, but guard anyway
            return

        # Tell user timestamp is always X
        print("\nTimestamp is always X-axis.")
        print("Select the primary Y-axis column:")
        for i, c in enumerate(graphable, 1):
            print(f"  {i}. {c}")
        sel = input("> ").strip()
        y1 = None
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(graphable):
                y1 = graphable[idx - 1]
        else:
            for c in graphable:
                if sel.lower() == c.lower():
                    y1 = c
                    break
        if not y1:
            cerr("Invalid selection for primary Y-axis.")
            return

        y2 = None
        if prompt_yes_no("Add a secondary Y-axis column?", default=False):
            if len(graphable) <= 1:
                cwarn("No second numeric column available.")
            else:
                y2 = prompt_choice_with_disabled(
                    "Select the secondary Y-axis column:", graphable, {y1}
                )

        # Time window (optional)
        # type: tuple[Optional[float], Optional[float]]
        epoch_window = (None, None)
        if prompt_yes_no("Do you want to restrict to a time window?", default=False):
            cinfo("(Time is optional.)")
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

        # Chart type
        style = prompt_choice("\nSelect chart type:", ["line", "scatter", "bar"])
        bar_mode = "group"
        line_layout = "overlay"
        scatter_trend = False
        if style == "bar":
            # One series only for bar charts
            if y2:
                cwarn("Bar charts support one series; removing secondary column.")
                y2 = None
            # Layout only matters if two bars; keep prompt for consistency
            bar_mode = prompt_choice(
                "Bar layout:", ["side-by-side", "overlay"]
            ).replace("side-by-side", "group")
        elif style == "line" and y2:
            line_layout = prompt_choice("Line layout:", ["overlay", "stacked"]).lower()
        elif style == "scatter":
            scatter_trend = prompt_yes_no("Add best-fit trend line?", default=False)

        # Timestamp display format
        tzmode = prompt_choice("\nTimestamp display:", ["LOCAL", "UTC", "OFFSET"])
        offset_minutes = 0
        if tzmode == "OFFSET":
            print("Enter offset in minutes (e.g., -300 for UTC-5, 120 for UTC+2):")
            while True:
                s = input("> ").strip()
                if s and (s.lstrip("+-").isdigit()):
                    offset_minutes = int(s)
                    break
                cwarn("Please enter an integer number of minutes (e.g., -300, 0, 60).")

        # Optional smoothing for line/scatter
        smooth_window = 1
        if style in ("line", "scatter") and prompt_yes_no(
            "Apply rolling-mean smoothing?", default=False
        ):
            print("Window size (number of points, e.g., 3, 5, 10). 1 = none")
            while True:
                s = input("> ").strip()
                if s.isdigit() and int(s) >= 1:
                    smooth_window = int(s)
                    break
                cwarn("Enter a positive integer (e.g., 3, 5, 10).")

        # Line shape for binary series
        step_for_binary = True
        if style == "line":
            step_for_binary = prompt_yes_no(
                "Use step lines for boolean (0/1) series?", default=True
            )

        # Fetch and render
        y_cols = [y1] + ([y2] if y2 else [])
        xs, ys = fetch_series(conn, table, ts_col, y_cols, epoch_window)
        if not xs:
            cerr("No data points matched the selection/time window.")
            return

        title = Path(db_path).stem
        html_path = render_plot_html(
            outdir=outdir,
            base_title=title,
            table=table,
            tzmode=tzmode,
            offset_minutes=offset_minutes,
            xs=xs,
            ys=ys,
            style=style,
            y2_col=y2,
            bar_mode=bar_mode,
            smooth_window=smooth_window,
            step_for_binary=step_for_binary,
        )
        try:
            cgood(f"HTML written: {html_path.resolve()}")
        except Exception:
            cgood(f"HTML written: {html_path}")

        # Re-create figure for PNG export only if user asks
        def fig_factory():
            # build the same figure again deterministically
            # This mirrors render_plot_html (without writing HTML)
            xs_used, ys_used, width_ms = (
                bin_time_series(xs, ys, target_bars=120)
                if style == "bar"
                else (xs, ys, 0.0)
            )
            # Detect boolean series on original values
            binary_map = {k: is_binary_series(ys[k]) for k in ys}
            # smoothing (same as HTML; skip boolean series)
            if style != "bar" and smooth_window and smooth_window > 1:
                ys_used = {
                    k: (
                        v
                        if binary_map.get(k, False)
                        else rolling_mean(v, smooth_window)
                    )
                    for k, v in ys_used.items()
                }
            x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs_used]
            fig = go.Figure()
            series = list(ys_used.keys())
            y1name = series[0]

            # Primary series
            if style == "line":
                shape1 = (
                    "hv"
                    if (step_for_binary and binary_map.get(y1name, False))
                    else "linear"
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_dt,
                        y=ys_used[y1name],
                        mode="lines+markers",
                        name=y1name,
                        yaxis="y1",
                        line_shape=shape1,
                    )
                )
            elif style == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=x_dt,
                        y=ys_used[y1name],
                        mode="markers",
                        name=y1name,
                        yaxis="y1",
                    )
                )
            elif style == "bar":
                fig.add_trace(
                    go.Bar(
                        x=x_dt,
                        y=ys_used[y1name],
                        name=y1name,
                        yaxis="y1",
                        width=width_ms if width_ms > 0 else None,
                        opacity=0.75,
                    )
                )

            # Optional secondary series
            y2name = series[1] if len(series) >= 2 else None
            y2_on_y2_axis = y2name and (
                style != "bar" or bar_mode.lower().startswith("over")
            )

            if y2name:
                y2_axis = "y2" if y2_on_y2_axis else "y1"
                if style == "line":
                    shape2 = (
                        "hv"
                        if (step_for_binary and binary_map.get(y2name, False))
                        else "linear"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_dt,
                            y=ys_used[y2name],
                            mode="lines+markers",
                            name=y2name,
                            yaxis=y2_axis,
                            line_shape=shape2,
                        )
                    )
                elif style == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=x_dt,
                            y=ys_used[y2name],
                            mode="markers",
                            name=y2name,
                            yaxis=y2_axis,
                        )
                    )
                elif style == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=x_dt,
                            y=ys_used[y2name],
                            name=y2name,
                            yaxis=y2_axis,
                            width=width_ms if width_ms > 0 else None,
                            opacity=0.55,
                        )
                    )

            # Layout
            title_str = f"{title} · {table} — {y1name}"
            if y2name:
                title_str += f" vs {y2name}"
            fig.update_layout(
                title=title_str,
                xaxis={
                    "title": "Time",
                    "showgrid": True,
                    "zeroline": False,
                    "automargin": True,
                },
                yaxis={
                    "title": y1name,
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
                annotations=[
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0,
                        "y": -0.28,
                        "text": "Re-rendered for PNG export.",
                        "showarrow": False,
                        "font": {"size": 11},
                    }
                ],
            )

            if y2_on_y2_axis:
                fig.update_layout(
                    yaxis2={
                        "title": y2name,
                        "overlaying": "y",
                        "side": "right",
                        "automargin": True,
                        "showgrid": False,
                        "zeroline": False,
                    }
                )
            elif y2name and style == "bar":
                # Grouped bars: show a matched right axis for labeling/ticks
                fig.update_layout(
                    yaxis2={
                        "title": y2name,
                        "overlaying": "y",
                        "side": "right",
                        "matches": "y",
                        "automargin": True,
                        "showgrid": False,
                        "zeroline": False,
                    }
                )

            if style == "bar":
                fig.update_layout(
                    barmode=("overlay" if bar_mode.startswith("overlay") else "group")
                )

            return fig

        png_path = maybe_export_png(html_path, fig_factory)
        if png_path:
            try:
                cgood(f"PNG written:  {png_path.resolve()}")
            except Exception:
                cgood(f"PNG written:  {png_path}")

        # Offer to generate another chart for easier testing
        while prompt_yes_no("Create another graph?", default=False):
            # Reuse time window or pick a new one
            if not prompt_yes_no("Reuse the same time window?", default=True):
                cinfo("(Time is optional.)")
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

            # Chart type and layout
            style = prompt_choice("\nSelect chart type:", ["line", "scatter", "bar"])
            bar_mode = "group"
            if style == "bar":
                bar_mode = prompt_choice(
                    "Bar layout:", ["side-by-side", "overlay"]
                ).replace("side-by-side", "group")

            # Timestamp display format
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

            # Optional smoothing for line/scatter
            smooth_window = 1
            if style in ("line", "scatter") and prompt_yes_no(
                "Apply rolling-mean smoothing?", default=False
            ):
                print("Window size (number of points, e.g., 3, 5, 10). 1 = none")
                while True:
                    s = input("> ").strip()
                    if s.isdigit() and int(s) >= 1:
                        smooth_window = int(s)
                        break
                    cwarn("Enter a positive integer (e.g., 3, 5, 10).")

            # Line shape for binary series
            step_for_binary = True
            if style == "line":
                step_for_binary = prompt_yes_no(
                    "Use step lines for boolean (0/1) series?", default=True
                )

            # Fetch and render
            y_cols = [y1] + ([y2] if y2 else [])
            xs, ys = fetch_series(conn, table, ts_col, y_cols, epoch_window)
            if not xs:
                cerr("No data points matched the selection/time window.")
                break

            html_path = render_plot_html(
                outdir=outdir,
                base_title=title,
                table=table,
                tzmode=tzmode,
                offset_minutes=offset_minutes,
                xs=xs,
                ys=ys,
                style=style,
                y2_col=y2,
                bar_mode=bar_mode,
                smooth_window=smooth_window,
                step_for_binary=step_for_binary,
                line_layout=line_layout,
                scatter_trend=scatter_trend,
            )
            try:
                cgood(f"HTML written: {html_path.resolve()}")
            except Exception:
                cgood(f"HTML written: {html_path}")

            def fig_factory_again():
                xs_used, ys_used, width_ms = (
                    bin_time_series(xs, ys, target_bars=120)
                    if style == "bar"
                    else (xs, ys, 0.0)
                )
                binary_map = {k: is_binary_series(ys[k]) for k in ys}
                if style != "bar" and smooth_window and smooth_window > 1:
                    ys_used = {
                        k: (
                            v
                            if binary_map.get(k, False)
                            else rolling_mean(v, smooth_window)
                        )
                        for k, v in ys_used.items()
                    }
                x_dt = [
                    human_time_from_epoch(x, tzmode, offset_minutes) for x in xs_used
                ]
                fig = go.Figure()
                series2 = list(ys_used.keys())
                y1name = series2[0]
                if style == "line":
                    shape1 = (
                        "hv"
                        if (step_for_binary and binary_map.get(y1name, False))
                        else "linear"
                    )
                    if line_layout.startswith("stack") and y2:
                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.07,
                            subplot_titles=(y1name, y2),
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_dt,
                                y=ys_used[y1name],
                                mode="lines+markers",
                                name=y1name,
                                line_shape=shape1,
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=x_dt,
                                y=ys_used[y1name],
                                mode="lines+markers",
                                name=y1name,
                                yaxis="y1",
                                line_shape=shape1,
                            )
                        )
                elif style == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=x_dt,
                            y=ys_used[y1name],
                            mode="markers",
                            name=y1name,
                            yaxis="y1",
                        )
                    )
                    if scatter_trend:
                        pts = [
                            (xs_used[i], float(ys_used[y1name][i]))
                            for i in range(min(len(xs_used), len(ys_used[y1name])))
                            if ys_used[y1name][i] is not None
                        ]
                        if len(pts) >= 2:
                            xs_, ys_ = zip(*pts)
                            n = len(xs_)
                            sx = sum(xs_)
                            sy = sum(ys_)
                            sxx = sum(x * x for x in xs_)
                            sxy = sum(x * y for x, y in pts)
                            denom = n * sxx - sx * sx
                            if denom != 0:
                                a = (n * sxy - sx * sy) / denom
                                b = (sy - a * sx) / n
                                yhat = [a * x + b for x in xs_used]
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_dt,
                                        y=yhat,
                                        mode="lines",
                                        name=f"{y1name} trend",
                                        yaxis="y1",
                                        line={"dash": "dash", "color": "#555"},
                                        showlegend=True,
                                    )
                                )
                elif style == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=x_dt,
                            y=ys_used[y1name],
                            name=y1name,
                            yaxis="y1",
                            width=width_ms if width_ms > 0 else None,
                            opacity=0.75,
                        )
                    )
                if len(series2) >= 2 and series2[1]:
                    y2name = series2[1]
                    if style == "line":
                        shape2 = (
                            "hv"
                            if (step_for_binary and binary_map.get(y2name, False))
                            else "linear"
                        )
                        if line_layout.startswith("stack") and y2:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_dt,
                                    y=ys_used[y2name],
                                    mode="lines+markers",
                                    name=y2name,
                                    line_shape=shape2,
                                ),
                                row=2,
                                col=1,
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_dt,
                                    y=ys_used[y2name],
                                    mode="lines+markers",
                                    name=y2name,
                                    yaxis="y2",
                                    line_shape=shape2,
                                )
                            )
                    elif style == "scatter":
                        fig.add_trace(
                            go.Scatter(
                                x=x_dt,
                                y=ys_used[y2name],
                                mode="markers",
                                name=y2name,
                                yaxis="y2",
                            )
                        )
                    elif style == "bar":
                        fig.add_trace(
                            go.Bar(
                                x=x_dt,
                                y=ys_used[y2name],
                                name=y2name,
                                yaxis=(
                                    "y2" if bar_mode.startswith("overlay") else "y1"
                                ),
                                width=width_ms if width_ms > 0 else None,
                                opacity=0.55,
                            )
                        )
                    if style == "bar" and not bar_mode.startswith("overlay"):
                        fig.update_layout(
                            yaxis2={
                                "title": y2name,
                                "overlaying": "y",
                                "side": "right",
                                "matches": "y",
                                "automargin": True,
                                "showgrid": False,
                                "zeroline": False,
                            }
                        )
                    else:
                        fig.update_layout(
                            yaxis2={
                                "title": y2name,
                                "overlaying": "y",
                                "side": "right",
                            }
                        )
                fig.update_layout(
                    title=f"{title} — {table}",
                    xaxis={
                        "title": "Time",
                        "showgrid": True,
                        "zeroline": False,
                        "automargin": True,
                    },
                    yaxis={
                        "title": series2[0],
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
                if style == "bar":
                    fig.update_layout(
                        barmode=(
                            "overlay" if bar_mode.startswith("overlay") else "group"
                        )
                    )
                return fig

            png_path = maybe_export_png(html_path, fig_factory_again)
            if png_path:
                try:
                    cgood(f"PNG written:  {png_path.resolve()}")
                except Exception:
                    cgood(f"PNG written:  {png_path}")
            # Loop again or exit while handled at top

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
