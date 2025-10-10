#!/usr/bin/env python3

"""
REPL-style Powerlog grapher (Plotly).

Inputs (CLI):
  - powerlog sqlite file (required)
  - optional: --rubric JSON (columns roles & hints)
  - optional: --schema CSV (Table,Column,Type)
  - optional: --outdir (defaults: ./Graphs)

Interactive flow:
  1) Show tables that (a) exist, (b) have ≥1 row, and (c) have a main 'timestamp' column
     (detected from rubric, schema, or name-contains 'timestamp').
  2) User selects a table.
  3) Show graphable numeric columns (INTEGER/REAL) that are non-null in data window.
  4) User selects 1 or 2 columns.
  5) Ask for "all data" or a time window (smart parsing: epoch, YYYY-MM-DD, YYYY-MM-DD HH:MM[:SS]).
  6) Ask chart type: line / scatter / bar.
  7) Ask timestamp presentation: epoch / UTC / local / offset (e.g., +02:00).
  8) Output HTML (and optionally PNG if Kaleido installed).

Requirements:
  - plotly (pip install plotly)
  - optional: kaleido (pip install -U kaleido) for image export
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- CLI parsing ----------


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive Powerlog grapher (Plotly).")
    ap.add_argument("db", help="Path to Powerlog SQLite DB (.PLSQL/.sqlite)")
    ap.add_argument("-o", "--outdir", default="Graphs", help="Output directory (default: ./Graphs)")
    ap.add_argument("--rubric", help="Optional rubric JSON")
    ap.add_argument("--schema", help="Optional schema CSV (Table,Column,Type)")
    return ap.parse_args()


# ---------- IO helpers ----------


def load_rubric(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f" Rubric not found: {p}")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to parse rubric JSON: {e}")
        return None


def load_schema(path: str | None) -> dict[str, list[tuple[str, str]]]:
    """
    Returns {table_name: [(column, type), ...]} preserving order.
    """
    result: dict[str, list[tuple[str, str]]] = {}
    if not path:
        return result
    p = Path(path)
    if not p.exists():
        print(f" Schema CSV not found: {p}")
        return result
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            # Expect either a header or raw rows; be forgiving
            for row in rdr if header else csv.reader(p.open("r", encoding="utf-8")):
                if not row:
                    continue
                # defensive parse
                if len(row) >= 3:
                    t, c, ty = row[0].strip(), row[1].strip(), (row[2] or "TEXT").strip()
                    if t and c:
                        result.setdefault(t, []).append((c, ty))
    except Exception as e:
        print(f"Failed to parse schema CSV: {e}")
    return result


# ---------- DB helpers ----------


def pragma_table_info(cur: sqlite3.Cursor, table: str) -> list[tuple[int, str, str, int, Any, int]]:
    """PRAGMA table_info(table) with safe quoting."""
    q = '"' + table.replace('"', '""') + '"'
    try:
        return cur.execute(f"PRAGMA table_info({q});").fetchall()
    except sqlite3.Error:
        return []


def table_has_rows(cur: sqlite3.Cursor, table: str) -> bool:
    q = '"' + table.replace('"', '""') + '"'
    try:
        r = cur.execute(f"SELECT 1 FROM {q} LIMIT 1;").fetchone()
        return r is not None
    except sqlite3.Error:
        return False


def list_all_tables(cur: sqlite3.Cursor) -> list[str]:
    try:
        rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error:
        return []


# ---------- Timestamp helpers ----------


def detect_timestamp_columns(table: str, columns: list[tuple[str, str]], rubric: dict | None) -> list[str]:
    """
    Return column names that are plausible primary time axes.
    Priority:
      1) rubric role == epoch_timestamp
      2) name contains 'timestamp' (case-insensitive)
    """
    names = [c for c, _ in columns]
    ts_from_rubric: list[str] = []
    if rubric:
        t = rubric.get("tables", {}).get(table, {})
        colmap = t.get("columns", {})
        for cname in names:
            meta = colmap.get(cname, {})
            if str(meta.get("role", "")).lower() == "epoch_timestamp":
                ts_from_rubric.append(cname)
    if ts_from_rubric:
        return ts_from_rubric

    # fallback by name
    return [c for c in names if "timestamp" in c.lower()]


def detect_numeric_columns(table: str, columns: list[tuple[str, str]], rubric: dict | None) -> list[str]:
    """
    Columns that are graphable: INTEGER or REAL.
    Use schema type first; if missing, use rubric 'type'; else heuristic by name.
    """
    numeric_cols: list[str] = []
    for cname, ctype in columns:
        t = (ctype or "").upper()
        is_numeric = t.startswith("INT") or t.startswith("REAL")
        if not ctype and rubric:
            meta = rubric.get("tables", {}).get(table, {}).get("columns", {}).get(cname, {})
            rt = str(meta.get("type", "")).upper()
            is_numeric = rt.startswith("INT") or rt.startswith("REAL")
        if is_numeric:
            numeric_cols.append(cname)
    return numeric_cols


def parse_human_datetime(s: str) -> float | None:
    """
    Accept:
      - epoch seconds (int/float string)
      - YYYY-MM-DD
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - with 'Z' suffix → UTC
    Returns epoch seconds (float) in UTC.
    """
    s = (s or "").strip()
    if not s:
        return None
    # epoch?
    try:
        return float(s)
    except ValueError:
        pass

    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    # handle trailing Z
    if s.endswith("Z"):
        core = s[:-1].strip()
        for f in fmts:
            try:
                dt = datetime.strptime(core, f).replace(tzinfo=UTC)
                return dt.timestamp()
            except ValueError:
                continue
        return None

    # naive → interpret as local time, convert to UTC epoch
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            # assume local time
            local_tz = datetime.now().astimezone().tzinfo
            dt_local = dt.replace(tzinfo=local_tz)
            return dt_local.astimezone(UTC).timestamp()
        except ValueError:
            continue
    return None


def present_time_axis(ts: float, mode: str, offset_minutes: int = 0) -> str:
    """
    Format a single epoch to a label according to mode.
    """
    if mode == "epoch":
        return f"{ts:.6f}"
    if mode == "utc":
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    if mode == "local":
        return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    if mode == "offset":
        tz = timezone(timedelta(minutes=offset_minutes))
        sign = "+" if offset_minutes >= 0 else "-"
        hh = abs(offset_minutes) // 60
        mm = abs(offset_minutes) % 60
        return datetime.fromtimestamp(ts, tz=tz).strftime(f"%Y-%m-%d %H:%M:%S GMT{sign}{hh:02d}:{mm:02d}")
    return f"{ts:.6f}"


# ---------- REPL prompts ----------


def prompt_choice(title: str, options: list[str], allow_quit: bool = True) -> int | None:
    print("\n" + title)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_quit:
        print("  q. quit")
    while True:
        s = input("Select: ").strip().lower()
        if allow_quit and s in ("q", "quit", "exit"):
            return None
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                return idx - 1
        print("  ✗ Invalid choice. Try again.")


def prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    suf = "[Y/n]" if default_yes else "[y/N]"
    while True:
        s = input(f"{msg} {suf}: ").strip().lower()
        if not s:
            return default_yes
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("  ✗ Please answer y/n.")


def prompt_time_window() -> tuple[float, float] | None:
    print("\nTime window:")
    if not prompt_yes_no("Plot ALL available timestamps?", default_yes=True):
        print("Enter start and end (epoch seconds OR 'YYYY-MM-DD[ HH:MM[:SS]]' — add 'Z' for UTC).")
        while True:
            s1 = input("  Start: ").strip()
            s2 = input("  End  : ").strip()
            t1 = parse_human_datetime(s1)
            t2 = parse_human_datetime(s2)
            if t1 is None or t2 is None or t2 < t1:
                print("  ✗ Could not parse, or end < start. Try again.")
                continue
            return (t1, t2)
    return None


def prompt_chart_type() -> str:
    opts = ["line", "scatter", "bar"]
    idx = prompt_choice("Chart type:", opts, allow_quit=False)
    return opts[idx or 0]


def prompt_time_display() -> tuple[str, int]:
    """
    Returns (mode, offset_minutes). mode ∈ {epoch, utc, local, offset}
    """
    print("\nTimestamp display format:")
    opts = ["Epoch (raw seconds)", "UTC (Z time)", "Local timezone", "Specific offset (e.g., +02:00)"]
    idx = prompt_choice("Pick one:", opts, allow_quit=False)
    if idx == 0:
        return ("epoch", 0)
    if idx == 1:
        return ("utc", 0)
    if idx == 2:
        return ("local", 0)
    # offset
    while True:
        s = input("Enter offset like +02:00 or -07:00: ").strip()
        try:
            sign = 1 if s.startswith("+") else -1
            hh, mm = s[1:].split(":")
            minutes = sign * (int(hh) * 60 + int(mm))
            return ("offset", minutes)
        except Exception:
            print("  ✗ Bad offset. Try again.")


# ---------- Plotting ----------


def build_plot(
    table: str,
    ts_name: str,
    series1_name: str,
    series2_name: str | None,
    rows: list[tuple[float, float, float | None]],
    chart_type: str,
    ts_mode: str,
    ts_offset: int,
    out_html: Path,
    out_png: Path | None = None,
):
    """
    rows: list of (ts, y1, y2?) with ts in epoch seconds.
    """
    # Prepare X labels depending on ts_mode; keep a numeric X for sorting/stability
    xs_epoch = [r[0] for r in rows]
    y1 = [r[1] for r in rows]
    y2 = [r[2] if len(r) > 2 else None for r in rows]

    # Construct figure with secondary y if needed
    secondary = series2_name is not None
    fig = make_subplots(specs=[[{"secondary_y": secondary}]])

    def add_trace(name: str, ys: list[float | None], secondary_y: bool):
        if chart_type == "line":
            fig.add_trace(go.Scatter(x=xs_epoch, y=ys, mode="lines+markers", name=name), secondary_y=secondary_y)
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(x=xs_epoch, y=ys, mode="markers", name=name), secondary_y=secondary_y)
        else:  # bar
            fig.add_trace(go.Bar(x=xs_epoch, y=ys, name=name), secondary_y=secondary_y)

    add_trace(series1_name, y1, secondary_y=False)
    if secondary:
        add_trace(series2_name, y2, secondary_y=True)

    # Axes + title + caption
    # X-axis tick formatting: show human-readable via hover and tick text (without actually transforming data)
    # We'll set a custom hovertemplate that prints both epoch and formatted time.
    def fmt_label(ts: float) -> str:
        return present_time_axis(ts, ts_mode, ts_offset)

    hovertemplate = (
        f"<b>%{{fullData.name}}</b><br>{ts_name}: %{{x:.6f}}<br>time: %{{customdata}}<br>value: %{{y}}<extra></extra>"
    )
    custom_times = [[fmt_label(v)] for v in xs_epoch]
    for tr in fig.data:
        tr.update(customdata=custom_times, hovertemplate=hovertemplate)

    # Layout
    fig.update_layout(
        title=f"{table} — {series1_name}" + (f" vs {series2_name}" if secondary else ""),
        xaxis_title=f"{ts_name} (epoch seconds; formatted per selection)",
        yaxis_title=series1_name,
        legend_title="Series",
        template="plotly_white",
        margin={"l": 60, "r": 60, "t": 70, "b": 60},
        annotations=[
            {
                "text": (
                    "Forensic note: X-axis uses raw epoch seconds from the Powerlog table.<br>"
                    "Displayed timestamps reflect the chosen presentation (epoch/UTC/local/offset).<br>"
                    "Values plotted without interpolation or smoothing."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "showarrow": False,
                "align": "left",
                "font": {"size": 11},
            }
        ],
        width=1100,
        height=650,
    )
    if secondary:
        fig.update_yaxes(title_text=series2_name, secondary_y=True)

    # Save
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Saved HTML → {out_html}")

    if out_png:
        try:
            import kaleido  # noqa: F401

            fig.write_image(str(out_png), scale=2)
            print(f" Saved image → {out_png}")
        except Exception as e:
            print(f" PNG export skipped (install kaleido): {e}")


# ---------- Main flow ----------


def main():
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"✗ DB not found: {db_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rubric = load_rubric(args.rubric)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    # Discover candidate tables
    tables = list_all_tables(cur)
    candidates: list[tuple[str, str, list[str], list[str]]] = []
    # (table, ts_col, numeric_cols, all_cols)

    for t in tables:
        # Skip internal sqlite tables
        if t.lower().startswith("sqlite_"):
            continue
        if not table_has_rows(cur, t):
            continue
        cols_info = pragma_table_info(cur, t)
        if not cols_info:
            continue
        cols = [(r[1], r[2] or "") for r in cols_info]  # (name, type)
        ts_cols = detect_timestamp_columns(t, cols, rubric)
        if not ts_cols:
            continue
        # pick the first timestamp col that actually has data
        picked_ts = None
        for ts_c in ts_cols:
            q_t = '"' + t.replace('"', '""') + '"'
            q_c = '"' + ts_c.replace('"', '""') + '"'
            try:
                row = cur.execute(f"SELECT {q_c} FROM {q_t} WHERE {q_c} IS NOT NULL LIMIT 1;").fetchone()
                if row is not None:
                    picked_ts = ts_c
                    break
            except sqlite3.Error:
                continue
        if not picked_ts:
            continue

        numeric_cols = detect_numeric_columns(t, cols, rubric)
        # ensure at least one numeric column with some data
        usable_numeric = []
        for c in numeric_cols:
            q_t = '"' + t.replace('"', '""') + '"'
            q_c = '"' + c.replace('"', '""') + '"'
            try:
                row = cur.execute(f"SELECT {q_c} FROM {q_t} WHERE {q_c} IS NOT NULL LIMIT 1;").fetchone()
                if row is not None:
                    usable_numeric.append(c)
            except sqlite3.Error:
                pass
        if usable_numeric:
            candidates.append((t, picked_ts, usable_numeric, [n for n, _ in cols]))

    if not candidates:
        print("✗ No tables with timestamp + numeric columns found.")
        sys.exit(2)

    # 1) Pick table
    opts = [f"{t}   (timestamp: {ts}, numeric: {', '.join(nums)})" for t, ts, nums, _ in candidates]
    idx = prompt_choice("Available tables:", opts)
    if idx is None:
        print("bye.")
        sys.exit(0)
    table, ts_col, num_cols, all_cols = candidates[idx]

    # 2) Pick first column
    idx1 = prompt_choice("Pick primary value column:", num_cols)
    if idx1 is None:
        print("bye.")
        sys.exit(0)
    col1 = num_cols[idx1]

    # 3) Optionally pick second column
    col2 = None
    remaining = [c for c in num_cols if c != col1]
    if remaining and prompt_yes_no("Add a second column on right Y-axis?"):
        idx2 = prompt_choice("Pick secondary column:", remaining)
        if idx2 is None:
            print("bye.")
            sys.exit(0)
        col2 = remaining[idx2]

    # 4) Time window
    window = prompt_time_window()

    # 5) Chart type
    chart_type = prompt_chart_type()

    # 6) Timestamp presentation
    ts_mode, ts_offset = prompt_time_display()

    # 7) Query data
    q_t = '"' + table.replace('"', '""') + '"'
    q_ts = '"' + ts_col.replace('"', '""') + '"'
    q_cols = [q_ts, '"' + col1.replace('"', '""') + '"']
    if col2:
        q_cols.append('"' + col2.replace('"', '""') + '"')
    select_cols_sql = ", ".join(q_cols)

    where = ""
    params: list[Any] = []
    if window:
        where = f"WHERE {q_ts} BETWEEN ? AND ?"
        params = [window[0], window[1]]

    sql = f"SELECT {select_cols_sql} FROM {q_t} {where} ORDER BY {q_ts};"

    try:
        rows = cur.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        print(f"✗ Query failed: {e}")
        sys.exit(3)

    # Filter out rows with null timestamp or null primary value
    cleaned = []
    for r in rows:
        if r[0] is None or r[1] is None:
            continue
        # coerce to float for ts & y
        try:
            ts_f = float(r[0])
            y1 = float(r[1])
            y2 = float(r[2]) if (len(r) > 2 and r[2] is not None) else None
        except Exception:
            continue
        cleaned.append((ts_f, y1, y2))

    if not cleaned:
        print("✗ No rows matched your selections.")
        sys.exit(4)

    # 8) Output filenames
    base = f"{db_path.stem}__{table}__{col1}" + (f"__{col2}" if col2 else "")
    html_path = outdir / f"{base}.html"
    png_path = None
    if prompt_yes_no("Also save a PNG image (requires kaleido)?", default_yes=False):
        png_path = outdir / f"{base}.png"

    # 9) Build & save plot
    build_plot(
        table=table,
        ts_name=ts_col,
        series1_name=col1,
        series2_name=col2,
        rows=cleaned,
        chart_type=chart_type,
        ts_mode=ts_mode,
        ts_offset=ts_offset,
        out_html=html_path,
        out_png=png_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
