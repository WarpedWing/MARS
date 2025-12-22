#!/usr/bin/env python3

"""
Database interaction functions for the plotter.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from mars.pipeline.matcher.rubric_utils import TimestampFormat
from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from mars.plotter.selection_buffer import SeriesSelection


def check_database_plottable(db_path: Path) -> dict[str, Any]:
    """
    Quick check if a database has any plottable data.

    Returns dict with:
        - plottable: bool - True if database has data that can be plotted
        - tables_with_data: int - Count of tables with timestamp + numeric columns
        - total_rows: int - Approximate row count across plottable tables
        - reason: str - Why not plottable (if plottable=False)
    """
    result = {
        "plottable": False,
        "tables_with_data": 0,
        "total_rows": 0,
        "reason": "",
    }

    try:
        rubric = load_rubric_for_database(db_path)

        with readonly_connection(db_path) as conn:
            tables = detect_tables_with_timestamp(conn, rubric)

            if not tables:
                result["reason"] = "No timestamp columns"
                return result

            # Filter empty tables and check for numeric columns
            plottable_count = 0
            total_rows = 0

            for tname, col_pairs, timestamp_cols, row_count in tables:
                if row_count == 0:
                    continue

                # Check for numeric columns
                cols = list_graphable_columns(col_pairs)
                cols = filter_columns_with_non_null(conn, tname, cols)

                if cols:
                    plottable_count += 1
                    total_rows += row_count

            if plottable_count == 0:
                result["reason"] = "No numeric columns"
                return result

            result["plottable"] = True
            result["tables_with_data"] = plottable_count
            result["total_rows"] = total_rows

    except sqlite3.Error as e:
        result["reason"] = f"DB error: {e}"
    except Exception as e:
        result["reason"] = str(e)

    return result


def load_rubric_for_database(db_path: Path) -> dict | None:
    """
    Try to find and load rubric for a database.

    Looks for rubric in:
    1. Same directory as database: {db_stem}.rubric.json
    2. Parent's schemas directory: ../schemas/{db_stem}/{db_stem}.rubric.json
    3. Exemplar schemas: {db_dir}/../../schemas/{db_stem}/...
    """
    if not db_path:
        return None

    stem = db_path.stem.removesuffix(".combined")
    self_schema_dir = db_path.parent.parent.parent / "schemas"
    candidate_to_exemplar_schema_dir = (
        db_path.parent.parent.parent.parent.parent.parent / "exemplar" / "databases" / "schemas"
    )

    candidates = []

    # Search paths for metamatch (for candidates) or exemplar (for exemplars) rubrics
    if self_schema_dir.exists():
        candidates.extend(
            [
                self_schema_dir / stem / f"{stem}.rubric.json",
                self_schema_dir / stem / f"{stem}_combined.rubric.json",
            ]
        )
    # Try exemplar schemas directory from candidates
    if candidate_to_exemplar_schema_dir.exists():
        candidates.extend(
            [
                candidate_to_exemplar_schema_dir / stem / f"{stem}.rubric.json",
                candidate_to_exemplar_schema_dir / stem / f"{stem}_combined.rubric.json",
            ]
        )

    for rubric_path in candidates:
        if rubric_path.exists():
            try:
                with rubric_path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

    return None


def get_timestamp_column_from_rubric(rubric: dict | None, table_name: str) -> tuple[str | None, str | None]:
    """
    Get first timestamp column and format from rubric for a table.

    Returns:
        (column_name, format) tuple, or (None, None) if not found
    """
    all_ts = get_all_timestamp_columns_from_rubric(rubric, table_name)
    if all_ts:
        return all_ts[0]
    return None, None


def get_all_timestamp_columns_from_rubric(rubric: dict | None, table_name: str) -> list[tuple[str, str | None]]:
    """
    Get ALL timestamp columns and formats from rubric for a table.

    Returns:
        List of (column_name, format) tuples, empty if none found
    """
    if not rubric:
        return []

    tables = rubric.get("tables", {})
    table_meta = tables.get(table_name, {})
    columns = table_meta.get("columns", {})

    # Handle columns as either dict or list format
    if isinstance(columns, dict):
        col_items = list(columns.items())
    elif isinstance(columns, list):
        col_items = [(c.get("name"), c) for c in columns if isinstance(c, dict)]
    else:
        return []

    timestamp_cols: list[tuple[str, str | None]] = []
    for col_name, col_meta in col_items:
        if not col_name or not isinstance(col_meta, dict):
            continue
        role = col_meta.get("role")
        # Handle both string and list roles
        roles = [role] if isinstance(role, str) else (role if isinstance(role, list) else [])

        if "timestamp" in roles:
            ts_format = col_meta.get("format")
            timestamp_cols.append((col_name, ts_format))

    return timestamp_cols


def detect_tables_with_timestamp(
    conn: sqlite3.Connection,
    rubric: dict | None = None,
) -> list[tuple[str, list[tuple[str, str]], list[tuple[str, str | None]], int]]:
    """
    Return list of (table_name, columns, timestamp_cols, row_count) where timestamp exists.

    Uses rubric if provided for semantic timestamp detection, otherwise falls back
    to column name matching.

    Returns:
        List of tuples: (table_name, col_pairs, timestamp_cols, row_count)
        where timestamp_cols is a list of (column_name, format) tuples
    """
    cur = conn.cursor()
    out = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

    for (tname,) in cur.fetchall():
        # Skip internal/FTS tables
        if tname.startswith("sqlite_") or "_fts" in tname.lower():
            continue

        # Get columns
        qtname = tname.replace('"', '""')
        cols = cur.execute(f'PRAGMA table_info("{qtname}");').fetchall()
        if not cols:
            continue

        col_pairs = [(r[1], (r[2] or "").upper()) for r in cols]

        # Get row count
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{qtname}";')
            row_count = cur.fetchone()[0]
        except sqlite3.DatabaseError:
            row_count = 0

        # Try rubric first for all timestamp columns
        timestamp_cols = get_all_timestamp_columns_from_rubric(rubric, tname)

        # Fall back to column name matching if no rubric match
        if not timestamp_cols:
            for col_name, _ in col_pairs:
                if "timestamp" in (col_name or "").lower():
                    timestamp_cols.append((col_name, None))  # Unknown format

        # Skip tables without any timestamp column
        if not timestamp_cols:
            continue

        out.append((tname, col_pairs, timestamp_cols, row_count))

    # Sort by row count descending (most data first)
    out.sort(key=lambda x: x[3], reverse=True)
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


def list_label_columns(col_pairs: list[tuple[str, str]]) -> list[str]:
    """
    Return TEXT/VARCHAR column names suitable for hover labels.
    Excludes obvious ID columns and very short columns that are likely codes.
    """
    out = []
    for name, ctype in col_pairs:
        nlow = (name or "").lower()
        # Skip ID columns and timestamp columns
        if nlow in ("id", "rowid", "timestamp", "uuid"):
            continue
        t = (ctype or "").upper()
        # Include TEXT, VARCHAR, CHAR types
        if "TEXT" in t or "VARCHAR" in t or "CHAR" in t:
            out.append(name)
    return out


def filter_columns_with_non_null(conn: sqlite3.Connection, table: str, cols: list[str]) -> list[str]:
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


def convert_timestamp_to_epoch(value: Any, ts_format: str | None) -> float | None:
    """
    Convert a timestamp value to Unix epoch seconds using format hint.

    Args:
        value: Raw timestamp value from database
        ts_format: Format hint from rubric (e.g., "mac_absolute_time", "unix_seconds")

    Returns:
        Unix epoch seconds, or None if conversion fails
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None

    # If no format hint provided, try to auto-detect
    effective_format = ts_format
    if effective_format is None:
        effective_format = TimestampFormat.detect_timestamp_format(val)
        # If still None after auto-detect, the value isn't a valid timestamp
        if effective_format is None:
            return None

    unix_seconds = TimestampFormat._to_unix_seconds(val, effective_format)
    return unix_seconds


def fetch_series(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    y_cols: list[str],
    epoch_window: tuple[float | None, float | None] | None = None,
    ts_format: str | None = None,
    label_col: str | None = None,
) -> tuple[list[float], dict[str, list[float | None]], list[str | None]]:
    """
    Fetch X (epoch timestamps), Y series, and optional labels.
    Applies optional [start,end] epoch window filtering at SQL level.

    Args:
        conn: SQLite connection
        table: Table name
        ts_col: Timestamp column name
        y_cols: List of Y-axis column names
        epoch_window: Optional (start, end) epoch filter
        ts_format: Timestamp format hint for conversion (from rubric)
        label_col: Optional TEXT column for hover labels

    Returns:
        Tuple of (xs, ys, labels) where labels is empty list if no label_col
    """
    cur = conn.cursor()
    qtable = table.replace('"', '""')
    q_ts = ts_col.replace('"', '""')
    select_cols = [q_ts] + [c.replace('"', '""') for c in y_cols]
    if label_col:
        select_cols.append(label_col.replace('"', '""'))
    sel = ", ".join(f'"{c}"' for c in select_cols)

    # Note: epoch_window filtering happens after conversion, not at SQL level
    # when we have a non-trivial timestamp format
    sql = f'SELECT {sel} FROM "{qtable}" ORDER BY "{q_ts}" ASC'

    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except sqlite3.DatabaseError as e:
        logger.info(f"Query failed: {e}")
        return [], {c: [] for c in y_cols}, []

    xs: list[float] = []
    ys: dict[str, list[float | None]] = {c: [] for c in y_cols}
    labels: list[str | None] = []

    for r in rows:
        # r[0] = timestamp
        ts_raw = r[0]
        if ts_raw is None:
            continue

        # Convert timestamp using format hint
        epoch_ts = convert_timestamp_to_epoch(ts_raw, ts_format)
        if epoch_ts is None:
            continue

        # Apply epoch window filter (after conversion)
        if epoch_window:
            if epoch_window[0] is not None and epoch_ts < epoch_window[0]:
                continue
            if epoch_window[1] is not None and epoch_ts > epoch_window[1]:
                continue

        xs.append(epoch_ts)
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

        # Extract label if provided
        if label_col:
            label_idx = len(y_cols) + 1
            label_val = r[label_idx] if label_idx < len(r) else None
            labels.append(str(label_val) if label_val is not None else None)

    return xs, ys, labels


def build_union_dataset(
    conn: sqlite3.Connection,
    selections: list[tuple[str, str]],
    ts_col_map: dict[str, str],
    epoch_window: tuple[float | None, float | None] | None,
    ts_format_map: dict[str, str | None] | None = None,
) -> tuple[list[float], dict[str, list[float | None]]]:
    """Fetch each (table, column) and align by global timestamps."""
    series_data: list[tuple[str, list[float], list[float | None]]] = []
    all_ts: set[float] = set()
    ts_format_map = ts_format_map or {}

    for table, column in selections:
        ts_format = ts_format_map.get(table)
        xs, ys, _ = fetch_series(conn, table, ts_col_map[table], [column], epoch_window, ts_format)
        if not xs:
            continue
        label = f"{table} · {column}"
        values = ys.get(column, [])
        series_data.append((label, xs, values))
        all_ts.update(xs)
    union_x = sorted(all_ts)
    if not union_x:
        return [], {}
    union_series: dict[str, list[float | None]] = {}
    for label, xs, values in series_data:
        lookup = dict(zip(xs, values))
        union_series[label] = [lookup.get(x) for x in union_x]
    return union_x, union_series


def fetch_series_from_selection(
    selection: SeriesSelection,
    epoch_window: tuple[float | None, float | None] | None = None,
) -> tuple[list[float], list[float | None], list[str | None]]:
    """
    Fetch data for a single SeriesSelection, opening its own connection.

    Args:
        selection: SeriesSelection with database path, table, column, and timestamp info
        epoch_window: Optional (start, end) epoch filter

    Returns:
        Tuple of (xs, ys, labels) for the single column
    """
    with readonly_connection(selection.db_path) as conn:
        xs, ys_dict, labels = fetch_series(
            conn,
            selection.table_name,
            selection.ts_col,
            [selection.column_name],
            epoch_window,
            selection.ts_format,
            selection.label_col,
        )
        ys = ys_dict.get(selection.column_name, [])
        return xs, ys, labels


def build_multi_db_dataset(
    selections: list[SeriesSelection],
    epoch_window: tuple[float | None, float | None] | None = None,
) -> tuple[list[float], dict[str, list[float | None]], dict[str, list[str | None]]]:
    """
    Fetch data from multiple databases and align by global timestamps.

    Args:
        selections: List of SeriesSelection objects (can span multiple databases)
        epoch_window: Optional (start, end) epoch filter applied to all series

    Returns:
        - union_xs: Sorted list of all unique epoch timestamps
        - series_dict: {short_name: aligned_ys} for each selection
        - labels_dict: {short_name: aligned_labels} for each selection
    """
    series_data: list[tuple[str, list[float], list[float | None], list[str | None]]] = []
    all_ts: set[float] = set()

    for sel in selections:
        xs, ys, labels = fetch_series_from_selection(sel, epoch_window)
        if not xs:
            continue
        label = sel.short_name()
        series_data.append((label, xs, ys, labels))
        all_ts.update(xs)

    union_x = sorted(all_ts)
    if not union_x:
        return [], {}, {}

    union_series: dict[str, list[float | None]] = {}
    union_labels: dict[str, list[str | None]] = {}

    for label, xs, values, lbls in series_data:
        lookup_vals = dict(zip(xs, values))
        union_series[label] = [lookup_vals.get(x) for x in union_x]

        if lbls:
            lookup_lbls = dict(zip(xs, lbls))
            union_labels[label] = [lookup_lbls.get(x) for x in union_x]
        else:
            union_labels[label] = []

    return union_x, union_series, union_labels
