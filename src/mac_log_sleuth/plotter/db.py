#!/usr/bin/env python3

"""
Database interaction functions for the plotter.
"""

import sqlite3
from typing import Any


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
        print(f"Query failed: {e}")
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
        return [], {}
    union_series: dict[str, list[float | None]] = {}
    for label, xs, values in series_data:
        lookup = dict(zip(xs, values))
        union_series[label] = [lookup.get(x) for x in union_x]
    return union_x, union_series
