#!/usr/bin/env python3
"""
Salvages recovered Powerlog databases by fingerprinting rows in the 'lost_and_found'
table and mapping them to known table structures using a schema and rubric.

This script operates as follows:
- Prompts for a profile, which consists of a schema and a rubric file.
- Iterates through each row of the 'lost_and_found' table to find the first
  plausible epoch timestamp.
- From that timestamp, it builds a typed "fingerprint" of the remaining cells
  in the row and attempts to match it against a table in the schema.
- Enforces strict type checking during matching:
    * TEXT: String-like, not purely numeric.
    * INTEGER: Integer values (no decimals).
    * REAL: Floating-point numbers (or promoted integers).
    * 'timestamp' columns: Must be a valid, non-null epoch timestamp within
      the bounds defined in the rubric.
    * 'date' columns: Can be 0.0 or a valid, non-null epoch timestamp.
- Extracts a row ID, defaulting to the 4th column. If null, it uses the
  first non-null integer in the row as a fallback.
- Ensures that each row from 'lost_and_found' is used at most once.

The script generates two output files in the '<db_source>/Salvaged' directory:
- A salvaged SQLite database: <basename>.<unix_timestamp>.PLSQL.sqlite
- A mapping CSV file: <basename>.<unix_timestamp>.PLSQL.sqlite.mapping.csv
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

SCHEMAS_DIR = Path("Schemas")
RUBRICS_DIR = Path("Rubrics")
SALVAGED_DIR = Path("Salvaged")

# lost_and_found layout heuristics
LNF_TABLE = "lost_and_found"
LNF_ID_COL_INDEX = 3  # 4th column (0-indexed) is the default row ID.

# Epoch bounds (rubric may override)
DEFAULT_EPOCH_MIN = 1262304000.0  # 2010-01-01
DEFAULT_EPOCH_MAX = 4102444800.0  # 2100-01-01

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$")

FILL_RATIO_STRONG_THRESHOLD = 0.85  # columns with >=85% fill rate are treated as "dense"


# ---------- utils ----------
def is_null(v):
    return v is None


def safe_float(x: Any) -> float | None:
    """
    Convert to float safely; return None if impossible.

    Returns:
        Optional[float]: The converted float value if possible, otherwise None.

    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_numericish_text(s: str) -> bool:
    return bool(NUMERIC_RE.match(s.strip()))


def coerce_numeric_kind(v):
    """
    Return ('int'|'real'|'text'|'null', value_as_python)
    Numeric-looking strings get converted to int/float for type checks.
    """
    if v is None:
        return "null", None
    if isinstance(v, (int, bool)):  # bool behaves like int in sqlite
        return "int", int(v)
    if isinstance(v, float):
        # normalize -0.0 edge
        if v == 0.0:
            v = 0.0
        return "real", float(v)
    if isinstance(v, (bytes, bytearray)):
        # bytes are not acceptable for TEXT unless caller decodes;
        return "text", v  # treat as text blob
    if isinstance(v, str):
        s = v.strip()
        if is_numericish_text(s):
            # distinguish int-like vs real-like
            try:
                if "." in s or "e" in s.lower():
                    fv = float(s)
                    # count it as int if exactly integral (e.g. "100.0")
                    if fv.is_integer():
                        return "int", int(fv)
                    return "real", fv
                return "int", int(s)
            except (ValueError, TypeError):
                return "text", v
        return "text", v
    # fallback
    return "text", v


def is_epoch_timestamp_ok(v, *, allow_zero=False):
    kind, val = coerce_numeric_kind(v)
    if kind in ("int", "real"):
        try:
            fv = safe_float(val)
        except (ValueError, TypeError):
            return False
        if allow_zero and not is_null(fv) and fv == 0.0:
            return True  # fv is guaranteed not null here
        return DEFAULT_EPOCH_MIN <= fv <= DEFAULT_EPOCH_MAX
    return False


def value_matches_declared_type(v, decl_type: str, *, col_name: str) -> bool:
    """
    Strictly validates if a value matches its declared SQLite type.

    Rules:
    - TEXT: Must be a string that does not look like a number.
    - INTEGER: Accepts integers or floats that are whole numbers (e.g., 123.0).
    - REAL: Accepts integers or floats.
    - 'timestamp' columns: Must be a valid, non-null epoch value.
    - 'date' columns: Can be 0 or a valid epoch value.
    """
    # normalize affinity string
    t = (decl_type or "").strip().upper()
    # special roles inferred from name
    name_lower = (col_name or "").lower()
    is_ts = "timestamp" in name_lower
    is_date = (not is_ts) and ("date" in name_lower)

    # epoch rules override base type where relevant
    if is_ts:
        return is_epoch_timestamp_ok(v, allow_zero=False)
    if is_date:
        # can be 0 or epoch
        return is_epoch_timestamp_ok(v, allow_zero=True)

    k, vv = coerce_numeric_kind(v)
    if t == "TEXT" or t == "":
        # must be textual; numeric-looking values are not allowed
        return k == "text"
    if t == "INTEGER":
        if k == "int":
            return True
        if k == "real":
            try:
                return safe_float(vv).is_integer()
            except (ValueError, TypeError):
                return False
        return False

    if t == "REAL":
        return k in ("int", "real")
    # Unknown types: be conservative; require textual unless numeric is explicit
    return k == "text"


def is_int_like(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_float_like(x: Any) -> bool:
    return isinstance(x, float)


def is_text_like(x: Any) -> bool:
    if isinstance(x, str):
        # TEXT should not be purely numeric string
        stripped = x.strip()
        if stripped == "":
            return True
        # reject plain ints/floats like "123" or "12.3"
        return not re.fullmatch(r"[+-]?\d+(\.\d+)?", stripped)
    # bytes → treat as text
    return bool(isinstance(x, (bytes, bytearray)))


def is_epoch(val: Any, lo: float, hi: float) -> bool:
    if is_int_like(val) or is_float_like(val):
        v = float(val)
        return lo <= v <= hi
    return False


def is_date_val(val: Any, lo: float, hi: float) -> bool:
    # date can be 0.0 or a valid epoch (non-null)
    if val is None:
        return False
    if (is_int_like(val) or is_float_like(val)) and float(val) == 0.0:
        return True
    return is_epoch(val, lo, hi)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_profiles() -> list[str]:
    """Return list of available base names (present in both Schemas and Rubrics)."""
    bases = set()
    bases.update(p.name.split(".powerlog.schema.csv")[0] for p in SCHEMAS_DIR.glob("*.powerlog.schema.csv"))
    profiles = []
    for b in sorted(bases):
        r = RUBRICS_DIR / f"{b}.powerlog.rubric.json"
        if r.exists():
            profiles.append(b)
    return profiles


def pick_profile_interactive(profiles: list[str]) -> str:
    print("\nAvailable profiles:")
    for i, b in enumerate(profiles, 1):
        print(f"  {i}. {b}")
    while True:
        s = input("Select profile by number: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(profiles):
                return profiles[idx - 1]
        print("Invalid selection. Try again.")


def load_schema(profile: str) -> dict[str, list[tuple[str, str]]]:
    """
    Return {table_name: [(col_name, type), ...]} preserving order.
    """
    path = SCHEMAS_DIR / f"{profile}.powerlog.schema.csv"
    by_table: dict[str, list[tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 3:
                continue
            if row[0] == "Table" and row[1] == "Column":
                continue  # skip header row
            t, c, ty = row[0], row[1], row[2]
            by_table.setdefault(t, []).append((c, ty or "TEXT"))
    return by_table


def load_rubric(profile: str) -> dict:
    path = RUBRICS_DIR / f"{profile}.powerlog.rubric.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_expectations(
    schema_cols: list[tuple[str, str]],
    rubric_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    col_names = [c for c, _ in schema_cols]
    col_types = [t for _, t in schema_cols]
    roles = []
    ts_index = None
    rubric_cols = (rubric_table or {}).get("columns", {}) if rubric_table else {}
    col_meta: list[dict[str, Any]] = []
    required_idxs: set[int] = set()
    dense_idxs: set[int] = set()
    fill_expectations: list[float | None] = []
    for idx, (cname, ctype) in enumerate(schema_cols):
        lower = cname.lower()
        if cname == "ID":
            role = "id"
        elif "timestamp" in lower:
            role = "timestamp"
            if ts_index is None:
                ts_index = idx
        elif "date" in lower:
            role = "date"
        elif (ctype or "").upper().startswith("INT"):
            role = "integer"
        elif (ctype or "").upper().startswith("REAL"):
            role = "real"
        else:
            role = "text"
        roles.append(role)

        meta = rubric_cols.get(cname, {}) if rubric_cols else {}
        col_meta.append(meta)

        if meta.get("notnull"):
            required_idxs.add(idx)

        observed_ratio = meta.get("observed_fill_ratio")
        fill_expectations.append(observed_ratio if isinstance(observed_ratio, (int, float)) else None)
        if isinstance(observed_ratio, (int, float)) and observed_ratio >= FILL_RATIO_STRONG_THRESHOLD:
            dense_idxs.add(idx)

    return {
        "columns": col_names,
        "column_list": col_names,
        "types": col_types,
        "roles": roles,
        "ts_index": ts_index,
        "col_meta": col_meta,
        "required_idxs": required_idxs,
        "dense_idxs": dense_idxs,
        "fill_expectations": fill_expectations,
    }


def classify_cell(val: Any, epoch_lo: float, epoch_hi: float) -> str:
    """
    Loose type classifier used for fingerprinting (not storage).
    """
    if val is None:
        return "null"
    if is_int_like(val):
        return "integer"
    if is_float_like(val):
        # could be real OR timestamp/date — caller handles role-specific checks
        return "real"
    if is_text_like(val):
        return "text"
    return "unknown"


def row_first_timestamp_index(row: tuple[Any, ...], epoch_lo: float, epoch_hi: float) -> int | None:
    for i, v in enumerate(row):
        if is_epoch(v, epoch_lo, epoch_hi):
            return i
    return None


def _coerce_lf_id(value: Any) -> int | None:
    """
    Attempt to coerce a lost_and_found cell into an integer ID.
    Accepts integers or floats that are integral after coercion.
    """
    kind, coerced = coerce_numeric_kind(value)
    if kind == "int":
        try:
            return int(coerced)
        except (ValueError, TypeError):
            return None
    if kind == "real":
        sf = safe_float(coerced)
        if sf is not None and float(sf).is_integer():
            return int(float(sf))
    return None


def row_pick_id(row: tuple[Any, ...]) -> int | None:
    # Prefer the known 4th column; else any first convertible numeric
    try:
        v = row[LNF_ID_COL_INDEX]
        candidate = _coerce_lf_id(v)
        if candidate is not None:
            return candidate
    except IndexError:
        pass
    for v in row:
        candidate = _coerce_lf_id(v)
        if candidate is not None:
            return candidate
    return None


def score_segment_against_table_strict(row_vals, start_idx, table_def):
    """
    Try to match a row segment starting at start_idx (first timestamp) to the given table.
    table_def must contain:
      - "column_list": ordered col names
      - "types": matching sqlite decl types
    Returns (ok, details) where:
      ok = True only if the ENTIRE remainder of the row matches up to schema length,
           and any cells beyond schema length are ALL NULL.
      details = {
         "from_schema_index": <int>,   # which schema index (col) the start_idx aligns to (usually 1 for 'timestamp')
         "mapped_values": <list>,      # aligned values sized to len(schema)
      }
    """
    cols = table_def.get("column_list") or []
    types = table_def.get("types") or ["TEXT"] * len(cols)
    col_meta_list = table_def.get("col_meta") or []
    required_idxs = table_def.get("required_idxs") or set()
    n_schema = len(cols)
    if n_schema < 2:
        return False, None  # must at least have ID + timestamp or legit short table

    # Find the 'timestamp' column index in schema (first occurrence)
    ts_schema_idx = None
    for i, name in enumerate(cols):
        if "timestamp" in (name or "").lower():
            ts_schema_idx = i
            break

    # Some tables legitimately don't have timestamp columns.
    # We don't match those here; another pass handles them.
    if ts_schema_idx is None:
        return False, None

    # Align row cells to schema columns starting at ts_schema_idx
    # Example: if timestamp is schema index 1, then:
    #   schema[1] <= row[start_idx]
    #   schema[2] <= row[start_idx+1], ...
    #   schema[0] will be filled later (ID from another cell)
    # Check we have enough cells to fill through the end of the schema
    remaining = len(row_vals) - start_idx
    needed = n_schema - ts_schema_idx
    if remaining < needed:
        return False, None  # row cuts off before schema ends

    mapped = [None] * n_schema
    # fill timestamp..end using the row segment
    seg = row_vals[start_idx : start_idx + needed]

    # Type-check each mapped value
    for offset, val in enumerate(seg):
        schema_idx = ts_schema_idx + offset
        decl = types[schema_idx]
        col_name = cols[schema_idx]
        meta = col_meta_list[schema_idx] if schema_idx < len(col_meta_list) else {}
        if is_null(val):
            # Required columns (including timestamps) must not be null.
            if schema_idx in required_idxs or "timestamp" in (col_name or "").lower():
                return False, None
        else:
            if not value_matches_declared_type(val, decl, col_name=col_name):
                return False, None
            hints = meta.get("hints") if isinstance(meta, dict) else None
            if hints and "range" in hints:
                try:
                    lo, hi = hints["range"]
                    fv = float(val)
                except (ValueError, TypeError):
                    return False, None
                if lo is not None:
                    try:
                        lo_f = float(lo)
                    except (ValueError, TypeError):
                        return False, None
                    if fv < lo_f:
                        return False, None
                if hi is not None:
                    try:
                        hi_f = float(hi)
                    except (ValueError, TypeError):
                        return False, None
                    if fv > hi_f:
                        return False, None
        mapped[schema_idx] = val

    # Everything AFTER the schema's end must be ALL NULL, or we reject
    extra_tail = row_vals[start_idx + needed :]
    if any(not is_null(x) for x in extra_tail):
        return False, None

    # OK — caller is responsible for filling ID if present in schema
    return True, {"from_schema_index": ts_schema_idx, "mapped_values": mapped}


def compute_match_metrics(mapped: list[Any], table_def: dict[str, Any]) -> dict[str, float]:
    total_cols = len(mapped) or 1
    non_null = sum(1 for v in mapped if not is_null(v))
    non_null_ratio = non_null / total_cols

    required_idxs = table_def.get("required_idxs") or set()
    required_total = len(required_idxs)
    required_hits = sum(
        1
        for idx in required_idxs
        if idx < len(mapped) and not is_null(mapped[idx])
    )
    required_ratio = required_hits / required_total if required_total else 1.0

    dense_idxs = table_def.get("dense_idxs") or set()
    dense_total = len(dense_idxs)
    dense_hits = sum(
        1
        for idx in dense_idxs
        if idx < len(mapped) and not is_null(mapped[idx])
    )
    dense_ratio = dense_hits / dense_total if dense_total else non_null_ratio

    confidence = (0.6 * required_ratio) + (0.25 * dense_ratio) + (0.15 * non_null_ratio)

    return {
        "confidence": confidence,
        "required_ratio": required_ratio,
        "dense_ratio": dense_ratio,
        "non_null_ratio": non_null_ratio,
    }


def quick_match_id_text_real(
    row: tuple[Any, ...],
    table_exp: dict[str, Any],
    epoch_lo: float,
    epoch_hi: float,
) -> float:
    """
    For tables without a timestamp (e.g., PLCoreStorage_schemaVersions: ID, tableName, schemaVersion).
    We look for a contiguous triple INT, TEXT, REAL anywhere in the row.
    Score 1.0 for a clean hit; else 0.0.
    """
    roles = table_exp["roles"]
    cols = table_exp["columns"]

    # Only support classic [ID, TEXT, REAL] shape with ID first
    if len(cols) < 3 or roles[0] != "id" or roles[1] != "text" or roles[2] not in ("real", "integer"):
        return 0.0

    n = len(row)
    for i in range(n - 2):
        a, b, c = row[i], row[i + 1], row[i + 2]
        if is_int_like(a) and is_text_like(b) and (is_float_like(c) or is_int_like(c)):
            return 1.0
    return 0.0


def ensure_table(dst_cur: sqlite3.Cursor, table: str, schema_cols: list[tuple[str, str]]):
    if table.lower() == "sqlite_sequence":
        return  # internal table, skip creation
    cols_sql = ", ".join(f"{quote_ident(c)} {t or 'TEXT'}" for c, t in schema_cols)
    dst_cur.execute(f"CREATE TABLE IF NOT EXISTS {quote_ident(table)} ({cols_sql});")


# ---------- main salvage ----------


def salvage_recovered_sqlite(recovered_path: Path) -> Path:
    profiles = load_profiles()
    if not profiles:
        print(
            "Error: No profiles found. Put your files in ./Schemas and ./Rubrics",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"\nSalvaging: {recovered_path}")
    profile = pick_profile_interactive(profiles)
    print(f"Using profile: {profile}")

    schema_by_table = load_schema(profile)
    rubric_data = load_rubric(profile)
    rubric_tables = rubric_data.get("tables", {}) if isinstance(rubric_data, dict) else {}

    salvaged_dir = recovered_path.parent / SALVAGED_DIR
    salvaged_dir.mkdir(parents=True, exist_ok=True)
    base = recovered_path.stem
    out_name = f"{base}.{int(time.time())}.PLSQL.sqlite"
    out_db = salvaged_dir / out_name
    map_csv = salvaged_dir / f"{out_name}.mapping.csv"

    # open recovered source (read-only)
    src = sqlite3.connect(f"file:{recovered_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    cur = src.cursor()

    # lost_and_found rows
    try:
        # Grab all raw rows as tuples
        raw_rows = cur.execute(f"SELECT rowid, * FROM {quote_ident(LNF_TABLE)}").fetchall()
        lf_rowids: list[int] = []
        rows: list[tuple[Any, ...]] = []
        for rec in raw_rows:
            lf_rowids.append(int(rec[0]))
            rows.append(tuple(rec[1:]))
        if not rows:
            print("No data in lost_and_found — nothing to salvage.")
            return out_db
    except sqlite3.Error as e:
        print(f"Error: Could not read {LNF_TABLE}: {e}", file=sys.stderr)
        sys.exit(3)

    # destination DB
    dst = sqlite3.connect(str(out_db))
    dst_cur = dst.cursor()
    # pre-create all schema tables (empty tables are desired)
    for tname, cols in schema_by_table.items():
        if tname.lower() == "sqlite_sequence":
            continue
        ensure_table(dst_cur, tname, cols)
    dst.commit()

    # prepare expectations
    tables_with_timestamp = {
        tname: build_expectations(cols, rubric_tables.get(tname))
        for tname, cols in schema_by_table.items()
    }

    mappings = []  # for the mapping CSV: (lf_index, lf_rowid, lf_id_value, table, score)
    consumed_rows = set()  # mark L&F row indices that we've already used
    written_tables: set[str] = set()

    for r_idx, row in enumerate(rows):
        if r_idx in consumed_rows:
            continue

        # Find the first plausible timestamp in this row, scanning left to right
        ts_positions = []
        for c_idx, val in enumerate(row):
            if is_epoch_timestamp_ok(val, allow_zero=False):
                ts_positions.append(c_idx)

        if not ts_positions:
            continue  # nothing to do with this row

        lf_id_value = row_pick_id(row)

        for start_c in ts_positions:
            # Collect matching candidates with their confidence scores
            candidates = []
            for table_name, tbl in tables_with_timestamp.items():
                ok, details = score_segment_against_table_strict(row, start_c, tbl)
                if not ok:
                    continue

                cols = tbl.get("column_list") or []
                types = tbl.get("types") or ["TEXT"] * len(cols)
                mapped = details["mapped_values"][:]

                if cols and cols[0] == "ID":
                    if lf_id_value is None:
                        continue
                    mapped[0] = lf_id_value

                metrics = compute_match_metrics(mapped, tbl)
                candidates.append(
                    (
                        metrics["confidence"],
                        -start_c,
                        len(cols),
                        table_name,
                        tbl,
                        mapped,
                        types,
                        metrics,
                    )
                )

            if not candidates:
                continue

            best_candidate = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            (
                _confidence,
                _neg_start,
                _schema_len,
                table_name,
                tbl,
                mapped,
                types,
                metrics,
            ) = best_candidate

            cols = tbl.get("column_list") or []
            q_table = quote_ident(table_name)
            col_defs = ", ".join(f"{quote_ident(c)} {(t or 'TEXT')}" for c, t in zip(cols, types, strict=False))
            if table_name.lower() != "sqlite_sequence":
                dst_cur.execute(f"CREATE TABLE IF NOT EXISTS {q_table} ({col_defs});")
                placeholders = ", ".join("?" for _ in cols)
                try:
                    dst_cur.execute(
                        f"INSERT OR IGNORE INTO {q_table} VALUES ({placeholders});",
                        tuple(mapped),
                    )
                except sqlite3.Error:
                    continue

            written_tables.add(table_name)
            mappings.append((r_idx, lf_rowids[r_idx], lf_id_value, table_name, metrics["confidence"]))
            consumed_rows.add(r_idx)
            break  # stop scanning more start positions for this row

        # (If no match, leave the row for 'unmatched' CSV if you're exporting it.)

    if written_tables:
        seq_exists = dst_cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()
        if not seq_exists:
            dst_cur.execute("DROP TABLE IF EXISTS _tmp_autoinc")
            dst_cur.execute("CREATE TABLE IF NOT EXISTS _tmp_autoinc(id INTEGER PRIMARY KEY AUTOINCREMENT)")
            dst_cur.execute("INSERT INTO _tmp_autoinc DEFAULT VALUES")
            dst_cur.execute("DELETE FROM _tmp_autoinc")
            dst_cur.execute("DROP TABLE _tmp_autoinc")
            seq_exists = dst_cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
            ).fetchone()

        if seq_exists:
            for table_name in sorted(written_tables):
                schema_cols = schema_by_table.get(table_name)
                if not schema_cols:
                    continue
                first_col_name = schema_cols[0][0]
                if first_col_name != "ID":
                    continue
                max_id_row = dst_cur.execute(
                    f"SELECT MAX({quote_ident(first_col_name)}) FROM {quote_ident(table_name)}"
                ).fetchone()
                max_id = max_id_row[0] if max_id_row and max_id_row[0] is not None else None
                if max_id is None:
                    continue
                dst_cur.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table_name,))
                dst_cur.execute("INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)", (table_name, int(max_id)))

    dst.commit()
    dst.close()
    src.close()

    # Mapping CSV
    with map_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lf_row_index", "lf_rowid", "lf_id_value", "table", "confidence"])
        for lf_idx, lf_rowid, lf_id, tname, score in mappings:
            w.writerow([lf_idx, lf_rowid, "" if lf_id is None else lf_id, tname, f"{score:.3f}"])

    print(f"Success: Rebuilt structured Powerlog DB -> {out_db}")
    print(f"Info: Mapping CSV -> {map_csv}")
    return out_db


# ---------- cli ----------


def main():
    ap = argparse.ArgumentParser(description="Salvage a recovered Powerlog SQLite using fingerprinted rows.")
    ap.add_argument(
        "recovered_db",
        help="Path to *_recovered.sqlite (or any recovered SQLite with lost_and_found)",
    )
    args = ap.parse_args()

    rec = Path(args.recovered_db)
    if not rec.exists():
        print(f"Error: Not found: {rec}", file=sys.stderr)
        sys.exit(1)

    salvage_recovered_sqlite(rec)


if __name__ == "__main__":
    main()
