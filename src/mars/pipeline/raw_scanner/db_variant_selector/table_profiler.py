"""
Compares exemplar and candidate tables to determine data similarity.
Used only for exemplar matches with corrupt original DBs.

Returns: Table score
"""

import math
import sqlite3
from collections import Counter

from mars.pipeline.matcher.rubric_utils import detect_pattern_type


# --------- tiny helpers ---------
def _q_ident(s: str) -> str:
    # double-quote identifiers safely (SQLite treats double-quotes as identifiers)
    return '"' + s.replace('"', '""') + '"'


def _safe_fetchone(cur):
    try:
        return cur.fetchone()
    except sqlite3.Error:
        return None


def _safe_exec(con, sql: str, params: tuple = ()):
    try:
        return con.execute(sql, params)
    except sqlite3.Error:
        return None


def _clip01(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, min(1.0, x))


# Heuristic time windows (adjust if you like)
UNIX_SEC_MIN = 946684800  # 2000-01-01
UNIX_SEC_MAX = 4102444800  # 2100-01-01


def _pct_epoch_like(con, table, col):
    qi = _q_ident
    # Count numeric values
    q_num = f"""
        SELECT
            SUM(CASE WHEN typeof({qi(col)}) IN ('integer','real') THEN 1 ELSE 0 END),
            SUM(CASE WHEN typeof({qi(col)}) IN ('integer','real') AND CAST({qi(col)} AS REAL) BETWEEN ? AND ? THEN 1 ELSE 0 END),
            SUM(CASE WHEN typeof({qi(col)}) IN ('integer','real') AND CAST({qi(col)} AS REAL)/1000 BETWEEN ? AND ? THEN 1 ELSE 0 END),
            SUM(CASE WHEN typeof({qi(col)}) IN ('integer','real') AND CAST({qi(col)} AS REAL)/1000000 BETWEEN ? AND ? THEN 1 ELSE 0 END),
            SUM(CASE WHEN typeof({qi(col)}) IN ('integer','real') AND CAST({qi(col)} AS REAL)/1000000000 BETWEEN ? AND ? THEN 1 ELSE 0 END)
        FROM {qi(table)};
    """  # noqa: E501
    cur = _safe_exec(con, q_num, (UNIX_SEC_MIN, UNIX_SEC_MAX) * 4)
    if not cur:
        return {"p_sec": 0, "p_ms": 0, "p_us": 0, "p_ns": 0}
    n_num, n_sec, n_ms, n_us, n_ns = _safe_fetchone(cur) or (0, 0, 0, 0, 0)
    if not n_num:
        return {"p_sec": 0, "p_ms": 0, "p_us": 0, "p_ns": 0}
    return {
        "p_sec": n_sec / n_num,
        "p_ms": n_ms / n_num,
        "p_us": n_us / n_num,
        "p_ns": n_ns / n_num,
    }


def _pct_like(con, table, col, like_clause: str):
    qi = _q_ident
    q = f"""SELECT
              SUM(CASE WHEN typeof({qi(col)})='text' AND ({like_clause}) THEN 1 ELSE 0 END),
              SUM(CASE WHEN typeof({qi(col)})='text' THEN 1 ELSE 0 END)
            FROM {qi(table)};"""
    cur = _safe_exec(con, q)
    if not cur:
        return 0.0
    hit, denom = _safe_fetchone(cur) or (0, 0)
    return (hit / denom) if denom else 0.0


def _pct_pattern_py(con, table, col, pattern: str, cap: int = 500) -> float:
    qi = _q_ident
    # Pull a sample of text values and detect with Python regexes
    q = f"""SELECT {qi(col)} FROM {qi(table)}
            WHERE typeof({qi(col)})='text' AND {qi(col)} IS NOT NULL
            LIMIT {int(cap)};"""
    cur = _safe_exec(con, q)
    if not cur:
        return 0.0
    rows = cur.fetchall() or []
    if not rows:
        return 0.0
    hits = 0
    total = 0
    for (v,) in rows:
        total += 1
        try:
            if detect_pattern_type(v) == pattern:
                hits += 1
        except Exception:
            continue
    return (hits / total) if total else 0.0


def _text_lengths(con, table, col):
    qi = _q_ident
    q = f"""SELECT MIN(length({qi(col)})),
                   MAX(length({qi(col)})),
                   AVG(length({qi(col)}))
            FROM {qi(table)}
            WHERE typeof({qi(col)})='text' AND {qi(col)} IS NOT NULL;"""
    cur = _safe_exec(con, q)
    if not cur:
        return (None, None, None)
    return _safe_fetchone(cur) or (None, None, None)


def _numeric_minmax(con, table, col):
    qi = _q_ident
    q = f"""SELECT MIN(CAST({qi(col)} AS REAL)),
                   MAX(CAST({qi(col)} AS REAL))
            FROM {qi(table)}
            WHERE typeof({qi(col)}) IN ('integer','real') AND {qi(col)} IS NOT NULL;"""
    cur = _safe_exec(con, q)
    if not cur:
        return (None, None)
    return _safe_fetchone(cur) or (None, None)


def _type_mix(con, table, col):
    qi = _q_ident
    q = f"""SELECT typeof({qi(col)}), COUNT(1)
            FROM {qi(table)}
            GROUP BY typeof({qi(col)});"""
    cur = _safe_exec(con, q)
    mix = Counter()
    if cur:
        for t, n in cur.fetchall():
            mix[t or "null"] += n or 0
    total = sum(mix.values()) or 1
    return {k: v / total for k, v in mix.items()}, total


def _null_count(con, table, col):
    qi = _q_ident
    q = f"""SELECT SUM(CASE WHEN {qi(col)} IS NULL THEN 1 ELSE 0 END),
                   COUNT(1)
            FROM {qi(table)};"""
    cur = _safe_exec(con, q)
    if not cur:
        return (0, 0, 0.0)
    n_null, n_all = _safe_fetchone(cur) or (0, 0)
    p = (n_null / n_all) if n_all else 0.0
    return n_null, n_all, p


def _distinct_ratio_sample(con, table, col, cap=5000):
    qi = _q_ident
    # Works even on WITHOUT ROWID; LIMIT is enough for a coarse ratio
    q = f"""WITH sample AS (
               SELECT {qi(col)} AS v FROM {qi(table)}
               WHERE {qi(col)} IS NOT NULL
               LIMIT {int(cap)}
           )
           SELECT COUNT(DISTINCT v), COUNT(1) FROM sample;"""
    cur = _safe_exec(con, q)
    if not cur:
        return 0.0
    d, n = _safe_fetchone(cur) or (0, 0)
    return (d / n) if n else 0.0


def profile_column(con: sqlite3.Connection, table: str, col: str) -> dict:
    mix, n_total = _type_mix(con, table, col)
    _, _, p_null = _null_count(con, table, col)
    p_text = mix.get("text", 0.0)
    p_num = mix.get("integer", 0.0) + mix.get("real", 0.0)
    tmin, tmax, tavg = _text_lengths(con, table, col)
    nmin, nmax = _numeric_minmax(con, table, col)
    p_url = _pct_like(
        con,
        table,
        col,
        f'{_q_ident(col)} LIKE "http%" OR '
        f'{_q_ident(col)} LIKE "imap%" OR '
        f'{_q_ident(col)} LIKE "https%" OR '
        f'{_q_ident(col)} LIKE "ftp%" OR '
        f'{_q_ident(col)} LIKE "file:%" OR '
        f'{_q_ident(col)} LIKE "mailto:%" OR '
        f'{_q_ident(col)} LIKE "chrome-extension:%"',
    )
    p_iso = _pct_like(con, table, col, f'{_q_ident(col)} GLOB "____-__-__*"')
    # crude UUID: 8-4-4-4-12 hex with dashes
    p_uuid = _pct_like(con, table, col, f'{_q_ident(col)} GLOB "????????-????-????-????-????????????"')
    # crude path: has a slash and not a URL
    p_path = _pct_like(
        con,
        table,
        col,
        f'{_q_ident(col)} LIKE "%/%" AND {_q_ident(col)} NOT LIKE "http%"',
    )
    # Additional semantic patterns (via regex sampling)
    p_email = _pct_pattern_py(con, table, col, "email", cap=300)
    p_domain = _pct_pattern_py(con, table, col, "domain", cap=300)
    p_timestamp_text = _pct_pattern_py(con, table, col, "timestamp_text", cap=300)
    p_distinct = _distinct_ratio_sample(con, table, col, cap=5000)
    epochs = _pct_epoch_like(con, table, col)

    return {
        "types": mix,  # dict: typeof -> fraction
        "p_null": p_null,
        "p_num": p_num,
        "p_text": p_text,
        "num_min": nmin,
        "num_max": nmax,
        "text_minlen": tmin,
        "text_maxlen": tmax,
        "text_avglens": tavg,
        "p_url": p_url,
        "p_iso": p_iso,
        "p_uuid": p_uuid,
        "p_path": p_path,
        "p_email": p_email,
        "p_domain": p_domain,
        "p_timestamp_text": p_timestamp_text,
        "p_distinct": p_distinct,
        "epoch_like": epochs,  # p_sec, p_ms, p_us, p_ns (over numeric values)
        "n_total": n_total,
    }


def profile_table(con: sqlite3.Connection, table: str, cap_cols: int = 512) -> dict:
    cols = []
    cur = _safe_exec(con, f"PRAGMA table_info({_q_ident(table)});")
    if cur:
        for row in cur.fetchall():
            name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
            if isinstance(name, str) and name:
                cols.append(name)
    cols = cols[:cap_cols]  # prevent absurd schemas from blowing up
    out = {}
    for c in cols:
        out[c] = profile_column(con, table, c)
    return out


def _l1_dist_dict(a: dict, b: dict, keys) -> float:
    return sum(abs((a.get(k, 0.0) or 0.0) - (b.get(k, 0.0) or 0.0)) for k in keys)


def _overlap_range(a_min, a_max, b_min, b_max) -> float:
    if a_min is None or a_max is None or b_min is None or b_max is None:
        return 0.0
    if a_max < b_min or b_max < a_min:
        return 0.0
    inter = min(a_max, b_max) - max(a_min, b_min)
    union = max(a_max, b_max) - min(a_min, b_min)
    return _clip01(inter / union) if union > 0 else 0.0


def compare_column_profiles(p_case: dict, p_ex: dict) -> dict:
    # 0..1 similarity; higher is better
    type_keys = set(p_case["types"]) | set(p_ex["types"])
    type_l1 = _l1_dist_dict(p_case["types"], p_ex["types"], type_keys)  # 0..2
    type_sim = 1.0 - (type_l1 / 2.0)

    # numeric range overlap and numeric-ness match
    num_overlap = _overlap_range(p_case["num_min"], p_case["num_max"], p_ex["num_min"], p_ex["num_max"])
    numness_gap = abs(p_case["p_num"] - p_ex["p_num"])
    num_sim = 0.5 * num_overlap + 0.5 * (1.0 - numness_gap)

    # text length overlap and text-ness match
    tmin_c, tmax_c = p_case["text_minlen"], p_case["text_maxlen"]
    tmin_e, tmax_e = p_ex["text_minlen"], p_ex["text_maxlen"]
    text_overlap = _overlap_range(tmin_c, tmax_c, tmin_e, tmax_e)
    textness_gap = abs(p_case["p_text"] - p_ex["p_text"])
    text_sim = 0.5 * text_overlap + 0.5 * (1.0 - textness_gap)

    # domain hints similarity
    hints = ["p_url", "p_iso", "p_uuid", "p_path", "p_distinct"]
    hint_sim = 1.0 - (sum(abs(p_case[h] - p_ex[h]) for h in hints) / len(hints))

    # epoch pattern similarity
    ekeys = ["p_sec", "p_ms", "p_us", "p_ns"]
    epoch_l1 = _l1_dist_dict(p_case["epoch_like"], p_ex["epoch_like"], ekeys)
    epoch_sim = 1.0 - (epoch_l1 / 2.0)

    # combine (weights are tunable)
    # Increased type_sim weight from 30% to 50% to prioritize type correctness
    score = 0.50 * type_sim + 0.15 * num_sim + 0.15 * text_sim + 0.15 * hint_sim + 0.05 * epoch_sim

    issues = []
    # Stricter type purity threshold: 0.85 allows max ~15% contamination (was 0.70 / ~30%)
    if type_sim < 0.85:
        issues.append(f"type-mix differs ({type_sim:.2f})")
    if num_sim < 0.60 and p_ex["p_num"] > 0.3:
        issues.append(f"numeric shape off ({num_sim:.2f})")
    if text_sim < 0.60 and p_ex["p_text"] > 0.3:
        issues.append(f"text shape off ({text_sim:.2f})")
    # Stricter semantic pattern matching for UUIDs and timestamps
    if hint_sim < 0.80:
        issues.append(f"domain hints differ ({hint_sim:.2f})")
    if epoch_sim < 0.80 and (max(p_ex["epoch_like"].values()) > 0.2):
        issues.append(f"time-epoch pattern off ({epoch_sim:.2f})")

    return {"score": _clip01(score), "issues": issues}


def compare_table_profiles(prof_case: dict, prof_ex: dict) -> dict:
    # Build a similarity matrix: case_col -> {ex_col: score}
    matrix = {}
    for ccol, pc in prof_case.items():
        row = {}
        for ecol, pe in prof_ex.items():
            row[ecol] = compare_column_profiles(pc, pe)["score"]
        matrix[ccol] = row

    results = []
    for ccol in prof_case:
        row = matrix[ccol]
        # diagonal candidate (same name) if present
        diag = row.get(ccol, None)
        best_alt_col = max(row, key=row.get) if row else None
        best_alt_score = row.get(best_alt_col, 0.0) if best_alt_col else 0.0

        verdict = "ok"
        issues = []
        if diag is None:
            verdict = "missing_in_exemplar"
        else:
            if diag < 0.65:
                verdict = "mismatch"
                issues.append(f"low diagonal score {diag:.2f}")
            if best_alt_col and best_alt_col != ccol and best_alt_score - (diag or 0.0) > 0.10:
                verdict = "possible_swap"
                issues.append(f"better match: {best_alt_col} ({best_alt_score:.2f}) > {diag or 0.0:.2f}")

        results.append(
            {
                "column": ccol,
                "diag_score": diag,
                "best_alt": best_alt_col,
                "best_alt_score": best_alt_score,
                "verdict": verdict,
                "notes": issues,
            }
        )

    table_score = _clip01(sum([(r["diag_score"] or 0.0) for r in results]) / max(1, len(results)))
    return {"table_score": table_score, "columns": results, "matrix": matrix}
