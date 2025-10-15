#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQLite Carver v3.2
by WarpedWing Labs

A page-oriented forensic SQLite data carver.

Features:
 - Detects and parses timestamps (Unix, ms, Apple NSDate, WebKit)
 - Picks the *likeliest* interpretation (closest to now; unix wins ties)
 - Converts timestamps to human-readable GMT (ts_human)
 - Extracts URLs, UTF-8 text, and protobuf blobs
 - Deduplicates and clusters pages (unless --no-cluster)
 - Outputs:
     • Carved/<base>_<UTC>_Carved_Recovered.sqlite
     • Carved/<base>_<UTC>_carved_all.csv
     • Carved/<base>_<UTC>_carved_protobufs.jsonl   ← JSON Lines
     • Carved/pages/page_XXXX.bin                    ← raw page dumps
     • Carved/pages/page_XXXX.csv                    ← per-page findings
 - Clean, ANSI-colored progress
 - Protobuf decoding ON by default (disable with --no-protobuf)
"""

from __future__ import annotations
import argparse, csv, json, re, sqlite3, sys, time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

# Local protobuf helper
try:
    from protobuf_extractor import maybe_decode_protobuf, to_json
except Exception:
    maybe_decode_protobuf = None
    to_json = None

# ---------------- Config ----------------

EPOCH_MIN = 1262304000.0  # 2010-01-01
EPOCH_MAX = 4102444800.0  # 2100-01-01
OUT_DIR = Path("Carved")
PAGES_DIR = OUT_DIR / "pages"

ANSI_RESET = "\x1b[0m"
ANSI_GREEN = "\x1b[32m"
ANSI_CYAN  = "\x1b[36m"
ANSI_YELLOW= "\x1b[33m"

URL_RE = re.compile(rb"https?://[^\s\0]+", re.IGNORECASE)
_NUM_RE = re.compile(rb"[0-9]{9,}\.?[0-9]*")  # biggish numbers only

# ---------------- Helpers ----------------

def read_sqlite_header_info(fp: Path) -> Tuple[int, str, int]:
    data = fp.read_bytes()[:100]
    if len(data) < 100 or data[:16] != b"SQLite format 3\0":
        raise ValueError("Not a valid SQLite header")
    page_size = int.from_bytes(data[16:18], "big") or 4096
    if page_size == 1:  # 65536 sentinel
        page_size = 65536
    enc_code = int.from_bytes(data[56:60], "big")
    encoding = {1: "UTF-8", 2: "UTF-16le", 3: "UTF-16be"}.get(enc_code, "UTF-8")
    fsize = fp.stat().st_size
    pages = fsize // page_size if page_size else 0
    return page_size, encoding, int(pages)

def to_gmt_str(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")
    except Exception:
        return ""

def percent_bar(done: int, total: int, width: int = 24) -> str:
    p = 1.0 if total == 0 else max(0.0, min(1.0, done / total))
    filled = int(p * width)
    return "[" + "=" * filled + " " * (width - filled) + f"] {int(p*100):3d}%"

# ---------------- Timestamp logic ----------------

def _candidate_epochs(num: float) -> list[tuple[str, float]]:
    out = []
    # unix seconds
    if EPOCH_MIN <= num <= EPOCH_MAX:
        out.append(("unix", num))
    # unix millis
    if 1e12 <= num <= 1e13:
        s = num / 1000.0
        if EPOCH_MIN <= s <= EPOCH_MAX:
            out.append(("millis", s))
    # WebKit/Chrome (µs since 1601-01-01)
    if 1.0e15 <= num <= 1.9e18:
        s = (num / 1e6) - 11644473600.0
        if EPOCH_MIN <= s <= EPOCH_MAX:
            out.append(("webkit", s))
    # Apple NSDate (seconds since 2001-01-01)
    if 0.0 <= num <= 1.6e9:
        s = num + 978307200.0
        if EPOCH_MIN <= s <= EPOCH_MAX:
            out.append(("nsdate", s))
    return out

def interpret_timestamp_best(num: float) -> tuple[Optional[float], str]:
    """
    Choose the likeliest interpretation:
      1) smallest |epoch - now|
      2) tie-break: unix > millis > nsdate > webkit
    Returns (epoch_seconds or None, kind_str).
    """
    cands = _candidate_epochs(num)
    if not cands:
        return None, "invalid"
    now = datetime.now(timezone.utc).timestamp()
    priority = {"unix": 0, "millis": 1, "nsdate": 2, "webkit": 3}
    # Prefer nearer to 'now'; break ties with priority above
    cands.sort(key=lambda kv: (abs(kv[1] - now), priority.get(kv[0], 9)))
    # Additional nudge: if top-2 within ~180 days, prefer unix if present
    if len(cands) >= 2:
        a, b = cands[0], cands[1]
        if abs(a[1] - b[1]) <= 180 * 86400:
            unix = next((x for x in cands if x[0] == "unix"), None)
            if unix is not None and unix != a:
                return unix[1], "unix"
    return cands[0][1], cands[0][0]

def find_timestamps(page: bytes) -> List[Tuple[int, float, str]]:
    out = []
    for m in _NUM_RE.finditer(page):
        try:
            raw = float(m.group(0))
        except Exception:
            continue
        epoch, kind = interpret_timestamp_best(raw)  # NB: (epoch, kind)
        if epoch is not None:
            out.append((m.start(), epoch, kind))
    return out

# ---------------- Other finders ----------------

def find_urls(page: bytes) -> List[Tuple[int, str]]:
    out = []
    for m in URL_RE.finditer(page):
        try:
            s = m.group(0).decode("utf-8", errors="replace")
            out.append((m.start(), s))
        except Exception:
            continue
    return out

def find_text_runs(page: bytes, min_len: int = 6) -> List[Tuple[int, str]]:
    out = []
    parts = []
    start = 0
    for i, b in enumerate(page):
        if b < 32 and b not in (9, 10, 13):
            if i > start:
                parts.append((start, page[start:i]))
            start = i + 1
    if start < len(page):
        parts.append((start, page[start:]))

    for off, chunk in parts:
        s = chunk.decode("utf-8", errors="replace").strip()
        if len(s) >= min_len and any(c.isalpha() for c in s):
            out.append((off, s))
    return out

def _read_varint_soft(buf: bytes, pos: int) -> Tuple[Optional[int], int]:
    result = 0
    shift = 0
    i = pos
    while i < len(buf) and shift <= 70:
        b = buf[i]
        result |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            return result, i
        shift += 7
    return None, pos

def find_blob_candidates(page: bytes, min_len: int = 16) -> List[Tuple[int, bytes]]:
    out = []
    i = 0
    while i < len(page):
        key, j = _read_varint_soft(page, i)
        if key is None:
            i += 1
            continue
        if (key & 0x7) != 2:
            i = j
            continue
        ln, k = _read_varint_soft(page, j)
        if ln is None or ln <= 0 or k + ln > len(page):
            i = j
            continue
        blob = page[k:k+ln]
        if len(blob) >= min_len:
            out.append((i, blob))
        i = k + ln
    return out

# ---------------- Output ----------------

def open_out_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE carved_all (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_no INTEGER,
            page_offset INTEGER,
            abs_offset INTEGER,
            cluster_id INTEGER,
            kind TEXT,                -- 'ts' | 'url' | 'text' | 'blob'
            value_text TEXT,
            value_num REAL,
            ts_kind TEXT,             -- 'unix' | 'webkit' | 'nsdate' | 'millis' | 'invalid'
            ts_human TEXT
        );
    """)
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_all
        ON carved_all(kind, value_text, COALESCE(value_num,-1), page_no);
    """)
    c.execute("""
        CREATE TABLE carved_protobufs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_abs_offset INTEGER,
            page_no INTEGER,
            abs_offset INTEGER,
            json_pretty TEXT
        );
    """)
    conn.commit()
    c.execute("PRAGMA journal_mode=DELETE;")
    c.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()
    return conn

def write_csv(path: Path, header: List[str], rows: List[Tuple[Any, ...]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="SQLite Carver (forensic)")
    ap.add_argument("db", help="SQLite file to carve")
    ap.add_argument("--no-cluster", action="store_true", help="Disable page clustering")
    ap.add_argument("--no-protobuf", action="store_true", help="Skip protobuf decoding")
    ap.add_argument("--no-pretty-protobuf", action="store_true", help="Compact JSON output for protobufs")
    args = ap.parse_args()

    src = Path(args.db)
    if not src.is_file():
        print(f"{ANSI_YELLOW}✗ File not found:{ANSI_RESET} {src}")
        sys.exit(1)

    page_size, encoding, page_count = read_sqlite_header_info(src)

    # Timestamped outputs
    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = src.stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    out_sqlite   = OUT_DIR / f"{base}_{now_tag}_Carved_Recovered.sqlite"
    out_csv_main = OUT_DIR / f"{base}_{now_tag}_carved_all.csv"
    out_jsonl_pb = OUT_DIR / f"{base}_{now_tag}_carved_protobufs.jsonl"

    print("╭──────────────────────────────╮")
    print("│      SQLite Carver v3.2      │")
    print("│      by WarpedWing Labs      │")
    print("╰──────────────────────────────╯\n")
    print(f"{ANSI_GREEN}[✓]{ANSI_RESET} File loaded: {src.name}")
    print(f"{ANSI_CYAN}[•]{ANSI_RESET} Page size: {page_size} bytes  |  Encoding: {encoding}")

    db = open_out_db(out_sqlite)
    cur = db.cursor()

    total = page_count or (src.stat().st_size // page_size)
    cluster_pages = 16
    clustering_on = not args.no_cluster

    print("\n─────────────────────────────────────────────")
    seen_ts: set[Tuple[int, float]] = set()
    seen_url: set[Tuple[int, str]]  = set()
    seen_txt: set[Tuple[int, str]]  = set()

    # For CSVs
    rows_all: List[Tuple[Any, ...]] = []

    with src.open("rb") as f:
        for pno in range(total):
            abs_page_start = pno * page_size
            page = f.read(page_size)
            if not page:
                break

            # Always dump the page as .bin
            bin_path = PAGES_DIR / f"page_{pno:04d}.bin"
            bin_path.write_bytes(page)

            cluster_id = (pno // cluster_pages) if clustering_on else pno
            print(f"\r{percent_bar(pno+1,total)}  Parsing page {pno+1:04d}/{total:04d}", end="", flush=True)

            # Timestamps
            for off, epoch, kind in find_timestamps(page):
                if (pno, epoch) in seen_ts:
                    continue
                seen_ts.add((pno, epoch))
                abs_off = abs_page_start + off
                gmt = to_gmt_str(epoch)
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind,ts_human) "
                    "VALUES (?,?,?,?, 'ts',NULL,?,?,?)",
                    (pno, off, abs_off, cluster_id, epoch, kind, gmt),
                )
                rows_all.append((pno, off, abs_off, cluster_id, "ts", "", epoch, kind, gmt))

            # URLs
            for off, url in find_urls(page):
                if (pno, url) in seen_url:
                    continue
                seen_url.add((pno, url))
                abs_off = abs_page_start + off
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind,ts_human) "
                    "VALUES (?,?,?,?, 'url',?,NULL,NULL,NULL)",
                    (pno, off, abs_off, cluster_id, url),
                )
                rows_all.append((pno, off, abs_off, cluster_id, "url", url, None, None, None))

            # Text
            for off, txt in find_text_runs(page, min_len=8):
                if (pno, txt) in seen_txt:
                    continue
                seen_txt.add((pno, txt))
                abs_off = abs_page_start + off
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind,ts_human) "
                    "VALUES (?,?,?,?, 'text',?,NULL,NULL,NULL)",
                    (pno, off, abs_off, cluster_id, txt),
                )
                rows_all.append((pno, off, abs_off, cluster_id, "text", txt, None, None, None))

            # Protobufs → JSONL
            if not args.no_protobuf and maybe_decode_protobuf is not None and to_json is not None:
                taken = 0
                for off, blob in find_blob_candidates(page, min_len=16):
                    parsed = maybe_decode_protobuf(blob, max_depth=4)
                    if not parsed:
                        continue
                    abs_off = abs_page_start + off
                    jtxt = to_json(parsed, pretty=(not args.no_pretty_protobuf))
                    cur.execute(
                        "INSERT INTO carved_protobufs(parent_abs_offset,page_no,abs_offset,json_pretty) VALUES (?,?,?,?)",
                        (abs_off, pno, abs_off, jtxt),
                    )
                    append_jsonl(out_jsonl_pb, {
                        "parent_abs_offset": abs_off,
                        "page_no": pno,
                        "abs_offset": abs_off,
                        "protobuf": json.loads(jtxt) if not args.no_pretty_protobuf else parsed
                    })
                    taken += 1
                    if taken >= 4:
                        break

            # Optional per-page CSV listing (mirrors carved_all columns for this page)
            page_csv = PAGES_DIR / f"page_{pno:04d}.csv"
            # Filter the new rows we just added for this page
            # (cheapest way is to revisit DB for this page only)
            cur.execute(
                "SELECT page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind,ts_human "
                "FROM carved_all WHERE page_no=? ORDER BY page_offset;",
                (pno,)
            )
            found = cur.fetchall()
            if found:
                write_csv(page_csv,
                          ["page_no","page_offset","abs_offset","cluster_id","kind","value_text","value_num","ts_kind","ts_human"],
                          found)

    # Finalize DB (single file, no WAL/SHM)
    db.commit()
    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    db.commit()
    db.close()

    # Master CSV
    write_csv(out_csv_main,
              ["page_no","page_offset","abs_offset","cluster_id","kind","value_text","value_num","ts_kind","ts_human"],
              rows_all)

    print("\n─────────────────────────────────────────────\n")
    print(f"{ANSI_GREEN}[✓]{ANSI_RESET} Exported:")
    print(f"   • {out_sqlite}")
    print(f"   • {out_csv_main}")
    print(f"   • {out_jsonl_pb}")
    print(f"   • {PAGES_DIR}/page_*.bin")
    print(f"   • {PAGES_DIR}/page_*.csv")
    print(f"\n{ANSI_GREEN}[✓]{ANSI_RESET} Done.")
    

if __name__ == "__main__":
    main()
