#!/usr/bin/env python3

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
     • Carved/<base>_<UTC>/Carved_Recovered.sqlite
     • Carved/<base>_<UTC>/carved_all.csv
     • Carved/<base>_<UTC>/carved_protobufs.jsonl   ← JSON Lines
     • Carved/<base>_<UTC>/pages/page_XXXX.bin      ← raw page dumps
 - Clean, ANSI-colored progress
 - Protobuf decoding ON by default (disable with --no-protobuf)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Local protobuf helper
try:
    from protobuf_extractor import maybe_decode_protobuf, to_json
except Exception:
    maybe_decode_protobuf = None
    to_json = None

# ---------------- Config ----------------

EPOCH_MIN = 1262304000.0  # 2010-01-01
EPOCH_MAX = 4102444800.0  # 2100-01-01
WIN_START = datetime(2015, 1, 1, tzinfo=UTC)
WIN_END = datetime(2030, 1, 1, tzinfo=UTC)

OUT_DIR = Path("Carved")

ANSI_RESET = "\x1b[0m"
ANSI_GREEN = "\x1b[32m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"

URL_RE = re.compile(rb"https?://[^\s\0]+", re.IGNORECASE)
_NUM_RE = re.compile(rb"[0-9]{9,}\.?[0-9]*")  # biggish numbers only


# ---------------- Helpers ----------------


def read_sqlite_header_info(fp: Path) -> tuple[int, str, int]:
    """
    Extracts DB header information.
    Returns page size (bytes), encoding type, and num of pages
    """
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
    # Takes input float and returns human-readable GMT datetime string
    try:
        return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d %H:%M:%S GMT")
    except Exception:
        return ""


def percent_bar(done: int, total: int, width: int = 24) -> str:
    # UI Percent done
    p = 1.0 if total == 0 else max(0.0, min(1.0, done / total))
    filled = int(p * width)
    return "[" + "=" * filled + " " * (width - filled) + f"] {int(p*100):3d}%"


# ---------------- Timestamp logic ----------------


TARGET_START = datetime(2015, 1, 1, tzinfo=UTC)
TARGET_END = datetime(2030, 1, 1, tzinfo=UTC)


def _in_range(dt: datetime) -> bool:
    return TARGET_START <= dt <= TARGET_END


def interpret_timestamp_best(value: float | int):
    """
    Strict datetime-driven timestamp classifier.

    Attempts to interpret `value` as one of several known epoch formats:
    - Unix (sec, ms, µs, ns)
    - Cocoa / NSDate (sec, ns since 2001-01-01)
    - WebKit (µs since 1601-01-01)
    Returns (kind:str, original_value:int, human_readable:str)
    Note: Returns the ORIGINAL value, not normalized to Unix seconds.
    """
    unix0 = datetime(1970, 1, 1, tzinfo=UTC)
    cocoa0 = datetime(2001, 1, 1, tzinfo=UTC)
    webkit0 = datetime(1601, 1, 1, tzinfo=UTC)

    try:
        val = int(float(value))
    except Exception:
        return None, None, ""

    now = datetime.now(UTC)
    candidates = []

    def add(label: str, dt: datetime):
        """Accept candidate if within hard 2015–2030 window."""
        if TARGET_START <= dt <= TARGET_END:
            candidates.append((label, dt))

    # --- WebKit (µs since 1601) ---
    if len(str(abs(val))) == 17:
        try:
            dt = webkit0 + timedelta(microseconds=val)
            add("webkit_micro", dt)
        except Exception:
            pass

    # --- Cocoa (since 2001) ---
    for div, label in [(1, "cocoa_sec"), (1e9, "cocoa_nano")]:
        try:
            dt = cocoa0 + timedelta(seconds=val / div)
            add(label, dt)
        except Exception:
            pass

    # --- Unix (since 1970) ---
    for div, label in [
        (1, "unix_sec"),
        (1e3, "unix_milli"),
        (1e6, "unix_micro"),
        (1e9, "unix_nano"),
    ]:
        try:
            dt = unix0 + timedelta(seconds=val / div)
            add(label, dt)
        except Exception:
            pass

    if not candidates:
        return None, None, ""

    # Choose candidate closest to now; tie-break prefers Unix variants
    candidates.sort(
        key=lambda kv: (
            abs(kv[1].timestamp() - now.timestamp()),
            0 if kv[0].startswith("unix") else 1,
        )
    )

    kind, dt = candidates[0]
    return kind, val, dt.strftime("%Y-%m-%d %H:%M:%S GMT")


# ---------------- Finders ----------------


def find_timestamps(page: bytes):
    out = []
    for m in _NUM_RE.finditer(page):
        try:
            raw = float(m.group(0))
        except Exception:
            continue
        kind, epoch, human = interpret_timestamp_best(raw)
        if epoch is not None:
            out.append((m.start(), epoch, kind, human))
    return out


def find_urls(page: bytes) -> list[tuple[int, str]]:
    out = []
    for m in URL_RE.finditer(page):
        try:
            s = m.group(0).decode("utf-8", errors="replace")
            out.append((m.start(), s))
        except Exception:
            continue
    return out


def find_text_runs(page: bytes, min_len: int = 6) -> list[tuple[int, str]]:
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


def _read_varint_soft(buf: bytes, pos: int) -> tuple[int | None, int]:
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


def find_blob_candidates(page: bytes, min_len: int = 16) -> list[tuple[int, bytes]]:
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
        blob = page[k : k + ln]
        if len(blob) >= min_len:
            out.append((i, blob))
        i = k + ln
    return out


# ---------------- Output ----------------


"""
Carved data recorded to:
- SQLite (data, possible protobufs)
- CSV (data)
- JSONL (possible protobufs)
"""


def open_out_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE carved_all (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_no INTEGER,
            page_offset INTEGER,
            abs_offset INTEGER,
            cluster_id INTEGER,
            kind TEXT,                -- 'ts' | 'url' | 'text' | 'blob'
            value_text TEXT,
            value_num INTEGER,        -- For timestamps: original raw value (not normalized)
            ts_kind_guess TEXT,       -- unix_sec | unix_milli | unix_micro | unix_nano |
                                      -- cocoa_sec | cocoa_nano | webkit_micro
            ts_human TEXT             -- Human-readable GMT timestamp
        );
    """
    )
    c.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_all
        ON carved_all(kind, value_text, COALESCE(value_num,-1), page_no);
    """
    )
    c.execute(
        """
        CREATE TABLE carved_protobufs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_abs_offset INTEGER,
            page_no INTEGER,
            abs_offset INTEGER,
            json_pretty TEXT
        );
    """
    )
    conn.commit()
    c.execute("PRAGMA journal_mode=DELETE;")
    c.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()
    return conn


def write_csv(path: Path, header: list[str], rows: list[tuple[Any, ...]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def append_jsonl(path: Path, obj: dict[str, Any]):
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
    ap.add_argument(
        "--no-pretty-protobuf",
        action="store_true",
        help="Compact JSON output for protobufs",
    )
    args = ap.parse_args()

    src = Path(args.db)
    if not src.is_file():
        print(f"{ANSI_YELLOW}✗ File not found:{ANSI_RESET} {src}")
        sys.exit(1)

    page_size, encoding, page_count = read_sqlite_header_info(src)

    # Timestamped outputs
    now_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = src.stem
    FILE_DIR = Path(f"{OUT_DIR}/{base}_{now_tag}")
    PAGES_DIR = Path(f"{FILE_DIR}/pages")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FILE_DIR.mkdir(parents=True, exist_ok=False)
    PAGES_DIR.mkdir(parents=True, exist_ok=False)

    out_sqlite = FILE_DIR / f"{base}_Carved_Recovered.sqlite"
    out_csv_main = FILE_DIR / f"{base}_carved_all.csv"
    out_jsonl_pb = FILE_DIR / f"{base}_carved_protobufs.jsonl"

    print("╭──────────────────────────────╮")
    print("│      SQLite Carver v3.2      │")
    print("│      by WarpedWing Labs      │")
    print("╰──────────────────────────────╯\n")
    print(f"{ANSI_GREEN}[✓]{ANSI_RESET} File loaded: {src.name}")
    print(
        f"{ANSI_CYAN}[•]{ANSI_RESET} Page size: {page_size} bytes  |  Encoding: {encoding}"
    )

    db = open_out_db(out_sqlite)
    cur = db.cursor()

    total = page_count or (src.stat().st_size // page_size)
    cluster_pages = 16
    clustering_on = not args.no_cluster

    print("\n─────────────────────────────────────────────")
    seen_ts_abs: set[int] = set()  # absolute offsets for TS
    seen_url: set[tuple[int, str]] = set()
    seen_txt: set[tuple[int, str]] = set()

    # For CSVs
    rows_all: list[tuple[Any, ...]] = []

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
            print(
                f"\r{percent_bar(pno+1,total)}  Parsing page {pno+1:04d}/{total:04d}",
                end="",
                flush=True,
            )

            # Timestamps
            for off, epoch, kind, human in find_timestamps(page):
                abs_off = abs_page_start + off
                if abs_off in seen_ts_abs:
                    continue
                seen_ts_abs.add(abs_off)
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                    "kind,value_text,value_num,ts_kind_guess,ts_human) "
                    "VALUES (?,?,?,?, 'ts',NULL,?,?,?)",
                    (pno, off, abs_off, cluster_id, epoch, kind, human),
                )
                rows_all.append(
                    (pno, off, abs_off, cluster_id, "ts", "", epoch, kind, human)
                )

            # URLs
            for off, url in find_urls(page):
                if (pno, url) in seen_url:
                    continue
                seen_url.add((pno, url))
                abs_off = abs_page_start + off
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind_guess,ts_human) "  # noqa: E501
                    "VALUES (?,?,?,?, 'url',?,NULL,NULL,NULL)",
                    (pno, off, abs_off, cluster_id, url),
                )
                rows_all.append(
                    (pno, off, abs_off, cluster_id, "url", url, None, None, None)
                )

            # Text
            for off, txt in find_text_runs(page, min_len=8):
                if (pno, txt) in seen_txt:
                    continue
                seen_txt.add((pno, txt))
                abs_off = abs_page_start + off
                cur.execute(
                    "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind_guess,ts_human) "  # noqa: E501
                    "VALUES (?,?,?,?, 'text',?,NULL,NULL,NULL)",
                    (pno, off, abs_off, cluster_id, txt),
                )
                rows_all.append(
                    (pno, off, abs_off, cluster_id, "text", txt, None, None, None)
                )

            # Protobufs → JSONL
            if (
                not args.no_protobuf
                and maybe_decode_protobuf is not None
                and to_json is not None
            ):
                taken = 0
                for off, blob in find_blob_candidates(page, min_len=16):
                    parsed = maybe_decode_protobuf(blob, max_depth=4)
                    if not parsed:
                        continue
                    abs_off = abs_page_start + off
                    jtxt = to_json(parsed, pretty=(not args.no_pretty_protobuf))
                    cur.execute(
                        "INSERT INTO carved_protobufs(parent_abs_offset,page_no,abs_offset,json_pretty) VALUES (?,?,?,?)",  # noqa: E501
                        (abs_off, pno, abs_off, jtxt),
                    )
                    append_jsonl(
                        out_jsonl_pb,
                        {
                            "parent_abs_offset": abs_off,
                            "page_no": pno,
                            "abs_offset": abs_off,
                            "protobuf": (
                                json.loads(jtxt)
                                if not args.no_pretty_protobuf
                                else parsed
                            ),
                        },
                    )
                    taken += 1
                    if taken >= 4:
                        break

    # Finalize DB (single file, no WAL/SHM)
    db.commit()
    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    db.commit()
    db.close()

    # Master CSV
    write_csv(
        out_csv_main,
        [
            "page_no",
            "page_offset",
            "abs_offset",
            "cluster_id",
            "kind",
            "value_text",
            "value_num",
            "ts_kind_guess",
            "ts_human",
        ],
        rows_all,
    )

    print("\n─────────────────────────────────────────────\n")
    print(f"{ANSI_GREEN}[✓]{ANSI_RESET} Exported:")
    print(f"   • {out_sqlite}")
    print(f"   • {out_csv_main}")
    print(f"   • {out_jsonl_pb}")
    print(f"   • {PAGES_DIR}/page_*.bin")
    print(f"\n{ANSI_GREEN}[✓]{ANSI_RESET} Done.")


if __name__ == "__main__":
    main()
