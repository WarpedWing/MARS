#!/usr/bin/env python3

"""
SQLite Carver v3.4
by WarpedWing Labs

A page-oriented forensic SQLite data carver with intelligent timestamp validation.

Features:
 - Detects and parses timestamps (Unix, ms, Apple NSDate, WebKit)
 - **NEW: Confidence scoring to distinguish real timestamps from IDs**
   - Analyzes context keywords (created_at, modified_at, etc.)
   - Detects sequential patterns (likely IDs vs scattered timestamps)
   - Checks for temporal clustering (multiple timestamps nearby)
   - Filters Snowflake IDs, Facebook IDs, and other timestamp-like values
 - Configurable confidence threshold (--min-confidence, default: 0.5)
 - Picks the *likeliest* interpretation (closest to now; unix wins ties)
 - Converts timestamps to human-readable GMT (ts_human)
 - Extracts URLs, UTF-8 text, and protobuf blobs
 - Deduplicates by absolute offset across all artifact types
 - Clusters pages (unless --no-cluster)
 - Batch commits for memory efficiency and crash recovery
 - Configurable timestamp validity range (--ts-start, --ts-end)
 - Outputs:
     • Carved/<base>_<UTC>/Carved_Recovered.sqlite (includes ts_confidence column)
     • Carved/<base>_<UTC>/carved_all.csv (includes ts_confidence column)
     • Carved/<base>_<UTC>/carved_protobufs.jsonl   ← JSON Lines
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

# Local protobuf timestamp extractor
try:
    from protobuf_timestamp_extractor import (
        analyze_protobuf_for_timestamps,
        should_keep_protobuf,
    )

    PROTOBUF_FILTER_AVAILABLE = True
except Exception:
    analyze_protobuf_for_timestamps = None
    should_keep_protobuf = None
    PROTOBUF_FILTER_AVAILABLE = False

# Local timestamp validation (V2 - Classification-based)
try:
    from timestamp_classifier import (
        ClassificationStats,
        TimestampClassification,
        should_keep_timestamp,
    )
    from timestamp_classifier_v2 import classify_page_timestamps
    from timestamp_patterns import find_timestamp_candidates, set_timestamp_range

    CLASSIFIER_V2_AVAILABLE = True
except Exception as e:
    print(f"Warning: V2 classifier not available ({e}), using fallback")
    CLASSIFIER_V2_AVAILABLE = False
    find_timestamp_candidates = None
    classify_page_timestamps = None
    set_timestamp_range = None
    TimestampClassification = None
    should_keep_timestamp = None
    ClassificationStats = None

# ---------------- Config ----------------

EPOCH_MIN = 1262304000.0  # 2010-01-01
EPOCH_MAX = 4102444800.0  # 2100-01-01
WIN_START = datetime(2015, 1, 1, tzinfo=UTC)
WIN_END = datetime(2030, 1, 1, tzinfo=UTC)

OUT_DIR = Path("Carved")
BATCH_SIZE = 100  # Commit every N pages

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
    with fp.open("rb") as f:
        data = f.read(100)
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
    """
    Find timestamps in page data.
    Pre-filters by digit count to avoid processing obviously invalid values.
    Valid ranges:
    - 10 digits: unix_sec (e.g., 1609459200)
    - 13 digits: unix_milli or cocoa_sec
    - 16-17 digits: unix_micro, cocoa_nano, webkit_micro
    - 19 digits: unix_nano
    """
    out = []
    for m in _NUM_RE.finditer(page):
        try:
            raw_str = m.group(0)
            # Quick filter: skip if all zeros or obviously invalid
            if raw_str.replace(b".", b"").replace(b"0", b"") == b"":
                continue

            raw = float(raw_str)
            # Pre-filter by digit count (ignore decimal for now)
            digit_count = len(str(int(abs(raw))))

            # Only process numbers with plausible timestamp lengths
            # 10=unix_sec, 13=unix_milli/cocoa_sec, 16-17=micro/webkit, 19=nano
            if digit_count not in (10, 13, 16, 17, 19):
                continue

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
            value_num TEXT,           -- For timestamps: original raw value (hex, decimal, etc.)
            ts_kind_guess TEXT,       -- unix_sec | unix_milli | unix_micro | unix_nano |
                                      -- cocoa_sec | cocoa_nano | webkit_micro | snowflake | unix_sec_hex
            ts_human TEXT,            -- Human-readable GMT timestamp
            ts_classification TEXT,   -- confirmed_timestamp | confirmed_id | likely_timestamp |
                                      -- likely_id | ambiguous | invalid
            ts_reason TEXT,           -- Why this classification was chosen
            ts_source_url TEXT,       -- Source URL if extracted from URL
            ts_field_name TEXT        -- Field name if detected
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
            json_pretty TEXT,
            has_timestamps INTEGER,     -- Boolean: 1 if timestamps found, 0 otherwise
            timestamp_count INTEGER,    -- Number of timestamps found
            timestamp_fields TEXT       -- JSON array of timestamp field names and values
        );
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_protobufs_page
        ON carved_protobufs(page_no);
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_protobufs_offset
        ON carved_protobufs(abs_offset);
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


def append_csv_batch(path: Path, rows: list[tuple[Any, ...]]):
    """Append rows to existing CSV file (no header)"""
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
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
    ap.add_argument(
        "--ts-start",
        type=str,
        default="2015-01-01",
        help="Start of timestamp validity range (YYYY-MM-DD, default: 2015-01-01)",
    )
    ap.add_argument(
        "--ts-end",
        type=str,
        default="2030-01-01",
        help="End of timestamp validity range (YYYY-MM-DD, default: 2030-01-01)",
    )
    ap.add_argument(
        "--filter-mode",
        type=str,
        choices=["strict", "balanced", "permissive", "all"],
        default="permissive",
        help="Timestamp filtering mode: strict (only confirmed), balanced (confirmed+likely), "
        "permissive (exclude only confirmed IDs - DEFAULT), all (no filtering)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for carved data (default: same directory as source database)",
    )
    args = ap.parse_args()

    # Parse timestamp range
    global TARGET_START, TARGET_END
    try:
        TARGET_START = datetime.strptime(args.ts_start, "%Y-%m-%d").replace(tzinfo=UTC)
        TARGET_END = datetime.strptime(args.ts_end, "%Y-%m-%d").replace(tzinfo=UTC)
        # Update timestamp_patterns module if available
        if set_timestamp_range:
            set_timestamp_range(TARGET_START, TARGET_END)
    except ValueError as e:
        print(f"{ANSI_YELLOW}✗ Invalid timestamp range:{ANSI_RESET} {e}")
        sys.exit(1)

    src = Path(args.db)
    if not src.is_file():
        print(f"{ANSI_YELLOW}✗ File not found:{ANSI_RESET} {src}")
        sys.exit(1)

    page_size, encoding, page_count = read_sqlite_header_info(src)

    # Determine output directory (default: same directory as source database)
    output_base = Path(args.output_dir) if args.output_dir else src.parent

    # Timestamped outputs
    now_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = src.stem
    FILE_DIR = output_base / "Carved" / f"{base}_{now_tag}"
    FILE_DIR.mkdir(parents=True, exist_ok=False)

    out_sqlite = FILE_DIR / f"{base}_Carved_Recovered.sqlite"
    out_csv_main = FILE_DIR / f"{base}_carved_all.csv"
    out_jsonl_pb = FILE_DIR / f"{base}_carved_protobufs.jsonl"

    print("╭──────────────────────────────╮")
    print("│      SQLite Carver v3.4      │")
    print("│      by WarpedWing Labs      │")
    print("╰──────────────────────────────╯\n")
    print(f"{ANSI_GREEN}[+]{ANSI_RESET} File loaded: {src.name}")
    print(
        f"{ANSI_CYAN}[•]{ANSI_RESET} Page size: {page_size} bytes  |  Encoding: {encoding}"
    )
    print(f"{ANSI_CYAN}[•]{ANSI_RESET} Timestamp filter mode: {args.filter_mode}")
    if CLASSIFIER_V2_AVAILABLE:
        print(f"{ANSI_GREEN}[+]{ANSI_RESET} Using V2 classifier (Unfurl + time_decode)")
    else:
        print(f"{ANSI_YELLOW}[!]{ANSI_RESET} Using fallback classifier")

    db = open_out_db(out_sqlite)
    cur = db.cursor()

    total = page_count or (src.stat().st_size // page_size)
    cluster_pages = 16
    clustering_on = not args.no_cluster

    print("\n─────────────────────────────────────────────")
    seen_ts_abs: set[int] = set()  # absolute offsets for TS
    seen_url_abs: set[int] = set()  # absolute offsets for URLs
    seen_txt_abs: set[int] = set()  # absolute offsets for text
    seen_pb_abs: set[int] = set()  # absolute offsets for protobufs

    # For CSVs - batch buffer
    rows_batch: list[tuple[Any, ...]] = []

    # Initialize CSV with header
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
            "ts_classification",
            "ts_reason",
            "ts_source_url",
            "ts_field_name",
        ],
        [],
    )

    # Track classification stats
    classification_stats = (
        ClassificationStats()
        if CLASSIFIER_V2_AVAILABLE and ClassificationStats
        else None
    )

    with src.open("rb") as f:
        for pno in range(total):
            try:
                abs_page_start = pno * page_size
                page = f.read(page_size)
                if not page:
                    break

                cluster_id = (pno // cluster_pages) if clustering_on else None
                print(
                    f"\r{percent_bar(pno+1,total)}  Parsing page {pno+1:04d}/{total:04d}",
                    end="",
                    flush=True,
                )

                # Timestamps with V2 classification (decimal timestamps)
                if (
                    CLASSIFIER_V2_AVAILABLE
                    and find_timestamp_candidates
                    and classify_page_timestamps
                ):
                    # Get URL offsets from this page first
                    page_url_offsets = find_urls(page)

                    # Find raw timestamp candidates (decimal only)
                    raw_candidates = find_timestamp_candidates(page)

                    # Classify them using V2 system
                    classified = classify_page_timestamps(
                        page, raw_candidates, page_url_offsets
                    )

                    # Filter and insert based on mode
                    for finding in classified:
                        # Track stats
                        if classification_stats:
                            classification_stats.add(finding.classification)

                        # Filter by mode
                        keep = True
                        if should_keep_timestamp:
                            keep = should_keep_timestamp(
                                finding.classification, args.filter_mode
                            )
                        if not keep:
                            continue

                        abs_off = abs_page_start + finding.offset
                        if abs_off in seen_ts_abs:
                            continue
                        seen_ts_abs.add(abs_off)

                        cur.execute(
                            "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                            "kind,value_text,value_num,ts_kind_guess,ts_human,"
                            "ts_classification,ts_reason,ts_source_url,ts_field_name) "
                            "VALUES (?,?,?,?, 'ts',NULL,?,?,?,?,?,?,?)",
                            (
                                pno,
                                finding.offset,
                                abs_off,
                                cluster_id,
                                str(finding.value),  # Store as text to preserve format
                                finding.format_type,
                                finding.human_readable,
                                finding.classification.value,
                                finding.reason,
                                finding.source_url,
                                finding.field_name,
                            ),
                        )
                        rows_batch.append(
                            (
                                pno,
                                finding.offset,
                                abs_off,
                                cluster_id,
                                "ts",
                                "",
                                str(finding.value),  # Store as text
                                finding.format_type,
                                finding.human_readable,
                                finding.classification.value,
                                finding.reason,
                                finding.source_url,
                                finding.field_name,
                            )
                        )

                    # Also find hex timestamps (processed separately)
                    try:
                        from timestamp_patterns import find_hex_timestamp_candidates

                        hex_timestamps = find_hex_timestamp_candidates(page)
                        for offset, hex_val, kind, human in hex_timestamps:
                            abs_off = abs_page_start + offset
                            if abs_off in seen_ts_abs:
                                continue
                            seen_ts_abs.add(abs_off)

                            # Hex timestamps stored in value_num (now TEXT)
                            cur.execute(
                                "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                                "kind,value_text,value_num,ts_kind_guess,ts_human,"
                                "ts_classification,ts_reason,ts_source_url,ts_field_name) "
                                "VALUES (?,?,?,?, 'ts',NULL,?,?,?,?,?,NULL,NULL)",
                                (
                                    pno,
                                    offset,
                                    abs_off,
                                    cluster_id,
                                    hex_val,  # Store hex string like "5b994548" in value_num
                                    kind,  # e.g., "unix_sec_hex"
                                    human,
                                    "confirmed_timestamp",  # Hex timestamps in Message-IDs are likely valid
                                    "Hexadecimal timestamp format",
                                ),
                            )
                            rows_batch.append(
                                (
                                    pno,
                                    offset,
                                    abs_off,
                                    cluster_id,
                                    "ts",
                                    "",  # value_text empty
                                    hex_val,  # Hex string in value_num (now TEXT field)
                                    kind,
                                    human,
                                    "confirmed_timestamp",
                                    "Hexadecimal timestamp format",
                                    None,
                                    None,
                                )
                            )
                    except Exception:
                        pass  # Hex timestamp support not available

                else:
                    # Fallback to old system if V2 not available
                    for off, epoch, kind, human in find_timestamps(page):
                        abs_off = abs_page_start + off
                        if abs_off in seen_ts_abs:
                            continue
                        seen_ts_abs.add(abs_off)
                        cur.execute(
                            "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                            "kind,value_text,value_num,ts_kind_guess,ts_human,"
                            "ts_classification,ts_reason,ts_source_url,ts_field_name) "
                            "VALUES (?,?,?,?, 'ts',NULL,?,?,?,?,?,?,?)",
                            (
                                pno,
                                off,
                                abs_off,
                                cluster_id,
                                epoch,
                                kind,
                                human,
                                "ambiguous",
                                "fallback classifier",
                                None,
                                None,
                            ),
                        )
                        rows_batch.append(
                            (
                                pno,
                                off,
                                abs_off,
                                cluster_id,
                                "ts",
                                "",
                                epoch,
                                kind,
                                human,
                                "ambiguous",
                                "fallback classifier",
                                None,
                                None,
                            )
                        )

                # URLs
                for off, url in find_urls(page):
                    abs_off = abs_page_start + off
                    if abs_off in seen_url_abs:
                        continue
                    seen_url_abs.add(abs_off)
                    cur.execute(
                        "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                        "kind,value_text,value_num,ts_kind_guess,ts_human,"
                        "ts_classification,ts_reason,ts_source_url,ts_field_name) "
                        "VALUES (?,?,?,?, 'url',?,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                        (pno, off, abs_off, cluster_id, url),
                    )
                    rows_batch.append(
                        (
                            pno,
                            off,
                            abs_off,
                            cluster_id,
                            "url",
                            url,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    )

                # Text
                for off, txt in find_text_runs(page, min_len=8):
                    abs_off = abs_page_start + off
                    if abs_off in seen_txt_abs:
                        continue
                    seen_txt_abs.add(abs_off)
                    cur.execute(
                        "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                        "kind,value_text,value_num,ts_kind_guess,ts_human,"
                        "ts_classification,ts_reason,ts_source_url,ts_field_name) "
                        "VALUES (?,?,?,?, 'text',?,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                        (pno, off, abs_off, cluster_id, txt),
                    )
                    rows_batch.append(
                        (
                            pno,
                            off,
                            abs_off,
                            cluster_id,
                            "text",
                            txt,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    )

                # Protobufs → JSONL
                if (
                    not args.no_protobuf
                    and maybe_decode_protobuf is not None
                    and to_json is not None
                ):
                    for off, blob in find_blob_candidates(page, min_len=16):
                        abs_off = abs_page_start + off
                        # Deduplicate by absolute offset
                        if abs_off in seen_pb_abs:
                            continue

                        parsed = maybe_decode_protobuf(blob, max_depth=4)
                        if not parsed:
                            continue

                        # Filter out garbage protobufs (optional, only if module available)
                        if PROTOBUF_FILTER_AVAILABLE:
                            keep, reason = should_keep_protobuf(parsed)
                            if not keep:
                                continue  # Skip this protobuf - it's noise

                            # Analyze for timestamps (add to output for context)
                            analysis = analyze_protobuf_for_timestamps(parsed)
                        else:
                            analysis = None

                        seen_pb_abs.add(abs_off)
                        jtxt = to_json(parsed, pretty=(not args.no_pretty_protobuf))

                        # Prepare timestamp analysis for database
                        if analysis and analysis.get("has_timestamps"):
                            has_ts = 1
                            ts_count = analysis["timestamp_count"]
                            ts_fields = json.dumps(analysis["timestamps"])
                        else:
                            has_ts = 0
                            ts_count = 0
                            ts_fields = None

                        # Insert into database with timestamp info
                        cur.execute(
                            "INSERT INTO carved_protobufs(parent_abs_offset,page_no,abs_offset,"
                            "json_pretty,has_timestamps,timestamp_count,timestamp_fields) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (abs_off, pno, abs_off, jtxt, has_ts, ts_count, ts_fields),
                        )

                        # Build JSONL output with optional timestamp analysis
                        jsonl_entry = {
                            "parent_abs_offset": abs_off,
                            "page_no": pno,
                            "abs_offset": abs_off,
                            "protobuf": (
                                json.loads(jtxt)
                                if not args.no_pretty_protobuf
                                else parsed
                            ),
                        }

                        # Add timestamp analysis to JSONL if available
                        if has_ts:
                            jsonl_entry["timestamp_analysis"] = {
                                "timestamp_count": ts_count,
                                "timestamps": json.loads(ts_fields),
                            }

                        append_jsonl(out_jsonl_pb, jsonl_entry)

                # Batch commit every BATCH_SIZE pages
                if (pno + 1) % BATCH_SIZE == 0:
                    db.commit()
                    append_csv_batch(out_csv_main, rows_batch)
                    rows_batch.clear()

            except Exception as e:
                print(
                    f"\n{ANSI_YELLOW}[!] Error processing page {pno}: {e}{ANSI_RESET}",
                    file=sys.stderr,
                )
                # Commit what we have so far to avoid losing all progress
                db.commit()
                append_csv_batch(out_csv_main, rows_batch)
                rows_batch.clear()
                # Continue with next page
                continue

    # Final commit for remaining rows
    if rows_batch:
        append_csv_batch(out_csv_main, rows_batch)
        rows_batch.clear()

    # Finalize DB (single file, no WAL/SHM)
    db.commit()
    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    db.commit()
    db.close()

    print("\n─────────────────────────────────────────────\n")

    # Print classification summary
    if classification_stats:
        print(classification_stats.get_summary())
        print()

    print(f"{ANSI_GREEN}[✓]{ANSI_RESET} Exported:")
    print(f"   • {out_sqlite}")
    print(f"   • {out_csv_main}")
    print(f"   • {out_jsonl_pb}")
    print(f"\n{ANSI_GREEN}[✓]{ANSI_RESET} Done.")


if __name__ == "__main__":
    main()
