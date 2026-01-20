#!/usr/bin/env python3
"""
ff_jsonlz4_salvage_classify.py

Scan a directory for Firefox JSONLZ4 (mozLz4) files, salvage decompressed JSON from
carved/partial blobs, export sessions and bookmarks to CSV, and (optionally) classify
and move/copy/link each source file into category subfolders under --out.

Categories created under --out:
  Firefox/Bookmarks/
  Firefox/Sessions/
  Firefox/Other/      (valid mozlz4 JSON but not parsable sessions/bookmarks)
  Unknown/            (JSON parsed, but not clearly Firefox session/bookmarks)
  Corrupt/            (no salvageable mozlz4 segments)

Usage:
  python3 ff_jsonlz4_salvage_classify.py INDIR --out OUTDIR [--classify move|copy|link|none]
                                               [--glob "**/*.jsonlz4"] [--append]
                                               [--preserve-tree] [--dry-run]
Requires: lz4
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import json
import os
import shutil
import string
import struct
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lz4.block

from mars.report_modules.progress_interface import get_progress
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

MAGIC = b"mozLz40\x00"
MAX_PARTIAL_OUTPUT = 512 * 1024 * 1024  # 512MB safety cap for ctypes fallback

_LZ4_PARTIAL: tuple[ctypes.CDLL, Any] | bool | None = None


def _load_lz4_partial() -> tuple[ctypes.CDLL, Any] | None:
    """Locate liblz4 and wire up LZ4_decompress_safe_partial."""
    global _LZ4_PARTIAL
    if _LZ4_PARTIAL is False:
        return None
    if isinstance(_LZ4_PARTIAL, tuple):
        return _LZ4_PARTIAL

    candidates: list[str] = []
    libname = ctypes.util.find_library("lz4")
    if libname:
        candidates.append(libname)

    if sys.platform == "darwin":
        candidates.extend(
            [
                "/opt/homebrew/lib/liblz4.dylib",
                "/usr/local/opt/lz4/lib/liblz4.dylib",
                "/usr/local/lib/liblz4.dylib",
            ]
        )
    elif sys.platform.startswith("linux"):
        candidates.extend(
            [
                "/usr/lib/x86_64-linux-gnu/liblz4.so",
                "/usr/lib64/liblz4.so",
                "/usr/lib/liblz4.so",
            ]
        )
    elif sys.platform.startswith("win"):
        try:
            # Try bundled DLL first (e.g., resources/windows/lib/lz4.dll)
            base = Path(__file__).resolve().parents[3]
            candidates.append(str(base / "resources" / "windows" / "lib" / "lz4.dll"))
        except Exception:
            pass
        candidates.append("lz4.dll")

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            func = getattr(lib, "LZ4_decompress_safe_partial", None)
            if func is None:
                continue
            func.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            func.restype = ctypes.c_int
            _LZ4_PARTIAL = (lib, func)
            return _LZ4_PARTIAL
        except Exception:
            continue

    _LZ4_PARTIAL = False
    return None


def _lz4_partial_decompress(block: bytes, expected_size: int | None, *, allow_partial: bool = False) -> bytes | None:
    """
    Attempt decompression via liblz4's LZ4_decompress_safe_partial.

    When allow_partial is True, progressively lower the requested output size so that
    truncated segments still yield partial JSON instead of failing.
    """
    if not expected_size or expected_size <= 0:
        return None
    if expected_size > MAX_PARTIAL_OUTPUT:
        return None

    entry = _load_lz4_partial()
    if not entry:
        return None

    _, func = entry
    src = ctypes.create_string_buffer(block, len(block))
    max_output = expected_size
    dest = (ctypes.c_char * max_output)()

    targets: list[int] = [expected_size]
    if allow_partial:
        shrink_factors = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
        targets.extend(max(1, expected_size // factor) for factor in shrink_factors if expected_size // factor > 0)
        step = max(1024, expected_size // 32)
        for target in range(expected_size - step, 0, -step):
            targets.append(target)
        targets.append(len(block))

    tried: set[int] = set()
    for target in targets:
        if target in tried or target > max_output:
            continue
        tried.add(target)
        try:
            res = func(
                ctypes.byref(src),
                dest,
                len(block),
                target,
                max_output,
            )
        except Exception:
            continue
        if res >= 0:
            return bytes(dest[:res])
    return None


# ---------- Types ----------


@dataclass
class SalvageResult:
    had_mozlz4: bool
    segments_ok: int
    segments_total: int
    kinds_found: set[str]  # {'session','bookmarks','other','unknown'}


# ---------- mozLz4 salvage ----------


def find_mozlz4_segments(raw: bytes):
    """
    Return list of (offset_after_magic, expected_size, block_bytes) for each mozlz4 segment.
    Layout: [8-byte magic][4-byte LE uncompressed size][raw LZ4 block...]
    """
    out = []
    i = 0
    mlen = len(MAGIC)
    while True:
        j = raw.find(MAGIC, i)
        if j < 0 or j + mlen + 4 > len(raw):
            break
        # 4-byte little-endian uncompressed size
        try:
            u_size = struct.unpack_from("<I", raw, j + mlen)[0]
        except struct.error:
            break
        block = raw[j + mlen + 4 :]
        out.append((j + mlen, u_size, block))
        i = j + 1
    return out


def decompress_with_tail_trimming(block: bytes, expected_size: int | None, max_attempts: int = 10) -> bytes:
    """
    Decompress a mozlz4 raw LZ4 block. If it fails, progressively trim the tail and retry.
    We try with expected_size first (if provided), then without, then with trimming.
    """
    last_exc = None

    # 1) Try with expected size (most reliable)
    if expected_size:
        try:
            return lz4.block.decompress(block, uncompressed_size=expected_size)
        except Exception as e:
            last_exc = e
        partial = _lz4_partial_decompress(block, expected_size)
        if partial is not None:
            return partial

    # 2) Try without size (library may figure it out for intact blocks)
    try:
        return lz4.block.decompress(block)
    except Exception as e:
        last_exc = e

    # 3) Tail-trim and retry (without size hint once we've trimmed data)
    end = len(block)
    for _ in range(max_attempts):
        if end <= 0:
            break

        trim = 1 if end <= 64 else min(4096, max(end // 8, 32))
        end -= trim
        if end <= 0:
            break

        # With a trimmed block the original expected_size is no longer trustworthy.
        try:
            return lz4.block.decompress(block[:end])
        except Exception as e:
            last_exc = e
        if expected_size:
            partial = _lz4_partial_decompress(block[:end], expected_size, allow_partial=True)
            if partial is not None:
                return partial

    if expected_size:
        partial = _lz4_partial_decompress(block, expected_size, allow_partial=True)
        if partial is not None:
            return partial

    raise last_exc or lz4.block.LZ4BlockError("Decompression failed")


def json_best_effort(data: bytes):
    """Parse JSON; if it fails, progressively sanitize and trim partial structures."""
    original = data.decode("utf-8", "replace")

    def _try_load(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            return err
        except Exception as err:
            return err

    def _strip_control_chars(raw: str) -> str:
        if not any(ord(ch) < 32 and ch not in "\t\n\r" for ch in raw):
            return raw
        return "".join(ch for ch in raw if ord(ch) >= 32 or ch in "\t\n\r")

    def _escape_invalid_backslashes(raw: str) -> str:
        allowed = {'"', "\\", "/", "b", "f", "n", "r", "t"}
        hexdigits = set(string.hexdigits)
        out_chars: list[str] = []
        i = 0
        length = len(raw)
        while i < length:
            ch = raw[i]
            if ch != "\\":
                out_chars.append(ch)
                i += 1
                continue
            if i + 1 >= length:
                out_chars.append("\\\\")
                i += 1
                continue
            nxt = raw[i + 1]
            if nxt in allowed:
                out_chars.append("\\" + nxt)
                i += 2
                continue
            if nxt == "u":
                hex_part = raw[i + 2 : i + 6]
                if len(hex_part) == 4 and all(c in hexdigits for c in hex_part):
                    out_chars.append("\\u" + hex_part)
                    i += 6
                    continue
                out_chars.append("\\\\u")
                i += 2
                continue
            out_chars.append("\\\\" + nxt)
            i += 2
        return "".join(out_chars)

    def _escape_inline_controls(raw: str) -> str:
        out: list[str] = []
        in_str = False
        esc = False
        for ch in raw:
            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                    continue
                if ch == "\\":
                    out.append("\\")
                    esc = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_str = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                if ord(ch) < 32:
                    out.append(f"\\u{ord(ch):04x}")
                    continue
                out.append(ch)
            else:
                out.append(ch)
                if ch == '"':
                    in_str = True
                    esc = False
        return "".join(out)

    def _close_unbalanced(raw: str) -> str:
        stack: list[str] = []
        out: list[str] = []
        in_str = False
        esc = False
        for ch in raw:
            out.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    stack.append("}")
                elif ch == "[":
                    stack.append("]")
                elif ch in "}]" and stack:
                    stack.pop()
        if in_str:
            out.append('"')
        while stack:
            out.append(stack.pop())
        return "".join(out)

    def _repair_tail(raw: str) -> str:
        allowed_tail = set(']}0123456789"eE.-tTrRuUaAlLfFsnN')
        s = raw.rstrip()
        while s and s[-1] not in allowed_tail:
            s = s[:-1].rstrip()
        while s.endswith((",", ":")):
            s = s[:-1].rstrip()
        if s.count('"') % 2 == 1:
            s = s.rsplit('"', 1)[0].rstrip(", :")
        if not s:
            return raw
        # Drop trailing keys that never received a value (missing colon/value)
        while s:
            stripped = s.rstrip()
            last_quote = stripped.rfind('"')
            last_colon = stripped.rfind(":")
            if last_quote > last_colon:
                s = stripped[:last_quote].rstrip(", :")
            else:
                break
        if not s:
            return raw
        return _close_unbalanced(s)

    # Stage 1: raw text
    result = _try_load(original)
    if not isinstance(result, Exception):
        return result

    work = _strip_control_chars(original)
    if work is not original:
        result = _try_load(work)
        if not isinstance(result, Exception):
            return result

    work2 = _escape_invalid_backslashes(work)
    if work2 is not work:
        work = work2
        result = _try_load(work)
        if not isinstance(result, Exception):
            return result

    work3 = _escape_inline_controls(work)
    if work3 is not work:
        work = work3
        result = _try_load(work)
        if not isinstance(result, Exception):
            return result

    # Stage 2: truncate to last balanced top-level closing brace/bracket
    depth = 0
    in_str = False
    esc = False
    cut = None
    for idx, ch in enumerate(work):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth = max(0, depth - 1)
                if depth == 0:
                    cut = idx + 1
    if cut:
        candidate = work[:cut]
        result = _try_load(candidate)
        if not isinstance(result, Exception):
            return result
        work = candidate

    repaired = _repair_tail(work)
    result = _try_load(repaired)
    if not isinstance(result, Exception):
        return result

    # Stage 3: iteratively trim at reported error position
    candidate = repaired
    for _ in range(16):
        err = _try_load(candidate)
        if not isinstance(err, Exception):
            return err
        if not isinstance(err, json.JSONDecodeError):
            break
        pos = err.pos
        if pos <= 0 or pos >= len(candidate):
            break
        back = pos
        while back > 0 and candidate[back - 1] not in ",[{":
            back -= 1
        candidate = candidate[:back].rstrip(", ")
        candidate = _repair_tail(candidate)

    # Final attempt (will raise if parsing still impossible)
    return json.loads(candidate)


def salvage_json_objects(
    path: Path,
) -> tuple[SalvageResult, list[tuple[int, str, object | None]]]:
    """
    Attempt to salvage JSON objects from mozLz4 segments in ``path``.

    Returns a tuple of (SalvageResult, [(segment_index, kind, json_obj), ...]).
    """
    try:
        raw = path.read_bytes()
    except Exception:
        logger.warning(f"read failed for {path}\n{traceback.format_exc()}")
        return (
            SalvageResult(
                had_mozlz4=False,
                segments_ok=0,
                segments_total=0,
                kinds_found=set(),
            ),
            [],
        )

    segs = find_mozlz4_segments(raw)
    out: list[tuple[int, str, object | None]] = []
    kinds_found: set[str] = set()

    for idx, (_, expected_size, block) in enumerate(segs):
        try:
            decompressed = decompress_with_tail_trimming(block, expected_size)
        except Exception:
            # Debug level - salvage failures are expected for carved/truncated files
            logger.debug(f"{path} seg#{idx}: salvage failed\n{traceback.format_exc()}")
            continue

        try:
            obj = json_best_effort(decompressed)
            kind = guess_kind(obj)
        except Exception:
            # Debug level - JSON decode failures are expected for corrupted data
            logger.debug(f"{path} seg#{idx}: JSON decode failed\n{traceback.format_exc()}")
            continue

        out.append((idx, kind, obj))
        kinds_found.add(kind)

    return (
        SalvageResult(
            had_mozlz4=bool(segs),
            segments_ok=len(out),
            segments_total=len(segs),
            kinds_found=kinds_found,
        ),
        out,
    )


# ---------- Extractors ----------


def iso_from_epoch_ms(ms):
    try:
        return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()
    except Exception:
        return None


def iso_from_epoch_us(us):
    try:
        return datetime.fromtimestamp(us / 1_000_000, tz=UTC).isoformat()
    except Exception:
        return None


def extract_session_rows(obj) -> list[dict]:
    rows = []

    def iter_windows():
        for key in ("windows", "_closedWindows"):
            wins = obj.get(key)
            if not isinstance(wins, list):
                continue
            for win in wins:
                if isinstance(win, dict):
                    yield win

    for win in iter_windows():
        tabs = win.get("tabs") or []
        if not isinstance(tabs, list):
            continue
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            idx = max(0, int(tab.get("index", 1)) - 1)
            entries = tab.get("entries") or []
            if not isinstance(entries, list):
                entries = []
            e = entries[idx] if (0 <= idx < len(entries)) else (entries[-1] if entries else {})
            if not isinstance(e, dict):
                e = {}
            url = e.get("url")
            ts = e.get("lastAccessed") or tab.get("lastAccessed")
            if url:
                rows.append(
                    {
                        "url": url,
                        "last_accessed_ms": (ts if isinstance(ts, (int, float)) else None),
                        "last_accessed_iso": (iso_from_epoch_ms(ts) if isinstance(ts, (int, float)) else None),
                    }
                )
    return rows


def _walk_bookmarks(node, bag, seen):
    if not isinstance(node, dict):
        return
    guid = node.get("guid")
    if guid and guid in seen:
        return
    if guid:
        seen.add(guid)
    t = node.get("type")
    if t in ("text/x-moz-place", "bookmark"):
        bag.append(
            {
                "title": node.get("title"),
                "uri": node.get("uri"),
                "dateAdded_us": node.get("dateAdded"),
                "dateAdded_iso": (
                    iso_from_epoch_us(node.get("dateAdded"))
                    if isinstance(node.get("dateAdded"), (int, float))
                    else None
                ),
                "lastModified_us": node.get("lastModified"),
                "lastModified_iso": (
                    iso_from_epoch_us(node.get("lastModified"))
                    if isinstance(node.get("lastModified"), (int, float))
                    else None
                ),
            }
        )
    for ch in node.get("children", []) or []:
        if isinstance(ch, dict):
            _walk_bookmarks(ch, bag, seen)


def extract_bookmark_rows(obj) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    if isinstance(obj, dict):
        _walk_bookmarks(obj, out, seen)
        root = obj.get("root")
        if isinstance(root, dict):
            _walk_bookmarks(root, out, seen)
        roots = obj.get("roots")
        if isinstance(roots, dict):
            for v in roots.values():
                if isinstance(v, dict):
                    _walk_bookmarks(v, out, seen)
    return out


def _normalize_csv_value(value):
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    return value


def extract_telemetry_event_rows(obj) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(obj, dict):
        return rows
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return rows

    app = obj.get("application") or {}
    base = {
        "doc_type": obj.get("type"),
        "client_id": obj.get("clientId"),
        "session_id": payload.get("sessionId"),
        "subsession_id": payload.get("subsessionId"),
        "reason": payload.get("reason"),
        "creation_date": obj.get("creationDate"),
        "application_name": app.get("name"),
        "application_version": app.get("version"),
        "application_channel": app.get("channel"),
    }

    process_lists: list[tuple[str, object]] = []

    events = payload.get("events")
    if isinstance(events, dict):
        process_lists.extend(events.items())
    elif isinstance(events, list):
        process_lists.append(("default", events))

    processes = payload.get("processes")
    if isinstance(processes, dict):
        for proc_name, proc_data in processes.items():
            if not isinstance(proc_data, dict):
                continue
            proc_events = proc_data.get("events")
            if isinstance(proc_events, dict):
                for sub_name, plist in proc_events.items():
                    label = f"{proc_name}:{sub_name}"
                    process_lists.append((label, plist))
            elif isinstance(proc_events, list):
                process_lists.append((proc_name, proc_events))

    for process, plist in process_lists:
        if not isinstance(plist, list):
            continue
        for entry in plist:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            timestamp = entry[0]
            category = entry[1] if len(entry) > 1 else None
            method = entry[2] if len(entry) > 2 else None
            obj_name = entry[3] if len(entry) > 3 else None
            value = entry[4] if len(entry) > 4 else None
            extra = entry[5] if len(entry) > 5 else None

            if isinstance(value, (dict, list)):
                value = _normalize_csv_value(value)

            extra_json = None
            if isinstance(extra, (dict, list)):
                extra_json = _normalize_csv_value(extra)
            elif extra is not None:
                extra_json = str(extra)

            rows.append(
                {
                    **base,
                    "process": process,
                    "timestamp_offset_ms": (int(timestamp) if isinstance(timestamp, (int, float)) else None),
                    "category": category,
                    "method": method,
                    "object": obj_name,
                    "value": value,
                    "extra_json": extra_json,
                }
            )
    return rows


def extract_telemetry_main_rows(obj) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(obj, dict):
        return rows
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return rows

    app = obj.get("application") or {}
    base = {
        "doc_type": obj.get("type"),
        "client_id": obj.get("clientId"),
        "session_id": payload.get("sessionId"),
        "subsession_id": payload.get("subsessionId"),
        "reason": payload.get("reason"),
        "creation_date": obj.get("creationDate"),
        "application_name": app.get("name"),
        "application_version": app.get("version"),
        "application_channel": app.get("channel"),
    }

    def add_row(metric_type, metric_name, value, *, process=None, key=None):
        if metric_name is None:
            return
        metric_name = str(metric_name)
        process_str = str(process) if process is not None else None
        key_str = str(key) if key is not None else None
        rows.append(
            {
                **base,
                "metric_type": metric_type,
                "metric_name": metric_name,
                "metric_value": _normalize_csv_value(value),
                "process": process_str,
                "key": key_str,
            }
        )

    simple = payload.get("simpleMeasurements")
    if isinstance(simple, dict):
        for name, value in simple.items():
            add_row("simple_measurement", name, value)

    info = payload.get("info")
    if isinstance(info, dict):
        for name, value in info.items():
            add_row("info", name, value)

    hist = payload.get("histograms")
    if isinstance(hist, dict):
        for name, value in hist.items():
            add_row("histogram", name, value)

    keyed_hist = payload.get("keyedHistograms")
    if isinstance(keyed_hist, dict):
        for name, key_map in keyed_hist.items():
            if not isinstance(key_map, dict):
                continue
            for key, value in key_map.items():
                add_row("keyed_histogram", name, value, key=key)

    processes = payload.get("processes")
    if isinstance(processes, dict):
        for proc_name, proc_data in processes.items():
            if not isinstance(proc_data, dict):
                continue
            scalars = proc_data.get("scalars")
            if isinstance(scalars, dict):
                for name, value in scalars.items():
                    add_row("scalar", name, value, process=proc_name)
            keyed = proc_data.get("keyedScalars")
            if isinstance(keyed, dict):
                for name, key_map in keyed.items():
                    if not isinstance(key_map, dict):
                        continue
                    for key, value in key_map.items():
                        add_row("keyed_scalar", name, value, process=proc_name, key=key)
            proc_hist = proc_data.get("histograms")
            if isinstance(proc_hist, dict):
                for name, value in proc_hist.items():
                    add_row("histogram", name, value, process=proc_name)
            proc_keyed_hist = proc_data.get("keyedHistograms")
            if isinstance(proc_keyed_hist, dict):
                for name, key_map in proc_keyed_hist.items():
                    if not isinstance(key_map, dict):
                        continue
                    for key, value in key_map.items():
                        add_row(
                            "keyed_histogram",
                            name,
                            value,
                            process=proc_name,
                            key=key,
                        )

    return rows


def guess_kind(obj) -> str:
    if isinstance(obj, dict):
        doc_type = obj.get("type")
        if isinstance(obj.get("windows"), list):
            return "session"
        if "roots" in obj or "root" in obj:
            return "bookmarks"
        if doc_type == "event":
            return "telemetry_event"
        if doc_type == "main":
            return "telemetry_main"
        # Looks like Firefox but not classified
        return "other"
    return "unknown"


# ---------- Classification / file ops ----------

CATS = {
    "sessions": Path("Firefox/Sessions"),
    "bookmarks": Path("Firefox/Bookmarks"),
    "telemetry": Path("Firefox/Telemetry"),
    "other": Path("Firefox/Other"),
    "unknown": Path("Unknown"),
    "corrupt": Path("Corrupt"),
}


def decide_category(sr: SalvageResult) -> str:
    if sr.segments_ok == 0:
        return "corrupt"
    # preference order if multiple kinds found
    if "session" in sr.kinds_found:
        return "sessions"
    if "bookmarks" in sr.kinds_found:
        return "bookmarks"
    if "telemetry_event" in sr.kinds_found or "telemetry_main" in sr.kinds_found:
        return "telemetry"
    if "other" in sr.kinds_found:
        return "other"
    return "unknown"


def classify_stage(
    src: Path,
    out_root: Path,
    category: str,
    classify: str,
    preserve_tree: bool,
    in_root: Path,
    dry_run: bool = False,
):
    """
    classify: 'move' | 'copy' | 'link' | 'none'
    """
    if classify == "none":
        return

    dest_base = out_root / CATS[category]
    if preserve_tree:
        # replicate path relative to input root
        try:
            rel = src.relative_to(in_root)
        except Exception:
            rel = src.name
        dest_path = dest_base / rel
    else:
        dest_path = dest_base / src.name

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(f"[dry-run] {classify} {src} -> {dest_path}")
        return

    if classify == "move":
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # If moving across devices, fallback to copy+remove
        try:
            shutil.move(str(src), str(dest_path))
        except Exception:
            shutil.copy2(str(src), str(dest_path))
            try:  # noqa: SIM105
                src.unlink()
            except Exception:
                pass
    elif classify == "copy":
        shutil.copy2(str(src), str(dest_path))
    elif classify == "link":
        try:
            if dest_path.exists():
                dest_path.unlink()
            os.link(src, dest_path)  # hardlink
        except OSError:
            # fallback to symlink if hardlink fails (e.g., cross-device)
            try:
                if dest_path.exists():
                    dest_path.unlink()
                os.symlink(src, dest_path)  # noqa: PTH211
            except OSError:
                # Windows requires admin privileges or Developer Mode for symlinks
                # Fall back to copying the file
                if dest_path.exists():
                    dest_path.unlink()
                shutil.copy2(str(src), str(dest_path))
    else:
        pass


# ---------- CLI ----------


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from root.glob(pattern)


def main():
    ap = argparse.ArgumentParser(description="Salvage Firefox JSONLZ4, export CSVs, and classify files.")
    ap.add_argument("indir", type=Path, help="Input directory (scanned recursively) for *.jsonlz4")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(),
        help="Output directory (default: current)",
    )
    ap.add_argument("--glob", default="**/*.*lz4*", help='Glob pattern (default: "**/*.*lz4*")')
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSVs instead of overwriting",
    )
    ap.add_argument(
        "--classify",
        choices=["move", "copy", "link", "none"],
        default="link",
        help="Move/copy/link each source into category folders under --out (default: link)",
    )
    ap.add_argument(
        "--preserve-tree",
        action="store_true",
        help="Preserve original relative path under category folders",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned file operations without performing them",
    )
    ap.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source JSONLZ4 files after successful CSV export (saves disk space)",
    )
    args = ap.parse_args()

    inroot: Path = args.indir
    outdir: Path = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV writers
    session_csv = outdir / "firefox_sessions.csv"
    bookmarks_csv = outdir / "firefox_bookmarks.csv"
    telemetry_events_csv = outdir / "firefox_telemetry_events.csv"
    telemetry_main_csv = outdir / "firefox_telemetry_main_metrics.csv"

    session_fields = [
        "source_file",
        "segment",
        "url",
        "last_accessed_iso",
        "last_accessed_ms",
    ]
    bookmark_fields = [
        "source_file",
        "segment",
        "title",
        "uri",
        "dateAdded_iso",
        "dateAdded_us",
        "lastModified_iso",
        "lastModified_us",
    ]
    telemetry_event_fields = [
        "source_file",
        "segment",
        "doc_type",
        "client_id",
        "session_id",
        "subsession_id",
        "reason",
        "creation_date",
        "application_name",
        "application_version",
        "application_channel",
        "process",
        "timestamp_offset_ms",
        "category",
        "method",
        "object",
        "value",
        "extra_json",
    ]
    telemetry_main_fields = [
        "source_file",
        "segment",
        "doc_type",
        "client_id",
        "session_id",
        "subsession_id",
        "reason",
        "creation_date",
        "application_name",
        "application_version",
        "application_channel",
        "metric_type",
        "process",
        "metric_name",
        "key",
        "metric_value",
    ]

    session_mode = "a" if args.append and session_csv.exists() else "w"
    bookmarks_mode = "a" if args.append and bookmarks_csv.exists() else "w"
    telemetry_events_mode = "a" if args.append and telemetry_events_csv.exists() else "w"
    telemetry_main_mode = "a" if args.append and telemetry_main_csv.exists() else "w"

    with (
        session_csv.open(session_mode, newline="", encoding="utf-8") as session_f,
        bookmarks_csv.open(bookmarks_mode, newline="", encoding="utf-8") as bookmarks_f,
        telemetry_events_csv.open(telemetry_events_mode, newline="", encoding="utf-8") as telemetry_events_f,
        telemetry_main_csv.open(telemetry_main_mode, newline="", encoding="utf-8") as telemetry_main_f,
    ):
        session_w = csv.DictWriter(session_f, fieldnames=session_fields)
        bookmarks_w = csv.DictWriter(bookmarks_f, fieldnames=bookmark_fields)
        telemetry_event_w = csv.DictWriter(telemetry_events_f, fieldnames=telemetry_event_fields)
        telemetry_main_w = csv.DictWriter(telemetry_main_f, fieldnames=telemetry_main_fields)
        if session_mode == "w":
            session_w.writeheader()
        if bookmarks_mode == "w":
            bookmarks_w.writeheader()
        if telemetry_events_mode == "w":
            telemetry_event_w.writeheader()
        if telemetry_main_mode == "w":
            telemetry_main_w.writeheader()

        files = list(iter_files(inroot, args.glob))
        if not files:
            logger.info(f"no files matched {args.glob!r} under {inroot}")

        # Set up progress reporting
        progress = get_progress()
        if progress:
            progress.set_total(len(files))

        total_segments = total_session_rows = total_bookmark_rows = 0
        total_event_rows = total_main_rows = 0
        processed = 0
        session_seen: set[tuple[str, int | None]] = set()
        bookmark_seen: set[tuple[str | None, int | None, int | None]] = set()

        for path in files:
            processed += 1
            sr, seg_objs = salvage_json_objects(path)

            for seg_idx, kind, obj in seg_objs:
                if kind == "session":
                    rows = extract_session_rows(obj)
                    for r in rows:
                        url = r.get("url")
                        if not isinstance(url, str):
                            continue
                        url = sys.intern(url)
                        ts = r.get("last_accessed_ms")
                        ts_int = int(ts) if isinstance(ts, (int, float)) else None
                        key: tuple[str, int | None] = (url, ts_int)
                        if key in session_seen:
                            continue
                        session_seen.add(key)
                        session_w.writerow({"source_file": str(path), "segment": seg_idx, **r})
                        total_session_rows += 1
                elif kind == "bookmarks":
                    rows = extract_bookmark_rows(obj)
                    for r in rows:
                        uri = r.get("uri")
                        last_mod = r.get("lastModified_us")
                        date_added = r.get("dateAdded_us")
                        try:
                            last_mod_key = int(last_mod) if last_mod is not None else None
                        except (TypeError, ValueError):
                            last_mod_key = None
                        try:
                            date_added_key = int(date_added) if date_added is not None else None
                        except (TypeError, ValueError):
                            date_added_key = None
                        if isinstance(uri, str):
                            uri = sys.intern(uri)
                        bkey = (uri, last_mod_key, date_added_key)
                        if uri and bkey in bookmark_seen:
                            continue
                        bookmark_seen.add(bkey)
                        bookmarks_w.writerow({"source_file": str(path), "segment": seg_idx, **r})
                        total_bookmark_rows += 1
                elif kind == "telemetry_event":
                    rows = extract_telemetry_event_rows(obj)
                    for r in rows:
                        telemetry_event_w.writerow({"source_file": str(path), "segment": seg_idx, **r})
                        total_event_rows += 1
                elif kind == "telemetry_main":
                    rows = extract_telemetry_main_rows(obj)
                    for r in rows:
                        telemetry_main_w.writerow({"source_file": str(path), "segment": seg_idx, **r})
                        total_main_rows += 1
                    event_rows = extract_telemetry_event_rows(obj)
                    for r in event_rows:
                        telemetry_event_w.writerow({"source_file": str(path), "segment": seg_idx, **r})
                        total_event_rows += 1
                else:
                    # 'other' or 'unknown' → no CSV output
                    pass

            total_segments += sr.segments_ok

            category = decide_category(sr)

            classify_stage(
                path,
                outdir,
                category,
                args.classify,
                args.preserve_tree,
                inroot,
                dry_run=args.dry_run,
            )

            # Delete source file if requested (saves disk space)
            # Only delete if successfully processed and not already moved
            if (
                args.delete_source
                and not args.dry_run
                and sr.segments_ok > 0
                and args.classify != "move"
                and path.exists()
            ):
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"failed to delete {path}: {e}")

            # Update progress
            if progress:
                progress.advance()

    # Make sure category directories exist even if empty (handy for post-run review)
    for p in CATS.values():
        (outdir / p).mkdir(parents=True, exist_ok=True)

    # Silent completion - stats tracked by processor


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\n[interrupt] aborted by user")
        sys.exit(130)
