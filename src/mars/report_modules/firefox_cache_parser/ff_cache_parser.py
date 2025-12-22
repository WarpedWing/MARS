#!/usr/bin/env python3
"""
Firefox cache2 entry parser.

Given a directory containing Firefox cache2 files (e.g. cache2/entries plus
index/index.log) this script walks every SHA1-named entry file, peels off the
metadata trailer, and emits a CSV summarising the HTTP artefacts we can recover.

The parser works purely off the entry files – the index is optional and currently
unused, but we validate its presence so future enhancements can pull last-used
times or frecency if available.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import mimetypes
import re
import string
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mars.report_modules.progress_interface import get_progress
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

URL_PATTERN = re.compile(rb"https?://[^\s\x00\"'>]+")
HEX_CHARS = set(string.hexdigits)
INDEX_HEADER_SIZE = 16
INDEX_HASH_SIZE = 4
INDEX_RECORD_SIZE = 41

FLAG_INITIALIZED = 0x80000000
FLAG_ANONYMOUS = 0x40000000
FLAG_REMOVED = 0x20000000
FLAG_DIRTY = 0x10000000
FLAG_FRESH = 0x08000000
FLAG_PINNED = 0x04000000
FLAG_HAS_ALT = 0x02000000
FLAG_RESERVED = 0x01000000
FLAG_FILESIZE_MASK = 0x00FFFFFF


@dataclass(slots=True)
class CacheEntry:
    name: str
    path: Path
    body: bytes
    metadata_raw: bytes
    metadata: dict[str, bytes | list[bytes]]
    index_info: dict[str, Any] | None = None


def is_hex_entry(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if len(name) != 40:
        return False
    return all(ch in HEX_CHARS for ch in name)


def iter_entries(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if is_hex_entry(path):
            yield path


def decode_index_record(chunk: bytes) -> dict[str, Any]:
    hash_hex = chunk[:20].hex().upper()
    frecency = int.from_bytes(chunk[20:24], "big", signed=False)
    origin_attrs_hash = int.from_bytes(chunk[24:32], "big", signed=False)
    on_start = int.from_bytes(chunk[32:34], "big", signed=False)
    on_stop = int.from_bytes(chunk[34:36], "big", signed=False)
    content_type = chunk[36]
    flags = int.from_bytes(chunk[37:41], "big", signed=False)
    info = {
        "hash": hash_hex,
        "frecency": frecency,
        "origin_attrs_hash": origin_attrs_hash,
        "on_start_time": on_start,
        "on_stop_time": on_stop,
        "content_type": content_type,
        "flags": flags,
        "file_size_kb": flags & FLAG_FILESIZE_MASK,
        "flag_initialized": bool(flags & FLAG_INITIALIZED),
        "flag_anonymous": bool(flags & FLAG_ANONYMOUS),
        "flag_removed": bool(flags & FLAG_REMOVED),
        "flag_dirty": bool(flags & FLAG_DIRTY),
        "flag_fresh": bool(flags & FLAG_FRESH),
        "flag_pinned": bool(flags & FLAG_PINNED),
        "flag_has_alt_data": bool(flags & FLAG_HAS_ALT),
        "flag_reserved": bool(flags & FLAG_RESERVED),
    }
    return info


def parse_index_file(path: Path) -> dict[str, dict[str, Any]]:
    data = path.read_bytes()
    if len(data) < INDEX_HEADER_SIZE + INDEX_HASH_SIZE:
        return {}
    body = data[INDEX_HEADER_SIZE:-INDEX_HASH_SIZE]
    return _parse_index_blob(body, path)


def parse_journal_file(path: Path) -> dict[str, dict[str, Any]]:
    data = path.read_bytes()
    if len(data) <= INDEX_HASH_SIZE:
        return {}
    body = data[:-INDEX_HASH_SIZE]
    return _parse_index_blob(body, path)


def _parse_index_blob(blob: bytes, origin: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    record_count = len(blob) // INDEX_RECORD_SIZE
    for i in range(record_count):
        chunk = blob[i * INDEX_RECORD_SIZE : (i + 1) * INDEX_RECORD_SIZE]
        rec = decode_index_record(chunk)
        result[rec["hash"]] = rec
    return result


def load_index_data(indir: Path, *, index_path: Path | None, journal_path: Path | None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    candidate_index = index_path or indir / "index"
    if candidate_index.exists():
        try:
            records = parse_index_file(candidate_index)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"failed to parse index file {candidate_index}: {exc}")
    candidate_journal = journal_path or indir / "index.log"
    if candidate_journal.exists():
        try:
            journal_records = parse_journal_file(candidate_journal)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"failed to parse index log {candidate_journal}: {exc}")
        else:
            records.update(journal_records)
    return records


def parse_metadata(raw: bytes) -> dict[str, bytes | list[bytes]]:
    """Firefox stores metadata as a NULL-delimited stream of key/value strings."""
    parts = raw.split(b"\x00")
    while parts and parts[-1] == b"":
        parts.pop()
    fields: dict[str, bytes | list[bytes]] = {}

    start = 0
    if parts and len(parts) % 2 == 1:
        # Empirically, odd counts indicate an unlabeled security-info blob.
        fields["security-info"] = parts[0]
        start = 1

    i = start
    while i + 1 < len(parts):
        key = parts[i].decode("utf-8", "replace")
        value = parts[i + 1]
        if not key:
            break
        if key in fields:
            existing = fields[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                fields[key] = [existing, value]
        else:
            fields[key] = value
        i += 2
    return fields


def display_text(value: bytes | list[bytes] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(display_text(v) or "" for v in value)
    try:
        text = bytes(value).decode("utf-8")
        if all((0x20 <= ord(ch) <= 0x7E) or ch in "\r\n\t" for ch in text):
            return text
        # Allow header-style text even if it contains lower control chars.
        stripped = text.replace("\r", "").replace("\n", "")
        if stripped and stripped.isprintable():
            return text
    except UnicodeDecodeError:
        pass
    return f"base64:{base64.b64encode(value).decode('ascii')}"


def parse_entry(path: Path) -> CacheEntry | None:
    data = path.read_bytes()
    if len(data) <= 4:
        return None
    meta_len = int.from_bytes(data[-4:], "big", signed=False)
    if meta_len <= 0 or meta_len > len(data) - 4:
        return None
    metadata_raw = data[-4 - meta_len : -4]
    body = data[: -4 - meta_len]
    metadata = parse_metadata(metadata_raw)
    return CacheEntry(
        name=path.name.upper(),
        path=path,
        body=body,
        metadata_raw=metadata_raw,
        metadata=metadata,
    )


def parse_response_head(head: str | None) -> tuple[str | None, dict[str, str]]:
    if not head:
        return None, {}
    lines = [line for line in head.splitlines() if line]
    if not lines:
        return None, {}
    status = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return status, headers


def extract_url(metadata: dict[str, bytes | list[bytes]], raw: bytes) -> str | None:
    cached_keys = ("key", "cache-key", "metadata-key")
    for k in cached_keys:
        if k in metadata:
            value = metadata[k]
            if isinstance(value, list):
                value = value[0]
            txt = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
            return txt.lstrip(":")
    match = URL_PATTERN.search(raw)
    if match:
        return match.group(0).decode("utf-8", "replace")
    return None


def maybe_decompress(body: bytes, headers: dict[str, str]) -> tuple[bytes, str]:
    encoding = headers.get("content-encoding", "").lower()
    looks_gzip = body.startswith(b"\x1f\x8b")
    if "gzip" in encoding or looks_gzip:
        for decoder in (
            lambda data: gzip.decompress(data),
            lambda data: zlib.decompress(data, 16 + zlib.MAX_WBITS),
        ):
            try:
                return decoder(body), "gzip"
            except Exception:
                continue
        return body, "gzip-error"
    return body, encoding or ""


# Magic byte signatures for common file types
# Format: (signature_bytes, offset, extension)
MAGIC_SIGNATURES: list[tuple[bytes, int, str]] = [
    # Images
    (b"\xff\xd8\xff", 0, ".jpg"),  # JPEG/JFIF
    (b"GIF87a", 0, ".gif"),  # GIF87a
    (b"GIF89a", 0, ".gif"),  # GIF89a
    (b"\x89PNG\r\n\x1a\n", 0, ".png"),  # PNG
    (b"RIFF", 0, ".webp"),  # WebP (needs secondary check for WEBP)
    (b"BM", 0, ".bmp"),  # BMP
    (b"\x00\x00\x01\x00", 0, ".ico"),  # ICO
    (b"\x00\x00\x02\x00", 0, ".cur"),  # CUR (cursor)
    # Fonts
    (b"wOFF", 0, ".woff"),  # WOFF
    (b"wOF2", 0, ".woff2"),  # WOFF2
    (b"OTTO", 0, ".otf"),  # OpenType with CFF
    (b"\x00\x01\x00\x00", 0, ".ttf"),  # TrueType (also used by some OTF)
    # Documents
    (b"%PDF", 0, ".pdf"),  # PDF
    (b"PK\x03\x04", 0, ".zip"),  # ZIP (also docx, xlsx, etc.)
    # Audio/Video
    (b"ID3", 0, ".mp3"),  # MP3 with ID3 tag
    (b"\xff\xfb", 0, ".mp3"),  # MP3 frame sync
    (b"\xff\xfa", 0, ".mp3"),  # MP3 frame sync
    (b"OggS", 0, ".ogg"),  # Ogg container
    (b"fLaC", 0, ".flac"),  # FLAC
    (b"\x1a\x45\xdf\xa3", 0, ".webm"),  # WebM/Matroska
    # ISO base media file format (ftyp box at offset 4)
    (b"ftyp", 4, ".mp4"),  # Generic - will be refined by brand check
    # Web content (check these after binary formats)
    (b"<?xml", 0, ".xml"),  # XML
    (b"\xef\xbb\xbf<?xml", 0, ".xml"),  # XML with BOM
]

# AVIF/HEIC brand identifiers (checked when ftyp is detected)
FTYP_BRANDS: dict[bytes, str] = {
    b"avif": ".avif",
    b"avis": ".avif",  # AVIF sequence
    b"heic": ".heic",
    b"heix": ".heic",
    b"hevc": ".heic",
    b"mif1": ".heif",  # Generic HEIF
    b"mp41": ".mp4",
    b"mp42": ".mp4",
    b"isom": ".mp4",
    b"M4A ": ".m4a",
    b"M4V ": ".m4v",
}

# Text-based signatures (checked on decoded content)
TEXT_SIGNATURES: list[tuple[bytes, str]] = [
    (b"<!DOCTYPE html", ".html"),
    (b"<!doctype html", ".html"),
    (b"<html", ".html"),
    (b"<HTML", ".html"),
    (b"<?xml", ".xml"),
    (b"<svg", ".svg"),  # SVG without XML declaration
    (b'{"', ".json"),  # JSON object
    (b"[{", ".json"),  # JSON array of objects
    (b"[\n", ".json"),  # JSON array
]


def guess_extension_from_magic(body: bytes) -> str | None:
    """Detect file type from magic bytes/file signature.

    Returns extension if detected, None otherwise.
    """
    if not body or len(body) < 4:
        return None

    # Check binary signatures first
    for signature, offset, ext in MAGIC_SIGNATURES:
        if len(body) <= offset + len(signature):
            continue
        if body[offset : offset + len(signature)] != signature:
            continue
        # Special case: WebP needs secondary WEBP check at offset 8
        if ext == ".webp" and (len(body) <= 12 or body[8:12] != b"WEBP"):
            continue  # Not actually WebP, skip
        # Special case: ftyp container - check brand for AVIF/HEIC/MP4
        if ext == ".mp4" and len(body) > 12:
            brand = body[8:12]
            if brand in FTYP_BRANDS:
                return FTYP_BRANDS[brand]
        return ext

    # Check for text-based content (HTML, JSON, etc.)
    # Look at first 512 bytes, stripped of whitespace
    head = body[:512].lstrip()
    for signature, ext in TEXT_SIGNATURES:
        if head.startswith(signature):
            return ext

    # Check for JavaScript (common patterns)
    if head.startswith(
        (
            b"(function",
            b"function ",
            b"var ",
            b"const ",
            b"let ",
            b'"use strict"',
            b"'use strict'",
            b"//",
            b"/*",
        )
    ):
        return ".js"

    # Check for CSS - heuristic based on common patterns
    css_starters = (
        b"@charset",
        b"@import",
        b"@media",
        b"@font-face",
        b"body{",
        b"body {",
    )
    has_css_structure = b"{" in head[:100] and b":" in head[:100] and b";" in head[:200]
    if head.startswith(css_starters) or has_css_structure:
        css_patterns = [
            b"font-",
            b"color:",
            b"margin:",
            b"padding:",
            b"display:",
            b"background",
        ]
        is_not_html = not head.startswith(b"<") and b"{" in head[:200]
        if is_not_html and any(p in head for p in css_patterns):
            return ".css"

    return None


def guess_dump_extension(content_type: str | None, body: bytes | None = None) -> str:
    """Determine file extension from magic bytes first, then Content-Type header.

    Args:
        content_type: HTTP Content-Type header value
        body: File body bytes for magic byte detection

    Returns:
        File extension including leading dot (e.g., ".jpg", ".bin")
    """
    # Try magic byte detection first (more reliable than headers)
    if body:
        magic_ext = guess_extension_from_magic(body)
        if magic_ext:
            return magic_ext

    # Fall back to Content-Type header
    if not content_type:
        return ".bin"
    ctype = content_type.split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(ctype)
    if ext == ".jpe":  # normalise rare alias
        return ".jpg"
    return ext or ".bin"


def build_row(
    entry: CacheEntry,
    dump_dir: Path | None = None,
    dump_mode: str = "copy",
) -> dict[str, object]:
    meta = entry.metadata
    response_head = display_text(meta.get("response-head"))
    status_line, headers = parse_response_head(response_head)
    url = extract_url(meta, entry.metadata_raw)
    body, body_encoding = maybe_decompress(entry.body, headers)
    shasum = hashlib.sha256(entry.body).hexdigest()
    dump_path: Path | None = None
    if dump_dir and (body or dump_mode == "link"):
        # Use magic byte detection first, then fall back to Content-Type
        ext = guess_dump_extension(headers.get("content-type"), body)
        dest = dump_dir / f"{entry.name}{ext}"
        try:
            # For .bin files (unknown type), always use symlinks to save space
            # For known file types, copy the decompressed content
            if ext == ".bin" or dump_mode == "link":
                # Symlink to original cache file (fallback to copy on Windows if symlink fails)
                target = entry.path.resolve()
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                try:
                    dest.symlink_to(target)
                except OSError:
                    # Windows requires admin privileges or Developer Mode for symlinks
                    # Fall back to copying the file
                    import shutil

                    shutil.copy2(target, dest)
            else:
                # Copy decompressed body for known file types
                dest.write_bytes(body)
            dump_path = dest
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to materialize body for {entry.path}: {exc}")

    index_info = entry.index_info
    origin_hash_hex = (
        f"{index_info['origin_attrs_hash']:016x}"
        if index_info and index_info["origin_attrs_hash"] is not None
        else None
    )
    file_size_kb = index_info["file_size_kb"] if index_info else None
    file_size_bytes = file_size_kb * 1024 if file_size_kb is not None else None
    flags = index_info["flags"] if index_info else None

    row = {
        "entry_name": entry.name,
        "source_file": str(entry.path),
        "url": url,
        "request_method": display_text(meta.get("request-method")),
        "status_line": status_line,
        "content_type": headers.get("content-type"),
        "content_length_header": headers.get("content-length"),
        "content_encoding": headers.get("content-encoding"),
        "response_date": headers.get("date"),
        "expires": headers.get("expires"),
        "last_modified": headers.get("last-modified"),
        "etag": headers.get("etag"),
        "cache_control": headers.get("cache-control"),
        "server": headers.get("server"),
        "age": headers.get("age"),
        "security_info_b64": display_text(meta.get("security-info")),
        "net_response_time_onstart": display_text(meta.get("net-response-time-onstart")),
        "net_response_time_onstop": display_text(meta.get("net-response-time-onstop")),
        "metadata_size": len(entry.metadata_raw),
        "body_size": len(entry.body),
        "decompressed_size": len(body),
        "body_encoding": body_encoding,
        "body_sha256": shasum,
        "response_head": response_head,
        "original_response_headers": display_text(meta.get("original-response-headers")),
        "index_present": bool(index_info),
        "index_frecency": index_info["frecency"] if index_info else None,
        "index_origin_attrs_hash": origin_hash_hex,
        "index_onstart_time": index_info["on_start_time"] if index_info else None,
        "index_onstop_time": index_info["on_stop_time"] if index_info else None,
        "index_content_type_id": index_info["content_type"] if index_info else None,
        "index_file_size_kb": file_size_kb,
        "index_file_size_bytes": file_size_bytes,
        "index_flags": flags,
        "index_flag_initialized": (index_info["flag_initialized"] if index_info else None),
        "index_flag_anonymous": index_info["flag_anonymous"] if index_info else None,
        "index_flag_removed": index_info["flag_removed"] if index_info else None,
        "index_flag_dirty": index_info["flag_dirty"] if index_info else None,
        "index_flag_fresh": index_info["flag_fresh"] if index_info else None,
        "index_flag_pinned": index_info["flag_pinned"] if index_info else None,
        "index_flag_has_alt_data": (index_info["flag_has_alt_data"] if index_info else None),
        "index_flag_reserved": index_info["flag_reserved"] if index_info else None,
        "dump_body_path": str(dump_path) if dump_path else None,
    }
    return row


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "entry_name",
        "source_file",
        "url",
        "request_method",
        "status_line",
        "content_type",
        "content_length_header",
        "content_encoding",
        "response_date",
        "expires",
        "last_modified",
        "etag",
        "cache_control",
        "server",
        "age",
        "security_info_b64",
        "net_response_time_onstart",
        "net_response_time_onstop",
        "metadata_size",
        "body_size",
        "decompressed_size",
        "body_encoding",
        "body_sha256",
        "response_head",
        "original_response_headers",
        "index_present",
        "index_frecency",
        "index_origin_attrs_hash",
        "index_onstart_time",
        "index_onstop_time",
        "index_content_type_id",
        "index_file_size_kb",
        "index_file_size_bytes",
        "index_flags",
        "index_flag_initialized",
        "index_flag_anonymous",
        "index_flag_removed",
        "index_flag_dirty",
        "index_flag_fresh",
        "index_flag_pinned",
        "index_flag_has_alt_data",
        "index_flag_reserved",
        "dump_body_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Firefox cache2 entry files and emit a CSV summary.")
    parser.add_argument("indir", type=Path, help="Directory containing cache2 entries")
    parser.add_argument("--out", type=Path, default=Path(), help="Output directory (default: cwd)")
    parser.add_argument(
        "--index-file",
        type=Path,
        help="Explicit path to the Firefox cache index file (defaults to INDIR/index)",
    )
    parser.add_argument(
        "--journal-file",
        type=Path,
        help="Explicit path to index.log (journal) (defaults to INDIR/index.log)",
    )
    parser.add_argument(
        "--dump-bodies",
        action="store_true",
        help="When set, write decoded response bodies to firefox_cache_bodies/",
    )
    parser.add_argument(
        "--dump-mode",
        choices=["copy", "link"],
        default="link",
        help="When dumping bodies, either copy bytes or create symlinks back to the original cache entry (default: copy)",
    )
    args = parser.parse_args()

    indir = args.indir
    if not indir.exists():
        parser.error(f"{indir} does not exist")
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    dump_dir = outdir / "firefox_cache_bodies" if args.dump_bodies else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    dump_mode = args.dump_mode

    index_records = load_index_data(indir, index_path=args.index_file, journal_path=args.journal_file)

    cache_entries = list(iter_entries(indir))
    if not cache_entries:
        logger.debug(f"[warn] No cache entry files detected under {indir}")

    # Set up progress reporting
    progress = get_progress()
    if progress:
        progress.set_total(len(cache_entries))

    rows: list[dict[str, object]] = []
    failures = 0
    for path in cache_entries:
        try:
            parsed = parse_entry(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[warn] Failed to parse {path}: {exc}")
            failures += 1
            if progress:
                progress.advance()
            continue
        if not parsed:
            failures += 1
            if progress:
                progress.advance()
            continue
        parsed.index_info = index_records.get(parsed.name)
        rows.append(build_row(parsed, dump_dir=dump_dir, dump_mode=dump_mode))

        # Update progress
        if progress:
            progress.advance()

    csv_path = outdir / "firefox_cache_entries.csv"
    write_csv(rows, csv_path)


if __name__ == "__main__":
    main()
