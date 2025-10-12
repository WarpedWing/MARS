#!/usr/bin/env python3
"""
PhotoRec Forensic Pipeline (macOS 10.13.6–friendly)

- Recursively scans a PhotoRec output tree for:
  • SQLite DBs: .sqlite / .db / .PLSQL (and gzipped SQLite)
  • FSEvents logs (magic: 1SLD, 2SLD, 3SLD; gz or raw)

- For SQLite:
  • If inside .gz, decompress to temp and test as DB
  • Try open; if fails, try sqlite3 '.recover'
  • If still fails, dump bytes to ./unsalvageable/<file>/
  • Classify by tables: Firefox (moz_), Powerlog (schema-driven via ./Schemas/), cfurl (cfurl_), Chrome (visit_source)
  • Copies only (never move) into ./categorized_SQLite/<Category>/
  • Copies matching -wal/-shm, merges WAL into the copied DB, then removes -wal/-shm from output
  • Powerlog copies are named *.PLSQL

- FSEvents:
  • If file (or gz) begins with 1SLD/2SLD/3SLD, copy into ./FSEvents/

- Final step:
  • Merge each Powerlog schema into ./combined/<Schema>/RecoveredPowerlog.<Schema>.PLSQL

Requirements:
  • Python 3 with stdlib
  • sqlite3 CLI on PATH (for '.recover' step)

This script NEVER modifies the PhotoRec source files.
"""
import argparse
import bz2
import contextlib
import gzip
import platform
import shutil
import sqlite3
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from schema_manager import (
    compare_schema,
    load_all_schemas,
    load_schema_catalog,
    save_missing_schema_report,
)

# ---------- Configuration ----------
SRC_DIR = Path()
OUTPUT_ROOT = Path("./Parsed")
OUT_SQL_DIR = OUTPUT_ROOT / "categorized_SQLite"
OUT_FSE_DIR = OUTPUT_ROOT / "FSEvents"
OUT_WIFI_LOGS = OUTPUT_ROOT / "WiFiLogs"
OUT_SYSLOGS = OUTPUT_ROOT / "SystemLogs"
OUT_UNSALV = OUTPUT_ROOT / "unsalvageable"
OUT_RECOVERED_SQLITE = OUTPUT_ROOT / "recovered_sqlite"


# Path to gzrecover and sqlite_dissect binaries (OS-specific)
def get_binary_path(binary_name: str) -> Path | None:
    """Get OS-specific binary path for packaging.

    Supports macOS and Windows with bundled binaries.
    For Linux or other platforms, only checks PATH.
    """
    # First check if binary is in PATH
    path_binary = shutil.which(binary_name)
    if path_binary:
        return Path(path_binary)

    # Determine OS-specific subdirectory (only macOS and Windows have bundled binaries)
    system = platform.system().lower()
    if system == "darwin":
        os_dir = "macos"
        ext = ""
    elif system == "windows":
        os_dir = "windows"
        ext = ".exe"
    else:
        # Linux and other platforms: no bundled binaries available
        # User must have binaries in PATH or install them manually
        return None

    # Check in resources/bin/<os>/ directory
    resources_path = (
        Path(__file__).parent.parent
        / "resources"
        / "bin"
        / os_dir
        / f"{binary_name}{ext}"
    )
    return resources_path if resources_path.exists() else None


GZRECOVER_PATH = get_binary_path("gzrecover")
SQLITE_DISSECT_PATH = get_binary_path("sqlite_dissect")

CATEGORIES = {
    "Firefox": ["moz_"],
    "Powerlog": ["PLBatteryAgent"],
    "cfurl": ["cfurl_"],
    "Chrome": ["visit_source"],
}
ALL_CATS = list(CATEGORIES.keys()) + ["Uncategorized"]

SQLITE_SUFFIXES = {".sqlite", ".db", ".plsql"}  # case-insensitive
FSEVENT_MAGICS = (b"1SLD", b"2SLD", b"3SLD")
WIFI_MAGIC_RAW = bytes.fromhex("425A6839314159265359")  # BZh91AY&SY
SYSLOG_MAGIC_RAW = bytes.fromhex("1F8B0800000000000013E5")


def configure_paths(input_dir: Path, output_root: Path):
    """Update global paths after CLI argument parsing."""
    global SRC_DIR, OUTPUT_ROOT, OUT_SQL_DIR, OUT_FSE_DIR, OUT_WIFI_LOGS, OUT_SYSLOGS, OUT_UNSALV
    global OUT_RECOVERED_SQLITE, OUT_CORRUPT_GZ

    SRC_DIR = input_dir
    OUTPUT_ROOT = output_root
    OUT_SQL_DIR = OUTPUT_ROOT / "categorized_SQLite"
    OUT_FSE_DIR = OUTPUT_ROOT / "FSEvents"
    OUT_WIFI_LOGS = OUTPUT_ROOT / "WiFiLogs"
    OUT_SYSLOGS = OUTPUT_ROOT / "SystemLogs"
    OUT_UNSALV = OUTPUT_ROOT / "unsalvageable"
    OUT_RECOVERED_SQLITE = OUTPUT_ROOT / "recovered_sqlite"
    OUT_CORRUPT_GZ = OUTPUT_ROOT / "corrupt_gz"


def prepare_output_dirs():
    """Ensure all output directories exist for the current configuration."""
    for p in [
        OUT_SQL_DIR,
        OUT_FSE_DIR,
        OUT_WIFI_LOGS,
        OUT_SYSLOGS,
        OUT_UNSALV,
        OUT_RECOVERED_SQLITE,
        OUT_CORRUPT_GZ,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    for cat in ALL_CATS:
        (OUT_SQL_DIR / cat).mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def looks_like_unix_timestamp(value):
    """Check if a value is a plausible Unix timestamp."""
    if not isinstance(value, (int, float)):
        return False
    return (
        946684800 <= value <= time.time() + 5 * 365 * 24 * 3600
    )  # between 2000 and ~5 years ahead


def archive_recovered_temp(temp_path: Path):
    """Move a recovered temp DB out of corrupt_gz once it has been categorized."""
    try:
        if temp_path and temp_path.exists():
            dest = OUT_RECOVERED_SQLITE / temp_path.name
            shutil.move(temp_path, dest)
            print(f"    Archived recovered temp to {dest}")
    except Exception as e:
        print(f"    Warning: Could not archive recovered temp {temp_path}: {e}")


def in_output_tree(path: Path) -> bool:
    """Skip scanning our own outputs to avoid re-processing."""
    try:
        path.resolve().relative_to(OUTPUT_ROOT)
        return True
    except ValueError:
        return False


def is_sqlite_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SQLITE_SUFFIXES


def read_first_bytes(path: Path, n: int = 100) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def gz_read_first_bytes(path: Path, n: int = 100) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read(n)


def gz_contains_sqlite(path: Path) -> bool:
    try:
        b = gz_read_first_bytes(path, 16)
        return b.startswith(b"SQLite format 3")
    except Exception:
        return False


def file_is_fsevent(path: Path) -> bool:
    try:
        return read_first_bytes(path, 4).startswith(FSEVENT_MAGICS)
    except Exception:
        return False


def gz_is_fsevent(path: Path) -> bool:
    try:
        return gz_read_first_bytes(path, 4).startswith(FSEVENT_MAGICS)
    except Exception:
        return False


def gz_is_wifi_log(path: Path) -> bool:
    """Detect Wi-Fi logs that are sometimes bzip payloads but stored as .gz."""
    try:
        raw = read_first_bytes(path, len(WIFI_MAGIC_RAW))
        if raw.startswith(WIFI_MAGIC_RAW):
            return True
    except Exception:
        pass

    try:
        decomp = gz_read_first_bytes(path, len(WIFI_MAGIC_RAW))
        return decomp.startswith(WIFI_MAGIC_RAW)
    except Exception:
        return False


def gz_is_system_log(path: Path) -> bool:
    """Detect classic macOS system.log gzip files via gzip header."""
    try:
        header = read_first_bytes(path, len(SYSLOG_MAGIC_RAW))
        return header.startswith(SYSLOG_MAGIC_RAW)
    except Exception:
        return False


def bz_is_wifi_log(path: Path) -> bool:
    """Detect Wi-Fi logs stored as bz2 archives."""
    try:
        header = read_first_bytes(path, len(WIFI_MAGIC_RAW))
        if header.startswith(WIFI_MAGIC_RAW):
            return True
    except Exception:
        pass

    # As a fallback, try a small decompress to cover truncated headers.
    try:
        with bz2.open(path, "rb") as f:
            sample = f.read(len(WIFI_MAGIC_RAW))
            return sample.startswith(WIFI_MAGIC_RAW)
    except Exception:
        return False


def name_indicates_wifi_log(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.startswith("wifi.log") or lowered.startswith("wifi_")


def name_indicates_system_log(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.startswith("system.log") or lowered.startswith("system_")


@lru_cache(maxsize=1)
def get_schema_catalog_cached():
    return load_schema_catalog()


@lru_cache(maxsize=1)
def get_powerlog_tables_lower() -> set[str]:
    tables: set[str] = set()
    for entry in get_schema_catalog_cached().values():
        tables.update(entry["tables_lower"])
    return tables


POWERLOG_SCHEMA_MIN_MATCH_RATIO = 0.1
POWERLOG_SCHEMA_MIN_MATCH_COUNT = 5
POWERLOG_UNKNOWN_SLUG = "unknown"


def match_powerlog_schema(table_names: list[str]) -> tuple[str, str] | None:
    """Best-effort match of a Powerlog DB to a known schema."""
    table_set = {t.lower() for t in table_names}
    catalog = get_schema_catalog_cached()

    best_slug = None
    best_score = (0.0, 0)  # (coverage, matches)

    for slug, entry in catalog.items():
        schema_tables = entry["tables_lower"]
        if not schema_tables:
            continue
        matches = len(table_set & schema_tables)
        if not matches:
            continue
        coverage = matches / len(schema_tables)
        score = (coverage, matches)
        if score > best_score:
            best_score = score
            best_slug = slug

    if best_slug:
        coverage, matches = best_score
        if (
            coverage >= POWERLOG_SCHEMA_MIN_MATCH_RATIO
            or matches >= POWERLOG_SCHEMA_MIN_MATCH_COUNT
        ):
            return best_slug, catalog[best_slug]["label"]

    return None


def list_tables(db_path: Path) -> list[str]:
    """Return list of table names or raise."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def detect_categories(tables: list[str]) -> list[str]:
    tables_lower = [t.lower() for t in tables]
    joined = " ".join(tables_lower)
    dests = [
        cat for cat, kws in CATEGORIES.items() if any(k.lower() in joined for k in kws)
    ]
    return dests or ["Uncategorized"]


def copy_with_companions(
    src_db: Path, dst_folder: Path, dst_name: str | None = None
) -> Path:
    """
    Copy a DB plus -wal/-shm companions from source folder to dst_folder.
    If dst_name is provided, rename the DB accordingly (e.g., Powerlog -> .PLSQL).
    Returns path to copied DB.
    """
    dst_folder.mkdir(parents=True, exist_ok=True)
    final_name = dst_name if dst_name else src_db.name
    dst_db = dst_folder / final_name
    shutil.copy2(src_db, dst_db)

    for ext in ("-wal", "-shm"):
        comp_src = src_db.with_name(src_db.name + ext)
        if comp_src.exists():
            shutil.copy2(comp_src, dst_folder / (final_name + ext))
    return dst_db


def merge_wal_in_place(dst_db: Path) -> bool:
    """
    If dst_db has a -wal file alongside it, merge WAL into dst_db using sqlite backup.
    Remove -wal/-shm after success. Returns True if merged or no wal present.
    """
    wal = dst_db.with_name(dst_db.name + "-wal")
    shm = dst_db.with_name(dst_db.name + "-shm")
    if not wal.exists():
        return True  # nothing to merge

    tmpdir = Path(tempfile.mkdtemp(prefix="walmerge_"))
    try:
        tmp_db = tmpdir / dst_db.name
        shutil.copy2(dst_db, tmp_db)
        shutil.copy2(wal, tmpdir / (dst_db.name + "-wal"))
        if shm.exists():
            shutil.copy2(shm, tmpdir / (dst_db.name + "-shm"))

        merged = tmpdir / "merged.db"
        src = sqlite3.connect(str(tmp_db))
        dst = sqlite3.connect(str(merged))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        shutil.move(str(merged), str(dst_db))
        # Clean extras in output
        for p in (wal, shm):
            if p.exists():
                p.unlink()
        return True
    except Exception as e:
        print(f"    Warning: WAL merge failed for {dst_db.name}: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def try_sqlite_recover_to_newdb(bad_db: Path) -> Path | None:
    """
    Use sqlite3 '.recover' to salvage a broken DB.
    Returns path to a new recovered DB (in OUT_RECOVERED), or None on failure.
    """
    sql_out = OUT_RECOVERED_SQLITE / (bad_db.stem + ".sql")
    new_db = OUT_RECOVERED_SQLITE / (bad_db.stem + "_recovered.sqlite")

    try:
        with sql_out.open("wb") as f:
            subprocess.run(
                ["sqlite3", str(bad_db), ".recover"],
                check=True,
                stdout=f,
                stderr=subprocess.DEVNULL,
            )
        subprocess.run(["sqlite3", str(new_db)], input=sql_out.read_bytes(), check=True)
        return new_db
    except Exception:
        return None
    finally:
        # keep sql_out; it can be useful
        pass


def try_sqlite_dissect(db_path: Path) -> Path | None:
    """
    Attempt deep recovery using sqlite-dissect CLI (if installed).
    Creates a carved SQLite DB in OUT_RECOVERED.
    Returns the new DB path or None if failed/unavailable.
    """
    if not SQLITE_DISSECT_PATH or not SQLITE_DISSECT_PATH.exists():
        print(
            f"    Warning: sqlite_dissect CLI not found (looked for {SQLITE_DISSECT_PATH})."
        )
        return None

    dst_dir = OUT_RECOVERED_SQLITE / (db_path.stem + "_dissect")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_db = dst_dir / (db_path.stem + "_dissect.sqlite")

    cmd = [
        str(SQLITE_DISSECT_PATH),
        str(db_path),
        "-d",
        str(dst_dir),
        "-e",
        "sqlite",
        "--carve",
        "--carve-freelists",
        "--disable-strict-format-checking",
    ]

    print(f"    Running sqlite_dissect CLI on {db_path.name} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # sqlite_dissect typically writes <prefix>_output.sqlite; rename it if necessary
        for f in dst_dir.glob("*.sqlite"):
            f.rename(dst_db)
            print(f"    [SUCCESS] sqlite_dissect recovered {dst_db.name}")
            return dst_db
    except FileNotFoundError:
        print("    [WARNING] sqlite_dissect CLI not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"    Warning: sqlite_dissect failed ({e.returncode}): {e.stderr[:300]}")
    except Exception as e:
        print(f"    Warning: sqlite_dissect error: {e}")
    return None


def dump_unreadable(src_file: Path):
    """Dump an unreadable DB into a human-viewable hexdump."""
    dstdir = OUT_UNSALV / (src_file.name + "_dump")
    dstdir.mkdir(parents=True, exist_ok=True)
    out_txt = dstdir / "dump.txt"

    try:
        with (
            src_file.open("rb") as f,
            out_txt.open("w", encoding="utf-8", errors="ignore") as out,
        ):
            while chunk := f.read(4096):
                # printable subset
                text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                out.write(text + "\n")
        print(f"    Dumped to {out_txt}")
    except Exception as e:
        print(f"    Warning: Dump failed: {e}")


def ensure_plsql_name(base_name: str) -> str:
    """
    For Powerlog, ensure destination filename ends with .PLSQL (preserve stem).
    """
    stem = Path(base_name).stem
    return stem + ".PLSQL"


# ---------- Main Processing ----------


def process_sqlite_candidate(src_db: Path):
    """Open/repair/classify/copy/merge a sqlite candidate (already a plain file)."""
    tables: list[str] = []
    try:
        tables = list_tables(src_db)
        ok = True
    except Exception:
        ok = False

    recovered_db = None
    if not ok:
        # Try sqlite .recover
        recovered_db = try_sqlite_recover_to_newdb(src_db)
        if recovered_db is not None:
            try:
                tables = list_tables(recovered_db)
                ok = True
            except Exception:
                ok = False

    if not ok:
        # Try sqlite_dissect if available
        dissected = try_sqlite_dissect(src_db)
        if dissected is not None:
            try:
                tables = list_tables(dissected)
                ok = True
                recovered_db = dissected
            except Exception:
                ok = False

    if not ok:
        # Still unreadable → dump bytes to text
        print(f"  Error: Unreadable after all recovery attempts: {src_db}")
        dump_unreadable(src_db)
        return
    tables_lower = [t.lower() for t in tables]
    if "lost_and_found" in tables_lower:
        print(
            f"  Warning: {src_db.name} contains 'lost_and_found' table; keeping for manual analysis only."
        )
        return

    # Determine categories
    dests = detect_categories(tables)
    powerlog_schema_slug: str | None = None
    powerlog_schema_label = "Unknown"

    powerlog_match = match_powerlog_schema(tables)
    if powerlog_match:
        powerlog_schema_slug, powerlog_schema_label = powerlog_match
        if "Powerlog" not in dests:
            dests.append("Powerlog")
    else:
        if "Powerlog" in dests:
            dests = [cat for cat in dests if cat != "Powerlog"]

    # Copy into each category, merging WAL if present
    for cat in dests:
        display_bucket = cat
        dst_folder = OUT_SQL_DIR / cat
        dst_name = src_db.name

        if cat == "Powerlog":
            if not powerlog_schema_slug:
                # This case should not be reached, but as a safeguard, explicitly skip copying to the root
                # Powerlog directory if a schema match is missing.
                print(
                    f"  [WARNING] Skipping Powerlog copy for {src_db.name} (no schema match)."
                )
                continue
            schema_folder = powerlog_schema_slug
            dst_folder = dst_folder / schema_folder
            dst_name = ensure_plsql_name(src_db.name)
            display_bucket = f"{cat}/{powerlog_schema_label}"

        src_for_copy = recovered_db if recovered_db else src_db
        dst_db = copy_with_companions(src_for_copy, dst_folder, dst_name=dst_name)
        # Merge WAL into the copied DB and clean
        merged_ok = merge_wal_in_place(dst_db)
        if not merged_ok:
            pass
        print(f"  Copied to {display_bucket}: {dst_db.name}")


def process_gz(path: Path):
    """Handle .gz: FSEvents? or SQLite inside? Copy accordingly."""
    # Raw magic detection for known non-SQLite logs
    if gz_is_wifi_log(path) or name_indicates_wifi_log(path):
        OUT_WIFI_LOGS.mkdir(parents=True, exist_ok=True)
        print(f"  WiFi log (gz): {path.name}")
        shutil.copy2(path, OUT_WIFI_LOGS / path.name)
        return

    if gz_is_system_log(path) or name_indicates_system_log(path):
        OUT_SYSLOGS.mkdir(parents=True, exist_ok=True)
        print(f"  System log (gz): {path.name}")
        shutil.copy2(path, OUT_SYSLOGS / path.name)
        return

    # FSEvents gz?
    if gz_is_fsevent(path):
        OUT_FSE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  FSEvents (gz): {path.name}")
        shutil.copy2(path, OUT_FSE_DIR / path.name)
        return

    # SQLite inside gz?
    try:
        if gz_contains_sqlite(path):
            tmp = Path(tempfile.mkdtemp(prefix="gzsqlite_"))
            try:
                out = tmp / Path(path.stem).name  # drop .gz
                with gzip.open(path, "rb") as g, out.open("wb") as f:
                    shutil.copyfileobj(g, f)
                print(f"  Decompressed SQLite from {path.name}")
                process_sqlite_candidate(out)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return
    except (gzip.BadGzipFile, OSError, EOFError) as e:
        print(f"  Warning: Gzip decompression failed: {path.name} ({e})")

        OUT_CORRUPT_GZ.mkdir(parents=True, exist_ok=True)

        if GZRECOVER_PATH and GZRECOVER_PATH.exists():
            try:
                recovered_base = path.stem + "_gzrecover.recovered"
                recovered_out = OUT_CORRUPT_GZ / recovered_base
                print(
                    f"    [INFO] Attempting to recover with gzrecover: {GZRECOVER_PATH}"
                )

                subprocess.run(
                    [
                        str(GZRECOVER_PATH),
                        "-o",
                        recovered_out.name,
                        str(path.resolve()),
                    ],  # Use absolute path for gzrecover
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(OUT_CORRUPT_GZ),
                )

                if recovered_out.exists() and recovered_out.stat().st_size > 0:
                    fb = read_first_bytes(recovered_out, 64)

                    if fb.startswith(b"SQLite format 3"):
                        renamed = OUT_CORRUPT_GZ / (path.stem + "_gzrecover.sqlite")
                        recovered_out.rename(renamed)
                        print(
                            f"    [SUCCESS] gzrecover recovered SQLite data to {renamed.name}"
                        )
                        process_sqlite_candidate(renamed)
                        # move the gzrecover artifact out of corrupt_gz → recovered_sqlite
                        try:
                            archive_recovered_temp(renamed)
                        except Exception as e:
                            print(
                                f"    Warning: Could not archive recovered temp {renamed.name}: {e}"
                            )
                        return

                    if fb[:4] in FSEVENT_MAGICS:
                        print(
                            f"    [SUCCESS] gzrecover recovered FSEvents data to {recovered_out.name}"
                        )
                        # move (not copy) the recovered artifact into FSEvents
                        with contextlib.suppress(Exception):
                            shutil.move(recovered_out, OUT_FSE_DIR / recovered_out.name)
                        return

                    try:
                        list_tables(recovered_out)
                        renamed = OUT_CORRUPT_GZ / (path.stem + "_gzrecover.sqlite")
                        recovered_out.rename(renamed)
                        print(
                            f"    [SUCCESS] gzrecover produced an openable SQLite DB at {renamed.name}"
                        )
                        process_sqlite_candidate(renamed)
                        try:
                            archive_recovered_temp(renamed)
                        except Exception as e:
                            print(f"    Warning: Could not archive {renamed.name}: {e}")
                        return

                    except Exception:
                        print(
                            f"    Warning: gzrecover output not recognized; kept in {OUT_CORRUPT_GZ}"
                        )
                else:
                    print("    Warning: gzrecover produced no usable output.")
            except subprocess.CalledProcessError as e2:
                print(
                    f"    Warning: gzrecover failed ({e2.returncode}): {e2.stderr[:200]}"
                )
            except Exception as e3:
                print(f"    Warning: gzrecover error: {e3}")

        else:
            print("    Warning: gzrecover not found; quarantining file.")

        # If we reach here, nothing worked → quarantine the .gz itself
        print(
            f"    [ERROR] The gzipped file is unrecoverable and has been moved to: {OUT_CORRUPT_GZ}"
        )
        try:
            shutil.copy2(path, OUT_CORRUPT_GZ / path.name)
        except Exception as e:
            print(
                f"    [ERROR] Failed to copy {path.name} to the corrupt_gz directory: {e}"
            )
        return


def process_bz2(path: Path):
    """Handle .bz2 logs."""
    if bz_is_wifi_log(path) or name_indicates_wifi_log(path):
        OUT_WIFI_LOGS.mkdir(parents=True, exist_ok=True)
        print(f"  WiFi log (bz2): {path.name}")
        shutil.copy2(path, OUT_WIFI_LOGS / path.name)
        return

    if name_indicates_system_log(path):
        OUT_SYSLOGS.mkdir(parents=True, exist_ok=True)
        print(f"  System log (bz2): {path.name}")
        shutil.copy2(path, OUT_SYSLOGS / path.name)


def process_plain_file(path: Path):
    """
    Plain file: Could be FSEvents or SQLite.
    """
    suffix = path.suffix.lower()

    # Ignore standalone WAL/SHM companions; we'll handle them when copying the DB.
    if suffix in {".sqlite-wal", ".sqlite-shm", "-wal", "-shm"}:
        return

    # FSEvents?
    if suffix not in SQLITE_SUFFIXES and file_is_fsevent(path):
        OUT_FSE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  FSEvents: {path.name}")
        shutil.copy2(path, OUT_FSE_DIR / path.name)
        return

    # SQLite by suffix
    if is_sqlite_file(path):
        process_sqlite_candidate(path)


def walk_tree():
    for p in SRC_DIR.rglob("*"):
        if not p.is_file():
            continue
        if in_output_tree(p):
            continue
        suffix = p.suffix.lower()

        if suffix == ".bz2":
            print(f"\nScanning (bz2): {p.relative_to(SRC_DIR)}")
            process_bz2(p)
            continue

        # .gz handling
        if suffix == ".gz":
            print(f"\nScanning (gz): {p.relative_to(SRC_DIR)}")
            process_gz(p)
            continue

        # Plain file handling
        if True:
            # We'll let process_plain decide what to do (fsevents/sqlite/ignore)
            print(f"\nScanning: {p.relative_to(SRC_DIR)}")
            process_plain_file(p)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ensure_dest_columns(dst_cur: sqlite3.Cursor, table: str, src_colinfo: list[tuple]):
    """Add any missing columns (by name) to destination table using source types where available."""
    q_t = quote_ident(table)
    dst_cols = [r[1] for r in dst_cur.execute(f"PRAGMA table_info({q_t});").fetchall()]
    for cid, name, ctype, *_ in src_colinfo:
        if name not in dst_cols:
            typ = ctype or "TEXT"
            q_col = quote_ident(name)
            with contextlib.suppress(sqlite3.DatabaseError):
                dst_cur.execute(f"ALTER TABLE {q_t} ADD COLUMN {q_col} {typ}")


def merge_powerlogs() -> list[Path]:
    """Merge Powerlog DBs grouped by schema slug with per-schema outputs."""
    base_dir = OUT_SQL_DIR / "Powerlog"
    if not base_dir.exists():
        print(f"\n(No Powerlog folder found at {base_dir}; skipping merge.)")
        return []

    groups: dict[str, list[Path]] = {}
    for entry in base_dir.iterdir():
        if entry.name == "combined":
            continue
        if entry.is_dir():
            dbs = [
                p
                for p in entry.iterdir()
                if p.is_file() and p.suffix.lower() in SQLITE_SUFFIXES
            ]
            if dbs:
                groups[entry.name] = dbs
        elif entry.is_file() and entry.suffix.lower() in SQLITE_SUFFIXES:
            groups.setdefault("legacy", []).append(entry)

    if not groups:
        print(f"\n(No Powerlog DBs to merge in {base_dir}.)")
        return []

    known_schemas = load_all_schemas()
    schema_catalog = get_schema_catalog_cached()
    combined_outputs: list[Path] = []

    partial_dir = OUT_RECOVERED_SQLITE / "partially_recovered"
    partial_dir.mkdir(parents=True, exist_ok=True)

    for slug, candidates in sorted(groups.items()):
        if slug == POWERLOG_UNKNOWN_SLUG:
            print(f"\nSkipping merge for Powerlog slug '{slug}' (no known schema).")
            continue

        entry = schema_catalog.get(slug)
        if not entry:
            print(
                f"\nSkipping merge for Powerlog slug '{slug}' (schema metadata missing)."
            )
            continue

        label = entry.get("label", slug)
        schema_tables_lower = entry["tables_lower"]
        allowed_tables_lower = set(schema_tables_lower) | {"sqlite_sequence"}

        combined_dir = base_dir / slug / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_db = combined_dir / "CurrentPowerlog.PLSQL"
        if combined_db.exists():
            combined_db.unlink()

        print(
            f"\nMerging {len(candidates)} Powerlog DB(s) for '{label}' to {combined_db.name}"
        )

        dst_conn = sqlite3.connect(str(combined_db))
        dst_cur = dst_conn.cursor()
        dst_cur.execute("PRAGMA journal_mode=OFF;")
        dst_conn.commit()

        merge_report = combined_dir / "PowerlogMergeReport.csv"
        rows_per_db: dict[str, list[int]] = {}

        with merge_report.open("w", encoding="utf-8") as rep:
            rep.write("Database,Table,RowsMerged\n")

            for db in candidates:
                try:
                    src_conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    cur = src_conn.cursor()
                    tables = cur.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table';"
                    ).fetchall()

                    db_schema = {}
                    for tname, _ in tables:
                        if tname.lower() not in allowed_tables_lower:
                            continue
                        try:
                            cur.execute(f"PRAGMA table_info('{tname}');")
                            db_schema[tname] = {
                                name: ctype for _, name, ctype, *_ in cur.fetchall()
                            }
                        except Exception:
                            continue

                    new_cols = compare_schema(db_schema, known_schemas)
                    if new_cols:
                        save_missing_schema_report(
                            new_cols, combined_dir / "new_columns_detected.csv"
                        )
                except Exception as e:
                    print(f"  Warning: Skipping unreadable {db.name}: {e}")
                    continue

                for tname, schema_sql in tables:
                    if not schema_sql:
                        continue

                    if tname.lower() not in allowed_tables_lower:
                        continue

                    q_tname = quote_ident(tname)
                    with contextlib.suppress(sqlite3.OperationalError):
                        dst_cur.execute(schema_sql)

                    try:
                        src_cols_info = cur.execute(
                            f"PRAGMA table_info({q_tname});"
                        ).fetchall()
                    except sqlite3.DatabaseError as e:
                        print(
                            f"  Warning: Cannot inspect columns for {tname} in {db.name}: {e}"
                        )
                        continue

                    try:
                        dst_cols_info = dst_cur.execute(
                            f"PRAGMA table_info({q_tname});"
                        ).fetchall()
                    except sqlite3.DatabaseError:
                        continue

                    src_cols = [c[1] for c in src_cols_info]
                    dst_cols = [c[1] for c in dst_cols_info]
                    common_cols = [c for c in src_cols if c in dst_cols]

                    if not {"ID", "timestamp"}.issubset(set(common_cols)):
                        print(
                            f"  Warning: Skipping {tname} in {db.name} (missing ID/timestamp)"
                        )
                        continue

                    ensure_dest_columns(dst_cur, tname, src_cols_info)
                    dst_cols_info = dst_cur.execute(
                        f"PRAGMA table_info({q_tname});"
                    ).fetchall()
                    dst_cols = [c[1] for c in dst_cols_info]
                    common_cols = [c for c in src_cols if c in dst_cols]

                    col_list = ", ".join(quote_ident(c) for c in common_cols)
                    placeholders = ", ".join("?" for _ in common_cols)

                    try:
                        rows_iter = cur.execute(f"SELECT {col_list} FROM {q_tname}; ")
                    except sqlite3.DatabaseError:
                        try:
                            print(
                                f"    Requerying {tname} WITHOUT indexes (index corruption suspected)"
                            )
                            rows_iter = cur.execute(
                                f"SELECT {col_list} FROM {q_tname} NOT INDEXED;"
                            )
                        except sqlite3.DatabaseError as e:
                            print(
                                f"  Warning: Cannot read table {tname} in {db.name}: {e}"
                            )
                            rep.write(f"{db.name},{tname},0\n")
                            rows_per_db.setdefault(db.name, []).append(0)
                            continue

                    try:
                        ts_idx = common_cols.index("timestamp")
                    except ValueError:
                        ts_idx = None

                    copied = 0
                    while True:
                        try:
                            row = rows_iter.fetchone()
                        except sqlite3.DatabaseError as e:
                            print(f"  Warning: Row read error {db.name}:{tname}: {e}")
                            break
                        if row is None:
                            break
                        if all(v is None for v in row):
                            continue
                        if ts_idx is not None and not looks_like_unix_timestamp(
                            row[ts_idx]
                        ):
                            continue
                        try:
                            dst_cur.execute(
                                f"INSERT OR IGNORE INTO {q_tname} ({col_list}) VALUES ({placeholders})",
                                row,
                            )
                            copied += 1
                        except sqlite3.DatabaseError:
                            continue

                    rep.write(f"{db.name},{tname},{copied}\n")
                    rows_per_db.setdefault(db.name, []).append(copied)

                src_conn.close()

        dst_conn.commit()
        dst_conn.close()
        print(
            f"  [SUCCESS] Powerlog merge complete for '{label}'. Report saved to {merge_report}"
        )

        for db_name, counts in rows_per_db.items():
            if counts and all(count == 0 for count in counts):
                candidate_path = (
                    base_dir / slug / db_name
                    if (base_dir / slug / db_name).exists()
                    else base_dir / db_name
                )
                if candidate_path.exists():
                    try:
                        dest = partial_dir / db_name
                        shutil.copy2(candidate_path, dest)
                        print(f"  [WARNING] Copied fully empty Powerlog DB to {dest}")
                    except Exception as e:
                        print(f"  [WARNING] Could not copy {db_name}: {e}")

        combined_outputs.append(combined_db)

    return combined_outputs


# ---------- Run ----------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a PhotoRec output directory for Apple SQLite artifacts and FSEvents logs."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the PhotoRec output directory to scan.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory where parsed output should be written (default: <input_dir>/Parsed).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        parser.error(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    input_dir = input_dir.resolve()

    if args.output_dir is None:
        output_root = (input_dir / "Parsed").resolve()
    else:
        output_root = Path(args.output_dir).expanduser()
        output_root = (
            (input_dir / output_root).resolve()
            if not output_root.is_absolute()
            else output_root.resolve()
        )

    configure_paths(input_dir, output_root)
    prepare_output_dirs()

    z_path_status = (
        f"{GZRECOVER_PATH} (missing)"
        if not GZRECOVER_PATH or not GZRECOVER_PATH.exists()
        else str(GZRECOVER_PATH)
    )
    dissect_status = (
        f"{SQLITE_DISSECT_PATH} (missing)"
        if not SQLITE_DISSECT_PATH or not SQLITE_DISSECT_PATH.exists()
        else str(SQLITE_DISSECT_PATH)
    )

    print(f"Source directory: {SRC_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"gzrecover: {z_path_status}")
    print(f"sqlite_dissect: {dissect_status}")

    walk_tree()
    combined_outputs = merge_powerlogs()
    if combined_outputs:
        summary_parts = []
        for p in combined_outputs:
            try:
                summary_parts.append(str(p.relative_to(OUTPUT_ROOT)))
            except ValueError:
                summary_parts.append(str(p))
        combined_summary = ", ".join(summary_parts)
    else:
        combined_summary = "(none created)"
    print(
        "\nComplete.\n"
        f"• Classified SQLite: {OUT_SQL_DIR}\n"
        f"• FSEvents gathered: {OUT_FSE_DIR}\n"
        f"• Unsalvageable dumps: {OUT_UNSALV}\n"
        f"• Recovered via .recover: {OUT_RECOVERED_SQLITE}\n"
        f"• Combined Powerlog: {combined_summary}"
    )


if __name__ == "__main__":
    main()
