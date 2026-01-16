#!/usr/bin/env python3
# Strict template matcher for carved SQLite DBs (generic; signature-free)
# - Default: STRICT mode -> only exact schema matches are accepted (0% otherwise).
# - Default: require exact column sets (can opt-out with --no-require-columns).
# - Robust to malformed DBs; falls back to bytes scan of CREATE TABLE statements.
# - Streams JSONL: header + one record per case.

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mars.utils.database_utils import quote_identifier
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    import argparse

    from mars.pipeline.raw_scanner.db_variant_selector.models import DBMeta

# -----------------------------
# Ignorable-table configuration
# Load from global config for single source of truth
# -----------------------------
try:
    from mars.config.schema import SchemaComparisonConfig

    _config = SchemaComparisonConfig()
    DEFAULT_IGNORABLE_TABLES = _config.ignorable_tables
    DEFAULT_IGNORABLE_PREFIXES = _config.ignorable_prefixes
    DEFAULT_IGNORABLE_SUFFIXES = _config.ignorable_suffixes
    SALVAGE_TABLE_CANON = _config.salvage_tables
except ImportError:
    # Fallback if config not available (standalone mode)
    # Use minimal set for standalone operation
    DEFAULT_IGNORABLE_TABLES = {
        "sqlite_sequence",
        "sqlite_stat1",
        "sqlite_stat4",
    }
    DEFAULT_IGNORABLE_PREFIXES = {"sqlite_", "sqlean_"}
    DEFAULT_IGNORABLE_SUFFIXES = {
        "_content",
        "_segments",
        "_segdir",
        "_docsize",
        "_stat",
    }
    SALVAGE_TABLE_CANON = {"lost_and_found"}


# -----------------------------
# Data models
# -----------------------------
@dataclass
class TableDef:
    """Definition of a database table for schema comparison.

    Attributes:
        name: Normalized table name for schema comparison.
        columns: List of column names in the table.
        is_virtual: Whether this is a virtual table (FTS, etc.).
        original_name: Original name from sqlite_master, used for querying.
    """

    name: str
    columns: list[str] = field(default_factory=list)
    is_virtual: bool = False
    original_name: str | None = None


# -----------------------------
# Magic Numbers to detect SQLite DBs without file extension
# -----------------------------

SQLITE_MAGIC = b"SQLite format 3\x00"
WAL_MAGIC = b"WAL\0"  # not a main DB
SHM_MAGIC = b"SQLiteShm"  # not a main DB


# -----------------------------
# Regexes for DDL salvage
# -----------------------------
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+([`\"'\[\]]?)(?P<name>[^`\"'\[\]\s]+)\1\s*\((?P<body>.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
COLUMN_NAME_RE = re.compile(
    r"(^|\s|,)([`\"'\[]?)(?P<col>[A-Za-z_][A-Za-z0-9_$.]*)\2\s+[A-Za-z(]",
    re.IGNORECASE,
)


# ==========================================================================
# ======================= CASE HELPERS =====================================
# ==========================================================================


def make_case_dir(variants_root: Path, src: Path) -> Path:
    """
    Build a stable per-case directory under variants_root.
    Uses '<stem>_<md5-8>' to avoid collisions.
    """
    tag = _short_md5(src, 8)
    d = variants_root / f"{src.stem}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_copy(src: Path, dst: Path) -> Path | None:
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def list_columns_safe(con: sqlite3.Connection, table: str, cap: int = 64) -> list[str]:
    """
    Return up to `cap` column names for `table` via PRAGMA table_info.
    Keeps names quoted safely. Falls back to empty list on error.
    """
    try:
        cur = con.execute(f"PRAGMA table_info('{table}')")
        cols = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cur]
        # Normalize to strings and respect cap
        return [c for c in cols if isinstance(c, str) and c][:cap]
    except sqlite3.Error:
        return []


def exemplar_label(path: Path, root: Path) -> str:
    """
    Return the top-level folder name under `root` that contains `path`.
    If `path` isn't under `root`, fall back to the immediate parent folder name.

    Special handling for OutputStructure:
    - If parts[0] is 'catalog' or 'originals', use parts[1] as the label
    - This handles the structure: databases/catalog/{db_name}/ or databases/originals/{filename}
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
        if not parts:
            return path.parent.name

        # Skip intermediate "catalog" or "originals" directories
        it = iter(parts)
        first = next(it, None)
        second = next(it, None)
        if first in ("catalog", "originals") and second:
            # For catalog: catalog/{db_name}/{file} → use db_name
            # For originals: originals/{file} → use filename stem
            if first == "catalog":
                return second
            # originals
            # Extract DB name from filename (remove extensions)
            filename = second
            # Remove .db, .sqlite extensions
            name = filename.replace(".sqlite", "").replace(".db", "")
            return name

        return first or path.parent.name
    except Exception:
        return path.parent.name


def discovery_report(root: Path, found: list[Path], report_path: Path):
    # Only report SQLite discovery counts; ignore non-DB files entirely.
    write_jsonl(
        report_path,
        {
            "type": "discovery_summary",
            "root": root.as_posix(),
            "sqlite_found": len(found),
        },
    )


def _read_head(path: Path, n: int = 512) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except Exception:
        return b""


# ==========================================================================
# ======================= SQLITE3 HELPERS ==================================
# ==========================================================================
def is_effectively_empty(meta: DBMeta) -> tuple[bool, str]:
    """
    A DB is 'effectively empty' if:
      - No non-ignorable tables exist, OR
      - No table has any non-NULL cell (rows with all NULLs don't count).
    """
    if not meta.effective_table_names:
        return True, "no_effective_tables"
    if not meta.nonnull_tables:
        return True, "no_nonnull_cells_in_tables"
    return False, ""


def _short_md5(path: Path, n: int = 8) -> str:
    try:
        return md5_bytes(path)[:n]
    except Exception:
        return "unknown"


def _looks_like_sqlite(head: bytes) -> bool:
    # True SQLite databases have the magic at offset 0
    return head.startswith(SQLITE_MAGIC)


def _is_wal_or_shm(head: bytes, name: str) -> bool:
    # quick exclusions
    if head.startswith(WAL_MAGIC) or head.startswith(SHM_MAGIC):
        return True
    # common suffix exclusions
    lname = name.lower()
    return lname.endswith("-wal") or lname.endswith("-shm") or lname.endswith("-journal")


def _should_try_sqlite_open(path: Path) -> bool:
    """
    Heuristic to decide if we should attempt the expensive SQLite open check.
    Returns False for files that are very unlikely to be SQLite databases.
    """
    try:
        # Skip very small files (SQLite header is 100 bytes minimum)
        size = path.stat().st_size
        if size < 100:
            return False

        # Skip very large files (>2GB) to avoid timeout issues
        if size > 2 * 1024 * 1024 * 1024:
            return False

        # Skip known non-SQLite extensions
        suffix = path.suffix.lower()
        non_sqlite_extensions = {
            ".txt",
            ".log",
            ".json",
            ".xml",
            ".html",
            ".csv",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".zip",
            ".tar",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".plist",
        }
        return suffix not in non_sqlite_extensions
    except Exception:
        return True  # If we can't check, be safe and try


def _openable_sqlite(path: Path) -> bool:
    # second-chance: if SQLite can open it read-only/immutable and answer a pragma, accept it
    # But first check if it's worth trying (optimization)
    if not _should_try_sqlite_open(path):
        return False

    con = None
    try:
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        con = sqlite3.connect(uri, uri=True, timeout=1.5)
        con.execute("PRAGMA schema_version;").fetchone()
        return True
    except sqlite3.Error:
        return False
    finally:
        with contextlib.suppress(Exception):
            con and con.close()  # type: ignore


def _maybe_gz_sqlite(path: Path) -> bool:
    # Optional: peek into small gz files (if you actually have gz exemplars)
    # Avoids heavy reads; only checks the very beginning of the stream.
    try:
        if path.suffix.lower() != ".gz":
            return False

        # Skip very large .gz files (>500MB compressed) to avoid long decompress
        size = path.stat().st_size
        if size > 500 * 1024 * 1024:
            return False

        with gzip.open(path, "rb") as f:
            head = f.read(32)
        return _looks_like_sqlite(head)
    except Exception:
        return False


def md5_bytes(path: Path, n: int = 65536) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_connect(path: Path, immutable: bool = False) -> sqlite3.Connection | None:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1" if immutable else f"file:{path.as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=2.5)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def run_pragma(con: sqlite3.Connection, pragma: str) -> str | None:
    try:
        cur = con.execute(f"PRAGMA {pragma}")
        row = cur.fetchone()
        if row is None:
            return None
        if len(row.keys()) == 1:
            # row is a tuple-like row; pull first value
            return str(list(row)[0])
        return " | ".join(str(v) for v in row)
    except sqlite3.Error:
        return None


def normalize_ident(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "`'\"":
        s = s[1:-1].strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def split_lower_csv(s: str) -> set[str]:
    return {t.strip().lower() for t in s.split(",") if t.strip()} if s else set()


def build_ignore_sets(ns: argparse.Namespace) -> tuple[set[str], set[str], set[str]]:
    tables = set(DEFAULT_IGNORABLE_TABLES) | split_lower_csv(ns.ignore_tables)
    prefixes = set(DEFAULT_IGNORABLE_PREFIXES) | split_lower_csv(ns.ignore_prefixes)
    suffixes = set(DEFAULT_IGNORABLE_SUFFIXES) | split_lower_csv(ns.ignore_suffixes)
    return tables, prefixes, suffixes


def discover_sqlites(root: Path, progress_callback=None) -> list[Path]:
    """
    Returns main DB files only (excludes WAL/SHM/journal).
    A file is accepted if:
      1) It has SQLite magic at offset 0, OR
      2) SQLite can open it in ro/immutable and answer PRAGMA, OR
      3) (Optional) It is a .gz whose first bytes decompress to SQLite magic.

    Args:
        root: Root directory to search
        progress_callback: Optional callback(current, total) for progress updates

    Performance notes:
        - First pass counts total files for progress tracking
        - Second pass checks each file with progress updates
        - Optimized to skip expensive checks on unlikely candidates
    """
    out: list[Path] = []

    # First pass: Count total files for progress tracking
    # Using list comprehension is faster than loop + append
    all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() != ".json"]
    total_files = len(all_files)

    # Second pass: Check each file with progress updates
    for idx, p in enumerate(all_files, 1):
        # Report progress
        if progress_callback:
            progress_callback(idx, total_files)

        # Fast head check
        head = _read_head(p, 32)

        # Exclude WAL/SHM/journal quickly
        if _is_wal_or_shm(head, p.name):
            continue

        # Check if file is SQLite (optimized to skip expensive checks)
        ok = False
        if _looks_like_sqlite(head) or _openable_sqlite(p) or _maybe_gz_sqlite(p):
            ok = True

        if ok:
            out.append(p)
    return out


def discover_sqlites_fast(root: Path) -> list[Path]:
    """
    Fast version: Only checks SQLite magic bytes and .gz files.
    Skips the slower _openable_sqlite() check.
    Use this when speed is critical and you can accept missing some edge cases.
    """
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() and p.suffix.lower() != ".json":
            continue

        # Fast head check
        head = _read_head(p, 32)

        # Exclude WAL/SHM/journal quickly
        if _is_wal_or_shm(head, p.name):
            continue

        # Only check magic bytes and .gz (skip slow SQLite open attempt)
        if _looks_like_sqlite(head) or _maybe_gz_sqlite(p):
            out.append(p)
    return out


def write_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_salvage_name(name: str) -> bool:
    return name.lower() in SALVAGE_TABLE_CANON


def table_has_nonnull(con: sqlite3.Connection, table: str, sample_rows: int = 50) -> tuple[bool, bool]:
    """
    Returns (has_row, has_nonnull_cell) for the given table.
    Uses a capped column list to avoid preparing queries with thousands of columns.
    """
    try:
        if is_salvage_name(table):
            return (False, False)
        cols = list_columns_safe(con, table, cap=64)
        quoted_table = quote_identifier(table)
        if not cols:
            # Fall back to existence probe
            try:
                cur2 = con.execute(f"SELECT 1 FROM {quoted_table} LIMIT 1;")
                has_row = cur2.fetchone() is not None
                return (has_row, has_row)
            except sqlite3.Error:
                return (False, False)

        # Quote all column names safely (handles special chars like " in column names)
        collist = ", ".join([quote_identifier(c) for c in cols])

        sql = f"SELECT {collist} FROM {quoted_table} LIMIT {int(sample_rows)};"
        cur = con.execute(sql)
        saw_any_row = False
        for row in cur:
            saw_any_row = True
            # sqlite3.Row supports index by position; safest is position iterate:
            if any(v is not None for v in row):
                return (True, True)
        return (saw_any_row, False)
    except sqlite3.Error:
        return (False, False)


def table_rowcount(con: sqlite3.Connection, table: str) -> int:
    """
    Return the rowcount for a given table using COUNT(*).
    Skips salvage tables. Returns 0 on error.
    """
    try:
        if is_salvage_name(table):
            return 0
        quoted_table = quote_identifier(table)
        cur = con.execute(f"SELECT COUNT(*) FROM {quoted_table};")
        row = cur.fetchone()
        if row is None:
            return 0
        v = row[0]
        try:
            return int(v) if v is not None else 0
        except Exception:
            return 0
    except sqlite3.Error:
        return 0


def _which_sqlite3() -> str | None:
    """Locate sqlite3 binary, cross-platform.

    On Windows, looks for sqlite3.exe in:
    1. PATH (via shutil.which)
    2. Bundled resources/windows/bin/
    3. Common installation locations

    On macOS/Linux, looks for sqlite3 in:
    1. PATH (via shutil.which)
    2. Common Unix paths
    """
    import sys

    is_windows = sys.platform == "win32"

    # Use shutil.which for cross-platform PATH lookup first
    which_result = shutil.which("sqlite3")
    if which_result:
        return which_result

    # Check bundled resources directory
    # __file__ is in db_variant_selector/, go up to src/
    script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    if is_windows:
        bundled = script_dir / "resources" / "windows" / "bin" / "sqlite3.exe"
    else:
        bundled = script_dir / "resources" / "macos" / "bin" / "sqlite3"

    if bundled.exists():
        return str(bundled)

    # Platform-specific fallback paths
    if is_windows:
        # Windows: common installation locations
        candidates = [
            r"C:\sqlite\sqlite3.exe",
            r"C:\Program Files\sqlite\sqlite3.exe",
            r"C:\Program Files (x86)\sqlite\sqlite3.exe",
        ]
    else:
        # Unix/macOS fallback paths
        candidates = [
            "/usr/bin/sqlite3",
            "/opt/homebrew/bin/sqlite3",
            "/usr/local/bin/sqlite3",
        ]

    for cand in candidates:
        try:
            r = subprocess.run([cand, "-version"], capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                return cand
        except Exception:
            continue

    return None


# -----------------------------
# Nearest (optional, for triage)
# -----------------------------
def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / (len(a | b) or 1)


def nearest_by_tables(case: DBMeta, exemplars: list[DBMeta], k: int = 5) -> list[dict]:
    """
    Find the k nearest exemplars by Jaccard similarity of table names.

    Optimization: Pre-compute lowercased table name sets once per exemplar
    instead of recomputing during both sorting and result building.
    """
    a = {t.lower() for t in (case.effective_table_names or case.table_names)}

    # Pre-compute lowercased sets once per exemplar (avoids O(2n) set creation)
    exemplar_sets = [(ex, {t.lower() for t in (ex.effective_table_names or ex.table_names)}) for ex in exemplars]

    # Sort using pre-computed sets
    ranked = sorted(
        exemplar_sets,
        key=lambda item: jaccard(a, item[1]),
        reverse=True,
    )[:k]

    out = []
    for ex, b in ranked:
        shared = a & b
        # Only include matches with shared tables (filter out noise)
        if shared:
            out.append(
                {
                    "exemplar": ex.path.as_posix(),
                    "tables_jaccard": round(jaccard(a, b), 3),
                    "shared": sorted(shared),
                }
            )
    return out


# ==========================================================================
# ======================= SQLITE DISSECT HELPERS ===========================
# ==========================================================================


def _which_sqlite_dissect(script_dir: Path) -> str | None:
    """
    Locate sqlite_dissect binary. Prefer resources bin dir, then script dir, then PATH.

    On Windows, looks for sqlite_dissect.exe in resources/windows/bin.
    On macOS/Linux, looks for sqlite_dissect in resources/macos/bin.
    """
    import sys

    is_windows = sys.platform == "win32"
    binary_name = "sqlite_dissect.exe" if is_windows else "sqlite_dissect"
    platform_dir = "windows" if is_windows else "macos"

    candidates = [
        script_dir / "resources" / platform_dir / "bin" / binary_name,
        script_dir / binary_name,
        Path(binary_name),
    ]

    for cand in candidates:
        try:
            # On Windows, .exe files are executable by default
            # On Unix, check executable permission
            if cand.exists() and (is_windows or os.access(str(cand), os.X_OK)):
                return str(cand)
        except Exception:
            continue

    # Last resort: rely on PATH using shutil.which (cross-platform)
    which_result = shutil.which("sqlite_dissect")
    if which_result and Path(which_result).exists():
        return which_result

    return None


def run_sqlite_dissect(bin_path: str, recovered_db: Path, case_dir: Path, debug: bool = False) -> tuple[bool, str]:
    """
    Execute sqlite_dissect against a recovered DB. Returns (success, error_msg).
    This tool's out can be flaky on recovered DBs;
    failures are expected and should not abort the pipeline.
    """

    try:
        # Use native path strings for subprocess (Windows expects backslashes)
        cmd = [
            bin_path,
            "-c",  # Carve
            "-f",  # Carve freelists
            "-k",  # No strict checking (will crash otherwise)
            str(recovered_db),
            "-e",  # export dissected DB
            "sqlite",  # as sqlite
            "-d",  # into the case directory
            str(case_dir),
        ]

        if debug:
            logger.debug(f"Command: {' '.join(cmd)}")
            logger.warning(f"     [DEBUG] Binary exists: {Path(bin_path).exists()}")
            logger.debug(f"DB exists: {recovered_db.exists()}")

        # Reduce timeout to 30s - sqlite_dissect should be fast if it works
        # Use encoding with errors="replace" to handle binary data in corrupted DBs
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        if debug:
            logger.debug(f"Return code: {r.returncode}")
            if r.stderr:
                first_err_line = r.stderr.strip().split("\n")[0] if r.stderr.strip() else ""
                logger.warning(f"     [DEBUG] First stderr line: {first_err_line[:150]}")

        if r.returncode == 0:
            return True, ""
        # Return stderr for debugging
        error_msg = (r.stderr or r.stdout or "No error output").strip()
        # Get just the first error/warning line
        first_line = error_msg.split("\n")[0] if error_msg else "No output"
        return False, first_line[:200]
    except subprocess.TimeoutExpired:
        return False, "Timeout (30s)"
    except Exception as e:
        return False, f"Exception: {str(e)[:100]}"


def find_sqlite_dissect_db(recovered_db: Path) -> Path | None:
    """
    Heuristic: sqlite_dissect writes '<recovered>-sqlite-dissect.db3' in the same folder.
    """
    cand = recovered_db.parent / f"{recovered_db.name}-sqlite-dissect.db3"
    if cand.exists():
        return cand
    # Fallback: scan for similarly-named files
    prefix = f"{recovered_db.name}-sqlite-dissect"
    for p in recovered_db.parent.glob("*"):
        try:
            if p.is_file() and p.name.startswith(prefix):
                return p
        except Exception:
            continue
    return None


def rebuild_sqlite_dissect_to_exemplar_shape(
    dissect_db: Path, exemplar_db: Path, out_db: Path, schemas_dir: Path | None = None
) -> dict:
    """
    Create a copy of `dissect_db` that matches the exemplar's schema:
    - For every exemplar table: create the same schema in `out_db`.
    - Copy data by selecting the shared column names (ignoring the sqlite_dissect
      metadata columns with `sd_` prefixes). Columns not present in the source are
      left to their defaults/NULLs in the rebuilt table.
    - Provide per-table diagnostics (rows found/inserted, columns copied/missing,
      and any copy errors).

    Args:
        dissect_db: Path to sqlite_dissect output database
        exemplar_db: Path to exemplar database
        out_db: Path for rebuilt output database
        schemas_dir: Optional path to schemas directory for rubric-based validation
    """
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        with contextlib.suppress(Exception):
            out_db.unlink()

    # Use ExitStack to ensure all connections are properly closed even if
    # later connections fail to open (prevents FD leaks on partial failures)
    with contextlib.ExitStack() as stack:
        con_src = stack.enter_context(sqlite3.connect(f"file:{dissect_db.as_posix()}?mode=ro", uri=True))
        con_src.row_factory = sqlite3.Row

        con_ex = stack.enter_context(sqlite3.connect(f"file:{exemplar_db.as_posix()}?mode=ro", uri=True))
        con_ex.row_factory = sqlite3.Row

        con_out = stack.enter_context(sqlite3.connect(str(out_db)))
        con_out.execute("PRAGMA journal_mode=WAL;")
        con_out.execute("PRAGMA synchronous=OFF;")

        tables_created = []
        rows_inserted = {}
        rows_source = {}
        copied_cols_by_table = {}
        missing_cols_by_table = {}
        errors_by_table = {}
        blob_filtered_rows_by_table = {}
        blob_cleaned_rows_by_table = {}
        rows_rejected_by_table = {}
        rejection_reasons_by_table = {}
        # Enumerate exemplar tables
        cur = con_ex.execute("SELECT name FROM sqlite_master WHERE type='table'")
        exemplar_tables = [r[0] for r in cur.fetchall() if r and r[0]]

        for ex_table in exemplar_tables:
            try:
                ex_schema = get_table_schema(con_ex, ex_table)
                # Create same table name in out_db
                fixed_name = ex_table
                create_fixed_table_like_exemplar(con_out, con_ex, ex_table, fixed_name)
                tables_created.append(fixed_name)

                # Collect columns from source table with the exact same name
                src_cols = []
                try:
                    sc = con_src.execute(f"PRAGMA table_info('{ex_table}')").fetchall()
                    src_cols = [row[1] if not isinstance(row, sqlite3.Row) else row["name"] for row in sc]
                except sqlite3.Error:
                    src_cols = []

                if not table_exists(con_src, ex_table):
                    # No same-named table in source; keep empty table shape only
                    rows_inserted[fixed_name] = 0
                    rows_source[fixed_name] = 0
                    copied_cols_by_table[fixed_name] = []
                    missing_cols_by_table[fixed_name] = [c["name"] for c in ex_schema]
                    blob_filtered_rows_by_table[fixed_name] = 0
                    blob_cleaned_rows_by_table[fixed_name] = 0
                    rows_rejected_by_table[fixed_name] = 0
                    copy_indexes_from_exemplar(con_out, con_ex, ex_table, fixed_name)
                    continue

                # Track source rowcount
                try:
                    src_cnt_row = con_src.execute(f'SELECT COUNT(1) FROM "{ex_table}"').fetchone()
                    src_cnt = src_cnt_row[0] if src_cnt_row else 0
                except sqlite3.Error:
                    src_cnt = 0
                rows_source[fixed_name] = int(src_cnt or 0)

                # Columns that exist in both exemplar and dissect source
                copied_cols = [c["name"] for c in ex_schema if c["name"] in src_cols]
                missing_cols = [c["name"] for c in ex_schema if c["name"] not in src_cols]
                copied_cols_by_table[fixed_name] = copied_cols
                missing_cols_by_table[fixed_name] = missing_cols

                if not copied_cols:
                    rows_inserted[fixed_name] = 0
                    blob_filtered_rows_by_table[fixed_name] = 0
                    blob_cleaned_rows_by_table[fixed_name] = 0
                    rows_rejected_by_table[fixed_name] = 0
                    copy_indexes_from_exemplar(con_out, con_ex, ex_table, fixed_name)
                    continue

                # Prepare SELECT from source and INSERT into rebuilt table
                def _qcol(col: str) -> str:
                    return '"' + col.replace('"', '""') + '"'

                select_cols_sql = ", ".join(_qcol(c) for c in copied_cols)
                insert_cols_sql = ", ".join(_qcol(c) for c in copied_cols)
                placeholders = ", ".join(["?"] * len(copied_cols))
                insert_sql = f'INSERT INTO "{fixed_name}" ({insert_cols_sql}) VALUES ({placeholders})'

                type_map = {c["name"]: (c.get("type") or "").upper() for c in ex_schema}

                # Load rubric metadata for data-driven validation
                rubric_metadata = {}
                rubric_loaded = False
                if schemas_dir:
                    # Try loading rubric for this table
                    for suffix in ["", "_combined"]:
                        rubric_path = schemas_dir / ex_table / f"{ex_table}{suffix}.rubric.json"
                        if rubric_path.exists():
                            try:
                                import json

                                with Path.open(rubric_path) as f:
                                    rubric_metadata = json.load(f)
                                rubric_loaded = True
                                break
                            except Exception as e:
                                logger.debug(f"Failed to load rubric {rubric_path}: {e}")
                                pass

                def validate_row_types(vals: list, cols: list) -> tuple[bool, str | None]:
                    """
                    Validate row data based on rubric roles (data-driven, not heuristic).

                    Only validates roles that were confidently set by analyzing actual data
                    during rubric generation. Does NOT infer roles from column names or types.

                    Supports both single roles ("timestamp") and multi-roles (["url", "path"]).

                    Returns (is_valid, rejection_reason).
                    """
                    if not rubric_metadata or "tables" not in rubric_metadata:
                        return True, None  # No rubric available, skip validation

                    # Navigate to the table's columns in the rubric
                    table_rubric = rubric_metadata.get("tables", {}).get(ex_table, {})
                    columns_list = table_rubric.get("columns", [])

                    # Build column metadata lookup (rubric uses list, not dict)
                    col_metadata = {c["name"]: c for c in columns_list if "name" in c}

                    for col, val in zip(cols, vals):
                        if val is None:
                            continue  # NULL is always valid

                        col_meta = col_metadata.get(col)
                        if not col_meta:
                            continue  # No metadata for this column

                        role = col_meta.get("role")
                        if not role:
                            continue  # No role set, skip validation

                        # Handle both single role (string) and multi-role (list)
                        roles = [role] if isinstance(role, str) else role

                        # Validate timestamp role (both regular and nullable)
                        is_timestamp_role = "timestamp" in roles or "nullable_timestamp" in roles
                        if is_timestamp_role:
                            if not isinstance(val, (int, float)):
                                continue  # Allow non-numeric in timestamp columns (might be NULL markers)

                            # Zero is a universal "null timestamp" sentinel across all platforms
                            # (Unix epoch, Cocoa Jan 1 2001, WebKit year 1601)
                            if val == 0:
                                continue  # Zero is valid as null timestamp sentinel

                            from mars.pipeline.matcher.rubric_utils import (
                                TimestampFormat,
                            )

                            # Allow Apple sentinel values (distantPast/distantFuture)
                            # These represent null/empty dates in Cocoa databases
                            if TimestampFormat.is_apple_timestamp_sentinel(val):
                                continue  # Sentinel values are valid

                            # Reject negative timestamps - Core Foundation absolute time should
                            # not be classified as role=timestamp during rubric generation
                            if val < 0:
                                return (
                                    False,
                                    f"{col}={val} (negative timestamp, roles={roles})",
                                )

                            # Reject unreasonably large values
                            if val > 4102444800000000000:  # Year 2100 in nanoseconds
                                return (
                                    False,
                                    f"{col}={val} (timestamp too large, roles={roles})",
                                )

                        # Validate UUID role
                        if "uuid" in roles:
                            try:
                                # Import pattern detection
                                from mars.pipeline.matcher.rubric_utils import (
                                    detect_pattern_type,
                                )

                                # Convert bytes to string if needed
                                text = (
                                    val.decode("utf-8", errors="ignore")
                                    if isinstance(val, bytes)
                                    else str(val)
                                    if val is not None
                                    else ""
                                )

                                # Check if value is a valid UUID
                                if detect_pattern_type(text) != "uuid":
                                    return (
                                        False,
                                        f"{col}={text[:50]} (not a valid UUID, roles={roles})",
                                    )
                            except Exception:
                                return (
                                    False,
                                    f"{col}=<error decoding> (UUID validation failed, roles={roles})",
                                )

                    return True, None

                def coerce_value(name: str, value):
                    if value is None:
                        return None
                    dtype = type_map.get(name, "")
                    if isinstance(value, memoryview):
                        value = value.tobytes()
                    if (
                        dtype.startswith(("CHAR", "CLOB", "TEXT", "VAR", "NVARCHAR", "NCHAR")) or "CHAR" in dtype
                    ) and isinstance(value, (bytes, bytearray)):
                        try:
                            raw = bytes(value)
                            text = raw.decode("utf-8", errors="ignore").replace("\x00", "")
                            if text:
                                return text
                            if all(b == 0 for b in raw):
                                return ""
                            return raw
                        except Exception:
                            return value
                    return value

                inserted = 0
                rejected = 0
                rejection_reasons = []
                try:
                    cur_src = con_src.execute(f'SELECT {select_cols_sql} FROM "{ex_table}"')
                    for src_row in cur_src:
                        vals = [coerce_value(col, src_row[col]) for col in copied_cols]

                        # Validate row types before inserting
                        is_valid, rejection_reason = validate_row_types(vals, copied_cols)
                        if not is_valid:
                            rejected += 1
                            if len(rejection_reasons) < 10:  # Keep first 10 reasons
                                rejection_reasons.append(rejection_reason)

                        con_out.execute(insert_sql, vals)
                        inserted += 1
                except sqlite3.Error as copy_err:
                    errors_by_table[fixed_name] = str(copy_err)
                    inserted = 0

                # Store rejection stats
                rows_rejected_by_table[fixed_name] = rejected
                if rejection_reasons:
                    rejection_reasons_by_table[fixed_name] = rejection_reasons

                # Debug summary for this table
                if rubric_loaded and rejected > 0:
                    logger.debug(
                        f"Validation summary for {ex_table}: {rejected} rows rejected, {inserted} rows inserted"
                    )
                    if rejection_reasons:
                        logger.debug(f"Sample rejection reasons: {rejection_reasons[:3]}")

                blob_rows_removed = 0
                blob_rows_cleaned = 0
                blob_sensitive_cols = [col for col in copied_cols if "BLOB" not in (type_map.get(col, ""))]
                if blob_sensitive_cols and inserted:
                    conds_parts = [f"typeof({_qcol(col)})='blob'" for col in blob_sensitive_cols]
                    conds = " OR ".join(conds_parts)
                    try:
                        select_cols = ", ".join(_qcol(c) for c in blob_sensitive_cols)
                        cur_blob = con_out.execute(f'SELECT rowid, {select_cols} FROM "{fixed_name}" WHERE {conds}')

                        def _blob_to_text(raw_val):
                            data = bytes(raw_val)
                            text = data.decode("utf-8", errors="ignore")
                            text = text.replace("\x00", "")
                            if text:
                                return text
                            if all(b == 0 for b in data):
                                return ""
                            # Return None for truly binary data (can't decode to text)
                            # This will cause the column to be skipped, leaving BLOB as-is
                            return None

                        rows = cur_blob.fetchall()
                        for row in rows:
                            rowid = row[0]
                            updates = {}
                            for col_name, raw_val in zip(blob_sensitive_cols, row[1:]):
                                if isinstance(raw_val, memoryview):
                                    raw_val = raw_val.tobytes()
                                if not isinstance(raw_val, (bytes, bytearray)):
                                    continue
                                as_text = _blob_to_text(raw_val)
                                if as_text is None:
                                    # Skip this column - leave BLOB as-is
                                    # Don't try to convert truly binary data
                                    continue
                                updates[col_name] = as_text
                            if updates:
                                set_clause = ", ".join(f"{_qcol(col)}=?" for col in updates)
                                params = list(updates.values()) + [rowid]
                                con_out.execute(
                                    f'UPDATE "{fixed_name}" SET {set_clause} WHERE rowid=?',
                                    params,
                                )
                                blob_rows_cleaned += 1
                    except sqlite3.Error:
                        blob_rows_removed = 0
                        blob_rows_cleaned = 0

                try:
                    remaining = con_out.execute(f'SELECT COUNT(1) FROM "{fixed_name}"').fetchone()[0] or 0
                except sqlite3.Error:
                    remaining = max(0, inserted - blob_rows_removed)

                rows_inserted[fixed_name] = int(remaining)
                blob_filtered_rows_by_table[fixed_name] = blob_rows_removed
                blob_cleaned_rows_by_table[fixed_name] = blob_rows_cleaned
                copy_indexes_from_exemplar(con_out, con_ex, ex_table, fixed_name)
            except sqlite3.Error as outer_err:
                rows_inserted[ex_table] = 0
                rows_source[ex_table] = 0
                copied_cols_by_table[ex_table] = []
                missing_cols_by_table[ex_table] = []
                errors_by_table[ex_table] = str(outer_err)
                blob_filtered_rows_by_table[ex_table] = 0
                blob_cleaned_rows_by_table[ex_table] = 0
                rows_rejected_by_table[ex_table] = 0
                continue

        con_out.commit()

        # Switch from WAL to DELETE mode to ensure data is in the main file
        con_out.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        con_out.execute("PRAGMA journal_mode=DELETE;")

        return {
            "tables": tables_created,
            "rows_inserted": rows_inserted,
            "rows_source": rows_source,
            "copied_columns": copied_cols_by_table,
            "missing_columns": missing_cols_by_table,
            "errors": errors_by_table,
            "blob_filtered_rows": blob_filtered_rows_by_table,
            "blob_cleaned_rows": blob_cleaned_rows_by_table,
            "rows_rejected": rows_rejected_by_table,
            "rejection_reasons": rejection_reasons_by_table,
        }
        # Note: ExitStack context manager automatically closes all connections


# ==========================================================================
# ========================== SCHEMA COPY HELPERS ===========================
# ==========================================================================
def get_table_schema(con: sqlite3.Connection, table: str):
    cols = []
    try:
        cur = con.execute(f'PRAGMA table_info("{table}")')
        for cid, name, ctype, notnull, dflt, pk in cur.fetchall():
            cols.append(
                {
                    "name": name,
                    "type": ctype or "",
                    "notnull": notnull,
                    "dflt": dflt,
                    "pk": pk,
                }
            )
    except sqlite3.Error:
        pass
    return cols


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    try:
        cur = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (table,),
        )
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False


def create_fixed_table_like_exemplar(
    con_out: sqlite3.Connection,
    con_ex: sqlite3.Connection,
    table_ex: str,
    fixed_name: str,
):
    ex_cols = get_table_schema(con_ex, table_ex)
    cols_sql = []
    pk_cols = []
    for c in ex_cols:
        line = f'"{c["name"]}" {c["type"]}' if c["type"] else f'"{c["name"]}"'
        if c["notnull"]:
            line += " NOT NULL"
        if c["dflt"] is not None:
            line += f" DEFAULT {c['dflt']}"
        cols_sql.append(line)
        if c["pk"]:
            pk_cols.append(f'"{c["name"]}"')
    pk_sql = f", PRIMARY KEY ({', '.join(pk_cols)})" if pk_cols else ""
    sql = f'CREATE TABLE "{fixed_name}" ({", ".join(cols_sql)}{pk_sql});'
    con_out.execute(sql)


def copy_indexes_from_exemplar(
    con_out: sqlite3.Connection,
    con_ex: sqlite3.Connection,
    table_ex: str,
    fixed_name: str,
):
    cur = con_ex.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
        """,
        (table_ex,),
    )
    for name, sql in cur.fetchall():
        sql_fixed = sql.replace(f' ON "{table_ex}"', f' ON "{fixed_name}"').replace(
            f" ON {table_ex}", f' ON "{fixed_name}"'
        )
        with contextlib.suppress(sqlite3.Error):
            con_out.execute(sql_fixed)


# ================================================================
# ========================== CLI & I/O ===========================
# ================================================================
def build_ignore_sets_from_catalog(
    catalog_path: Path | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Build ignore sets from catalog YAML.

    Combines DEFAULT_IGNORABLE_TABLES with per-database ignorable tables
    from the catalog.

    Args:
        catalog_path: Optional path to catalog YAML. If None, uses default location.

    Returns:
        Tuple of (ignore_tables, ignore_prefixes, ignore_suffixes)
    """
    try:
        # Import catalog manager
        from mars.pipeline.common.catalog_manager import (
            GLOBAL_IGNORABLE_TABLES,
            CatalogManager,
        )

        # Initialize catalog manager
        catalog_mgr = CatalogManager(catalog_path=catalog_path)
        catalog = catalog_mgr.get_catalog()

        # Start with defaults
        ignore_tables = set(DEFAULT_IGNORABLE_TABLES)
        ignore_tables.update(GLOBAL_IGNORABLE_TABLES)

        if catalog:
            # Add per-database ignorable tables from catalog
            for category_name, entries in catalog.items():
                if category_name == "catalog_metadata" or category_name == "skip_databases":
                    continue

                if not isinstance(entries, list):
                    continue

                for entry in entries:
                    if isinstance(entry, dict):
                        ignorable = entry.get("ignorable_tables", [])
                        if ignorable:
                            # Lowercase all ignorable tables for case-insensitive matching
                            ignore_tables.update(t.lower() for t in ignorable)

        ignore_prefixes = set(DEFAULT_IGNORABLE_PREFIXES)
        ignore_suffixes = set(DEFAULT_IGNORABLE_SUFFIXES)

        return ignore_tables, ignore_prefixes, ignore_suffixes

    except ImportError:
        # Fallback if catalog manager not available
        return (
            set(DEFAULT_IGNORABLE_TABLES),
            set(DEFAULT_IGNORABLE_PREFIXES),
            set(DEFAULT_IGNORABLE_SUFFIXES),
        )


# ================================================================
# ===================== SKIP DATABASES ===========================
# ================================================================
def load_skip_databases_from_catalog(
    catalog_path: Path | None = None,
) -> dict[str, list[str]]:
    """Load skip_databases patterns from catalog YAML.

    Returns a dict mapping skip database name to list of table patterns.
    For example: {"geoservices": ["GeoPlaces", "GeoLookup_*", "RKTimeZone", ...]}

    Args:
        catalog_path: Optional path to catalog YAML. If None, uses default location.

    Returns:
        Dict of skip database name -> table patterns
    """
    try:
        from mars.pipeline.common.catalog_manager import CatalogManager

        catalog_mgr = CatalogManager(catalog_path=catalog_path)
        catalog = catalog_mgr.get_catalog()

        skip_patterns = {}
        if catalog and "skip_databases" in catalog:
            skip_dbs = catalog["skip_databases"]
            for skip_name, skip_info in skip_dbs.items():
                if isinstance(skip_info, dict) and "table_patterns" in skip_info:
                    skip_patterns[skip_name] = skip_info["table_patterns"]

        if not skip_patterns:
            logger.warning("No skip_databases found in catalog")

        return skip_patterns

    except ImportError as e:
        logger.warning(f"Could not import CatalogManager: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Error loading skip_databases from catalog: {e}")
        import traceback

        traceback.print_exc()
        return {}


def should_skip_database(
    db_path: Path,
    skip_patterns: dict[str, list[str]],
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
) -> tuple[bool, str | None]:
    """Check if a database should be skipped based on catalog skip patterns.

    Args:
        db_path: Path to database to check
        skip_patterns: Dict of skip_name -> table_patterns from catalog
        ignore_tables: Tables to ignore when introspecting
        ignore_prefixes: Table prefixes to ignore
        ignore_suffixes: Table suffixes to ignore

    Returns:
        Tuple of (should_skip, skip_reason)
    """
    if not skip_patterns:
        return False, None

    try:
        # Lazy import to avoid circular dependency
        from mars.pipeline.raw_scanner.db_variant_selector.db_variant_selector import introspect_db

        # Introspect the database to get its tables
        db_meta = introspect_db(db_path, ignore_tables, ignore_prefixes, ignore_suffixes)
        db_tables = set(db_meta.table_names or [])

        # Check against each skip pattern
        for skip_name, patterns in skip_patterns.items():
            matched = 0
            for pattern in patterns:
                # Check if pattern matches any table
                if "*" in pattern:
                    # Wildcard pattern
                    import fnmatch

                    for table in db_tables:
                        if fnmatch.fnmatch(table, pattern):
                            matched += 1
                            break
                else:
                    # Exact match
                    if pattern in db_tables:
                        matched += 1

            # If majority of patterns match, skip this database
            if matched >= len(patterns) * 0.7:  # 70% threshold
                return True, skip_name

        return False, None

    except Exception:
        # If introspection fails, don't skip (let normal processing handle it)
        return False, None


# ==========================================================================
# =========================== SCHEMA EXTRACTION ============================
# ==========================================================================
def parse_sqlite_master(con: sqlite3.Connection, meta: DBMeta) -> tuple[dict[str, TableDef], set[str], float]:
    tables: dict[str, TableDef] = {}
    indices: set[str] = set()
    ok = 0
    total = 0
    try:
        cur = con.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL")
        rows = cur.fetchall()
    except sqlite3.Error as e:
        meta.notes.append(f"sqlite_master unreadable: {e}")
        return tables, indices, 0.0

    for row in rows:
        total += 1
        t = (row["type"] or "").lower()
        original_name = row["name"] or ""  # Keep original for querying
        name = normalize_ident(original_name)  # Normalized for schema comparison
        sql = row["sql"] or ""
        if t == "table":
            m = CREATE_TABLE_RE.search(sql)
            is_virtual = "VIRTUAL TABLE" in sql.upper()
            if m:
                body = m.group("body")
                cols = []
                for m2 in COLUMN_NAME_RE.finditer(body):
                    c = normalize_ident(m2.group("col"))
                    if c.upper() in {
                        "PRIMARY",
                        "UNIQUE",
                        "CONSTRAINT",
                        "FOREIGN",
                        "CHECK",
                    }:
                        continue
                    cols.append(c)
                tables[name] = TableDef(
                    name=name,
                    columns=cols,
                    is_virtual=is_virtual,
                    original_name=original_name,
                )
                ok += 1
            else:
                # Heuristic salvage
                cols = []
                body = sql.partition("(")[2].rpartition(")")[0]
                for frag in body.split(","):
                    frag = frag.strip()
                    if not frag:
                        continue
                    cand = normalize_ident(frag.split()[0])
                    if re.match(r"[A-Za-z_][A-Za-z0-9_$.]*", cand):
                        cols.append(cand)
                tables[name] = TableDef(
                    name=name,
                    columns=cols,
                    is_virtual=is_virtual,
                    original_name=original_name,
                )
                ok += 1
        elif t == "index":
            indices.add(name)
            ok += 1
        else:
            ok += 1

    rate = (ok / total) if total else 0.0
    return tables, indices, rate


def byte_scan_tables(path: Path) -> dict[str, TableDef]:
    td: dict[str, TableDef] = {}
    try:
        text = path.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return td
    for m in CREATE_TABLE_RE.finditer(text):
        original_name = m.group("name")  # Keep original for querying
        name = normalize_ident(original_name)  # Normalized for schema comparison
        body = m.group("body")
        cols = []
        for m2 in COLUMN_NAME_RE.finditer(body):
            c = normalize_ident(m2.group("col"))
            if c.upper() in {"PRIMARY", "UNIQUE", "CONSTRAINT", "FOREIGN", "CHECK"}:
                continue
            cols.append(c)
        td[name] = TableDef(
            name=name,
            columns=cols,
            is_virtual=("VIRTUAL TABLE" in m.group(0).upper()),
            original_name=original_name,
        )
    return td


# ==========================================================================
# ====================== FILTERING/NORMALIZATION ===========================
# ==========================================================================
def is_ignorable_table(
    name: str,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
) -> bool:
    n = name.lower()
    if n in ignore_tables:
        return True
    if n in SALVAGE_TABLE_CANON:  # <-- exclude salvage from matching/probes
        return True
    if any(n.startswith(p) for p in ignore_prefixes):
        return True
    if any(n.endswith(s) for s in ignore_suffixes):
        return True
    return bool(n.startswith("z_") and n.endswith("_cache"))


def normalize_columns(cols: list[str]) -> set[str]:
    return {normalize_ident(c) for c in cols if c}
