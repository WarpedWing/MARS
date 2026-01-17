#!/usr/bin/env python3
"""
Rubric Matching Utilities for MARS
by WarpedWing Labs

Shared constants, timestamp validation, and type coercion utilities
for rubric-based database matching.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# ============================================================================
# Constants
# ============================================================================

# lost_and_found table from sqlite_dissect
LNF_TABLE = "lost_and_found"
LNF_ID_COL_INDEX = 3  # 4th column (0-indexed) is often the row ID

# Numeric pattern for string coercion
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$")

# Semantic pattern detection
# URL: Common protocols in databases (http, https, ftp, imap, smtp, file, etc.)
URL_RE = re.compile(r"^(https?|ftps?|imaps?|smtps?|file|mailto|local|data|ws|wss)://", re.IGNORECASE)
URLISH_RE = re.compile(
    r"^(about|message|data|chrome-extension|chrome-error):",
    re.IGNORECASE,
)
DOMAINISH_RE = re.compile(r"^(knowledge-agent)", re.IGNORECASE)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
DOMAIN_RE = re.compile(
    r"^\.?([a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?\.)+[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?\.?$",
    re.IGNORECASE,
)
# Path regex - basic check, but use is_filesystem_path() for proper validation
# The regex alone is too permissive (matches base64 starting with '/')
PATH_RE = re.compile(r"^(/|~/)[^\0]*$")
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
EMAILISH_RE = re.compile(r"^(undisclosed-recipients|unlisted-recipients):;?$", re.IGNORECASE)

# Human-readable timestamp patterns
# Matches formats like:
#   2020/07/26-20:40:53.493 (LevelDB/RocksDB logs)
#   2020-07-26T20:40:53.493Z (ISO 8601)
#   2020-07-26 20:40:53 (Common log format)
#   07/26/2020 20:40:53 (US format)
TIMESTAMP_TEXT_RE = re.compile(
    r"^\d{4}[-/]\d{2}[-/]\d{2}[-T\s]\d{2}:\d{2}:\d{2}"  # Core: YYYY-MM-DD HH:MM:SS
    r"([.,]\d{1,9})?"  # Optional: subseconds (.123 or ,123)
    r"([Z\s]|[+-]\d{2}:?\d{2})?"  # Optional: timezone (Z, +00:00, -0700)
    r"|"  # OR
    r"^\d{2}[-/]\d{2}[-/]\d{4}[-T\s]\d{2}:\d{2}:\d{2}"  # MM/DD/YYYY HH:MM:SS
    r"([.,]\d{1,9})?"  # Optional: subseconds
    r"([Z\s]|[+-]\d{2}:?\d{2})?"  # Optional: timezone
)

# Fill ratio threshold for "dense" columns
FILL_RATIO_STRONG_THRESHOLD = 0.85

# Programming case patterns (snake_case, camelCase, PascalCase, kebab-case, SCREAMING_SNAKE)
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
CAMEL_CASE_RE = re.compile(r"^[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*$")
PASCAL_CASE_RE = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+$")
KEBAB_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
SCREAMING_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

# Characters that disqualify a value from being programming_case
# (spaces, punctuation commonly found in natural language)
PROGRAMMING_CASE_PROHIBITED = set(" !@#$%^&*()+=[]{}|\\;:'\",.<>?/")

# Threshold for programming_case role detection
# At least this fraction of values must be programming_case matches
PROGRAMMING_CASE_THRESHOLD = 0.6


def is_filesystem_path(value: str) -> bool:
    """
    Check if a string is likely a filesystem path (not base64 or other encoded data).

    Base64 strings starting with '/' (like '/KS47+1OObRwg...') are NOT paths.
    Real paths have directory structure like /dir/file or ~/Documents/file.txt.

    Args:
        value: String to check

    Returns:
        True if the string appears to be a real filesystem path
    """
    if not value:
        return False

    # Must start with / or ~/
    if not (value.startswith("/") or value.startswith("~/")):
        return False

    # Check path segments for base64-like content
    segments = value.split("/")
    non_empty_segments = [s for s in segments if s]

    # Require at least one non-empty segment
    if not non_empty_segments:
        return False

    # Analyze each segment for base64-like patterns
    long_alphanumeric_count = 0
    for seg in non_empty_segments:
        # Base64 padding character = is almost never in real paths
        if "=" in seg:
            return False

        # Check for base64-like segment characteristics:
        # - Contains + (base64 char) AND
        # - Is predominantly alphanumeric with /+ chars
        # - Has length > 20 (typical base64 segments)
        # Long segments with + are likely base64 (short ones like foo+bar are OK)
        if "+" in seg and len(seg) > 20:
            return False

        # Long segments (>30 chars) that are purely alphanumeric + base64 chars
        # are likely base64, not path components
        # Real path segments are usually shorter or have dots, hyphens, underscores
        if len(seg) > 30:
            # Check if segment is base64-like (mostly alphanumeric with +/)
            base64_chars = sum(1 for c in seg if c.isalnum() or c in "+/")
            if base64_chars / len(seg) > 0.95:
                long_alphanumeric_count += 1

    # If multiple segments look like base64, it's probably encoded data
    if long_alphanumeric_count >= 2:
        return False

    # Single segment without second slash - be stricter
    if len(non_empty_segments) == 1:
        seg = non_empty_segments[0]
        # Root-level paths like /etc, /tmp are valid
        # But long strings without subdirs are suspicious
        if len(seg) > 20:
            # Long single segment - check if it looks like base64
            base64_chars = sum(1 for c in seg if c.isalnum() or c in "+/=")
            if base64_chars / len(seg) > 0.9:
                return False

    return True


def matches_programming_case(s: str) -> bool:
    """
    Check if a string matches any programming case pattern.

    Matches: snake_case, camelCase, PascalCase, kebab-case, SCREAMING_SNAKE_CASE

    Args:
        s: String to check (should already be stripped)

    Returns:
        True if string matches any programming case pattern
    """
    return bool(
        SNAKE_CASE_RE.match(s)
        or CAMEL_CASE_RE.match(s)
        or PASCAL_CASE_RE.match(s)
        or KEBAB_CASE_RE.match(s)
        or SCREAMING_SNAKE_RE.match(s)
    )


def is_multi_word_identifier(s: str) -> bool:
    """
    Check if a string appears to be a multi-word identifier.

    Multi-word identifiers have separators (underscore, hyphen) or mixed case
    that indicates multiple words joined together.

    All-caps single words (like "UUID", "HTTP", "API") are NOT multi-word -
    they're acronyms that happen to be uppercase.

    Args:
        s: String to check

    Returns:
        True if string appears to be multi-word
    """
    # Has explicit separator - definitely multi-word
    if "_" in s or "-" in s:
        return True

    # All-caps single word is NOT multi-word (it's an acronym like UUID, HTTP)
    if s.isupper():
        return False

    # Has mixed case (uppercase after first char) - indicates camelCase/PascalCase
    return bool(len(s) > 1 and any(c.isupper() for c in s[1:]))


def detect_programming_case_role(col_values: list) -> str | None:
    """
    Detect programming_case role using threshold-based matching.

    Rules:
    - 100% of multi-word values must match programming_case patterns
    - At least 60% of all non-NULL values must be programming_case matches
    - Single words are allowed but don't count toward the threshold
    - Any value with spaces/prohibited punctuation disqualifies the column

    Args:
        col_values: Sample values from column

    Returns:
        "programming_case" if column qualifies, None otherwise
    """
    total_values = 0
    programming_case_count = 0

    for val in col_values:
        if val is None or val == "":
            continue
        s = str(val).strip()
        if not s:
            continue

        total_values += 1

        # Check for prohibited characters (spaces, punctuation)
        if any(c in PROGRAMMING_CASE_PROHIBITED for c in s):
            return None  # Disqualify column

        # Determine if multi-word identifier
        if is_multi_word_identifier(s):
            # Multi-word values MUST match programming_case (100% requirement)
            if matches_programming_case(s):
                programming_case_count += 1
            else:
                return None  # Multi-word that doesn't match = disqualify
        # Single words are neutral - allowed but don't count

    if total_values == 0:
        return None

    # At least PROGRAMMING_CASE_THRESHOLD of all values must be matches
    if programming_case_count / total_values >= PROGRAMMING_CASE_THRESHOLD:
        return "programming_case"

    return None


# ============================================================================
# Epoch Conversion Utilities
# ============================================================================


def date_to_epoch(date_str: str) -> float:
    """
    Convert a date string (YYYY-MM-DD) to Unix epoch timestamp.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Unix timestamp as float (seconds since 1970-01-01 UTC)

    Raises:
        ValueError: If date string is not in valid YYYY-MM-DD format
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    return dt.timestamp()


def epoch_to_year(epoch: float) -> int:
    """
    Convert Unix epoch timestamp to year.

    Args:
        epoch: Unix timestamp (seconds since 1970-01-01)

    Returns:
        Year as integer
    """
    return datetime.fromtimestamp(epoch, tz=UTC).year


# ============================================================================
# Timestamp Format Support (Enhanced)
# ============================================================================


class TimestampFormat:
    """
    Enhanced timestamp format detection using rubric metadata.

    Supports formats detected in exemplar rubrics:
    - unix_seconds
    - unix_milliseconds
    - unix_microseconds
    - unix_nanoseconds
    - cocoa_seconds (macOS: seconds since 2001-01-01)
    - cocoa_nanoseconds
    - hfs_seconds (HFS+: seconds since 1904-01-01, used in Apple file systems)
    - webkit_microseconds (microseconds since 1601-01-01)
    - hex_timestamp (hexadecimal timestamps)
    """

    # Epoch references
    COCOA_EPOCH = 978307200.0  # 2001-01-01 00:00:00 UTC
    HFS_EPOCH = -2082844800.0  # 1904-01-01 00:00:00 UTC (in Unix time)
    WEBKIT_EPOCH = -11644473600.0  # 1601-01-01 00:00:00 UTC (in Unix time)

    # Apple NSDate sentinel values (in Cocoa seconds, i.e., seconds from 2001-01-01)
    # These represent "no date" / "distant past" and "forever" / "distant future"
    APPLE_DISTANT_PAST = -63114076800  # [NSDate distantPast] ≈ Jan 1, 0001
    APPLE_DISTANT_FUTURE = 63114076800  # [NSDate distantFuture] ≈ Dec 31, 4000

    @classmethod
    def is_apple_timestamp_sentinel(cls, value: int | float) -> bool:
        """
        Check if value is an Apple NSDate sentinel (distantPast/distantFuture).

        These are valid values in Apple/Cocoa databases representing null/empty dates.

        Args:
            value: Timestamp value to check (in Cocoa seconds)

        Returns:
            True if value matches a known Apple sentinel value
        """
        # Allow some tolerance for floating point comparison
        return abs(value - cls.APPLE_DISTANT_PAST) < 1 or abs(value - cls.APPLE_DISTANT_FUTURE) < 1

    @classmethod
    def validate_timestamp(
        cls,
        value: Any,
        format_hint: str | None = None,
        min_year: int = 2015,
        max_year: int = 2030,
    ) -> bool:
        """
        Validate if value is a plausible timestamp in the given format.

        Args:
            value: Value to check
            format_hint: Timestamp format from rubric (e.g., "unix_seconds")
                         Use "unknown" to accept any valid timestamp format.
            min_year: Minimum plausible year
            max_year: Maximum plausible year

        Returns:
            True if value is a plausible timestamp
        """
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False

        # "unknown" format: Accept any recognized timestamp format
        # Used when column name suggests timestamp but no exemplar data exists
        if format_hint == "unknown":
            detected = cls.detect_timestamp_format(val, min_year, max_year)
            return detected is not None

        # Convert to Unix seconds for validation
        unix_seconds = cls._to_unix_seconds(val, format_hint)

        if unix_seconds is None:
            return False

        # Check year bounds
        try:
            dt = datetime.fromtimestamp(unix_seconds, tz=UTC)
            return min_year <= dt.year <= max_year
        except (ValueError, OSError):
            return False

    @classmethod
    def _to_unix_seconds(cls, value: float, format_hint: str | None) -> float | None:
        """Convert timestamp to Unix seconds based on format hint."""
        if format_hint == "unix_milliseconds":
            return value / 1000.0
        if format_hint == "unix_microseconds":
            return value / 1_000_000.0
        if format_hint == "unix_nanoseconds":
            return value / 1_000_000_000.0
        if format_hint == "cocoa_seconds":
            return cls.COCOA_EPOCH + value
        if format_hint == "cocoa_nanoseconds":
            return cls.COCOA_EPOCH + (value / 1_000_000_000.0)
        if format_hint == "hfs_seconds":
            return cls.HFS_EPOCH + value
        if format_hint == "webkit_microseconds":
            return cls.WEBKIT_EPOCH + (value / 1_000_000.0)
        # Default: assume unix_seconds
        return value

    @classmethod
    def detect_timestamp_format(
        cls,
        value: Any,
        min_year: int = 2015,
        max_year: int = 2030,
    ) -> str | None:
        """
        Auto-detect if value is a timestamp in ANY known format.

        Uses conservative magnitude checks to avoid false positives.
        Small integers (< 10000) are never considered timestamps.

        Args:
            value: Value to check (typically int or float)
            min_year: Minimum plausible year
            max_year: Maximum plausible year

        Returns:
            Format name if detected (e.g., "webkit_microseconds"), else None
        """
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None

        # Handle infinity and NaN
        if not (-1e100 < val < 1e100):  # Extreme bounds check
            return None

        # Small numbers are almost never timestamps
        # This prevents false positives on small integers
        if abs(val) < 10000:
            return None

        # Try all known timestamp formats with magnitude constraints
        # Order matters: try more specific formats first
        formats_with_constraints = [
            # (format_name, min_magnitude, max_magnitude)
            ("webkit_microseconds", 1e14, 1e18),  # ~500M+ seconds in microseconds
            ("unix_nanoseconds", 1e18, 1e20),  # ~1B+ seconds in nanoseconds
            ("cocoa_nanoseconds", 6e17, 2e18),  # ~600M+ seconds in nanoseconds
            ("unix_microseconds", 1e15, 1e17),  # ~1B+ seconds in microseconds
            ("unix_milliseconds", 1e12, 1e14),  # ~1B+ seconds in milliseconds
            (
                "hfs_seconds",
                3.0e9,
                4.3e9,
            ),  # 2000-2038 in HFS+ seconds (Mac epoch: 1904)
            ("unix_seconds", 9e8, 3e9),  # 2000-2065 in Unix seconds
            ("cocoa_seconds", 1e7, 2e9),  # 2001-2064 in Cocoa seconds
        ]

        for fmt, min_mag, max_mag in formats_with_constraints:
            # Quick magnitude check first
            if not (min_mag <= abs(val) <= max_mag):
                continue

            try:
                unix_seconds = cls._to_unix_seconds(val, fmt)
                if unix_seconds is None:
                    continue

                # Check if this produces a valid date
                dt = datetime.fromtimestamp(unix_seconds, tz=UTC)
                if min_year <= dt.year <= max_year:
                    return fmt
            except (ValueError, OSError, OverflowError):
                continue

        return None


# ============================================================================
# Type Coercion & Validation
# ============================================================================


def is_null(v):
    """Check if value is NULL."""
    return v is None


def safe_float(x: Any) -> float | None:
    """Convert to float safely; return None if impossible."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_numericish_text(s: str) -> bool:
    """Check if string looks like a number."""
    return bool(NUMERIC_RE.match(s.strip()))


def is_urlish_text(s: str) -> bool:
    """Check if string looks like an url-ish thing found in some system and Chrome DBs."""
    return bool(URLISH_RE.match(s.strip()))


def coerce_numeric_kind(v):
    """
    Return ('int'|'real'|'text'|'null', value_as_python).

    Numeric-looking strings get converted to int/float for type checks.
    """
    if v is None:
        return "null", None
    if isinstance(v, (int, bool)):  # bool behaves like int in sqlite
        return "int", int(v)
    if isinstance(v, float):
        # Normalize -0.0 edge case
        return "real", 0.0 if v == 0.0 else float(v)
    if isinstance(v, (bytes, bytearray)):
        return "text", v  # Treat as text blob
    if isinstance(v, str):
        s = v.strip()
        if is_numericish_text(s):
            # Distinguish int-like vs real-like
            try:
                if "." in s or "e" in s.lower():
                    fv = float(s)
                    # Count as int if exactly integral (e.g. "100.0")
                    if fv.is_integer():
                        return "int", int(fv)
                    return "real", fv
                return "int", int(s)
            except (ValueError, TypeError):
                return "text", v
        return "text", v
    # Fallback
    return "text", v


# ============================================================================
# Semantic Pattern Detection
# ============================================================================


def detect_enhanced_pattern_type(value: Any) -> str | None:
    """
    Enhanced pattern detection including composite IDs, alphanumeric IDs, etc.

    This extends detect_pattern_type with additional patterns for better
    semantic differentiation in recovered data.
    """
    import re

    if value is None:
        return None

    # Convert to string for pattern matching
    if isinstance(value, bytes):
        try:
            s = value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
    elif not isinstance(value, str):
        return None
    else:
        s = value.strip()

    # Check for null placeholders first
    if not s or s in ("-", "NULL", "null"):
        return None

    # Try basic patterns first (url, uuid, email, path, domain)
    # Pass string value to detect_pattern_type
    basic_pattern = detect_pattern_type(s)
    if basic_pattern:
        return basic_pattern

    # Additional number-ish patterns
    # Ordered from most specific to least specific

    # Compound UUID: Multiple UUIDs concatenated
    uuid_pattern = r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    uuid_matches = re.findall(uuid_pattern, s)
    if len(uuid_matches) >= 2:
        return "compound_uuid"

    # Composite ID with UUID: number:UUID, text:UUID, etc.
    if re.search(uuid_pattern, s) and re.search(r"[:-]", s):
        return "composite_id_with_uuid"

    # Composite ID: number:number or number-number
    if re.match(r"^\d+[:-]\d+$", s):
        return "composite_id"

    # Hash-like: long hex string (32+ chars, typical for MD5/SHA hashes)
    if len(s) >= 32 and re.match(r"^[0-9a-fA-F]+$", s):
        return "hash_string"

    # Version string: Number.Number[.Number...]
    if re.match(r"^\d+(\.\d+)+$", s):
        return "version_string"

    # Programming case text: snake_case, camelCase, PascalCase, kebab-case, SCREAMING_SNAKE
    # These are common in config keys, DB column names, etc.
    # Check these before alphanumeric_id since they're more specific
    # Uses the compiled regex constants for efficiency
    if matches_programming_case(s):
        return "programming_case"

    # Alphanumeric ID: mixed letters and numbers, no spaces
    has_letter = bool(re.search(r"[a-zA-Z]", s))
    has_digit = bool(re.search(r"\d", s))
    has_space = " " in s

    if has_letter and has_digit and not has_space:
        return "alphanumeric_id"

    # Numeric string
    if is_numericish_text(s):
        return "number_string"

    return None


def detect_pattern_type(value: str) -> str | None:
    """
    Detect semantic pattern type from a string value.

    Returns:
        "url", "uuid", "timestamp_text", "email", "domain", "path", or None
    """
    if not isinstance(value, str) or not value:
        return None

    # Check URL
    if URL_RE.match(value) or URLISH_RE.match(value):
        return "url"

    # Check UUID
    if UUID_RE.match(value):
        return "uuid"

    # Check human-readable timestamp
    # Examples: 2020/07/26-20:40:53.493, 2020-07-26T20:40:53Z
    if TIMESTAMP_TEXT_RE.match(value):
        return "timestamp_text"

    # Check email (including email-like patterns like "undisclosed-recipients:;")
    if EMAIL_RE.match(value) or EMAILISH_RE.match(value):
        return "email"

    # Check domain (must come before path to avoid false positives)
    if DOMAIN_RE.match(value) or DOMAINISH_RE.match(value):
        return "domain"

    # Check filesystem path (use strict validation to avoid base64 false positives)
    if is_filesystem_path(value):
        return "path"

    return None


def infer_pattern_from_examples(examples: list) -> str | None:
    """
    Infer consistent semantic pattern from rubric examples.

    Returns:
        Pattern type if >= 80% of examples match the same pattern, else None
    """
    if not examples:
        return None

    # Count pattern types across examples
    pattern_counts: dict[str, int] = {}
    total = 0

    for ex in examples:
        if ex is None or ex == "":
            continue  # Skip null/empty

        pattern = detect_pattern_type(str(ex))
        if pattern:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total += 1

    if total == 0:
        return None

    # Find most common pattern
    if not pattern_counts:
        return None

    dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])
    pattern_type, count = dominant_pattern

    # Require 100% consistency
    if count / total == 1.0:
        return pattern_type

    return None


def value_matches_declared_type(v, decl_type: str, col_meta: dict[str, Any] | None = None) -> bool:
    """
    Strictly validate if value matches declared SQLite type.

    Uses enhanced rubric metadata for timestamp and semantic pattern validation.

    IMPORTANT: When values are read from SQLite databases (like lost_and_found),
    the Python type already reflects the stored type. We check the actual Python
    type BEFORE coercion to prevent false positives (e.g., "980" TEXT matching
    INTEGER columns).

    Rules:
    - TEXT: String-like, not purely numeric, must match semantic pattern if detected
    - INTEGER: Actual integers (not numeric strings like "980")
    - REAL: Integers or floats (not numeric strings)
    - Timestamp columns: Valid epoch value in specified format
    - Semantic patterns: URLs, UUIDs, domains, paths, emails (inferred from examples)
    """
    col_meta = col_meta or {}

    # NULL is valid for any type (unless column is NOT NULL, but we don't
    # enforce that here). In SQLite, NULL can be stored in any column
    # regardless of declared type
    if v is None:
        return True

    # Check if column has timestamp role from rubric
    role = col_meta.get("role")
    timestamp_format = col_meta.get("timestamp_format")

    # Handle both single role (string) and multi-role (list)
    roles = [role] if isinstance(role, str) else (role if role else [])

    # Handle both regular timestamp and nullable_timestamp roles
    is_timestamp_role = "timestamp" in roles or "nullable_timestamp" in roles
    if is_timestamp_role:
        # Zero is a universal "null timestamp" sentinel across all platforms
        # (Unix epoch, Cocoa Jan 1 2001, WebKit year 1601)
        if v == 0:
            return True
        # Apple sentinel values (distantPast/distantFuture) are also valid
        if isinstance(v, (int, float)) and TimestampFormat.is_apple_timestamp_sentinel(v):
            return True
        # Use enhanced timestamp validation
        return TimestampFormat.validate_timestamp(v, timestamp_format)

    # STRICT TYPE CHECK: Check actual Python type BEFORE coercion
    # If value is a Python str and target is INTEGER/REAL, reject immediately
    # This prevents "980" (TEXT) from matching INTEGER columns
    t = (decl_type or "").strip().upper()

    if isinstance(v, str):
        # Python string values can ONLY match TEXT columns (not INTEGER/REAL/BLOB)
        # Reject for INTEGER/REAL even if string looks numeric (e.g., "980")
        if "INT" in t or t in ("REAL", "DOUBLE", "FLOAT") or "FLOAT" in t or "DOUBLE" in t:
            # String cannot match numeric type (even if it looks numeric)
            return False

        # BLOB type: Reject TEXT strings for BLOB columns
        # If rubric shows TEXT data in BLOB column, effective_type will be
        # TEXT (not BLOB). So if we reach here with t="BLOB", it means rubric
        # shows binary data
        if t == "BLOB":
            # TEXT string cannot match binary BLOB column
            return False
        # Continue with TEXT validation below (after coercion check)

    # Regular type checking with coercion (for actual int/float values)
    k, vv = coerce_numeric_kind(v)

    if t == "TEXT" or t == "":
        # FTS3/4 STRICT VALIDATION (highest priority)
        # FTS virtual tables accept ANY data type and convert to TEXT, causing
        # massive false positives. Enforce strict text-only validation for FTS columns.
        is_fts = col_meta.get("is_fts", False)

        if is_fts:
            # FTS columns: STRICT text-only, reject bare integers/floats/BLOBs
            if k != "text":
                return False

            # Additional: Reject very short strings if column has long average
            # Prevents single words/numbers from matching metadata/description columns
            # Example: Reject "admin" in StringTable column with avg 100+ chars
            string_stats = col_meta.get("string_stats", {})
            avg_length = string_stats.get("avg_length", 0)
            if avg_length > 20 and isinstance(v, str) and len(v.strip()) < 5:
                # Column typically has long strings (paths, descriptions, metadata)
                # Reject suspiciously short values (single words, numbers)
                return False

        # Check for semantic role - these are strict
        role = col_meta.get("role")

        # Semantic columns (url, email, path, etc.) must be text only
        # This prevents false positives like integers matching URL columns
        # text_only: Enforces actual string values for TEXT columns with string exemplar data
        if role in (
            "url",
            "uuid",
            "email",
            "path",
            "domain",
            "timestamp_text",
            "text_only",
        ):
            if k != "text":
                return False
        else:
            # Check column name for ID heuristic
            col_name = col_meta.get("name", "").lower()

            # ID columns (file_id, fs_id, user_id, etc.) often store integers
            # even when declared as TEXT for flexibility
            if "id" in col_name:
                # Allow integers in ID columns
                if k not in ("text", "int", "real"):
                    return False
            else:
                # Generic TEXT columns accept all types (SQLite dynamic typing)
                # Integers and floats can be stored in TEXT columns and will be
                # converted to text. This handles self-match rubrics where
                # schema declares TEXT but data was stored as INTEGER.
                if k not in ("text", "int", "real"):
                    return False

        # Enhanced semantic validation using stored role field (fast path)
        # Falls back to inferring from examples if role not set (slower path)

        # Only validate semantic patterns for non-empty text values
        if isinstance(v, str) and v.strip():
            # Get expected pattern from role field first (optimization)
            # Role field stores pre-computed patterns: url, uuid, email, path, domain
            if role in (  # noqa: SIM108
                "url",
                "uuid",
                "email",
                "path",
                "domain",
                "timestamp_text",
            ):
                expected_pattern = role
            else:
                expected_pattern = None
                # No role set - infer from examples (slower fallback)
                # examples = col_meta.get("examples", [])
                # expected_pattern = (
                #     infer_pattern_from_examples(examples) if examples else None
                # )

            # Detect actual pattern in value
            actual_pattern = detect_pattern_type(v)

            # Reject if patterns don't match (bidirectional check)
            if expected_pattern and actual_pattern != expected_pattern:
                # Examples show a pattern but value doesn't match
                # e.g., examples are URLs but value is plain text
                return False

            # DISABLE this check if there are NO examples at all (empty schema)
            # Only apply for catalog matches where we have examples but they're plain text
            # examples = col_meta.get("examples", [])
            # if not expected_pattern and actual_pattern and examples:
            # Examples exist and are plain text, but value has a strong pattern
            # e.g., examples are "version" but value is a URL
            # This is almost certainly wrong for catalog matches
            # return False

        return True

    # SQLite INTEGER affinity: INT, INTEGER, TINYINT, SMALLINT, BIGINT, etc.
    if "INT" in t:
        if k == "int":
            return True
        if k == "real":
            try:
                fv = safe_float(vv)
                return fv is not None and fv.is_integer()
            except (ValueError, TypeError, AttributeError):
                return False
        return False

    # SQLite REAL affinity: REAL, DOUBLE, FLOAT, etc.
    if t in ("REAL", "DOUBLE", "FLOAT") or "FLOAT" in t or "DOUBLE" in t:
        return k in ("int", "real")

    # SQLite BLOB type: can store binary data or text, but NOT pure numeric values
    # This prevents false positives where numeric data matches BLOB columns
    if t == "BLOB":
        # Accept bytes, bytearray, or text - but not pure int/real
        # This is more conservative but prevents many false positives
        return k in ("text",)  # Only accept text-like values for BLOB

    # Unknown types: conservative check
    return k == "text"
