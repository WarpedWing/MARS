"""Glob pattern validation and parsing for ARC entries.

Provides utilities for parsing user-entered glob patterns and validating
catalog entries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlobParseResult:
    """Result of parsing a user-entered glob pattern."""

    glob_pattern: str
    """The normalized glob pattern."""

    glob_base: str
    """The glob pattern without the terminal segment."""

    glob_terminal: str
    """The final segment of the glob pattern (filename or pattern)."""

    scope: str
    """Detected scope: 'user' or 'system'."""

    errors: list[str] = field(default_factory=list)
    """List of validation errors, if any."""

    @property
    def is_valid(self) -> bool:
        """Check if the parse result has no errors."""
        return len(self.errors) == 0


def normalize_glob_pattern(pattern: str) -> str:
    """Normalize a glob pattern for catalog storage.

    Args:
        pattern: The user-entered glob pattern.

    Returns:
        Normalized pattern with forward slashes and no leading slash.
    """
    # Convert backslashes to forward slashes
    result = pattern.replace("\\", "/")

    # Remove leading slash if present (catalog format doesn't use leading /)
    if result.startswith("/"):
        result = result[1:]

    # Remove trailing slash
    result = result.rstrip("/")

    # Collapse multiple slashes
    while "//" in result:
        result = result.replace("//", "/")

    return result


def detect_scope(pattern: str) -> str:
    """Detect scope (user or system) from a glob pattern.

    Args:
        pattern: The glob pattern to analyze.

    Returns:
        'user' if pattern is under Users/, otherwise 'system'.
    """
    normalized = normalize_glob_pattern(pattern)

    # Check for user directory patterns
    if (
        normalized.startswith("Users/*/")
        or "/Users/*/" in normalized
        or normalized.startswith("Users\\")
        or "\\Users\\" in normalized
    ):
        return "user"

    return "system"


def split_glob_terminal(pattern: str) -> tuple[str, str]:
    """Split a glob pattern into base and terminal.

    The terminal is the last path segment (filename or pattern).

    Args:
        pattern: The normalized glob pattern.

    Returns:
        Tuple of (base, terminal). Base excludes the final segment.
    """
    normalized = normalize_glob_pattern(pattern)

    if "/" not in normalized:
        # Single segment - it's all terminal
        return "", normalized

    # Split on last slash
    last_slash = normalized.rfind("/")
    base = normalized[:last_slash]
    terminal = normalized[last_slash + 1 :]

    return base, terminal


def validate_glob_terminal(terminal: str, file_type: str) -> list[str]:
    """Validate that a glob terminal is appropriate for the file type.

    Rules:
    - database: Must be filename (no wildcards) or *.ext pattern
    - log/cache: Can be *, **, **/* or specific filename

    Args:
        terminal: The terminal segment of the glob.
        file_type: The file type (database, log, cache).

    Returns:
        List of validation errors, empty if valid.
    """
    errors = []

    if not terminal:
        errors.append("Glob pattern cannot end with a slash.")
        return errors

    if file_type == "database":
        # Databases must have a specific file target, not a wildcard-all pattern
        if terminal == "*":
            errors.append(
                "Database targets must specify a filename or extension pattern "
                "(e.g., 'History.db' or '*.sqlite'), not just '*'."
            )
        elif terminal == "**/*":
            errors.append(
                "Database targets cannot use '**/*'. Use a specific filename or extension pattern (e.g., '*.db')."
            )
        elif terminal == "**":
            errors.append("Database targets cannot end with '**'.")

    return errors


def count_wildcards_in_base(glob_base: str, start_pos: int = 0) -> int:
    """Count wildcard segments in a glob base path.

    Used to detect multi-profile browser patterns.

    Args:
        glob_base: The base portion of the glob pattern.
        start_pos: Character position to start counting from.

    Returns:
        Number of wildcard segments (/*/ patterns).
    """
    # Count occurrences of /*/ in the base after start position
    search_string = glob_base[start_pos:]
    return search_string.count("/*/") + search_string.count("/*")


def user_glob_parser(glob_string: str, file_type: str = "database") -> GlobParseResult:
    """Parse and validate a user-entered glob pattern.

    Args:
        glob_string: The raw glob pattern entered by the user.
        file_type: The file type for validation (database, log, cache).

    Returns:
        GlobParseResult with parsed components and any errors.
    """
    errors: list[str] = []

    # Basic input validation
    if not glob_string or not glob_string.strip():
        return GlobParseResult(
            glob_pattern="",
            glob_base="",
            glob_terminal="",
            scope="system",
            errors=["Glob pattern cannot be empty."],
        )

    raw_input = glob_string.strip()

    # Must contain a path separator (looks like a path, not random words)
    if "/" not in raw_input and "\\" not in raw_input:
        return GlobParseResult(
            glob_pattern="",
            glob_base="",
            glob_terminal="",
            scope="system",
            errors=["Glob pattern must be a path (contain '/')."],
        )

    # No unquoted spaces (spaces in paths must be handled by quoting in the YAML)
    if " " in raw_input:
        errors.append("Glob pattern cannot contain spaces. Use paths without spaces or escape them in the YAML.")

    # Normalize the pattern
    normalized = normalize_glob_pattern(raw_input)

    # Check for invalid characters (very restrictive - filesystem-safe)
    invalid_chars = re.findall(r'[<>"|;`$!]', normalized)
    if invalid_chars:
        errors.append(f"Glob pattern contains invalid characters: {', '.join(set(invalid_chars))}")

    # Detect scope
    scope = detect_scope(normalized)

    # Split into base and terminal
    glob_base, glob_terminal = split_glob_terminal(normalized)

    # Validate terminal based on file type
    terminal_errors = validate_glob_terminal(glob_terminal, file_type)
    errors.extend(terminal_errors)

    return GlobParseResult(
        glob_pattern=normalized,
        glob_base=glob_base,
        glob_terminal=glob_terminal,
        scope=scope,
        errors=errors,
    )


def generate_exemplar_pattern(name: str, file_type: str = "database") -> str:
    """Generate an exemplar_pattern from a target name.

    Args:
        name: The target name (e.g., "Safari History").
        file_type: The file type (database, log, cache).

    Returns:
        The exemplar pattern (e.g., "databases/catalog/Safari History*").
    """
    # Map file type to folder
    folder_map = {
        "database": "databases/catalog",
        "log": "logs",
        "cache": "caches",
        "keychain": "keychains",
    }
    folder = folder_map.get(file_type, "databases/catalog")

    # Sanitize name for filesystem (keep most chars, just remove problematic ones)
    # The name is used as a directory/file prefix
    sanitized = name.strip()

    return f"{folder}/{sanitized}*"


def validate_entry(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a complete catalog entry.

    Args:
        entry: The entry dictionary to validate.

    Returns:
        Tuple of (is_valid, error_list).
    """
    errors: list[str] = []

    # Required field: name
    if not entry.get("name"):
        errors.append("Entry must have a 'name' field.")

    # Required: glob_pattern OR primary.glob_pattern
    has_glob = bool(entry.get("glob_pattern"))
    has_primary_glob = bool(
        entry.get("primary", {}).get("glob_pattern") if isinstance(entry.get("primary"), dict) else False
    )

    if not has_glob and not has_primary_glob:
        errors.append("Entry must have 'glob_pattern' or 'primary.glob_pattern'.")

    # Validate scope if present
    scope = entry.get("scope")
    if scope and scope not in ("user", "system"):
        errors.append(f"Invalid scope '{scope}'. Must be 'user' or 'system'.")

    # Validate file_type if present
    file_type = entry.get("file_type")
    valid_file_types = ("database", "log", "cache", "keychain")
    if file_type and file_type not in valid_file_types:
        errors.append(f"Invalid file_type '{file_type}'. Must be one of: {', '.join(valid_file_types)}.")

    # Validate exemplar_pattern if present
    exemplar_pattern = entry.get("exemplar_pattern")
    if exemplar_pattern and not exemplar_pattern.endswith("*"):
        errors.append("exemplar_pattern must end with '*'.")

    # Validate combine_strategy if has_archives is true
    if entry.get("has_archives"):
        strategy = entry.get("combine_strategy")
        valid_strategies = (
            "decompress_and_merge",
            "decompress_and_concatenate",
            "decompress_only",
        )
        if strategy and strategy not in valid_strategies:
            errors.append(f"Invalid combine_strategy '{strategy}'. Must be one of: {', '.join(valid_strategies)}.")

    # Validate ignorable_tables is a list
    ignorable = entry.get("ignorable_tables")
    if ignorable is not None and not isinstance(ignorable, list):
        errors.append("ignorable_tables must be a list.")

    return len(errors) == 0, errors


def has_extra_wildcards(glob_pattern: str) -> bool:
    """Check if glob pattern has wildcards beyond the Users/*/ segment.

    Used to determine if multi_profile option should be available.
    A pattern like Users/*/Library/Safari/History.db has no extra wildcards.
    A pattern like Users/*/Library/Google/Chrome/*/History has extra wildcards.

    Args:
        glob_pattern: The glob pattern to check.

    Returns:
        True if there are wildcards beyond Users/*/.
    """
    normalized = normalize_glob_pattern(glob_pattern)

    # Remove the Users/*/ prefix if present
    remainder = normalized[8:] if normalized.startswith("Users/*/") else normalized

    # Check for wildcard segments in the remainder
    return "/*/" in remainder or remainder.endswith("/*")


def suggest_multi_profile(glob_pattern: str, scope: str) -> bool:
    """Suggest whether an entry might be multi-profile.

    Multi-profile is typically for browsers where the pattern has
    an extra wildcard for profile directories (e.g., Chrome/*/History).

    Args:
        glob_pattern: The glob pattern.
        scope: The detected scope.

    Returns:
        True if the pattern looks like a multi-profile browser path.
    """
    if scope != "user":
        return False

    # Must have wildcards beyond Users/*/
    if not has_extra_wildcards(glob_pattern):
        return False

    normalized = normalize_glob_pattern(glob_pattern)

    # Common browser paths that indicate multi-profile
    browser_indicators = [
        "Google/Chrome/*/",
        "Microsoft Edge/*/",
        "Firefox/Profiles/*/",
        "BraveSoftware/Brave-Browser/*/",
        "Vivaldi/*/",
        "Opera/*/",
    ]

    return any(indicator in normalized for indicator in browser_indicators)


def infer_file_type_from_glob(glob_pattern: str) -> tuple[str | None, list[str]]:
    """Infer file_type from glob pattern and return available choices.

    Analyzes the terminal segment of the glob pattern to suggest a file type
    and filter available choices.

    Args:
        glob_pattern: The glob pattern to analyze.

    Returns:
        Tuple of (suggested_type, available_choices).
        suggested_type is None if no strong inference can be made.
        available_choices is the filtered list of valid file types.
    """
    if not glob_pattern:
        return (None, ["database", "log", "cache"])

    normalized = normalize_glob_pattern(glob_pattern)
    terminal = normalized.split("/")[-1] if "/" in normalized else normalized
    terminal_lower = terminal.lower()

    # Database extensions - strong inference
    db_extensions = (".db", ".sqlite", ".sqlite3", ".sqlitedb")
    if any(terminal_lower.endswith(ext) for ext in db_extensions):
        return ("database", ["database", "log", "cache"])

    # Log/plist extensions - strong inference
    log_extensions = (".plist", ".log", ".txt", ".asl")
    if any(terminal_lower.endswith(ext) for ext in log_extensions):
        return ("log", ["database", "log", "cache"])

    # Wildcard terminal - exclude database from choices
    # Can't target "all files" as a database entry
    if terminal == "*" or terminal == "**/*":
        return (None, ["log", "cache"])

    # No strong inference - all choices available
    return (None, ["database", "log", "cache"])
