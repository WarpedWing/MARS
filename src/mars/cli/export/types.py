"""Export types, constants, and dataclasses.

Extracted from export_packager.py for better organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

# Log folders to skip during export
LOG_FOLDERS_TO_SKIP = {
    "Unified Log (All Diagnostics)",  # Merged into .logarchive (exemplar)
    "UUID Text",  # Merged into .logarchive (exemplar)
    "WiFi (Corrupt)",  # Corrupt files, not useful for export
    "WiFi (Unknown)",  # Unclassified, not useful for export
}

# Log folders where files should be concatenated (not deduplicated) in combined exports
# These are text log folders where exemplar + candidate fragments should be merged
LOG_FOLDERS_TO_CONCATENATE = {
    "System Log",
    "WiFi Log",
    "Install Log",
}

# Map MARS log folder names to canonical macOS paths
# Used for FULL_PATH export when no provenance exists (carved candidate files)
# Note: "_user" is a placeholder replaced with actual username for user-scoped folders
LOG_FOLDER_TO_PATH: dict[str, str] = {
    # Hardcoded log types (from file_categorizer.py)
    "WiFi Log": "private/var/log/wifi.log",
    "System Log": "private/var/log/system.log",
    "Install Log": "private/var/log/install.log",
    "Apple System Log (ASL)": "private/var/log/asl",
    # WiFi plists (from artifact_recovery_catalog.yaml)
    "WiFi Known Networks (Airport legacy)": "Library/Preferences/SystemConfiguration",
    "Wi-Fi Known Networks (new)": "Library/Preferences",
    "Network Services Mapping (context)": "Library/Preferences/SystemConfiguration",
    "Network Hardware Interfaces (context)": "Library/Preferences/SystemConfiguration",
    "Wi-Fi Analytics (message tracer 1)": "Library/Preferences",
    "Wi-Fi Analytics (message tracer 2)": "Library/Preferences/SystemConfiguration",
    "Wi-Fi Analytics (explicit analytics file)": "Library/Preferences",
    "EAPOL Client (per-user)": "Users/_user/Library/Preferences",
    "EAPOL Client (system)": "Library/Preferences/SystemConfiguration",
    "DHCP Leases": "private/var/db/dhcpclient/leases",
    "WiFi Agent": "Users/_user/Library/Preferences",
    # Additional log types from catalog
    "WiFi Analytics": "private/var/db/analyticsd",
    "Diagnostic Messages ASL": "private/var/log/DiagnosticMessages",
    "AirDrop Hash": "Users/_user/Library/Sharing/AirDropHashDB",
    "Bash History": "Users/_user",
    "Zsh History": "Users/_user",
    "Zsh Sessions": "Users/_user/.zsh_sessions",
    "Recent Items": "Users/_user/Library/Application Support/com.apple.sharedfilelist",
    "Recent Application Documents": "Users/_user/Library/Application Support/com.apple.sharedfilelist/com.apple.LSSharedFileList.ApplicationRecentDocuments",
    "Spotlight Shortcuts": "Users/_user/Library/Application Support/com.apple.spotlight",
    "Spotlight Volume Index": ".Spotlight-V100",
    "Spotlight Volume Index v2": "private/var/db/Spotlight-V100",
    "Spotlight Volume Index v3": "private/var/db/Spotlight",
}


# =============================================================================
# Enums
# =============================================================================


class ExportSource(Enum):
    """Source type for export."""

    EXEMPLAR = "exemplar"
    CANDIDATE = "candidate"
    COMBINED = "combined"


class ExportStructure(Enum):
    """Directory structure type for export."""

    FLAT = "flat"  # Minimal folder structure (_system, _user_*)
    FULL_PATH = "full_path"  # Full macOS directory structure


class ExportMethod(Enum):
    """Method for exporting files."""

    COPY = "copy"
    SYMLINK = "symlink"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ExportedFile:
    """Information about an exported file."""

    mars_name: str  # MARS catalog folder name
    canonical_name: str  # Canonical macOS filename
    canonical_path: str  # Full canonical path (for full_path mode)
    source_path: Path  # Original path in MARS output
    export_path: Path  # Destination path in export
    username: str | None = None  # Username for user-scoped DBs
    scope: str = "system"  # user or system
    is_combined: bool = False  # True if from combined databases
    method: str = "copy"  # copy or symlink


@dataclass
class ExportedLog:
    """Information about an exported log or keychain file."""

    folder_name: str  # MARS folder name (e.g., "WiFi Log")
    original_filename: str  # Original macOS filename
    relative_path: str  # Original macOS path (from provenance)
    source_path: Path  # Path in MARS output
    export_path: Path  # Destination path in export
    md5_hash: str  # MD5 hash for deduplication tracking
    artifact_type: str = "log"  # "log" or "keychain"
    method: str = "copy"  # copy or symlink


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    export_dir: Path
    exported_files: list[ExportedFile] = field(default_factory=list)
    exported_logs: list[ExportedLog] = field(default_factory=list)
    exported_keychains: list[ExportedLog] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    skipped_logs: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (name, error)
    total_size: int = 0  # Total bytes exported
