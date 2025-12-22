"""
Cross-platform utilities for MARS.

Provides platform detection and OS-specific functionality.
"""

from __future__ import annotations

import base64
import sys
import webbrowser
from functools import lru_cache
from pathlib import Path


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith("linux")


def is_admin() -> bool:
    """
    Check if running with elevated privileges.

    Returns:
        True if running as admin/root, False otherwise.
    """
    if is_windows():
        import ctypes

        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    import os

    return os.geteuid() == 0


def sanitize_windows_filename(name: str) -> str:
    """
    Replace Windows-reserved characters with underscores.

    Windows prohibits: < > : " / \\ | ? *
    Also removes control characters (0x00-0x1F).

    Args:
        name: Filename component (not a full path)

    Returns:
        Sanitized filename safe for Windows
    """
    if not is_windows():
        return name
    # Replace reserved characters (/ and \\ are path separators, handled separately)
    reserved = '<>:"|?*'
    for char in reserved:
        name = name.replace(char, "_")
    # Remove control characters
    name = "".join(c if ord(c) >= 32 else "_" for c in name)
    return name


def sanitize_windows_path(virtual_path: str) -> str:
    """
    Sanitize each path component for Windows compatibility.

    Applies sanitize_windows_filename to each component of the path.

    Args:
        virtual_path: Path using forward slashes (e.g., from dfVFS)

    Returns:
        Path with each component sanitized for Windows
    """
    if not is_windows():
        return virtual_path
    parts = virtual_path.split("/")
    sanitized = [sanitize_windows_filename(p) for p in parts]
    return "/".join(sanitized)


@lru_cache(maxsize=1)
def get_logo_data_uri() -> str:
    """
    Get the WarpedWing Labs logo as a base64 data URI for HTML embedding.

    Using a data URI ensures the logo displays correctly on all platforms,
    avoiding issues with Windows backslash paths and browser security restrictions.

    Returns:
        Base64 data URI string for use in HTML img src attribute
    """
    # Logo is in src/resources/, not src/mars/resources/
    # __file__ is in src/mars/utils/, so go up 3 levels to src/
    logo_path = Path(__file__).parent.parent.parent / "resources" / "images" / "WarpedWingLabsLogo_Horizontal_W500.png"

    if not logo_path.exists():
        return ""

    logo_bytes = logo_path.read_bytes()
    logo_b64 = base64.b64encode(logo_bytes).decode("ascii")
    return f"data:image/png;base64,{logo_b64}"


def open_help(section: str = "") -> None:
    """Open HTML help file in default browser.

    Args:
        section: Optional anchor (e.g., "settings" opens mars_help.html#settings)
    """
    help_path = Path(__file__).parent.parent.parent / "resources" / "help" / "mars_help.html"
    if not help_path.exists():
        return  # Silently fail if help file missing

    url = help_path.as_uri()
    if section:
        url += f"#{section}"
    webbrowser.open(url)
