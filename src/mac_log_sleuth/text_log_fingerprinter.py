#!/usr/bin/env python3
"""
Text Log Fingerprinter for Mac Log Sleuth
by WarpedWing Labs

Fingerprints macOS text-based log files by analyzing their structure and content patterns.
More accurate than magic bytes for distinguishing real logs from random compressed files.

Supported log types:
- WiFi logs (wifi.log, wifi.log.*.bz2)
- System logs (system.log, system.log.*.gz)
- Install logs (install.log)
- ASL logs (*.asl - binary format, header-based detection)

Usage:
    from text_log_fingerprinter import identify_log_type, LogType

    log_type = identify_log_type(file_path)
    if log_type != LogType.UNKNOWN:
        print(f"Identified: {log_type.name}")
"""

from __future__ import annotations

import bz2
import gzip
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import BinaryIO


class LogType(Enum):
    """Identified log file types."""

    UNKNOWN = "unknown"
    WIFI_LOG = "wifi_log"
    SYSTEM_LOG = "system_log"
    INSTALL_LOG = "install_log"
    ASL_BINARY = "asl_binary"
    DIAGNOSTIC_ASL = "diagnostic_asl"
    POWER_MANAGEMENT_ASL = "power_management_asl"
    PLIST_XML = "plist_xml"
    PLIST_BINARY = "plist_binary"
    JSON = "json"
    JSONLZ4 = "jsonlz4"  # Firefox compressed JSON


@dataclass
class LogFingerprint:
    """Result of log fingerprinting analysis."""

    log_type: LogType
    confidence: float  # 0.0 to 1.0
    matched_patterns: int
    total_lines_analyzed: int
    first_timestamp: str | None
    last_timestamp: str | None
    sample_lines: list[str]
    reasons: list[str]


# ============================================================================
# Pattern Definitions
# ============================================================================

# WiFi log pattern: Thu Jul 23 00:48:51.636 <airportd[128]> message
WIFI_PATTERN = re.compile(
    r"^(?P<dow>[A-Z][a-z]{2})\s+"  # Day of week
    r"(?P<mon>[A-Z][a-z]{2})\s+"  # Month
    r"(?P<day>\d{1,2})\s+"  # Day
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+"  # HH:MM:SS.mmm
    r"<(?P<process>[^>]+)>\s+",  # <process[pid]>
    re.MULTILINE,
)

# System/Install log pattern: Nov 23 17:45:15 hostname process[pid]: message
SYSLOG_PATTERN = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+"  # Month
    r"(?P<day>\d{1,2})\s+"  # Day
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"  # HH:MM:SS
    r"(?P<host>\S+)\s+"  # Hostname
    r"(?P<process>[^\[:]+)[\[:]",  # Process name
    re.MULTILINE,
)

# WiFi-specific keywords (high confidence indicators)
WIFI_KEYWORDS = {
    b"airportd",
    b"IO80211",
    b"AirPort_Brcm",
    b"EAPOL",
    b"_processIPv4Changes",
    b"WoWEnable",
    b"link down event",
    b"Received EAPOL packet",
}

# System log keywords
SYSLOG_KEYWORDS = {
    b"syslogd",
    b"kernel",
    b"configd",
    b"opendirectoryd",
    b"com.apple",
    b"launchd",
}

# Install log keywords
INSTALL_KEYWORDS = {
    b"opendirectoryd",
    b"Language Chooser",
    b"Installer",
    b"Setting system to install",
    b"kern.boottime",
}

# Magic bytes for various formats
ASL_MAGIC = b"ASL DB"
PLIST_XML_MAGIC = b"<?xml"
PLIST_BINARY_MAGIC = b"bplist"
JSONLZ4_MAGIC = b"mozLz40\x00"  # Firefox JSONLZ4 magic


# ============================================================================
# Recovery Tool Paths
# ============================================================================


def get_recovery_tool_path(tool_name: str) -> Path | None:
    """
    Get path to recovery tool (gzrecover or bzip2recover).

    Checks in order:
    1. System PATH
    2. Bundled binaries in resources/bin/<os>/

    Args:
        tool_name: Tool name (gzrecover or bzip2recover)

    Returns:
        Path to tool if found, None otherwise
    """
    # Check PATH first
    system_tool = shutil.which(tool_name)
    if system_tool:
        return Path(system_tool)

    # Check bundled binaries
    system = platform.system().lower()
    if system == "darwin":
        os_dir = "macos"
        ext = ""
    elif system == "windows":
        os_dir = "windows"
        ext = ".exe"
    else:
        # Linux - no bundled binaries
        return None

    # Check resources/bin/<os>/
    bundled_path = (
        Path(__file__).parent.parent
        / "resources"
        / "bin"
        / os_dir
        / f"{tool_name}{ext}"
    )

    return bundled_path if bundled_path.exists() else None


# ============================================================================
# Helper Functions
# ============================================================================


def read_file_header(file_path: Path, size: int = 8192, try_recovery: bool = True) -> bytes:
    """
    Read file header, handling compressed files automatically.

    Args:
        file_path: Path to file
        size: Number of bytes to read (default: 8KB)
        try_recovery: Try recovery tools for corrupted compressed files

    Returns:
        Uncompressed header bytes
    """
    try:
        # Read more compressed data for better decompression
        # Compressed files need more input to produce same output
        with open(file_path, "rb") as f:
            header = f.read(size * 4)  # Read 4x for compression

        # Try to decompress if it looks compressed
        if header[:2] == b"\x1f\x8b":  # gzip magic
            try:
                with gzip.open(file_path, "rb") as gz:
                    return gz.read(size)
            except Exception as e:
                # Try gzrecover for corrupted files
                if try_recovery:
                    recovered = _try_gzrecover(file_path, size)
                    if recovered:
                        return recovered
        elif header[:2] == b"BZ":  # bzip2 magic
            try:
                with bz2.open(file_path, "rb") as bz:
                    return bz.read(size)
            except Exception as e:
                # Try bzip2recover for corrupted files
                if try_recovery:
                    recovered = _try_bzip2recover(file_path, size)
                    if recovered:
                        return recovered

        return header[:size]  # Return original if not compressed

    except Exception:
        return b""


def _try_gzrecover(file_path: Path, size: int) -> bytes | None:
    """
    Attempt to recover data from corrupted gzip file using gzrecover.

    Args:
        file_path: Path to corrupted gzip file
        size: Bytes to recover

    Returns:
        Recovered bytes or None if recovery failed
    """
    gzrecover = get_recovery_tool_path("gzrecover")
    if not gzrecover:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # gzrecover writes to current directory
            result = subprocess.run(
                [str(gzrecover), str(file_path)],
                cwd=tmpdir,
                capture_output=True,
                timeout=10,
            )

            # Look for recovered file
            recovered_files = list(tmpdir_path.glob("*"))
            if recovered_files:
                with open(recovered_files[0], "rb") as f:
                    return f.read(size)

    except Exception:
        pass

    return None


def _try_bzip2recover(file_path: Path, size: int) -> bytes | None:
    """
    Attempt to recover data from corrupted bzip2 file using bzip2recover.

    Args:
        file_path: Path to corrupted bzip2 file
        size: Bytes to recover

    Returns:
        Recovered bytes or None if recovery failed
    """
    bzip2recover = get_recovery_tool_path("bzip2recover")
    if not bzip2recover:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Copy file to temp dir (bzip2recover works in place)
            temp_file = tmpdir_path / file_path.name
            shutil.copy(file_path, temp_file)

            # Run bzip2recover
            result = subprocess.run(
                [str(bzip2recover), str(temp_file)],
                cwd=tmpdir,
                capture_output=True,
                timeout=10,
            )

            # Look for recovered file (usually rec*.bz2)
            recovered_files = list(tmpdir_path.glob("rec*.bz2"))
            if recovered_files:
                # Try to decompress recovered file
                with bz2.open(recovered_files[0], "rb") as bz:
                    return bz.read(size)

    except Exception:
        pass

    return None


def is_printable_text(data: bytes, min_ratio: float = 0.7) -> bool:
    """
    Check if data is mostly printable text.

    Args:
        data: Bytes to analyze
        min_ratio: Minimum ratio of printable characters required

    Returns:
        True if data is mostly printable text
    """
    if not data:
        return False

    printable_count = sum(
        1 for b in data if (32 <= b < 127) or b in (9, 10, 13)  # tab, LF, CR
    )

    return (printable_count / len(data)) >= min_ratio


def extract_lines(data: bytes, max_lines: int = 50) -> list[str]:
    """
    Extract text lines from bytes, handling encoding issues.

    Args:
        data: Bytes to parse
        max_lines: Maximum lines to extract

    Returns:
        List of text lines
    """
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        try:
            text = data.decode("latin-1", errors="replace")
        except Exception:
            return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[:max_lines]


def count_pattern_matches(lines: list[str], pattern: re.Pattern) -> int:
    """Count lines matching a regex pattern."""
    return sum(1 for line in lines if pattern.match(line))


def extract_timestamps_from_lines(lines: list[str], pattern: re.Pattern) -> list[str]:
    """Extract timestamps from lines matching pattern."""
    timestamps = []
    for line in lines:
        match = pattern.match(line)
        if match:
            try:
                # Construct timestamp string from groups
                groups = match.groupdict()
                if "dow" in groups:  # WiFi format
                    timestamps.append(
                        f"{groups['mon']} {groups['day']} {groups['time']}"
                    )
                else:  # System/Install format
                    timestamps.append(
                        f"{groups['mon']} {groups['day']} {groups['time']}"
                    )
            except Exception:
                continue

    return timestamps


def check_keyword_presence(data: bytes, keywords: set[bytes]) -> tuple[bool, int]:
    """
    Check if any keywords are present in data.

    Returns:
        (found_any, count_found)
    """
    count = sum(1 for keyword in keywords if keyword in data)
    return count > 0, count


# ============================================================================
# Log Type Detection
# ============================================================================


def detect_asl_binary(header: bytes) -> LogFingerprint | None:
    """Detect ASL binary format files."""
    if header.startswith(ASL_MAGIC):
        return LogFingerprint(
            log_type=LogType.ASL_BINARY,
            confidence=1.0,
            matched_patterns=1,
            total_lines_analyzed=0,
            first_timestamp=None,
            last_timestamp=None,
            sample_lines=[],
            reasons=["ASL DB magic bytes detected"],
        )
    return None


def detect_plist(header: bytes) -> LogFingerprint | None:
    """Detect plist files (XML or binary format)."""
    # Binary plist
    if header.startswith(PLIST_BINARY_MAGIC):
        return LogFingerprint(
            log_type=LogType.PLIST_BINARY,
            confidence=1.0,
            matched_patterns=1,
            total_lines_analyzed=0,
            first_timestamp=None,
            last_timestamp=None,
            sample_lines=[],
            reasons=["Binary plist magic bytes detected"],
        )

    # XML plist
    if header.startswith(PLIST_XML_MAGIC) or header.startswith(b"<!DOCTYPE plist"):
        # Verify it contains plist-specific tags
        if b"<plist" in header[:512]:
            return LogFingerprint(
                log_type=LogType.PLIST_XML,
                confidence=1.0,
                matched_patterns=1,
                total_lines_analyzed=0,
                first_timestamp=None,
                last_timestamp=None,
                sample_lines=[],
                reasons=["XML plist format detected"],
            )

    return None


def detect_json(header: bytes) -> LogFingerprint | None:
    """Detect JSON files."""
    if not is_printable_text(header, min_ratio=0.9):
        return None

    # Look for JSON structure markers
    text = header[:1024].decode('utf-8', errors='ignore').strip()

    # Must start with { or [
    if not (text.startswith('{') or text.startswith('[')):
        return None

    # Check for common JSON patterns
    json_indicators = [
        b'":', b'":',  # Key-value pairs
        b'"{', b'"[',  # Nested objects/arrays
        b',"', b'",',  # Comma-separated values
    ]

    indicator_count = sum(1 for indicator in json_indicators if indicator in header[:1024])

    if indicator_count >= 3:
        return LogFingerprint(
            log_type=LogType.JSON,
            confidence=min(0.6 + (indicator_count * 0.1), 1.0),
            matched_patterns=indicator_count,
            total_lines_analyzed=0,
            first_timestamp=None,
            last_timestamp=None,
            sample_lines=[text[:100]],
            reasons=[f"JSON structure detected with {indicator_count} indicators"],
        )

    return None


def detect_jsonlz4(header: bytes) -> LogFingerprint | None:
    """Detect Firefox JSONLZ4 compressed JSON files."""
    if header.startswith(JSONLZ4_MAGIC):
        return LogFingerprint(
            log_type=LogType.JSONLZ4,
            confidence=1.0,
            matched_patterns=1,
            total_lines_analyzed=0,
            first_timestamp=None,
            last_timestamp=None,
            sample_lines=[],
            reasons=["Firefox JSONLZ4 magic bytes detected"],
        )
    return None


def detect_wifi_log(header: bytes) -> LogFingerprint | None:
    """Detect WiFi log format."""
    # Quick rejection: must be text
    if not is_printable_text(header, min_ratio=0.8):
        return None

    lines = extract_lines(header, max_lines=30)
    if len(lines) < 5:
        return None

    # Count pattern matches
    pattern_matches = count_pattern_matches(lines, WIFI_PATTERN)
    match_ratio = pattern_matches / len(lines)

    # Check for WiFi-specific keywords
    has_keywords, keyword_count = check_keyword_presence(header, WIFI_KEYWORDS)

    # Build confidence score
    reasons = []
    confidence = 0.0

    if match_ratio >= 0.5:
        confidence += 0.5
        reasons.append(f"{pattern_matches}/{len(lines)} lines match WiFi pattern")

    if has_keywords:
        confidence += 0.3
        reasons.append(f"Found {keyword_count} WiFi-specific keywords")

    # Bonus: check for <process[pid]> format
    angle_bracket_count = sum(1 for line in lines if "<" in line and ">" in line)
    if angle_bracket_count >= len(lines) * 0.3:
        confidence += 0.2
        reasons.append("High density of <process> markers")

    if confidence >= 0.6:
        timestamps = extract_timestamps_from_lines(lines, WIFI_PATTERN)
        return LogFingerprint(
            log_type=LogType.WIFI_LOG,
            confidence=min(confidence, 1.0),
            matched_patterns=pattern_matches,
            total_lines_analyzed=len(lines),
            first_timestamp=timestamps[0] if timestamps else None,
            last_timestamp=timestamps[-1] if len(timestamps) > 1 else None,
            sample_lines=lines[:5],
            reasons=reasons,
        )

    return None


def detect_system_log(header: bytes) -> LogFingerprint | None:
    """Detect system.log format."""
    if not is_printable_text(header, min_ratio=0.8):
        return None

    lines = extract_lines(header, max_lines=30)
    if len(lines) < 5:
        return None

    pattern_matches = count_pattern_matches(lines, SYSLOG_PATTERN)
    match_ratio = pattern_matches / len(lines)

    has_keywords, keyword_count = check_keyword_presence(header, SYSLOG_KEYWORDS)

    reasons = []
    confidence = 0.0

    if match_ratio >= 0.5:
        confidence += 0.5
        reasons.append(f"{pattern_matches}/{len(lines)} lines match syslog pattern")

    if has_keywords:
        confidence += 0.3
        reasons.append(f"Found {keyword_count} system log keywords")

    # Check for common syslog markers
    if b"syslogd[" in header or b"kernel:" in header:
        confidence += 0.2
        reasons.append("Contains syslog daemon or kernel messages")

    if confidence >= 0.6:
        timestamps = extract_timestamps_from_lines(lines, SYSLOG_PATTERN)
        return LogFingerprint(
            log_type=LogType.SYSTEM_LOG,
            confidence=min(confidence, 1.0),
            matched_patterns=pattern_matches,
            total_lines_analyzed=len(lines),
            first_timestamp=timestamps[0] if timestamps else None,
            last_timestamp=timestamps[-1] if len(timestamps) > 1 else None,
            sample_lines=lines[:5],
            reasons=reasons,
        )

    return None


def detect_install_log(header: bytes) -> LogFingerprint | None:
    """Detect install.log format."""
    if not is_printable_text(header, min_ratio=0.8):
        return None

    lines = extract_lines(header, max_lines=30)
    if len(lines) < 5:
        return None

    pattern_matches = count_pattern_matches(lines, SYSLOG_PATTERN)
    match_ratio = pattern_matches / len(lines)

    has_keywords, keyword_count = check_keyword_presence(header, INSTALL_KEYWORDS)

    reasons = []
    confidence = 0.0

    if match_ratio >= 0.5:
        confidence += 0.4
        reasons.append(f"{pattern_matches}/{len(lines)} lines match syslog pattern")

    if has_keywords:
        confidence += 0.4
        reasons.append(f"Found {keyword_count} installation keywords")

    # Install logs often mention opendirectoryd during boot
    if b"opendirectoryd" in header and b"launched" in header:
        confidence += 0.2
        reasons.append("Contains opendirectoryd launch messages")

    if confidence >= 0.6:
        timestamps = extract_timestamps_from_lines(lines, SYSLOG_PATTERN)
        return LogFingerprint(
            log_type=LogType.INSTALL_LOG,
            confidence=min(confidence, 1.0),
            matched_patterns=pattern_matches,
            total_lines_analyzed=len(lines),
            first_timestamp=timestamps[0] if timestamps else None,
            last_timestamp=timestamps[-1] if len(timestamps) > 1 else None,
            sample_lines=lines[:5],
            reasons=reasons,
        )

    return None


# ============================================================================
# Main API
# ============================================================================


def identify_log_type(
    file_path: Path, header_size: int = 8192, min_confidence: float = 0.6
) -> LogFingerprint:
    """
    Identify log file type by analyzing its structure and content.

    Args:
        file_path: Path to file to analyze
        header_size: Bytes to read for analysis (default: 8KB)
        min_confidence: Minimum confidence threshold (default: 0.6)

    Returns:
        LogFingerprint with detection results
    """
    header = read_file_header(file_path, size=header_size)

    if not header:
        return LogFingerprint(
            log_type=LogType.UNKNOWN,
            confidence=0.0,
            matched_patterns=0,
            total_lines_analyzed=0,
            first_timestamp=None,
            last_timestamp=None,
            sample_lines=[],
            reasons=["Could not read file"],
        )

    # Try each detector in priority order
    detectors = [
        detect_asl_binary,  # Binary format - check first
        detect_plist,  # Binary/XML plist
        detect_jsonlz4,  # Firefox compressed JSON
        detect_wifi_log,  # Most distinctive pattern
        detect_install_log,  # Specific keywords
        detect_system_log,  # Generic syslog format
        detect_json,  # JSON - check last (most generic)
    ]

    best_result = None
    best_confidence = 0.0

    for detector in detectors:
        result = detector(header)
        if result and result.confidence >= min_confidence:
            if result.confidence > best_confidence:
                best_result = result
                best_confidence = result.confidence

    if best_result:
        return best_result

    # Unknown file type
    return LogFingerprint(
        log_type=LogType.UNKNOWN,
        confidence=0.0,
        matched_patterns=0,
        total_lines_analyzed=0,
        first_timestamp=None,
        last_timestamp=None,
        sample_lines=[],
        reasons=["No matching log patterns found"],
    )


def is_log_file(file_path: Path, min_confidence: float = 0.6) -> bool:
    """
    Quick check if file is a recognized log type.

    Args:
        file_path: Path to file
        min_confidence: Minimum confidence threshold

    Returns:
        True if file is identified as a log with sufficient confidence
    """
    result = identify_log_type(file_path, min_confidence=min_confidence)
    return result.log_type != LogType.UNKNOWN and result.confidence >= min_confidence


# ============================================================================
# CLI for Testing
# ============================================================================


def main():
    """CLI for testing log fingerprinting."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fingerprint macOS text log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify a single file
  python text_log_fingerprinter.py wifi.log

  # Scan a directory
  python text_log_fingerprinter.py /path/to/logs/*.log

  # Lower confidence threshold
  python text_log_fingerprinter.py --min-confidence 0.5 suspicious.log
        """,
    )
    parser.add_argument("files", nargs="+", type=Path, help="Files to analyze")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed analysis"
    )

    args = parser.parse_args()

    for file_path in args.files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}", file=sys.stderr)
            continue

        result = identify_log_type(file_path, min_confidence=args.min_confidence)

        if result.log_type != LogType.UNKNOWN:
            print(f"✅ {file_path.name}")
            print(f"   Type: {result.log_type.value}")
            print(f"   Confidence: {result.confidence:.1%}")
            if result.first_timestamp:
                print(f"   First timestamp: {result.first_timestamp}")
            if result.last_timestamp:
                print(f"   Last timestamp: {result.last_timestamp}")

            if args.verbose:
                print(f"   Matched patterns: {result.matched_patterns}")
                print(f"   Lines analyzed: {result.total_lines_analyzed}")
                print("   Reasons:")
                for reason in result.reasons:
                    print(f"     - {reason}")
                if result.sample_lines:
                    print("   Sample lines:")
                    for line in result.sample_lines[:3]:
                        print(f"     {line[:80]}")
        else:
            print(f"❓ {file_path.name}")
            print(f"   Type: Unknown")
            print(f"   Reasons: {', '.join(result.reasons)}")

        print()


if __name__ == "__main__":
    main()
