#!/usr/bin/env python3

"""
URL Analysis using Unfurl
by WarpedWing Labs

Extracts timestamps and IDs from URLs using the Unfurl library.
Provides context for distinguishing real timestamps from ID values.
"""

from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

try:
    import unfurl.core as unfurl_core

    UNFURL_AVAILABLE = True
except ImportError:
    UNFURL_AVAILABLE = False


class URLTimestampInfo(NamedTuple):
    """Information about timestamps/IDs extracted from URL"""

    url: str
    platform: str | None  # 'facebook', 'youtube', 'twitter', etc.
    timestamps: list[tuple[int | float, str]]  # [(value, description), ...]
    ids: list[tuple[int | str, str]]  # [(value, description), ...]
    metadata: dict  # Additional context from Unfurl


class URLContext:
    """Context about URLs and their contents for a page"""

    def __init__(self):
        self.url_infos: dict[str, URLTimestampInfo] = {}  # url -> info
        self.value_to_url: dict[int | str, str] = {}  # numeric value -> source URL
        self.timestamp_values: set[int | float] = set()  # Known timestamp values
        self.id_values: set[int | str] = set()  # Known ID values

    def add_url_info(self, info: URLTimestampInfo):
        """Add information about a URL"""
        self.url_infos[info.url] = info

        # Map values back to URLs
        for ts_val, _ in info.timestamps:
            self.timestamp_values.add(ts_val)
            self.value_to_url[ts_val] = info.url

        for id_val, _ in info.ids:
            self.id_values.add(id_val)
            self.value_to_url[id_val] = info.url

    def is_confirmed_timestamp(self, value: int | float) -> tuple[bool, str | None]:
        """Check if value is confirmed timestamp from URL"""
        if value in self.timestamp_values:
            url = self.value_to_url.get(value)
            return True, url
        return False, None

    def is_confirmed_id(self, value: int | str) -> tuple[bool, str | None]:
        """Check if value is confirmed ID from URL"""
        if value in self.id_values:
            url = self.value_to_url.get(value)
            return True, url
        return False, None

    def get_url_near_offset(
        self, target_offset: int, url_offsets: list[tuple[int, str]], window: int = 200
    ) -> str | None:
        """Find URL near a given offset"""
        for url_offset, url in url_offsets:
            if abs(url_offset - target_offset) <= window:
                return url
        return None


def parse_url_with_unfurl(url: str) -> URLTimestampInfo:
    """
    Parse URL using Unfurl to extract timestamps and IDs.

    Returns URLTimestampInfo with extracted values and context.
    """
    timestamps = []
    ids = []
    platform = None
    metadata = {}

    if not UNFURL_AVAILABLE:
        # Fallback: manual parsing
        return _parse_url_manual(url)

    try:
        # Use Unfurl to parse
        parsed = unfurl_core.unfurl(url)

        # Extract platform
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if "facebook.com" in domain or "fb.com" in domain:
            platform = "facebook"
        elif "twitter.com" in domain or "x.com" in domain:
            platform = "twitter"
        elif "instagram.com" in domain:
            platform = "instagram"
        elif "youtube.com" in domain or "youtu.be" in domain:
            platform = "youtube"
        elif "tiktok.com" in domain:
            platform = "tiktok"
        elif "linkedin.com" in domain:
            platform = "linkedin"
        elif "reddit.com" in domain:
            platform = "reddit"
        elif "discord.com" in domain:
            platform = "discord"

        # Process Unfurl output
        if hasattr(parsed, "nodes"):
            for node in parsed.nodes:
                _extract_from_unfurl_node(node, timestamps, ids, metadata)

        # Fallback: also try manual extraction
        manual_results = _parse_url_manual(url)
        timestamps.extend(manual_results.timestamps)
        ids.extend(manual_results.ids)

        # Deduplicate
        timestamps = list(set(timestamps))
        ids = list(set(ids))

    except Exception:
        # If Unfurl fails, fall back to manual parsing
        return _parse_url_manual(url)

    return URLTimestampInfo(
        url=url, platform=platform, timestamps=timestamps, ids=ids, metadata=metadata
    )


def _extract_from_unfurl_node(node, timestamps: list, ids: list, metadata: dict):
    """Extract timestamps and IDs from Unfurl node"""
    try:
        # Check node label and value
        label = getattr(node, "label", "").lower()
        value = getattr(node, "value", None)

        # Timestamp indicators
        if any(
            kw in label for kw in ["time", "date", "timestamp", "created", "modified"]
        ):
            if value and isinstance(value, (int, float)):
                timestamps.append((value, f"Unfurl: {label}"))
                metadata[label] = value

        # ID indicators
        elif any(
            kw in label for kw in ["id", "uid", "key", "token", "post", "message"]
        ):
            if value:
                ids.append((value, f"Unfurl: {label}"))
                metadata[label] = value

        # Snowflake detection
        elif "snowflake" in label.lower():
            if value:
                ids.append((value, "Unfurl: Snowflake ID"))
                # Snowflakes embed timestamps - extract it
                if isinstance(value, int) and len(str(value)) >= 18:
                    embedded_ts = (value >> 22) + 1420070400000  # Discord/Twitter epoch
                    timestamps.append(
                        (embedded_ts / 1000, "Extracted from Snowflake ID")
                    )
                    metadata["snowflake_embedded_timestamp"] = embedded_ts / 1000

        # Recurse into children
        if hasattr(node, "children"):
            for child in node.children:
                _extract_from_unfurl_node(child, timestamps, ids, metadata)

    except Exception:
        pass  # Skip problematic nodes


def _parse_url_manual(url: str) -> URLTimestampInfo:
    """
    Manual URL parsing as fallback when Unfurl not available.
    Extracts common timestamp and ID patterns.
    """
    timestamps = []
    ids = []
    platform = None
    metadata = {}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Detect platform
        if "facebook.com" in domain or "fb.com" in domain:
            platform = "facebook"
        elif "twitter.com" in domain or "x.com" in domain:
            platform = "twitter"
        elif "instagram.com" in domain:
            platform = "instagram"
        elif "youtube.com" in domain:
            platform = "youtube"

        # Parse query parameters
        query_params = parse_qs(parsed.query)

        # Common timestamp parameters
        for ts_param in ["t", "time", "timestamp", "ts", "date", "created"]:
            if ts_param in query_params:
                try:
                    val = int(query_params[ts_param][0])
                    timestamps.append((val, f"URL param: {ts_param}"))
                except ValueError:
                    pass

        # Extract Snowflake IDs from path (18-19 digits)
        snowflake_pattern = r"/(\d{18,19})(?:/|$|\?)"
        for match in re.finditer(snowflake_pattern, parsed.path):
            snowflake_id = int(match.group(1))
            ids.append((snowflake_id, f"{platform or 'unknown'} Snowflake ID"))

            # Extract embedded timestamp
            embedded_ts = (snowflake_id >> 22) + 1420070400000
            timestamps.append((embedded_ts / 1000, "Extracted from Snowflake ID"))

        # Facebook/Instagram specific patterns
        if platform in ["facebook", "instagram"]:
            # Post IDs, photo IDs, etc. (15-19 digits)
            fb_id_pattern = r"/(?:posts|p|photo|photos|videos)/(\d{15,19})"
            for match in re.finditer(fb_id_pattern, parsed.path):
                ids.append((int(match.group(1)), f"{platform} post/media ID"))

        # YouTube video position timestamp
        if platform == "youtube" and "t" in query_params:
            try:
                # YouTube t= is seconds offset, not Unix timestamp
                val = int(query_params["t"][0])
                timestamps.append((val, "YouTube video position (seconds)"))
            except ValueError:
                pass

    except Exception:
        pass

    return URLTimestampInfo(
        url=url, platform=platform, timestamps=timestamps, ids=ids, metadata=metadata
    )


def analyze_urls_in_page(page: bytes, url_offsets: list[tuple[int, str]]) -> URLContext:
    """
    Analyze all URLs found in a page to build context.

    Args:
        page: Raw page bytes
        url_offsets: List of (offset, url_string) tuples

    Returns:
        URLContext with all extracted information
    """
    context = URLContext()

    for offset, url in url_offsets:
        info = parse_url_with_unfurl(url)
        context.add_url_info(info)

    return context
