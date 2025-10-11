#!/usr/bin/env python3

"""
Generic helper utilities for plotting.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_floatable(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def try_parse_datetime(s: str) -> datetime | None:
    """
    Accept flexible inputs:
      - Raw epoch seconds (int/float)
      - YYYY-MM-DD
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
      - YYYY/MM/DD ...
      - ISO-like 'YYYY-MM-DDTHH:MM[:SS]'
    Returns naive datetime in local time.
    """
    s = (s or "").strip()
    if not s:
        return None
    # epoch?
    if is_floatable(s):
        try:
            v = float(s)
            # heuristics: treat large numbers as epoch seconds
            if 1000000000 <= v <= 9999999999:
                return datetime.fromtimestamp(v)
        except Exception:
            pass

    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def to_epoch(dt: datetime) -> float:
    # Treat naive as local time
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.astimezone(UTC).timestamp()


def human_time_from_epoch(
    epoch_s: float,
    tzmode: str,
    offset_minutes: int = 0,
) -> datetime:
    if tzmode == "UTC":
        return datetime.fromtimestamp(epoch_s, tz=UTC)
    if tzmode == "OFFSET":
        return datetime.fromtimestamp(
            epoch_s, tz=timezone(timedelta(minutes=offset_minutes))
        )
    # Local
    return datetime.fromtimestamp(epoch_s)


def _nice_bucket_seconds(cand: float) -> float:
    """Choose a human-friendly bucket size >= candidate seconds."""
    NICE = [
        1,
        2,
        5,
        10,
        15,
        30,
        60,
        120,
        300,
        600,
        900,
        1800,
        3600,
        7200,
        14400,
        21600,
        43200,
        86400,
    ]
    for n in NICE:
        if cand <= n:
            return float(n)
    return float(NICE[-1])


def bin_time_series(
    xs: list[float],
    ys: dict[str, list[float | None]],
    target_bars: int = 120,
) -> tuple[list[float], dict[str, list[float | None]], float]:
    """
    Downsample into time buckets for bar charts to keep bars visible.
    Returns (bucket_times_epoch, binned_series, suggested_bar_width_ms).
    """
    if not xs:
        return xs, ys, 0.0
    if len(xs) < 2:
        return xs, ys, 60_000.0
    start, end = xs[0], xs[-1]
    span = max(1.0, end - start)
    cand = span / max(1, target_bars)
    bucket_s = _nice_bucket_seconds(cand)
    if bucket_s <= 0:
        bucket_s = 60.0

    # Prepare accumulators per bucket and per series
    sums: dict[int, dict[str, float]] = {}
    counts: dict[int, dict[str, int]] = {}

    cols = list(ys.keys())
    for i, ts in enumerate(xs):
        bidx = int((ts - start) // bucket_s)
        srow = sums.setdefault(bidx, {})
        crow = counts.setdefault(bidx, {})
        for c in cols:
            v = ys[c][i] if i < len(ys[c]) else None
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            srow[c] = srow.get(c, 0.0) + fv
            crow[c] = crow.get(c, 0) + 1

    # Build outputs ordered by bucket index
    bidxs = sorted(sums.keys())
    bx: list[float] = [start + idx * bucket_s for idx in bidxs]
    by: dict[str, list[float | None]] = {c: [] for c in cols}
    for idx in bidxs:
        srow = sums.get(idx, {})
        crow = counts.get(idx, {})
        for c in cols:
            cnt = crow.get(c, 0)
            if cnt <= 0:
                by[c].append(None)
            else:
                by[c].append(srow.get(c, 0.0) / cnt)

    width_ms = bucket_s * 1000.0 * 0.8
    return bx, by, width_ms


def truncate_label(label: str, max_len: int = 50) -> str:
    if len(label) <= max_len:
        return label
    return "..." + label[-max_len:]


def is_binary_series(values: Iterable[float | None]) -> bool:
    """True if non-null values are a subset of {0,1}."""
    found = set()
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            return False
        if fv not in (0.0, 1.0):
            return False
        found.add(int(fv))
    return True


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return list(values)
    out: list[float | None] = []
    acc: list[float | None] = []
    s = 0.0
    c = 0
    for v in values:
        acc.append(v)
        if v is not None:
            s += float(v)
            c += 1
        if len(acc) > window:
            old = acc.pop(0)
            if old is not None:
                s -= float(old)
                c -= 1
        out.append((s / c) if c > 0 else None)
    return out


def compute_chart_title(
    base_title: str,
    xs: list[float],
    x_range: tuple[float | None, float | None] | None,
    tzmode: str,
    offset_minutes: int,
) -> tuple[str, float | None, float | None]:
    data_min = min(xs) if xs else None
    data_max = max(xs) if xs else None
    start_epoch = x_range[0] if (x_range and x_range[0] is not None) else data_min
    end_epoch = x_range[1] if (x_range and x_range[1] is not None) else data_max
    if start_epoch is None and end_epoch is None:
        return base_title, None, None
    if start_epoch is None:
        start_epoch = end_epoch
    if end_epoch is None:
        end_epoch = start_epoch
    if start_epoch > end_epoch:
        start_epoch, end_epoch = end_epoch, start_epoch
    start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
    end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    title = (
        f"{base_title}: {start_str}"
        if start_str == end_str
        else f"{base_title}: {start_str} to {end_str}"
    )
    return title, start_epoch, end_epoch
