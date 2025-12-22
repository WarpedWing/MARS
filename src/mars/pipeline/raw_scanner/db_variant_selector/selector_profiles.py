from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mars.pipeline.raw_scanner.db_variant_selector.selector_config import (
    PROFILE_MIN_ROWS,
    PROFILE_SCORE_THRESHOLD,
    PROFILE_TABLE_SAMPLE_LIMIT,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import Protocol

    class Variant(Protocol):
        tag: str
        path: Path
        meta: Any

    ProfileTableFn = Callable[[Any, str], dict[str, Any]]
    CompareProfilesFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    SafeConnectFn = Callable[[Path], Any | None]


@dataclass
class ProfileComputationResult:
    """Structured return payload for variant profiling."""

    info_by_path: dict[str, dict[str, Any]]
    profile_scores: list[dict[str, Any]]
    profile_score_map: dict[str, float | None]
    variants_for_choice: list[Variant]


def _candidate_tables(meta_rows: Mapping[str, int], nonnull_tables: Iterable[str]) -> list[str]:
    """Return tables ordered by rowcount, preferring those with non-null data."""
    nonnull = set(nonnull_tables)
    pairs = [
        (name, meta_rows.get(name, 0) or 0) for name in meta_rows if (meta_rows.get(name, 0) or 0) >= PROFILE_MIN_ROWS
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda item: (item[1], item[0]), reverse=True)
    preferred = [name for name, _ in pairs if name in nonnull]
    others = [name for name, _ in pairs if name not in nonnull]
    return preferred + others


def select_profile_tables(
    variants: Sequence[Variant],
    limit: int = PROFILE_TABLE_SAMPLE_LIMIT,
) -> list[str]:
    """Pick shared tables to profile by scanning variants in preference order."""
    ordered: list[str] = []
    seen: set[str] = set()
    preferred_order = ("O", "C", "R", "D")
    by_tag = {v.tag: v for v in variants}

    def _add_tables(variant: Variant) -> None:
        rows_map = variant.meta.rows_by_table or {}
        if not rows_map:
            return
        candidates = _candidate_tables(rows_map, variant.meta.nonnull_tables or [])
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
            if len(ordered) >= limit:
                return

    for tag in preferred_order:
        variant = by_tag.get(tag)
        if not variant:
            continue
        _add_tables(variant)
        if len(ordered) >= limit:
            return ordered[:limit]

    for variant in variants:
        _add_tables(variant)
        if len(ordered) >= limit:
            break

    return ordered[:limit]


def compute_variant_profiles(
    variants: Sequence[Variant],
    exemplar_path: Path | None,
    profile_tables: Sequence[str],
    *,
    safe_connect: SafeConnectFn,
    profile_table_fn: ProfileTableFn,
    compare_profiles_fn: CompareProfilesFn,
    profile_score_threshold: float = PROFILE_SCORE_THRESHOLD,
) -> ProfileComputationResult:
    """
    Build per-variant profiling metadata and determine eligible variants.

    Returns overall score structures and the filtered variant list that should
    participate in final selection.
    """
    info_by_path: dict[str, dict[str, Any]] = {}
    profile_scores: list[dict[str, Any]] = []
    exemplar_cache: dict[str, dict[str, Any]] = {}
    exemplar_conn = safe_connect(exemplar_path) if exemplar_path else None

    for variant in variants:
        info: dict[str, Any] = {
            "tables": list(profile_tables),
            "table_scores": {},
            "score": None,
        }
        rows_map: dict[str, int] = dict(variant.meta.rows_by_table or {})

        if not profile_tables:
            info["error"] = "no table"
        elif exemplar_path is None:
            info["error"] = "no exemplar"
        else:
            var_conn = safe_connect(Path(variant.path))
            if var_conn is None:
                info["error"] = "variant_open_failed"
            else:
                try:
                    for table_name in profile_tables:
                        row_count = rows_map.get(table_name, 0) or 0
                        if table_name not in rows_map or row_count <= 0:
                            info.setdefault("table_errors", {})[table_name] = "no_rows"
                            info["table_scores"][table_name] = None
                            continue
                        if row_count < PROFILE_MIN_ROWS:
                            info.setdefault("table_errors", {})[table_name] = f"fewer_than_{PROFILE_MIN_ROWS}_rows"
                            info["table_scores"][table_name] = None
                            continue
                        try:
                            if table_name not in exemplar_cache:
                                exemplar_cache[table_name] = profile_table_fn(exemplar_conn, table_name)
                            ex_prof = exemplar_cache.get(table_name)
                            if not ex_prof:
                                info["table_scores"][table_name] = None
                                continue
                            var_prof = profile_table_fn(var_conn, table_name)
                            if not var_prof:
                                info["table_scores"][table_name] = None
                                continue
                            score = compare_profiles_fn(var_prof, ex_prof).get("table_score")
                            info["table_scores"][table_name] = float(score) if score is not None else None
                        except Exception as prof_err:  # pragma: no cover
                            info.setdefault("table_errors", {})[table_name] = repr(prof_err)
                            info["table_scores"][table_name] = None

                    valid_scores = [s for s in info["table_scores"].values() if s is not None]
                    if valid_scores:
                        info["score"] = sum(valid_scores) / len(valid_scores)
                finally:
                    with contextlib.suppress(Exception):
                        if var_conn is not None:
                            var_conn.close()

        if info.get("score") is not None and info["score"] < profile_score_threshold:
            info.setdefault("warnings", []).append(
                f"combined profile score {info['score']:.3f} < {profile_score_threshold:.2f}"
            )

        info_by_path[variant.path.as_posix()] = info
        profile_scores.append(
            {
                "tag": variant.tag,
                "tables": list(profile_tables),
                "score": info.get("score"),
                "table_scores": info.get("table_scores"),
                "table_errors": info.get("table_errors"),
                "warnings": info.get("warnings"),
                "error": info.get("error"),
                "excluded": info.get("excluded_below_threshold", False),
            }
        )

    with contextlib.suppress(Exception):
        if exemplar_conn is not None:
            exemplar_conn.close()

    profile_score_map = {path: info.get("score") for path, info in info_by_path.items()}
    variants_for_choice: list[Variant] = list(variants)

    good_scores_present = any(
        (score is not None) and score >= profile_score_threshold for score in profile_score_map.values()
    )
    if good_scores_present:
        eligible_paths = {
            path for path, score in profile_score_map.items() if score is None or score >= profile_score_threshold
        }
        for path, info in info_by_path.items():
            score_val = info.get("score")
            if score_val is not None and score_val < profile_score_threshold:
                info["excluded_below_threshold"] = True
        filtered_variants = [v for v in variants if v.path.as_posix() in eligible_paths]
        if filtered_variants:
            variants_for_choice = filtered_variants

    return ProfileComputationResult(
        info_by_path=info_by_path,
        profile_scores=profile_scores,
        profile_score_map=profile_score_map,
        variants_for_choice=variants_for_choice,
    )
