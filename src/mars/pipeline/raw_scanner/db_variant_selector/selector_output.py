from __future__ import annotations

import hashlib
from enum import Enum
from typing import TYPE_CHECKING, Any

from .schema_matcher import compute_case_schema_hash

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from mars.pipeline.raw_scanner.db_variant_selector.db_variant_selector import Variant


class VerbosityLevel(Enum):
    """JSON output verbosity levels for file size optimization.

    MINIMAL: For unmatched/empty databases (very brief)
    STANDARD: For matched databases (normal detail)
    VERBOSE: For debug/development (full detail)

    Expected size reduction: 4.7 MB → 1.5 MB (68% reduction) for typical runs
    """

    MINIMAL = "minimal"  # Only decision, basic meta, chosen variant tag
    STANDARD = "standard"  # + match details, chosen variant summary, meta snapshot
    VERBOSE = "verbose"  # + all variants, profiles, nearest exemplars


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def table_name_hashes(names: Iterable[str]) -> dict[str, str]:
    return {t: md5_hex(t.lower()) for t in sorted(names)}


def generate_schema_label(
    table_names: Iterable[str],
    schema_hash: str,
    max_table_len: int = 20,
    hash_len: int = 8,
) -> str:
    """Generate unique label for schema identification.

    Format: {TruncatedFirstTable}_{hash8}
    Example: "Messages_abc123de"

    Args:
        table_names: Table names in schema
        schema_hash: Full MD5 hash of schema
        max_table_len: Maximum length for table name portion
        hash_len: Number of hash characters to include

    Returns:
        Label string for schema identification
    """
    sorted_names = sorted(table_names) if table_names else []
    first_table = sorted_names[0] if sorted_names else "unknown"

    # Truncate and sanitize (remove special chars)
    table_part = first_table[:max_table_len]
    table_part = "".join(c if c.isalnum() else "_" for c in table_part)

    # Use first N chars of hash (handle None safely)
    hash_part = (schema_hash or "unknown")[:hash_len]

    return f"{table_part}_{hash_part}"


def determine_verbosity_level(
    matched: bool,
    empty: bool,
    has_nearest: bool = False,
    force_verbose: bool = False,
) -> VerbosityLevel:
    """Determine appropriate verbosity level for a database record.

    Args:
        matched: Whether database matched an exemplar
        empty: Whether database is empty/has no meaningful data
        has_nearest: Whether nearest exemplar data is available (--emit-nearest)
        force_verbose: Override to always use VERBOSE (for debugging)

    Returns:
        VerbosityLevel enum value

    Rules:
        - force_verbose → VERBOSE (full)
        - Empty WITH nearest data → STANDARD (preserve nearest for chimera detection)
        - Empty/unmatched → MINIMAL (brief)
        - Matched → STANDARD (normal)
    """
    if force_verbose:
        return VerbosityLevel.VERBOSE

    # Preserve nearest data for empty/X variants when --emit-nearest is set
    # This enables chimera detection (databases with mixed table types)
    if empty and has_nearest:
        return VerbosityLevel.STANDARD

    if empty or not matched:
        return VerbosityLevel.MINIMAL

    return VerbosityLevel.STANDARD


def detect_chimera(nearest: Sequence[Mapping[str, Any]] | None) -> tuple[bool, str]:
    """Detect if database is likely a chimera (mixed DB types).

    A chimera database contains tables from multiple different database types,
    typically due to corruption or byte-level data mixing. Detection indicators:
    - Multiple partial matches (>1 match with shared tables)
    - No strong/perfect match (top match < 90% similarity)

    If the top match is near-perfect (≥90%), additional matches are just noise
    from shared common table names (e.g., "thumbnails", "meta"), not chimera.

    Args:
        nearest: List of nearest exemplar matches with Jaccard scores

    Returns:
        Tuple of (is_chimera, reason)
        - is_chimera: True if chimera pattern detected
        - reason: Human-readable explanation with exemplar names

    Example:
        For a database with Chrome and Apple tables mixed:
        >>> detect_chimera([
        ...     {"exemplar": "/path/Apple_Interactions.sqlite", "tables_jaccard": 0.421, "shared": [...]},
        ...     {"exemplar": "/path/Chrome_cookies.sqlite", "tables_jaccard": 0.053, "shared": [...]},
        ... ])
        (True, "Multiple database types detected: Apple_Interactions, Chrome_cookies")
    """
    if not nearest:
        return False, ""

    from pathlib import Path

    # If top match is near-perfect (≥90%), it's not a chimera
    # Additional matches are just noise from shared common table names
    top_score = nearest[0]["tables_jaccard"]
    if top_score >= 0.9:
        return False, ""

    # Chimera detected when >1 partial match exists (after zero-match filtering)
    # Multiple weak/partial matches indicate mixed table types from different DBs
    if len(nearest) > 1:
        # Extract exemplar base names (without paths or extensions)
        exemplar_names = [Path(n["exemplar"]).stem for n in nearest]
        return True, f"Multiple database types detected: {', '.join(exemplar_names)}"

    return False, ""


def filter_record_by_verbosity(
    record: dict[str, Any],
    level: VerbosityLevel,
) -> dict[str, Any]:
    """Filter record fields based on verbosity level.

    Args:
        record: Full database record
        level: Desired verbosity level

    Returns:
        Filtered record with only relevant fields

    Field Selection:
        MINIMAL: type, case_path, decision, variant_chosen, basic meta
        STANDARD: + exact_matches, chosen_variant, sqlite_dissect, meta_snapshot
        VERBOSE: All fields (no filtering)
    """
    if level == VerbosityLevel.VERBOSE:
        # Include everything
        return record

    # MINIMAL: Core decision fields only
    minimal_record = {
        "type": record["type"],
        "case_path": record["case_path"],
        "variant_chosen": record["variant_chosen"],
        "decision": record["decision"],
        "meta_snapshot": {
            "opens": record["meta_snapshot"]["opens"],
            "tables_n_effective": record["meta_snapshot"]["tables_n_effective"],
            "tables_effective_label": record["meta_snapshot"].get("tables_effective_label"),
            "tables_effective_md5": record["meta_snapshot"].get("tables_effective_md5"),
            "rows_total": record["meta_snapshot"]["rows_total"],
            "has_lost_and_found": record["meta_snapshot"]["has_lost_and_found"],
            "lost_and_found_tables": record["meta_snapshot"]["lost_and_found_tables"],
        },
    }

    if level == VerbosityLevel.MINIMAL:
        # Always add variant_outputs (needed for residue cleanup to delete files)
        # Empty databases still have variant files on disk that need cleanup
        minimal_record["variant_outputs"] = record["variant_outputs"]

        # Add variant scores for debugging variant selection
        if "variant_scores" in record:
            minimal_record["variant_scores"] = record["variant_scores"]

        # Add chosen_variant summary for debugging
        if "chosen_variant" in record and record["chosen_variant"]:
            minimal_record["chosen_variant"] = {
                "tag": record["chosen_variant"]["tag"],
                "priority": record["chosen_variant"]["priority"],
                "closeness": record["chosen_variant"].get("closeness", 0),
                "rows_total": record["chosen_variant"]["rows_total"],
                "tables_n_effective": record["chosen_variant"]["tables_n_effective"],
            }

        # Preserve nearest exemplars and chimera detection for X variants
        # (helps identify mixed/corrupt databases even when empty)
        if "nearest_exemplars" in record:
            minimal_record["nearest_exemplars"] = record["nearest_exemplars"]
        if "chimera_detected" in record:
            minimal_record["chimera_detected"] = record["chimera_detected"]
        if "chimera_reason" in record:
            minimal_record["chimera_reason"] = record["chimera_reason"]

        return minimal_record

    # STANDARD: Add match details and chosen variant summary
    standard_record = minimal_record.copy()
    standard_record.update(
        {
            "exact_matches": record["exact_matches"],
            "chosen_variant": {
                "tag": record["chosen_variant"]["tag"],
                "priority": record["chosen_variant"]["priority"],
                "path": record["chosen_variant"]["path"],
                "rows_total": record["chosen_variant"]["rows_total"],
                "tables_n_effective": record["chosen_variant"]["tables_n_effective"],
                "best_profile_table": record["chosen_variant"].get("best_profile_table"),
                "best_profile_score": record["chosen_variant"].get("best_profile_score"),
            },
            "sqlite_dissect": record["sqlite_dissect"],
            "meta_snapshot": record["meta_snapshot"],  # Full meta snapshot for matched
        }
    )

    # Always add variant_outputs (needed for residue cleanup to delete files)
    # Even empty databases have variant files on disk that need cleanup
    standard_record["variant_outputs"] = record["variant_outputs"]

    # Add variant scores for debugging variant selection
    if "variant_scores" in record:
        standard_record["variant_scores"] = record["variant_scores"]

    # Preserve nearest exemplars and chimera detection for STANDARD verbosity
    # (needed for matched+empty and unmatched+empty cases)
    if "nearest_exemplars" in record:
        standard_record["nearest_exemplars"] = record["nearest_exemplars"]
    if "chimera_detected" in record:
        standard_record["chimera_detected"] = record["chimera_detected"]
    if "chimera_reason" in record:
        standard_record["chimera_reason"] = record["chimera_reason"]

    return standard_record


def aggregate_table_names_md5(names: Iterable[str]) -> str:
    payload = "\n".join(sorted(n.lower() for n in names))
    return md5_hex(payload)


def build_case_record(
    *,
    case_path: Path,
    best: Variant,
    best_priority: int,
    best_closeness: int,
    variants: Sequence[Variant],
    variant_score_map: Mapping[str, Sequence[Any]],
    variant_priorities: Mapping[str, Sequence[int]],
    lf_tables_from_recover: Iterable[str],
    saved_paths: Mapping[str, str],
    exact_matches: Sequence[Mapping[str, Any]],
    empty_bool: bool,
    empty_reason: str,
    dissect_attempted: bool,
    dissect_pass: bool,
    dissect_rebuilt: bool,
    dissect_rebuilt_path: str | None,
    nearest: Sequence[Mapping[str, Any]] | None = None,
    verbosity: VerbosityLevel | None = None,
) -> dict[str, Any]:
    case = best.meta
    lf_tables = sorted(set(lf_tables_from_recover))

    best_score_tuple = tuple(variant_score_map.get(best.path.as_posix(), ()))
    chosen_variant = {
        "tag": best.tag,
        "priority": best_priority,
        "closeness": best_closeness,
        "score": (
            best_score_tuple
            if best_score_tuple
            else (
                1 if case.integrity_ok else 0,
                int(case.total_rows or 0),
                len(case.nonempty_tables or []),
                len(case.effective_table_names or []),
                case.ddl_parse_rate or 0.0,
                1 if case.opens else 0,
            )
        ),
        "path": best.path.as_posix(),
        "rows_total": int(case.total_rows or 0),
        "tables_n_effective": len(case.effective_table_names or []),
        "notes": case.notes,
    }

    variant_scores_entries = []
    for variant in variants:
        v_priority, v_closeness = variant_priorities.get(variant.tag, (0, 0))
        score_tuple = tuple(variant_score_map.get(variant.path.as_posix(), ()))
        entry = {
            "tag": variant.tag,
            # "path": variant.path.as_posix(),
            "priority": v_priority,
            "closeness": v_closeness,
            "score_tuple": (
                score_tuple
                if score_tuple
                else (
                    1 if variant.meta.integrity_ok else 0,
                    int(variant.meta.total_rows or 0),
                    len(variant.meta.nonempty_tables or []),
                    len(variant.meta.effective_table_names or []),
                    variant.meta.ddl_parse_rate or 0.0,
                    1 if variant.meta.opens else 0,
                )
            ),
            "tables_n_effective": len(variant.meta.effective_table_names),
            "tables_effective_md5": compute_case_schema_hash(variant.meta),
            "rows_total": int(variant.meta.total_rows or 0),
            "nonempty_n": len(variant.meta.nonempty_tables),
            "integrity_ok": variant.meta.integrity_ok,
            "ddl_parse_rate": variant.meta.ddl_parse_rate,
            "opens": variant.meta.opens,
            "notes": variant.meta.notes,
        }
        variant_scores_entries.append(entry)

    record = {
        "type": "case",
        "case_path": case_path.as_posix(),
        "chosen_variant": chosen_variant,
        "variant_chosen": best.tag,
        "decision": {
            "matched": bool(exact_matches),
            "match_count": len(exact_matches),
            "policy": "tables_equal+columns_equal",
            "empty": empty_bool,
            "empty_reason": empty_reason,
        },
        "exact_matches": list(exact_matches),
        "sqlite_dissect": {
            "attempted": bool(dissect_attempted),
            "pass": bool(dissect_pass),
            "rebuilt": bool(dissect_rebuilt),
            "rebuilt_path": dissect_rebuilt_path,
        },
        "meta_snapshot": {
            "opens": next(
                (v["opens"] for v in variant_scores_entries if v["tag"] == best.tag),
                case.opens,  # fallback
            ),
            "tables_n_raw": len(case.table_names),
            "tables_n_effective": len(case.effective_table_names),
            "tables_effective_md5": compute_case_schema_hash(case),
            "tables_effective_label": generate_schema_label(
                case.effective_table_names or set(),
                compute_case_schema_hash(case),
            ),
            "rows_total": int(case.total_rows or 0),
            # "rowcounts_by_table": {
            #     name: int(count)
            #     for name, count in sorted((case.rows_by_table or {}).items())
            # },
            "nonempty_n": len(case.nonempty_tables),
            "nonnull_n": len(case.nonnull_tables),
            "notes": case.notes,
            "has_lost_and_found": bool(lf_tables),
            "lost_and_found_tables": lf_tables,
        },
        "variant_outputs": dict(saved_paths),
        "variant_scores": variant_scores_entries,
    }

    if nearest:
        record["nearest_exemplars"] = list(nearest)

        # Chimera detection for empty/X variants with nearest data
        # Helps identify corrupt databases with mixed table types
        if empty_bool:
            is_chimera, chimera_reason = detect_chimera(nearest)
            if is_chimera:
                record["chimera_detected"] = True
                record["chimera_reason"] = chimera_reason

    # Apply verbosity filtering if requested
    if verbosity is not None:
        return filter_record_by_verbosity(record, verbosity)

    # Auto-determine verbosity based on outcome (if not explicitly set)
    # This provides smart defaults: minimal for unmatched, standard for matched
    # Preserve nearest data for empty databases when --emit-nearest is set
    auto_verbosity = determine_verbosity_level(
        matched=bool(exact_matches),
        empty=empty_bool,
        has_nearest=bool(nearest),
    )
    return filter_record_by_verbosity(record, auto_verbosity)
