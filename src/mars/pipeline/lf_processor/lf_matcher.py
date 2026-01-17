"""
Lost and Found table matching logic.

Matches LF split tables against exemplar rubrics with strict type checking,
nfield validation, and confidence scoring.
"""

import gc
import json
import sqlite3
from pathlib import Path
from typing import Any

from mars.config.schema import GLOBAL_IGNORABLE_TABLES
from mars.pipeline.matcher.rubric_generator import generate_table_rubric

# Build lowercase set of ignorable tables for case-insensitive matching
_IGNORABLE_TABLES_LOWER = {t.lower() for t in GLOBAL_IGNORABLE_TABLES}


# Role normalization constants and functions (moved outside loop for performance)
_ROLE_NORMALIZATIONS = {
    "nullable_signature": "signature",
    "nullable_timestamp": "timestamp",
}


def _normalize_role(role: str) -> str:
    """
    Normalize nullable role variants to canonical form.

    Maps 'nullable_signature' -> 'signature' and 'nullable_timestamp' -> 'timestamp'
    to allow matching between exemplars and candidates with different null handling.
    """
    return _ROLE_NORMALIZATIONS.get(role, role)


def _normalize_roles(role: str | list | None) -> list[str]:
    """
    Convert role (str or list) to normalized sorted list.

    Handles both single role (string) and multi-role (list) formats,
    returning a sorted list of normalized role names for comparison.
    """
    if not role:
        return []
    roles = [role] if isinstance(role, str) else role
    return sorted(_normalize_role(r) for r in roles)


def _roles_compatible_int(cand_set: set, ex_set: set) -> bool:
    """
    Check if signature/timestamp roles are compatible for INTEGER columns.

    On INTEGER columns, classification as 'signature' vs 'timestamp' depends on
    the value distribution, not semantic meaning. Allow cross-matching between
    signature-like and timestamp-like roles.
    """
    signature_roles = {"signature", "nullable_signature"}
    timestamp_roles = {"timestamp", "nullable_timestamp"}
    return bool(
        (cand_set & signature_roles and ex_set & timestamp_roles)
        or (cand_set & timestamp_roles and ex_set & signature_roles)
    )


def load_exemplar_rubric(rubric_path: Path) -> dict:
    """Load exemplar rubric JSON."""
    with Path.open(rubric_path) as f:
        return json.load(f)


def infer_pk_type_from_lf_table(split_db: Path, table_name: str) -> str:
    """
    Infer original table's PK type from original_id pattern.

    The `original_id` column in LF tables comes from SQLite's .recover command.
    It contains the rowid if the source table had an INTEGER PRIMARY KEY.

    Patterns:
    - original_id > 0 AND c0 populated → source had INTEGER PK, c0 is NOT the rowid
    - original_id > 0 AND c0 is NULL → source had INTEGER PK, c0 WAS the rowid alias
    - original_id NULL/0 → source likely had composite PK (no rowid to recover)

    Returns:
        "integer_pk_not_c0" - source had INTEGER PK, but c0 is a different column
        "integer_pk_as_c0" - source had INTEGER PK as rowid alias (c0 was the PK)
        "composite_pk" - source likely had composite PK or WITHOUT ROWID
        "unknown" - couldn't determine PK type
    """
    with sqlite3.connect(split_db) as con:
        cur = con.cursor()

        # Get original_id and c0 values
        try:
            cur.execute(f"SELECT original_id, c0 FROM [{table_name}]")
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return "unknown"

    if not rows:
        return "unknown"

    # Count patterns
    has_original_id_and_c0 = sum(1 for oid, c0 in rows if oid and oid > 0 and c0 is not None)
    has_original_id_null_c0 = sum(1 for oid, c0 in rows if oid and oid > 0 and c0 is None)
    no_original_id = sum(1 for oid, _ in rows if oid is None or oid == 0)

    total = len(rows)

    # If most rows have original_id with c0 populated,
    # the source table had INTEGER PK (not as c0)
    if has_original_id_and_c0 / total > 0.5:
        return "integer_pk_not_c0"  # c0 is NOT the PK

    # If most rows have original_id with c0 NULL,
    # the source table had INTEGER PK as rowid alias (c0 was rowid)
    if has_original_id_null_c0 / total > 0.5:
        return "integer_pk_as_c0"  # c0 IS the PK (rowid alias)

    # If most rows have no original_id, likely composite PK
    if no_original_id / total > 0.5:
        return "composite_pk"

    return "unknown"


def generate_lf_table_rubric(split_db: Path, table_name: str) -> dict:
    """
    Generate rubric for a lost_and_found split table.

    Filters out metadata columns (rootpgno, pgno, nfield, original_id)
    and keeps only data columns (c0, c1, c2, ...).

    Also detects NULL PK columns (c0 with >95% NULL values, likely ROWID placeholder).
    """
    with sqlite3.connect(split_db) as con:
        rubric = generate_table_rubric(
            con,
            table_name,
            stats_sample_size=10000,
            infer_fks=False,  # Skip FK inference for lost_and_found tables
        )

        # Filter out metadata columns (rootpgno, pgno, nfield, original_id)
        # Only keep data columns (c0, c1, c2, ...)
        data_columns = [
            col for col in rubric.get("columns", []) if col["name"].startswith("c") and col["name"][1:].isdigit()
        ]

        # Check if first column (c0) is likely a NULL PK column
        # If c0 has null_likelihood close to 1.0, it's probably a recovered ROWID/PK
        if data_columns and data_columns[0]["name"] == "c0":
            null_likelihood = data_columns[0].get("null_likelihood", 0.0)
            # If c0 is >95% NULL, it's likely a ROWID/auto-increment PK that wasn't in the data
            if null_likelihood > 0.95:
                data_columns[0]["is_likely_null_pk"] = True

        rubric["columns"] = data_columns

        # Infer PK type from original_id pattern
        # This helps distinguish tables that came from INTEGER PK vs composite PK sources
        rubric["pk_type"] = infer_pk_type_from_lf_table(split_db, table_name)

    return rubric


def normalize_sqlite_type(type_name: str) -> str:
    """
    Normalize SQLite declared type to its type affinity.

    SQLite has 5 type affinities: INTEGER, REAL, TEXT, BLOB, NUMERIC
    See: https://www.sqlite.org/datatype3.html
    """
    type_upper = type_name.upper()

    # INTEGER affinity
    if "INT" in type_upper:
        return "INTEGER"

    # TEXT affinity (most permissive - matches CHAR, VARCHAR, TEXT, CLOB, etc.)
    if any(t in type_upper for t in ["CHAR", "CLOB", "TEXT"]):
        return "TEXT"

    # BLOB affinity (no type or contains "BLOB")
    if "BLOB" in type_upper or not type_upper:
        return "BLOB"

    # REAL affinity (floating point)
    if any(t in type_upper for t in ["REAL", "FLOA", "DOUB"]):
        return "REAL"

    # NUMERIC affinity (everything else - NUMERIC, DECIMAL, BOOLEAN, DATE, DATETIME, TIMESTAMP)
    return "NUMERIC"


def types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two SQLite types are compatible for L&F matching.

    Allows safe widening conversions where data won't be lost:
    - INTEGER → REAL is safe (integers fit in floats without precision loss)
    - REAL → INTEGER is NOT safe (truncation/precision loss)
    - TEXT ≠ INTEGER/REAL (incompatible data types)

    This is appropriate for L&F matching because types are INFERRED from actual
    values. A column with only integer values (like sizes 128, 90) infers as
    INTEGER, but the target schema may declare REAL for flexibility.
    """
    # Normalize to type affinities
    affinity1 = normalize_sqlite_type(type1)
    affinity2 = normalize_sqlite_type(type2)

    # Exact match
    if affinity1 == affinity2:
        return True

    # INTEGER can safely widen to REAL (no data loss)
    # This handles cases where L&F infers INTEGER from whole-number values,
    # but the exemplar schema declares REAL for the column
    return affinity1 == "INTEGER" and affinity2 == "REAL"


def calculate_match_confidence(
    candidate_cols: list[dict],
    exemplar_cols: list[dict],
    column_scores: list[tuple[int, float, dict]],
    match_score: float,
    all_exemplar_tables: dict | None = None,
    original_exemplar_col_count: int | None = None,
) -> dict:
    """
    Calculate match confidence based on semantic anchors and schema specificity.

    Conservative approach: Penalize matches that lack semantic validation.
    Better to miss a correct match than accept an incorrect one.

    BUT: Reward highly unique schema fingerprints (e.g., 124 columns is extremely specific).

    Returns:
        {
            "confidence": "high" | "medium" | "low" | "very_low",
            "confidence_score": 0.0-1.0,
            "reasons": [list of reason strings],
        }
    """
    reasons = []
    penalties = []
    bonuses = []

    # 0. Check for unique schema fingerprint (BOOST confidence)
    # If this is the ONLY table with this exact column count, that's extremely strong evidence
    # Use original column count (before PK handling) for this check
    num_cols = original_exemplar_col_count if original_exemplar_col_count is not None else len(exemplar_cols)

    if all_exemplar_tables and num_cols > 10:  # Only for non-trivial schemas
        # Count how many tables have the same column count (using original counts)
        tables_with_same_count = sum(
            1 for table_rubric in all_exemplar_tables.values() if len(table_rubric.get("columns", [])) == num_cols
        )

        if tables_with_same_count == 1:
            # Unique column count - extremely strong fingerprint!
            # This is such a strong signal it should override weak individual column scores
            if num_cols >= 100:
                bonuses.append(("unique_large_schema", 0.7))
                reasons.append(f"Extremely unique schema ({num_cols} columns - only table with this count)")
            elif num_cols >= 50:
                bonuses.append(("unique_medium_schema", 0.5))
                reasons.append(f"Very unique schema ({num_cols} columns - only table with this count)")
            elif num_cols >= 20:
                bonuses.append(("unique_schema", 0.3))
                reasons.append(f"Unique schema ({num_cols} columns - only table with this count)")
        elif tables_with_same_count == 2 and num_cols >= 50:
            # Rare column count
            bonuses.append(("rare_schema", 0.2))
            reasons.append(f"Rare schema ({num_cols} columns - only 2 tables with this count)")

    # 1. Assess semantic anchor strength in exemplar
    # NOTE: Type mismatch checking was removed - tables with ANY type incompatibility
    # are rejected entirely in compare_rubrics() before reaching this function
    semantic_anchor_score = 0.0
    total_possible_anchors = 0

    for ex_col in exemplar_cols:
        # Check for role (strongest anchor)
        if ex_col.get("role"):
            semantic_anchor_score += 3.0
        total_possible_anchors += 3.0

        # Check for NULL likelihood data (medium anchor)
        if "null_likelihood" in ex_col:
            semantic_anchor_score += 2.0
        total_possible_anchors += 2.0

        # Check for examples (medium anchor)
        if ex_col.get("examples"):
            semantic_anchor_score += 2.0
        total_possible_anchors += 2.0

        # Check for timestamp format (strong anchor if present)
        if ex_col.get("timestamp_format"):
            semantic_anchor_score += 2.0
        total_possible_anchors += 2.0

    anchor_strength = semantic_anchor_score / total_possible_anchors if total_possible_anchors > 0 else 0.0

    if anchor_strength < 0.1:
        penalties.append(("no_semantic_anchors", 0.4))
        reasons.append("Exemplar has no semantic anchors (no roles, NULL data, or examples)")
    elif anchor_strength < 0.3:
        penalties.append(("weak_semantic_anchors", 0.2))
        reasons.append("Exemplar has very weak semantic anchors")

    # 2. Check for schema specificity
    num_cols = len(exemplar_cols)

    # Count unique types in exemplar
    exemplar_types = set()
    for ex_col in exemplar_cols:
        ex_type = ex_col.get("type", "")
        if "inferred_type" in ex_col:
            ex_type = ex_col["inferred_type"]
        exemplar_types.add(ex_type.upper())

    # Generic schemas (all same type, few columns)
    if num_cols <= 2 and len(exemplar_types) == 1:
        penalties.append(("generic_schema", 0.3))
        type_name = list(exemplar_types)[0] if exemplar_types else "UNKNOWN"
        reasons.append(f"Very generic schema ({num_cols} columns, all {type_name})")
    elif num_cols <= 3 and len(exemplar_types) <= 2:
        penalties.append(("simple_schema", 0.15))
        reasons.append(f"Simple schema ({num_cols} columns, {len(exemplar_types)} types)")

    # 3. Check for candidate semantic types without exemplar validation
    # NOTE: Only apply this check if exemplar has reasonable semantic anchors.
    # For weak exemplars (no roles, no examples), candidate having strong semantics
    # is actually GOOD evidence, not bad. We only penalize when exemplar COULD have
    # validated the semantics but didn't (i.e., has anchors but they don't match).
    unvalidated_semantics = 0
    conflicting_semantics = 0
    total_candidate_semantics = 0

    def roles_to_set(role):
        """Convert role (str or list) to set for comparison.

        Normalizes timestamp variants (timestamp, nullable_timestamp) to a common
        base 'timestamp' so they compare as compatible for matching purposes.
        """
        if not role:
            return set()
        roles = {role} if isinstance(role, str) else set(role)
        # Normalize timestamp variants to common base
        # nullable_timestamp is a specialization of timestamp
        if "nullable_timestamp" in roles:
            roles.add("timestamp")
        return roles

    for cand_col in candidate_cols:
        cand_role = cand_col.get("role", "")
        # If candidate has a semantic role
        if cand_role and cand_role != "":
            total_candidate_semantics += 1
            # Check if corresponding exemplar column has that role
            cand_idx = candidate_cols.index(cand_col)
            if cand_idx < len(exemplar_cols):
                ex_role = exemplar_cols[cand_idx].get("role", "")
                if not ex_role:
                    # Exemplar has no role - just unvalidated (not conflicting)
                    unvalidated_semantics += 1
                else:
                    # Both have roles - check if they overlap
                    cand_roles_set = roles_to_set(cand_role)
                    ex_roles_set = roles_to_set(ex_role)
                    if not (cand_roles_set & ex_roles_set):
                        # No overlap - conflicting semantics (more serious)
                        conflicting_semantics += 1

    # Conflicting semantics are always penalized (different roles = bad match)
    if conflicting_semantics > 0:
        conflict_ratio = conflicting_semantics / total_candidate_semantics
        if conflict_ratio > 0.2:
            penalties.append(("conflicting_semantics", 0.35))
            reasons.append(
                f"Candidate has {conflicting_semantics}/{total_candidate_semantics} semantic type(s) that CONFLICT with exemplar ({conflict_ratio:.0%})"
            )
        else:
            penalties.append(("conflicting_semantics", 0.15))
            reasons.append(
                f"Candidate has {conflicting_semantics}/{total_candidate_semantics} conflicting semantic type(s)"
            )

    # Unvalidated semantics are only penalized if exemplar has decent semantic anchors
    # If exemplar is weak (anchor_strength < 0.2), don't penalize candidate for
    # having semantics that couldn't be validated - the exemplar just lacks data
    if (
        unvalidated_semantics > 0
        and total_candidate_semantics > 0
        and anchor_strength >= 0.2  # Only penalize if exemplar has decent anchors
    ):
        unvalidated_ratio = unvalidated_semantics / total_candidate_semantics

        if unvalidated_ratio > 0.5:
            # More than half unvalidated - moderate penalty
            penalties.append(("unvalidated_semantics", 0.15))
            reasons.append(
                f"Candidate has {unvalidated_semantics}/{total_candidate_semantics} semantic type(s) that couldn't be validated ({unvalidated_ratio:.0%})"
            )
        elif unvalidated_ratio > 0.3:
            # Significant proportion unvalidated - small penalty
            penalties.append(("unvalidated_semantics", 0.08))
            reasons.append(
                f"Candidate has {unvalidated_semantics}/{total_candidate_semantics} semantic type(s) that couldn't be validated ({unvalidated_ratio:.0%})"
            )
        # If <30% unvalidated, no penalty - this is acceptable variance

    # 4. Check match score quality - be VERY strict here
    # Score below 0.3 is catastrophically bad and should override other factors
    if match_score < 0.3:
        penalties.append(("catastrophic_match_score", 0.6))
        reasons.append(f"Catastrophically low match score ({match_score:.3f})")
    elif match_score < 0.5:
        penalties.append(("low_match_score", 0.3))
        reasons.append(f"Low match score ({match_score:.3f})")

    # Calculate final confidence score
    confidence_score = 1.0

    # Apply penalties
    for _, penalty in penalties:
        confidence_score -= penalty

    # Apply bonuses (can exceed 1.0 if schema is extremely unique)
    for _, bonus in bonuses:
        confidence_score += bonus

    # Clamp to 0.0-1.5 (allow bonuses to push above 1.0 for unique schemas)
    confidence_score = max(0.0, min(1.5, confidence_score))

    # Determine confidence level
    if confidence_score >= 0.75:
        confidence = "high"
    elif confidence_score >= 0.5:
        confidence = "medium"
    elif confidence_score >= 0.25:
        confidence = "low"
    else:
        confidence = "very_low"

    if not reasons:
        reasons.append("Strong semantic validation and specific schema")

    return {
        "confidence": confidence,
        "confidence_score": confidence_score,
        "reasons": reasons,
    }


def compare_rubrics(
    candidate_rubric: dict,
    exemplar_rubric: dict,
    all_exemplar_tables: dict | None = None,
) -> dict:
    """
    Compare a candidate LF table rubric against an exemplar table rubric.

    Handles NULL PK columns: If candidate's first column is mostly NULL and
    exemplar's first column is a PK, both are skipped for comparison since
    the NULL column is a placeholder for the unavailable ROWID/auto-increment PK.

    Returns:
        {
            "score": 0.0-1.0,
            "matched_columns": int,
            "total_columns": int,
            "column_scores": [(col_idx, score, details), ...],
            "confidence": "high" | "medium" | "low" | "very_low",
            "confidence_score": 0.0-1.5,
            "confidence_reasons": [str, ...],
        }
    """
    candidate_cols = candidate_rubric.get("columns", [])
    exemplar_cols = exemplar_rubric.get("columns", [])

    # Store original column counts before PK handling (for schema uniqueness check)
    original_exemplar_col_count = len(exemplar_cols)

    if not candidate_cols or not exemplar_cols:
        return {"score": 0.0, "matched_columns": 0, "total_columns": 0}

    # Check for NULL PK column case:
    # If candidate's first column is >95% NULL AND exemplar's first column is a PK,
    # skip both columns (they represent the same ROWID/auto-increment PK)
    if (
        candidate_cols
        and exemplar_cols
        and candidate_cols[0].get("is_likely_null_pk", False)
        and exemplar_cols[0].get("primary_key", False)
    ):
        # Create new lists without the first column
        candidate_cols = list(candidate_cols[1:])
        exemplar_cols = list(exemplar_cols[1:])

    if not candidate_cols or not exemplar_cols:
        return {"score": 0.0, "matched_columns": 0, "total_columns": 0}

    # Must have same number of columns (after PK handling)
    if len(candidate_cols) != len(exemplar_cols):
        return {"score": 0.0, "matched_columns": 0, "total_columns": len(exemplar_cols)}

    # Compare each column position
    column_scores = []
    total_score = 0.0

    for idx, (cand_col, ex_col) in enumerate(zip(candidate_cols, exemplar_cols)):
        col_score = 0.0
        details = {}

        # 1. Check type compatibility (30% of score) - STRICT REQUIREMENT
        cand_type = cand_col.get("type", "").upper()
        ex_type = ex_col.get("type", "").upper()

        # Use inferred type if available, but only if it's a SQLite base type
        # (not a semantic pattern like EMAIL, UUID, etc.)
        SQLITE_BASE_TYPES = {
            "INTEGER",
            "REAL",
            "TEXT",
            "BLOB",
            "NUMERIC",
            "VARCHAR",
            "TIMESTAMP",
        }
        if "inferred_type" in cand_col:
            inferred = cand_col["inferred_type"].upper()
            if any(base in inferred for base in SQLITE_BASE_TYPES):
                cand_type = inferred
        if "inferred_type" in ex_col:
            inferred = ex_col["inferred_type"].upper()
            if any(base in inferred for base in SQLITE_BASE_TYPES):
                ex_type = inferred

        # STRICT TYPE CHECK: If base types are incompatible, REJECT THE ENTIRE TABLE
        # EXCEPTION: If candidate column is all-NULL (>95% NULL), we can't know its actual type,
        # so skip type checking. But if candidate has data, it MUST match the exemplar schema,
        # even if the exemplar has no data rows (the schema is the authoritative contract).
        cand_null_likelihood = cand_col.get("null_likelihood", 0.0)
        ex_null_likelihood = ex_col.get("null_likelihood", 0.0)

        is_all_null = cand_null_likelihood > 0.95  # Only check if candidate has no data

        if not is_all_null and not types_compatible(cand_type, ex_type):
            # Type mismatch with actual data is a dealbreaker - ENTIRE TABLE REJECTED
            # You cannot store incompatible data in a schema
            return {
                "score": 0.0,
                "matched_columns": 0,
                "total_columns": len(candidate_cols),
                "column_scores": [],
                "confidence": "very_low",
                "confidence_score": 0.0,
                "confidence_reasons": [
                    f"Type incompatibility at column {idx}: {cand_col.get('name', f'col{idx}')} "
                    f"({cand_type}) cannot fit into {ex_col.get('name', f'col{idx}')} ({ex_type})"
                ],
            }

        # Types are compatible - give credit
        if is_all_null:
            # All-NULL column: can't verify type, give partial credit
            col_score += 0.15
            details["type_match"] = "null_column"
        elif cand_type == ex_type:
            col_score += 0.3
            details["type_match"] = "exact"
        else:
            col_score += 0.15
            details["type_match"] = "compatible"

        # 1.5. Numeric range validation (BONUS: up to +0.15 for good range match)
        # If both columns are numeric and exemplar has stats, validate ranges
        if {cand_type, ex_type} <= {"INTEGER", "REAL"} and "stats" in ex_col and "stats" in cand_col:
            ex_stats = ex_col["stats"]
            cand_stats = cand_col["stats"]

            # Get min/max from both
            ex_min = ex_stats.get("min")
            ex_max = ex_stats.get("max")
            cand_min = cand_stats.get("min")
            cand_max = cand_stats.get("max")

            if all(v is not None for v in [ex_min, ex_max, cand_min, cand_max]):
                # Calculate how much overlap there is between the ranges
                # Perfect overlap = candidate range fully within exemplar range
                # Partial overlap = some overlap
                # No overlap = disjoint ranges

                # Check if candidate range is fully within exemplar range (ideal)
                if cand_min >= ex_min and cand_max <= ex_max:
                    # Perfect: candidate values fall within exemplar range
                    col_score += 0.15
                    details["numeric_range"] = "within"
                # Check if ranges overlap
                elif cand_min <= ex_max and cand_max >= ex_min:
                    # Calculate overlap percentage
                    overlap_min = max(cand_min, ex_min)
                    overlap_max = min(cand_max, ex_max)
                    overlap_size = overlap_max - overlap_min

                    cand_range = cand_max - cand_min if cand_max != cand_min else 1
                    ex_range = ex_max - ex_min if ex_max != ex_min else 1

                    # Use smaller range for overlap calculation (more conservative)
                    smaller_range = min(cand_range, ex_range)
                    overlap_ratio = overlap_size / smaller_range if smaller_range > 0 else 0

                    if overlap_ratio >= 0.8:
                        # High overlap (80%+)
                        col_score += 0.12
                        details["numeric_range"] = "high_overlap"
                    elif overlap_ratio >= 0.5:
                        # Moderate overlap (50-80%)
                        col_score += 0.08
                        details["numeric_range"] = "moderate_overlap"
                    else:
                        # Low overlap (<50%)
                        col_score += 0.04
                        details["numeric_range"] = "low_overlap"
                else:
                    # Disjoint ranges - apply penalty
                    # This penalizes matches where LF data has values completely
                    # outside the exemplar's expected range (e.g., ActivityID column
                    # with exemplar range 1-5 receiving values in hundreds/negatives)
                    details["numeric_range"] = "disjoint"
                    col_score -= 0.20

                # Additionally check mean/median proximity for fine-grained validation
                ex_mean = ex_stats.get("mean")
                cand_mean = cand_stats.get("mean")
                if ex_mean is not None and cand_mean is not None and ex_max > ex_min:
                    # If means are very close, add small bonus (avoid division by zero)
                    mean_diff_normalized = abs(ex_mean - cand_mean) / (ex_max - ex_min)
                    if mean_diff_normalized < 0.1:  # Within 10% of range
                        col_score += 0.05
                        details["numeric_mean"] = "close"

        # 2. Check semantic role (30% of score)
        cand_role = cand_col.get("role", "")
        ex_role = ex_col.get("role", "")

        # Use module-level functions for role normalization (avoids function creation per iteration)
        cand_roles_norm = _normalize_roles(cand_role)
        ex_roles_norm = _normalize_roles(ex_role)

        # Also keep original roles for compatibility check
        cand_roles = (
            [cand_role]
            if isinstance(cand_role, str) and cand_role
            else (cand_role if isinstance(cand_role, list) else [])
        )
        ex_roles = [ex_role] if isinstance(ex_role, str) and ex_role else (ex_role if isinstance(ex_role, list) else [])

        if cand_roles and ex_roles and cand_roles_norm == ex_roles_norm:
            col_score += 0.3
            details["role_match"] = "exact"
        elif (
            cand_roles
            and ex_roles
            and "INT" in cand_type
            and "INT" in ex_type
            and _roles_compatible_int(set(cand_roles), set(ex_roles))
        ):
            # INTEGER columns with signature/timestamp cross-match
            # (classification depends on value distribution, not semantic meaning)
            col_score += 0.25  # Slightly less than exact match
            details["role_match"] = "int_compatible"
        elif cand_roles or ex_roles:
            # One has a role, other doesn't - partial penalty
            col_score += 0.1
            details["role_match"] = "partial"
        else:
            # Neither has a role - neutral
            col_score += 0.15
            details["role_match"] = "neutral"

        # 2.5. String length similarity (helps disambiguate TEXT columns)
        # Tables with same column count and same roles can be distinguished by avg_length
        # (e.g., 'subjects' has longer strings than 'properties')
        # Use proportional scoring so closer matches score better
        cand_string_stats = cand_col.get("string_stats", {})
        ex_string_stats = ex_col.get("string_stats", {})
        cand_avg_len = cand_string_stats.get("avg_length", 0)
        ex_avg_len = ex_string_stats.get("avg_length", 0)

        if cand_avg_len > 0 and ex_avg_len > 0:
            # Calculate similarity ratio (0 to 1, where 1 = identical)
            ratio = min(cand_avg_len, ex_avg_len) / max(cand_avg_len, ex_avg_len)

            # Proportional scoring based on ratio:
            # ratio >= 0.8: bonus +0.15 (very similar)
            # ratio 0.5-0.8: slight penalty, proportional to distance from 0.8
            # ratio 0.3-0.5: moderate penalty, proportional
            # ratio < 0.3: heavy penalty -0.25
            if ratio >= 0.8:
                col_score += 0.15
                details["string_length"] = f"match ({ratio:.2f})"
            elif ratio >= 0.5:
                # Linear penalty from 0 at ratio=0.8 to -0.1 at ratio=0.5
                penalty = (0.8 - ratio) * (0.1 / 0.3)
                col_score -= penalty
                details["string_length"] = f"similar ({ratio:.2f})"
            elif ratio >= 0.3:
                # Linear penalty from -0.1 at ratio=0.5 to -0.2 at ratio=0.3
                penalty = 0.1 + (0.5 - ratio) * (0.1 / 0.2)
                col_score -= penalty
                details["string_length"] = f"different ({ratio:.2f})"
            else:
                # Very different lengths - heavy penalty
                col_score -= 0.25
                details["string_length"] = f"mismatch ({ratio:.2f})"

        # 3. Check NULL likelihood similarity (20% of score)
        # Note: cand_null_likelihood and ex_null_likelihood already retrieved above for type checking

        # Calculate similarity of NULL rates (1.0 = identical, 0.0 = opposite extremes)
        null_diff = abs(cand_null_likelihood - ex_null_likelihood)
        null_similarity = 1.0 - null_diff

        if null_similarity >= 0.9:
            # Very similar NULL rates (within 10%)
            col_score += 0.2
            details["null_match"] = "exact"
        elif null_similarity >= 0.7:
            # Similar NULL rates (within 30%)
            col_score += 0.15
            details["null_match"] = "similar"
        elif null_similarity >= 0.5:
            # Moderately similar (within 50%)
            col_score += 0.1
            details["null_match"] = "moderate"
        else:
            # Different NULL patterns
            col_score += 0.05
            details["null_match"] = "different"

        # 4. Check timestamp format (20% of score, only if both have timestamp role)
        if "timestamp" in cand_roles and "timestamp" in ex_roles:
            cand_format = cand_col.get("timestamp_format", "")
            ex_format = ex_col.get("timestamp_format", "")

            if cand_format == ex_format:
                col_score += 0.2
                details["timestamp_format"] = "match"
            else:
                col_score += 0.1
                details["timestamp_format"] = "mismatch"
        else:
            # Not a timestamp column - give default points
            col_score += 0.2

        column_scores.append((idx, col_score, details))
        total_score += col_score

    # Overall score is average of column scores
    avg_score = total_score / len(candidate_cols) if candidate_cols else 0.0

    # PK type compatibility check
    # If LF data came from an INTEGER PK table but exemplar has composite PK (or vice versa),
    # apply a penalty since this is a strong structural mismatch
    cand_pk_type = candidate_rubric.get("pk_type", "unknown")
    ex_pk_type = exemplar_rubric.get("pk_type", "unknown")
    pk_mismatch_detail = None

    if cand_pk_type != "unknown" and ex_pk_type != "unknown":
        # Mismatch cases that should be penalized:
        # 1. LF has integer_pk_not_c0 but exemplar has composite_pk
        #    (LF came from table with INTEGER PK, exemplar doesn't have one)
        # 2. LF has composite_pk but exemplar has integer_pk

        if cand_pk_type == "integer_pk_not_c0" and ex_pk_type == "composite_pk":
            # LF came from INTEGER PK table, but exemplar has composite PK
            # This is a strong structural mismatch
            avg_score -= 0.25
            pk_mismatch_detail = "PK type mismatch: LF has INTEGER PK (c0 is not PK), exemplar has composite PK"

        elif cand_pk_type == "composite_pk" and ex_pk_type == "integer_pk":
            # LF came from composite PK table, exemplar has single INTEGER PK
            avg_score -= 0.25
            pk_mismatch_detail = "PK type mismatch: LF has composite PK, exemplar has INTEGER PK"

    matched_count = sum(1 for _, score, _ in column_scores if score >= 0.6)

    # Calculate match confidence based on semantic anchors and schema specificity
    confidence_metrics = calculate_match_confidence(
        candidate_cols,
        exemplar_cols,
        column_scores,
        avg_score,
        all_exemplar_tables if all_exemplar_tables is not None else {},
        original_exemplar_col_count,
    )

    # Add PK mismatch detail to confidence reasons and apply penalty to confidence_score
    confidence_reasons = confidence_metrics["reasons"]
    confidence_score = confidence_metrics["confidence_score"]
    confidence = confidence_metrics["confidence"]

    if pk_mismatch_detail:
        confidence_reasons = [pk_mismatch_detail] + confidence_reasons
        # Apply same penalty to confidence_score
        confidence_score -= 0.25
        confidence_score = max(0.0, confidence_score)
        # Recalculate confidence level
        if confidence_score >= 0.75:
            confidence = "high"
        elif confidence_score >= 0.5:
            confidence = "medium"
        elif confidence_score >= 0.25:
            confidence = "low"
        else:
            confidence = "very_low"

    return {
        "score": avg_score,
        "matched_columns": matched_count,
        "total_columns": len(candidate_cols),
        "column_scores": column_scores,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "confidence_reasons": confidence_reasons,
    }


def get_lf_tables(split_db: Path) -> list[str]:
    """Get list of lost_and_found split tables (lf_table_1, lf_table_2, ...)."""
    with sqlite3.connect(split_db) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'lf_table_%'")
        tables = [row[0] for row in cursor.fetchall()]

    # Sort by table number
    return sorted(tables, key=lambda t: int(t.split("_")[-1]))


def get_table_info(split_db: Path, table_name: str) -> dict[str, Any]:
    """Get row count and nfield for an LF table."""
    with sqlite3.connect(split_db) as con:
        # Get row count
        cursor = con.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # Get nfield (count of c0, c1, c2... columns)
        cursor = con.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        nfield = len([c for c in columns if c[1].startswith("c") and c[1][1:].isdigit()])

    return {
        "row_count": row_count,
        "nfield": nfield,
    }


def match_lf_table_to_exemplars(
    split_db: Path,
    lf_table: str,
    exemplar_rubrics: list[dict],
    per_db_ignorable_tables: dict[str, set[str]] | None = None,
) -> list[dict]:
    """
    Match a single LF table against multiple exemplar rubrics.

    Args:
        split_db: Path to split database containing LF tables
        lf_table: Name of LF table to match (e.g., "lf_table_1")
        exemplar_rubrics: List of dicts with keys:
            - "name": Exemplar name
            - "rubric_path": Path to rubric JSON
            - "rubric": Optional pre-loaded rubric dict
        per_db_ignorable_tables: Optional dict mapping exemplar_name -> set of
            table names to ignore for that specific exemplar. These are loaded
            from artifact_recovery_catalog.yaml (e.g., "settings" for Kext Policy Database).

    Returns:
        List of matches sorted by score (highest first):
        [
            {
                "exemplar_name": str,
                "exemplar_table": str,
                "score": float,
                "matched_columns": int,
                "total_columns": int,
                "confidence": str,
                "confidence_score": float,
                "confidence_reasons": [str, ...],
                "table_info": {"row_count": int, "nfield": int},
            },
            ...
        ]
    """
    # Generate rubric for LF table
    lf_rubric = generate_lf_table_rubric(split_db, lf_table)

    # Get table info
    table_info = get_table_info(split_db, lf_table)

    # Match against each exemplar
    matches = []

    for exemplar_info in exemplar_rubrics:
        exemplar_name = exemplar_info["name"]

        # Load rubric if not pre-loaded
        if "rubric" in exemplar_info:
            exemplar_rubric = exemplar_info["rubric"]
        else:
            exemplar_rubric = load_exemplar_rubric(exemplar_info["rubric_path"])

        # Test against each table in exemplar
        all_exemplar_tables = exemplar_rubric.get("tables", {})

        # Get per-DB ignorable tables for this specific exemplar (lowercase for matching)
        exemplar_ignorable = set()
        if per_db_ignorable_tables and exemplar_name in per_db_ignorable_tables:
            exemplar_ignorable = {t.lower() for t in per_db_ignorable_tables[exemplar_name]}

        for table_name, table_rubric in all_exemplar_tables.items():
            # Skip global ignorable tables (z_primarykey, sqlite_sequence, etc.)
            # These tables have generic schemas that match too broadly
            if table_name.lower() in _IGNORABLE_TABLES_LOWER:
                continue

            # Skip per-DB ignorable tables (e.g., 'settings' for Kext Policy Database)
            # These are tables marked as ignorable in artifact_recovery_catalog.yaml
            if table_name.lower() in exemplar_ignorable:
                continue

            # Compare rubrics (pass all tables for schema uniqueness check)
            match_result = compare_rubrics(lf_rubric, table_rubric, all_exemplar_tables)

            if match_result["score"] > 0:
                matches.append(
                    {
                        "exemplar_name": exemplar_name,
                        "exemplar_table": table_name,
                        "score": match_result["score"],
                        "matched_columns": match_result["matched_columns"],
                        "total_columns": match_result["total_columns"],
                        "confidence": match_result["confidence"],
                        "confidence_score": match_result["confidence_score"],
                        "confidence_reasons": match_result["confidence_reasons"],
                        "table_info": table_info,
                    }
                )

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


def match_lf_tables_to_exemplars(
    split_db: Path,
    exemplar_rubrics: list[dict],
    exact_match_name: str | None = None,
    nearest_exemplar_names: list[str] | None = None,
    per_db_ignorable_tables: dict[str, set[str]] | None = None,
) -> dict[str, list[dict]]:
    """
    Match all LF tables in a split database against exemplar rubrics.

    CRITICAL: If exact_match_name is provided and appears in nearest_exemplar_names,
    it will be filtered out to avoid duplicate testing.

    Args:
        split_db: Path to split database with LF tables
        exemplar_rubrics: List of exemplar info dicts (see match_lf_table_to_exemplars)
        exact_match_name: Name of exact match exemplar (if any)
        nearest_exemplar_names: List of nearest exemplar names
        per_db_ignorable_tables: Optional dict mapping exemplar_name -> set of
            table names to ignore for that specific exemplar. These are loaded
            from artifact_recovery_catalog.yaml (e.g., "settings" for Kext Policy Database).

    Returns:
        {
            "lf_table_1": [match_dict, ...],
            "lf_table_2": [match_dict, ...],
            ...
        }
    """
    # Filter out exact_match from nearest_exemplars to avoid duplicate testing
    filtered_rubrics = exemplar_rubrics

    if exact_match_name and nearest_exemplar_names and exact_match_name in nearest_exemplar_names:
        # Remove exact_match from the list
        filtered_rubrics = [r for r in exemplar_rubrics if r["name"] != exact_match_name]

    # Get all LF tables
    lf_tables = get_lf_tables(split_db)

    # Match each LF table
    results = {}
    for idx, lf_table in enumerate(lf_tables, 1):
        matches = match_lf_table_to_exemplars(split_db, lf_table, filtered_rubrics, per_db_ignorable_tables)
        results[lf_table] = matches

        # Periodic gc.collect() every 50 LF tables to release SQLite connections
        # CRITICAL: Prevents file handle exhaustion (ERRNO 24)
        if idx % 50 == 0:
            gc.collect()

    # Force garbage collection after processing all LF tables to release SQLite connections
    gc.collect()

    return results
