"""
Logic for combining multiple LF tables that match the same exemplar table.

Groups and combines LF split tables (lf_table_1, lf_table_2, lf_table_5) that
all match the same table in the same exemplar, preparing them for insertion
into a reconstructed database.
"""

from collections import defaultdict
from typing import Any


def group_lf_tables_by_match(
    match_results: dict[str, list[dict]],
    confidence_threshold: str = "medium",
) -> dict[tuple[str, str], list[str]]:
    """
    Group LF tables by their best match (exemplar_name, exemplar_table).

    Only includes matches at or above the confidence threshold.

    Args:
        match_results: Dict mapping lf_table_name -> list of match dicts
        confidence_threshold: Minimum confidence level to accept
            ("high", "medium", "low", "very_low")

    Returns:
        Dict mapping (exemplar_name, exemplar_table) -> [lf_table_1, lf_table_2, ...]

    Example:
        {
            ("Chrome History", "urls"): ["lf_table_1", "lf_table_3"],
            ("Chrome Cookies", "cookies"): ["lf_table_2"],
        }
    """
    # Confidence level ordering
    confidence_levels = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
    min_confidence = confidence_levels.get(confidence_threshold, 2)

    grouped = defaultdict(list)

    for lf_table, matches in match_results.items():
        if not matches:
            continue

        # Take best match (already sorted by score)
        best_match = matches[0]

        # Check confidence threshold
        match_confidence_level = confidence_levels.get(best_match["confidence"], 0)

        if match_confidence_level >= min_confidence:
            key = (best_match["exemplar_name"], best_match["exemplar_table"])
            grouped[key].append(lf_table)

    return dict(grouped)


def get_column_mapping(
    exemplar_rubric: dict,
    exemplar_table: str,
    nfield: int,
) -> dict[str, str] | None:
    """
    Generate column mapping from LF columns (c0, c1, c2...) to exemplar columns.

    Args:
        exemplar_rubric: Loaded exemplar rubric dict
        exemplar_table: Name of table in exemplar
        nfield: Number of fields in LF table

    Returns:
        Dict mapping "c0" -> "column_name", "c1" -> "other_column", etc.
        Returns None if nfield doesn't match exemplar column count.

    Example:
        {
            "c0": "id",
            "c1": "url",
            "c2": "title",
            "c3": "visit_count",
        }
    """
    table_rubric = exemplar_rubric.get("tables", {}).get(exemplar_table)
    if not table_rubric:
        return None

    exemplar_columns = table_rubric.get("columns", [])

    # Handle NULL PK column case
    # If first column is a PK and LF table has nfield = len(columns) - 1,
    # assume LF table is missing the auto-increment PK
    first_col_is_pk = exemplar_columns and exemplar_columns[0].get("primary_key", False)

    if first_col_is_pk and nfield == len(exemplar_columns) - 1:
        # Skip first column (PK) in mapping
        mapping_columns = exemplar_columns[1:]
    elif nfield == len(exemplar_columns):
        # Exact match
        mapping_columns = exemplar_columns
    else:
        # Column count mismatch
        return None

    # Build mapping
    column_mapping = {}
    for idx, col in enumerate(mapping_columns):
        lf_col_name = f"c{idx}"
        exemplar_col_name = col["name"]
        column_mapping[lf_col_name] = exemplar_col_name

    return column_mapping


def prepare_combined_tables(
    grouped_lf_tables: dict[tuple[str, str], list[str]],
    match_results: dict[str, list[dict]],
    exemplar_rubrics: dict[str, dict],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Prepare data structures for combining LF tables into reconstructed tables.

    Args:
        grouped_lf_tables: Output from group_lf_tables_by_match
        match_results: Original match results (for getting nfield)
        exemplar_rubrics: Dict mapping exemplar_name -> rubric dict

    Returns:
        Dict mapping (exemplar_name, exemplar_table) -> {
            "lf_tables": [lf_table_1, lf_table_2, ...],
            "column_mapping": {"c0": "id", "c1": "url", ...},
            "total_rows": int (sum of all LF table rows),
            "confidence": "high" | "medium" | "low",
        }
    """
    prepared = {}

    for (exemplar_name, exemplar_table), lf_tables in grouped_lf_tables.items():
        # Get exemplar rubric
        exemplar_rubric = exemplar_rubrics.get(exemplar_name)
        if not exemplar_rubric:
            continue

        # Get nfield from first LF table's match result
        first_lf_table = lf_tables[0]
        matches = match_results.get(first_lf_table, [])
        if not matches:
            continue

        # Find match for this exemplar table
        matching_entry = None
        for match in matches:
            if match["exemplar_name"] == exemplar_name and match["exemplar_table"] == exemplar_table:
                matching_entry = match
                break

        if not matching_entry:
            continue

        nfield = matching_entry["table_info"]["nfield"]
        confidence = matching_entry["confidence"]

        # Generate column mapping
        column_mapping = get_column_mapping(exemplar_rubric, exemplar_table, nfield)
        if not column_mapping:
            continue

        # Calculate total rows
        total_rows = sum(
            match_results[lf_table][0]["table_info"]["row_count"]
            for lf_table in lf_tables
            if lf_table in match_results and match_results[lf_table]
        )

        prepared[(exemplar_name, exemplar_table)] = {
            "lf_tables": lf_tables,
            "column_mapping": column_mapping,
            "total_rows": total_rows,
            "confidence": confidence,
        }

    return prepared


def get_unmatched_lf_tables(
    all_lf_tables: list[str],
    matched_lf_tables: set[str],
) -> list[str]:
    """
    Get list of LF tables that weren't matched at sufficient confidence.

    Args:
        all_lf_tables: All LF table names
        matched_lf_tables: Set of LF table names that were matched

    Returns:
        List of unmatched LF table names
    """
    return [lf for lf in all_lf_tables if lf not in matched_lf_tables]
