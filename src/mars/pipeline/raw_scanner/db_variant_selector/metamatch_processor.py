#!/usr/bin/env python3
"""Metamatch Processor for unclassified and partial-match databases.

Groups databases by exact schema hash (tables_effective_md5) and annotates them
with metamatch metadata. This helps identify patterns in databases that didn't
fully match known exemplars.

Includes:
- Unmatched databases (no exemplar match)
- Partial matches (match='tables' - table names match but not columns)

Excludes:
- Full schema matches (match='tables_equal+columns_equal' or 'hash')

Every processed database gets a label, even singletons (group_size: 1).

Usage:
    python metamatch_processor.py results.jsonl
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mars.utils.file_utils import append_jsonl, read_jsonl, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path


def read_unmatched_cases(results_path: Path) -> list[dict[str, Any]]:
    """
    Read and filter unmatched/partial-match, non-empty case records from results.

    Includes:
    - Unmatched databases (no exemplar match)
    - Partial matches (match='tables' - table names only, not columns)

    Excludes:
    - Full schema matches (match='tables_equal+columns_equal' or 'hash')
    - Empty databases

    Args:
        results_path: Path to sqlite_scan_results.jsonl

    Returns:
        List of unmatched/partial-match case records for metamatch grouping
    """
    unmatched = []

    # Read case records from JSONL file
    for record in read_jsonl(results_path, filter_type="case"):
        # Only process unmatched/partial-match and non-empty records
        if record.get("skipped"):
            continue

        decision = record.get("decision", {})
        if decision.get("empty"):
            continue

        # Include unmatched databases and partial matches (match='tables' only)
        # Exclude full schema matches (match='tables_equal+columns_equal' or 'hash')
        is_matched = decision.get("matched", False)
        if is_matched:
            # Check if this is a partial match (tables only) or full match
            exact_matches = record.get("exact_matches", [])
            if exact_matches:
                match_type = exact_matches[0].get("match", "")
                # Only exclude full schema matches from metamatch
                if match_type in ["tables_equal+columns_equal", "hash"]:
                    continue
                # match='tables' falls through - these should be in metamatch groups
            else:
                # Matched but no exact_matches array (shouldn't happen, but exclude)
                continue

        unmatched.append(record)

    return unmatched


def group_by_schema_hash(
    cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group cases by their schema hash (tables_effective_md5).

    Args:
        cases: List of unmatched case records

    Returns:
        Dictionary mapping schema_hash to list of cases
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        meta = case.get("meta_snapshot", {})
        schema_hash = meta.get("tables_effective_md5")
        # Handle None or missing hash values
        if not schema_hash:
            schema_hash = "unknown"
        groups[schema_hash].append(case)

    return dict(groups)


def annotate_with_metamatch(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    Create metamatch annotations for each group.

    Args:
        groups: Dictionary of schema_hash -> cases

    Returns:
        Dictionary of schema_hash -> metamatch_info
    """
    metamatch_info = {}

    for schema_hash, cases in groups.items():
        # Get schema info from first case
        first_case = cases[0]
        meta = first_case.get("meta_snapshot", {})

        label = meta.get("tables_effective_label", "unknown_unknown")
        # Extract first table from label (format: "FirstTable_hash8")
        first_table = label.rsplit("_", 1)[0]
        table_count = meta.get("tables_n_effective", 0)

        # Build metamatch metadata
        metamatch_info[schema_hash] = {
            "group_id": schema_hash,
            "group_label": label,
            "group_size": len(cases),
            "first_table": first_table,
            "table_count": table_count,
        }

    return metamatch_info


def generate_metamatch_summary(
    groups: dict[str, list[dict[str, Any]]],
    metamatch_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate summary statistics for metamatch groups.

    Args:
        groups: Dictionary of schema_hash -> cases
        metamatch_info: Dictionary of schema_hash -> metamatch metadata

    Returns:
        Summary record for JSONL output
    """
    # Count singletons vs multi-member groups
    singleton_count = sum(1 for cases in groups.values() if len(cases) == 1)
    multi_member_count = sum(1 for cases in groups.values() if len(cases) >= 2)

    # Find largest group
    largest_group_size = max(len(cases) for cases in groups.values()) if groups else 0
    largest_group_label = None
    if largest_group_size > 1:
        for schema_hash, cases in groups.items():
            if len(cases) == largest_group_size:
                largest_group_label = metamatch_info[schema_hash]["group_label"]
                break

    # Build group details for summary
    group_details = []
    for schema_hash, cases in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        meta_info = metamatch_info[schema_hash]

        group_details.append(
            {
                "group_label": meta_info["group_label"],
                "group_id": schema_hash,
                "member_count": len(cases),
                "first_table": meta_info["first_table"],
                "table_count": meta_info["table_count"],
                "members": [
                    {
                        "path": case.get("case_path"),
                        "rows_total": case.get("meta_snapshot", {}).get("rows_total", 0),
                        "variant_chosen": case.get("variant_chosen"),
                    }
                    for case in cases
                ],
            }
        )

    return {
        "type": "metamatch_summary",
        "timestamp": datetime.now(UTC).isoformat(),
        "total_unmatched": sum(len(cases) for cases in groups.values()),
        "unique_schemas": len(groups),
        "singleton_count": singleton_count,
        "multi_member_count": multi_member_count,
        "largest_group_size": largest_group_size,
        "largest_group_label": largest_group_label,
        "groups": group_details,
    }


def process_results(results_path: Path) -> dict[str, Any]:
    """
    Process sqlite_scan_results.jsonl and add metamatch metadata.

    This function:
    1. Reads all records from results file
    2. Groups unmatched cases by schema hash
    3. Annotates each unmatched case with metamatch metadata
    4. Rewrites results file with annotations + summary

    Args:
        results_path: Path to sqlite_scan_results.jsonl

    Returns:
        Summary statistics
    """
    # Read all records (we need to keep non-case records too)
    all_records = read_jsonl(results_path)

    # Find unmatched cases
    unmatched_cases = read_unmatched_cases(results_path)

    if not unmatched_cases:
        return {
            "unique_schemas": 0,
            "singleton_count": 0,
            "multi_member_count": 0,
            "largest_group": 0,
        }

    # Group by schema hash
    groups = group_by_schema_hash(unmatched_cases)

    # Generate metamatch annotations
    metamatch_info = annotate_with_metamatch(groups)

    # Create backup before modifying
    backup_path = results_path.with_suffix(".jsonl.backup")
    shutil.copy2(results_path, backup_path)

    # Annotate records with metamatch metadata
    annotated_count = 0
    for record in all_records:
        # Only annotate case records eligible for metamatch grouping
        if record.get("type") != "case":
            continue
        if record.get("skipped"):
            continue

        decision = record.get("decision", {})
        if decision.get("empty"):
            continue

        # Use same logic as read_unmatched_cases():
        # Include unmatched databases and partial matches (match='tables' only)
        # Exclude full schema matches (match='tables_equal+columns_equal' or 'hash')
        is_matched = decision.get("matched", False)
        if is_matched:
            # Check if this is a partial match (tables only) or full match
            exact_matches = record.get("exact_matches", [])
            if exact_matches:
                match_type = exact_matches[0].get("match", "")
                # Only exclude full schema matches from metamatch annotation
                if match_type in ["tables_equal+columns_equal", "hash"]:
                    continue
                # match='tables' falls through - these get metamatch annotation
            else:
                # Matched but no exact_matches array (shouldn't happen, but exclude)
                continue

        # Get schema hash and add metamatch field
        meta = record.get("meta_snapshot", {})
        schema_hash = meta.get("tables_effective_md5", "unknown")

        if schema_hash in metamatch_info:
            record["metamatch"] = metamatch_info[schema_hash]
            annotated_count += 1

    # Generate summary
    summary = generate_metamatch_summary(groups, metamatch_info)

    # Write updated records + summary
    write_jsonl(results_path, all_records)
    append_jsonl(results_path, summary)

    return {
        "unique_schemas": summary["unique_schemas"],
        "singleton_count": summary["singleton_count"],
        "multi_member_count": summary["multi_member_count"],
        "largest_group": summary["largest_group_size"],
    }
