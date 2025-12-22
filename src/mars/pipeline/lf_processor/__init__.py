"""
Lost and Found (LF) table processor.

This module handles extraction and matching of recovered lost_and_found data
against exemplar databases and metamatches.
"""

# Orchestrator
# Utilities
# Processor modules (Merge/Catalog/Nearest/Orphan)
from . import (
    lf_catalog,
    lf_merge,
    lf_nearest,
    lf_orchestrator,
    lf_orphan,
    lf_reconstruction,
    lf_splitter,
    uc_helpers,
)

# Core modules
from .db_reconstructor import (
    copy_table_data_with_provenance,
    deduplicate_table,
    get_table_schema,
    insert_lf_data_into_table,
    reconstruct_database_with_schema,
)
from .lf_combiner import (
    get_column_mapping,
    get_unmatched_lf_tables,
    group_lf_tables_by_match,
    prepare_combined_tables,
)
from .lf_matcher import (
    get_lf_tables,
    load_exemplar_rubric,
    match_lf_table_to_exemplars,
    match_lf_tables_to_exemplars,
)
from .lf_orchestrator import LFOrchestrator

__all__ = [
    # Orchestrator
    "LFOrchestrator",
    "lf_orchestrator",
    # Utilities
    "lf_splitter",
    # Processor modules (Merge/Catalog/Nearest/Orphan)
    "lf_merge",
    "lf_catalog",
    "lf_nearest",
    "lf_orphan",
    "lf_reconstruction",
    "uc_helpers",
    # Matching
    "match_lf_tables_to_exemplars",
    "match_lf_table_to_exemplars",
    "load_exemplar_rubric",
    "get_lf_tables",
    # Combining
    "group_lf_tables_by_match",
    "prepare_combined_tables",
    "get_column_mapping",
    "get_unmatched_lf_tables",
    # Reconstruction
    "reconstruct_database_with_schema",
    "get_table_schema",
    "insert_lf_data_into_table",
    "copy_table_data_with_provenance",
    "deduplicate_table",
]
