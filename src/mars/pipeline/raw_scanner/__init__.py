"""Raw file scanner for recovered/carved files.

This module processes recovered files from forensic tools (PhotoRec, Scalpel, etc.)
and carved data, organizing and classifying databases and logs.

Modules:
    candidate_processor: Main orchestrator for raw file processing
    file_categorizer: Fingerprint and categorize recovered files
    database_carver: Byte-carve SQLite databases from raw blocks
"""
