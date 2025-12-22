# Lost & Found (LF) Database Reconstruction

This module handles the reconstruction of SQLite databases from fragmented data
recovered from disk images. When SQLite databases are carved from
raw disk images, they often contain `lost_and_found` tables with orphaned
data fragments. This processor matches these fragments against known database
schemas (exemplars) and reconstructs coherent databases.

## Processing Modes

Four modes handle recovered fragments based on their match quality
and processing requirements.

## The Four Modes

### MERGE: Metamatch Groups

**Multiple databases with identical schemas that don't match any exemplar**

When multiple recovered databases share identical table and column structures
but don't match any known exemplar, they're grouped by schema hash and
processed together. The schema hash is computed from both table names AND
column names, so only truly identical schemas are grouped.

**Processing:**

1. Group databases by schema hash (tables + columns)
2. Combine all databases in the group
3. Generate a "superrubric" from the merged database
4. Match lost_and_found fragments against the superrubric
5. Reconstruct combined database with both intact and recovered data

**Output:** `databases/metamatches/{group_label}/{group_label}.sqlite`

**Example:** Three non-catalog databases,
all with identical schema structure, combined into a single output database.

### CATALOG: Exact Matches

**Databases with exact schema matches to known exemplars**

When a recovered database exactly matches a known exemplar schema from
the catalog, we can confidently reconstruct it using that exemplar's rubric.

**Processing:**

1. Match lost_and_found fragments against the exact exemplar rubric
2. Reconstruct database using the canonical exemplar schema
3. Combine intact data + matched fragments from all matching databases
4. Create remnant databases for unmatched fragments

**Output:** `databases/catalog/{exemplar_name}/{exemplar_name}.sqlite`

**Example:** A Chrome history database that exactly matches the known
Chrome exemplar schema.

### NEAREST: Best-Fit Exemplar

**Databases matched to nearest (but not exact) exemplar**

When a database doesn't exactly match any exemplar but is close enough for
useful reconstruction, we match it to the nearest exemplar schema.

**Processing:**

1. Find nearest matching exemplar based on schema similarity
2. Match lost_and_found fragments against nearest exemplar rubric
3. Reconstruct database using nearest exemplar schema as template
4. Initially output to `found_data/`, then reclassified in Phase 7

**Final Output:** If L&F rows recovered → `databases/catalog/{exemplar_name}/`
(merged with existing CATALOG entry or created new). If no recovery → `databases/empty/`.

**Example:** A database with L&F fragments that resemble Firefox Places.
After reconstruction, if fragments were recovered, the output is promoted to
`catalog/Firefox Places/`.

### ORPHAN: Unmatched Tables

**Lost & found fragments that don't match any schema**

When lost_and_found fragments can't be matched to any exemplar (catalog,
metamatch, or nearest), they're preserved as "orphans" for manual review.

**Processing:**

1. Collect all unmatched lost_and_found tables across MERGE/CATALOG/NEAREST
2. Create standalone databases preserving the original fragment structure
3. Name using match hints when available (e.g., database had partial matches)

**Output:** `databases/found_data/{match_label}_{db_name}_orphans/{match_label}_{db_name}_orphans.sqlite`

**Example:** Fragments from an unknown application database that don't
match any known schema.

## Processing Order & Rationale

The orchestrator processes use cases in this specific order:

### Phase 1: Prepare Split Databases

Extract lost_and_found tables from all recovered databases into separate
"split" databases for matching.

### Phase 2: Group Databases

Classify databases into catalog matches, metamatch groups, or individual
databases based on their schema matching results.

### Phase 3: MERGE

Metamatch processing combines multiple databases and generates
superrubrics. Metamatches:

- Creates new composite schemas that might help match fragments
- Reduces the number of databases to process in later phases

### Phase 4: CATALOG

Catalog matches are high-confidence exact matchesthat use canonical
schemas for highest quality reconstruction.

### Phase 5: NEAREST

For databases that don't fit MERGE or CATALOG, NEAREST attempts to
match their L&F fragments against the most similar exemplar schema.
Processing them after exact matches:

- Ensures we've exhausted higher-confidence matching strategies first
- Provides match hints for fragments that resemble known schemas
- Rebuilds using the nearest exemplar schema as template
- Results go to `found_data/` initially (may be promoted to `catalog/` in Phase 7)

### Phase 6: ORPHAN

Orphan processing collects **all unmatched fragments** from
MERGE/CATALOG/NEAREST. Must run last because:

- Needs to know which fragments were successfully matched in previous phases
- Only processes remnants that couldn't be matched anywhere else
- Preserves everything for manual forensic review

### Phase 7: Reclassification

NEAREST results initially go to `found_data/` because they're based on
schema similarity rather than exact matches. However, NEAREST **rebuilds
databases using the exemplar schema as template**, making them structurally
compatible with CATALOG results.

Phase 7 reclassifies NEAREST results based on recovery success:

1. **Successful recovery** (`total_lf_rows > 0`): Promoted to `catalog/`.
   Since NEAREST outputs use the exemplar schema, they can be merged with
   existing CATALOG entries for the same exemplar.

2. **No recovery** (`total_lf_rows == 0`): Moved to `empty/`—matched a
   schema but had no recoverable L&F fragments.

3. **Orphans** (no manifest): Remain in `found_data/` for manual review.

Phase 7 also scans `catalog/` for databases that are effectively empty
(only contain ignorable tables like `sqlite_sequence`) and moves them to `empty/`.

## Output Directory Structure

```text
databases/
├── selected_variants/                # Input: Variant selection results
│   └── {f_offset}_{hash}/            # Best variant for each carved database
│       └── {variant}.sqlite          # O/C/R/D variant chosen
│
├── catalog/                          # CATALOG + promoted NEAREST
│   └── {exemplar_name}/
│       ├── {exemplar_name}.sqlite
│       ├── {exemplar_name}_manifest.json
│       └── rejected/
│           └── {exemplar_name}_rejected.sqlite
│
├── metamatches/                      # MERGE: Identical schema groups
│   └── {group_label}/
│       ├── {group_label}.sqlite
│       ├── {group_label}_manifest.json
│       └── rejected/
│           └── {group_label}_rejected.sqlite
│
├── found_data/                       # ORPHAN
│   └── {match_hint}_{db_name}_orphans/
│       └── {match_hint}_{db_name}_orphans.sqlite
│
├── empty/                            # Databases with no recoverable data
│   └── {exemplar_name}/              # Matched schema but total_lf_rows == 0
│
├── carved/                           # Byte-carved residue (variant X)
│   └── {exemplar_name}_{f_offset}_carved/
│       └── {exemplar_name}_{f_offset}_carved.sqlite
│
└── schemas/                          # Generated rubrics and schemas
    ├── {exemplar_name}/
    │   └── {exemplar_name}.rubric.json
    └── {group_label}/
        └── {group_label}.superrubric.json
```

## Manifest Files

Each reconstructed database includes a `*_manifest.json` file documenting:

- **Source databases**: Which original databases contributed data
- **Intact rows**: Rows copied from original database tables
- **LF rows**: Rows recovered from lost_and_found fragments
- **Remnant tables**: Number of unmatched fragments
- **Duplicates removed**: Deduplication statistics
- **Table-level stats**: Row counts per table

Example manifest:

```json
{
  "output_type": "catalog",
  "output_name": "Chrome_History",
  "created": "2025-01-18T10:30:00",
  "source_databases": [
    {
      "db_name": "f12345678",
      "intact_rows": 1500,
      "lf_rows": 342,
      "remnant_tables": 2
    }
  ],
  "combined_stats": {
    "total_intact_rows": 1500,
    "total_lf_rows": 342,
    "total_remnant_tables": 2,
    "duplicates_removed": 45,
    "table_stats": [
      {"name": "urls", "rows": 1200},
      {"name": "visits", "rows": 642}
    ]
  }
}
```

## Data Source Tracking

All reconstructed databases include a `data_source` column in every table
(except FTS virtual tables) to track data provenance:

- `carved_{db_name}`: Intact data from carved database
- `found_{db_name}`: Reconstructed data from lost_and_found fragments

This allows forensic analysts to distinguish between original intact
data and recovered fragments, which can be especially helpful when
combining with exemplar data.

## Key Modules

### Orchestrator

- **`lf_orchestrator.py`**: Main orchestrator coordinating all 7 phases

### Processor Modules

- **`lf_merge.py`**: MERGE - Metamatch group processing
- **`lf_catalog.py`**: CATALOG - Exact match processing
- **`lf_nearest.py`**: NEAREST - Best-fit exemplar matching
- **`lf_orphan.py`**: ORPHAN - Unmatched table preservation

### Shared Logic

- **`lf_reconstruction.py`**: Shared reconstruction logic for CATALOG/NEAREST
- **`uc_helpers.py`**: Utility functions (FTS detection, labeling, sanitization)

### Core Components

- **`lf_matcher.py`**: Fragment-to-exemplar matching engine with schema validation
- **`lf_combiner.py`**: Fragment combination and column mapping
- **`db_reconstructor.py`**: Low-level database reconstruction
- **`lf_splitter.py`**: Extract lost_and_found tables into split databases
