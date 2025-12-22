# Database Variant Selector

The variant selector recovers and evaluates carved SQLite databases, choosing the
best variant for delivery. It attempts multiple recovery strategies and uses schema
matching to correlate databases with known exemplars.

## Variant Types

Each database goes through multiple recovery attempts, producing up to 5 variants:

| Tag | Name | Description |
| ----- | ------ | ------------- |
| **O** | Original | Raw carved file, opened directly |
| **C** | Clone | SQLite3 `.clone` operation (repairs minor corruption) |
| **R** | Recover | SQLite3 `.recover` operation (extracts surviving data) |
| **D** | Dissect | Rebuilt via sqlite_dissect (forensic page-level recovery) |
| **X** | Empty | Marker for empty databases kept for byte carving |

## Processing Flow

### 1. Discovery

Walk case directories and emit discovery summaries of SQLite files.

### 2. Skip Detection

Check databases against the skip list from the artifact catalog. Some databases
(like system caches) are explicitly excluded from processing.

### 3. Per-Database Introspection

For each database, attempt O → C → R → D variants in sequence:

- Extract schema (tables, columns, indices)
- Count rows per table
- Run integrity checks
- Detect lost_and_found tables (from `.recover`)
- Note any anomalies

### 4. Schema Matching

Two-tier matching strategy:

1. **Fast O(1) hash lookup** - Pre-computed schema hashes enable instant matching
   against `exemplar_hash_lookup.json`
2. **Full comparison fallback** - Table name and column structure matching when
   hash lookup unavailable

Match levels: `hash` (exact) → `tables_equal+columns_equal` → `tables` (partial) → none

### 5. sqlite_dissect Rebuild (Matched DBs)

When a recovered database matches an exemplar, optionally rebuild via sqlite_dissect
into the exemplar schema (variant D). The `--dissect-all` flag attempts this for
all databases, not just matches.

### 6. Profiling & Weighting

Call `select_profile_tables()` to sample up to `PROFILE_TABLE_SAMPLE_LIMIT` shared
tables, generate per-table statistics, and weight variants. Rows below
`PROFILE_MIN_ROWS` are ignored for noisy comparison avoidance.

### 7. Variant Selection

`choose_best_variant()` combines profile scores with base heuristics (integrity,
row count, schema completeness) to select the best variant.

### 8. Metamatch Processing

Databases that didn't fully match get grouped by schema hash in `metamatch_processor.py`:

- **Unmatched** - No exemplar match at all
- **Partial match** - Table names match but columns differ

This helps identify patterns in unknown database types.

### 9. Residue Processing

`residue_processor.py` handles post-selection cleanup:

- Extract `lost_and_found` tables from matched databases into separate DBs
- Clean up unnecessary variant files for empty/unmatched databases
- Generate cleanup reports

## Module Structure

| File | Purpose |
| ------ | --------- |
| `db_variant_selector.py` | Main orchestration and introspection |
| `db_variant_selector_helpers.py` | Discovery, utilities, sqlite_dissect integration |
| `variant_operations.py` | Variant creation (clone, recover) and selection |
| `selector_profiles.py` | Table sampling and profile scoring |
| `schema_matcher.py` | Hash-based and full schema matching |
| `metamatch_processor.py` | Groups unmatched/partial databases by schema |
| `residue_processor.py` | Lost & found extraction, variant cleanup |
| `generate_hash_lookup.py` | Pre-generates exemplar hash lookup table |
| `models.py` | `DBMeta` and `Variant` dataclasses |
| `selector_config.py` | Tunable thresholds and feature flags |
| `table_profiler.py` | Per-table data quality profiling |

## Configuration

Thresholds in `selector_config.py`:

```python
PROFILE_TABLE_SAMPLE_LIMIT = 3   # Sample up to 3 tables for comparison
PROFILE_MIN_ROWS = 10            # Minimum rows for reliable comparison
PROFILE_SCORE_THRESHOLD = 0.70   # Variants below this are excluded
FLAG_DISSECT_ALL = False         # Try sqlite_dissect on all DBs (testing)
```