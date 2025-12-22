# Pipeline Module

The pipeline module contains all the core processing workflows for MARS.
It's organized into specialized subdirectories, each handling a specific aspect of the forensic analysis pipeline.

## Module Overview

### raw_scanner/

**Candidate scan processing for recovered/carved files**

Processes files from disk images, including categorization, database variant selection, and output organization.

**Key Components:**

- `candidate_orchestrator.py` - Main orchestrator coordinating candidate scan phases
- `file_categorizer.py` - File type detection and classification
- `database_carver.py` - SQLite database carving and recovery
- `catalog_wifi_mapper.py` - WiFi artifact catalog mapping
- `self_rubric_generator.py` - Generate rubrics from recovered databases
- `stats_reporter.py` - Processing statistics reporting

**Subdirectory:**

- `db_variant_selector/` - Multi-variant database selection (O/C/R/D variants). See [db_variant_selector/README.md](raw_scanner/db_variant_selector/README.md)

---

### exemplar_scanner/

**Exemplar scan for live/mounted macOS systems**

Scans macOS systems to generate exemplar databases, schemas, and rubrics for forensic classification.

**Key Components:**

- `exemplar_orchestrator.py` - Main orchestrator for exemplar scanning
- `exemplar_cataloger.py` - Catalogs databases into exemplar catalog structure (Phases 3-4)
- `database_processor.py` - Database copying and provenance tracking
- `database_scanner.py` - Database discovery and scanning
- `schema_generator.py` - Schema and rubric generation from exemplar databases
- `dfvfs_manager.py` - dfVFS integration for disk image mounting
- `workspace_manager.py` - Workspace and archive decompression management
- `cleanup_manager.py` - Post-processing cleanup operations

**Use When:** Creating exemplar rubrics from known-good systems or mounted disk images

---

### matcher/

**Rubric generation and schema utilities**

Generates and manages rubrics (schema fingerprints) used for database classification.

**Key Components:**

- `rubric_generator.py` - Generate rubrics from exemplar databases
- `rubric_utils.py` - Rubric loading, matching, and comparison utilities
- `generate_sqlite_schema_rubric.py` - Low-level schema extraction and rubric creation
- `multi_user_splitter.py` - Split multi-user rubrics into user-specific versions

**Use When:** Working with rubrics for database classification

---

### output/

**Output structure and database combination**

Manages output directory structure and combines multiple database instances.

**Key Components:**

- `database_combiner.py` - Merge multiple SQLite databases with deduplication
- `structure.py` - Output directory structure management and file organization

**Use When:** Combining database fragments or organizing output files

---

### lf_processor/

**Lost & Found fragment reconstruction**

Reconstructs databases from SQLite `lost_and_found` tables created by `.recover` operations.

See [lf_processor/README.md](lf_processor/README.md) for detailed documentation.

**Key Components:**

- `lf_orchestrator.py` - Main orchestrator for L&F processing
- `lf_merge.py` - Merge L&F fragments with exemplar schemas
- `lf_catalog.py` - Catalog exact matches
- `lf_nearest.py` - Find nearest schema matches
- `lf_orphan.py` - Handle unmatched fragments
- `lf_splitter.py` - Split L&F tables by schema

**Use When:** Recovering data from corrupted SQLite databases

---

### fingerprinter/

**File type fingerprinting**

Identifies file types by analyzing content structure rather than relying on extensions.

See [fingerprinter/README.md](fingerprinter/README.md) for detailed documentation.

**Key Components:**

- `text_fingerprinter.py` - Identifies log types from text content patterns

**Use When:** Classifying recovered text files by type

---

### mount_utils/

**Disk image mounting utilities**

Handles mounting of E01 and other forensic disk image formats using dfVFS.

**Key Components:**

- `dfvfs_glob_utils.py` - Glob pattern matching within mounted images
- `e01_mounter.py` - E01 image mounting operations
- `workflow.py` - End-to-end disk image processing workflow

**Use When:** Working with E01 or other forensic disk images

---

### common/

**Shared pipeline utilities**

Common utilities used across multiple pipeline modules.

**Key Components:**

- `catalog_manager.py` - Database catalog operations and skip list management

---

### comparison/

**Database comparison and reporting**

Compares databases and generates HTML comparison reports.

**Key Components:**

- `html_report.py` - Generate HTML comparison reports

---

## Processing Workflows

### Exemplar Scan Flow

```text
1. Mount/access source system
   ↓
2. exemplar_scanner/ - Discover and export databases
   ↓
3. matcher/ - Generate rubrics from exemplar schemas
   ↓
4. output/ - Organize into catalog structure
```

### Candidate Scan Flow

```text
1. Mount disk image or access carved files
   ↓
2. raw_scanner/ - Categorize and process files
   ↓
3. db_variant_selector/ - Select best database variants
   ↓
4. lf_processor/ - Reconstruct from lost_and_found
   ↓
5. output/ - Generate final organized output
```

## Import Examples

```python
# Exemplar scanning
from mars.pipeline.exemplar_scanner.exemplar_orchestrator import ExemplarScanner

# Rubric utilities
from mars.pipeline.matcher.rubric_utils import load_rubric, match_schema

# Database combination
from mars.pipeline.output.database_combiner import merge_sqlite_databases

# Text fingerprinting
from mars.pipeline.fingerprinter.text_fingerprinter import identify_log_type
```

## See Also

- [Main README](../../../README.md) - Project overview
- [Carver Module](../carver/) - Data carving utilities
- [CLI Module](../cli/) - Command-line interface
- [Catalog](../catalog/) - Artifact recovery catalog
