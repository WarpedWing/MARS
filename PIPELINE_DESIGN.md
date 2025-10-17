# Mac Log Sleuth Pipeline Design

## Overview
Two-phase forensic recovery pipeline: Exemplar phase creates ground truth, Search phase recovers and reconstructs data.

---

## Phase 1: Exemplar Processing

### Goal
Build ground truth from a known-good system to guide recovery.

### Steps

#### 1. Exemplar Selection
```
Input: Path to exemplar system drive
Output: Validated system root
```

**Actions:**
- Verify it's a macOS system (check `/System/Library/CoreServices/SystemVersion.plist`)
- Detect macOS version
- Check for common system directories

#### 2. Database Discovery
```
Input: System root + database catalog
Output: List of discovered databases with metadata
```

**Actions:**
- Load `macos_databases.yaml` catalog
- Glob search for each database pattern
- Filter by macOS version compatibility
- User can add custom paths (browser profiles, apps, etc.)

**Data Structure:**
```python
@dataclass
class DiscoveredDatabase:
    name: str
    category: str
    original_path: Path
    relative_path: str  # Relative to system root
    size: int
    modified: datetime
    intact: bool
    schema_hash: str  # For deduplication
```

#### 3. Schema/Rubric Creation
```
Input: Discovered databases
Output: Schema definitions + rubrics
```

**Actions:**
- For each intact database:
  - Extract schema (tables, columns, types, indexes)
  - Create fingerprint (table names, column patterns)
  - Generate rubric (identifying characteristics)
  - Save to `schemas/{category}/{name}.schema.json`

**Schema Format:**
```json
{
  "name": "TCC.db",
  "category": "tcc",
  "macos_version": "14.0",
  "tables": {
    "access": {
      "columns": ["service", "client", "auth_value", "last_modified"],
      "primary_key": ["service", "client"],
      "timestamp_columns": ["last_modified"]
    }
  },
  "fingerprint": {
    "table_count": 3,
    "table_names_hash": "abc123...",
    "signature": "TCC database pattern"
  }
}
```

#### 4. Archive Processing
```
Input: System archives (Time Machine, etc.)
Output: Merged database copies in working folder
```

**Actions:**
- Scan `/Volumes/*/Backups.backupdb/` (Time Machine)
- Scan other archive directories user specifies
- For each target database:
  - Extract from .gz/.bz2 if needed (gzip2recover/bzip2recover)
  - Attempt merge with main database copy
  - Deduplicate by primary key or timestamp
  - Track data provenance (which archive contributed what)

**Working Folder Structure:**
```
working/
├── databases/
│   ├── tcc/
│   │   ├── system_TCC.db  (merged from all sources)
│   │   └── provenance.json
│   ├── safari/
│   │   ├── History.db
│   │   └── provenance.json
│   └── ...
├── schemas/
│   ├── tcc/
│   │   └── TCC.schema.json
│   └── ...
└── exemplar_metadata.json
```

---

## Phase 2: Search & Recovery

### Goal
Process carved/recovered files and reconstruct databases using exemplar knowledge.

### Steps

#### 1. Search Folder Selection
```
Input: Path to search folder (carved files, forensic export, etc.)
Output: Folder scan with file inventory
```

**Actions:**
- Recursive scan for target filetypes: `.db`, `.sqlite*`, `.gz`, `.bz2`, `.txt`, etc.
- Build file inventory with metadata

#### 2. File Processing Loop
```
For each file in inventory:
    1. Crack open if compressed
    2. Classify file type
    3. Apply recovery strategy
    4. Integrate with working databases
```

##### 2a. Decompression
```python
def crack_open(file_path: Path) -> Path:
    """
    Decompress if needed, return path to uncompressed file.

    Strategies:
    - .gz: try gunzip, fallback to gzip2recover
    - .bz2: try bunzip2, fallback to bzip2recover
    - Others: pass through
    """
```

##### 2b. Classification
```python
def classify_file(file_path: Path, schemas: dict) -> Classification:
    """
    Identify what kind of database this is.

    Methods:
    1. Magic number check (SQLite header)
    2. Schema fingerprint matching (against exemplar schemas)
    3. Rubric scoring (how well does it match known patterns)
    4. Content analysis (table names, column patterns)

    Returns:
        Classification(
            type="tcc",  # Matched category
            confidence=0.95,  # How confident we are
            schema_match="TCC.db",  # Which schema matched
            corruption_level="partial"  # intact, partial, severe, unknown
        )
    """
```

##### 2c. Recovery Strategy Selection
```python
def select_recovery_strategy(classification: Classification) -> list[RecoveryStrategy]:
    """
    Choose recovery methods based on classification and corruption level.

    Returns ordered list of strategies to try:

    If intact:
        1. Direct copy

    If partial corruption:
        1. sqlite3 PRAGMA integrity_check
        2. sqlite3 .recover
        3. sqlite_dissect
        4. Carver (last resort)

    If severe corruption:
        1. Carver
        2. sqlite_dissect (may still find fragments)

    If unknown type:
        1. Carver with generic settings
        2. Save to "unknown" folder for manual review
    """
```

##### 2d. Database Integration
```python
def integrate_recovered_data(
    recovered_db: Path,
    classification: Classification,
    working_databases: dict
) -> IntegrationResult:
    """
    Merge recovered database with working copy (from exemplar phase).

    Steps:
    1. Load schema for this database type
    2. Identify primary keys and timestamp columns
    3. For each table:
        - Extract rows from recovered db
        - Deduplicate against working db
        - Insert new rows
        - Track provenance (which file contributed data)
    4. Generate statistics (rows added, duplicates found)

    Returns:
        IntegrationResult(
            rows_added=150,
            duplicates_skipped=25,
            tables_updated=["access", "admin"],
            timestamp_range=(datetime_min, datetime_max)
        )
    """
```

#### 3. Carved Data Handling
```
If database is completely trashed:
    1. Run carver
    2. Save output to carved/{category}/{filename}/
    3. Generate metadata about carved content
    4. Optionally: attempt partial integration if tables match schema
```

---

## Phase 3: Visualization & Reporting

### Goal
Present findings with interactive visualizations and comprehensive reports.

### Components

#### 1. Plotly Dashboard
```python
def generate_timeline_plot(working_databases: dict) -> plotly.Figure:
    """
    Timeline showing:
    - Original data timestamps (from exemplar)
    - Newly recovered data timestamps
    - Data sources (which archives/carved files)

    Interactive features:
    - Zoom to time ranges
    - Filter by database category
    - Click to see details
    """
```

#### 2. Data Addition Visualization
```python
def plot_data_addition(db_name: str, before: int, after: int) -> plotly.Figure:
    """
    Bar chart showing:
    - Original row count
    - Rows added from archives
    - Rows added from carved files
    - Total final count

    With percentage increase annotation.
    """
```

#### 3. Interactive Query Interface
```python
def generate_query_dashboard(unified_db: Path) -> html:
    """
    Web interface with:
    - Dropdown to select database/table
    - Column selector
    - Time range filter
    - SQL query box (for advanced users)
    - Export to CSV button

    Uses Plotly Dash for interactivity.
    """
```

#### 4. HTML Report
```markdown
# Mac Log Sleuth Recovery Report
Generated: {timestamp}

## Case Summary
- Exemplar System: macOS {version}
- Search Folder: {path}
- Files Processed: {count}
- Databases Recovered: {count}

## Recovery Statistics

### By Category
| Category | Original Rows | Recovered Rows | % Increase |
|----------|---------------|----------------|------------|
| TCC      | 1,234         | 567            | +46%       |
| Safari   | 5,678         | 2,345          | +41%       |
| ...      | ...           | ...            | ...        |

### Timeline Coverage
[Interactive Plotly timeline embedded]

### Top Recovered Databases
1. **TCC.db** - Added 567 new permissions records
   - [View Details](./details/tcc.html)
   - [Download CSV](./exports/tcc.csv)
2. **History.db** - Added 2,345 browsing history entries
   - [View Details](./details/safari.html)
   - [Download CSV](./exports/safari.csv)

## Files Processed

### Successfully Recovered
- `carved_001.db` → Identified as TCC.db (95% confidence)
- `backup_20201215.gz` → Extracted Safari history
- ...

### Partially Recovered (Carved)
- `corrupted_023.db` → Carved {n} records
  - [View Carved Data](./carved/corrupted_023/)

### Failed Recovery
- `unknown_file.bin` → Could not classify
  - Saved to [unknown/](./unknown/)

## Recommendations
- Review databases with low confidence scores manually
- Check "unknown" folder for unidentified artifacts
- Run additional analysis on {suspicious_items}

---
*Generated by Mac Log Sleuth v{version}*
```

---

## Implementation Phases

### MVP (Phase 1)
- [x] Database catalog (YAML)
- [ ] Exemplar scanner
- [ ] Schema generator
- [ ] Simple archive merger
- [ ] Basic file classifier
- [ ] Single recovery strategy (sqlite3 .recover)
- [ ] Basic HTML report

### Phase 2
- [ ] Multiple recovery strategies with fallback
- [ ] Advanced deduplication
- [ ] Plotly visualizations
- [ ] Interactive query dashboard

### Phase 3
- [ ] Toga GUI wrapper
- [ ] Forensic image support (pyewf, libaff4)
- [ ] Advanced correlation (cross-database queries)
- [ ] Export to timeline formats (Plaso JSONL, Timesketch)

---

## File Structure

```
mac_log_sleuth/
├── cli.py                          # Main CLI entry point
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py             # Main pipeline coordinator
│   ├── exemplar.py                 # Phase 1: Exemplar processing
│   ├── search.py                   # Phase 2: Search & recovery
│   └── reporting.py                # Phase 3: Visualization & reports
├── catalog/
│   ├── macos_databases.yaml        # System database catalog
│   └── loader.py                   # Catalog loader
├── recovery/
│   ├── __init__.py
│   ├── strategies.py               # Recovery strategy selector
│   ├── sqlite_native.py            # sqlite3 .recover wrapper
│   ├── sqlite_dissect.py           # sqlite-dissect wrapper
│   ├── compressed.py               # gzip/bzip recovery
│   └── carver.py                   # Wrapper for your carver
├── classification/
│   ├── __init__.py
│   ├── magic.py                    # Magic number detection
│   ├── fingerprint.py              # Schema fingerprinting
│   └── rubric.py                   # Rubric scoring
├── integration/
│   ├── __init__.py
│   ├── merger.py                   # Database merging logic
│   └── deduplication.py            # Dedupe strategies
├── visualization/
│   ├── __init__.py
│   ├── timeline.py                 # Plotly timeline
│   ├── dashboard.py                # Interactive dashboard
│   └── reports.py                  # HTML report generation
├── carver/                         # Your existing carving code
│   └── ...
└── gui/                            # Optional Toga GUI
    └── main.py
```

---

## Configuration File Example

```yaml
# case_config.yaml
case:
  name: "2020_macOS_Investigation"
  investigator: "Analyst Name"
  date_range: ["2020-01-01", "2020-12-31"]

exemplar:
  system_root: "/Volumes/ExemplarDrive"
  custom_databases:
    - "/Users/victim/Library/Application Support/Slack/storage/slack.db"
    - "/Users/victim/Library/Safari/History.db"

  archives:
    - "/Volumes/TimeMachine/Backups.backupdb/MacBook/Latest"
    - "/Volumes/Backup/archived_logs/"

search:
  input_folder: "/Volumes/Carved/recovered_files"
  recursive: true
  file_types: ["db", "sqlite", "sqlite3", "gz", "bz2"]

output:
  folder: "/Cases/2020_macOS/output"
  formats:
    - sqlite  # Unified database
    - csv     # Per-table exports
    - jsonl   # Timeline format
    - html    # Interactive report

recovery:
  strategies:
    intact: ["copy"]
    partial: ["sqlite_recover", "sqlite_dissect", "carve"]
    severe: ["carve", "sqlite_dissect"]

  deduplication:
    method: "timestamp_and_primary_key"
    conflict_resolution: "keep_newest"

visualization:
  timeline_resolution: "hour"
  include_categories: ["tcc", "safari", "knowledgeC"]
  exclude_categories: ["powerlog"]  # Too noisy
```

---

## Next Steps

1. **Start with Exemplar Scanner** - Get the foundation right
2. **Build Classification Engine** - Schema fingerprinting is key
3. **Implement Recovery Strategies** - Chain your existing tools
4. **Add Visualization** - Plotly makes this easy
5. **Polish UX** - Config files, progress bars, clear messaging

Would you like me to start implementing any of these components?
