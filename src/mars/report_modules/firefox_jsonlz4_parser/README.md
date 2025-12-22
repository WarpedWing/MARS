# Firefox JSONLZ4 Salvage Parser

This module salvages Firefox JSONLZ4 (mozLz4) compressed JSON files, extracting sessions, bookmarks,
and telemetry data from both intact and carved/truncated files.

## Disclaimer

**This report is for informational and investigative purposes only.** The data presented should be
independently verified before being relied upon for any legal, regulatory, or evidentiary purpose.

### Limitations

- **Partial data recovery**: Carved or truncated files may yield incomplete data. The parser uses
progressive decompression and JSON repair heuristics, but some records may be lost or corrupted.
- **JSON repair heuristics**: Corrupted JSON is repaired using multi-stage heuristics (brace
balancing, escape fixing, truncation recovery). Repaired JSON may not perfectly represent the original data.
- **Telemetry structure variance**: Firefox telemetry JSON structure varies across Firefox versions.
Some fields may be missing or differently named in older/newer versions.
- **Timestamp interpretation**: Session timestamps represent last-accessed times, not necessarily
when tabs were first opened. Bookmark timestamps may reflect import dates rather than original creation.
- **Deduplication**: The parser deduplicates extracted records within a single run. Records may
appear duplicated across multiple source files from the same profile.

## What Are JSONLZ4 Files?

Firefox uses a custom LZ4-based compression format called "mozLz4" for storing JSON data. These
files have a `mozLz40\0` magic header followed by LZ4-compressed JSON content.

Common JSONLZ4 files include:

- **Session files** (`sessionstore.jsonlz4`, `recovery.jsonlz4`): Open tabs, closed tabs, window state
- **Bookmarks** (`bookmarks.jsonlz4`): Bookmark tree with folders and URLs
- **Telemetry** (`*.jsonlz4` in datareporting): Browser metrics and events

## Salvage Capabilities

This module is specifically designed for forensic recovery from:

- **Carved files**: JSONLZ4 segments extracted from disk images or unallocated space
- **Truncated files**: Partially overwritten or incomplete files
- **Corrupted files**: Files with bit errors or missing data

### Recovery Techniques

1. **Segment detection**: Scans files for `mozLz40\0` magic bytes to find embedded JSONLZ4 segments
2. **Progressive tail trimming**: Attempts decompression with progressively shorter input until success
3. **Partial decompression**: Uses liblz4 C library for partial decompression when available
4. **JSON repair**: Multi-stage repair process:
   - Control character removal
   - Backslash escape fixing
   - Brace/bracket balancing
   - Truncation to last valid JSON boundary

## Parsed Data

### Sessions

Extracted from session restore files:

- **Tab URLs**: URLs from open and recently closed tabs
- **Last accessed**: Timestamp of last tab interaction
- **Window context**: Which window/tab group contained the tab

### Bookmarks

Extracted from bookmark backup files:

- **Title**: Bookmark display name
- **URL**: Bookmarked URI
- **Date added**: When bookmark was created
- **Date modified**: Last modification timestamp
- **Folder path**: Location in bookmark hierarchy

### Telemetry Events

Extracted from telemetry event files:

- **Timestamp**: Event occurrence time
- **Category**: Event category (e.g., "navigation", "security")
- **Method**: Event action type
- **Object**: Target of the event
- **Value**: Event payload

### Telemetry Metrics

Extracted from telemetry main ping files:

- **Metric name**: Flattened metric path
- **Value**: Metric value (histograms, scalars, etc.)
- **Process**: Which Firefox process generated the metric

## Data Sources

| Artifact | Path Pattern | Purpose |
| ---------- | -------------- | --------- |
| Session store | `sessionstore.jsonlz4` | Current session state |
| Session backup | `sessionstore-backups/*.jsonlz4` | Session recovery files |
| Bookmarks | `bookmarkbackups/*.jsonlz4` | Bookmark snapshots |
| Telemetry | `datareporting/archived/**/*.jsonlz4` | Browser telemetry |
| Carved files | `**/*.jsonlz4`, `**/*.mozlz4` | Recovered segments |

## Not Currently Parsed

- **Session history**: Individual tab navigation history within sessions
- **Form data**: Saved form field values in session files
- **Pinned tabs**: Tab pinning state
- **Container tabs**: Multi-Account Container associations
- **Sync data**: Firefox Sync encrypted payloads

## Output Files

Four CSV files are generated:

- `firefox_sessions.csv` - Tab URLs and access timestamps
  - `source_file`, `url`, `last_accessed`

- `firefox_bookmarks.csv` - Bookmark entries
  - `source_file`, `title`, `url`, `date_added`, `date_modified`

- `firefox_telemetry_events.csv` - Flattened telemetry events
  - `source_file`, `timestamp`, `category`, `method`, `object`, `value`, `extra_*`

- `firefox_telemetry_main_metrics.csv` - Flattened telemetry metrics
  - `source_file`, `metric`, `value`, `process`

## File Classification

When `--classify` is specified, source files are organized into categories:

```text
Firefox/
  Sessions/      - Session restore files
  Bookmarks/     - Bookmark backup files
  Telemetry/     - Telemetry data files
  Other/         - Valid mozLz4 JSON (unclassified type)
Unknown/         - JSON parsed but not Firefox format
Corrupt/         - No salvageable mozLz4 segments found
```

## Command-Line Options

| Option | Description |
| -------- | ------------- |
| `--glob` | File pattern to match (default: `**/*.*lz4`) |
| `--append` | Append to existing CSV files instead of overwriting |
| `--classify` | Organize files: `move`, `copy`, `link`, or `none` |
| `--preserve-tree` | Maintain source directory structure in classification |
| `--dry-run` | Show what would be done without making changes |
| `--delete-source` | Delete source files after successful processing |

## Dependencies

- `lz4` Python library (required)
- `liblz4` C library (optional, enables partial decompression)

## Scan Type

**Exemplar AND Candidate** - This module runs during both live system scans and carved file
processing, making it particularly useful for forensic recovery scenarios.
