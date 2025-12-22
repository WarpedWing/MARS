# Artifact Recovery Catalog (ARC)

The Artifact Recovery Catalog defines the macOS artifacts that MARS collects during Exemplar scans.
Each entry specifies where to find an artifact, how to process it, and how to handle archives.

## Catalog File

**Location:** `artifact_recovery_catalog.yaml`

The catalog uses YAML format with comment preservation for editability.

## Structure

```yaml
catalog_metadata:
  version: 2.5.0
  created: '2025-05-16'
  updated: '2025-12-15'

skip_databases:
  geoservices:
    name: GeoServices / Location Services Cache
    reason: Read-only Apple system cache
    table_patterns: [GeoPlaces, GeoLookup_*, ...]
    description: Geographic lookup tables
    notes: Contains no user activity

category_name:
  - name: Artifact Name
    glob_pattern: 'path/to/artifact'
    file_type: database
    scope: user
    # ... additional fields
```

## Sections

### catalog_metadata

Version tracking for the catalog file.

| Field | Description |
| ------- | ------------- |
| `version` | Semantic version string |
| `created` | Creation date |
| `updated` | Last modification date (auto-updated by editor) |

### skip_databases

Databases to exclude from processing. Used for system caches with no forensic value.

| Field | Description |
| ------- | ------------- |
| `name` | Display name |
| `reason` | Why it's skipped |
| `table_patterns` | Table name patterns to match |
| `description` | What the database contains |
| `notes` | Additional context |

### Category Groups

Artifacts are organized into category groups (e.g., `accounts`, `airdrop`, `biome`).
Each group contains one or more artifact entries.

## Entry Fields

### Required Fields

| Field | Type | Description |
| ------- | ------ | ------------- |
| `name` | string | Display name (e.g., "Chrome History") |
| `glob_pattern` | string | Path pattern to locate the artifact |
| `file_type` | enum | `database`, `log`, or `cache` |
| `scope` | enum | `user` or `system` |
| `exemplar_pattern` | string | Output path pattern for exemplar files |

### Optional Fields

| Field | Type | Default | Description |
| ------- | ------ | --------- | ------------- |
| `description` | string | - | Brief description |
| `notes` | string | - | Implementation notes |
| `ignorable_tables` | list | `[]` | Tables to skip (databases only) |
| `multi_profile` | bool | `false` | Has browser-style profiles |
| `has_archives` | bool | `false` | Has associated compressed files |
| `archives` | list | - | Archive configuration (see below) |
| `combine_strategy` | enum | - | How to combine archives |
| `preserve_structure` | bool | `false` | Copy folder structure intact |

## Glob Pattern Syntax

Patterns follow Python's `pathlib.glob()` syntax:

```yaml
# User-scoped: use * for username
glob_pattern: 'Users/*/Library/Application Support/com.apple.TCC/TCC.db'

# Variable folder depth
glob_pattern: 'Users/*/Library/Caches/**/Cache.db'

# Variable filename
glob_pattern: 'private/var/folders/*/*/0/com.apple.routined/dv/Cache/Cloud*.sqlite'

# Multi-profile (browser)
glob_pattern: 'Users/*/Library/Application Support/Google/Chrome/*/History'

# All files in directory
glob_pattern: 'Users/*/Library/Caches/Firefox/Profiles/*/cache2/*'

# Recursive with structure preservation
glob_pattern: 'private/var/db/uuidtext/**/*'
```

**Scope Detection:**

- Paths starting with `Users/` are `user` scope
- All other paths are `system` scope

## Archive Configuration

For artifacts with compressed archives (e.g., Powerlog .gz files):

```yaml
has_archives: true
archives:
  - name: Archives
    subpath: ''              # Relative to artifact location (empty = same folder)
    pattern: '*.gz'          # Archive file pattern
    combine_strategy: decompress_and_merge
```

### Combine Strategies

| Strategy | Use Case |
| ---------- | ---------- |
| `decompress_and_merge` | SQLite databases - merge rows |
| `decompress_and_concatenate` | Text logs - append content |
| `decompress_only` | Just decompress, don't combine |

## Multi-Profile Support

For browser-style artifacts with profile folders:

```yaml
name: Chrome History
glob_pattern: 'Users/*/Library/Application Support/Google/Chrome/*/History'
multi_profile: true
```

The final `*` before the filename is treated as the profile name (e.g., "Default", "Profile 1").

## Module Files

| File | Purpose |
| ------ | --------- |
| `artifact_recovery_catalog.yaml` | The catalog data |
| `catalog_editor.py` | Programmatic editing with format preservation |
| `glob_validator.py` | Validate glob patterns |

## Usage

The catalog is loaded automatically during Exemplar scans. To edit:

1. Use the ARC Manager in the MARS TUI (Settings > ARC Manager)
2. Or edit `artifact_recovery_catalog.yaml` directly

The `CatalogEditor` class preserves YAML comments and formatting when making programmatic changes.

## Adding New Artifacts

1. Choose the appropriate category group (or create a new one)
2. Add entry with required fields: `name`, `glob_pattern`, `file_type`, `scope`, `exemplar_pattern`
3. Test the glob pattern matches files on a sample system
4. Add optional fields as needed (archives, ignorable_tables, etc.)

Example minimal entry:

```yaml
my_category:
  - name: My Database
    glob_pattern: 'Users/*/Library/My App/data.db'
    file_type: database
    scope: user
    exemplar_pattern: databases/catalog/My Database*
```
