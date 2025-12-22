# MARS Report Modules

Report modules are plugins that run post-processing tasks after the main scan pipeline completes.
Each module is self-contained in its own directory with a `mars_module.yaml` configuration file.

## Overview

The module system provides:

- **Automatic discovery** - Modules are discovered by scanning for `mars_module.yaml` files
- **Scan type filtering** - Modules specify when they run (exemplar, candidate, or free scans)
- **Target resolution** - Modules can target specific database types or the scan root
- **Argument building** - Arguments are configured in YAML and built automatically
- **Progress reporting** - Modules can optionally report progress to the UI

## Directory Structure

```text
report_modules/
│
├── __init__.py
├── module_config.py          # Configuration parsing
├── module_runner.py          # Module execution
├── report_module_manager.py  # Discovery and orchestration
├── argument_builder.py       # Argument construction
├── target_resolver.py        # Target path resolution
├── progress_interface.py     # Progress reporting API
├── README.md                 # This file
│
├── my_module/                # Example module
│   ├── mars_module.yaml       # Module configuration (required)
│   ├── my_entry.py           # Entry point with main() function
│   └── helpers.py            # Additional module files
│
└── ...
```

## Module Configuration (mars_module.yaml)

Each module requires an `mars_module.yaml` file in its directory.

### Required Fields

```yaml
module_info:
  name: "Display Name"              # Human-readable name for logs and UI
  report_folder_name: "output_dir"  # Created in reports/ directory
  version: "1.0"                    # Version string (informational)
  description: "What this does"     # Description (informational)
  scan_type: ["exemplar"]           # When to run (see Scan Types below)
  target: "root"                    # What to scan (see Targets below)
  entry: "my_entry"                 # Entry point filename (without .py)
  active: True                      # Enable/disable module
```

### Optional Fields

```yaml
module_info:
  dependencies: ["pandas"]          # External dependencies (informational)
  readme: "README.md"               # Optional readme file in module dir
```

### Scan Types

The `scan_type` field controls when a module runs. Valid values:

| Value | Description |
| ------- | ------------- |
| `exemplar` | Runs during exemplar scans |
| `candidate` | Runs during candidate scans |
| `free` | Runs during Free Match scans |

Examples:

```yaml
scan_type: ["exemplar"]                    # Only exemplar scans
scan_type: ["candidate"]                   # Only candidate scans
scan_type: ["exemplar", "candidate"]       # Both standard scan types
scan_type: ["exemplar", "candidate", "free"]  # All scan types
```

### Targets

The `target` field specifies what the module processes:

| Target | Description |
| -------- | ------------- |
| `"root"` | Module receives the scan root path; it handles its own file discovery |
| `"Database Name"` | Catalog entry name (e.g., "Firefox Cache"); resolved to actual path(s) |

For catalog-based targets, the module may be called multiple times if multiple matches exist (e.g.,
per-user databases). The output directory will include the username suffix automatically. For example, a target of
`Chrome History`will match `Chrome History_Default`, `Chrome History_admin`, `Chrome_History_usernameX`, etc.

To find the catalog names, reference `catalog/artifact_recovery_catalog.yaml`.

**Important for Free Match modules:** Modules with `scan_type: ["free"]` must use `target: "root"`
because Free Match scans don't use the database catalog. Catalog-based targets won't resolve during free scans.

## Arguments Configuration

Arguments define the command-line interface for the module's `main()` function.

### Argument Structure

```yaml
arguments:
  entry_file_name:        # Must match the module 'entry' field
    - name: "input_path"      # Argument identifier
      flag: null              # null = positional, "--flag" = named
      type: "Path"            # Type: Path, str, int, bool
      help: "Description"     # Help text
      required: True          # Is this required?
      default: null           # Default value
      value: null             # Pre-configured value (overrides default)
      set: False              # For optional args: include if True
      choices: null           # Valid choices (optional)
```

### Argument Types

| Type | Description | Example |
| ------ | ------------- | --------- |
| `Path` | Filesystem path | `/path/to/file` |
| `str` | String value | `"value"` |
| `int` | Integer | `42` |
| `bool` | Boolean flag | `--verbose` (flag only, no value. is either set or not set.) |

### Special Arguments

Two argument names have special handling:

- **`input_path`** - Automatically set to the resolved target path
- **`output_path`** (with `flag: "--out"`) - Automatically set to the output directory

### Argument Examples

**Positional argument:**

```yaml
- name: "input_path"
  flag: null                  # No flag = positional
  type: "Path"
  required: True
```

**Named argument:**

```yaml
- name: "output_path"
  flag: "--out"               # Will be: --out /path/to/output
  type: "Path"
  required: True
```

**Boolean flag:**

```yaml
- name: "verbose"
  flag: "--verbose"
  type: "bool"
  set: True                   # Include the flag (no value)
```

**Optional with default:**

```yaml
- name: "workers"
  flag: "--workers"
  type: "int"
  default: 4
  set: False                  # Don't include unless set: True
```

## Entry Point Requirements

The entry file must have a `main()` function that:

1. Uses `argparse` to parse arguments matching the YAML configuration
2. Reads from `sys.argv` (set by the module runner)
3. Returns `None` (success determined by lack of exceptions)

### Minimal Example

```python
#!/usr/bin/env python3
"""My module entry point."""

import argparse
from pathlib import Path


def main() -> None:
    """Entry point called by ModuleRunner."""
    parser = argparse.ArgumentParser(description="Process files")
    parser.add_argument("input_path", type=Path, help="Input directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Module logic here
    output_file = args.out / "results.csv"
    # ...


if __name__ == "__main__":
    main()
```

## Progress Reporting

Modules can optionally report progress to the UI. This is especially useful for modules that process
many files or have multiple steps.

### Using Progress Reporting

```python
from mars.report_modules.progress_interface import get_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    # ... parse args ...
    args = parser.parse_args()

    files = list(args.input_path.rglob("*.db"))

    # Get progress interface (None if running standalone)
    progress = get_progress()

    # Set total for determinate progress bar
    if progress:
        progress.set_total(len(files))

    for file in files:
        process_file(file)

        # Update progress
        if progress:
            progress.advance()  # Increment by 1
```

### Progress API

```python
from mars.report_modules.progress_interface import get_progress

progress = get_progress()  # Returns ModuleProgress or None

if progress:
    # Set total items (None for indeterminate "throbbing" progress)
    progress.set_total(100)
    progress.set_total(None)      # Indeterminate

    # Update progress
    progress.advance()            # Increment by 1
    progress.advance(5)           # Increment by 5
    progress.update(current=50)   # Set to absolute value

    # Update with message
    progress.advance(message="Processing config.db")
    progress.set_message("Finalizing...")
```

### Progress Display

When modules report progress:

- **Determinate** (total set): Shows `[current/total]` and percentage
- **Indeterminate** (total=None): Shows message only, bar "throbs"
- **No progress calls**: Shows "Running: Module Name"

### Backward Compatibility

Progress reporting is completely optional:

- Modules that don't import `get_progress` work normally
- `get_progress()` returns `None` when running standalone (not via ModuleRunner)
- Always check `if progress:` before calling progress methods

## Inputs and Outputs

### Inputs

Modules receive:

1. **Input path** - Resolved from `target` field:
   - For `"root"`: The scan root directory
   - For catalog names: Path to the matching database/directory

2. **Output directory** - Created automatically:
   - Path: `{scan_root}/reports/{report_folder_name}/`
   - For multi-user targets: `{scan_root}/reports/{report_folder_name}_{username}/`

### Outputs

Modules should write output files to the provided output directory. Common patterns:

```python
# CSV report
csv_path = args.out / "results.csv"
df.to_csv(csv_path, index=False)

# JSON report
json_path = args.out / "summary.json"
with json_path.open("w") as f:
    json.dump(data, f, indent=2)

# Multiple outputs
(args.out / "details").mkdir(exist_ok=True)
for item in items:
    item_path = args.out / "details" / f"{item.name}.json"
    # ...
```

## Complete Example

### mars_module.yaml

```yaml
module_info:
  name: "File Analyzer"
  report_folder_name: "file_analysis"
  version: "1.0"
  description: "Analyze and categorize files by type"
  dependencies: ["python-magic"]
  readme: ""
  scan_type: ["exemplar", "candidate"]
  target: "root"
  entry: "file_analyzer"
  active: True

arguments:
  file_analyzer:
    - name: "input_path"
      flag: null
      type: "Path"
      help: "Directory to analyze"
      required: True
    - name: "output_path"
      flag: "--out"
      type: "Path"
      help: "Output directory"
      required: True
    - name: "recursive"
      flag: "--recursive"
      type: "bool"
      help: "Scan subdirectories"
      set: True
    - name: "min_size"
      flag: "--min-size"
      type: "int"
      default: 0
      help: "Minimum file size in bytes"
      required: False
      set: False
```

### file_analyzer.py

```python
#!/usr/bin/env python3
"""File analyzer module."""

import argparse
import csv
from pathlib import Path

from mars.report_modules.progress_interface import get_progress


def main() -> None:
    """Analyze files and generate report."""
    parser = argparse.ArgumentParser(description="Analyze files")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-size", type=int, default=0)

    args = parser.parse_args()

    # Find files
    pattern = "**/*" if args.recursive else "*"
    files = [f for f in args.input_path.glob(pattern)
             if f.is_file() and f.stat().st_size >= args.min_size]

    # Set up progress
    progress = get_progress()
    if progress:
        progress.set_total(len(files))

    # Process files
    results = []
    for file in files:
        results.append({
            "path": str(file.relative_to(args.input_path)),
            "size": file.stat().st_size,
            "suffix": file.suffix,
        })

        if progress:
            progress.advance()

    # Write output
    output_file = args.out / "file_analysis.csv"
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "suffix"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
```

## Testing Modules

Modules can be tested standalone by running them directly:

```bash
# Test with arguments
python -m mars.report_modules.my_module.my_entry \
    /path/to/input --out /path/to/output --verbose

# Or if entry point is set up
python src/mars/report_modules/my_module/my_entry.py \
    /path/to/input --out /path/to/output
```

When running standalone, `get_progress()` returns `None`, so progress reporting is safely skipped.

## Troubleshooting

### Module not running

1. Check `active: True` in mars_module.yaml
2. Verify `scan_type` includes the scan type you're running
3. Check for validation errors in debug mode

### Target not found

1. Verify the target name matches a catalog entry exactly
2. For custom targets, ensure the path pattern matches files in the scan

### Arguments not working

1. Ensure argument names in YAML match argparse definitions
2. Check `flag` is `null` for positional args
3. For boolean flags, use `set: True` to include them

### Progress not showing

1. Import `get_progress` from `mars.report_modules.progress_interface`
2. Always check `if progress:` before calling methods
3. Call `set_total()` before `advance()` for determinate progress
