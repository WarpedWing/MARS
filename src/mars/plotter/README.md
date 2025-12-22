# MARS Plotter

Interactive timeline visualization module for exploring recovered SQLite database data.

## Features

### Automatic Column Detection

The plotter automatically identifies plottable data by finding:

- **Timestamp columns**: Identified via rubric metadata or column name patterns
- **Numeric columns**: INTEGER and REAL columns suitable for Y-axis plotting

### Multi-Database Support

Select and combine up to 5 series from different databases:

- Mix exemplar and candidate scan data
- Compare metrics across different artifacts
- Overlay related time series for correlation analysis

### Chart Types

| Type | Best For |
| ------ | ---------- |
| **Line** | Continuous time series, default for single series |
| **Step** | Binary (0/1) data, automatically detected |
| **Scatter** | Sparse or irregular data points |
| **Bar** | Aggregated counts, automatically time-binned |
| **Overlay** | Comparing 1-2 series on same axes |
| **Stacked** | Multiple series in vertically aligned panels |

### Interactive Features

Generated HTML charts include:

- Pan and zoom
- Hover tooltips with data values
- Series toggle (click legend to show/hide)
- Range selection
- Export to PNG (via Plotly toolbar, or with Kaleido in MARS)

### Timestamp Format Support

The plotter handles various macOS timestamp formats:

| Format | Description |
| ------ | ----------- |
| `mac_absolute_time` | Seconds since 2001-01-01 (Core Data, many Apple DBs) |
| `unix_seconds` | Seconds since 1970-01-01 |
| `unix_milliseconds` | Milliseconds since 1970-01-01 |
| `webkit_timestamp` | Microseconds since 1601-01-01 (Chrome/WebKit) |
| ISO 8601 strings | Parsed automatically |

## Module Components

| File | Purpose |
| ------ | ---------- |
| `sqlite_plotter.py` | Core REPL interface and series selection UI |
| `rendering.py` | Plotly figure creation and HTML generation |
| `db.py` | Database interaction, schema detection, data fetching |
| `selection_buffer.py` | Multi-database series accumulation (max 5 series) |
| `utils.py` | Time handling, binning, formatting utilities |
| `tui.py` | Rich-based terminal UI styling |

## Workflow

### Interactive Mode

1. **Source Selection**: Choose between Exemplar or Candidate databases
2. **Database Selection**: Pick from plottable databases (pre-filtered)
3. **Table Selection**: Choose table with timestamp and numeric columns
4. **Column Selection**: Select Y-axis columns to plot
5. **Label Column** (optional): Select TEXT column for hover tooltips
6. **Chart Configuration**: Choose type, timezone, dark mode
7. **Export**: HTML saved to `reports/plots/`, opens in browser

### Series Buffer

The selection buffer accumulates series across multiple database selections:

```text
Buffer: [Series 1] [Series 2] [Series 3] ... (max 5)
         ↑ from DB A  ↑ from DB B  ↑ from DB A
```

Render when ready to generate combined visualization.

## Output

Charts are saved to:

```text
output/MARS_*/reports/plots/
```

Filename format: `{db_name}_{table}_{columns}_{timestamp}.html`

## Data Processing

### Time Window Filtering

Optionally restrict plots to specific date ranges to focus on relevant time periods.

### Rolling Mean Smoothing

Apply moving average smoothing to noisy time series data.

### Binary Detection

Series containing only 0 and 1 values are automatically rendered as step functions for clarity.

### Intelligent Binning

Bar charts automatically compute appropriate time bins based on data range:

- Sub-day ranges: Minute/hour bins
- Multi-day ranges: Day bins
- Multi-month ranges: Week/month bins

## Integration

The plotter integrates with MARS via:

- **CLI entry**: Main menu → Chart Plotter
- **Project context**: Accesses exemplar and candidate scan results
- **Rubric system**: Uses schema rubrics for timestamp format detection

## Dependencies

| Package | Required | Purpose |
| --------- | ---------- | --------- |
| `plotly` | Yes | Interactive chart generation |
| `rich` | Yes | Terminal UI |
| `kaleido` | No | PDF/PNG static export |

## Usage Tips

- **Start with single series** to understand data patterns before combining
- **Use overlay for comparison** when series share similar scales
- **Use stacked for correlation** when comparing different metrics
- **Check rubric availability** if timestamp conversion seems wrong
- **Export PNG** from Plotly toolbar for reports (or use Kaleido for scripted export)
