"""
HTML Report Generator for Comparison Results.

Generates self-contained HTML reports with Plotly.js charts showing:
- Timeline coverage comparison (exemplar vs candidate date ranges)
- Recovery metrics per database
- Category breakdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger
from mars.utils.platform_utils import get_logo_data_uri

if TYPE_CHECKING:
    from typing import Any

    from mars.pipeline.comparison.types import (
        ComparisonResult,
        DatabaseComparison,
    )


def generate_report(
    result: ComparisonResult,
    output_dir: Path,
    filename: str = "comparison_report.html",
    exemplar_description: str | None = None,
    candidate_description: str | None = None,
    exemplar_report_path: Path | None = None,
    candidate_report_path: Path | None = None,
) -> Path:
    """
    Generate HTML comparison report.

    Args:
        result: ComparisonResult from ComparisonCalculator
        output_dir: Directory to write report to
        filename: Output filename (default: comparison_report.html)
        exemplar_description: Optional description for the exemplar scan
        candidate_description: Optional description for the candidate scan
        exemplar_report_path: Optional path to exemplar's HTML report
        candidate_report_path: Optional path to candidate's HTML report

    Returns:
        Path to generated HTML file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    html_content = _build_html(
        result,
        exemplar_description=exemplar_description,
        candidate_description=candidate_description,
        exemplar_report_path=exemplar_report_path,
        candidate_report_path=candidate_report_path,
    )

    with Path.open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Comparison report generated: {output_path}")
    return output_path


def _build_html(
    result: ComparisonResult,
    exemplar_description: str | None = None,
    candidate_description: str | None = None,
    exemplar_report_path: Path | None = None,
    candidate_report_path: Path | None = None,
) -> str:
    """Build complete HTML document."""
    # Prepare data for charts
    timeline_data = _prepare_timeline_data(result)
    heatmap_data = _prepare_heatmap_data(result)

    # Build table rows
    table_rows = _build_database_table_rows(result.databases)

    # Build source info with descriptions and links
    source_info = _build_source_info(
        result,
        exemplar_description,
        candidate_description,
        exemplar_report_path,
        candidate_report_path,
    )

    logo_data_uri = get_logo_data_uri()

    return rf"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Recovery Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #ca8a04;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        [data-theme="dark"] {{
            --primary: #60a5fa;
            --success: #4ade80;
            --warning: #fbbf24;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 1rem 2rem 2rem 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            font-size: 1.875rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: var(--text);
        }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card-title {{
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-value.success {{ color: var(--success); }}
        .stat-label {{
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        @media (max-width: 1024px) {{
            .charts-row {{ grid-template-columns: 1fr; }}
        }}
        .chart-container {{ min-height: 400px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: var(--bg); }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .badge {{
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        .source-info {{
            margin-top: 0.5rem;
            padding: 0.75rem 1rem;
            background: var(--bg);
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .source-info p {{
            margin: 0.25rem 0;
        }}
        .report-link {{
            color: var(--primary);
            text-decoration: none;
            font-size: 0.9rem;
        }}
        .report-link:hover {{
            text-decoration: underline;
        }}
        /* Theme toggle button */
        .theme-toggle {{
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.875rem;
            color: var(--text);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            z-index: 100;
            transition: background 0.2s, color 0.2s, border-color 0.2s;
        }}
        .theme-toggle:hover {{
            background: var(--bg);
        }}
        /* Timeline range controls */
        .range-controls {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        .range-controls label {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}
        .range-controls input[type="date"] {{
            padding: 0.375rem 0.5rem;
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            background: var(--bg);
            color: var(--text);
            font-size: 0.875rem;
        }}
        .dual-slider {{
            flex: 1;
            min-width: 200px;
            position: relative;
            height: 24px;
        }}
        .dual-slider input[type="range"] {{
            position: absolute;
            width: 100%;
            height: 24px;
            top: 0;
            margin: 0;
            pointer-events: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            background: transparent;
        }}
        .dual-slider input[type="range"]::-webkit-slider-runnable-track {{
            height: 6px;
            background: transparent;
        }}
        .dual-slider input[type="range"]::-moz-range-track {{
            height: 6px;
            background: transparent;
            border: none;
        }}
        .dual-slider input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            pointer-events: auto;
            width: 18px;
            height: 18px;
            margin-top: -6px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: 2px solid var(--card-bg);
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .dual-slider input[type="range"]::-moz-range-thumb {{
            pointer-events: auto;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: 2px solid var(--card-bg);
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        .dual-slider #slider-min {{
            z-index: 1;
        }}
        .dual-slider #slider-max {{
            z-index: 2;
        }}
        .slider-track {{
            position: absolute;
            width: 100%;
            height: 6px;
            top: 9px;
            border-radius: 3px;
            background: var(--border);
        }}
        .slider-range {{
            position: absolute;
            height: 6px;
            top: 9px;
            border-radius: 3px;
            background: var(--primary);
        }}
        /* Dark mode badge adjustments */
        [data-theme="dark"] .badge-success {{ background: #166534; color: #dcfce7; }}
        [data-theme="dark"] .badge-warning {{ background: #92400e; color: #fef3c7; }}
        [data-theme="dark"] .badge-info {{ background: #1e40af; color: #dbeafe; }}
    </style>
</head>
<body>
    <button class="theme-toggle" onclick="toggleTheme()">☀️ Light</button>
    <script>
        // Theme persistence - dark mode is default
        (function() {{
            const saved = localStorage.getItem('mars-theme');
            if (saved === 'light') {{
                document.body.removeAttribute('data-theme');
                document.querySelector('.theme-toggle').textContent = '🌙 Dark';
            }} else {{
                // Default to dark mode
                document.body.setAttribute('data-theme', 'dark');
            }}
        }})();
        function toggleTheme() {{
            const body = document.body;
            const btn = document.querySelector('.theme-toggle');
            const isDark = body.getAttribute('data-theme') === 'dark';
            if (isDark) {{
                body.removeAttribute('data-theme');
                btn.textContent = '🌙 Dark';
                localStorage.setItem('mars-theme', 'light');
                updatePlotlyTheme('light');
            }} else {{
                body.setAttribute('data-theme', 'dark');
                btn.textContent = '☀️ Light';
                localStorage.removeItem('mars-theme');  // Dark is default, no need to store
                updatePlotlyTheme('dark');
            }}
        }}
        function updatePlotlyTheme(theme) {{
            const layout = theme === 'dark' ? {{
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: {{ color: '#e2e8f0' }},
                xaxis: {{ gridcolor: '#334155', tickfont: {{ color: '#94a3b8' }} }},
                yaxis: {{ gridcolor: '#334155', tickfont: {{ color: '#94a3b8' }} }}
            }} : {{
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#ffffff',
                font: {{ color: '#1e293b' }},
                xaxis: {{ gridcolor: '#e2e8f0', tickfont: {{ color: '#64748b' }} }},
                yaxis: {{ gridcolor: '#e2e8f0', tickfont: {{ color: '#64748b' }} }}
            }};
            ['timeline-chart', 'heatmap-chart'].forEach(id => {{
                const el = document.getElementById(id);
                if (el && el.data) Plotly.relayout(id, layout);
            }});
            // Update heatmap colorscale for dark/light mode
            const heatmapEl = document.getElementById('heatmap-chart');
            if (heatmapEl && heatmapEl.data) {{
                const colorscale = theme === 'dark' ? [
                    [0, '#1e293b'],
                    [0.25, '#1e3a5f'],
                    [0.5, '#2563eb'],
                    [0.75, '#60a5fa'],
                    [1, '#93c5fd']
                ] : [
                    [0, '#f1f5f9'],
                    [0.25, '#bfdbfe'],
                    [0.5, '#60a5fa'],
                    [0.75, '#2563eb'],
                    [1, '#1e40af']
                ];
                Plotly.restyle('heatmap-chart', {{ colorscale: [colorscale] }});
            }}
        }}
    </script>
    <div class="container">
        <div
      style="
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.2rem;
      "
        >
            <div style="padding-top: 1.2rem;">
                <h1>Data Recovery Comparison Report</h1>
                <p class="subtitle">Generated: {result.generated_at}</p>
            </div>
            <div>
                <img src="{logo_data_uri}" alt="WarpedWing Labs Logo" height="100px">
            </div>
        </div>

        <!-- Source Info -->
        <div class="source-info">
            {source_info}
        </div>

        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{result.candidate_matched_count} / {result.exemplar_database_count}</div>
                <div class="stat-label">Databases Matched</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{result.total_unique_rows_recovered:,}</div>
                <div class="stat-label">Unique Rows Recovered</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #8b5cf6;">{result.total_lf_rows_recovered:,}</div>
                <div class="stat-label">Reconstituted Rows (L&F)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{result.candidate_with_new_data_count}</div>
                <div class="stat-label">Databases with New Data</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{_count_timeline_extensions(result)}</div>
                <div class="stat-label">Added Days of Data</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #f59e0b;">{result.rebuilt_databases_count}</div>
                <div class="stat-label">Rebuilt DBs (L&F only)</div>
            </div>
        </div>

        <!-- Database Details Table -->
        <div class="card">
            <h3 class="card-title">Database Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>Database</th>
                        <th>Source</th>
                        <th class="num">Exemplar Rows</th>
                        <th class="num">Candidate Rows</th>
                        <th class="num">L&F Rows</th>
                        <th class="num">Overlap</th>
                        <th class="num">Unique Recovered</th>
                        <th>Added Days</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <!-- Timeline Chart -->
        <div class="card">
            <h3 class="card-title">Timeline Coverage</h3>
            <p style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1rem;">
                Shows dates with data: exemplar (blue squares) vs candidate (green squares).
                Gaps indicate missing data. "New Days" counts days where candidate has data but exemplar doesn't
                (includes both range extension AND gap-filling).
            </p>
            <div class="range-controls" id="timeline-controls" style="display: none;">
                <label>Start:</label>
                <input type="date" id="start-date">
                <div class="dual-slider">
                    <div class="slider-track"></div>
                    <div class="slider-range" id="slider-range"></div>
                    <input type="range" id="slider-min" min="0" max="100" value="0">
                    <input type="range" id="slider-max" min="0" max="100" value="100">
                </div>
                <label>End:</label>
                <input type="date" id="end-date">
            </div>
            <div id="timeline-chart" class="chart-container"></div>
        </div>

        <!-- Activity Heatmap -->
        <div class="card">
            <h3 class="card-title">Activity Heatmap</h3>
            <p style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1rem;">
                Calendar heatmap showing data activity intensity by day across all databases.
                Darker colors indicate more timestamps recorded on that date.
            </p>
            <div id="heatmap-chart" style="min-height: 300px; padding: 1rem;"></div>
        </div>

        <div class="footer">
            <p>MARS - Data Recovery Comparison</p>
            {_format_error_info(result)}
        </div>
    </div>

    <script>
        // Timeline Chart - show per-day data points with gaps
        const timelineData = {json.dumps(timeline_data)};
        if (timelineData.length > 0) {{
            const traces = [];
            // Calculate dynamic height: 40px per row, minimum 300px
            const chartHeight = Math.max(300, timelineData.length * 40 + 100);

            // Track whether we've shown legend for each type
            let exemplarLegendShown = false;
            let candidateLegendShown = false;

            timelineData.forEach((db, i) => {{
                // Exemplar dates - show as individual points if available
                if (db.exemplar_dates && db.exemplar_dates.length > 0) {{
                    traces.push({{
                        x: db.exemplar_dates,
                        y: db.exemplar_dates.map(() => db.name),
                        mode: 'markers',
                        marker: {{ size: 25, color: '#2563eb', symbol: 'square' }},
                        name: !exemplarLegendShown ? 'Exemplar' : '',
                        showlegend: !exemplarLegendShown,
                        hoverinfo: 'text',
                        text: db.exemplar_dates.map(d => `Exemplar: ${{d}}`)
                    }});
                    exemplarLegendShown = true;
                }} else if (db.exemplar_start) {{
                    // Fallback to range display
                    traces.push({{
                        x: [db.exemplar_start, db.exemplar_end],
                        y: [db.name, db.name],
                        mode: 'lines',
                        line: {{ width: 25, color: '#2563eb' }},
                        name: !exemplarLegendShown ? 'Exemplar' : '',
                        showlegend: !exemplarLegendShown,
                        hoverinfo: 'text',
                        text: `Exemplar: ${{db.exemplar_start}} to ${{db.exemplar_end}}`
                    }});
                    exemplarLegendShown = true;
                }}

                // Candidate dates - show as individual points if available
                if (db.candidate_dates && db.candidate_dates.length > 0) {{
                    traces.push({{
                        x: db.candidate_dates,
                        y: db.candidate_dates.map(() => db.name),
                        mode: 'markers',
                        marker: {{ size: 15, color: '#4ade80', symbol: 'square' }},
                        name: !candidateLegendShown ? 'Candidate' : '',
                        showlegend: !candidateLegendShown,
                        hoverinfo: 'text',
                        text: db.candidate_dates.map(d => `Candidate: ${{d}}`)
                    }});
                    candidateLegendShown = true;
                }} else if (db.candidate_start) {{
                    // Fallback to range display
                    traces.push({{
                        x: [db.candidate_start, db.candidate_end],
                        y: [db.name, db.name],
                        mode: 'lines',
                        line: {{ width: 15, color: '#4ade80' }},
                        name: !candidateLegendShown ? 'Candidate' : '',
                        showlegend: !candidateLegendShown,
                        hoverinfo: 'text',
                        text: `Candidate: ${{db.candidate_start}} to ${{db.candidate_end}}`
                    }});
                    candidateLegendShown = true;
                }}
            }});

            Plotly.newPlot('timeline-chart', traces, {{
                height: chartHeight,
                margin: {{ t: 40, r: 40, b: 60, l: 250 }},
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: {{ color: '#e2e8f0' }},
                xaxis: {{
                    title: 'Date',
                    type: 'date',
                    tickformat: '%Y-%m-%d',
                    gridcolor: '#334155',
                    tickfont: {{ color: '#94a3b8' }},
                    rangeslider: {{ visible: false }}
                }},
                yaxis: {{
                    automargin: true,
                    tickfont: {{ size: 11, color: '#94a3b8' }},
                    gridcolor: '#334155'
                }},
                legend: {{ orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center' }}
            }}, {{ responsive: true }});

            // Custom range controls
            const allDates = timelineData.flatMap(db => [
                ...(db.exemplar_dates || []),
                ...(db.candidate_dates || [])
            ]).filter(d => d).map(d => new Date(d).getTime());

            if (allDates.length > 0) {{
                const minTime = Math.min(...allDates);
                const maxTime = Math.max(...allDates);
                const minDate = new Date(minTime);
                const maxDate = new Date(maxTime);

                // Show controls
                document.getElementById('timeline-controls').style.display = 'flex';

                // Initialize date inputs
                const startInput = document.getElementById('start-date');
                const endInput = document.getElementById('end-date');
                const sliderMin = document.getElementById('slider-min');
                const sliderMax = document.getElementById('slider-max');
                const sliderRange = document.getElementById('slider-range');

                const formatDate = d => d.toISOString().split('T')[0];
                startInput.value = formatDate(minDate);
                endInput.value = formatDate(maxDate);
                startInput.min = endInput.min = formatDate(minDate);
                startInput.max = endInput.max = formatDate(maxDate);

                // Convert timestamp to slider position (0-100)
                const toSlider = ts => ((ts - minTime) / (maxTime - minTime)) * 100;
                const fromSlider = val => minTime + (val / 100) * (maxTime - minTime);

                // Update chart range
                const updateChart = () => {{
                    Plotly.relayout('timeline-chart', {{
                        'xaxis.range': [startInput.value, endInput.value]
                    }});
                }};

                // Update slider range highlight
                const updateSliderRange = () => {{
                    const minVal = parseFloat(sliderMin.value);
                    const maxVal = parseFloat(sliderMax.value);
                    sliderRange.style.left = minVal + '%';
                    sliderRange.style.width = (maxVal - minVal) + '%';
                }};

                // Date input handlers
                startInput.addEventListener('change', () => {{
                    if (new Date(startInput.value) > new Date(endInput.value)) {{
                        startInput.value = endInput.value;
                    }}
                    sliderMin.value = toSlider(new Date(startInput.value).getTime());
                    updateSliderRange();
                    updateChart();
                }});

                endInput.addEventListener('change', () => {{
                    if (new Date(endInput.value) < new Date(startInput.value)) {{
                        endInput.value = startInput.value;
                    }}
                    sliderMax.value = toSlider(new Date(endInput.value).getTime());
                    updateSliderRange();
                    updateChart();
                }});

                // Slider handlers
                sliderMin.addEventListener('input', () => {{
                    if (parseFloat(sliderMin.value) > parseFloat(sliderMax.value)) {{
                        sliderMin.value = sliderMax.value;
                    }}
                    startInput.value = formatDate(new Date(fromSlider(sliderMin.value)));
                    updateSliderRange();
                    updateChart();
                }});

                sliderMax.addEventListener('input', () => {{
                    if (parseFloat(sliderMax.value) < parseFloat(sliderMin.value)) {{
                        sliderMax.value = sliderMin.value;
                    }}
                    endInput.value = formatDate(new Date(fromSlider(sliderMax.value)));
                    updateSliderRange();
                    updateChart();
                }});

                // Swap z-index on interaction so the active thumb is always on top
                sliderMin.addEventListener('mousedown', () => {{
                    sliderMin.style.zIndex = 3;
                    sliderMax.style.zIndex = 2;
                }});
                sliderMax.addEventListener('mousedown', () => {{
                    sliderMax.style.zIndex = 3;
                    sliderMin.style.zIndex = 1;
                }});

                updateSliderRange();
            }}
        }} else {{
            document.getElementById('timeline-chart').innerHTML =
                '<p style="color: #64748b; text-align: center; padding: 3rem;">No timeline data available</p>';
        }}

        // Activity Heatmap - calendar view of data activity
        const heatmapData = {json.dumps(heatmap_data)};
        if (heatmapData.dates.length > 0) {{
            // Group dates by week for calendar layout
            const dateCounts = {{}};
            heatmapData.dates.forEach(d => {{
                dateCounts[d] = (dateCounts[d] || 0) + 1;
            }});

            // Get date range
            const sortedDates = Object.keys(dateCounts).sort();
            const minDate = new Date(sortedDates[0]);
            const maxDate = new Date(sortedDates[sortedDates.length - 1]);

            // Create week-based grid
            const weeks = [];
            const days = [];
            const counts = [];
            const hoverTexts = [];

            let currentDate = new Date(minDate);
            // Align to start of week (Sunday)
            currentDate.setDate(currentDate.getDate() - currentDate.getDay());

            while (currentDate <= maxDate) {{
                const weekStart = new Date(currentDate);
                for (let dayOfWeek = 0; dayOfWeek < 7; dayOfWeek++) {{
                    const d = new Date(currentDate);
                    d.setDate(d.getDate() + dayOfWeek);

                    const dateStr = d.toISOString().split('T')[0];
                    const count = dateCounts[dateStr] || 0;

                    weeks.push(weekStart.toISOString().split('T')[0]);
                    days.push(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][dayOfWeek]);
                    counts.push(count);
                    hoverTexts.push(count > 0 ? `${{dateStr}}: ${{count}} record(s)` : dateStr);
                }}
                currentDate.setDate(currentDate.getDate() + 7);
            }}

            // Get unique weeks for x-axis
            const uniqueWeeks = [...new Set(weeks)];

            // Reshape data for heatmap (7 rows x N weeks)
            const zData = [[], [], [], [], [], [], []];
            const textData = [[], [], [], [], [], [], []];
            for (let i = 0; i < counts.length; i++) {{
                const dayIdx = i % 7;
                zData[dayIdx].push(counts[i]);
                textData[dayIdx].push(hoverTexts[i]);
            }}

            Plotly.newPlot('heatmap-chart', [{{
                z: zData,
                x: uniqueWeeks,
                y: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
                type: 'heatmap',
                colorscale: [
                    [0, '#1e293b'],
                    [0.25, '#1e3a5f'],
                    [0.5, '#2563eb'],
                    [0.75, '#60a5fa'],
                    [1, '#93c5fd']
                ],
                showscale: true,
                hoverinfo: 'text',
                text: textData,
                colorbar: {{
                    title: 'Records',
                    thickness: 15
                }}
            }}], {{
                margin: {{ t: 20, r: 80, b: 60, l: 60 }},
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: {{ color: '#e2e8f0' }},
                xaxis: {{
                    title: 'Date',
                    type: 'date',
                    tickformat: '%Y-%m-%d',
                    gridcolor: '#334155',
                    tickfont: {{ color: '#94a3b8' }}
                }},
                yaxis: {{
                    autorange: 'reversed',
                    tickfont: {{ color: '#94a3b8' }}
                }}
            }}, {{ responsive: true }});
        }} else {{
            document.getElementById('heatmap-chart').innerHTML =
                '<p style="color: #64748b; text-align: center; padding: 3rem;">No activity data available</p>';
        }}

        // Charts are created with dark mode colors by default
        // Apply light theme to charts if light mode was saved
        if (localStorage.getItem('mars-theme') === 'light') {{
            updatePlotlyTheme('light');
        }}
    </script>
</body>
</html>"""


def _build_source_info(
    result: ComparisonResult,
    exemplar_description: str | None,
    candidate_description: str | None,
    exemplar_report_path: Path | None,
    candidate_report_path: Path | None,
) -> str:
    """Build HTML for source information section with descriptions and links."""
    lines = []

    # Exemplar info - "Exemplar:" is the clickable link, description follows
    exemplar_text = exemplar_description or result.exemplar_scan_dir or "Unknown"
    if exemplar_report_path and exemplar_report_path.exists():
        lines.append(
            f'<p><a href="{exemplar_report_path}" class="report-link">'
            f"<strong>Exemplar:</strong></a> {exemplar_text}</p>"
        )
    else:
        lines.append(f"<p><strong>Exemplar:</strong> {exemplar_text}</p>")

    # Candidate info - "Candidate:" is the clickable link, description follows
    candidate_text = candidate_description or result.candidate_run_dir or "Unknown"
    if candidate_report_path and candidate_report_path.exists():
        lines.append(
            f'<p><a href="{candidate_report_path}" class="report-link">'
            f"<strong>Candidate:</strong></a> {candidate_text}</p>"
        )
    else:
        lines.append(f"<p><strong>Candidate:</strong> {candidate_text}</p>")

    return "\n                ".join(lines)


def _prepare_timeline_data(result: ComparisonResult) -> list[dict[str, Any]]:
    """Prepare timeline data for Plotly chart with per-day gap visualization."""
    timeline_data: list[dict[str, Any]] = []

    for db in result.databases:
        for table in db.tables:
            if table.exemplar_date_range or table.candidate_date_range:
                entry: dict[str, Any] = {"name": f"{db.name}.{table.name}"}

                if table.exemplar_date_range:
                    entry["exemplar_start"] = table.exemplar_date_range[0].strftime("%Y-%m-%d")
                    entry["exemplar_end"] = table.exemplar_date_range[1].strftime("%Y-%m-%d")
                else:
                    entry["exemplar_start"] = None
                    entry["exemplar_end"] = None

                if table.candidate_date_range:
                    entry["candidate_start"] = table.candidate_date_range[0].strftime("%Y-%m-%d")
                    entry["candidate_end"] = table.candidate_date_range[1].strftime("%Y-%m-%d")
                else:
                    entry["candidate_start"] = None
                    entry["candidate_end"] = None

                # Add per-day date lists for gap visualization
                # (sentinel dates already filtered by comparison_calculator)
                entry["exemplar_dates"] = table.exemplar_dates
                entry["candidate_dates"] = table.candidate_dates

                timeline_data.append(entry)

    return timeline_data


def _prepare_heatmap_data(result: ComparisonResult) -> dict:
    """Prepare activity heatmap data by aggregating all dates across databases."""
    all_dates = []

    for db in result.databases:
        for table in db.tables:
            # Include both exemplar and candidate dates for activity view
            # (sentinel dates already filtered by comparison_calculator)
            all_dates.extend(table.exemplar_dates)
            all_dates.extend(table.candidate_dates)

    return {"dates": all_dates}


def _build_database_table_rows(databases: list[DatabaseComparison]) -> str:
    """Build HTML table rows for database details."""
    rows = []

    # Sort: first by unique recovered, then by L&F rows (for rebuilt DBs)
    sorted_dbs = sorted(
        databases,
        key=lambda d: (d.total_unique_recovered, d.lost_and_found.total_rows),
        reverse=True,
    )

    for db in sorted_dbs:
        # Calculate timeline extension badge (sum of unique days across all tables)
        if db.has_timeline_extension:
            days = db.total_added_days  # SUM of unique_candidate_days per table
            timeline_badge = f'<span class="badge badge-success">+{days} days</span>'
        else:
            timeline_badge = '<span class="badge badge-info">-</span>'

        # Unique recovered badge
        unique_class = "badge-success" if db.total_unique_recovered > 0 else "badge-info"

        # Lost and found badge - use manifest count (matched L&F rows before dedup)
        lf_rows = db.lost_and_found.total_rows
        lf_badge = f'<span class="badge badge-warning">{lf_rows:,}</span>' if lf_rows > 0 else "-"

        # Determine source type: Carved, Carved + L&F, or Rebuilt
        if db.rebuilt_from_lf:
            source_badge = '<span class="badge" style="background: #fef3c7; color: #92400e;">L&F</span>'
        elif lf_rows > 0:
            source_badge = '<span class="badge" style="background: #dbeafe; color: #1e40af;">Carved + L&F</span>'
        else:
            source_badge = '<span class="badge" style="background: #dcfce7; color: #166534;">Carved</span>'

        row = f"""<tr>
            <td>{db.name}</td>
            <td>{source_badge}</td>
            <td class="num">{db.total_exemplar_rows:,}</td>
            <td class="num">{db.total_candidate_rows:,}</td>
            <td class="num">{lf_badge}</td>
            <td class="num">{db.total_overlap:,}</td>
            <td class="num"><span class="badge {unique_class}">{db.total_unique_recovered:,}</span></td>
            <td>{timeline_badge}</td>
        </tr>"""
        rows.append(row)

    return "\n".join(rows)


def _count_timeline_extensions(result: ComparisonResult) -> int:
    """Sum total added days across all databases and tables."""
    total_days = 0
    for db in result.databases:
        for table in db.tables:
            total_days += table.unique_candidate_days
    return total_days


def _format_error_info(result: ComparisonResult) -> str:
    """Format error information for footer."""
    if result.databases_with_errors == 0:
        return ""

    return f"""<p style="color: #dc2626; margin-top: 0.5rem;">
        Errors: {result.databases_with_errors} databases had comparison errors
        ({result.path_resolution_failures} path resolution failures)
    </p>"""
