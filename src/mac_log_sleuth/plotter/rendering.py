#!/usr/bin/env python3

"""
Plotly rendering functions for the plotter.
"""

from datetime import datetime
from pathlib import Path

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from mac_log_sleuth.plotter.tui import cerr, cwarn
from mac_log_sleuth.plotter.utils import (
    bin_time_series,
    compute_chart_title,
    ensure_outdir,
    human_time_from_epoch,
    is_binary_series,
    rolling_mean,
    truncate_label,
)


def create_overlay_figure(
    xs: list[float],
    ys: dict[str, list[float | None]],
    style: str,
    tzmode: str,
    offset_minutes: int,
    step_for_binary: bool,
    title: str,
    x_range: tuple[float | None, float | None] | None = None,
) -> go.Figure:
    labels = list(ys.keys())
    x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs]
    fig = go.Figure()

    if not labels:
        return fig

    primary = labels[0]
    primary_data = ys[primary]
    if style == "line":
        shape1 = (
            "hv" if (step_for_binary and is_binary_series(primary_data)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=primary_data,
                mode="lines+markers",
                name=primary,
                yaxis="y1",
                line_shape=shape1,
            )
        )
    elif style == "scatter":
        fig.add_trace(
            go.Scatter(x=x_dt, y=primary_data, mode="markers", name=primary, yaxis="y1")
        )
    elif style == "bar":
        fig.add_trace(
            go.Bar(x=x_dt, y=primary_data, name=primary, yaxis="y1", opacity=0.75)
        )

    if len(labels) >= 2:
        secondary = labels[1]
        secondary_data = ys[secondary]
        if style == "line":
            shape2 = (
                "hv"
                if (step_for_binary and is_binary_series(secondary_data))
                else "linear"
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=secondary_data,
                    mode="lines+markers",
                    name=secondary,
                    yaxis="y2",
                    line_shape=shape2,
                )
            )
        elif style == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=secondary_data,
                    mode="markers",
                    name=secondary,
                    yaxis="y2",
                )
            )
        elif style == "bar":
            fig.add_trace(
                go.Bar(
                    x=x_dt,
                    y=secondary_data,
                    name=secondary,
                    yaxis="y1",
                    opacity=0.55,
                )
            )

    fig.update_layout(
        title=title,
        xaxis={
            "title": "Time",
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": labels[0],
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
        height=600,
    )

    if len(labels) >= 2 and style != "bar":
        fig.update_layout(
            yaxis2={
                "title": labels[1],
                "overlaying": "y",
                "side": "right",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
    elif len(labels) >= 2 and style == "bar":
        fig.update_layout(
            yaxis2={
                "title": labels[1],
                "overlaying": "y",
                "side": "right",
                "matches": "y",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
        fig.update_layout(barmode="group")

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    fig.update_layout(
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "text": f"X-axis: {tzlabel}. Data plotted as acquired from Powerlog.",
                "showarrow": False,
                "font": {"size": 11},
            }
        ]
    )

    return fig

def render_stacked_lines(
    outdir: Path,
    base_title: str,
    tzmode: str,
    offset_minutes: int,
    panels: list[dict],  # each: {label:str, xs:list[float], ys:list[float|None]}
    step_for_binary: bool = True,
    smooth_window: int = 1,
    x_range: tuple[float | None, float | None] | None = None,
) -> Path:
    """
    Render stacked line panels (1 per series). Each panel is independent and
    shares the time axis. Returns HTML path. PNG export handled by caller.
    """
    if not panels:
        raise ValueError("No panels to plot.")

    rows = len(panels)
    subplot_titles = [truncate_label(p.get("label", "")) for p in panels]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
    )
    all_xs: list[float] = []
    for i, panel in enumerate(panels, start=1):
        xs = panel.get("xs", [])
        ys = panel.get("ys", [])
        label = panel.get("label", "")
        all_xs.extend(xs)
        is_bool = is_binary_series(ys)
        yplot = (
            ys if (is_bool or smooth_window <= 1) else rolling_mean(list(ys), smooth_window)
        )
        x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs]
        shape = "hv" if (step_for_binary and is_bool) else "linear"
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=yplot,
                mode="lines+markers",
                name=label,  # Full label for legend
                line_shape=shape,
            ),
            row=i,
            col=1,
        )

    chart_title, start_epoch, end_epoch = compute_chart_title(
        base_title, all_xs, x_range, tzmode, offset_minutes
    )
    fig.update_layout(
        title=chart_title,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
        height=max(600, 350 + 220 * (rows - 1)),
    )

    if start_epoch is not None and end_epoch is not None:
        start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
        end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
        for row in range(1, rows + 1):
            fig.update_xaxes(range=[start_dt, end_dt], row=row, col=1)

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    annotations = list(fig.layout.annotations)
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.28,
            "text": f"X-axis: {tzlabel}. Data plotted as acquired from Powerlog.",
            "showarrow": False,
            "font": {"size": 11},
        }
    )
    fig.update_layout(annotations=annotations)

    outdir = ensure_outdir(outdir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    html_path = outdir / f"stacked.{rows}.{ts}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    return html_path


def render_plot_html(
    outdir: Path,
    base_title: str,
    table: str,
    tzmode: str,
    offset_minutes: int,
    xs: list[float],
    ys: dict[str, list[float | None]],
    style: str,
    y2_col: str | None = None,
    *,
    bar_mode: str = "group",  # "group" or "overlay"
    smooth_window: int = 1,
    step_for_binary: bool = True,
    line_layout: str = "overlay",  # "overlay" or "stacked"
    x_range: tuple[float | None, float | None] | None = None,
) -> Path:
    """
    Build Plotly figure and save as HTML. Returns the HTML path.
    """
    # For bar charts, bin densely-sampled data so bars are visible
    xs_used, ys_used = xs, ys
    width_ms = 0.0
    if style == "bar":
        xs_used, ys_used, width_ms = bin_time_series(xs, ys, target_bars=120)
    # Detect boolean series on original data (pre-smoothing)
    binary_map = {k: is_binary_series(ys[k]) for k in ys}
    # Convert X to human-readable datetimes in chosen tz
    x_dt = [human_time_from_epoch(x, tzmode, offset_minutes) for x in xs_used]

    fig = go.Figure()

    # Optional smoothing for non-bar styles (skip boolean series)
    if style != "bar" and smooth_window and smooth_window > 1:
        ys_used = {
            k: (v if binary_map.get(k, False) else rolling_mean(v, smooth_window))
            for k, v in ys_used.items()
        }

    series = list(ys_used.keys())
    if not series:
        raise ValueError("No Y series to plot.")

    # width_ms is suggested by bin_time_series when style == "bar"
    chart_title, start_epoch, end_epoch = compute_chart_title(
        base_title, xs_used, x_range, tzmode, offset_minutes
    )

    # This is dead code as of the current REPL logic, but kept for potential future use.
    if style == "line" and y2_col and (line_layout.lower().startswith("stack")):
        # Build stacked subplots sharing X
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(truncate_label(series[0]), truncate_label(y2_col)),
        )
        shape1 = (
            "hv" if (step_for_binary and binary_map.get(series[0], False)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[series[0]],
                mode="lines+markers",
                name=truncate_label(series[0]),
                line_shape=shape1,
            ),
            row=1,
            col=1,
        )
        shape2 = (
            "hv" if (step_for_binary and binary_map.get(y2_col, False)) else "linear"
        )
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[y2_col],
                mode="lines+markers",
                name=truncate_label(y2_col),
                line_shape=shape2,
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            title=chart_title,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
            },
            margin={"l": 60, "r": 60, "t": 70, "b": 100},
            height=700,
        )
        if start_epoch is not None and end_epoch is not None:
            start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
            end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
            fig.update_xaxes(range=[start_dt, end_dt], row=1, col=1)
            fig.update_xaxes(range=[start_dt, end_dt], row=2, col=1)

        panel_titles = {truncate_label(series[0]), truncate_label(y2_col)}
        for ann in fig.layout.annotations:
            if ann.text in panel_titles:
                ann.update(
                    x=0.5, xanchor="center", y=1.08, yanchor="bottom", font={"size": 14}
                )

        tzlabel = {
            "LOCAL": "Local time",
            "UTC": "UTC",
            "OFFSET": f"UTC{offset_minutes:+d}m",
        }.get(tzmode, "Local time")
        annotations = list(fig.layout.annotations)
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "text": (
                    f"X-axis: human-readable {tzlabel}. "
                    f"Series: {truncate_label(series[0])} (top), {truncate_label(y2_col)} (bottom). "
                    "Data plotted as acquired from Powerlog."
                ),
                "showarrow": False,
                "font": {"size": 11},
            }
        )
        fig.update_layout(annotations=annotations)
        # Write and return (reuse writer below)
        outdir = ensure_outdir(outdir)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")

        def safe_name(s: str | None) -> str:
            s = (s or "").strip()
            safe = []
            for ch in s:
                if ch.isalnum() or ch in ("_", "-", ".", "+"):
                    safe.append(ch)
                else:
                    safe.append("_")
            out = "".join(safe)
            return out or "series"

        fname = f"{safe_name(table)}.{safe_name(series[0])}-{safe_name(y2_col)}.line-stacked.{ts}.html"
        html_path = outdir / fname
        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        except Exception:
            from plotly.offline import plot as plotly_plot  # type: ignore

            plotly_plot(
                fig, filename=str(html_path), auto_open=True, include_plotlyjs="cdn"
            )
        if not html_path.exists():
            cwarn(f"HTML not found after write attempt: {html_path}")
        return html_path

    # Primary series
    y1 = series[0]
    if style == "line":
        shape1 = "hv" if (step_for_binary and binary_map.get(y1, False)) else "linear"
        fig.add_trace(
            go.Scatter(
                x=x_dt,
                y=ys_used[y1],
                mode="lines+markers",
                name=y1,
                yaxis="y1",
                line_shape=shape1,
            )
        )
    elif style == "scatter":
        fig.add_trace(
            go.Scatter(x=x_dt, y=ys_used[y1], mode="markers", name=y1, yaxis="y1")
        )
    elif style == "bar":
        fig.add_trace(
            go.Bar(
                x=x_dt,
                y=ys_used[y1],
                name=y1,
                yaxis="y1",
                width=width_ms if width_ms > 0 else None,
                opacity=0.75,
            )
        )
    else:
        raise ValueError("Unknown style.")

    # Optional secondary series on y2 (overlay)
    y2_on_y2_axis = (
        y2_col
        and y2_col in ys_used
        and (style != "bar" or bar_mode.lower().startswith("over"))
    )

    if y2_col and y2_col in ys_used:
        y2_axis = "y2" if y2_on_y2_axis else "y1"
        if style == "line":
            shape2 = (
                "hv"
                if (step_for_binary and binary_map.get(y2_col, False))
                else "linear"
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=ys_used[y2_col],
                    mode="lines+markers",
                    name=y2_col,
                    yaxis=y2_axis,
                    line_shape=shape2,
                )
            )
        elif style == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=x_dt,
                    y=ys_used[y2_col],
                    mode="markers",
                    name=y2_col,
                    yaxis=y2_axis,
                )
            )
        elif style == "bar":
            fig.add_trace(
                go.Bar(
                    x=x_dt,
                    y=ys_used[y2_col],
                    name=y2_col,
                    yaxis=y2_axis,
                    width=width_ms if width_ms > 0 else None,
                    opacity=0.55,
                )
            )
        else:
            pass

    # Layout
    fig.update_layout(
        title=chart_title,
        xaxis={
            "title": "Time",
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": series[0],
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 60, "t": 70, "b": 100},
    )

    if y2_on_y2_axis:
        fig.update_layout(
            yaxis2={
                "title": y2_col,
                "overlaying": "y",
                "side": "right",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )
    elif (y2_col in series) and style == "bar":
        fig.update_layout(
            yaxis2={
                "title": y2_col,
                "overlaying": "y",
                "side": "right",
                "matches": "y",
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            },
            barmode="group",
        )
    elif style == "bar":
        fig.update_layout(
            barmode=("overlay" if bar_mode.lower().startswith("over") else "group")
        )

    if start_epoch is not None and end_epoch is not None:
        start_dt = human_time_from_epoch(start_epoch, tzmode, offset_minutes)
        end_dt = human_time_from_epoch(end_epoch, tzmode, offset_minutes)
        fig.update_xaxes(range=[start_dt, end_dt])

    tzlabel = {
        "LOCAL": "Local time",
        "UTC": "UTC",
        "OFFSET": f"UTC{offset_minutes:+d}m",
    }.get(tzmode, "Local time")
    foot = (
        f"X-axis: human-readable {tzlabel}. "
        f"Series plotted: {', '.join(series)}. Data plotted as acquired from Powerlog."
    )
    annotations = list(fig.layout.annotations)
    annotations.append(
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.28,
            "text": foot,
            "showarrow": False,
            "font": {"size": 11},
        }
    )
    fig.update_layout(annotations=annotations, height=600)

    def safe_name(s: str | None) -> str:
        s = (s or "").strip()
        # replace any path-separator or problematic chars
        safe = []
        for ch in s:
            if ch.isalnum() or ch in ("_", "-", ".", "+"):
                safe.append(ch)
            else:
                safe.append("_")
        out = "".join(safe)
        return out or "series"

    outdir = ensure_outdir(outdir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"{safe_name(table)}.{safe_name(series[0])}{('-' + safe_name(y2_col)) if y2_col else ''}.{style}.{ts}.html"
    html_path = outdir / fname

    try:
        # Prefer the explicit writer to ensure file creation across Plotly versions
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        # Fallback to offline.plot if available
        try:
            from plotly.offline import plot as plotly_plot  # type: ignore

            plotly_plot(
                fig, filename=str(html_path), auto_open=False, include_plotlyjs="cdn"
            )
        except Exception as e:
            cerr(f"Failed to write HTML: {e}")
            raise

    if not html_path.exists():
        cwarn(f"HTML not found after write attempt: {html_path}")
    return html_path
