import json
import logging
import math
from datetime import datetime
from urllib.parse import parse_qs

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, ctx, dcc, html

from tricys.analysis import metric
from tricys.visualizer.context import resolve_context_token
from tricys.visualizer.data import (
    load_baseline_data,
    load_h5_data,
    load_h5_overview,
    load_results_subset,
    load_summary_data,
)
from tricys.visualizer.filtering import filter_dataframe
from tricys.visualizer.layout import render_log_content

logger = logging.getLogger(__name__)


THEME_COLORS = {
    "paper": "rgba(11, 14, 20, 0.96)",
    "plot": "rgba(5, 7, 10, 0.88)",
    "grid": "rgba(139, 148, 158, 0.16)",
    "font": "#c9d1d9",
    "muted": "#8b949e",
    "accent": "#00d2ff",
}


def _build_viewer_error_copy(viewer_error):
    message = str(viewer_error or "").strip()
    if not message:
        return "", "", ""

    lowered = message.lower()
    if message == "Missing viewer token":
        return (
            "Missing viewer token",
            "This HDF5 viewer page was opened without the signed token required to resolve a file.",
            "Open the file again from TRICYS, or use a viewer URL that includes the token query parameter.",
        )
    if message == "Malformed viewer token":
        return (
            "Malformed viewer token",
            "The viewer URL contains a token, but its structure is incomplete or unreadable.",
            "Open the HDF5 file again from TRICYS to generate a fresh URL instead of reusing a manually edited or truncated link.",
        )
    if message == "Malformed viewer token payload":
        return (
            "Malformed viewer token payload",
            "The viewer token was present, but its payload could not be decoded into a valid request.",
            "Generate a fresh viewer URL and retry. If the problem persists, verify that the token is not being modified by a proxy or copied incorrectly.",
        )
    if message == "Invalid viewer token signature":
        return (
            "Invalid viewer token signature",
            "The viewer token signature does not match the secret configured on the HDF5 shared service.",
            "Make sure the token was issued by the same TRICYS environment and that the backend and HDF5 service are using the same HDF5_VISUALIZER_SECRET value.",
        )
    if "context not found" in lowered:
        return (
            "Viewer context not found",
            "The shared HDF5 service could not find the requested viewer context for this token.",
            "The token may be stale, the shared service may still be using an older context directory, or the context record may have expired. Reopen the file and retry.",
        )
    if message == "Viewer context payload is invalid":
        return (
            "Viewer context payload is invalid",
            "The shared HDF5 service found the stored viewer context file, but its contents could not be parsed.",
            "Delete the broken context file if needed, then reopen the HDF5 file to regenerate a clean viewer context.",
        )
    if "expired" in lowered:
        return (
            "Viewer token expired",
            "The signed token or its stored viewer context has expired before the page finished loading.",
            "Reopen the HDF5 file to generate a fresh token and context.",
        )
    if message == "Viewer token missing context id":
        return (
            "Viewer token missing context id",
            "The viewer token was accepted structurally, but it does not identify any stored viewer context.",
            "Open the HDF5 file again from TRICYS to create a valid token with a context reference.",
        )
    if message == "Viewer context missing file path":
        return (
            "Viewer context missing file path",
            "The stored viewer context exists, but it does not contain the HDF5 file path required to open the results.",
            "Regenerate the viewer context by reopening the HDF5 file. If this repeats, inspect the stored context JSON for incomplete writes.",
        )
    if message == "Viewer context points to an unsupported file":
        return (
            "Unsupported viewer file",
            "The stored viewer context points to a file that is not an .h5 result file.",
            "Open a valid .h5 file from TRICYS and ensure any manually injected context data is removed.",
        )
    if message == "Requested HDF5 file is no longer available":
        return (
            "HDF5 file no longer available",
            "The viewer token and context were valid, but the referenced HDF5 file no longer exists at the recorded path.",
            "Restore the file, or reopen the result from its current location so TRICYS can issue a new viewer context.",
        )
    return (
        "Unable to load HDF5 viewer",
        message,
        "The viewer could not resolve the file for this request. Check that the HDF5 file still exists and that the shared service is using the expected context directory.",
    )


def _apply_figure_theme(figure, height=None):
    figure.update_layout(
        autosize=False,
        width=None,
        template=None,
        paper_bgcolor=THEME_COLORS["paper"],
        plot_bgcolor=THEME_COLORS["plot"],
        font={"color": THEME_COLORS["font"]},
        margin=dict(l=40, r=20, t=50, b=40),
    )
    if height is not None:
        figure.update_layout(height=height)
    figure.update_xaxes(
        gridcolor=THEME_COLORS["grid"],
        zerolinecolor=THEME_COLORS["grid"],
        linecolor=THEME_COLORS["grid"],
        tickfont={"color": THEME_COLORS["muted"]},
        title_font={"color": THEME_COLORS["muted"]},
    )
    figure.update_yaxes(
        gridcolor=THEME_COLORS["grid"],
        zerolinecolor=THEME_COLORS["grid"],
        linecolor=THEME_COLORS["grid"],
        tickfont={"color": THEME_COLORS["muted"]},
        title_font={"color": THEME_COLORS["muted"]},
    )
    return figure


def _empty_figure(kind="line"):
    figure = px.line() if kind == "line" else px.scatter()
    figure.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    figure.add_annotation(
        text="No data available",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": THEME_COLORS["muted"], "size": 14},
    )
    figure.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        range=[0, 1],
        domain=[0, 1],
    )
    figure.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        range=[0, 1],
        domain=[0, 1],
        scaleanchor=None,
    )
    return _apply_figure_theme(figure, height=320)


def _build_baseline_options(jobs_data):
    return [
        {"label": f"Job {job.get('id')}", "value": job.get("id")}
        for job in (jobs_data or [])
        if job.get("id") is not None
    ]


def _get_jobs_df(jobs_data):
    if not jobs_data:
        return pd.DataFrame()
    return pd.DataFrame(jobs_data)


def _get_h5_file(viewer_context):
    if not viewer_context:
        return None
    return viewer_context.get("file_path")


def _load_context_bundle(h5_file_path):
    (
        variable_options,
        parameter_options,
        table_columns,
        jobs_data,
        config_data,
        log_data,
    ) = load_h5_data(h5_file_path)
    return {
        "variable_options": variable_options,
        "parameter_options": parameter_options,
        "table_columns": table_columns,
        "jobs_data": jobs_data,
        "config_data": config_data,
        "log_data": log_data,
    }


def _coerce_job_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_job_ids(job_ids):
    normalized = []
    seen = set()
    for job_id in job_ids or []:
        normalized_id = _coerce_job_id(job_id)
        if normalized_id is None or normalized_id in seen:
            continue
        seen.add(normalized_id)
        normalized.append(normalized_id)
    return normalized


def _get_current_page_job_ids(selected_rows, table_data):
    if not selected_rows or not table_data:
        return []
    current_job_ids = []
    for idx in selected_rows:
        if idx >= len(table_data):
            continue
        current_job_ids.append(table_data[idx].get("id"))
    return _normalize_job_ids(current_job_ids)


def _format_file_size(num_bytes):
    if num_bytes in (None, 0):
        return "--"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "--"


def _format_time_range(start, end):
    if start is None or end is None:
        return "Unavailable", "No /results time axis"
    if start == end:
        return f"{start}", "Single sampled timestamp"
    return f"{start} -> {end}", "Derived from /results"


def _format_modified_at(timestamp):
    if not timestamp:
        return "--", "File metadata unavailable"
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S"), "Local file timestamp"


def _find_jobs_by_ids(jobs_data, selected_job_ids):
    jobs_df = _get_jobs_df(jobs_data)
    if jobs_df.empty:
        return []
    normalized_ids = _normalize_job_ids(selected_job_ids)
    if not normalized_ids:
        return []
    selected_df = jobs_df[jobs_df["id"].isin(normalized_ids)].copy()
    selected_df["_sort_order"] = selected_df["id"].map(
        {job_id: index for index, job_id in enumerate(normalized_ids)}
    )
    selected_df = selected_df.sort_values("_sort_order").drop(columns="_sort_order")
    return selected_df.to_dict("records")


def register_callbacks(app, server_mode=False, context_secret=None, context_dir=None):
    if server_mode:

        @app.callback(
            Output("full-jobs-data-store", "data"),
            Output("variable-options-store", "data"),
            Output("parameter-options-store", "data"),
            Output("config-store", "data"),
            Output("log-store", "data"),
            Output("viewer-context-store", "data"),
            Output("viewer-error-store", "data"),
            Output("jobs-table", "columns"),
            Output("variable-dropdown", "options"),
            Output("baseline-dropdown", "options"),
            Input("viewer-location", "search"),
        )
        def initialize_viewer_from_token(search):
            query = parse_qs((search or "").lstrip("?"))
            token = (query.get("token") or [None])[0]
            if not token:
                return [], [], [], None, None, None, "Missing viewer token", [], [], []

            try:
                viewer_context = resolve_context_token(
                    token,
                    context_secret or "",
                    context_dir or ".hdf5_contexts",
                )
                bundle = _load_context_bundle(viewer_context["file_path"])
                return (
                    bundle["jobs_data"],
                    bundle["variable_options"],
                    bundle["parameter_options"],
                    bundle["config_data"],
                    bundle["log_data"],
                    viewer_context,
                    None,
                    bundle["table_columns"],
                    bundle["variable_options"],
                    _build_baseline_options(bundle["jobs_data"]),
                )
            except Exception as exc:
                return [], [], [], None, None, None, str(exc), [], [], []

    @app.callback(
        Output("viewer-status-alert", "children"),
        Output("viewer-status-alert", "is_open"),
        Input("viewer-context-store", "data"),
        Input("viewer-error-store", "data"),
    )
    def update_viewer_status(viewer_context, viewer_error):
        if viewer_error:
            return "", False
        if viewer_context and viewer_context.get("display_path"):
            return "", False
        return "Open an .h5 file to begin.", True

    @app.callback(
        Output("viewer-main-content", "style"),
        Output("viewer-fatal-error", "style"),
        Output("viewer-fatal-error-title", "children"),
        Output("viewer-fatal-error-message", "children"),
        Output("viewer-fatal-error-detail", "children"),
        Input("viewer-error-store", "data"),
    )
    def toggle_fatal_viewer_error(viewer_error):
        if not viewer_error:
            return {}, {"display": "none"}, "", "", ""

        title, message, detail = _build_viewer_error_copy(viewer_error)
        return (
            {"display": "none"},
            {"display": "flex"},
            title,
            message,
            detail,
        )

    @app.callback(
        Output("metrics-availability-store", "data"),
        Output("metrics-section", "style"),
        Output("metrics-unavailable-alert", "is_open"),
        Input("viewer-context-store", "data"),
    )
    def update_metrics_availability(viewer_context):
        h5_file = _get_h5_file(viewer_context)
        if not h5_file:
            return {"has_summary": False}, {"display": "none"}, False

        overview = load_h5_overview(h5_file)
        has_summary = bool(overview.get("has_summary"))
        return (
            {"has_summary": has_summary},
            {} if has_summary else {"display": "none"},
            not has_summary,
        )

    @app.callback(
        Output("overview-job-count", "children"),
        Output("overview-job-detail", "children"),
        Output("overview-variable-count", "children"),
        Output("overview-variable-detail", "children"),
        Output("overview-time-range", "children"),
        Output("overview-time-detail", "children"),
        Output("overview-dataset-status", "children"),
        Output("overview-dataset-detail", "children"),
        Output("overview-file-size", "children"),
        Output("overview-file-detail", "children"),
        Output("overview-modified-at", "children"),
        Output("overview-modified-detail", "children"),
        Input("viewer-context-store", "data"),
        Input("full-jobs-data-store", "data"),
        Input("variable-options-store", "data"),
        Input("config-store", "data"),
        Input("log-store", "data"),
    )
    def update_overview_cards(
        viewer_context, jobs_data, variable_options, config_data, log_data
    ):
        h5_file = _get_h5_file(viewer_context)
        if not h5_file:
            return (
                "--",
                "No HDF5 file loaded",
                "--",
                "No variables loaded",
                "Unavailable",
                "No /results time axis",
                "No datasets",
                "Open a file to inspect datasets",
                "--",
                "No file metadata available",
                "--",
                "No file metadata available",
            )

        overview = load_h5_overview(h5_file)
        dataset_flags = []
        if overview["has_results"]:
            dataset_flags.append("results")
        if overview["has_summary"]:
            dataset_flags.append("summary")
        if overview["has_config"] or config_data is not None:
            dataset_flags.append("config")
        if overview["has_log"] or log_data is not None:
            dataset_flags.append("log")

        time_range_value, time_range_detail = _format_time_range(
            overview.get("time_start"), overview.get("time_end")
        )
        modified_value, modified_detail = _format_modified_at(
            overview.get("modified_at")
        )

        return (
            str(len(jobs_data or [])),
            "Rows available in jobs table",
            str(len(variable_options or [])),
            "Selectable result variables",
            time_range_value,
            time_range_detail,
            ", ".join(dataset_flags) if dataset_flags else "No datasets",
            f"summary={'yes' if overview['has_summary'] else 'no'} | config={'yes' if (overview['has_config'] or config_data is not None) else 'no'} | log={'yes' if (overview['has_log'] or log_data is not None) else 'no'}",
            _format_file_size(overview.get("file_size_bytes")),
            h5_file,
            modified_value,
            modified_detail,
        )

    @app.callback(
        Output("jobs-table", "data"),
        Output("jobs-table", "page_count"),
        Output("jobs-table", "page_current"),
        Input("full-jobs-data-store", "data"),
        Input("jobs-table", "sort_by"),
        Input("jobs-table", "filter_query"),
        Input("jobs-table", "page_current"),
        Input("jobs-table", "page_size"),
    )
    def update_jobs_table(data, sort_by, filter_query, page_current, page_size):
        if not data:
            return [], 0, 0

        df = pd.DataFrame(data)

        # Filter
        if filter_query:
            try:
                df = filter_dataframe(df, filter_query)
            except Exception:
                logger.exception(
                    "Failed to filter jobs table", extra={"filter_query": filter_query}
                )

        # Sort
        if sort_by and len(sort_by):
            col_name = sort_by[0]["column_id"]
            direction = sort_by[0]["direction"]
            ascending = direction == "asc"
            df = df.sort_values(by=col_name, ascending=ascending)
        else:
            # Default sort by id
            if "id" in df.columns:
                df = df.sort_values(by="id")

        # Pagination
        if page_current is None:
            page_current = 0
        if page_size is None or page_size <= 0:
            page_size = 50

        total_rows = len(df)
        page_count = math.ceil(total_rows / page_size) if total_rows else 0
        if page_count == 0:
            return [], 0, 0
        if page_current >= page_count:
            page_current = max(page_count - 1, 0)

        start = page_current * page_size
        end = start + page_size
        return df.iloc[start:end].to_dict("records"), page_count, page_current

    @app.callback(
        Output("main-data-store", "data"),
        Input("analysis-selection-store", "data"),
        Input("variable-dropdown", "value"),
        State("viewer-context-store", "data"),
    )
    def update_main_data_store(selected_job_ids, selected_variables, viewer_context):
        """Update plot data based on the applied analysis selection."""
        h5_file = _get_h5_file(viewer_context)
        normalized_job_ids = _normalize_job_ids(selected_job_ids)
        if not normalized_job_ids or not selected_variables or not h5_file:
            return None
        try:
            return load_results_subset(h5_file, normalized_job_ids, selected_variables)
        except Exception:
            logger.exception(
                "Failed to update main data store",
                extra={
                    "file_path": h5_file,
                    "job_ids": normalized_job_ids,
                    "variables": list(selected_variables or []),
                },
            )
            return None

    @app.callback(
        Output("baseline-job-store", "data"),
        Output("baseline-job-id-store", "data"),
        Output("btn-view-baseline-details", "disabled"),
        Output("btn-clear-baseline", "disabled"),
        Input("baseline-dropdown", "value"),
        State("viewer-context-store", "data"),
        prevent_initial_call=True,
    )
    def update_baseline_store(selected_job_id, viewer_context):
        if not selected_job_id:
            return None, None, True, True

        try:
            baseline_df = load_baseline_data(
                _get_h5_file(viewer_context), selected_job_id
            )
            if baseline_df is None:
                return None, None, True, True
            return baseline_df.to_dict("records"), selected_job_id, False, False
        except Exception:
            logger.exception(
                "Failed to set baseline",
                extra={
                    "file_path": _get_h5_file(viewer_context),
                    "job_id": selected_job_id,
                },
            )
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output("baseline-dropdown", "value"),
        Input("btn-clear-baseline", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_baseline_selection(n_clicks):
        if not n_clicks:
            return dash.no_update
        return None

    @app.callback(
        Output("plot-type-radio", "options"),
        Output("plot-type-radio", "value"),
        Input("baseline-job-id-store", "data"),
        Input("plot-type-radio", "value"),
    )
    def update_baseline_mode_controls(baseline_job_id, plot_mode):
        has_baseline = baseline_job_id is not None
        options = [
            {"label": "Absolute Values", "value": "absolute"},
            {
                "label": "Difference from Baseline",
                "value": "difference",
                "disabled": not has_baseline,
            },
        ]

        effective_mode = plot_mode
        if plot_mode == "difference" and not has_baseline:
            effective_mode = "absolute"

        return options, effective_mode

    @app.callback(
        Output("results-graph", "figure"),
        Input("main-data-store", "data"),
        Input("baseline-job-store", "data"),
        Input("plot-type-radio", "value"),
        State("variable-dropdown", "value"),
    )
    def update_results_graph(data, baseline_data, plot_type, selected_variables):
        if not data:
            return _empty_figure("line")
        df_wide = pd.DataFrame(data)
        if not selected_variables:
            return _empty_figure("line")

        if plot_type == "difference":
            if not baseline_data:
                return _empty_figure("line")
            baseline_df = pd.DataFrame(baseline_data).set_index("time")
            if not all(v in baseline_df.columns for v in selected_variables):
                return _empty_figure("line")

            diff_dfs = []
            for job_id, group in df_wide.groupby("job_id"):
                group = group.set_index("time")
                # Align series and compute difference
                aligned_group, aligned_baseline = group.align(
                    baseline_df, join="outer", axis=0
                )
                diff = (
                    aligned_group[selected_variables]
                    - aligned_baseline[selected_variables]
                )
                diff["job_id"] = f"Job {job_id} - Baseline"
                diff_dfs.append(diff.reset_index())

            df_long = pd.melt(
                pd.concat(diff_dfs),
                id_vars=["time", "job_id"],
                value_vars=selected_variables,
                var_name="variable",
                value_name="value",
            )
            # Extract baseline job id for title
            y_title = "Difference"
        else:
            df_long = pd.melt(
                df_wide,
                id_vars=["time", "job_id"],
                value_vars=selected_variables,
                var_name="variable",
                value_name="value",
            )
            y_title = "Absolute Value"

        fig = px.line(
            df_long,
            x="time",
            y="value",
            color="job_id",
            facet_row="variable",
            labels={"value": y_title, "time": "Time (h)", "job_id": "Job ID"},
            height=350 * len(selected_variables),
            render_mode="webgl",
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.update_yaxes(matches=None)
        return _apply_figure_theme(fig, height=350 * len(selected_variables))

    @app.callback(
        Output("metrics-summary-container", "style"),
        Output("metrics-plot-container", "style"),
        Output("heatmap-container", "style"),
        Output("parallel-coords-container", "style"),
        Input("metrics-tabs", "active_tab"),
    )
    def toggle_tab_visibility(active_tab):
        show = {"marginTop": "20px"}
        hide = {"display": "none", "marginTop": "20px"}
        if active_tab == "tab-metrics-summary":
            return show, hide, hide, hide
        if active_tab == "tab-metrics-plots":
            return hide, show, hide, hide
        if active_tab == "tab-heatmap-analysis":
            return hide, hide, show, hide
        if active_tab == "tab-parallel-coords":
            return hide, hide, hide, show
        return hide, hide, hide, hide

    @app.callback(
        Output("parcoords-dims-dropdown", "options"),
        Input("metrics-data-store", "data"),
    )
    def update_parcoords_options(metrics_data):
        if not metrics_data:
            return []
        df = pd.DataFrame(metrics_data)
        # Filter numeric columns for parallel coordinates
        numeric_cols = df.select_dtypes(include=["number"]).columns
        options = [{"label": col, "value": col} for col in numeric_cols if col != "id"]
        return options

    @app.callback(
        Output("parcoords-graph", "figure"),
        Input("metrics-data-store", "data"),
        Input("parcoords-dims-dropdown", "value"),
    )
    def update_parallel_coordinates(metrics_data, selected_dims):
        if not metrics_data or not selected_dims:
            return _empty_figure()
        df = pd.DataFrame(metrics_data)

        try:
            fig = px.parallel_coordinates(
                df, dimensions=selected_dims, color=selected_dims[-1]
            )
            return _apply_figure_theme(fig, height=420)
        except Exception:
            logger.exception("Failed to render parallel coordinates plot")
            return _empty_figure()

    @app.callback(
        Output("metrics-data-store", "data"),
        Input("analysis-selection-store", "data"),
        State("full-jobs-data-store", "data"),
        State("variable-dropdown", "value"),
        State("viewer-context-store", "data"),
        State("metrics-availability-store", "data"),
    )
    def calculate_metrics_data(
        selected_job_ids,
        jobs_data,
        selected_variables,
        viewer_context,
        metrics_availability,
    ):
        """
        Loads pre-calculated metrics from HDF5 summary table.
        Merges with job parameters for display.
        """
        h5_file = _get_h5_file(viewer_context)
        if not h5_file or not jobs_data:
            return None

        if not (metrics_availability or {}).get("has_summary"):
            return []

        normalized_job_ids = _normalize_job_ids(selected_job_ids)
        if not normalized_job_ids:
            return []

        try:
            selected_jobs = _find_jobs_by_ids(jobs_data, normalized_job_ids)
            job_ids = [
                row.get("id") for row in selected_jobs if row.get("id") is not None
            ]

            if not job_ids:
                return []

            # Load summary metrics directly
            summary_records = load_summary_data(h5_file, job_ids)
            if not summary_records:
                return []

            # --- Success Logic (Existing) ---
            summary_df = pd.DataFrame(summary_records)

            # Merge with parameters for display context
            metrics_data = []

            # We want to iterate over the jobs in the table order probably,
            # but summary_df usually returns in job_id order or storage order.

            # Create a lookup for parameters
            # Force IDs to int to ensure matching succeeds despite format differences (int vs float vs str)
            params_lookup = {}
            for row in selected_jobs:
                try:
                    rid = int(row.get("id"))
                    params_lookup[rid] = row
                except:
                    continue

            for _, metric_row in summary_df.iterrows():
                try:
                    raw_jid = metric_row.get("job_id")
                    job_id = int(raw_jid)
                except:
                    continue

                if job_id not in params_lookup:
                    continue

                table_row = params_lookup[job_id].copy()

                # Format Job Title
                # Remove internal id for display params
                display_params = {
                    k: v for k, v in table_row.items() if k != "id" and k != "job_id"
                }
                table_row["Job"] = (
                    f"Job {job_id} ({', '.join([f'{k}={v}' for k, v in display_params.items()])})"
                )

                # Update with metric values
                # metric_row has job_id, MetricA, MetricB...
                for k, v in metric_row.items():
                    if k != "job_id":
                        table_row[k] = v

                metrics_data.append(table_row)

            return metrics_data

        except Exception:
            logger.exception(
                "Failed to build metrics summary",
                extra={"file_path": h5_file, "job_ids": normalized_job_ids},
            )
            return None

    @app.callback(
        Output("metrics-summary-table", "data"),
        Output("metrics-summary-table", "columns"),
        Output("xaxis-param-dropdown", "options"),
        Output("yaxis-metric-dropdown", "options"),
        Output("heatmap-x-dropdown", "options"),
        Output("heatmap-y-dropdown", "options"),
        Output("heatmap-z-dropdown", "options"),
        Input("metrics-data-store", "data"),
        State("full-jobs-data-store", "data"),
    )
    def update_metrics_ui(metrics_data, jobs_data):
        if not metrics_data:
            return [], [], [], [], [], [], []

        jobs_df = _get_jobs_df(jobs_data)
        if jobs_df.empty:
            return [], [], [], [], [], [], []

        # Cols for table
        cols = [
            {"name": k.replace("_", " "), "id": k}
            for k in metrics_data[0].keys()
            if k != "id"
        ]

        # Metric options
        metric_keys = [
            k
            for k in metrics_data[0].keys()
            if k not in jobs_df.columns and k != "Job" and k != "id"
        ]

        # Param options
        param_keys = [c for c in jobs_df.columns if c != "id"]

        return (
            metrics_data,
            cols,
            param_keys,
            metric_keys,
            param_keys,
            param_keys,
            metric_keys,
        )

    @app.callback(
        Output("metric-plot-graph", "figure"),
        Input("xaxis-param-dropdown", "value"),
        Input("yaxis-metric-dropdown", "value"),
        State("metrics-data-store", "data"),
        State("full-jobs-data-store", "data"),
    )
    def update_metric_plot(xaxis, yaxis, data, jobs_data):
        if not all([xaxis, yaxis, data]):
            return _empty_figure()
        jobs_df = _get_jobs_df(jobs_data)
        if jobs_df.empty:
            return _empty_figure()
        df = pd.DataFrame(data).sort_values(by=xaxis)
        df[yaxis] = pd.to_numeric(
            df[yaxis].astype(str).str.replace(",", ""), errors="coerce"
        )

        # Identify grouping parameters (all parameters except job_id and the selected xaxis)
        all_params = [c for c in jobs_df.columns if c != "id"]
        grouping_cols = [c for c in all_params if c != xaxis]

        if grouping_cols:
            # Create a 'Condition' column for the legend
            df["Condition"] = df.apply(
                lambda row: ", ".join([f"{col}={row[col]}" for col in grouping_cols]),
                axis=1,
            )
            color_arg = "Condition"
        else:
            color_arg = None

        fig = px.scatter(
            df.dropna(subset=[yaxis]),
            x=xaxis,
            y=yaxis,
            color=color_arg,
            hover_data=["id", "Job"],
        )
        fig.update_traces(mode="lines+markers")
        return _apply_figure_theme(fig, height=360)

    @app.callback(
        Output("heatmap-graph", "figure"),
        Input("heatmap-x-dropdown", "value"),
        Input("heatmap-y-dropdown", "value"),
        Input("heatmap-z-dropdown", "value"),
        State("metrics-data-store", "data"),
    )
    def update_heatmap_plot(x_param, y_param, z_metric, data):
        if not all([x_param, y_param, z_metric, data]):
            return _empty_figure()
        import plotly.graph_objects as go

        df = pd.DataFrame(data)
        # Convert Z to numeric
        df[z_metric] = pd.to_numeric(
            df[z_metric].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=[x_param, y_param, z_metric])

        if df.empty:
            return _empty_figure()

        fig = go.Figure(
            data=go.Contour(
                z=df[z_metric],
                x=df[x_param],
                y=df[y_param],
                colorscale="Viridis",
                connectgaps=True,
                line_smoothing=0.85,
                colorbar=dict(title=z_metric.split("_")[0]),
            )
        )

        fig.update_layout(
            xaxis_title=x_param,
            yaxis_title=y_param,
            autosize=True,
        )
        return _apply_figure_theme(fig, height=420)

    @app.callback(
        Output("jobs-table", "style_data_conditional"),
        Input("metric-plot-graph", "clickData"),
        Input("jobs-table", "selected_rows"),
        Input("analysis-selection-store", "data"),
    )
    def update_table_highlighting(clickData, selected_rows, analysis_selection):
        styles = []
        for job_id in _normalize_job_ids(analysis_selection):
            styles.append(
                {
                    "if": {"filter_query": f"{{id}} = {job_id}"},
                    "backgroundColor": "rgba(0, 210, 255, 0.1)",
                    "border": "1px solid rgba(0, 210, 255, 0.4)",
                }
            )

        if ctx.triggered_id != "jobs-table" and clickData:
            clicked_job_id = clickData["points"][0]["customdata"][0]
            styles.append(
                {
                    "if": {"filter_query": f"{{id}} = {clicked_job_id}"},
                    "backgroundColor": "rgba(0, 116, 217, 0.3)",
                    "border": "1px solid #007bff",
                }
            )

        return styles

    @app.callback(
        Output("download-selected-csv", "data"),
        Input("btn-download-selected", "n_clicks"),
        State("analysis-selection-store", "data"),
        State("viewer-context-store", "data"),
        State("full-jobs-data-store", "data"),
        prevent_initial_call=True,
    )
    def download_selected_jobs_batch(
        n_clicks, analysis_selection, viewer_context, jobs_data
    ):
        job_ids_numeric = _normalize_job_ids(analysis_selection)
        if not job_ids_numeric:
            return dash.no_update

        # Load all selected jobs
        # Note: 'where' clause with 'in' is efficient in PyTables/Pandas HDF
        try:
            df = pd.read_hdf(
                _get_h5_file(viewer_context),
                "results",
                where=f"job_id in {job_ids_numeric}",
            )
        except Exception:
            logger.exception(
                "Failed to export selected batch jobs",
                extra={
                    "file_path": _get_h5_file(viewer_context),
                    "job_ids": job_ids_numeric,
                },
            )
            return dash.no_update

        if df.empty:
            return dash.no_update

        # Merge parameter info into the dataframe columns for clarity
        # We'll pivot or just append columns?
        # Appending columns 'paramA', 'paramB' to each row is safer for "Long Format" export
        # Or we can do "Wide Format" similar to download-all.
        # "Batch" usually implies raw data for multiple jobs. Let's keep it Long Format (standard simulation output)
        # but add parameter columns so users can distinguish/group by parameters in their analysis tools.

        jobs_df = _get_jobs_df(jobs_data)
        if jobs_df.empty:
            return dash.no_update

        params_to_merge = jobs_df[jobs_df["id"].isin(job_ids_numeric)].copy()
        params_to_merge = params_to_merge.rename(columns={"id": "job_id"})

        # Left join to add parameters to the time-series data
        merged_df = pd.merge(df, params_to_merge, on="job_id", how="left")

        return dcc.send_data_frame(
            merged_df.to_csv, "batch_selected_jobs.csv", index=False
        )

    @app.callback(
        Output("download-all-csv", "data"),
        Input("btn-download-all", "n_clicks"),
        State("main-data-store", "data"),
        State("full-jobs-data-store", "data"),
        prevent_initial_call=True,
    )
    def download_all_csv(n_clicks, data, jobs_data):
        if not data:
            return dash.no_update
        jobs_df = _get_jobs_df(jobs_data)
        if jobs_df.empty:
            return dash.no_update
        df_wide, final_df = (
            pd.DataFrame(data),
            pd.DataFrame({"time": pd.DataFrame(data)["time"].unique()}).sort_values(
                "time"
            ),
        )
        for job_id in df_wide["job_id"].unique():
            params = jobs_df.loc[jobs_df["id"] == job_id].iloc[0]
            params_str = "(" + ", ".join([f"{k}={v}" for k, v in params.items()]) + ")"
            vars_df = (
                df_wide[df_wide["job_id"] == job_id]
                .drop(columns="job_id")
                .rename(
                    columns={
                        col: f"{col} {params_str}"
                        for col in df_wide.columns
                        if col not in ["time", "job_id"]
                    }
                )
            )
            final_df = pd.merge(final_df, vars_df, on="time", how="outer")
        return dcc.send_data_frame(final_df.to_csv, "tricys_data_wide.csv", index=False)

    @app.callback(Output("config-view", "children"), Input("config-store", "data"))
    def update_config_view(data):
        if not data:
            return "No configuration data found."
        return json.dumps(data, indent=2)

    @app.callback(Output("log-view", "children"), Input("log-store", "data"))
    def update_log_view(data):
        if not data:
            return "No log data found."
        return render_log_content(data)

    @app.callback(
        Output("analysis-selection-store", "data"),
        Input("btn-apply-selection", "n_clicks"),
        Input("btn-clear-analysis-selection", "n_clicks"),
        State("jobs-table", "selected_rows"),
        State("jobs-table", "data"),
        prevent_initial_call=True,
    )
    def update_analysis_selection(
        apply_clicks, clear_clicks, selected_rows, current_data
    ):
        if ctx.triggered_id == "btn-clear-analysis-selection":
            return []
        if ctx.triggered_id == "btn-apply-selection":
            return _get_current_page_job_ids(selected_rows, current_data)
        return dash.no_update

    @app.callback(
        Output("analysis-selection-feedback-store", "data"),
        Output("analysis-selection-feedback-timer", "disabled"),
        Input("btn-apply-selection", "n_clicks"),
        Input("btn-clear-analysis-selection", "n_clicks"),
        State("jobs-table", "selected_rows"),
        State("jobs-table", "data"),
        prevent_initial_call=True,
    )
    def trigger_analysis_selection_feedback(
        apply_clicks, clear_clicks, selected_rows, current_data
    ):
        if ctx.triggered_id == "btn-clear-analysis-selection":
            return {"flash": False}, True
        if ctx.triggered_id == "btn-apply-selection":
            applied_job_ids = _get_current_page_job_ids(selected_rows, current_data)
            if applied_job_ids:
                return {"flash": True}, False
        return dash.no_update, dash.no_update

    @app.callback(
        Output("analysis-selection-feedback-store", "data", allow_duplicate=True),
        Output("analysis-selection-feedback-timer", "disabled", allow_duplicate=True),
        Input("analysis-selection-feedback-timer", "n_intervals"),
        State("analysis-selection-feedback-store", "data"),
        prevent_initial_call=True,
    )
    def clear_analysis_selection_feedback(n_intervals, feedback_state):
        if not feedback_state or not feedback_state.get("flash"):
            return dash.no_update, True
        return {"flash": False}, True

    @app.callback(
        Output("analysis-selection-panel", "className"),
        Input("analysis-selection-store", "data"),
        Input("analysis-selection-feedback-store", "data"),
    )
    def update_analysis_selection_panel_class(analysis_selection, feedback_state):
        classes = ["selection-stage-panel", "selection-stage-panel--applied", "h-100"]
        if _normalize_job_ids(analysis_selection):
            classes.append("selection-stage-panel--active")
        if feedback_state and feedback_state.get("flash"):
            classes.append("selection-stage-panel--flash")
        return " ".join(classes)

    @app.callback(
        Output("plot-metrics-panel", "className"),
        Output("plot-metrics-lock-overlay", "className"),
        Input("analysis-selection-store", "data"),
    )
    def update_plot_metrics_lock_state(analysis_selection):
        has_applied_jobs = bool(_normalize_job_ids(analysis_selection))
        if has_applied_jobs:
            return "plot-metrics-panel", "plot-metrics-lock-overlay"
        return (
            "plot-metrics-panel plot-metrics-panel--locked",
            "plot-metrics-lock-overlay plot-metrics-lock-overlay--visible",
        )

    @app.callback(
        Output("jobs-table", "selected_rows"),
        Output("select-all-checkbox", "value"),
        Input("select-all-checkbox", "value"),
        Input("jobs-table", "data"),
        State("jobs-table", "data"),
        prevent_initial_call=True,
    )
    def sync_current_page_selection(select_all_checked, current_page_data, table_data):
        if ctx.triggered_id == "jobs-table":
            return [], False

        current_data = table_data or current_page_data
        if not current_data:
            return [], False

        if not select_all_checked:
            return [], False

        return list(range(len(current_data))), True

    @app.callback(
        Output("selection-alert", "is_open"),
        Output("selection-alert", "children"),
        Output("selection-alert", "color"),
        Output("btn-apply-selection", "disabled"),
        Output("btn-apply-selection", "children"),
        Output("btn-clear-analysis-selection", "disabled"),
        Output("btn-download-selected", "disabled"),
        Output("current-selection-summary", "children"),
        Output("current-selection-detail", "children"),
        Output("analysis-selection-summary", "children"),
        Output("analysis-selection-detail", "children"),
        Input("jobs-table", "selected_rows"),
        Input("analysis-selection-store", "data"),
        State("jobs-table", "data"),
    )
    def update_selection_alert(selected_rows, analysis_selection, current_data):
        current_page_job_ids = _get_current_page_job_ids(selected_rows, current_data)
        current_page_count = len(current_page_job_ids)
        applied_job_ids = _normalize_job_ids(analysis_selection)
        applied_count = len(applied_job_ids)

        if current_page_count == 0 and applied_count == 0:
            return (
                True,
                "Select one or more jobs in the table, then click Apply to render charts and enable export actions.",
                "info",
                True,
                "Apply Selection",
                True,
                True,
                "No jobs currently selected",
                "Selections can change at any time. Charts and export remain unchanged until Apply is clicked.",
                "No jobs applied",
                "Charts, metrics, and exports use only the most recently applied selection.",
            )

        if current_page_count > 0 and applied_count == 0:
            return (
                True,
                f"{current_page_count} job(s) are selected. Click Apply to update charts, metrics, and export actions.",
                "primary",
                False,
                f"Apply {current_page_count} Job(s)",
                True,
                True,
                f"{current_page_count} job(s) currently selected",
                "This selection is pending. Nothing has been rendered from it yet.",
                "No jobs applied",
                "No applied job set yet. Apply the current selection to activate analysis.",
            )

        if current_page_count == 0 and applied_count > 0:
            return (
                True,
                f"{applied_count} job(s) are currently applied. You can export them, clear them, or select a new set and click Apply to replace them.",
                "success",
                True,
                "Apply Selection",
                False,
                False,
                "No jobs currently selected",
                "Pick another set in the table if you want to replace the applied jobs.",
                f"{applied_count} job(s) applied for analysis",
                "Charts, metrics, and exports are currently bound to this applied job set.",
            )

        return (
            True,
            f"{current_page_count} job(s) are currently selected, while {applied_count} job(s) remain applied. Click Apply to replace the active analysis set.",
            "warning",
            current_page_count == 0,
            f"Apply {current_page_count} Job(s)",
            applied_count == 0,
            applied_count == 0,
            f"{current_page_count} job(s) currently selected",
            "Click Apply to replace the active analysis set with the current table selection.",
            f"{applied_count} job(s) applied for analysis",
            "The applied set stays active until you clear it or apply a different selection.",
        )

    @app.callback(
        Output("baseline-details-offcanvas", "is_open"),
        Output("baseline-details-content", "children"),
        Input("btn-view-baseline-details", "n_clicks"),
        State("baseline-job-id-store", "data"),
        State("baseline-details-offcanvas", "is_open"),
        State("full-jobs-data-store", "data"),
        State("viewer-context-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_baseline_offcanvas(
        n_clicks, baseline_job_id, is_open, jobs_data, viewer_context
    ):
        if not n_clicks:
            return is_open, dash.no_update

        if not baseline_job_id:
            return not is_open, html.Div(
                "No baseline selected.", className="text-muted"
            )

        try:
            # Fetch Params
            job_id_int = int(baseline_job_id)
            jobs_df = _get_jobs_df(jobs_data)
            if jobs_df.empty:
                return not is_open, html.Div(
                    "No job metadata loaded.", className="text-muted"
                )

            params = jobs_df[jobs_df["id"] == job_id_int].iloc[0]
            param_items = params.drop("id").items()

            param_table = dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")])),
                    html.Tbody(
                        [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in param_items]
                    ),
                ],
                striped=True,
                bordered=True,
                size="sm",
            )

            # Fetch/Calc Metrics for this specific job
            # We reuse calculate validation logic or just read from H5 if needed.
            # Ideally we have metrics pre-calculated or we calc on fly.
            # Let's calc on fly similar to calculate_metrics_data but for one job.

            metrics_content = html.Div(
                "Metrics not available (requires main calculation)"
            )

            # Try to fetch full data for this job to compute metrics
            try:
                h5_file = _get_h5_file(viewer_context)
                if h5_file:
                    full_df = pd.read_hdf(
                        h5_file, "results", where=f"job_id == {job_id_int}"
                    )
                    if not full_df.empty:
                        # Calculate metrics for all available columns except time and job_id
                        metric_rows = []
                        data_cols = [
                            c for c in full_df.columns if c not in ["time", "job_id"]
                        ]

                        for col in data_cols:
                            series = full_df[col]
                            final_val = metric.get_final_value(series)
                            startup_inv = metric.calculate_startup_inventory(series)

                            # Simple metrics display
                            metric_rows.append(
                                html.Tr(
                                    [
                                        html.Td(f"{col} Final"),
                                        html.Td(f"{final_val:,.3f}"),
                                    ]
                                )
                            )
                            metric_rows.append(
                                html.Tr(
                                    [
                                        html.Td(f"{col} Startup Inv."),
                                        html.Td(f"{startup_inv:,.3f}"),
                                    ]
                                )
                            )

                        metrics_content = dbc.Table(
                            [
                                html.Thead(
                                    html.Tr([html.Th("Metric"), html.Th("Value")])
                                ),
                                html.Tbody(metric_rows),
                            ],
                            striped=True,
                            bordered=True,
                            size="sm",
                        )
            except Exception as e:
                metrics_content = html.Div(
                    f"Error calculating metrics: {str(e)}", className="text-danger"
                )

            content = html.Div(
                [
                    html.H5(f"Job ID: {job_id_int}", className="mb-3"),
                    html.H6("Parameters", className="mt-4 mb-2"),
                    param_table,
                    html.H6("Key Metrics (Approx)", className="mt-4 mb-2"),
                    metrics_content,
                ]
            )

            return not is_open, content

        except Exception as e:
            return not is_open, html.Div(
                f"Error loading details: {str(e)}", className="text-danger"
            )
