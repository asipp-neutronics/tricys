import base64
import json
import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, ctx, dcc, html

from tricys.analysis import metric
from tricys.visualizer.data import load_h5_data
from tricys.visualizer.filtering import filter_dataframe
from tricys.visualizer.layout import render_log_content

# Global state for callbacks
H5_FILE = ""
JOBS_DF = None


def register_callbacks(app):
    @app.callback(
        Output("full-jobs-data-store", "data"),
        Output("jobs-table", "columns"),
        Output("variable-dropdown", "options"),
        Output("parameter-options-store", "data"),
        Output("config-store", "data"),
        Output("log-store", "data"),
        Output("baseline-dropdown", "options"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def update_output(contents, filename):
        global H5_FILE, JOBS_DF
        if contents is None:
            raise dash.exceptions.PreventUpdate
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        temp_path = os.path.join(os.getcwd(), f"temp_{filename}")
        with open(temp_path, "wb") as f:
            f.write(decoded)

        v_opts, p_opts, t_cols, j_data, c_data, l_data = load_h5_data(temp_path)

        # Update globals
        H5_FILE = temp_path
        if j_data:
            JOBS_DF = pd.DataFrame(j_data)
            # Ensure job_id is column if it was index or renamed to 'id'
            # The load_h5_data returns records with 'id' instead of 'job_id' for the table.
            # but we need consistency. Let's look at load_h5_data implementation.
            # It renames job_id -> id.
            # So JOBS_DF should have 'id'.

        # Generate baseline options
        baseline_options = []
        if JOBS_DF is not None:
            baseline_options = [
                {"label": f"Job {row['id']}", "value": row["id"]}
                for _, row in JOBS_DF.iterrows()
            ]

        return j_data, t_cols, v_opts, p_opts, c_data, l_data, baseline_options

    @app.callback(
        Output("jobs-table", "data"),
        Input("full-jobs-data-store", "data"),
        Input("jobs-table", "sort_by"),
        Input("jobs-table", "filter_query"),
    )
    def update_jobs_table(data, sort_by, filter_query):
        global JOBS_DF
        if not data:
            return []

        # Sync JOBS_DF if it's None (e.g. init from file path)
        if JOBS_DF is None:
            JOBS_DF = pd.DataFrame(data)

        df = pd.DataFrame(data)

        # Filter
        if filter_query:
            try:
                df = filter_dataframe(df, filter_query)
            except Exception as e:
                print(f"Filtering error: {e}")

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

        # No Pagination - return all rows
        return df.to_dict("records")

    @app.callback(
        Output("main-data-store", "data"),
        Input("jobs-table", "selected_rows"),
        Input("variable-dropdown", "value"),
        State("jobs-table", "data"),
    )
    def update_main_data_store(selected_rows, selected_variables, table_data):
        """Update main data based on selected rows (indices) and variables.
        Maps row indices to job IDs using the current table data (which respects sort/filter).
        """
        if not selected_rows or not selected_variables or not H5_FILE or not table_data:
            return None
        # Map row indices to job IDs
        job_ids = []
        for idx in selected_rows:
            try:
                # Use table_data which aligns with the visual table rows
                if idx < len(table_data):
                    job_ids.append(table_data[idx].get("id"))
            except Exception:
                continue
            except Exception:
                continue
        if not job_ids:
            return None
        try:
            # Ensure job IDs are integers for backend query
            job_ids_numeric = [int(jid) for jid in job_ids]
            df = pd.read_hdf(
                H5_FILE,
                "results",
                where=f"job_id in {job_ids_numeric}",
                columns=list(set(["time", "job_id"] + selected_variables)),
            )
            return df.to_dict("records")
        except Exception:
            return None

    @app.callback(
        Output("baseline-job-store", "data"),
        Output("baseline-job-id-store", "data"),
        Output("btn-view-baseline-details", "disabled"),
        Input("baseline-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_baseline_store(selected_job_id):
        if not selected_job_id:
            return None, None, True

        try:
            baseline_df = pd.read_hdf(
                H5_FILE, "results", where=f"job_id == {selected_job_id}"
            )
            return baseline_df.to_dict("records"), selected_job_id, False
        except Exception as e:
            print(f"Error setting baseline: {e}")
            return dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output("results-graph", "figure"),
        Input("main-data-store", "data"),
        Input("baseline-job-store", "data"),
        Input("plot-type-radio", "value"),
        State("variable-dropdown", "value"),
    )
    def update_results_graph(data, baseline_data, plot_type, selected_variables):
        if not data:
            return px.line()
        df_wide = pd.DataFrame(data)
        if not selected_variables:
            return px.line()

        if plot_type == "difference":
            if not baseline_data:
                return px.line()
            baseline_df = pd.DataFrame(baseline_data).set_index("time")
            if not all(v in baseline_df.columns for v in selected_variables):
                return px.line()

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
            margin=dict(l=40, r=20, t=60, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.update_yaxes(matches=None)
        return fig

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
            return {}
        df = pd.DataFrame(metrics_data)

        try:
            fig = px.parallel_coordinates(
                df, dimensions=selected_dims, color=selected_dims[-1]
            )
            fig.update_layout(margin=dict(l=40, r=40, t=60, b=20))
            return fig
        except Exception as e:
            print(f"Parcoords error: {e}")
            return {}

    @app.callback(
        Output("metrics-data-store", "data"),
        Input("main-data-store", "data"),
        State("variable-dropdown", "value"),
    )
    def calculate_metrics_data(data, selected_variables):
        if not data or not selected_variables:
            return None
        df_wide = pd.DataFrame(data)
        metrics_data = []
        for job_id in df_wide["job_id"].unique():
            try:
                params, row_data = JOBS_DF[JOBS_DF["id"] == job_id].iloc[0], {}
                row_data.update(params.to_dict())
                row_data["id"] = job_id
                row_data["Job"] = (
                    f"Job {job_id} ({', '.join([f'{k}={v}' for k, v in params.drop('id').items()])})"
                )
                job_df, time_series = (
                    df_wide[df_wide["job_id"] == job_id],
                    df_wide[df_wide["job_id"] == job_id]["time"],
                )
                for variable in selected_variables:
                    series = job_df[variable].dropna()
                    if series.empty:
                        continue
                    aligned_time = time_series.loc[series.index]
                    # Store as numeric types for plots, they will be formatted in the table if needed or we keep precision
                    # Actually for the metrics summary table, strings are fine, but for plots we need numbers.
                    # The metrics-data-store is used for both. Let's store as numbers. The table will display them as is or we format there.
                    # Or simpler: we store as numbers here, and rely on Dash DataTable formatting (if defined) or default float display.

                    row_data[f"{variable}_Final Value (g)"] = metric.get_final_value(
                        series
                    )
                    row_data[f"{variable}_Startup Inventory (g)"] = (
                        metric.calculate_startup_inventory(series)
                    )
                    try:
                        row_data[f"{variable}_Self-sufficient Time (h)"] = (
                            metric.time_of_turning_point(series, aligned_time)
                        )
                    except:
                        row_data[f"{variable}_Self-sufficient Time (h)"] = None
                    try:
                        row_data[f"{variable}_Doubling Time (days)"] = (
                            metric.calculate_doubling_time(series, aligned_time) / 24
                        )
                    except:
                        row_data[f"{variable}_Doubling Time (days)"] = None
                metrics_data.append(row_data)
            except Exception as e:
                print(f"Error calculating metrics for job {job_id}: {e}")
        return metrics_data

    @app.callback(
        Output("metrics-summary-table", "data"),
        Output("metrics-summary-table", "columns"),
        Output("xaxis-param-dropdown", "options"),
        Output("yaxis-metric-dropdown", "options"),
        Output("heatmap-x-dropdown", "options"),
        Output("heatmap-y-dropdown", "options"),
        Output("heatmap-z-dropdown", "options"),
        Input("metrics-data-store", "data"),
    )
    def update_metrics_ui(metrics_data):
        if not metrics_data:
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
            if k not in JOBS_DF.columns and k != "Job" and k != "id"
        ]

        # Param options
        param_keys = [c for c in JOBS_DF.columns if c != "id"]

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
    )
    def update_metric_plot(xaxis, yaxis, data):
        if not all([xaxis, yaxis, data]):
            return px.scatter()
        df = pd.DataFrame(data).sort_values(by=xaxis)
        df[yaxis] = pd.to_numeric(
            df[yaxis].astype(str).str.replace(",", ""), errors="coerce"
        )

        # Identify grouping parameters (all parameters except job_id and the selected xaxis)
        all_params = [c for c in JOBS_DF.columns if c != "id"]
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
        return fig

    @app.callback(
        Output("heatmap-graph", "figure"),
        Input("heatmap-x-dropdown", "value"),
        Input("heatmap-y-dropdown", "value"),
        Input("heatmap-z-dropdown", "value"),
        State("metrics-data-store", "data"),
    )
    def update_heatmap_plot(x_param, y_param, z_metric, data):
        if not all([x_param, y_param, z_metric, data]):
            return px.scatter()
        import plotly.graph_objects as go

        df = pd.DataFrame(data)
        # Convert Z to numeric
        df[z_metric] = pd.to_numeric(
            df[z_metric].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=[x_param, y_param, z_metric])

        if df.empty:
            return px.scatter()

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
            margin=dict(l=40, r=20, t=60, b=20),
        )
        return fig

    @app.callback(
        Output("jobs-table", "style_data_conditional"),
        Input("metric-plot-graph", "clickData"),
        Input("jobs-table", "selected_rows"),
    )
    def update_table_highlighting(clickData, selected_rows):
        if ctx.triggered_id == "jobs-table" or not clickData:
            return []
        clicked_job_id = clickData["points"][0]["customdata"][0]
        return [
            {
                "if": {"filter_query": f"{{id}} = {clicked_job_id}"},
                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                "border": "1px solid #007bff",
            }
        ]

    @app.callback(
        Output("download-selected-csv", "data"),
        Input("btn-download-selected", "n_clicks"),
        State("jobs-table", "selected_rows"),
        State("jobs-table", "data"),
        prevent_initial_call=True,
    )
    def download_selected_jobs_batch(n_clicks, selected_rows, table_data):
        if not selected_rows:
            return dash.no_update

        # Map indices to job IDs using the currently displayed data
        try:
            job_ids = [
                table_data[idx].get("id")
                for idx in selected_rows
                if idx < len(table_data)
            ]
            if not job_ids:
                return dash.no_update
        except:
            return dash.no_update

        # Ensure numeric IDs
        job_ids_numeric = [int(jid) for jid in job_ids]

        # Load all selected jobs
        # Note: 'where' clause with 'in' is efficient in PyTables/Pandas HDF
        try:
            df = pd.read_hdf(H5_FILE, "results", where=f"job_id in {job_ids_numeric}")
        except Exception as e:
            print(f"Error reading batch jobs: {e}")
            return dash.no_update

        if df.empty:
            return dash.no_update

        # Merge parameter info into the dataframe columns for clarity
        # We'll pivot or just append columns?
        # Appending columns 'paramA', 'paramB' to each row is safer for "Long Format" export
        # Or we can do "Wide Format" similar to download-all.
        # "Batch" usually implies raw data for multiple jobs. Let's keep it Long Format (standard simulation output)
        # but add parameter columns so users can distinguish/group by parameters in their analysis tools.

        params_to_merge = JOBS_DF[JOBS_DF["id"].isin(job_ids_numeric)].copy()
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
        prevent_initial_call=True,
    )
    def download_all_csv(n_clicks, data):
        if not data:
            return dash.no_update
        df_wide, final_df = pd.DataFrame(data), pd.DataFrame(
            {"time": pd.DataFrame(data)["time"].unique()}
        ).sort_values("time")
        for job_id in df_wide["job_id"].unique():
            params = JOBS_DF.loc[JOBS_DF["id"] == job_id].iloc[0]
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
        Output("jobs-table", "selected_rows"),
        Output("selection-alert", "is_open"),
        Output("selection-alert", "children"),
        Input("select-all-checkbox", "value"),
        State("jobs-table", "derived_virtual_row_ids"),
        State("jobs-table", "derived_virtual_data"),
        State("jobs-table", "data"),
        prevent_initial_call=True,
    )
    def update_selection(
        select_all_checked, virtual_row_ids, virtual_data, current_data
    ):
        """Select all rows when the safe checkbox is checked.
        Returns a list of row indices (selected_rows) instead of IDs.
        """
        if not select_all_checked:
            return [], False, ""
        # Determine which rows are currently displayed (after filtering/pagination)
        target_data = current_data or virtual_data
        if not target_data:
            return [], False, ""
        # Use the length of the displayed data to generate indices
        row_count = len(target_data)
        all_indices = list(range(row_count))
        # Safety limit to avoid performance issues
        SAFE_LIMIT = 50
        if row_count > SAFE_LIMIT:
            return (
                all_indices[:SAFE_LIMIT],
                True,
                f"Safety Limit: Only the first {SAFE_LIMIT} jobs were selected to prevent performance issues.",
            )
        return all_indices, False, ""

    @app.callback(
        Output("baseline-details-offcanvas", "is_open"),
        Output("baseline-details-content", "children"),
        Input("btn-view-baseline-details", "n_clicks"),
        State("baseline-job-id-store", "data"),
        State("baseline-details-offcanvas", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_baseline_offcanvas(n_clicks, baseline_job_id, is_open):
        if not n_clicks:
            return is_open, dash.no_update

        if not baseline_job_id:
            return not is_open, html.Div(
                "No baseline selected.", className="text-muted"
            )

        try:
            # Fetch Params
            job_id_int = int(baseline_job_id)
            params = JOBS_DF[JOBS_DF["id"] == job_id_int].iloc[0]
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
                # We need H5_FILE.
                if H5_FILE:
                    full_df = pd.read_hdf(
                        H5_FILE, "results", where=f"job_id == {job_id_int}"
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


def initialize_data(h5_file_path):
    global H5_FILE, JOBS_DF
    if h5_file_path:
        H5_FILE = h5_file_path
        v_opts, p_opts, t_cols, jobs_data, config_data, log_data = load_h5_data(
            h5_file_path
        )
        if jobs_data:
            JOBS_DF = pd.DataFrame(jobs_data)
        return v_opts, p_opts, t_cols, jobs_data, config_data, log_data
    return [], [], [], [], None, None
