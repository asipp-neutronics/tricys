import json

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

filter_help_text = dcc.Markdown(
    """
    **Operators:** `=`, `!=`, `>`, `<`, `>=`, `<=`
    *Example: `> 300`*

    **Contains text:** `contains`
    *Example: `contains run_1`*

    **Is Empty:** `is blank`

    **Multiple Values:** `{val1, val2}`
    *Example: `{10, 20}` finds 10 OR 20.*

    ---
    *Combine column filters with AND logic.*
    """
)


def render_log_content(data):
    """Renders log data as a list of styled HTML components."""
    if not data:
        return "No log data found."
    if isinstance(data, list):
        rows = []
        for entry in data:
            level = (
                entry.get("levelname", "INFO") if isinstance(entry, dict) else "INFO"
            )
            msg = (
                entry.get("message", str(entry))
                if isinstance(entry, dict)
                else str(entry)
            )
            timestamp = entry.get("asctime", "") if isinstance(entry, dict) else ""
            color = "black"
            if level == "ERROR":
                color = "red"
            elif level == "WARNING":
                color = "orange"
            rows.append(
                html.Div(
                    [
                        html.Span(
                            f"[{timestamp}] ",
                            style={"color": "#6c757d", "fontSize": "0.9em"},
                        ),
                        html.Span(
                            f"[{level}] ", style={"color": color, "fontWeight": "bold"}
                        ),
                        html.Span(msg),
                    ],
                    style={
                        "fontFamily": "monospace",
                        "whiteSpace": "pre-wrap",
                        "marginBottom": "2px",
                    },
                )
            )
        return rows
    else:
        return str(data)


def create_layout(
    variable_options, parameter_options, table_columns, jobs_data, config_data, log_data
):
    """
    Constructs the application layout.
    """
    initial_config_content = (
        json.dumps(config_data, indent=2)
        if config_data
        else "No configuration data found."
    )
    initial_log_content = (
        render_log_content(log_data) if log_data else "No log data found."
    )

    return dbc.Container(
        [
            # Stores
            dcc.Store(id="full-jobs-data-store", data=jobs_data),
            dcc.Store(id="main-data-store"),
            dcc.Store(id="metrics-data-store"),
            dcc.Store(id="baseline-job-store"),
            dcc.Store(id="baseline-job-id-store"),
            dcc.Store(id="variable-options-store", data=variable_options),
            dcc.Store(id="parameter-options-store", data=parameter_options),
            dcc.Store(id="config-store", data=config_data),
            dcc.Store(id="log-store", data=log_data),
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        html.H1("TRICYS HDF5 Visualizer", className="my-4"), width=True
                    ),
                    dbc.Col(
                        dcc.Upload(
                            id="upload-data",
                            children=html.I(
                                className="bi bi-folder2-open",
                                style={"fontSize": "2rem", "cursor": "pointer"},
                            ),
                            multiple=False,
                            className="my-4",
                        ),
                        width="auto",
                    ),
                ],
                align="center",
            ),
            # 1. Run Details
            dbc.Card(
                [
                    dbc.CardHeader("1. Run Details (Config & Logs)"),
                    dbc.CardBody(
                        [
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            html.Pre(
                                                id="config-view",
                                                children=initial_config_content,
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "15px",
                                                    "maxHeight": "500px",
                                                    "overflow": "auto",
                                                },
                                            )
                                        ],
                                        title="Configuration",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.Div(
                                                id="log-view",
                                                children=initial_log_content,
                                                style={
                                                    "maxHeight": "500px",
                                                    "overflow": "auto",
                                                },
                                            )
                                        ],
                                        title="Simulation Logs",
                                    ),
                                ],
                                start_collapsed=True,
                            )
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # 2. Jobs Table
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            "2. Select Simulation Jobs ",
                                            html.I(
                                                className="bi bi-info-circle-fill ms-1",
                                                id="filter-help-icon",
                                                style={"cursor": "pointer"},
                                            ),
                                        ],
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id="select-all-checkbox",
                                            label="Select All (Safe)",
                                            value=False,
                                        ),
                                        width="auto",
                                    ),
                                ],
                                justify="between",
                                align="center",
                            )
                        ]
                    ),
                    dbc.Popover(
                        [
                            dbc.PopoverHeader("Filter Expression Syntax"),
                            dbc.PopoverBody(filter_help_text),
                        ],
                        target="filter-help-icon",
                        trigger="legacy",
                        placement="bottom",
                    ),
                    dbc.CardBody(
                        [
                            dbc.Alert(
                                "",
                                id="selection-alert",
                                color="warning",
                                is_open=False,
                                dismissable=True,
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Download All (Wide Format)",
                                            id="btn-download-all",
                                            className="mb-3",
                                            size="sm",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Download Selected (Batch)",
                                            id="btn-download-selected",
                                            className="mb-3",
                                            size="sm",
                                            color="secondary",
                                            outline=True,
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dcc.Download(id="download-selected-csv"),
                            dcc.Download(id="download-all-csv"),
                            dash_table.DataTable(
                                id="jobs-table",
                                columns=table_columns,
                                data=jobs_data,
                                sort_action="custom",
                                filter_action="custom",
                                page_action="none",
                                row_selectable="multi",
                                style_table={
                                    "overflowX": "auto",
                                    "maxHeight": "800px",
                                    "overflowY": "auto",
                                },
                                style_as_list_view=True,
                                style_cell={"textAlign": "center", "padding": "10px"},
                                style_header={
                                    "backgroundColor": "#f8f9fa",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #dee2e6",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "rgba(0, 0, 0, 0.02)",
                                    },
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "rgba(0, 123, 255, 0.2)",
                                        "border": "1px solid #007bff",
                                    },
                                ],
                                style_filter={"textAlign": "center"},
                                css=[
                                    {
                                        "selector": ".dash-filter input",
                                        "rule": "text-align: center !important",
                                    }
                                ],
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            # 3. Plot & Metrics
            dbc.Card(
                [
                    dbc.CardHeader("3. Plot Results & Key Metrics"),
                    dbc.CardBody(
                        [
                            dcc.Dropdown(
                                id="variable-dropdown",
                                options=variable_options,
                                multi=True,
                                placeholder="Select variables to plot...",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="plot-type-radio",
                                            options=[
                                                {
                                                    "label": "Absolute Values",
                                                    "value": "absolute",
                                                },
                                                {
                                                    "label": "Difference from Baseline",
                                                    "value": "difference",
                                                },
                                            ],
                                            value="absolute",
                                            inline=True,
                                            className="mt-3",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Baseline Job:",
                                                    className="me-2 d-inline-block fw-bold",
                                                ),
                                                dcc.Dropdown(
                                                    id="baseline-dropdown",
                                                    options=(
                                                        [
                                                            {
                                                                "label": f"Job {job.get('id')}",
                                                                "value": job.get("id"),
                                                            }
                                                            for job in jobs_data
                                                        ]
                                                        if jobs_data
                                                        else []
                                                    ),
                                                    placeholder="Select Baseline Job...",
                                                    style={
                                                        "width": "200px",
                                                        "display": "inline-block",
                                                        "verticalAlign": "middle",
                                                    },
                                                ),
                                                dbc.Button(
                                                    "View Details",
                                                    id="btn-view-baseline-details",
                                                    size="sm",
                                                    className="ms-2",
                                                    color="info",
                                                    outline=True,
                                                    disabled=True,
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                justify="between",
                                align="center",
                            ),
                            dcc.Loading(
                                dcc.Graph(id="results-graph"), className="mt-3"
                            ),
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Metrics Summary",
                                        tab_id="tab-metrics-summary",
                                    ),
                                    dbc.Tab(
                                        label="Metrics Plots",
                                        tab_id="tab-metrics-plots",
                                    ),
                                    dbc.Tab(
                                        label="Heatmap Analysis",
                                        tab_id="tab-heatmap-analysis",
                                    ),
                                    dbc.Tab(
                                        label="Multidimensional Analysis",
                                        tab_id="tab-parallel-coords",
                                    ),
                                ],
                                id="metrics-tabs",
                                active_tab="tab-metrics-summary",
                                className="mt-4",
                            ),
                            # Persistent Containers for each tab content
                            html.Div(
                                [
                                    dcc.Loading(
                                        dash_table.DataTable(
                                            id="metrics-summary-table",
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "textAlign": "center",
                                                "minWidth": "150px",
                                            },
                                        )
                                    )
                                ],
                                id="metrics-summary-container",
                                style={"marginTop": "20px"},
                            ),
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="xaxis-param-dropdown",
                                                    placeholder="X-Axis (Parameter)",
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="yaxis-metric-dropdown",
                                                    placeholder="Y-Axis (Metric)",
                                                ),
                                                width=6,
                                            ),
                                        ]
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="metric-plot-graph", className="mt-3"
                                        )
                                    ),
                                ],
                                id="metrics-plot-container",
                                style={"display": "none", "marginTop": "20px"},
                            ),
                            html.Div(
                                [
                                    dbc.Alert(
                                        "Select X and Y parameters and one Metric (Z) to generate a contour map.",
                                        color="info",
                                        className="mb-2",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="heatmap-x-dropdown",
                                                    placeholder="X-Axis (Parameter A)",
                                                ),
                                                width=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="heatmap-y-dropdown",
                                                    placeholder="Y-Axis (Parameter B)",
                                                ),
                                                width=4,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="heatmap-z-dropdown",
                                                    placeholder="Z-Axis (Metric)",
                                                ),
                                                width=4,
                                            ),
                                        ]
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id="heatmap-graph", className="mt-3")
                                    ),
                                ],
                                id="heatmap-container",
                                style={"display": "none", "marginTop": "20px"},
                            ),
                            html.Div(
                                [
                                    dbc.Alert(
                                        "Visualize high-dimensional trade-offs. Each line is a job.",
                                        color="info",
                                        className="mb-2",
                                    ),
                                    html.Label(
                                        "Select Metrics/Parameters to visualize:"
                                    ),
                                    dcc.Dropdown(
                                        id="parcoords-dims-dropdown",
                                        multi=True,
                                        placeholder="Select dimensions...",
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="parcoords-graph", className="mt-3"
                                        )
                                    ),
                                ],
                                id="parallel-coords-container",
                                style={"display": "none", "marginTop": "20px"},
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            dbc.Offcanvas(
                html.Div(id="baseline-details-content"),
                id="baseline-details-offcanvas",
                title="Baseline Job Details",
                is_open=False,
                placement="end",
                scrollable=True,
            ),
        ],
        fluid=True,
    )
