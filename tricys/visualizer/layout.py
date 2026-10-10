import json

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": False,
    "scrollZoom": True,
}

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
            color = "#c9d1d9"
            if level == "ERROR":
                color = "#ffb4b4"
            elif level == "WARNING":
                color = "#ffd28a"
            rows.append(
                html.Div(
                    [
                        html.Span(
                            f"[{timestamp}] ",
                            style={"color": "#8b949e", "fontSize": "0.9em"},
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


def _overview_card(title, value_id, detail_id=None):
    children = [
        html.Div(title, className="overview-card-label"),
        html.Div("--", id=value_id, className="overview-card-value"),
    ]
    if detail_id:
        children.append(html.Div("", id=detail_id, className="overview-card-detail"))
    return dbc.Card(dbc.CardBody(children), className="overview-card h-100")


def create_layout(
    variable_options,
    parameter_options,
    table_columns,
    jobs_data,
    config_data,
    log_data,
    initial_context=None,
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
            dcc.Location(id="viewer-location", refresh=False),
            dcc.Store(id="viewer-context-store", data=initial_context),
            dcc.Store(id="viewer-error-store"),
            dcc.Store(id="analysis-selection-store", data=[]),
            dcc.Store(id="analysis-selection-feedback-store", data={"flash": False}),
            dcc.Store(id="metrics-availability-store", data={"has_summary": False}),
            dcc.Store(id="full-jobs-data-store", data=jobs_data),
            dcc.Store(id="main-data-store"),
            dcc.Store(id="metrics-data-store"),
            dcc.Store(id="baseline-job-store"),
            dcc.Store(id="baseline-job-id-store"),
            dcc.Store(id="variable-options-store", data=variable_options),
            dcc.Store(id="parameter-options-store", data=parameter_options),
            dcc.Store(id="config-store", data=config_data),
            dcc.Store(id="log-store", data=log_data),
            dcc.Interval(
                id="analysis-selection-feedback-timer",
                interval=1400,
                n_intervals=0,
                disabled=True,
            ),
            html.Div(
                [
                    html.Div(
                        "HDF5 Viewer Error", className="viewer-fatal-error-eyebrow"
                    ),
                    html.Div(
                        "Unable to load HDF5 viewer",
                        id="viewer-fatal-error-title",
                        className="viewer-fatal-error-title",
                    ),
                    html.Div(
                        id="viewer-fatal-error-message",
                        className="viewer-fatal-error-message",
                    ),
                    html.Div(
                        id="viewer-fatal-error-detail",
                        className="viewer-fatal-error-detail",
                    ),
                ],
                id="viewer-fatal-error",
                className="viewer-fatal-error",
                style={"display": "none"},
            ),
            html.Div(
                [
                    dbc.Alert(
                        id="viewer-status-alert",
                        is_open=False,
                        color="danger",
                        className="mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _overview_card(
                                    "Jobs", "overview-job-count", "overview-job-detail"
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _overview_card(
                                    "Variables",
                                    "overview-variable-count",
                                    "overview-variable-detail",
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _overview_card(
                                    "Time Range",
                                    "overview-time-range",
                                    "overview-time-detail",
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _overview_card(
                                    "Datasets",
                                    "overview-dataset-status",
                                    "overview-dataset-detail",
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _overview_card(
                                    "File Size",
                                    "overview-file-size",
                                    "overview-file-detail",
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                            dbc.Col(
                                _overview_card(
                                    "Last Modified",
                                    "overview-modified-at",
                                    "overview-modified-detail",
                                ),
                                md=6,
                                xl=2,
                                className="mb-3",
                            ),
                        ],
                        className="mb-2",
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
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "2. Select Simulation Jobs"
                                                        ),
                                                        html.I(
                                                            className="bi bi-info-circle-fill ms-2",
                                                            id="filter-help-icon",
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ],
                                                    className="d-flex align-items-center",
                                                ),
                                                html.Div(
                                                    "Select one or more jobs in the table. Charts, metrics, and export update only after you click Apply.",
                                                    className="jobs-section-subtitle",
                                                ),
                                            ],
                                            width=True,
                                        ),
                                    ],
                                    align="center",
                                )
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
                                        color="info",
                                        is_open=False,
                                        dismissable=True,
                                        className="mb-3 selection-alert",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            id="current-selection-summary",
                                                            style={"display": "none"},
                                                        ),
                                                        html.Div(
                                                            id="current-selection-detail",
                                                            style={"display": "none"},
                                                        ),
                                                        html.Div(
                                                            [
                                                                dbc.Checkbox(
                                                                    id="select-all-checkbox",
                                                                    label="Select all on current page",
                                                                    value=False,
                                                                    className="mb-0",
                                                                ),
                                                                dbc.Button(
                                                                    "Apply Selection",
                                                                    id="btn-apply-selection",
                                                                    size="sm",
                                                                    color="primary",
                                                                    disabled=True,
                                                                ),
                                                            ],
                                                            className="d-flex align-items-center gap-3 flex-wrap",
                                                        ),
                                                    ],
                                                    className="selection-stage-panel selection-stage-panel--current d-flex align-items-center px-3 py-1",
                                                    style={
                                                        "minHeight": "40px",
                                                        "width": "fit-content",
                                                    },
                                                ),
                                                lg=6,
                                                className="mb-2 d-flex align-items-center",
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            id="analysis-selection-summary",
                                                            style={"display": "none"},
                                                        ),
                                                        html.Div(
                                                            id="analysis-selection-detail",
                                                            style={"display": "none"},
                                                        ),
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Clear",
                                                                    id="btn-clear-analysis-selection",
                                                                    size="sm",
                                                                    color="secondary",
                                                                    outline=True,
                                                                    disabled=True,
                                                                    className="selection-toolbar-button",
                                                                ),
                                                                dbc.Button(
                                                                    "Batch CSV",
                                                                    id="btn-download-selected",
                                                                    size="sm",
                                                                    color="secondary",
                                                                    outline=True,
                                                                    disabled=True,
                                                                    className="selection-toolbar-button",
                                                                ),
                                                                dbc.Button(
                                                                    "Wide CSV",
                                                                    id="btn-download-all",
                                                                    size="sm",
                                                                    className="selection-toolbar-button selection-toolbar-button--primary",
                                                                ),
                                                            ],
                                                            className="d-flex align-items-center justify-content-end gap-2",
                                                        ),
                                                    ],
                                                    id="analysis-selection-panel",
                                                    className="selection-stage-panel selection-stage-panel--applied d-flex align-items-center px-2 py-1",
                                                    style={
                                                        "minHeight": "40px",
                                                        "width": "fit-content",
                                                        "marginLeft": "auto",
                                                    },
                                                ),
                                                lg=6,
                                                className="mb-2 d-flex justify-content-end align-items-center",
                                            ),
                                        ],
                                        className="jobs-workspace-row align-items-center",
                                    ),
                                    dcc.Download(id="download-selected-csv"),
                                    dcc.Download(id="download-all-csv"),
                                    dash_table.DataTable(
                                        id="jobs-table",
                                        columns=table_columns,
                                        data=jobs_data,
                                        sort_action="custom",
                                        filter_action="custom",
                                        page_action="custom",
                                        page_current=0,
                                        page_size=50,
                                        row_selectable="multi",
                                        style_table={
                                            "overflowX": "auto",
                                        },
                                        style_as_list_view=True,
                                        style_cell={
                                            "textAlign": "center",
                                            "padding": "10px",
                                            "height": "36px",
                                            "lineHeight": "36px",
                                        },
                                        style_header={
                                            "fontWeight": "bold",
                                            "backgroundColor": "#11141a",
                                            "color": "#f0f6fc",
                                            "borderBottom": "2px solid #30363d",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "rgba(255, 255, 255, 0.015)",
                                            },
                                            {
                                                "if": {"state": "active"},
                                                "backgroundColor": "rgba(0, 210, 255, 0.08)",
                                                "border": "1px solid rgba(0, 210, 255, 0.28)",
                                                "color": "#f0f6fc",
                                            },
                                            {
                                                "if": {"state": "selected"},
                                                "backgroundColor": "rgba(0, 210, 255, 0.12)",
                                                "border": "1px solid rgba(0, 210, 255, 0.38)",
                                                "color": "#f0f6fc",
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
                                    html.Div(
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
                                                                    options=[
                                                                        {
                                                                            "label": f"Job {job.get('id')}",
                                                                            "value": job.get(
                                                                                "id"
                                                                            ),
                                                                        }
                                                                        for job in (
                                                                            jobs_data
                                                                            or []
                                                                        )
                                                                        if job.get("id")
                                                                        is not None
                                                                    ],
                                                                    placeholder="Select Baseline Job...",
                                                                    style={
                                                                        "width": "200px",
                                                                        "display": "inline-block",
                                                                        "verticalAlign": "middle",
                                                                    },
                                                                ),
                                                                dbc.Button(
                                                                    "Clear",
                                                                    id="btn-clear-baseline",
                                                                    size="sm",
                                                                    className="ms-2",
                                                                    color="secondary",
                                                                    outline=True,
                                                                    disabled=True,
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
                                                dcc.Graph(
                                                    id="results-graph",
                                                    className="tricys-plotly-graph",
                                                    config=GRAPH_CONFIG,
                                                    responsive=False,
                                                    style={
                                                        "minHeight": "420px",
                                                        "width": "100%",
                                                    },
                                                ),
                                                className="mt-3",
                                            ),
                                            dbc.Alert(
                                                id="metrics-unavailable-alert",
                                                children="Summary dataset missing. Metrics Summary and derived metrics charts are unavailable for this file.",
                                                color="secondary",
                                                is_open=False,
                                                className="mt-4 mb-0 metrics-note-alert",
                                            ),
                                            html.Div(
                                                [
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
                                                        className="mt-4 tricys-metrics-tabs",
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Loading(
                                                                dash_table.DataTable(
                                                                    id="metrics-summary-table",
                                                                    sort_action="native",
                                                                    style_table={
                                                                        "overflowX": "auto"
                                                                    },
                                                                    style_cell={
                                                                        "textAlign": "center",
                                                                        "minWidth": "150px",
                                                                        "padding": "10px",
                                                                        "height": "36px",
                                                                        "lineHeight": "36px",
                                                                    },
                                                                    style_header={
                                                                        "fontWeight": "bold",
                                                                        "backgroundColor": "#11141a",
                                                                        "color": "#f0f6fc",
                                                                        "borderBottom": "2px solid #30363d",
                                                                    },
                                                                    style_data_conditional=[
                                                                        {
                                                                            "if": {
                                                                                "row_index": "odd"
                                                                            },
                                                                            "backgroundColor": "rgba(255, 255, 255, 0.015)",
                                                                        },
                                                                        {
                                                                            "if": {
                                                                                "state": "active"
                                                                            },
                                                                            "backgroundColor": "rgba(0, 210, 255, 0.08)",
                                                                            "border": "1px solid rgba(0, 210, 255, 0.28)",
                                                                            "color": "#f0f6fc",
                                                                        },
                                                                        {
                                                                            "if": {
                                                                                "state": "selected"
                                                                            },
                                                                            "backgroundColor": "rgba(0, 210, 255, 0.12)",
                                                                            "border": "1px solid rgba(0, 210, 255, 0.38)",
                                                                            "color": "#f0f6fc",
                                                                        },
                                                                    ],
                                                                    style_filter={
                                                                        "textAlign": "center"
                                                                    },
                                                                    css=[
                                                                        {
                                                                            "selector": ".dash-filter input",
                                                                            "rule": "text-align: center !important",
                                                                        }
                                                                    ],
                                                                )
                                                            )
                                                        ],
                                                        id="metrics-summary-container",
                                                        className="metrics-tab-panel",
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
                                                                    id="metric-plot-graph",
                                                                    className="tricys-plotly-graph mt-3",
                                                                    config=GRAPH_CONFIG,
                                                                    responsive=False,
                                                                    style={
                                                                        "minHeight": "360px",
                                                                        "width": "100%",
                                                                    },
                                                                )
                                                            ),
                                                        ],
                                                        id="metrics-plot-container",
                                                        className="metrics-tab-panel",
                                                        style={
                                                            "display": "none",
                                                            "marginTop": "20px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Alert(
                                                                "Select X and Y parameters and one Metric (Z) to generate a contour map.",
                                                                color="info",
                                                                className="mb-2 chart-help-alert",
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
                                                                dcc.Graph(
                                                                    id="heatmap-graph",
                                                                    className="tricys-plotly-graph mt-3",
                                                                    config=GRAPH_CONFIG,
                                                                    responsive=False,
                                                                    style={
                                                                        "minHeight": "420px",
                                                                        "width": "100%",
                                                                    },
                                                                )
                                                            ),
                                                        ],
                                                        id="heatmap-container",
                                                        className="metrics-tab-panel",
                                                        style={
                                                            "display": "none",
                                                            "marginTop": "20px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Alert(
                                                                "Visualize high-dimensional trade-offs. Each line is a job.",
                                                                color="info",
                                                                className="mb-2 chart-help-alert",
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
                                                                    id="parcoords-graph",
                                                                    className="tricys-plotly-graph mt-3",
                                                                    config=GRAPH_CONFIG,
                                                                    responsive=False,
                                                                    style={
                                                                        "minHeight": "420px",
                                                                        "width": "100%",
                                                                    },
                                                                )
                                                            ),
                                                        ],
                                                        id="parallel-coords-container",
                                                        className="metrics-tab-panel",
                                                        style={
                                                            "display": "none",
                                                            "marginTop": "20px",
                                                        },
                                                    ),
                                                ],
                                                id="metrics-section",
                                            ),
                                        ],
                                        id="plot-metrics-panel",
                                        className="plot-metrics-panel",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "Please select jobs.",
                                                className="plot-metrics-lock-title",
                                            ),
                                            html.Div(
                                                "Please select and apply jobs from the table above before analyzing charts and metrics.",
                                                className="plot-metrics-lock-detail",
                                            ),
                                        ],
                                        id="plot-metrics-lock-overlay",
                                        className="plot-metrics-lock-overlay plot-metrics-lock-overlay--visible",
                                    ),
                                ],
                                className="plot-metrics-card-body",
                            ),
                        ],
                        className="mb-4",
                    ),
                    # 4. Engineering Report & Multi-Format Delivery Hub
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Span(
                                                    "4. Engineering Report Hub (Multi-Format Export)",
                                                    className="fw-bold",
                                                ),
                                                html.Div(
                                                    "Built-in engineering report template rendering with live preview and multi-format export (Markdown, Word .docx, PDF)",
                                                    className="text-muted small mt-1",
                                                ),
                                            ],
                                            md=8,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Export MD",
                                                    id="btn-export-report-md",
                                                    size="sm",
                                                    color="secondary",
                                                    outline=True,
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Export Word (.docx)",
                                                    id="btn-export-report-docx",
                                                    size="sm",
                                                    color="primary",
                                                    outline=True,
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    "Export PDF (.pdf)",
                                                    id="btn-export-report-pdf",
                                                    size="sm",
                                                    color="danger",
                                                    outline=True,
                                                ),
                                            ],
                                            md=4,
                                            className="text-end d-flex justify-content-end align-items-center",
                                        ),
                                    ],
                                    align="center",
                                )
                            ),
                            dbc.CardBody(
                                [
                                    dcc.Download(id="download-report-md"),
                                    dcc.Download(id="download-report-docx"),
                                    dcc.Download(id="download-report-pdf"),
                                    dcc.Store(id="report-markdown-store"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Task Dedicated Engineering Report:",
                                                        className="fw-bold small mb-1",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Badge(
                                                                        "Dedicated",
                                                                        id="report-task-badge",
                                                                        color="primary",
                                                                        className="me-2 p-2 align-middle",
                                                                    ),
                                                                    html.Span(
                                                                        "Matching engineering task template...",
                                                                        id="report-active-title",
                                                                        className="fw-bold text-light align-middle me-2",
                                                                    ),
                                                                    html.Small(
                                                                        id="report-source-indicator",
                                                                        className="text-info align-middle",
                                                                    ),
                                                                ],
                                                                className="d-flex align-items-center flex-wrap me-auto",
                                                            ),
                                                            dbc.Button(
                                                                html.I(
                                                                    className="bi bi-arrow-clockwise fs-5"
                                                                ),
                                                                id="btn-refresh-report",
                                                                size="sm",
                                                                color="info",
                                                                outline=True,
                                                                title="Refresh Report Preview",
                                                                className="rounded-circle d-flex align-items-center justify-content-center p-0 ms-2",
                                                                style={
                                                                    "width": "34px",
                                                                    "height": "34px",
                                                                    "minWidth": "34px",
                                                                },
                                                            ),
                                                            dbc.Tooltip(
                                                                "Refresh Report Preview",
                                                                target="btn-refresh-report",
                                                                placement="left",
                                                            ),
                                                        ],
                                                        className="p-2 border rounded bg-dark d-flex align-items-center justify-content-between flex-wrap",
                                                        style={"minHeight": "42px"},
                                                    ),
                                                ],
                                                md=12,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            dcc.Markdown(
                                                id="report-markdown-preview",
                                                children="*Loading report preview...*",
                                                dangerously_allow_html=True,
                                                mathjax=True,
                                                className="report-markdown-view p-4 rounded border",
                                                style={
                                                    "backgroundColor": "#0d1117",
                                                    "color": "#e6edf3",
                                                    "minHeight": "260px",
                                                    "maxHeight": "850px",
                                                    "overflowY": "auto",
                                                },
                                            )
                                        )
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
                id="viewer-main-content",
            ),
        ],
        fluid=True,
        className="tricys-dash-theme py-3",
    )
