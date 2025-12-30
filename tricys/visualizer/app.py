# tricys/visualizer/app.py
import dash
import dash_bootstrap_components as dbc

from tricys.visualizer.callbacks import initialize_data, register_callbacks
from tricys.visualizer.layout import create_layout


def create_app(h5_file_path: str):
    """
    Creates and configures the Dash application.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # Initialize data and state in callbacks module
    # We call initialize_data to seed the globals in callbacks.py and get initial data for layout
    (
        variable_options,
        parameter_options,
        table_columns,
        jobs_data,
        config_data,
        log_data,
    ) = initialize_data(h5_file_path)

    # Set Layout
    app.layout = create_layout(
        variable_options,
        parameter_options,
        table_columns,
        jobs_data,
        config_data,
        log_data,
    )

    # Register Callbacks
    register_callbacks(app)

    return app
