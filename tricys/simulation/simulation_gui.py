import json
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from tricys.core.modelica import (
    get_all_parameters_details,
    get_om_session,
    load_modelica_package,
)
from tricys.simulation.simulation import run_simulation
from tricys.utils.log_utils import delete_old_logs
from tricys.utils.sqlite_utils import (
    create_parameters_table,
    get_parameters_from_db,
    store_parameters_in_db,
    update_sweep_values_in_db,
)

logger = logging.getLogger(__name__)


class GUILogHandler(logging.Handler):
    """Custom log handler that sends log messages to a GUI window.

    Attributes:
        log_queue: A queue to which log messages are sent for GUI display.

    Note:
        Prevents logging errors from breaking the application by catching all
        exceptions in emit(). Thread-safe via queue communication.
    """

    def __init__(self, log_queue: queue.Queue) -> None:
        """Initializes the log handler.

        Args:
            log_queue: A queue to which log messages will be sent.
        """
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emits a log record to the queue for GUI display.

        Args:
            record: The log record to be emitted.

        Note:
            Silently catches all exceptions to prevent logging errors from breaking
            the application. Uses queue.put() for thread-safe communication.
        """
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            # Prevent logging errors from breaking the application
            pass


class LogWindow:
    """A separate window for displaying log messages in real-time.

    Attributes:
        parent: The parent tkinter widget.
        log_window: The Toplevel window for logs.
        log_queue: Queue for receiving log messages from the handler.
        log_handler: GUILogHandler instance attached to the root logger.

    Note:
        Updates log display every 100ms via Tkinter's after() mechanism.
        Supports log clearing and copying. Handles window lifecycle safely.
    """

    def __init__(self, parent: tk.Widget) -> None:
        """Initializes the LogWindow.

        Args:
            parent: The parent tkinter widget.
        """
        self.parent = parent
        self.log_window = None
        self.log_queue = queue.Queue()
        self.log_handler = None

    def create_window(self) -> None:
        """Creates and shows the log window."""
        if self.log_window is not None and self.log_window.winfo_exists():
            # Window already exists, just bring it to front
            self.log_window.lift()
            self.log_window.focus()
            return

        self.log_window = tk.Toplevel(self.parent)
        self.log_window.title("Real-time logs")
        self.log_window.geometry("800x600")

        # Create main frame
        main_frame = ttk.Frame(self.log_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(toolbar, text="Clear logs", command=self.clear_logs).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(toolbar, text="Copy all", command=self.copy_all_logs).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Create log text area with scrollbar
        self.log_text = ScrolledText(
            main_frame, wrap=tk.WORD, width=100, height=30, font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for different log levels
        self.log_text.tag_configure("DEBUG", foreground="gray")
        self.log_text.tag_configure("INFO", foreground="black")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("CRITICAL", foreground="red", background="yellow")

        # Start the log handler
        self.start_logging()

        # Handle window close event
        self.log_window.protocol("WM_DELETE_WINDOW", self.on_window_close)

        # Start processing log messages
        self.process_log_queue()

    def start_logging(self) -> None:
        """Starts capturing log messages by adding a custom handler to the root logger."""
        # Create and add the GUI log handler
        self.log_handler = GUILogHandler(self.log_queue)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)

    def stop_logging(self) -> None:
        """Stops capturing log messages by removing the custom handler."""
        if self.log_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
            self.log_handler = None

    def process_log_queue(self) -> None:
        """Processes log messages from the queue and displays them in the text widget."""
        try:
            while True:
                try:
                    message = self.log_queue.get_nowait()
                    self.add_log_message(message)
                except queue.Empty:
                    break
        except tk.TclError:
            # Window has been destroyed
            return

        # Schedule next check
        if self.log_window and self.log_window.winfo_exists():
            self.log_window.after(100, self.process_log_queue)

    def add_log_message(self, message) -> None:
        """Adds a formatted log message to the text widget.

        Args:
            message (str): The log message string to add.
        """
        if not self.log_window or not self.log_window.winfo_exists():
            return

        # Determine log level for coloring
        tag = "INFO"  # default
        if "DEBUG" in message:
            tag = "DEBUG"
        elif "WARNING" in message:
            tag = "WARNING"
        elif "ERROR" in message:
            tag = "ERROR"
        elif "CRITICAL" in message:
            tag = "CRITICAL"

        # Insert message with appropriate tag
        self.log_text.insert(tk.END, message + "\n", tag)

        # Auto-scroll to bottom
        self.log_text.see(tk.END)

        # Limit the number of lines to prevent memory issues
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 1000:  # Keep only last 1000 lines
            self.log_text.delete("1.0", f"{lines-1000}.0")

    def clear_logs(self) -> None:
        """Clears all log messages from the display."""
        if self.log_text:
            self.log_text.delete("1.0", tk.END)
            messagebox.showinfo("Success", "The log has been cleared.")

    def copy_all_logs(self) -> None:
        """Copies all log text to the system clipboard."""
        if self.log_text:
            content = self.log_text.get("1.0", tk.END)
            self.log_window.clipboard_clear()
            self.log_window.clipboard_append(content)
            messagebox.showinfo(
                "Success", "The log content has been copied to the clipboard."
            )

    def on_window_close(self) -> None:
        """Handles the window close event by stopping logging and destroying the window."""
        self.stop_logging()
        if self.log_window:
            self.log_window.destroy()
            self.log_window = None


class InteractiveSimulationUI:
    """A GUI for managing simulation parameters and settings, runnable from any directory."""

    def __init__(self, root: tk.Tk) -> None:
        """Initializes the main application UI.

        Args:
            root (tk.Tk): The root tkinter window.
        """
        self.root = root
        self.original_title = "Tricys Interactive Simulation Runner"
        self.root.title(self.original_title)
        self.root.geometry("1100x800")
        self.params_widgets = {}
        self.workspace_path_var = tk.StringVar(value=os.path.abspath(os.getcwd()))

        # Initialize log window
        self.log_window = LogWindow(self.root)

        self.create_settings_vars()
        self.create_widgets()
        package_path = self._get_abs_path(self.package_path_var.get())
        if not os.path.exists(self._get_abs_path(self.package_path_var.get())):
            messagebox.showwarning(
                "Model Not Found",
                f"The specified model package could not be found at:\n{package_path}",
            )
            return
        # Delay database check and logging setup until after mainloop starts
        self.root.after(100, self._delayed_initialization)

    def _delayed_initialization(self) -> None:
        """Performs initialization that requires the main loop to be running."""
        self.setup_logging()
        self.db_path_updated()
        self.load_parameters()

    def _get_abs_path(self, path: str) -> str:
        """Resolves a path against the workspace directory if it's not absolute.

        Args:
            path (str): The path to resolve.

        Returns:
            str: The absolute path.
        """
        if os.path.isabs(path):
            return path
        return Path(
            os.path.join(Path(self.workspace_path_var.get()).as_posix(), path)
        ).as_posix()

    def _convert_relative_paths_to_absolute(self, config_data) -> Any:
        """Recursively converts relative paths in the configuration to absolute paths.

        Args:
            config_data: The configuration data (dict or list) to traverse.

        Returns:
            Any: The configuration data with paths converted.
        """
        if isinstance(config_data, dict):
            converted_config = {}
            for key, value in config_data.items():
                if key.endswith("_path") and isinstance(value, str):
                    if not os.path.isabs(value):
                        abs_path = self._get_abs_path(value)
                        converted_config[key] = abs_path
                        logger.info(
                            f"Converted relative path '{value}' to absolute path '{abs_path}' for key '{key}'"
                        )
                    else:
                        converted_config[key] = value
                else:
                    converted_config[key] = self._convert_relative_paths_to_absolute(
                        value
                    )
            return converted_config
        elif isinstance(config_data, list):
            return [
                self._convert_relative_paths_to_absolute(item) for item in config_data
            ]
        else:
            return config_data

    def create_settings_vars(self) -> None:
        """Initializes all Tkinter StringVars for configuration settings."""
        # Path and Model Settings
        self.package_path_var = tk.StringVar(value="example_model/package.mo")
        self.db_path_var = tk.StringVar(value="data/parameters.db")
        self.results_dir_var = tk.StringVar(value="results")
        self.temp_dir_var = tk.StringVar(value="temp")
        self.model_name_var = tk.StringVar(value="example_model.Cycle")

        # Simulation Settings
        self.variable_filter_var = tk.StringVar(value=r"time|sds\.I\[1\]")
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        self.step_size_var = tk.DoubleVar(value=1.0)
        self.tolerance_var = tk.StringVar(value="1e-6")
        self.max_workers_var = tk.IntVar(value=4)
        self.keep_temp_files_var = tk.BooleanVar(value=True)
        self.concurrent_var = tk.BooleanVar(value=True)

        # Logging Settings
        self.log_dir_var = tk.StringVar(value="log")
        self.log_level_var = tk.StringVar(value="INFO")
        self.log_count_var = tk.IntVar(value=5)
        self.log_to_console_var = tk.BooleanVar(value=True)

        # Co-simulation Settings
        self.enable_co_simulation_var = tk.BooleanVar(value=False)
        self.co_sim_config_path_var = tk.StringVar(value="")

    def create_widgets(self) -> None:
        """Creates the main frames and widgets for the GUI layout."""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, expand=False)

        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.create_settings_widgets(top_frame)
        self.create_params_widgets(bottom_frame)

    def _set_widget_state(self, parent, state) -> None:
        """Recursively sets the state of all child widgets.

        Args:
            parent: The parent widget.
            state (str): The state to set (e.g., 'disabled', 'normal').
        """
        for widget in parent.winfo_children():
            try:
                # Exclude scrollbars as disabling them can look odd.
                if "scrollbar" not in widget.winfo_class():
                    widget.configure(state=state)
            except tk.TclError:
                # This widget doesn't have a 'state' option (e.g., a Frame).
                pass
            self._set_widget_state(widget, state)

    def _toggle_ui_lock(self, locked: bool) -> None:
        """Disables or enables the entire UI for long-running tasks.

        Args:
            locked (bool): If True, lock the UI; otherwise, unlock it.
        """
        if locked:
            self.root.title(f"Executing... - {self.original_title}")
            self._set_widget_state(self.main_frame, "disabled")
        else:
            self.root.title(self.original_title)
            self._set_widget_state(self.main_frame, "normal")

    def create_settings_widgets(self, parent: ttk.Frame) -> None:
        """Creates and arranges the widgets for the settings sections.

        Args:
            parent (ttk.Frame): The parent frame to contain the settings widgets.
        """
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, expand=True)

        # Workspace display
        workspace_frame = ttk.Frame(settings_frame)
        workspace_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(workspace_frame, text="Workspace:").pack(side=tk.LEFT, padx=(0, 5))
        workspace_entry = ttk.Entry(
            workspace_frame, textvariable=self.workspace_path_var, justify="center"
        )
        workspace_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            workspace_frame, text="Browse...", command=self.select_workspace
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Path and Sim Settings
        path_sim_frame = ttk.Frame(settings_frame)
        path_sim_frame.pack(fill=tk.X)

        ttk.Label(path_sim_frame, text="Package Path:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.package_path_var, width=40).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Database Path:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        db_entry = ttk.Entry(path_sim_frame, textvariable=self.db_path_var, width=40)
        db_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        db_entry.bind("<FocusOut>", self.db_path_updated)
        ttk.Label(path_sim_frame, text="Results Dir:").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.results_dir_var, width=40).grid(
            row=2, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Temp Dir:").grid(
            row=3, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.temp_dir_var, width=40).grid(
            row=3, column=1, sticky="ew", padx=5, pady=2
        )

        ttk.Label(path_sim_frame, text="Model Name:").grid(
            row=0, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.model_name_var, width=40).grid(
            row=0, column=3, sticky="ew", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Variable Filter:").grid(
            row=1, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.variable_filter_var, width=40).grid(
            row=1, column=3, sticky="ew", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Stop Time:").grid(
            row=2, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.stop_time_var, width=15).grid(
            row=2, column=3, sticky="w", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Step Size:").grid(
            row=3, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.step_size_var, width=15).grid(
            row=3, column=3, sticky="w", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Tolerance:").grid(
            row=4, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.tolerance_var, width=15).grid(
            row=4, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(path_sim_frame, text="Max Workers:").grid(
            row=4, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.max_workers_var, width=15).grid(
            row=4, column=3, sticky="w", padx=5, pady=2
        )
        ttk.Checkbutton(
            path_sim_frame, text="Keep Temp Files", variable=self.keep_temp_files_var
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(
            path_sim_frame, text="Concurrent Execution", variable=self.concurrent_var
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=15, pady=2)

        # Logging Settings
        log_frame = ttk.LabelFrame(settings_frame, text="Logging", padding="10")
        log_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(log_frame, text="Log Directory:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(log_frame, textvariable=self.log_dir_var, width=40).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Label(log_frame, text="Log Level:").grid(
            row=0, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Combobox(
            log_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=10,
        ).grid(row=0, column=3, sticky="w", padx=5, pady=2)
        ttk.Label(log_frame, text="Log Count:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(log_frame, textvariable=self.log_count_var, width=10).grid(
            row=1, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Checkbutton(
            log_frame, text="Log to Console", variable=self.log_to_console_var
        ).grid(row=1, column=2, sticky="w", padx=15, pady=2)
        ttk.Button(
            log_frame, text="Apply Logging Settings", command=self.setup_logging
        ).grid(row=1, column=3, sticky="e", padx=5, pady=2)
        ttk.Button(
            log_frame, text="Open the log window", command=self.show_log_window
        ).grid(row=0, column=4, sticky="e", padx=5, pady=2)

        # Co-simulation Settings
        cosim_frame = ttk.LabelFrame(
            settings_frame, text="Co-simulation (Optional)", padding="10"
        )
        cosim_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Checkbutton(
            cosim_frame,
            text="Enable Co-simulation",
            variable=self.enable_co_simulation_var,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(cosim_frame, text="Co-sim Config (JSON):").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Entry(cosim_frame, textvariable=self.co_sim_config_path_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(
            cosim_frame, text="Browse...", command=self.select_co_sim_config
        ).grid(row=1, column=2, padx=(5, 0), pady=2)

        cosim_frame.columnconfigure(1, weight=1)

        path_sim_frame.columnconfigure(1, weight=1)
        path_sim_frame.columnconfigure(3, weight=1)
        log_frame.columnconfigure(1, weight=1)
        log_frame.columnconfigure(4, weight=0)  # Keep button column fixed width

    def select_co_sim_config(self) -> None:
        """Opens a dialog to select a co-simulation configuration file."""
        config_file = filedialog.askopenfilename(
            title="Select Co-simulation Configuration File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.workspace_path_var.get(),
        )
        if config_file:
            # Convert to relative path if within workspace
            workspace = Path(self.workspace_path_var.get())
            try:
                relative_path = Path(config_file).relative_to(workspace)
                self.co_sim_config_path_var.set(str(relative_path))
                messagebox.showinfo(
                    "Success",
                    f"The selected co-simulation configuration file:\n{relative_path}",
                )
            except ValueError:
                # File is outside workspace, use absolute path
                self.co_sim_config_path_var.set(config_file)
                messagebox.showinfo(
                    "成功",
                    f"The selected co-simulation configuration file:\n{config_file}",
                )
        else:
            messagebox.showinfo("Cancel", "No file selected")

    def show_log_window(self) -> None:
        """Displays the log window."""
        self.log_window.create_window()

    def select_workspace(self) -> None:
        """Opens a dialog to select a new workspace directory."""
        initial_dir = self.workspace_path_var.get()
        new_workspace = filedialog.askdirectory(
            initialdir=initial_dir, title="Select Workspace Directory"
        )
        if new_workspace and new_workspace != initial_dir:
            self.workspace_path_var.set(new_workspace)
            self.db_path_updated()
            self.load_parameters()
            self.setup_logging()
            messagebox.showinfo(
                "Success", f"The workspace has been changed to:\n{new_workspace}"
            )
        elif new_workspace:
            messagebox.showinfo("Tip", "The workspace has not changed.")
        else:
            messagebox.showinfo("Cancel", "No directory selected")

    def create_params_widgets(self, parent: ttk.Frame) -> None:
        """Creates the toolbar and scrollable frame for displaying model parameters.

        Args:
            parent (ttk.Frame): The parent frame to contain the parameters widgets.
        """
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(params_frame)
        toolbar.pack(fill=tk.X, pady=5)
        ttk.Button(
            toolbar, text="Load Model to DB", command=self.load_model_to_db_thread
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            toolbar, text="Refresh From DB", command=self.refresh_parameters_from_db
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            toolbar, text="Save Sweep Values to DB", command=self.save_sweep_parameters
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            toolbar, text="Run Simulation", command=self.run_simulation_thread
        ).pack(side=tk.RIGHT, padx=5)
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = event.width
            canvas.itemconfig(canvas.find_all()[0], width=canvas_width)

        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_to_mousewheel)
        canvas.bind("<Leave>", _unbind_from_mousewheel)
        self.scrollable_frame.bind("<Enter>", _bind_to_mousewheel)
        self.scrollable_frame.bind("<Leave>", _unbind_from_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        headers = ["Name", "Default Value", "Sweep Value", "Description"]
        for i, header in enumerate(headers):
            if header == "Sweep Value":
                bg_color = "#d6ebf5"
            else:
                bg_color = "lightgray"

            header_label = ttk.Label(
                self.scrollable_frame,
                text=header,
                font=("Helvetica", 10, "bold"),
                relief="solid",
                borderwidth=1,
                background=bg_color,
                anchor="center",
            )
            header_label.grid(
                row=0, column=i, padx=0, pady=0, sticky="ew", ipadx=5, ipady=3
            )

        for col in range(len(headers)):
            self.scrollable_frame.columnconfigure(col, weight=1)

        min_widths = [
            150,
            100,
            200,
            250,
        ]  # Name, Default Value, Sweep Value, Description
        for col, min_width in enumerate(min_widths):
            self.scrollable_frame.grid_columnconfigure(col, minsize=min_width)

    def setup_logging(self) -> None:
        """Configures the logging module based on settings from the GUI."""
        try:
            log_level_str = self.log_level_var.get().upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            log_to_console = self.log_to_console_var.get()
            log_dir_path = self.log_dir_var.get()
            log_count = self.log_count_var.get()

            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)

            # Remove only non-GUI handlers to preserve log window functionality
            handlers_to_remove = []
            for handler in root_logger.handlers[:]:
                if not isinstance(handler, GUILogHandler):
                    handlers_to_remove.append(handler)

            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)
                handler.close()

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            if log_to_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            if log_dir_path:
                abs_log_dir = self._get_abs_path(log_dir_path)
                os.makedirs(abs_log_dir, exist_ok=True)
                delete_old_logs(abs_log_dir, log_count)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(
                    abs_log_dir, f"gui_simulation_{timestamp}.log"
                )
                file_handler = logging.FileHandler(
                    log_file_path, mode="a", encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                logger.info(f"Logging to file: {log_file_path}")

            logger.info("Logging settings applied.")

            # Prepare success message with log file path info
            message = "Log settings have been applied"
            if log_dir_path:
                log_file_path = Path(self._get_abs_path(log_file_path)).as_posix()
                message += f",File save location：\n\n{log_file_path}"

            messagebox.showinfo("Success", message)
        except Exception as e:
            messagebox.showerror("Logging Error", f"Failed to configure logger: {e}")

    def db_path_updated(self, event=None) -> None:
        """Handles the event when the database path is updated.

        Checks for model and database existence and loads parameters accordingly.

        Args:
            event: The event object (optional).
        """
        package_path = self._get_abs_path(self.package_path_var.get())
        if not os.path.exists(package_path):
            self.load_parameters()
            messagebox.showwarning(
                "Model Not Found",
                f"The specified model package could not be found at:\n{package_path}",
            )
            return

        self.db_path = self._get_abs_path(self.db_path_var.get())

        if os.path.exists(self.db_path):
            logger.info(f"Database exists at {self.db_path}, loading parameters.")
            self.load_parameters()
        else:
            logger.info(
                f"Database not found at {self.db_path}, creating and loading from model."
            )
            create_parameters_table(self.db_path)
            self.load_model_to_db_thread()

    def load_model_to_db_thread(self) -> None:
        """Loads model parameters to the database in a separate thread."""
        # Get values from UI in main thread before starting background thread
        package_path = self._get_abs_path(self.package_path_var.get())
        model_name = self.model_name_var.get()
        db_path = self._get_abs_path(self.db_path_var.get())

        # Lock UI and show status in title
        self._toggle_ui_lock(True)
        self.root.title(f"Loading model parameters... - {self.original_title}")

        # Pass the values to the background thread
        threading.Thread(
            target=self.execute_load_model_to_db,
            args=(package_path, model_name, db_path),
            daemon=True,
        ).start()

    def execute_load_model_to_db(self, package_path, model_name, db_path) -> None:
        """Executes model loading in a background thread.

        Args:
            package_path (str): Path to the Modelica package.
            model_name (str): The name of the model.
            db_path (str): Path to the SQLite database.
        """
        logger.info("=" * 50)
        logger.info("Starting to load model parameters into the database.")
        logger.info("=" * 50)
        omc = None
        try:
            if not package_path or not model_name:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", "Package Path and Model Name must be set."
                    ),
                )
                return
            omc = get_om_session()
            # Ensure path is in POSIX format for OMPython
            if not load_modelica_package(omc, Path(package_path).as_posix()):
                raise RuntimeError(f"Failed to load Modelica package: {package_path}")
            params_details = get_all_parameters_details(omc, model_name)
            if not params_details:
                raise RuntimeError("No parameters found in the model.")
            store_parameters_in_db(db_path, params_details)
            logger.info("=" * 30)
            logger.info(
                f"Successfully loaded {len(params_details)} parameters from model '{model_name}' into the database."
            )
            logger.info("=" * 30)

            # Update parameters in UI
            self.root.after(0, self.load_parameters)

            # Show success message after everything is done
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Successfully loaded {len(params_details)} parameters into the database",
                ),
            )
        except Exception as e:
            logger.error("=" * 30)
            logger.error(f"Failed to load model parameters: {e}", exc_info=True)
            logger.error("=" * 30)
            self.root.after(
                0,
                lambda e=e: messagebox.showerror(
                    "Error", f"Failed to load model parameters：{e}"
                ),
            )
        finally:
            if omc:
                omc.sendExpression("quit()")
            # Always unlock UI from main thread
            self.root.after(0, self._toggle_ui_lock, False)

    def refresh_parameters_from_db(self) -> None:
        """Refreshes parameters from the database when the user clicks the button."""
        try:
            self.load_parameters()
            # Only show message if this is called directly by button click
            params = get_parameters_from_db(self._get_abs_path(self.db_path_var.get()))
            if params:
                messagebox.showinfo(
                    "Success", f"Refreshed {len(params)} parameters from the database"
                )
            else:
                messagebox.showwarning(
                    "Warning",
                    "No parameters were found in the database, please load the model into the database first.",
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh parameters:{e}")

    def load_parameters(self) -> None:
        """Loads parameters from the database and displays them in the UI."""
        for widget in self.scrollable_frame.winfo_children():
            if widget.grid_info()["row"] > 0:
                widget.destroy()
        self.params_widgets = {}
        try:
            db_path = self._get_abs_path(self.db_path_var.get())
            params = get_parameters_from_db(db_path)
            for i, param in enumerate(params, start=1):
                name_label = tk.Label(
                    self.scrollable_frame,
                    text=param["name"],
                    relief="solid",
                    borderwidth=1,
                    background="white",
                    anchor="w",
                )
                name_label.grid(
                    row=i, column=0, padx=0, pady=0, sticky="ew", ipadx=5, ipady=2
                )

                default_label = tk.Label(
                    self.scrollable_frame,
                    text=str(param.get("default_value", "")),
                    relief="solid",
                    borderwidth=1,
                    background="white",
                    anchor="w",
                )
                default_label.grid(
                    row=i, column=1, padx=0, pady=0, sticky="ew", ipadx=5, ipady=2
                )

                sweep_var = tk.StringVar(value=str(param.get("sweep_values", "")))
                sweep_entry = tk.Entry(
                    self.scrollable_frame,
                    textvariable=sweep_var,
                    relief="solid",
                    borderwidth=1,
                    background="#f0f8ff",
                    insertbackground="black",
                    selectbackground="#b3d9ff",
                )
                sweep_entry.grid(
                    row=i, column=2, padx=0, pady=0, sticky="ew", ipadx=5, ipady=2
                )

                desc_label = tk.Label(
                    self.scrollable_frame,
                    text=param["description"],
                    relief="solid",
                    borderwidth=1,
                    background="white",
                    anchor="w",
                )
                desc_label.grid(
                    row=i, column=3, padx=0, pady=0, sticky="ew", ipadx=5, ipady=2
                )

                self.params_widgets[param["name"]] = {
                    "default_value": param["default_value"],
                    "sweep_var": sweep_var,
                }

            for col in range(4):
                self.scrollable_frame.columnconfigure(col, weight=1)

            self.scrollable_frame.update_idletasks()

            logger.info(f"Loaded {len(params)} parameters into the UI from {db_path}.")

            # Show success message if parameters were loaded
            if params:
                logger.info(
                    f"Successfully loaded {len(params)} parameters from the database"
                )
            else:
                logger.warning(
                    "No parameters were found in the database. Please load the model into the database first."
                )
        except Exception as e:
            logger.error(f"Failed to load parameters from DB: {e}", exc_info=True)
            for widget in self.scrollable_frame.winfo_children():
                if widget.grid_info()["row"] > 0:
                    widget.destroy()
            self.params_widgets = {}
            self.scrollable_frame.update_idletasks()
            logger.error(f"Could not load parameters from database: {e}")

    def save_sweep_parameters(self) -> None:
        """Collects and saves the sweep values from the UI to the database."""
        params_to_save = {}
        for name, widgets in self.params_widgets.items():
            sweep_value = widgets["sweep_var"].get()
            if sweep_value:
                params_to_save[name] = sweep_value
        self.db_path = self._get_abs_path(self.db_path_var.get())
        if os.path.exists(self.db_path):
            try:
                update_sweep_values_in_db(
                    self._get_abs_path(self.db_path_var.get()), params_to_save
                )
                messagebox.showinfo(
                    "Success", "Sweep values saved successfully to the database."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save sweep values: {e}")
                logger.error(f"Failed to save sweep values: {e}", exc_info=True)
        else:
            messagebox.showerror("Error", f"Database not found at {self.db_path}.")
            logger.info(f"Database not found at {self.db_path}.")

    def run_simulation_thread(self) -> None:
        """Starts the simulation in a separate thread with UI feedback."""
        # Show log window when starting simulation
        self.show_log_window()

        # Get all values from UI in main thread before starting background thread
        try:
            paths_config = {
                "package_path": self._get_abs_path(self.package_path_var.get()),
                "results_dir": self._get_abs_path(self.results_dir_var.get()),
                "temp_dir": self._get_abs_path(self.temp_dir_var.get()),
                "log_dir": self._get_abs_path(self.log_dir_var.get()),
            }

            sim_config = {
                "model_name": self.model_name_var.get(),
                "variableFilter": self.variable_filter_var.get(),
                "stop_time": self.stop_time_var.get(),
                "step_size": self.step_size_var.get(),
                "tolerance": self.tolerance_var.get(),
                "max_workers": self.max_workers_var.get(),
                "keep_temp_files": self.keep_temp_files_var.get(),
                "concurrent": self.concurrent_var.get(),
            }

            # Hybrid parameter parsing
            sim_params = {}
            for name, widgets in self.params_widgets.items():
                value_str = widgets["sweep_var"].get().strip()
                if not value_str:
                    continue

                # Remove surrounding quotes if present
                if (value_str.startswith('"') and value_str.endswith('"')) or (
                    value_str.startswith("'") and value_str.endswith("'")
                ):
                    value_str = value_str[1:-1]

                # If it's a special format for sim_utils.py, pass it as a raw string.
                if (":" in value_str) or (
                    value_str.startswith("{") and value_str.endswith("}")
                ):
                    sim_params[name] = value_str
                else:
                    # Otherwise, try to parse as a JSON literal (number, list, bool, etc.)
                    try:
                        sim_params[name] = json.loads(value_str)
                    except json.JSONDecodeError:
                        # If JSON fails, show error message and return
                        messagebox.showerror(
                            "Incorrect parameter format",
                            f"The value '{value_str}' for parameter '{name}' is not in a valid format.\n\nPlease check if the parameter settings are correct.",
                        )
                        return

            # Show parameter check dialog before proceeding
            param_count = len(sim_params)
            if param_count > 0:
                param_list = "\n".join(
                    [f"• {name}: {value}" for name, value in sim_params.items()]
                )
                check_message = (
                    f"The simulation is about to run, please confirm the parameter settings:\n\n"
                    f"Total {param_count} parameters:\n{param_list}\n\n"
                    f"Continue the simulation?"
                )
            else:
                check_message = "The simulation is about to run, and no scan parameters have been set.\n\nDo you want to continue running the simulation?"

            # Ask user confirmation
            result = messagebox.askyesno("Parameter confirmation", check_message)
            if not result:
                return  # User cancelled

            # Co-simulation configuration
            co_sim_config = None
            if self.enable_co_simulation_var.get():
                co_sim_config_path = self.co_sim_config_path_var.get().strip()
                if co_sim_config_path:
                    try:
                        abs_co_sim_path = self._get_abs_path(co_sim_config_path)
                        with open(abs_co_sim_path, "r") as f:
                            co_sim_config = json.load(f)
                        co_sim_config = self._convert_relative_paths_to_absolute(
                            co_sim_config
                        )
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        messagebox.showerror(
                            "Co-simulation Config Error",
                            f"Failed to load co-simulation configuration:\n{e}",
                        )
                        return
                else:
                    messagebox.showwarning(
                        "Co-simulation Warning",
                        "Co-simulation is enabled but no configuration file is specified.",
                    )
                    return

            # Pass the values to the background thread
            threading.Thread(
                target=self.execute_simulation,
                args=(paths_config, sim_config, sim_params, co_sim_config),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare simulation: {e}")

    def execute_simulation(
        self, paths_config, sim_config, sim_params, co_sim_config
    ) -> None:
        """Executes the simulation in a background thread.

        Args:
            paths_config (dict): Configuration for paths.
            sim_config (dict): Configuration for the simulation.
            sim_params (dict): Simulation parameters for the sweep.
            co_sim_config (dict): Configuration for co-simulation, if enabled.
        """
        # Lock the UI from the main thread
        self.root.after(0, self._toggle_ui_lock, True)
        self.root.after(
            0, lambda: self.root.title(f"Running Simulation... - {self.original_title}")
        )
        try:
            logger.info("=" * 50)
            logger.info("Starting simulation from GUI.")
            logger.info("=" * 50)
            # Generate run timestamp
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build complete config structure
            config = {
                "run_timestamp": run_timestamp,
                "paths": paths_config,
                "simulation": sim_config,
                "simulation_parameters": sim_params,
            }

            # Add co-simulation configuration if provided
            if co_sim_config:
                config["co_simulation"] = co_sim_config
                logger.info("Co-simulation configuration loaded")

            # Call run_simulation with the new structure
            run_simulation(config)

            # Show success message with directory paths
            results_dir = paths_config["results_dir"]
            temp_dir = paths_config["temp_dir"]
            success_message = (
                "The simulation run was successfully completed!\n\n"
                f"Output directory:\n{results_dir}\n\n"
                f"Temporary file directory:\n{temp_dir}"
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo("Success", success_message),
            )
            logger.info("=" * 50)
            logger.info("Simulation run finished successfully.")
            logger.info("=" * 50)
        except Exception as e:
            self.root.after(
                0, lambda e=e: messagebox.showerror("Error", f"Simulation failed: {e}")
            )
            logger.error("=" * 50)
            logger.error(f"Simulation run failed: {e}", exc_info=True)
            logger.error("=" * 50)
        finally:
            # No matter what, unlock the UI from the main thread
            self.root.after(0, self._toggle_ui_lock, False)


def main() -> None:
    """Main function to initialize and run the GUI."""
    root = tk.Tk()
    InteractiveSimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()


if __name__ == "__main__":
    main()
