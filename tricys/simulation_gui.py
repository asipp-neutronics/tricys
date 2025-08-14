import json
import logging
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from tricys.simulation import run_simulation
from tricys.utils.db_utils import (
    create_parameters_table,
    get_parameters_from_db,
    store_parameters_in_db,
    update_sweep_values_in_db,
)
from tricys.utils.file_utils import delete_old_logs
from tricys.utils.om_utils import (
    get_all_parameters_details,
    get_om_session,
    load_modelica_package,
)

logger = logging.getLogger(__name__)


class InteractiveSimulationUI:
    """A GUI for managing simulation parameters and settings, runnable from any directory."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tricys Interactive Simulation Runner")
        self.root.geometry("1100x800")
        self.params_widgets = {}
        self.workspace_path_var = tk.StringVar(value=os.path.abspath(os.getcwd()))

        self.create_settings_vars()
        self.create_widgets()
        package_path = self._get_abs_path(self.package_path_var.get())
        if not os.path.exists(self._get_abs_path(self.package_path_var.get())):
            messagebox.showwarning(
                "Model Not Found",
                f"The specified model package could not be found at:\n{package_path}",
            )
            return
        self.db_path_updated()
        self.setup_logging()
        self.load_parameters()

    def _get_abs_path(self, path: str) -> str:
        """Resolves a path against the workspace directory if it's not absolute."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_path_var.get(), path)

    def create_settings_vars(self):
        """Initializes all Tkinter StringVars for configuration settings with default values."""
        # Path and Model Settings
        self.package_path_var = tk.StringVar(value="example_model/package.mo")
        self.db_path_var = tk.StringVar(value="data/parameters.db")
        self.results_dir_var = tk.StringVar(value="results")
        self.temp_dir_var = tk.StringVar(value="temp")
        self.model_name_var = tk.StringVar(value="example_model.Cycle")

        # Simulation Settings
        self.variable_filter_var = tk.StringVar(value="time|sds\\.I\\[1\\]")
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        self.step_size_var = tk.DoubleVar(value=1.0)
        self.max_workers_var = tk.IntVar(value=4)
        self.keep_temp_files_var = tk.BooleanVar(value=False)
        self.concurrent_var = tk.BooleanVar(value=True)

        # Logging Settings
        self.log_dir_var = tk.StringVar(value="log")
        self.log_level_var = tk.StringVar(value="INFO")
        self.log_count_var = tk.IntVar(value=5)
        self.log_to_console_var = tk.BooleanVar(value=True)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, expand=False)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.create_settings_widgets(top_frame)
        self.create_params_widgets(bottom_frame)

    def create_settings_widgets(self, parent: ttk.Frame):
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, expand=True)

        # Workspace display
        workspace_frame = ttk.Frame(settings_frame)
        workspace_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(workspace_frame, text="Workspace:").pack(side=tk.LEFT, padx=(0, 5))
        workspace_entry = ttk.Entry(
            workspace_frame, textvariable=self.workspace_path_var, justify='center' 
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
        ttk.Label(path_sim_frame, text="Max Workers:").grid(
            row=4, column=2, sticky="w", padx=15, pady=2
        )
        ttk.Entry(path_sim_frame, textvariable=self.max_workers_var, width=15).grid(
            row=4, column=3, sticky="w", padx=5, pady=2
        )
        ttk.Checkbutton(
            path_sim_frame, text="Keep Temp Files", variable=self.keep_temp_files_var
        ).grid(row=5, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(
            path_sim_frame, text="Concurrent Execution", variable=self.concurrent_var
        ).grid(row=5, column=2, sticky="w", padx=15, pady=2)

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

        path_sim_frame.columnconfigure(1, weight=1)
        path_sim_frame.columnconfigure(3, weight=1)
        log_frame.columnconfigure(1, weight=1)

    def select_workspace(self):
        """Opens a dialog to select a new workspace directory."""
        initial_dir = self.workspace_path_var.get()
        new_workspace = filedialog.askdirectory(
            initialdir=initial_dir, title="Select Workspace Directory"
        )
        if new_workspace != initial_dir:
            self.workspace_path_var.set(new_workspace)
            self.db_path_updated()
            self.load_parameters()
            self.setup_logging()

    def create_params_widgets(self, parent: ttk.Frame):
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(params_frame)
        toolbar.pack(fill=tk.X, pady=5)
        ttk.Button(
            toolbar, text="Load Model to DB", command=self.load_model_to_db_thread
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Refresh From DB", command=self.load_parameters).pack(
            side=tk.LEFT, padx=5
        )
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
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        headers = ["Name", "Default Value", "Sweep Value", "Description"]
        for i, header in enumerate(headers):
            ttk.Label(
                self.scrollable_frame, text=header, font=("Helvetica", 10, "bold")
            ).grid(row=0, column=i, padx=10, pady=5, sticky="w")

    def setup_logging(self):
        """Configures the logging module based on settings from the GUI."""
        try:
            log_level_str = self.log_level_var.get().upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            log_to_console = self.log_to_console_var.get()
            log_dir_path = self.log_dir_var.get()
            log_count = self.log_count_var.get()

            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)

            for handler in root_logger.handlers[:]:
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
        except Exception as e:
            messagebox.showerror("Logging Error", f"Failed to configure logger: {e}")

    def db_path_updated(self, event=None):
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

    def load_model_to_db_thread(self):
        threading.Thread(target=self.execute_load_model_to_db, daemon=True).start()

    def execute_load_model_to_db(self):
        logger.info("Starting to load model parameters into the database.")
        omc = None
        try:
            package_path = self._get_abs_path(self.package_path_var.get())
            model_name = self.model_name_var.get()
            if not package_path or not model_name:
                messagebox.showerror(
                    "Error", "Package Path and Model Name must be set."
                )
                return
            omc = get_om_session()
            if not load_modelica_package(omc, package_path):
                raise RuntimeError(f"Failed to load Modelica package: {package_path}")
            params_details = get_all_parameters_details(omc, model_name)
            if not params_details:
                raise RuntimeError("No parameters found in the model.")
            store_parameters_in_db(self.db_path, params_details)
            messagebox.showinfo(
                "Success",
                f"Successfully loaded {len(params_details)} parameters from model '{model_name}' into the database.",
            )
            self.load_parameters()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model parameters: {e}")
            logger.error(f"Failed to load model parameters: {e}", exc_info=True)
        finally:
            if omc:
                omc.sendExpression("quit()")

    def load_parameters(self):
        for widget in self.scrollable_frame.winfo_children():
            if widget.grid_info()["row"] > 0:
                widget.destroy()
        self.params_widgets = {}
        try:
            params = get_parameters_from_db(self._get_abs_path(self.db_path_var.get()))
            for i, param in enumerate(params, start=1):
                ttk.Label(self.scrollable_frame, text=param["name"]).grid(
                    row=i, column=0, padx=10, pady=2, sticky="w"
                )
                ttk.Label(
                    self.scrollable_frame, text=str(param.get("default_value", ""))
                ).grid(row=i, column=1, padx=10, pady=2, sticky="w")
                sweep_var = tk.StringVar(value=str(param.get("sweep_values", "")))
                ttk.Entry(self.scrollable_frame, textvariable=sweep_var, width=30).grid(
                    row=i, column=2, padx=10, pady=2, sticky="w"
                )
                ttk.Label(self.scrollable_frame, text=param["description"]).grid(
                    row=i, column=3, padx=10, pady=2, sticky="w"
                )
                self.params_widgets[param["name"]] = {
                    "default_value": param["default_value"],
                    "sweep_var": sweep_var,
                }
            logger.info(
                f"Loaded {len(params)} parameters into the UI from {self.db_path}."
            )
        except Exception as e:
            logger.error(f"Failed to load parameters from DB: {e}", exc_info=True)
            for widget in self.scrollable_frame.winfo_children():
                if widget.grid_info()["row"] > 0:
                    widget.destroy()
            self.params_widgets = {}
            self.scrollable_frame.update_idletasks()
            messagebox.showerror(
                "DB Error",
                f"Could not load parameters from database \n\nError: {e}",
            )

    def save_sweep_parameters(self):
        params_to_save = {}
        for name, widgets in self.params_widgets.items():
            sweep_value = widgets["sweep_var"].get()
            if sweep_value:
                params_to_save[name] = sweep_value
        self.db_path = self._get_abs_path(self.db_path_var.get())
        if os.path.exists(self.db_path):
            try:
                update_sweep_values_in_db(self._get_abs_path(self.db_path_var.get()), params_to_save)
                messagebox.showinfo(
                    "Success", "Sweep values saved successfully to the database."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save sweep values: {e}")
                logger.error(f"Failed to save sweep values: {e}", exc_info=True)
        else:
            messagebox.showerror("Error", f"Database not found at {self.db_path}.")
            logger.info(
                f"Database not found at {self.db_path}."
            )

    def run_simulation_thread(self):
        threading.Thread(target=self.execute_simulation, daemon=True).start()

    def execute_simulation(self):
        logger.info("Starting simulation from GUI.")
        try:
            sim_config = {
                "model_name": self.model_name_var.get(),
                "variableFilter": self.variable_filter_var.get(),
                "stop_time": self.stop_time_var.get(),
                "step_size": self.step_size_var.get(),
                "max_workers": self.max_workers_var.get(),
                "keep_temp_files": self.keep_temp_files_var.get(),
                "concurrent": self.concurrent_var.get(),
            }
            sim_params = {}
            for name, widgets in self.params_widgets.items():
                value_str = widgets["sweep_var"].get().strip()
                if not value_str:
                    continue
                # Try parsing as a list
                if value_str.startswith('[') and value_str.endswith(']'):
                    try:
                        sim_params[name] = json.loads(value_str)
                        continue
                    except json.JSONDecodeError:
                        messagebox.showerror("Invalid Parameter", f"Invalid list format for parameter '{name}':\n{value_str}")
                        return
                # Try parsing as a range string 'start:stop:step'
                if value_str.count(':') == 2:
                    sim_params[name] = value_str
                    continue
                # Try parsing as a number
                try:
                    sim_params[name] = int(value_str)
                    continue
                except ValueError:
                    try:
                        sim_params[name] = float(value_str)
                        continue
                    except ValueError:
                        pass # Not a number, fall through to the final error
                messagebox.showerror(
                    "Invalid Parameter",
                    f"Invalid format for parameter '{name}':\n{value_str}\n\n"
                    "Must be a number, a list like [1, 2], or a range like '1:10:1'."
                )
                return

            config = {"simulation": sim_config, "simulation_parameters": sim_params}
            package_path = self._get_abs_path(self.package_path_var.get())
            results_dir = self._get_abs_path(self.results_dir_var.get())
            temp_dir = self._get_abs_path(self.temp_dir_var.get())
            run_simulation(
                config=config,
                package_path=package_path,
                results_dir=results_dir,
                temp_dir=temp_dir,
            )
            messagebox.showinfo("Success", "Simulation completed successfully!")
            logger.info("Simulation run finished successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
            logger.error(f"Simulation run failed: {e}", exc_info=True)



def main():
    """Main function to initialize and run the GUI."""
    root = tk.Tk()
    InteractiveSimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()


if __name__ == "__main__":
    main()