import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from SALib.analyze import fast, sobol
from SALib.analyze import morris as morris_analyze
from SALib.sample import fast_sampler, latin, morris, saltelli

# Configure Chinese fonts in matplotlib
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class TricysSALibAnalyzer:
    """Integrated SALib's Tricys Sensitivity Analyzer.

    Supported Analysis Methods:
    - Sobol: Variance-based global sensitivity analysis
    - Morris: Screening-based sensitivity analysis
    - FAST: Fourier Amplitude Sensitivity Test
    - LHS: Latin Hypercube Sampling uncertainty analysis

    Attributes:
        base_config: Copy of the Tricys base configuration.
        problem: SALib problem definition dictionary.
        parameter_samples: Generated parameter samples array.
        simulation_results: Results from simulations.
        sensitivity_results: Dictionary storing sensitivity analysis results by method.

    Note:
        Automatically sets up Chinese font support and validates Tricys configuration
        on initialization. Supports multiple sensitivity analysis methods with appropriate
        sampling strategies.
    """

    def __init__(self, base_config: Dict[str, Any]) -> None:
        """Initialize the analyzer.

        Args:
            base_config: Tricys base configuration dictionary.

        Note:
            Creates a deep copy of base_config. Initializes problem, samples, and results
            to None. Calls _setup_chinese_font() and _validate_tricys_config() automatically.
        """
        self.base_config = base_config.copy()
        self.problem = None
        self.parameter_samples = None
        self.simulation_results = None
        self.sensitivity_results = {}

        self._setup_chinese_font()
        self._validate_tricys_config()

    def _setup_chinese_font(self) -> None:
        """Set the Chinese font to ensure proper display of Chinese characters in charts.

        Note:
            Tries multiple Chinese fonts in order of preference. Falls back to default
            if no Chinese font found. Also sets axes.unicode_minus to False for proper
            minus sign display. Logs warnings if font setup fails.
        """
        try:
            import matplotlib.font_manager as fm

            chinese_fonts = [
                "SimHei",  # 黑体
                "Microsoft YaHei",  # 微软雅黑
                "KaiTi",  # 楷体
                "FangSong",  # 仿宋
                "STSong",  # 华文宋体
                "STKaiti",  # 华文楷体
                "STHeiti",  # 华文黑体
                "DejaVu Sans",  # 备用字体
                "Arial Unicode MS",  # 备用字体
            ]

            available_font = None
            system_fonts = [f.name for f in fm.fontManager.ttflist]

            for font in chinese_fonts:
                if font in system_fonts:
                    available_font = font
                    break

            if available_font:
                plt.rcParams["font.sans-serif"] = [available_font] + plt.rcParams[
                    "font.sans-serif"
                ]
                logger.info("Using Chinese font", extra={"font": available_font})
            else:
                logger.warning(
                    "No suitable Chinese font found, which may affect Chinese display"
                )

            plt.rcParams["axes.unicode_minus"] = False

        except Exception as e:
            logger.warning(
                "Failed to set Chinese font, using default font",
                extra={"error": str(e)},
            )

    def _handle_nan_values(
        self, Y: np.ndarray, method_name: str = "Sensitivity analysis"
    ) -> np.ndarray:
        """
        Handling NaN values with maximum interpolation

        Args:
            Y: Output array that may contain NaN values
            method_name: Analysis method name for logging

        Returns:
            Processed output array
        """
        nan_indices = np.isnan(Y)
        if np.any(nan_indices):
            n_nan = np.sum(nan_indices)
            logger.info(
                "Found NaN values, using maximum value for imputation",
                extra={
                    "method_name": method_name,
                    "nan_count": n_nan,
                },
            )

            valid_values = Y[~nan_indices]

            if len(valid_values) > 0:
                max_value = np.max(valid_values)
                Y_processed = Y.copy()
                Y_processed[nan_indices] = max_value
                return Y_processed
            else:
                logger.error(
                    "All values are NaN, analysis cannot be performed",
                    extra={
                        "method_name": method_name,
                    },
                )
                raise ValueError(
                    f"{method_name}: All simulation results are NaN, sensitivity analysis cannot be performed"
                )
        return Y

    def _validate_tricys_config(self) -> None:
        """Validate the Tricys configuration for required sections and keys."""
        required_keys = {
            "paths": ["package_path"],
            "simulation": ["model_name", "stop_time"],
        }

        for section, keys in required_keys.items():
            if section not in self.base_config:
                logger.warning(
                    "Missing configuration section, default values will be used",
                    extra={
                        "section": section,
                    },
                )
                continue

            for key in keys:
                if key not in self.base_config[section]:
                    logger.warning(
                        "Missing configuration item, using default value",
                        extra={
                            "section": section,
                            "key": key,
                        },
                    )

        package_path = self.base_config.get("paths", {}).get("package_path")
        if package_path and not os.path.exists(package_path):
            logger.warning(
                "Model file does not exist, which may cause simulation failure",
                extra={
                    "package_path": package_path,
                },
            )

    def _find_unit_config(self, var_name: str, unit_map: dict) -> dict | None:
        """
        Finds the unit configuration for a variable name from the unit_map.
        1. Checks for an exact match.
        2. Checks if the last part of a dot-separated name matches.
        3. Checks for a simple substring containment as a fallback, matching longest keys first.
        """
        if not unit_map or not var_name:
            return None
        if var_name in unit_map:
            return unit_map[var_name]
        components = var_name.split(".")
        if len(components) > 1 and components[-1] in unit_map:
            return unit_map[components[-1]]
        # Fallback to substring match, longest key first
        for key in sorted(unit_map.keys(), key=len, reverse=True):
            if key in var_name:
                return unit_map[key]
        return None

    def define_problem(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        param_distributions: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Define SALib problem space.

        Args:
            param_bounds: Parameter bounds dictionary {'param_name': (min_val, max_val)}.
            param_distributions: Parameter distribution type dictionary {'param_name': 'unif'/'norm'/etc}.
                Valid distribution types: 'unif', 'triang', 'norm', 'truncnorm', 'lognorm'.

        Returns:
            SALib problem definition dictionary.

        Note:
            Defaults to 'unif' distribution if not specified. Validates distribution types
            and warns if invalid. Logs parameter definitions including bounds and distributions.
        """
        if param_distributions is None:
            param_distributions = {name: "unif" for name in param_bounds.keys()}

        valid_dists = ["unif", "triang", "norm", "truncnorm", "lognorm"]
        for name, dist in param_distributions.items():
            if dist not in valid_dists:
                logger.warning(
                    "Invalid distribution type, using 'unif' instead",
                    extra={
                        "parameter_name": name,
                        "invalid_distribution": dist,
                    },
                )
                param_distributions[name] = "unif"

        self.problem = {
            "num_vars": len(param_bounds),
            "names": list(param_bounds.keys()),
            "bounds": list(param_bounds.values()),
            "dists": [
                param_distributions.get(name, "unif") for name in param_bounds.keys()
            ],
        }

        logger.info(
            "Defined a problem space",
            extra={
                "num_parameters": self.problem["num_vars"],
            },
        )
        for i, name in enumerate(self.problem["names"]):
            logger.info(
                "Parameter definition",
                extra={
                    "parameter_name": name,
                    "bounds": self.problem["bounds"][i],
                    "distribution": self.problem["dists"][i],
                },
            )

        return self.problem

    def generate_samples(
        self, method: str = "sobol", N: int = 1024, **kwargs
    ) -> np.ndarray:
        """Generate parameter samples.

        Args:
            method: Sampling method ('sobol', 'morris', 'fast', 'latin').
            N: Number of samples (for Sobol this is the base sample count, actual count is N*(2*D+2)).
            **kwargs: Method-specific parameters.

        Returns:
            Parameter sample array (n_samples, n_params).

        Raises:
            ValueError: If problem not defined or unsupported method.

        Note:
            Sobol generates N*(2*D+2) samples. Morris generates N trajectories. Samples are
            rounded to 5 decimal places. Stores last sampling method for compatibility checking.
        """
        if self.problem is None:
            raise ValueError(
                "You must first call define_problem() to define the problem space."
            )

        logger.info(
            "Generating samples",
            extra={
                "method": method,
                "base_sample_count": N,
            },
        )

        if method.lower() == "sobol":
            # Sobol method: generate N*(2*D+2) samples
            self.parameter_samples = saltelli.sample(self.problem, N, **kwargs)
            actual_samples = N * (2 * self.problem["num_vars"] + 2)

        elif method.lower() == "morris":
            # Morris method: Generate N trajectories
            # Note: Different versions of SALib may have different parameter names
            morris_kwargs = {"num_levels": 4}
            # Check the SALib version and use the correct parameter names
            try:
                morris_kwargs.update(kwargs)
                self.parameter_samples = morris.sample(self.problem, N, **morris_kwargs)
            except TypeError as e:
                if "grid_jump" in str(e):
                    morris_kwargs = {
                        k: v for k, v in morris_kwargs.items() if k != "grid_jump"
                    }
                    morris_kwargs.update(
                        {k: v for k, v in kwargs.items() if k != "grid_jump"}
                    )
                    self.parameter_samples = morris.sample(
                        self.problem, N, **morris_kwargs
                    )
                else:
                    raise e

            actual_samples = len(self.parameter_samples)

        elif method.lower() == "fast":
            # FAST method
            fast_kwargs = {"M": 4}
            fast_kwargs.update(kwargs)
            self.parameter_samples = fast_sampler.sample(self.problem, N, **fast_kwargs)
            actual_samples = len(self.parameter_samples)

        elif method.lower() == "latin":
            # Latin Hypercube Sampling
            self.parameter_samples = latin.sample(self.problem, N, **kwargs)
            actual_samples = N

        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        logger.info(
            "Successfully generated samples", extra={"actual_samples": actual_samples}
        )

        if self.parameter_samples is not None:
            self.parameter_samples = np.round(self.parameter_samples, decimals=5)
            logger.info("Parameter sample precision adjusted to 5 decimal places")

        self._last_sampling_method = method.lower()

        return self.parameter_samples

    def run_tricys_simulations(self, output_metrics: List[str] = None) -> str:
        """
        Generate sampling parameters and output them as a CSV file, which can be subsequently read by the Tricys simulation engine.

        Args:
            output_metrics: List of output metrics to be extracted (for recording but does not affect CSV generation)
            max_workers: Number of concurrent worker processes (reserved for compatibility, currently unused)

        Returns:
            Path to the generated CSV file
        """
        if self.parameter_samples is None:
            raise ValueError(
                "You must first call generate_samples() to generate samples."
            )

        if output_metrics is None:
            output_metrics = [
                "Startup_Inventory",
                "Self_Sufficiency_Time",
                "Doubling_Time",
            ]

        logger.info("Target output metrics", extra={"output_metrics": output_metrics})

        sampled_param_names = self.problem["names"]

        base_params = self.base_config.get("simulation_parameters", {}).copy()
        csv_output_path = (
            Path(self.base_config.get("paths", {}).get("temp_dir"))
            / "salib_sampling.csv"
        )

        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

        param_data = []
        for i, sample in enumerate(self.parameter_samples):
            sampled_params = {
                sampled_param_names[j]: sample[j]
                for j in range(len(sampled_param_names))
            }

            job_params = base_params.copy()
            job_params.update(sampled_params)

            param_data.append(job_params)

        df = pd.DataFrame(param_data)

        for col in df.columns:
            if df[col].dtype in ["float64", "float32"]:
                df[col] = df[col].round(5)

        df.to_csv(csv_output_path, index=False, encoding="utf-8")

        logger.info(
            "Successfully generated parameter samples",
            extra={"num_samples": len(param_data)},
        )
        logger.info("Parameter file saved", extra={"file_path": csv_output_path})
        logger.info("Parameter file columns", extra={"columns": list(df.columns)})
        logger.info("Parameter precision set to 5 decimal places")
        logger.info("Sample statistics", extra={"statistics": df.describe().to_dict()})

        self.sampling_csv_path = csv_output_path

        return csv_output_path

    def generate_tricys_config(
        self, csv_file_path: str = None, output_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Tricys configuration file for reading CSV parameter files and executing simulations
        This function reuses the base configuration and specifically modifies simulation_parameters and analysis_case for file-based SALib runs

        Args:
            csv_file_path: Path to the CSV parameter file. If None, the last generated file is used
            output_metrics: List of output metrics to be calculated

        Returns:
            Path of the generated configuration file
        """
        if csv_file_path is None:
            if hasattr(self, "sampling_csv_path"):
                csv_file_path = self.sampling_csv_path
            else:
                raise ValueError(
                    "CSV file path not found, please first call run_tricys_simulations() or specify csv_file_path"
                )

        if output_metrics is None:
            output_metrics = [
                "Startup_Inventory",
                "Self_Sufficiency_Time",
                "Doubling_Time",
            ]

        csv_abs_path = os.path.abspath(csv_file_path)

        import copy

        tricys_config = copy.deepcopy(self.base_config)
        tricys_config["simulation_parameters"] = {"file": csv_abs_path}

        if "sensitivity_analysis" not in tricys_config:
            tricys_config["sensitivity_analysis"] = {"enabled": True}

        tricys_config["sensitivity_analysis"]["analysis_case"] = {
            "name": "SALib_Analysis",
            "independent_variable": "file",
            "independent_variable_sampling": csv_abs_path,
            "dependent_variables": output_metrics,
        }

        return tricys_config

    def load_tricys_results(
        self, sensitivity_summary_csv: str, output_metrics: List[str] = None
    ) -> np.ndarray:
        """
        Read simulation results from the sensitivity_analysis_summary.csv file output by Tricys

        Args:
            sensitivity_summary_csv: Path to the sensitivity analysis summary CSV file output by Tricys
            output_metrics: List of output metrics to extract

        Returns:
            Simulation result array (n_samples, n_metrics)
        """
        if output_metrics is None:
            output_metrics = [
                "Startup_Inventory",
                "Self_Sufficiency_Time",
                "Doubling_Time",
            ]

        logger.info(f"Read data from the Tricys result file: {sensitivity_summary_csv}")

        df = pd.read_csv(sensitivity_summary_csv)

        logger.info(f"Read {len(df)} simulation results")
        logger.info(f"Result file columns: {list(df.columns)}")

        param_cols = []
        metric_cols = []

        for col in df.columns:
            if col in output_metrics:
                metric_cols.append(col)
            elif col in self.problem["names"] if self.problem else False:
                param_cols.append(col)

        logger.info(f"Recognized parameter columns: {param_cols}")
        logger.info(f"Identified metric columns: {metric_cols}")

        ordered_metric_cols = []
        for metric in output_metrics:
            if metric in metric_cols:
                ordered_metric_cols.append(metric)
            else:
                logger.warning(f"Metric column not found: {metric}")

        if not ordered_metric_cols:
            raise ValueError(f"No valid output metrics columns found: {output_metrics}")

        results_data = df[ordered_metric_cols].values

        self.simulation_results = results_data

        logger.info(f"Successfully loaded simulation results: {results_data.shape}")
        logger.info(
            f"Result Statistics:\n{pd.DataFrame(results_data, columns=ordered_metric_cols).describe()}"
        )
        logger.info(
            f"Result preview:\n{pd.DataFrame(results_data, columns=metric_cols).head()}"
        )
        return self.simulation_results

    def get_compatible_analysis_methods(self, sampling_method: str) -> List[str]:
        """
        Get analysis methods compatible with the specified sampling method.

        Args:
            sampling_method: Sampling method

        Returns:
            List of compatible analysis methods
        """
        compatibility_map = {
            "sobol": ["sobol"],
            "morris": ["morris"],
            "fast": ["fast"],
            "latin": ["latin"],
            "unknown": [],
        }

        return compatibility_map.get(sampling_method, [])

    def run_tricys_analysis(
        self, csv_file_path: str = None, output_metrics: List[str] = None
    ) -> str:
        """
        Run the Tricys simulation using the generated CSV parameter file and obtain the sensitivity analysis results

        Args:
            csv_file_path: Path to the CSV parameter file. If None, the last generated file will be used
            output_metrics: List of output metrics to be calculated
            config_output_path: Path for the configuration file output. If None, it will be automatically generated

        Returns:
            Path to the sensitivity_analysis_summary.csv file
        """
        # Generate Tricys configuration file
        tricys_config = self.generate_tricys_config(
            csv_file_path=csv_file_path, output_metrics=output_metrics
        )

        logger.info("Starting Tricys simulation analysis...")

        try:
            # Call the Tricys simulation engine
            from datetime import datetime

            from tricys.simulation.simulation_analysis import run_simulation

            tricys_config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

            run_simulation(tricys_config)

            results_dir = tricys_config["paths"]["results_dir"]

            return Path(results_dir) / "sensitivity_analysis_summary.csv"

        except Exception as e:
            logger.error(f"Tricys simulation execution failed: {e}")
            raise

    def analyze_sobol(self, output_index: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Perform Sobol Sensitivity Analysis

        Args:
            output_index: Output variable index
            **kwargs: Sobol analysis parameters

        Returns:
            Sobol sensitivity analysis results

        Note:
            Sobol analysis requires samples generated using the Saltelli sampling method!
            Results from Morris or FAST sampling cannot be used.
        """
        if self.simulation_results is None:
            raise ValueError("The simulation must be run first to obtain the results.")

        # Check sampling method compatibility
        if (
            hasattr(self, "_last_sampling_method")
            and self._last_sampling_method != "sobol"
        ):
            logger.warning(
                f"⚠️ Currently using {self._last_sampling_method} sampling, but Sobol analysis requires Saltelli sampling!"
            )
            logger.warning(
                "Suggestion: Regenerate samples using generate_samples('sobol')"
            )

        Y = self.simulation_results[:, output_index]

        Y = self._handle_nan_values(Y, "Sobol分析")

        # Remove NaN values
        # valid_indices = ~np.isnan(Y)
        # if not np.all(valid_indices):
        #    logger.warning(f"发现{np.sum(~valid_indices)}个无效结果，将被排除")
        #    Y = Y[valid_indices]
        #    X = self.parameter_samples[valid_indices]
        # else:
        #    X = self.parameter_samples

        try:
            Si = sobol.analyze(self.problem, Y, **kwargs)

            if "sobol" not in self.sensitivity_results:
                self.sensitivity_results["sobol"] = {}

            metric_name = f"metric_{output_index}"
            self.sensitivity_results["sobol"][metric_name] = {
                "output_index": output_index,
                "Si": Si,
                "S1": Si["S1"],
                "ST": Si["ST"],
                "S2": Si.get("S2", None),
                "S1_conf": Si["S1_conf"],
                "ST_conf": Si["ST_conf"],
                "sampling_method": getattr(self, "_last_sampling_method", "unknown"),
            }

            logger.info(f"Sobol sensitivity analysis completed (index {output_index})")
            return self.sensitivity_results["sobol"][metric_name]

        except Exception as e:
            if "saltelli" in str(e).lower() or "sample" in str(e).lower():
                raise ValueError(
                    f"Sobol analysis failed, possibly due to incompatible sampling method: {e}\nPlease regenerate samples using generate_samples('sobol')"
                ) from e
            else:
                raise

    def analyze_morris(self, output_index: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Perform Morris sensitivity analysis

        Args:
            output_index: Output variable index
            **kwargs: Morris analysis parameters

        Returns:
            Morris sensitivity analysis results
        """
        if self.simulation_results is None:
            raise ValueError("The simulation must be run first to obtain the results.")

        Y = self.simulation_results[:, output_index]

        Y = self._handle_nan_values(Y, "Morris分析")
        X = self.parameter_samples

        # Remove NaN values
        # valid_indices = ~np.isnan(Y)
        # if not np.all(valid_indices):
        #    logger.warning(f"发现{np.sum(~valid_indices)}个无效结果，将被排除")
        #    Y = Y[valid_indices]
        #    X = self.parameter_samples[valid_indices]
        # else:
        #    X = self.parameter_samples

        # Perform Morris analysis
        logger.info(
            f"Start Morris sensitivity analysis: X.shape={X.shape}, Y.shape={Y.shape}, X.dtype={X.dtype}"
        )

        try:
            Si = morris_analyze.analyze(self.problem, X, Y, **kwargs)
        except Exception as e:
            logger.error(f"Morris analysis execution failed: {e}")
            logger.error(f"problem: {self.problem}")
            logger.error(f"X shape: {X.shape}, type: {X.dtype}")
            logger.error(f"Yshape: {Y.shape}, type: {Y.dtype}")
            if hasattr(X, "dtype") and X.dtype == "object":
                logger.error(
                    "X contains non-numeric data, please check the sampled data"
                )
            raise

        if "morris" not in self.sensitivity_results:
            self.sensitivity_results["morris"] = {}

        metric_name = f"metric_{output_index}"
        self.sensitivity_results["morris"][metric_name] = {
            "output_index": output_index,
            "Si": Si,
            "mu": Si["mu"],
            "mu_star": Si["mu_star"],
            "sigma": Si["sigma"],
            "mu_star_conf": Si["mu_star_conf"],
        }

        logger.info(f"Morris sensitivity analysis completed (metric {output_index})")
        return self.sensitivity_results["morris"][metric_name]

    def analyze_fast(self, output_index: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Perform FAST sensitivity analysis

        Args:
            output_index: Output variable index
            **kwargs: FAST analysis parameters

        Returns:
            FAST sensitivity analysis results

        Note:
            FAST analysis requires samples generated by the fast_sampler sampling method!
            Results from Morris or Sobol sampling cannot be used.
        """
        if self.simulation_results is None:
            raise ValueError("The simulation must be run first to obtain the results.")

        if (
            hasattr(self, "_last_sampling_method")
            and self._last_sampling_method != "fast"
        ):
            logger.warning(
                f"⚠️ The current sampling method is {self._last_sampling_method}, but FAST analysis requires FAST sampling!"
            )
            logger.warning(
                "Suggestion: Regenerate samples using generate_samples('fast')"
            )

        Y = self.simulation_results[:, output_index]

        Y = self._handle_nan_values(Y, "FAST分析")

        # Remove NaN values
        # valid_indices = ~np.isnan(Y)
        # if not np.all(valid_indices):
        #    logger.warning(f"发现{np.sum(~valid_indices)}个无效结果，将被排除")
        #    Y = Y[valid_indices]

        try:
            # Perform FAST analysis
            Si = fast.analyze(self.problem, Y, **kwargs)

            if "fast" not in self.sensitivity_results:
                self.sensitivity_results["fast"] = {}

            metric_name = f"metric_{output_index}"
            self.sensitivity_results["fast"][metric_name] = {
                "output_index": output_index,
                "Si": Si,
                "S1": Si["S1"],
                "ST": Si["ST"],
                "sampling_method": getattr(self, "_last_sampling_method", "unknown"),
            }

            logger.info(
                f"FAST sensitivity analysis completed (indicator {output_index})"
            )
            return self.sensitivity_results["fast"][metric_name]

        except Exception as e:
            if "fast" in str(e).lower() or "sample" in str(e).lower():
                raise ValueError(
                    f"FAST analysis failed, possibly due to incompatible sampling method: {e}\nPlease regenerate samples using generate_samples('fast')"
                ) from e
            else:
                raise

    def analyze_lhs(self, output_index: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Perform LHS (Latin Hypercube Sampling) uncertainty analysis

        Note: This is a basic statistical analysis method for LHS samples,
        providing descriptive statistics and basic sensitivity indices.

        Args:
            output_index: Output variable index
            **kwargs: Analysis parameters (reserved for future use)

        Returns:
            LHS uncertainty analysis results
        """
        if self.simulation_results is None:
            raise ValueError("The simulation must be run first to obtain the results.")

        if (
            hasattr(self, "_last_sampling_method")
            and self._last_sampling_method != "latin"
        ):
            logger.warning(
                f"⚠️ The current sampling method is {self._last_sampling_method}, but LHS analysis is designed for Latin Hypercube Sampling!"
            )
            logger.warning(
                "Suggestion: Regenerate samples using generate_samples('latin')"
            )

        Y = self.simulation_results[:, output_index]

        # Handle NaN values
        Y = self._handle_nan_values(Y, "LHS分析")

        # Basic statistical analysis
        mean_val = np.mean(Y)
        std_val = np.std(Y)
        min_val = np.min(Y)
        max_val = np.max(Y)
        percentile_5 = np.percentile(Y, 5)
        percentile_95 = np.percentile(Y, 95)

        # Create results dictionary
        Si = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95,
        }

        if "latin" not in self.sensitivity_results:
            self.sensitivity_results["latin"] = {}

        metric_name = f"metric_{output_index}"
        self.sensitivity_results["latin"][metric_name] = {
            "output_index": output_index,
            "Si": Si,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "percentile_5": percentile_5,
            "percentile_95": percentile_95,
            "sampling_method": getattr(self, "_last_sampling_method", "unknown"),
        }

        logger.info(f"LHS uncertainty analysis completed (指标 {output_index})")
        return self.sensitivity_results["latin"][metric_name]

    def run_salib_analysis_from_tricys_results(
        self,
        sensitivity_summary_csv: str,
        param_bounds: Dict[str, Tuple[float, float]] = None,
        output_metrics: List[str] = None,
        methods: List[str] = ["sobol", "morris", "fast"],
        save_dir: str = None,
    ) -> Dict[str, Any]:
        """
        Run a complete SALib sensitivity analysis from the sensitivity analysis results file output by Tricys

        Args:
            sensitivity_summary_csv: Path to the sensitivity summary CSV file output by Tricys
            param_bounds: Dictionary of parameter bounds, inferred from the CSV file if None
            output_metrics: List of output metrics to analyze
            methods: List of sensitivity analysis methods to execute
            save_dir: Directory to save the results

        Returns:
            Dictionary containing all analysis results
        """
        if output_metrics is None:
            output_metrics = [
                "Startup_Inventory",
                "Self_Sufficiency_Time",
                "Doubling_Time",
            ]

        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(sensitivity_summary_csv), "salib_analysis"
            )
        os.makedirs(save_dir, exist_ok=True)

        df = pd.read_csv(sensitivity_summary_csv)

        if param_bounds is None:
            param_bounds = {}
            param_candidates = []
            for col in df.columns:
                if col not in output_metrics and "." in col:
                    param_candidates.append(col)

            for param in param_candidates:
                param_data = df[param].dropna()
                if len(param_data) > 0:
                    param_bounds[param] = (param_data.min(), param_data.max())

        if not param_bounds:
            raise ValueError(
                "Unable to determine parameter boundaries, please provide the param_bounds parameter"
            )

        self.define_problem(param_bounds)

        self.load_tricys_results(sensitivity_summary_csv, output_metrics)

        detected_method = self._last_sampling_method

        methods = self.get_compatible_analysis_methods(detected_method)

        all_results = {}

        for metric_idx, metric_name in enumerate(output_metrics):
            if metric_idx >= self.simulation_results.shape[1]:
                logger.warning(f"The metric {metric_name} is out of range, skipping")
                continue

            logger.info(f"\n=== Analysis indicators: {metric_name} ===")
            metric_results = {}

            # Check data validity
            Y = self.simulation_results[:, metric_idx]
            valid_ratio = np.sum(~np.isnan(Y)) / len(Y)
            logger.info(f"Valid data ratio: {valid_ratio:.2%}")

            if valid_ratio < 0.5:
                logger.warning(
                    f"The metric {metric_name} has less than 50% valid data, which may affect the analysis quality."
                )

            # Sobol analysis
            if "sobol" in methods:
                try:
                    logger.info("Performing Sobol sensitivity analysis...")
                    sobol_result = self.analyze_sobol(output_index=metric_idx)
                    metric_results["sobol"] = sobol_result

                    # Display Sobol results summary
                    logger.info("\nSobol sensitivity index:")
                    for i, param_name in enumerate(self.problem["names"]):
                        s1 = sobol_result["S1"][i]
                        st = sobol_result["ST"][i]
                        logger.info(f"  {param_name}: S1={s1:.4f}, ST={st:.4f}")

                except Exception as e:
                    logger.error(f"Sobol analysis failed: {e}")

            # Morris analysis
            if "morris" in methods:
                try:
                    logger.info("Performing Morris sensitivity analysis...")
                    morris_result = self.analyze_morris(output_index=metric_idx)
                    metric_results["morris"] = morris_result

                    # Display Morris results summary
                    logger.info("\nMorris sensitivity index:")
                    for i, param_name in enumerate(self.problem["names"]):
                        mu_star = morris_result["mu_star"][i]
                        sigma = morris_result["sigma"][i]
                        logger.info(f"  {param_name}: μ*={mu_star:.4f}, σ={sigma:.4f}")

                except Exception as e:
                    logger.error(f"Morris analysis failed: {e}")

            # FAST analysis
            if "fast" in methods:
                try:
                    logger.info("Performing FAST sensitivity analysis...")
                    fast_result = self.analyze_fast(output_index=metric_idx)
                    metric_results["fast"] = fast_result

                    # Display FAST results summary
                    logger.info("\nFAST sensitivity index:")
                    for i, param_name in enumerate(self.problem["names"]):
                        s1 = fast_result["S1"][i]
                        st = fast_result["ST"][i]
                        logger.info(f"  {param_name}: S1={s1:.4f}, ST={st:.4f}")

                except Exception as e:
                    logger.error(f"FAST analysis failed: {e}")

            # LHS analysis
            if "latin" in methods:
                try:
                    logger.info("Performing LHS uncertainty analysis...")
                    lhs_result = self.analyze_lhs(output_index=metric_idx)
                    metric_results["latin"] = lhs_result

                    # Display LHS results summary
                    logger.info("\nLHS分析结果:")
                    logger.info(f"  均值: {lhs_result['mean']:.4f}")
                    logger.info(f"  标准差: {lhs_result['std']:.4f}")
                    logger.info(f"  最小值: {lhs_result['min']:.4f}")
                    logger.info(f"  最大值: {lhs_result['max']:.4f}")
                    logger.info(f"  5%分位数: {lhs_result['percentile_5']:.4f}")
                    logger.info(f"  95%分位数: {lhs_result['percentile_95']:.4f}")

                except Exception as e:
                    logger.error(f"LHS分析失败: {e}")

            all_results[metric_name] = metric_results

        try:
            if "sobol" in methods and "sobol" in self.sensitivity_results:
                self.plot_sobol_results(save_dir=save_dir, metric_names=output_metrics)

            if "morris" in methods and "morris" in self.sensitivity_results:
                self.plot_morris_results(save_dir=save_dir, metric_names=output_metrics)

            if "fast" in methods and "fast" in self.sensitivity_results:
                self.plot_fast_results(save_dir=save_dir, metric_names=output_metrics)

            # Plot LHS results
            if "latin" in methods and "latin" in self.sensitivity_results:
                self.plot_lhs_results(save_dir=save_dir, metric_names=output_metrics)

        except Exception as e:
            logger.warning(f"Drawing failed: {e}")

        try:
            self.save_results(
                save_dir=save_dir, format="csv", metric_names=output_metrics
            )

            report_content = self._save_sensitivity_report(all_results, save_dir)
            report_path = os.path.join(save_dir, "analysis_report.md")

            load_dotenv()

            # --- LLM Calls for analysis ---
            api_key = os.environ.get("API_KEY")
            base_url = os.environ.get("BASE_URL")
            ai_model = os.environ.get("AI_MODEL")

            sa_config = self.base_config.get("sensitivity_analysis", {})
            case_config = sa_config.get("analysis_case", {})
            ai_config = case_config.get("ai")

            ai_enabled = False
            if isinstance(ai_config, bool):
                ai_enabled = ai_config
            elif isinstance(ai_config, dict):
                ai_enabled = ai_config.get("enabled", False)

            if api_key and base_url and ai_model and ai_enabled:
                # First LLM call for initial analysis
                wrapper_prompt, llm_summary = call_llm_for_salib_analysis(
                    report_content=report_content,
                    api_key=api_key,
                    base_url=base_url,
                    ai_model=ai_model,
                    method=detected_method,
                )
                if wrapper_prompt and llm_summary:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write("\n\n---\n\n# AI模型分析提示词\n\n")
                        f.write("```markdown\n")
                        f.write(wrapper_prompt)
                        f.write("\n```\n\n")
                        f.write("\n\n---\n\n# AI模型分析结果\n\n")
                        f.write(llm_summary)
                    logger.info(f"Appended LLM prompt and summary to {report_path}")

                    # Second LLM call for academic report
                    glossary_path = None
                    if isinstance(case_config, dict):
                        glossary_path = sa_config.get("glossary_path")

                    if glossary_path and os.path.exists(glossary_path):
                        try:
                            with open(glossary_path, "r", encoding="utf-8") as f:
                                glossary_content = f.read()

                            (
                                academic_wrapper_prompt,
                                academic_report,
                            ) = call_llm_for_academic_report(
                                analysis_report=llm_summary,
                                glossary_content=glossary_content,
                                api_key=api_key,
                                base_url=base_url,
                                ai_model=ai_model,
                                problem_details=self.problem,
                                metric_names=output_metrics,
                                method=detected_method,
                                save_dir=save_dir,
                            )

                            if academic_wrapper_prompt and academic_report:
                                academic_report_path = os.path.join(
                                    save_dir, "academic_report.md"
                                )
                                with open(
                                    academic_report_path, "w", encoding="utf-8"
                                ) as f:
                                    f.write(academic_report)
                                logger.info(
                                    f"Generated academic report: {academic_report_path}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to generate or save academic report: {e}"
                            )
                    elif glossary_path:
                        logger.warning(
                            f"Glossary file not found at {glossary_path}, skipping academic report generation."
                        )

            else:
                logger.warning(
                    "API_KEY, BASE_URL, or AI_MODEL not set, or AI analysis is disabled. Skipping LLM summary generation."
                )

        except Exception as e:
            logger.warning(f"Failed to save result: {e}")

        logger.info("\n✅ SALib sensitivity analysis completed!")
        logger.info(f"📁 The result has been saved to: {save_dir}")

        return all_results

    def _save_sensitivity_report(
        self, all_results: Dict[str, Any], save_dir: str
    ) -> str:
        """The result has been saved to: {save_dir}"""
        report_file = os.path.join(save_dir, "analysis_report.md")
        # Determine analysis type based on sampling method
        is_uncertainty_analysis = (
            hasattr(self, "_last_sampling_method")
            and self._last_sampling_method == "latin"
        )
        report_title = (
            "# SALib 不确定性分析报告\n\n"
            if is_uncertainty_analysis
            else "# SALib 敏感性分析报告\n\n"
        )
        report_lines = []
        report_lines.append(report_title)
        report_lines.append(f"生成时间: {pd.Timestamp.now()}\n\n")
        # Get unit_map from config
        sensitivity_analysis_config = self.base_config.get("sensitivity_analysis", {})
        unit_map = sensitivity_analysis_config.get("unit_map", {})
        report_lines.append("## 分析参数\n\n")
        if self.problem:
            for i, param_name in enumerate(self.problem["names"]):
                bounds = self.problem["bounds"][i]
                # --- Unit Conversion Logic for Bounds ---
                unit_config = self._find_unit_config(param_name, unit_map)
                display_bounds = list(bounds)
                unit_str = ""
                if unit_config:
                    unit = unit_config.get("unit")
                    factor = unit_config.get("conversion_factor")
                    if factor:
                        display_bounds[0] /= float(factor)
                        display_bounds[1] /= float(factor)
                    if unit:
                        unit_str = f" ({unit})"
                # --- End Conversion Logic ---
                report_lines.append(
                    f"- **{param_name}**: [{display_bounds[0]:.4f}, {display_bounds[1]:.4f}]{unit_str}\n"
                )
        report_lines.append("\n")
        for metric_name, metric_results in all_results.items():
            metric_section_title = (
                f"## {metric_name} 不确定性分析结果\n\n"
                if is_uncertainty_analysis
                else f"## {metric_name} 敏感性分析结果\n\n"
            )
            report_lines.append(metric_section_title)
            if "sobol" in metric_results:
                report_lines.append("### Sobol敏感性指数\n\n")
                report_lines.append(
                    "| 参数 | S1 (一阶) | ST (总) | S1置信区间 | ST置信区间 |\n"
                )
                report_lines.append(
                    "|------|----------|---------|------------|------------|\n"
                )
                sobol_data = metric_results["sobol"]
                for i, param_name in enumerate(self.problem["names"]):
                    s1 = sobol_data["S1"][i]
                    st = sobol_data["ST"][i]
                    s1_conf = sobol_data["S1_conf"][i]
                    st_conf = sobol_data["ST_conf"][i]
                    report_lines.append(
                        f"| {param_name} | {s1:.4f} | {st:.4f} | ±{s1_conf:.4f} | ±{st_conf:.4f} |\n"
                    )
                report_lines.append("\n")
                plot_filename = (
                    f'sobol_sensitivity_indices_{metric_name.replace(" ", "_")}.png'
                )
                report_lines.append(
                    f"![Sobol Analysis for {metric_name}]({plot_filename})\n\n"
                )
            if "morris" in metric_results:
                report_lines.append("### Morris敏感性指数\n\n")
                report_lines.append(
                    "| 参数 | μ* (平均绝对效应) | σ (标准差) | μ*置信区间 |\n"
                )
                report_lines.append(
                    "|------|-------------------|------------|------------|\n"
                )
                morris_data = metric_results["morris"]
                for i, param_name in enumerate(self.problem["names"]):
                    mu_star = morris_data["mu_star"][i]
                    sigma = morris_data["sigma"][i]
                    mu_star_conf = morris_data["mu_star_conf"][i]
                    report_lines.append(
                        f"| {param_name} | {mu_star:.4f} | {sigma:.4f} | ±{mu_star_conf:.4f} |\n"
                    )
                report_lines.append("\n")
                plot_filename = (
                    f'morris_sensitivity_analysis_{metric_name.replace(" ", "_")}.png'
                )
                report_lines.append(
                    f"![Morris Analysis for {metric_name}]({plot_filename})\n\n"
                )
            if "fast" in metric_results:
                report_lines.append("### FAST敏感性指数\n\n")
                report_lines.append("| 参数 | S1 (一阶) | ST (总) |\n")
                report_lines.append("|------|----------|---------|\n")
                fast_data = metric_results["fast"]
                for i, param_name in enumerate(self.problem["names"]):
                    s1 = fast_data["S1"][i]
                    st = fast_data["ST"][i]
                    report_lines.append(f"| {param_name} | {s1:.4f} | {st:.4f} |\n")
                report_lines.append("\n")
                plot_filename = (
                    f'fast_sensitivity_indices_{metric_name.replace(" ", "_")}.png'
                )
                report_lines.append(
                    f"![FAST Analysis for {metric_name}]({plot_filename})\n\n"
                )
            if "latin" in metric_results:
                # --- Unit Conversion Logic for Metrics ---
                unit_config = self._find_unit_config(metric_name, unit_map)
                unit_str = ""
                factor = 1.0
                if unit_config:
                    unit = unit_config.get("unit")
                    conv_factor = unit_config.get("conversion_factor")
                    if conv_factor:
                        factor = float(conv_factor)
                    if unit:
                        unit_str = f" ({unit})"
                # --- End Conversion Logic ---
                # 1. Get raw data and clean it
                output_index = metric_results["latin"]["output_index"]
                Y = self.simulation_results[:, output_index]
                Y_clean = Y[~np.isnan(Y)]
                # --- Modify report generation ---
                report_lines.append("### 统计摘要\n\n")
                lhs_data = metric_results["latin"]
                report_lines.append(
                    f"- 均值: {lhs_data['mean']/factor:.4f}{unit_str}\n"
                )
                report_lines.append(
                    f"- 标准差: {lhs_data['std']/factor:.4f}{unit_str}\n"
                )
                report_lines.append(
                    f"- 最小值: {lhs_data['min']/factor:.4f}{unit_str}\n"
                )
                report_lines.append(
                    f"- 最大值: {lhs_data['max']/factor:.4f}{unit_str}\n\n"
                )
                # 2. Calculate more percentiles
                if len(Y_clean) > 0:
                    percentiles_to_calc = [5, 10, 25, 50, 75, 90, 95]
                    percentile_values = np.percentile(Y_clean, percentiles_to_calc)
                    report_lines.append("### 分布关键点 (CDF)\n\n")
                    report_lines.append(
                        f"- 5%分位数: {percentile_values[0]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 10%分位数: {percentile_values[1]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 25%分位数 (Q1): {percentile_values[2]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 50%分位数 (中位数): {percentile_values[3]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 75%分位数 (Q3): {percentile_values[4]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 90%分位数: {percentile_values[5]/factor:.4f}{unit_str}\n"
                    )
                    report_lines.append(
                        f"- 95%分位数: {percentile_values[6]/factor:.4f}{unit_str}\n\n"
                    )
                    # 3. Calculate histogram data
                    hist_freq, bin_edges = np.histogram(Y_clean, bins=10)
                    report_lines.append("### 输出分布 (直方图数据)\n\n")
                    report_lines.append("| 数值范围 | 频数 |\n")
                    report_lines.append("|:---|---:|\n")
                    for i in range(len(hist_freq)):
                        lower_bound = bin_edges[i] / factor
                        upper_bound = bin_edges[i + 1] / factor
                        freq = hist_freq[i]
                        report_lines.append(
                            f"| {lower_bound:.2f} - {upper_bound:.2f} | {freq} |\n"
                        )
                    report_lines.append("\n")
                plot_filename = f'lhs_analysis_{metric_name.replace(" ", "_")}.png'
                report_lines.append(
                    f"![LHS Analysis for {metric_name}]({plot_filename})\n\n"
                )
        report_content = "".join(report_lines)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"The sensitivity analysis report has been saved.: {report_file}")
        return report_content

    def plot_sobol_results(
        self,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        metric_names: List[str] = None,
    ) -> None:
        """Plot Sobol analysis results"""
        if "sobol" not in self.sensitivity_results:
            raise ValueError("No analysis results for the Sobol method were found.")

        # Ensure Chinese font settings
        self._setup_chinese_font()

        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        # Get the results of all indicators
        sobol_results = self.sensitivity_results["sobol"]

        if not sobol_results:
            raise ValueError("Sobol analysis results not found")

        # Generate charts for each metric
        for metric_key, results in sobol_results.items():
            Si = results["Si"]
            output_index = results["output_index"]

            if metric_names and output_index < len(metric_names):
                metric_display_name = metric_names[output_index]
            else:
                metric_display_name = f"Metric_{output_index}"

            # Bar chart of first-order and total sensitivity indices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # First-order sensitivity index
            y_pos = np.arange(len(self.problem["names"]))
            ax1.barh(y_pos, Si["S1"], xerr=Si["S1_conf"], alpha=0.7, color="skyblue")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(self.problem["names"], fontsize=10)
            ax1.set_xlabel("First-order sensitivity index (S1)", fontsize=12)
            ax1.set_title(
                f"First-order Sensitivity Indices\n{metric_display_name}",
                fontsize=14,
                pad=20,
            )
            ax1.grid(True, alpha=0.3)

            # # Total Sensitivity Index
            ax2.barh(y_pos, Si["ST"], xerr=Si["ST_conf"], alpha=0.7, color="orange")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(self.problem["names"], fontsize=10)
            ax2.set_xlabel("Total Sensitivity Index (ST)", fontsize=12)
            ax2.set_title(
                f"Total Sensitivity Indices\n{metric_display_name}", fontsize=14, pad=20
            )
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = (
                f'sobol_sensitivity_indices_{metric_display_name.replace(" ", "_")}.png'
            )
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            logger.info(f"Sobol result chart has been saved: {filename}")

    def plot_morris_results(
        self,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        metric_names: List[str] = None,
    ) -> None:
        """Plot the Morris analysis results"""
        if "morris" not in self.sensitivity_results:
            raise ValueError("No analysis results were found for the Morris method.")

        # Ensure Chinese font settings
        self._setup_chinese_font()

        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        # Obtain the results of all indicators
        morris_results = self.sensitivity_results["morris"]

        if not morris_results:
            raise ValueError("No Morris analysis results found")

        for metric_key, results in morris_results.items():
            Si = results["Si"]
            output_index = results["output_index"]

            if metric_names and output_index < len(metric_names):
                metric_display_name = metric_names[output_index]
            else:
                metric_display_name = f"Metric_{output_index}"

            # Morris μ*-σ diagram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # μ*-σ scatter plot
            ax1.scatter(Si["mu_star"], Si["sigma"], s=100, alpha=0.7, color="red")
            for i, name in enumerate(self.problem["names"]):
                ax1.annotate(
                    name,
                    (Si["mu_star"][i], Si["sigma"][i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            ax1.set_xlabel("μ*(Average Absolute Effect)", fontsize=12)
            ax1.set_ylabel("σ (Standard Deviation)", fontsize=12)
            ax1.set_title(
                f"Morris μ*-σ Plot\n{metric_display_name}", fontsize=14, pad=20
            )
            ax1.grid(True, alpha=0.3)

            y_pos = np.arange(len(self.problem["names"]))
            ax2.barh(
                y_pos, Si["mu_star"], xerr=Si["mu_star_conf"], alpha=0.7, color="green"
            )
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(self.problem["names"], fontsize=10)
            ax2.set_xlabel("μ*(Average Absolute Effect)", fontsize=12)
            ax2.set_title(
                f"Morris Elementary Effects\n{metric_display_name}", fontsize=14, pad=20
            )
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'morris_sensitivity_analysis_{metric_display_name.replace(" ", "_")}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            logger.info(f"Morris result chart has been saved: {filename}")

    def plot_fast_results(
        self,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        metric_names: List[str] = None,
    ) -> None:
        """Plot FAST analysis results"""
        if "fast" not in self.sensitivity_results:
            raise ValueError("No analysis results found for the FAST method")

        # No analysis results found for the FAST method
        self._setup_chinese_font()

        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        # Get the results of all indicators
        fast_results = self.sensitivity_results["fast"]

        if not fast_results:
            raise ValueError("FAST analysis results not found")

        # Generate a chart for each metric
        for metric_key, results in fast_results.items():
            Si = results["Si"]
            output_index = results["output_index"]

            # Determine the indicator name
            if metric_names and output_index < len(metric_names):
                metric_display_name = metric_names[output_index]
            else:
                metric_display_name = f"Metric_{output_index}"

            # Bar charts of first-order and total sensitivity indices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # first-order sensitivity index
            y_pos = np.arange(len(self.problem["names"]))
            ax1.barh(y_pos, Si["S1"], alpha=0.7, color="purple")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(self.problem["names"], fontsize=10)
            ax1.set_xlabel("一阶敏感性指数 (S1)", fontsize=12)
            ax1.set_title(
                f"FAST First-order Sensitivity Indices\n{metric_display_name}",
                fontsize=14,
                pad=20,
            )
            ax1.grid(True, alpha=0.3)

            # Total Sensitivity Index
            ax2.barh(y_pos, Si["ST"], alpha=0.7, color="darkgreen")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(self.problem["names"], fontsize=10)
            ax2.set_xlabel("总敏感性指数 (ST)", fontsize=12)
            ax2.set_title(
                f"FAST Total Sensitivity Indices\n{metric_display_name}",
                fontsize=14,
                pad=20,
            )
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = (
                f'fast_sensitivity_indices_{metric_display_name.replace(" ", "_")}.png'
            )
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            logger.info(f"The FAST result chart has been saved: {filename}")

    def plot_lhs_results(
        self,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        metric_names: List[str] = None,
    ) -> None:
        """Plot LHS (Latin Hypercube Sampling) uncertainty analysis results"""
        if "latin" not in self.sensitivity_results:
            raise ValueError("No analysis results found for the LHS method")

        # Ensure Chinese font settings
        self._setup_chinese_font()

        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        # Get the results of all indicators
        lhs_results = self.sensitivity_results["latin"]

        if not lhs_results:
            raise ValueError("LHS analysis results not found")

        # Generate charts for each metric
        for metric_key, results in lhs_results.items():
            Si = results["Si"]
            output_index = results["output_index"]

            # Determine the indicator name
            if metric_names and output_index < len(metric_names):
                metric_display_name = metric_names[output_index]
            else:
                metric_display_name = f"Metric_{output_index}"

            # Get unit from config
            sensitivity_analysis_config = self.base_config.get(
                "sensitivity_analysis", {}
            )
            unit_map = sensitivity_analysis_config.get("unit_map", {})
            unit_config = self._find_unit_config(metric_display_name, unit_map)
            unit_str = ""
            if unit_config:
                unit = unit_config.get("unit")
                if unit:
                    unit_str = f" ({unit})"

            xlabel = f"{metric_display_name}{unit_str}"

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Plot 1: Distribution histogram
            ax1.hist(
                self.simulation_results[:, output_index],
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax1.set_xlabel(xlabel, fontsize=12)
            ax1.set_ylabel("频率", fontsize=12)
            ax1.set_title("输出分布直方图", fontsize=14, pad=10)
            ax1.grid(True, alpha=0.3)

            # Add statistics text to the histogram plot
            stats_text = f"均值: {Si['mean']:.4f}\n标准差: {Si['std']:.4f}\n最小值: {Si['min']:.4f}\n最大值: {Si['max']:.4f}"
            ax1.text(
                0.05,
                0.95,
                stats_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 2: Cumulative distribution function
            sorted_data = np.sort(self.simulation_results[:, output_index])
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax2.plot(sorted_data, y_vals, linewidth=2, color="darkgreen")
            ax2.set_xlabel(xlabel, fontsize=12)
            ax2.set_ylabel("累积概率", fontsize=12)
            ax2.set_title("累积分布函数", fontsize=14, pad=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'lhs_analysis_{metric_display_name.replace(" ", "_")}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            logger.info(f"LHS分析结果图表已保存: {filename}")

    def save_results(
        self, save_dir: str = None, format: str = "csv", metric_names: List[str] = None
    ) -> None:
        """
        Save sensitivity analysis results

        Args:
            save_dir: Save directory
            format: Save format ('csv
        """
        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        for method, method_results in self.sensitivity_results.items():
            if not method_results:
                continue

            for metric_key, results in method_results.items():
                output_index = results["output_index"]

                if metric_names and output_index < len(metric_names):
                    metric_display_name = metric_names[output_index]
                else:
                    metric_display_name = f"Metric_{output_index}"

                if format == "csv":
                    if method == "sobol":
                        sobol_df = pd.DataFrame(
                            {
                                "Parameter": self.problem["names"],
                                "S1": results["S1"],
                                "ST": results["ST"],
                                "S1_conf": results["S1_conf"],
                                "ST_conf": results["ST_conf"],
                            }
                        )
                        filename = (
                            f'sobol_indices_{metric_display_name.replace(" ", "_")}.csv'
                        )
                        sobol_df.to_csv(os.path.join(save_dir, filename), index=False)
                        logger.info(f"Sobol results have been saved: {filename}")

                    elif method == "morris":
                        morris_df = pd.DataFrame(
                            {
                                "Parameter": self.problem["names"],
                                "mu": results["mu"],
                                "mu_star": results["mu_star"],
                                "sigma": results["sigma"],
                                "mu_star_conf": results["mu_star_conf"],
                            }
                        )
                        filename = f'morris_indices_{metric_display_name.replace(" ", "_")}.csv'
                        morris_df.to_csv(os.path.join(save_dir, filename), index=False)
                        logger.info(f"Morris results have been saved: {filename}")

                    elif method == "fast":
                        fast_df = pd.DataFrame(
                            {
                                "Parameter": self.problem["names"],
                                "S1": results["S1"],
                                "ST": results["ST"],
                            }
                        )
                        filename = (
                            f'fast_indices_{metric_display_name.replace(" ", "_")}.csv'
                        )
                        fast_df.to_csv(os.path.join(save_dir, filename), index=False)
                        logger.info(f"FAST results have been saved: {filename}")

                    elif method == "latin":
                        # Save LHS statistics
                        lhs_stats_df = pd.DataFrame(
                            {
                                "Metric": [metric_display_name],
                                "Mean": [results["mean"]],
                                "Std": [results["std"]],
                                "Min": [results["min"]],
                                "Max": [results["max"]],
                                "Percentile_5": [results["percentile_5"]],
                                "Percentile_95": [results["percentile_95"]],
                            }
                        )
                        filename_stats = (
                            f'lhs_stats_{metric_display_name.replace(" ", "_")}.csv'
                        )
                        lhs_stats_df.to_csv(
                            os.path.join(save_dir, filename_stats), index=False
                        )
                        logger.info(f"LHS统计结果已保存: {filename_stats}")

                        # Remove LHS sensitivity indices saving
                        # lhs_sens_df = pd.DataFrame({
                        #     "Parameter": self.problem["names"],
                        #     "Partial_Correlation": results["partial_correlations"]
                        # })
                        # filename_sens = f'lhs_sensitivity_{metric_display_name.replace(" ", "_")}.csv'
                        # lhs_sens_df.to_csv(os.path.join(save_dir, filename_sens), index=False)
                        # logger.info(f"LHS敏感性结果已保存: {filename_sens}")

        logger.info(f"The result has been saved to: {save_dir}")


def call_llm_for_salib_analysis(
    report_content: str, api_key: str, base_url: str, ai_model: str, method: str
) -> Tuple[str, str]:
    """Sends a SALib analysis report to an LLM for summarization and returns the prompt and summary."""
    try:
        logger.info("Proceeding with LLM analysis for SALib report.")

        PROMPT_TEMPLATES = {
            "sobol": """**角色：** 你是一名在氚燃料循环领域具有深厚背景的敏感性分析专家。

**任务：** 请仔细审查并解读以下这份由SALib库生成的**Sobol敏感性分析**报告。你的目标是：
1.  **总结核心发现**：对于报告中提到的每一个输出指标（如“启动氚量”等），总结其敏感性分析结果。
2.  **识别关键参数**：明确指出哪些输入参数的**一阶敏感性指数（S1）**和**总敏感性指数（ST）**最高。
3.  **解读指数含义**：解释S1和ST指数的含义。例如，高S1值表示参数对输出有重要的直接影响，而ST与S1的显著差异表示参数存在强烈的交互作用或非线性效应。
4.  **提供综合结论**：基于所有分析结果，对模型的整体行为、参数间的相互作用，以及这些发现对工程实践的潜在启示，给出一个综合性的结论。

请确保你的分析清晰、专业，并直接切入要点。
""",
            "morris": """**角色：** 你是一名在氚燃料循环领域具有深厚背景的敏感性分析专家。

**任务：** 请仔细审查并解读以下这份由SALib库生成的**Morris敏感性分析**报告。你的目标是：
1.  **总结核心发现**：对于报告中提到的每一个输出指标（如“启动氚量”等），总结其敏感性分析结果。
2.  **识别关键参数**：根据**μ* (mu_star)**值对参数进行排序，识别出对模型输出影响最大的参数。
3.  **解读参数效应**：解释**μ***和**σ (sigma)**的含义。高μ*表示参数有重要影响，高σ表示参数存在非线性影响或与其他参数有交互作用。结合μ*-σ图进行分析。
4.  **提供综合结论**：基于所有分析结果，对模型的整体行为、参数间的相互作用，以及这些发现对工程实践的潜在启示，给出一个综合性的结论。

请确保你的分析清晰、专业，并直接切入要点。
""",
            "fast": """**角色：** 你是一名在氚燃料循环领域具有深厚背景的敏感性分析专家。

**任务：** 请仔细审查并解读以下这份由SALib库生成的**FAST敏感性分析**报告。你的目标是：
1.  **总结核心发现**：对于报告中提到的每一个输出指标（如“启动氚量”等），总结其敏感性分析结果。
2.  **识别关键参数**：明确指出哪些输入参数的**一阶敏感性指数（S1）**和**总敏感性指数（ST）**最高。
3.  **解读指数含义**：解释S1和ST指数的含义。高S1值表示参数对输出有重要的直接影响，而ST与S1的差异表示参数可能存在交互作用。
4.  **提供综合结论**：基于所有分析结果，对模型的整体行为、参数间的相互作用，以及这些发现对工程实践的潜在启示，给出一个综合性的结论。

请确保你的分析清晰、专业，并直接切入要点。
""",
            "latin": """**角色：** 你是一名在氚燃料循环领域具有深厚背景的统计学和不确定性分析专家。

**任务：** 请仔细审查并解读以下这份由拉丁超立方采样（LHS）生成的不确定性分析报告。你的目标是：
1.  **解读统计数据**：对于报告中的每一个输出指标（如“启动氚量”等），解读其均值、标准差、最大/最小值和百分位数。
2.  **评估不确定性**：基于标准差和5%/95%百分位数的范围，评估模型输出结果的不确定性或波动范围有多大。
3.  **提供综合结论**：总结在给定的参数不确定性下，模型的关键性能指标（KPIs）表现如何，是否存在较大的风险（例如，输出值波动范围过大），并对模型的稳定性给出评价。

请确保你的分析聚焦于不确定性的量化和解读，而不是参数的敏感性排序。
""",
        }

        wrapper_prompt = PROMPT_TEMPLATES.get(
            method,
            """**角色：** 你是一名在氚燃料循环领域具有深厚背景的敏感性分析专家。

**任务：** 请仔细审查并解读以下这份由SALib库生成的敏感性分析报告。你的目标是：
1.  **总结核心发现**：简明扼要地总结报告中的关键信息。
2.  **识别关键参数**：对于报告中提到的每一个输出指标（如“启动氚量”、“倍增时间”等），明确指出哪些输入参数对它的影响最大（即最敏感）。
3.  **提供综合结论**：基于所有分析结果，对模型的整体行为、参数间的相互作用（如果可能）以及这些发现对工程实践的潜在启示，给出一个综合性的结论。

请确保你的分析清晰、专业，并直接切入要点。
""",
        )

        full_prompt = f"{wrapper_prompt}\n\n---\n**分析报告原文：**\n\n{report_content}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending SALib report to LLM (Attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=ai_model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=4000,
                )
                llm_summary = response.choices[0].message.content

                logger.info("LLM analysis successful for SALib report.")
                return wrapper_prompt, llm_summary  # Return wrapper prompt and summary

            except Exception as e:
                logger.error(
                    f"Error calling LLM for SALib report on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to get LLM summary for SALib report after {max_retries} attempts."
                    )
                    return None, None  # Return None on failure

    except Exception as e:
        logger.error(f"Error in call_llm_for_salib_analysis: {e}", exc_info=True)
        return None, None


def call_llm_for_academic_report(
    analysis_report: str,
    glossary_content: str,
    api_key: str,
    base_url: str,
    ai_model: str,
    problem_details: dict,
    metric_names: list,
    method: str,
    save_dir: str,
) -> Tuple[str, str]:
    """Sends an analysis report and a glossary to an LLM to generate a professional academic report."""
    try:
        logger.info("Proceeding with LLM for academic report generation.")

        param_names_str = ", ".join(
            [f"`{name}`" for name in problem_details.get("names", [])]
        )
        metric_names_str = ", ".join([f"`{name}`" for name in metric_names])

        all_plots = [f for f in os.listdir(save_dir) if f.endswith((".svg", ".png"))]
        plot_list_str = "\n".join([f"    *   `{plot}`" for plot in all_plots])

        method_details = {
            "sobol": {
                "name": "Sobol",
                "methodology": "指出本次分析采用了SALib库，并使用了**Sobol方法**。这是一种基于方差的全局敏感性分析技术，能够量化单个参数以及参数间交互作用对模型输出方差的贡献。",
                "results_discussion": """*   对于每个性能指标，哪些输入参数的一阶敏感性（S1）和总体敏感性（ST）最高？请结合图表（如条形图）进行解读。
        *   S1和ST指数之间的差异揭示了什么？（例如，ST显著大于S1意味着该参数与其他参数存在显著的交互作用或其影响是非线性的）。
        *   分析不同指标之间的**权衡关系 (Trade-offs)**。例如，某个参数对某个指标 (e.g., `Startup_Inventory`) 有正面影响，但可能对另一个指标 (e.g., `Doubling_Time`) 有负面影响。""",
            },
            "morris": {
                "name": "Morris",
                "methodology": "指出本次分析采用了SALib库，并使用了**Morris方法**。这是一种基于轨迹的“一次性”设计方法，常用于在高维参数空间中进行参数筛选，以识别出影响最大的少数几个参数。",
                "results_discussion": """*   对于每个性能指标，哪些参数的 `μ*` (mu_star) 值最高，表明其对输出的总体影响最重要？
        *   `σ` (sigma) 值的大小又说明了什么？较高的 `σ` 值通常表明参数具有非线性效应或与其他参数存在强烈的交互作用。
        *   请结合 `μ*-σ` 图进行分析，对参数进行分类（例如，高 `μ*`/高 `σ` vs. 高 `μ*`/低 `σ`），并解释其含义。
        *   分析不同指标之间的**权衡关系 (Trade-offs)**。例如，某个参数对某个指标 (e.g., `Startup_Inventory`) 有正面影响，但可能对另一个指标 (e.g., `Doubling_Time`) 有负面影响。""",
            },
            "fast": {
                "name": "FAST",
                "methodology": "指出本次分析采用了SALib库，并使用了**FAST（傅里叶幅度敏感性检验）方法**。这是一种基于频率的全局敏感性分析技术，通过将参数在傅里叶级数中展开来计算敏感性指数。",
                "results_discussion": """*   对于每个性能指标，哪些输入参数的一阶敏感性（S1）最高？
        *   （如果可用）总体敏感性（ST）与一阶敏感性（S1）的比较揭示了什么？较大的差异通常表明存在参数交互。
        *   分析不同指标之间的**权衡关系 (Trade-offs)**。例如，某个参数对某个指标 (e.g., `Startup_Inventory`) 有正面影响，但可能对另一个指标 (e.g., `Doubling_Time`) 有负面影响。""",
            },
        }

        if method == "latin":
            ACADEMIC_REPORT_PROMPT_WRAPPER = f"""**角色：** 您是一位在核聚变工程，特别是氚燃料循环领域，具有深厚学术背景的资深科学家，擅长进行**不确定性量化 (UQ)** 和风险评估。

**任务：** 您收到了一个基于**拉丁超立方采样 (LHS)** 的不确定性分析初步报告和一份专业术语表。请您基于这两份文件，撰写一份更加专业、正式、符合学术发表标准的深度分析总结报告。

**指令：**

1.  **专业化语言：** 将初步报告中的模型参数/缩写替换为术语表中对应的专业词汇。
2.  **学术化重述：** 用严谨、客观的学术语言重新组织和阐述初步报告中的发现，聚焦于**不确定性**的量化和解读。
3.  **图表和表格的呈现与引用：**
    *   **显示图表：** 在报告的“结果与讨论”部分，您**必须**使用Markdown语法 `![图表标题](图表文件名)` 来**直接嵌入**和显示初步报告中包含的所有图表。可用的图表文件如下：
{plot_list_str}
    *   **引用图表：** 在正文中分析和讨论图表内容时，请使用“如图1所示...”等方式对图表进行编号和文字引用。
    *   **显示表格：** 当呈现数据时（例如，统计摘要、分布数据等），您**必须**使用Markdown的管道表格（pipe-table）格式来清晰地展示它们。您可以直接复用或重新格式化初步报告中的数据表格。
4.  **结构化报告：** 您的报告是关于一项**不确定性分析**。报告应包含以下部分：
    *   **摘要 (Abstract):** 简要概括本次不确定性研究的目的，明确指出分析的输入参数是 {param_names_str}，总结这些参数的不确定性对关键性能指标 ({metric_names_str}) 的输出分布（如均值、标准差、置信区间）有何影响。
    *   **引言 (Introduction):** 描述进行这项不确定性分析的背景和重要性。阐述研究目标，即量化评估当输入参数 {param_names_str} 在其定义域内变化时，氚燃料循环系统关键性能指标的统计分布和稳定性。
    *   **方法 (Methodology):** 简要说明分析方法。指出本次分析采用了拉丁超立方采样（LHS）方法来对输入参数空间进行抽样。说明被评估的关键性能指标是 {metric_names_str}，以及输入参数的概率分布和范围。
    *   **结果与讨论 (Results and Discussion):** 这是报告的核心。请结合初步报告中的统计数据和您嵌入的图表（如直方图、累积分布图），分点详细论述：
        *   对于每个性能指标，其输出的**概率分布**是怎样的？（例如，是正态分布、偏态分布还是双峰分布？）
        *   输出指标的**不确定性范围**有多大？（参考标准差和5%-95%百分位数区间）。这个范围在工程实践中是否可以接受？
        *   是否存在某些指标的波动范围过大，可能导致系统性能低于设计要求或存在运行风险？
    *   **结论 (Conclusion):** 总结本次不确定性分析得出的主要学术结论（例如，模型的稳定性、输出指标的可靠性等），并对降低关键指标不确定性或未来的风险评估提出具体建议。
5.  **输出格式：** 请直接输出完整的学术分析报告正文，确保所有内容都遵循正确的Markdown语法。

**输入文件：**
"""
        else:
            selected_method = method_details.get(method)
            if not selected_method:
                # Fallback for unknown methods
                selected_method = {
                    "name": method.capitalize(),
                    "methodology": f"指出本次分析采用了SALib库，并提及具体的敏感性分析方法为**{method.capitalize()}**。",
                    "results_discussion": "*   对于每个性能指标，识别出最重要的输入参数。\n*   讨论这些发现的意义。",
                }

            ACADEMIC_REPORT_PROMPT_WRAPPER = f"""**角色：** 您是一位在核聚变工程，特别是氚燃料循环领域，具有深厚学术背景的资深科学家。

**任务：** 您收到了一个关于**SALib {selected_method['name']} 方法敏感性分析**的程序生成的初步报告和一份专业术语表。请您基于这两份文件，撰写一份更加专业、正式、符合学术发表标准的深度分析总结报告。

**指令：**

1.  **专业化语言：** 将初步报告中的模型参数/缩写（例如 `sds.I[1]`, `Startup_Inventory`）替换为术语表中对应的“中文翻译”或“英文术语”。
2.  **学术化重述：** 用严谨、客观的学术语言重新组织和阐述初步报告中的发现。
3.  **图表和表格的呈现与引用：**
    *   **显示图表：** 在报告的“结果与讨论”部分，您**必须**使用Markdown语法 `![图表标题](图表文件名)` 来**直接嵌入**和显示初步报告中包含的所有图表。可用的图表文件如下：
{plot_list_str}
    *   **引用图表：** 在正文中分析和讨论图表内容时，请使用“如图1所示...”等方式对图表进行编号和文字引用。
    *   **显示表格：** 当呈现数据时（例如，敏感性指数表），您**必须**使用Markdown的管道表格（pipe-table）格式来清晰地展示它们。您可以直接复用或重新格式化初步报告中的数据表格。
4.  **结构化报告：** 您的报告是关于一项**敏感性分析**。报告应包含以下部分：
    *   **摘要 (Abstract):** 简要概括本次敏感性研究的目的，明确指明分析的输入参数是 {param_names_str}，总结哪些参数对关键性能指标 ({metric_names_str}) 影响最显著，并陈述核心结论。
    *   **引言 (Introduction):** 描述进行这项敏感性分析的背景和重要性。阐述研究目标，即量化评估输入参数的变化对氚燃料循环系统性能的影响。
    *   **方法 (Methodology):** {selected_method['methodology']} 说明被评估的关键性能指标是 {metric_names_str}，以及输入参数 {param_names_str} 的变化范围。
    *   **结果与讨论 (Results and Discussion):** 这是报告的核心。请结合初步报告中的数据和您嵌入的图表，分点详细论述：
{selected_method['results_discussion']}
    *   **结论 (Conclusion):** 总结本次敏感性分析得出的主要学术结论，并对反应堆设计或未来研究方向提出具体建议。
5.  **输出格式：** 请直接输出完整的学术分析报告正文，确保所有内容都遵循正确的Markdown语法。

**输入文件：**
"""

        full_prompt = f"{ACADEMIC_REPORT_PROMPT_WRAPPER}\n\n---\n### 1. 初步分析报告\n---\n{analysis_report}\n\n---\n### 2. 专业术语表\n---\n{glossary_content}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending data for academic report to LLM (Attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=ai_model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=4000,
                )
                academic_report = response.choices[0].message.content

                logger.info("LLM academic report generation successful.")
                return ACADEMIC_REPORT_PROMPT_WRAPPER, academic_report

            except Exception as e:
                logger.error(
                    f"Error calling LLM for academic report on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to get LLM academic report after {max_retries} attempts."
                    )
                    return None, None

    except Exception as e:
        logger.error(f"Error in call_llm_for_academic_report: {e}", exc_info=True)
        return None, None


def run_salib_analysis(config: Dict[str, Any]) -> None:
    """
    Orchestrates the SALib sensitivity analysis workflow.

    This function extracts the necessary configuration, defines the problem
    space for SALib, and then runs the analysis.

    Args:
        config: The main configuration dictionary.
    """
    # 1. Extract sensitivity analysis configuration
    sa_config = config.get("sensitivity_analysis")
    if not sa_config or not sa_config.get("enabled"):
        logger.info("Sensitivity analysis is not enabled in the configuration file.")
        return

    # 2. Create analyzer
    analyzer = TricysSALibAnalyzer(config)

    # 3. Define the problem space from configuration
    analysis_case = sa_config.get("analysis_case", {})
    param_names = analysis_case.get("independent_variable")
    sampling_details = analysis_case.get("independent_variable_sampling")

    if not isinstance(param_names, list):
        raise ValueError("'independent_variable' must be a list of parameter names.")
    if not isinstance(sampling_details, dict):
        raise ValueError(
            "'independent_variable_sampling' must be an object with parameter details."
        )

    param_bounds = {
        name: sampling_details[name]["bounds"]
        for name in param_names
        if name in sampling_details
    }
    param_dists = {
        name: sampling_details[name].get("distribution", "unif")
        for name in param_names
        if name in sampling_details
    }

    if len(param_bounds) != len(param_names):
        raise ValueError(
            "The keys of 'independent_variable' and 'independent_variable_sampling' do not match"
        )

    problem = analyzer.define_problem(param_bounds, param_dists)
    logger.info(
        f"\n🔍 The problem space with {problem['num_vars']} parameters was defined from the configuration file"
    )

    # 4. Generate samples from configuration
    analyzer_config = analysis_case.get("analyzer", {})
    enabled_method_name = analyzer_config.get("method")
    if not enabled_method_name:
        raise ValueError(
            "No method found in 'sensitivity_analysis.analysis_case.analyzer'"
        )

    N = analyzer_config.get("sample_N", 1024)

    sample_kwargs = {}

    samples = analyzer.generate_samples(
        method=enabled_method_name, N=N, **sample_kwargs
    )
    logger.info(f"✓ Generated {len(samples)} parameter samples")

    # 5. Run Tricys simulation
    output_metrics = analysis_case.get("dependent_variables", [])

    csv_file_path = analyzer.run_tricys_simulations(output_metrics=output_metrics)
    logger.info(f"✓ Parameter file has been generated: {csv_file_path}")

    summary_file = None
    try:
        logger.info("\nAttempting to run Tricys analysis directly...")
        summary_file = analyzer.run_tricys_analysis(
            csv_file_path=csv_file_path, output_metrics=output_metrics
        )
        if summary_file:
            logger.info(f"✓ Tricys analysis completed, result file: {summary_file}")
        else:
            logger.info("⚠️  Tricys analysis result file not found")
            return
    except Exception as e:
        logger.info(f"⚠️  Tricys analysis failed: {e}")
        logger.info("Please check if the model path and configuration are correct")
        return

    # 6. Run SALib analysis from Tricys results
    try:
        logger.info("\nRunning SALib analysis from Tricys results...")
        all_results = analyzer.run_salib_analysis_from_tricys_results(
            sensitivity_summary_csv=summary_file,
            param_bounds=param_bounds,
            output_metrics=output_metrics,
            methods=[enabled_method_name],
            save_dir=os.path.dirname(summary_file),
        )

        logger.info(f"\n✅ SALib {enabled_method_name.upper()} analysis completed!")
        logger.info(
            f"📁 The results have been saved to: {os.path.join(os.path.dirname(summary_file), f'salib_analysis_{enabled_method_name}')}"
        )

        logger.info("\n📈 Brief results:")
        for metric_name, metric_results in all_results.items():
            logger.info(f"\n--- {metric_name} ---")
            if enabled_method_name in metric_results:
                result_data = metric_results[enabled_method_name]
                if enabled_method_name == "sobol":
                    logger.info("🔥 Most sensitive parameters (Sobol ST):")
                    st_values = list(zip(analyzer.problem["names"], result_data["ST"]))
                    st_values.sort(key=lambda x: x[1], reverse=True)
                    for param, st in st_values[:3]:
                        logger.info(f"   {param}: {st:.4f}")
                elif enabled_method_name == "morris":
                    logger.info("📊 Most Sensitive Parameter (Morris μ*):")
                    mu_star_values = list(
                        zip(analyzer.problem["names"], result_data["mu_star"])
                    )
                    mu_star_values.sort(key=lambda x: x[1], reverse=True)
                    for param, mu_star in mu_star_values[:3]:
                        logger.info(f"   {param}: {mu_star:.4f}")
                elif enabled_method_name == "fast":
                    logger.info("⚡ Most Sensitive Parameter (Morris μ*):")
                    st_values = list(zip(analyzer.problem["names"], result_data["ST"]))
                    st_values.sort(key=lambda x: x[1], reverse=True)
                    for param, st in st_values[:3]:
                        logger.info(f"   {param}: {st:.4f}")

        return analyzer, all_results

    except Exception as e:
        logger.error(f"SALib analysis failed: {e}", exc_info=True)
        raise
