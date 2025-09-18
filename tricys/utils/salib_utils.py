import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """
    Integrated SALib's Tricys Sensitivity Analyzer

    Supported Analysis Methods:
    - Sobol: Variance-based global sensitivity analysis
    - Morris: Screening-based sensitivity analysis
    - FAST: Fourier Amplitude Sensitivity Test
    - LHS: Latin Hypercube Sampling uncertainty analysis
    """

    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize the analyzer

        Args:
            base_config: Tricys base configuration dictionary
        """
        self.base_config = base_config.copy()
        self.problem = None
        self.parameter_samples = None
        self.simulation_results = None
        self.sensitivity_results = {}

        self._setup_chinese_font()
        self._validate_tricys_config()

    def _setup_chinese_font(self):
        """Set the Chinese font to ensure proper display of Chinese characters in the chart"""
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
                logger.info(f"Using Chinese fonts: {available_font}")
            else:
                logger.warning(
                    "No suitable Chinese font found, which may affect Chinese display"
                )

            plt.rcParams["axes.unicode_minus"] = False

        except Exception as e:
            logger.warning(f"Failed to set Chinese font: {e}, using default font")

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
                f"{method_name}: Found {n_nan} NaN values, using maximum value for imputation"
            )

            valid_values = Y[~nan_indices]

            if len(valid_values) > 0:
                max_value = np.max(valid_values)
                Y_processed = Y.copy()
                Y_processed[nan_indices] = max_value
                return Y_processed
            else:
                logger.error(
                    f"{method_name}: All values are NaN, analysis cannot be performed"
                )
                raise ValueError(
                    f"{method_name}: All simulation results are NaN, sensitivity analysis cannot be performed"
                )
        return Y

    def _validate_tricys_config(self):
        required_keys = {
            "paths": ["package_path"],
            "simulation": ["model_name", "stop_time"],
        }

        for section, keys in required_keys.items():
            if section not in self.base_config:
                logger.warning(
                    f"Missing configuration section: {section}, default values will be used"
                )
                continue

            for key in keys:
                if key not in self.base_config[section]:
                    logger.warning(
                        f"Missing configuration item: {section}.{key}, using default value"
                    )

        package_path = self.base_config.get("paths", {}).get("package_path")
        if package_path and not os.path.exists(package_path):
            logger.warning(
                f"Model file does not exist: {package_path}, which may cause simulation failure"
            )

    def define_problem(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        param_distributions: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Define SALib problem space

        Args:
            param_bounds: Parameter bounds dictionary {'param_name': (min_val, max_val)}
            param_distributions: Parameter distribution type dictionary {'param_name': 'unif'/'norm'/etc}
                                Valid distribution types: 'unif', 'triang', 'norm', 'truncnorm', 'lognorm'

        Returns:
            SALib problem definition dictionary
        """
        if param_distributions is None:
            param_distributions = {name: "unif" for name in param_bounds.keys()}

        valid_dists = ["unif", "triang", "norm", "truncnorm", "lognorm"]
        for name, dist in param_distributions.items():
            if dist not in valid_dists:
                logger.warning(
                    f"The distribution type '{dist}' for parameter {name} is invalid, changed to use 'unif'"
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
            f"Defined a problem space containing {self.problem['num_vars']} parameters:"
        )
        for i, name in enumerate(self.problem["names"]):
            logger.info(
                f"  {name}: {self.problem['bounds'][i]} (distribution: {self.problem['dists'][i]})"
            )

        return self.problem

    def generate_samples(
        self, method: str = "sobol", N: int = 1024, **kwargs
    ) -> np.ndarray:
        """
        Generate parameter samples

        Args:
            method: Sampling method ('sobol', 'morris', 'fast', 'latin')
            N: Number of samples (for Sobol this is the base sample count, actual sample count is N*(2*D+2))
            **kwargs: Method-specific parameters

        Returns:
            Parameter sample array (n_samples, n_params)
        """
        if self.problem is None:
            raise ValueError(
                "You must first call define_problem() to define the problem space."
            )

        logger.info(
            f"Generate samples using the {method} method, base sample count: {N}"
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

        logger.info(f"Successfully generated {actual_samples} samples")

        if self.parameter_samples is not None:
            self.parameter_samples = np.round(self.parameter_samples, decimals=5)
            logger.info(
                "The parameter sample precision has been adjusted to 5 decimal places."
            )

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

        logger.info(f"Target output metrics: {output_metrics}")

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

        logger.info(f"Successfully generated {len(param_data)} parameter samples")
        logger.info(f"Parameter file saved to: {csv_output_path}")
        logger.info(f"Column list: {list(df.columns)}")
        logger.info("Parameter precision: 5 decimal places")
        logger.info(f"Sample statistics:\n{df.describe()}")

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

            from tricys.simulation_analysis import run_simulation

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

                    # Remove parameter sensitivity (correlation coefficient) display
                    # logger.info("\n参数敏感性 (相关系数):")
                    # for i, param_name in enumerate(self.problem["names"]):
                    #     corr = lhs_result["partial_correlations"][i]
                    #     logger.info(f"  {param_name}: {corr:.4f}")

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

            self._save_sensitivity_report(all_results, save_dir)

        except Exception as e:
            logger.warning(f"Failed to save result: {e}")

        logger.info("\n✅ SALib sensitivity analysis completed!")
        logger.info(f"📁 The result has been saved to: {save_dir}")

        return all_results

    def _save_sensitivity_report(self, all_results: Dict[str, Any], save_dir: str):
        """The result has been saved to: {save_dir}"""
        report_file = os.path.join(save_dir, "sensitivity_analysis_report.md")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# SALib Sensitivity Analysis Report\n\n")
            f.write(f"Generation time: {pd.Timestamp.now()}\n\n")

            f.write("## Analyze parameters\n\n")
            if self.problem:
                for i, param_name in enumerate(self.problem["names"]):
                    bounds = self.problem["bounds"][i]
                    f.write(f"- **{param_name}**: [{bounds[0]:.4f}, {bounds[1]:.4f}]\n")
            f.write("\n")

            for metric_name, metric_results in all_results.items():
                f.write(f"## {metric_name} Sensitivity analysis results\n\n")

                if "sobol" in metric_results:
                    f.write("### Sobol敏感性指数\n\n")
                    f.write(
                        "| 参数 | S1 (一阶) | ST (总) | S1置信区间 | ST置信区间 |\n"
                    )
                    f.write("|------|----------|---------|------------|------------|\n")

                    sobol_data = metric_results["sobol"]
                    for i, param_name in enumerate(self.problem["names"]):
                        s1 = sobol_data["S1"][i]
                        st = sobol_data["ST"][i]
                        s1_conf = sobol_data["S1_conf"][i]
                        st_conf = sobol_data["ST_conf"][i]
                        f.write(
                            f"| {param_name} | {s1:.4f} | {st:.4f} | ±{s1_conf:.4f} | ±{st_conf:.4f} |\n"
                        )
                    f.write("\n")

                if "morris" in metric_results:
                    f.write("### Morris敏感性指数\n\n")
                    f.write("| 参数 | μ* (平均绝对效应) | σ (标准差) | μ*置信区间 |\n")
                    f.write("|------|-------------------|------------|------------|\n")

                    morris_data = metric_results["morris"]
                    for i, param_name in enumerate(self.problem["names"]):
                        mu_star = morris_data["mu_star"][i]
                        sigma = morris_data["sigma"][i]
                        mu_star_conf = morris_data["mu_star_conf"][i]
                        f.write(
                            f"| {param_name} | {mu_star:.4f} | {sigma:.4f} | ±{mu_star_conf:.4f} |\n"
                        )
                    f.write("\n")

                if "fast" in metric_results:
                    f.write("### FAST敏感性指数\n\n")
                    f.write("| 参数 | S1 (一阶) | ST (总) |\n")
                    f.write("|------|----------|---------|\n")

                    fast_data = metric_results["fast"]
                    for i, param_name in enumerate(self.problem["names"]):
                        s1 = fast_data["S1"][i]
                        st = fast_data["ST"][i]
                        f.write(f"| {param_name} | {s1:.4f} | {st:.4f} |\n")
                    f.write("\n")

                if "latin" in metric_results:
                    f.write("### LHS不确定性分析结果\n\n")
                    lhs_data = metric_results["latin"]
                    f.write(f"- 均值: {lhs_data['mean']:.4f}\n")
                    f.write(f"- 标准差: {lhs_data['std']:.4f}\n")
                    f.write(f"- 最小值: {lhs_data['min']:.4f}\n")
                    f.write(f"- 最大值: {lhs_data['max']:.4f}\n")
                    f.write(f"- 5%分位数: {lhs_data['percentile_5']:.4f}\n")
                    f.write(f"- 95%分位数: {lhs_data['percentile_95']:.4f}\n\n")

                    # Remove parameter sensitivity (correlation coefficient) data
                    # f.write("#### 参数敏感性 (相关系数)\n\n")
                    # f.write("| 参数 | 相关系数 |\n")
                    # f.write("|------|----------|\n")
                    # for i, param_name in enumerate(self.problem["names"]):
                    #     corr = lhs_data["partial_correlations"][i]
                    #     f.write(f"| {param_name} | {corr:.4f} |\n")
                    # f.write("\n")

        logger.info(f"The sensitivity analysis report has been saved.: {report_file}")

    def plot_sobol_results(
        self,
        save_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        metric_names: List[str] = None,
    ):
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
    ):
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
    ):
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
        figsize: Tuple[int, int] = (15, 10),
        metric_names: List[str] = None,
    ):
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

            # Create a figure with multiple subplots
            plt.figure(figsize=figsize)

            # 1. Distribution histogram
            ax1 = plt.subplot(2, 3, 1)
            ax1.hist(
                self.simulation_results[:, output_index],
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax1.set_xlabel("输出值", fontsize=12)
            ax1.set_ylabel("频率", fontsize=12)
            ax1.set_title(f"输出分布直方图\n{metric_display_name}", fontsize=14, pad=10)
            ax1.grid(True, alpha=0.3)

            # Add statistics text
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

            # 2. Box plot of output
            ax2 = plt.subplot(2, 3, 2)
            ax2.boxplot(
                self.simulation_results[:, output_index],
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
            )
            ax2.set_ylabel("输出值", fontsize=12)
            ax2.set_title(f"输出箱线图\n{metric_display_name}", fontsize=14, pad=10)
            ax2.grid(True, alpha=0.3)

            # 3. Cumulative distribution function
            ax3 = plt.subplot(2, 3, 3)
            sorted_data = np.sort(self.simulation_results[:, output_index])
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax3.plot(sorted_data, y_vals, linewidth=2, color="darkgreen")
            ax3.set_xlabel("输出值", fontsize=12)
            ax3.set_ylabel("累积概率", fontsize=12)
            ax3.set_title(f"累积分布函数\n{metric_display_name}", fontsize=14, pad=10)
            ax3.grid(True, alpha=0.3)

            # Remove parameter sensitivity plots (related coefficient plots)
            # These were in positions (2,3,4), (2,3,5), and (2,3,6) but we only need the first 3 plots
            # So we'll leave the remaining subplots empty or add additional analysis if needed

            plt.tight_layout()
            filename = f'lhs_analysis_{metric_display_name.replace(" ", "_")}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            logger.info(f"LHS分析结果图表已保存: {filename}")

    def save_results(
        self, save_dir: str = None, format: str = "csv", metric_names: List[str] = None
    ):
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


def run_salib_analysis(config: Dict[str, Any]):

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
