from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_salib_problem(
    sampling_specs: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds a standard SALib problem dictionary from variable sampling definitions.

    Expected sampling_specs format:
        {
            "pulseSource.width": {"bounds": [50, 90], "distribution": "unif"},
            "plasma.nf": [0.1, 0.9],  # shorthand list format
        }
    """
    names: List[str] = []
    bounds: List[List[float]] = []
    dists: List[str] = []

    valid_dists = ["unif", "triang", "norm", "truncnorm", "lognorm"]

    for param_name, spec in sampling_specs.items():
        if isinstance(spec, dict):
            param_bounds = spec.get("bounds", [0.0, 1.0])
            dist = spec.get("distribution", "unif")
            if dist not in valid_dists:
                dist = "unif"
        elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
            param_bounds = [float(spec[0]), float(spec[1])]
            dist = "unif"
        else:
            continue

        names.append(param_name)
        bounds.append([float(param_bounds[0]), float(param_bounds[1])])
        dists.append(dist)

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
        "dists": dists,
    }


def generate_salib_samples(
    problem: Dict[str, Any],
    method: str = "sobol",
    N: int = 16,
    calc_second_order: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generates a numpy sample matrix using SALib algorithms."""
    if seed is not None:
        np.random.seed(seed)

    method_clean = method.lower()
    if method_clean in ("sobol", "saltelli"):
        # Prefer modern SALib sobol.sample if available to prevent deprecation warnings
        try:
            from SALib.sample import sobol

            return sobol.sample(
                problem, N, calc_second_order=calc_second_order, seed=seed
            )
        except (ImportError, AttributeError, TypeError):
            from SALib.sample import saltelli

            return saltelli.sample(problem, N, calc_second_order=calc_second_order)

    elif method_clean in ("latin", "lhs"):
        from SALib.sample import latin

        try:
            return latin.sample(problem, N, seed=seed)
        except TypeError:
            return latin.sample(problem, N)

    elif method_clean == "morris":
        from SALib.sample import morris

        try:
            return morris.sample(problem, N, seed=seed)
        except TypeError:
            return morris.sample(problem, N)

    elif method_clean == "fast":
        from SALib.sample import fast_sampler

        try:
            return fast_sampler.sample(problem, N, seed=seed)
        except TypeError:
            return fast_sampler.sample(problem, N)

    else:
        raise ValueError(
            f"Unsupported statistical sampling method: '{method}'. "
            f"Supported methods are: 'sobol', 'saltelli', 'latin', 'lhs', 'morris', 'fast'."
        )


class SamplingEngine:
    """Unified Parameter Sampling and Exploration Engine.

    Encapsulates statistical sampling (Sobol, Latin Hypercube, Morris, FAST)
    as well as full-factorial / grid parameter sweeps into a consistent interface.
    """

    SUPPORTED_STATISTICAL_METHODS = {
        "sobol",
        "saltelli",
        "latin",
        "lhs",
        "morris",
        "fast",
    }
    SUPPORTED_METHODS = SUPPORTED_STATISTICAL_METHODS | {"grid", "sweep"}

    def __init__(
        self,
        specs: Dict[str, Any],
        method: str = "sobol",
        sample_n: int = 16,
        calc_second_order: bool = False,
        seed: Optional[int] = None,
        base_params: Optional[Dict[str, Any]] = None,
    ):
        self.specs = specs or {}
        self.method = method.lower()
        self.sample_n = sample_n
        self.calc_second_order = calc_second_order
        self.seed = seed
        self.base_params = base_params.copy() if base_params else {}

        self.problem: Optional[Dict[str, Any]] = None
        if self.is_statistical():
            self.problem = build_salib_problem(self.specs)

    def is_statistical(self) -> bool:
        return self.method in self.SUPPORTED_STATISTICAL_METHODS

    def is_grid(self) -> bool:
        return self.method in ("grid", "sweep")

    def sample(self) -> np.ndarray:
        """Returns the raw sample matrix as a 2D numpy array for statistical methods."""
        if not self.is_statistical():
            raise ValueError(
                f"sample() is only valid for statistical methods, got: '{self.method}'"
            )
        if not self.problem or self.problem["num_vars"] == 0:
            return np.empty((0, 0))

        return generate_salib_samples(
            problem=self.problem,
            method=self.method,
            N=self.sample_n,
            calc_second_order=self.calc_second_order,
            seed=self.seed,
        )

    def generate_jobs(self) -> List[Dict[str, Any]]:
        """Generates the discrete job parameter dictionaries."""
        if self.is_statistical():
            return self._generate_statistical_jobs()
        elif self.is_grid():
            return self._generate_grid_jobs()
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _generate_statistical_jobs(self) -> List[Dict[str, Any]]:
        if not self.problem or self.problem["num_vars"] == 0:
            return []

        samples = self.sample()
        jobs: List[Dict[str, Any]] = []
        for row in samples:
            job = self.base_params.copy()
            for idx, name in enumerate(self.problem["names"]):
                job[name] = float(row[idx])
            jobs.append(job)

        logger.info(
            f"Generated {len(jobs)} simulation job(s) via {self.method.upper()} sampling "
            f"for {self.problem['num_vars']} parameter(s)"
        )
        return jobs

    def _generate_grid_jobs(self) -> List[Dict[str, Any]]:
        from tricys.core.jobs import generate_simulation_jobs

        merged_grid_specs = self.base_params.copy()
        merged_grid_specs.update(self.specs)

        jobs = generate_simulation_jobs(merged_grid_specs)
        logger.info(
            f"Generated {len(jobs)} simulation job(s) via GRID sweep "
            f"for {len(self.specs)} parameter(s)"
        )
        return jobs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[SamplingEngine]:
        """Factory method to construct a SamplingEngine by auto-detecting config schema."""
        # 1. Schema 2.0 explicit sampling: preprocess.sampling or root-level sampling
        sampling_cfg = config.get("preprocess", {}).get("sampling") or config.get(
            "sampling"
        )
        if sampling_cfg and isinstance(sampling_cfg, dict):
            mode = sampling_cfg.get("mode", "sobol").lower()
            mode_opts = (
                sampling_cfg.get(mode, {})
                if isinstance(sampling_cfg.get(mode), dict)
                else sampling_cfg
            )
            specs = (
                mode_opts.get("parameters")
                or mode_opts.get("variables")
                or sampling_cfg.get("parameters")
                or {}
            )
            sample_n = mode_opts.get("sample_n", sampling_cfg.get("sample_n", 16))
            calc_second_order = mode_opts.get("calc_second_order", False)
            seed = mode_opts.get("seed", sampling_cfg.get("seed"))
            base_params = (
                mode_opts.get("base_params")
                or sampling_cfg.get("base_params")
                or config.get("default_simulation_values")
            )
            if specs:
                return cls(
                    specs=specs,
                    method=mode,
                    sample_n=int(sample_n),
                    calc_second_order=bool(calc_second_order),
                    seed=seed,
                    base_params=base_params,
                )

        # 2. Schema 1.0 Sensitivity Analysis style: sensitivity_analysis.analysis_cases
        sa_cfg = config.get("sensitivity_analysis", {})
        if sa_cfg and isinstance(sa_cfg, dict):
            cases = sa_cfg.get("analysis_cases") or [sa_cfg.get("analysis_case", {})]
            if cases and isinstance(cases, list) and isinstance(cases[0], dict):
                case = cases[0]
                analyzer = case.get("analyzer", {})
                specs = case.get("independent_variable_sampling")
                base_params = case.get("default_simulation_values")

                # Case A: Statistical analyzer present (Sobol, Latin, etc.)
                if isinstance(specs, dict) and analyzer:
                    method = analyzer.get("method", "sobol")
                    sample_n = analyzer.get("sample_N", analyzer.get("sample_n", 16))
                    calc_second_order = analyzer.get("calc_second_order", False)
                    seed = analyzer.get("seed")
                    return cls(
                        specs=specs,
                        method=method,
                        sample_n=int(sample_n),
                        calc_second_order=bool(calc_second_order),
                        seed=seed,
                        base_params=base_params,
                    )

                # Case B: Grid/discrete sampling within analysis_case
                indep_var = case.get("independent_variable")
                sim_params = case.get("simulation_parameters", {})
                if indep_var and isinstance(specs, list):
                    grid_specs = {}
                    if isinstance(indep_var, str):
                        grid_specs[indep_var] = specs
                    elif isinstance(indep_var, list) and len(indep_var) == 1:
                        grid_specs[indep_var[0]] = specs

                    if isinstance(sim_params, dict):
                        grid_specs.update(sim_params)

                    return cls(
                        specs=grid_specs,
                        method="grid",
                        base_params=base_params,
                    )

        # 3. Standard simulation_parameters grid sweep
        sweep_params = config.get("simulation_parameters")
        if sweep_params and isinstance(sweep_params, dict):
            return cls(
                specs=sweep_params,
                method="grid",
            )

        return None


def generate_sampling_jobs(
    sampling_specs: Dict[str, Any],
    method: str = "sobol",
    sample_n: int = 16,
    calc_second_order: bool = False,
    base_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Legacy backward-compatible functional wrapper for sampling job generation."""
    if not sampling_specs:
        return []

    engine = SamplingEngine(
        specs=sampling_specs,
        method=method,
        sample_n=sample_n,
        calc_second_order=calc_second_order,
        base_params=base_params,
        seed=seed,
    )
    return engine.generate_jobs()
