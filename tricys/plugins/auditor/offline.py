import fnmatch
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class AuditorConfig:
    enabled: bool = False
    warn_threshold_g: float = 0.5
    kill_threshold_g: float = 50.0
    inventory_patterns: list[str] = field(default_factory=list)
    source_patterns: list[str] = field(default_factory=list)
    leak_patterns: list[str] = field(default_factory=list)
    burn_patterns: list[str] = field(default_factory=list)
    decay_patterns: list[str] = field(default_factory=list)
    cumulative_source_patterns: list[str] = field(default_factory=list)
    cumulative_leak_patterns: list[str] = field(default_factory=list)
    cumulative_burn_patterns: list[str] = field(default_factory=list)
    cumulative_decay_patterns: list[str] = field(default_factory=list)


def parse_auditor_config(config_dict: dict[str, Any]) -> AuditorConfig:
    auditor_data = config_dict.get("auditor", {})
    if not auditor_data:
        return AuditorConfig()

    patterns_data = auditor_data.get("patterns", {})

    return AuditorConfig(
        enabled=bool(auditor_data.get("enabled", False)),
        warn_threshold_g=float(auditor_data.get("warn_threshold_g", 0.5)),
        kill_threshold_g=float(auditor_data.get("kill_threshold_g", 50.0)),
        inventory_patterns=list(patterns_data.get("inventory", [])),
        source_patterns=list(patterns_data.get("sources", [])),
        leak_patterns=list(patterns_data.get("leak", [])),
        burn_patterns=list(patterns_data.get("burn", [])),
        decay_patterns=list(patterns_data.get("decay", patterns_data.get("dcay", []))),
        cumulative_source_patterns=list(patterns_data.get("cumulative_sources", [])),
        cumulative_leak_patterns=list(patterns_data.get("cumulative_leak", [])),
        cumulative_burn_patterns=list(patterns_data.get("cumulative_burn", [])),
        cumulative_decay_patterns=list(patterns_data.get("cumulative_decay", [])),
    )


def audit_dataframe_mass_balance(df: pd.DataFrame, config: AuditorConfig) -> dict:
    """
    Perform mass balance audit directly on a simulation DataFrame.
    Matches variables against configured patterns and calculates integral balance error.
    """
    if df is None or df.empty:
        return {"status": "error", "error": "Empty dataframe provided for audit."}

    if not config.enabled and not (config.inventory_patterns or config.source_patterns):
        return {
            "status": "unconfigured",
            "configured": False,
            "message": "未配置质量守恒审计规则（未定义 inventory / sources 匹配模式）",
        }

    columns = df.columns.tolist()

    # 1. Discovery Phase
    inv_cols = [
        c for c in columns for p in config.inventory_patterns if fnmatch.fnmatch(c, p)
    ]
    src_cols = [
        c for c in columns for p in config.source_patterns if fnmatch.fnmatch(c, p)
    ]
    leak_cols = [
        c for c in columns for p in config.leak_patterns if fnmatch.fnmatch(c, p)
    ]
    burn_cols = [
        c for c in columns for p in config.burn_patterns if fnmatch.fnmatch(c, p)
    ]
    decay_cols = [
        c for c in columns for p in config.decay_patterns if fnmatch.fnmatch(c, p)
    ]

    cum_src_cols = [
        c
        for c in columns
        for p in config.cumulative_source_patterns
        if fnmatch.fnmatch(c, p)
    ]
    cum_leak_cols = [
        c
        for c in columns
        for p in config.cumulative_leak_patterns
        if fnmatch.fnmatch(c, p)
    ]
    cum_burn_cols = [
        c
        for c in columns
        for p in config.cumulative_burn_patterns
        if fnmatch.fnmatch(c, p)
    ]
    cum_decay_cols = [
        c
        for c in columns
        for p in config.cumulative_decay_patterns
        if fnmatch.fnmatch(c, p)
    ]

    # Remove duplicates
    inv_cols = sorted(list(set(inv_cols)))
    src_cols = sorted(list(set(src_cols)))
    leak_cols = sorted(list(set(leak_cols)))
    burn_cols = sorted(list(set(burn_cols)))
    decay_cols = sorted(list(set(decay_cols)))
    cum_src_cols = sorted(list(set(cum_src_cols)))
    cum_leak_cols = sorted(list(set(cum_leak_cols)))
    cum_burn_cols = sorted(list(set(cum_burn_cols)))
    cum_decay_cols = sorted(list(set(cum_decay_cols)))

    if not inv_cols and not src_cols and not cum_src_cols:
        return {
            "status": "no_variables_matched",
            "configured": True,
            "message": "已配置审计规则，但未在输出数据中匹配到相关变量列",
            "warn_threshold_g": config.warn_threshold_g,
        }

    if "time" not in df.columns:
        return {"status": "error", "error": "No 'time' column found in result file."}

    times = df["time"].values

    if inv_cols:
        total_inv = df[inv_cols].sum(axis=1).values
        initial_inv = float(total_inv[0])
        current_inv = float(total_inv[-1])
    else:
        initial_inv = 0.0
        current_inv = 0.0

    cumulative_sources = 0.0
    cumulative_leak = 0.0
    cumulative_burn = 0.0
    cumulative_decay = 0.0

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    if src_cols:
        src_rates = df[src_cols].sum(axis=1).values
        cumulative_sources = float(_trapz(src_rates, times))

    if leak_cols:
        leak_rates = df[leak_cols].sum(axis=1).values
        cumulative_leak = float(_trapz(leak_rates, times))

    if burn_cols:
        burn_rates = df[burn_cols].sum(axis=1).values
        cumulative_burn = float(_trapz(burn_rates, times))

    if decay_cols:
        decay_rates = df[decay_cols].sum(axis=1).values
        cumulative_decay = float(_trapz(decay_rates, times))

    if cum_src_cols:
        cumulative_sources += float(df[cum_src_cols].sum(axis=1).values[-1])

    if cum_leak_cols:
        cumulative_leak += float(df[cum_leak_cols].sum(axis=1).values[-1])

    if cum_burn_cols:
        cumulative_burn += float(df[cum_burn_cols].sum(axis=1).values[-1])

    if cum_decay_cols:
        cumulative_decay += float(df[cum_decay_cols].sum(axis=1).values[-1])

    expected_mass = (
        initial_inv
        + cumulative_sources
        - cumulative_leak
        - cumulative_burn
        - cumulative_decay
    )
    mass_error = current_inv - expected_mass

    return {
        "status": "success",
        "configured": True,
        "warn_threshold_g": config.warn_threshold_g,
        "kill_threshold_g": config.kill_threshold_g,
        "initial_inventory": float(initial_inv),
        "final_inventory": float(current_inv),
        "cumulative_sources": float(cumulative_sources),
        "cumulative_leak": float(cumulative_leak),
        "cumulative_burn": float(cumulative_burn),
        "cumulative_decay": float(cumulative_decay),
        "expected_final_mass": float(expected_mass),
        "mass_balance_error": float(mass_error),
        "discovered_inventory_vars": inv_cols,
        "discovered_source_vars": src_cols + cum_src_cols,
        "discovered_leak_vars": leak_cols + cum_leak_cols,
        "discovered_burn_vars": burn_cols + cum_burn_cols,
        "discovered_decay_vars": decay_cols + cum_decay_cols,
    }


def perform_offline_audit(result_file: str, config: AuditorConfig) -> dict:
    """
    Perform an offline mass balance audit on a simulation result file (CSV or HDF5).
    Reads the file, matches variables against configured patterns, and calculates the integral error.
    """
    try:
        if result_file.endswith(".csv"):
            df = pd.read_csv(result_file)
        elif result_file.endswith(".h5") or result_file.endswith(".hdf5"):
            try:
                df = pd.read_hdf(result_file, "results")
            except KeyError:
                df = pd.read_hdf(result_file, "data")
        else:
            raise ValueError("Unsupported result file format. Must be CSV or HDF5.")
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"Failed to read result file {result_file} for auditing: {e}"
        )
        return {"error": str(e)}

    return audit_dataframe_mass_balance(df, config)
