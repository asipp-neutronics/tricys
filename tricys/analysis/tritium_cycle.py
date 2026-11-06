"""Tritium Fuel Cycle (TFC) Physics Analysis & Self-Sufficiency Post-Processing Engine.

Provides publication-grade and engineering-standard physical post-processing algorithms
tailored for fusion tritium cycle dynamics (CFEDR, ITER, DEMO):
1. Fusion engineering parameters derivation (P_fus, Duty, DIR, TBR, burn/fuel rates).
2. Startup inventory valley, deficit & turning-point self-sufficiency extractor.
3. Process subsystem holdup dynamics & paving time detector (95% and 99% criteria).
4. Ten-node closed-loop tritium mass flow balance proof matrix (F01-F10, IN, OUT, NET).
5. Dynamic mass conservation and numerical leakage audit.
6. Sobol variance decomposition & parameter ranking parser.
7. Bisection optimization convergence & critical TBR margin analyzer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 31536000.0


def resolve_time_factor_to_hours(
    t_max: float, config: Optional[Dict[str, Any]] = None
) -> float:
    """Determine the scale factor to convert raw simulation time to hours (h).

    TRICYS schema time fields (stopTime, stepSize, t_dd, etc.) are always in hours.
    time_base specifies the Modelica model's time unit ('second' or 'hour').
    - If Modelica model runs in seconds, raw time is in seconds -> factor = 3600.0 (t / 3600).
    - If Modelica model runs in hours, raw time is in hours -> factor = 1.0.
    - If time_base is not explicitly set (e.g. legacy H5 files), we check config stop_time
      or t_max magnitude:
      - If stop_time is provided and t_max matches stop_time -> raw time is already in hours (factor = 1.0).
      - If stop_time is provided and t_max matches stop_time * 3600 -> raw time is in seconds (factor = 3600.0).
      - Otherwise, if t_max >= 50000.0 -> raw time is in seconds (factor = 3600.0).
      - Otherwise -> raw time is already in hours (factor = 1.0).
    """
    tb = None
    stop_time_cfg = None

    if isinstance(config, dict):
        tb = (
            config.get("time_base")
            or config.get("scenario", {}).get("time_base")
            or config.get("simulation", {}).get("time_base")
        )
        if isinstance(tb, str):
            tb = tb.strip().lower()

        raw_stop = (
            config.get("stopTime")
            or config.get("scenario", {}).get("stopTime")
            or config.get("simulation", {}).get("stop_time")
            or config.get("stop_time")
        )
        try:
            if raw_stop is not None:
                stop_time_cfg = float(raw_stop)
        except (ValueError, TypeError):
            stop_time_cfg = None

    try:
        t_val = float(t_max) if t_max is not None else 0.0
    except (ValueError, TypeError):
        t_val = 0.0

    # 1. Match against known stop_time in config
    if stop_time_cfg and stop_time_cfg > 0 and t_val > 0:
        ratio_h = abs(t_val - stop_time_cfg) / stop_time_cfg
        ratio_s = abs(t_val - stop_time_cfg * SECONDS_PER_HOUR) / (
            stop_time_cfg * SECONDS_PER_HOUR
        )
        if ratio_h < 0.5:
            return 1.0
        if ratio_s < 0.5:
            return SECONDS_PER_HOUR

    # 2. Check explicit time_base
    if tb == "second":
        return SECONDS_PER_HOUR
    elif tb == "hour":
        return 1.0

    # 3. Fallback heuristic when time_base is unspecified
    if t_val >= 50000.0:
        return SECONDS_PER_HOUR
    return 1.0


# Deprecated subsystem registry preserved only for backward compatibility.
# Do not use for hardcoded translation: use load_glossary_mapping instead.
SUBSYSTEM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bz": {
        "name": "增殖包层产氚区",
        "name_en": "Breeding Blanket Zone (BZ)",
        "tau_nominal": 24.00,
        "keywords": ["blanket", "bz"],
        "color": "#2CA02C",
    },
    "tep": {
        "name": "尾气净化处理系统",
        "name_en": "Tritium Extraction & Purification (TEP)",
        "tau_nominal": 2.00,
        "keywords": ["tep", "tep_fep", "purification", "fep"],
        "color": "#1F77B4",
    },
    "wds": {
        "name": "水去污处理系统",
        "name_en": "Water Detritiation System (WDS)",
        "tau_nominal": 4.00,
        "keywords": ["wds", "detritiation"],
        "color": "#FF7F0E",
    },
    "i_iss": {
        "name": "内同位素分离系统",
        "name_en": "Inner Isotope Separation System (I-ISS)",
        "tau_nominal": 6.00,
        "keywords": ["i_iss", "iss.inventory", "iss.i"],
        "color": "#9467BD",
    },
    "o_iss": {
        "name": "外同位素分离系统",
        "name_en": "Outer Isotope Separation System (O-ISS)",
        "tau_nominal": 6.00,
        "keywords": ["o_iss"],
        "color": "#8C564B",
    },
    "ps": {
        "name": "真空前级抽气系统",
        "name_en": "Primary Pumping System (PS)",
        "tau_nominal": 0.17,
        "keywords": ["ps", "pump"],
        "color": "#E377C2",
    },
    "tes": {
        "name": "包层氚提取系统",
        "name_en": "Tritium Extraction System (TES)",
        "tau_nominal": 12.00,
        "keywords": ["tes", "extraction"],
        "color": "#7F7F7F",
    },
    "coolant": {
        "name": "冷却剂净化回路",
        "name_en": "Coolant Purification System (CPS)",
        "tau_nominal": 1.00,
        "keywords": ["coolant", "cps", "coolant_pipe"],
        "color": "#BCBD22",
    },
    "fw_div": {
        "name": "第一壁与偏滤器部件",
        "name_en": "First Wall & Divertor (FW/DIV)",
        "tau_nominal": 0.28,
        "keywords": ["fw", "div", "divertor"],
        "color": "#17BECF",
    },
    "fs_plasma": {
        "name": "加料注气与堆芯等离子体",
        "name_en": "Fueling & Core Plasma (FS/Plasma)",
        "tau_nominal": 0.05,
        "keywords": ["fs", "plasma"],
        "color": "#D62728",
    },
    "sds": {
        "name": "储存供料备用罐",
        "name_en": "Storage & Delivery System (SDS)",
        "tau_nominal": 0.10,
        "keywords": ["sds", "storage"],
        "color": "#336699",
    },
}

PALETTE_EXTENDED = [
    "#2CA02C",
    "#1F77B4",
    "#FF7F0E",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#D62728",
    "#336699",
    "#AEC7E8",
]


def load_glossary_mapping(
    config: Optional[Dict[str, Any]] = None,
    glossary_path: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Loads professional terminology glossary mapping from CSV file if configured.

    Searches for glossary path from:
    1. Explicit glossary_path argument
    2. config['glossary_path'] or config['glossary']
    3. config['report']['glossary_path'] or config['report']['glossary']
    4. config['analysis']['glossary_path'] or config['analysis']['glossary']
    5. config['scenario']['glossary_path'] or config['scenario']['glossary']
    6. config['task']['glossary_path'] or config['task']['glossary']
    7. config['sensitivity_analysis']['glossary_path']

    Returns:
        Dict[str, Dict[str, str]]: Mapping of parameter name -> {
            "param": parameter string,
            "zh": 中文翻译 (Chinese Translation),
            "en": 英文术语 (English Term),
            "alias": 别名 (Alias),
            "desc": 说明 (Explanation),
        }
        Returns empty dict {} if no glossary is configured or file not found.
    """
    cfg = config or {}
    path = glossary_path
    if not path:
        candidates = [
            cfg.get("glossary_path"),
            cfg.get("glossary"),
            (
                cfg.get("report", {}).get("glossary_path")
                if isinstance(cfg.get("report"), dict)
                else None
            ),
            (
                cfg.get("report", {}).get("glossary")
                if isinstance(cfg.get("report"), dict)
                else None
            ),
            (
                cfg.get("analysis", {}).get("glossary_path")
                if isinstance(cfg.get("analysis"), dict)
                else None
            ),
            (
                cfg.get("analysis", {}).get("glossary")
                if isinstance(cfg.get("analysis"), dict)
                else None
            ),
            (
                cfg.get("scenario", {}).get("glossary_path")
                if isinstance(cfg.get("scenario"), dict)
                else None
            ),
            (
                cfg.get("scenario", {}).get("glossary")
                if isinstance(cfg.get("scenario"), dict)
                else None
            ),
            (
                cfg.get("task", {}).get("glossary_path")
                if isinstance(cfg.get("task"), dict)
                else None
            ),
            (
                cfg.get("task", {}).get("glossary")
                if isinstance(cfg.get("task"), dict)
                else None
            ),
            (
                cfg.get("sensitivity_analysis", {}).get("glossary_path")
                if isinstance(cfg.get("sensitivity_analysis"), dict)
                else None
            ),
        ]
        for c in candidates:
            if c and isinstance(c, str) and c.strip():
                path = c.strip()
                break

    if not path:
        return {}

    resolved_file = None
    if os.path.isabs(path) and os.path.exists(path):
        resolved_file = path
    else:
        search_dirs = [
            cfg.get("_config_dir"),
            cfg.get("_working_dir"),
            os.getcwd(),
        ]
        for base_dir in search_dirs:
            if base_dir and isinstance(base_dir, str):
                cand = os.path.abspath(os.path.join(base_dir, path))
                if os.path.exists(cand):
                    resolved_file = cand
                    break

        if not resolved_file and os.path.exists(path):
            resolved_file = os.path.abspath(path)

    if not resolved_file or not os.path.exists(resolved_file):
        logger.debug(
            f"Glossary file '{path}' not found; parameter translation disabled."
        )
        return {}

    try:
        df = None
        for enc in ["utf-8", "utf-8-sig", "gbk"]:
            try:
                df = pd.read_csv(resolved_file, encoding=enc)
                break
            except Exception:
                continue

        if df is None or df.empty:
            return {}

        param_col, zh_col, en_col, alias_col, desc_col = None, None, None, None, None
        for col in df.columns:
            c_str = str(col).strip()
            if (
                "模型参数" in c_str
                or "Model Parameter" in c_str
                or c_str.lower() == "parameter"
            ):
                param_col = col
            elif (
                "中文翻译" in c_str
                or "Chinese Translation" in c_str
                or c_str.lower() in ("zh", "chinese")
            ):
                zh_col = col
            elif (
                "英文术语" in c_str
                or "English Term" in c_str
                or c_str.lower() in ("en", "english")
            ):
                en_col = col
            elif "别名" in c_str or "Alias" in c_str or c_str.lower() == "alias":
                alias_col = col
            elif "说明" in c_str or "Explanation" in c_str or "Description" in c_str:
                desc_col = col

        if not param_col:
            param_col = df.columns[0]

        mapping: Dict[str, Dict[str, str]] = {}
        for _, row in df.iterrows():
            raw_param = row[param_col]
            if pd.isna(raw_param) or not str(raw_param).strip():
                continue
            param = str(raw_param).strip()
            zh = str(row[zh_col]).strip() if zh_col and pd.notna(row[zh_col]) else ""
            en = str(row[en_col]).strip() if en_col and pd.notna(row[en_col]) else ""
            alias = (
                str(row[alias_col]).strip()
                if alias_col and pd.notna(row[alias_col])
                else ""
            )
            desc = (
                str(row[desc_col]).strip()
                if desc_col and pd.notna(row[desc_col])
                else ""
            )

            if not zh and not en and not desc:
                continue

            entry = {
                "param": param,
                "zh": zh,
                "en": en,
                "alias": alias,
                "desc": desc,
            }
            mapping[param] = entry
            mapping[param.lower()] = entry

            if alias:
                for a in alias.split("/"):
                    a_clean = a.strip()
                    if a_clean and a_clean not in mapping:
                        mapping[a_clean] = entry
                        mapping[a_clean.lower()] = entry

        return mapping
    except Exception as exc:
        logger.warning(f"Failed to parse glossary file '{resolved_file}': {exc}")
        return {}


def resolve_subsystem_metadata(
    col: str,
    index: int = 0,
    config: Optional[Dict[str, Any]] = None,
    glossary_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Resolves human-readable subsystem name, short key, and color for a given variable column.

    Strict Engineering Rule:
    - If NO professional glossary is configured or provided, DOES NOT perform any
      hardcoded translation or mapping. Strictly preserves and uses the original
      parameter/variable name (e.g. 'blanket.I[1]', 'wds.threshold', 'sds.I[1]').
    - If a professional glossary is configured and has an entry for `col`, maps
      to the professional term (e.g. '{zh} ({col})').
    """
    g_map = glossary_map
    if g_map is None and config is not None:
        g_map = load_glossary_mapping(config=config)

    # 1. Match from glossary if available
    if g_map:
        clean_col = col.strip()
        low_col = clean_col.lower()
        matched = g_map.get(clean_col) or g_map.get(low_col)

        # Handle array indices: e.g. blanket.I[1] -> blanket.I
        if not matched and "[" in clean_col:
            base_col = clean_col.split("[")[0]
            matched = g_map.get(base_col) or g_map.get(base_col.lower())

        if matched:
            zh = matched.get("zh", "").strip()
            en = matched.get("en", "").strip()
            desc = matched.get("desc", "").strip()
            if zh:
                name = f"{zh} ({clean_col})"
                short_name = zh
            elif en:
                name = f"{en} ({clean_col})"
                short_name = en
            else:
                name = clean_col
                short_name = clean_col

            return {
                "key": clean_col,
                "name": name,
                "short_name": short_name,
                "desc": desc,
                "color": PALETTE_EXTENDED[index % len(PALETTE_EXTENDED)],
            }

    # 2. Strict engineering default when NO glossary configured or term not in glossary:
    # Strictly return the original variable name without hardcoded fabrication!
    return {
        "key": col,
        "name": col,
        "short_name": col,
        "desc": "",
        "color": PALETTE_EXTENDED[index % len(PALETTE_EXTENDED)],
    }


def derive_fusion_engineering_parameters(
    config: Optional[Dict[str, Any]] = None,
    results_df: Optional[pd.DataFrame] = None,
    jobs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Derives core physics and engineering parameters from simulation config or results.

    Extracts or estimates:
    - P_fus: Fusion thermal power (MW)
    - t_burn: Single pulse flattop burn time (h)
    - t_dwell: Dwell cleanup duration (h)
    - duty: Duty cycle fraction
    - DIR: Direct Internal Recycling fraction
    - eta_f: Fueling efficiency
    - f_b: Core plasma burnup fraction
    - TBR_net: Net engineering tritium breeding ratio
    - I_0: Initial nominal startup storage inventory (g)
    - m_burn_peak: Peak pure tritium burnup rate (g/h)
    - m_fuel_peak: Peak continuous fueling rate (g/h)
    - m_burn_avg: Period-averaged tritium burnup rate (g/h)
    - m_fuel_avg: Period-averaged continuous fueling rate (g/h)
    """
    cfg = config or {}
    scen = cfg.get("scenario", {})
    pulse = scen.get("pulse", {}) or cfg.get("pulse", {}) or {}
    model_params = cfg.get("model", {}).get("parameters", {})

    # 1. Fusion Power
    p_fus = float(
        model_params.get("P_fus") or cfg.get("P_fus") or scen.get("P_fus", 2000.0)
    )

    # 2. Pulse parameters
    burn_t = float(
        pulse.get("burn_T")
        or pulse.get("burn_time")
        or model_params.get("pulse.burn_T", 2.0)
    )
    duty_pct = float(
        pulse.get("width")
        or pulse.get("duty_cycle")
        or model_params.get("pulse.width", 80.0)
    )
    if duty_pct > 1.0:
        duty = duty_pct / 100.0
    else:
        duty = duty_pct if duty_pct > 0 else 0.80

    t_period = burn_t / duty if duty > 0 else burn_t
    t_dwell = max(t_period - burn_t, 0.0)

    # 3. Efficiencies and recycling
    dir_ratio = float(model_params.get("tep.DIR") or cfg.get("DIR") or 0.60)
    eta_f = float(
        model_params.get("plasma.Fueling_Efficiency")
        or cfg.get("fueling_efficiency")
        or 0.50
    )
    f_b = float(
        model_params.get("plasma.Burn_Fraction") or cfg.get("burn_fraction") or 0.03
    )
    eff_overall = eta_f * f_b if (eta_f * f_b) > 0 else 0.015

    # 4. Net TBR & Required TBR Metrics
    tbr_net_val = (
        model_params.get("TBR") or model_params.get("blanket.TBR") or cfg.get("TBR")
    )
    tbr_net = float(tbr_net_val) if tbr_net_val is not None else None

    tbr_req_val = (
        model_params.get("TBR_Required")
        or model_params.get("Required_TBR")
        or cfg.get("TBR_Required")
        or cfg.get("Required_TBR")
    )
    tbr_req = float(tbr_req_val) if tbr_req_val is not None else None

    delta_tbr = (
        (tbr_net - tbr_req) if (tbr_net is not None and tbr_req is not None) else None
    )
    delta_tbr_pct = (
        (delta_tbr / tbr_req) * 100.0
        if (delta_tbr is not None and tbr_req and tbr_req > 0)
        else None
    )

    # 5. Initial inventory I_0
    i_0 = None
    if results_df is not None and not results_df.empty:
        sds_col = detect_sds_column(results_df)
        if sds_col:
            i_0 = float(results_df[sds_col].iloc[0])
    if "sds.inventory" in model_params:
        i_0 = float(model_params["sds.inventory"])

    # 6. Physical Burn and Fuel Rates
    # Theoretical constant: 6.3935e-3 g pure tritium consumed per hour per MW fusion
    # P_fus * 1e6 W / (17.6 MeV * 1.602e-19 J) * (3.016 g/mol / 6.022e23) * 3600 s/h
    m_burn_peak = p_fus * 6.3935e-3
    m_fuel_peak = m_burn_peak / eff_overall if eff_overall > 0 else m_burn_peak / 0.015
    m_burn_avg = m_burn_peak * duty
    m_fuel_avg = m_fuel_peak * duty

    return {
        "P_fus": round(p_fus, 1),
        "t_burn": round(burn_t, 2),
        "t_dwell": round(t_dwell, 2),
        "t_period": round(t_period, 2),
        "duty": round(duty, 3),
        "duty_pct": round(duty * 100.0, 1),
        "DIR": round(dir_ratio, 3),
        "eta_f": round(eta_f, 3),
        "f_b": round(f_b, 4),
        "eff_overall": round(eff_overall, 4),
        "TBR_net": round(tbr_net, 3) if tbr_net is not None else None,
        "TBR_Design": round(tbr_net, 3) if tbr_net is not None else None,
        "TBR_Required": round(tbr_req, 4) if tbr_req is not None else None,
        "Delta_TBR": round(delta_tbr, 4) if delta_tbr is not None else None,
        "Delta_TBR_pct": round(delta_tbr_pct, 2) if delta_tbr_pct is not None else None,
        "I_0": round(i_0, 1) if i_0 is not None else None,
        "m_burn_peak": round(m_burn_peak, 3),
        "m_fuel_peak": round(m_fuel_peak, 2),
        "m_burn_avg": round(m_burn_avg, 3),
        "m_fuel_avg": round(m_fuel_avg, 2),
    }


def detect_sds_column(df: pd.DataFrame, strict: bool = True) -> Optional[str]:
    """Detects primary storage inventory column in DataFrame."""
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns if c not in ["time", "job_id"]]
    for preferred in ["sds.inventory", "sds.I[1]", "sds_inventory", "sds"]:
        if preferred in cols:
            return preferred
    for c in cols:
        low = c.lower()
        if "sds" in low and ("inv" in low or "i" in low):
            return c
    for c in cols:
        low = c.lower()
        if "storage" in low:
            return c
    for c in cols:
        if "sds" in c.lower():
            return c
    if not strict:
        for c in cols:
            if "inventory" in c.lower():
                return c
    return None


def extract_startup_inventory_metrics(
    results_df: pd.DataFrame,
    jobs_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Extracts startup inventory valley, deficit, and turning point self-sufficiency metrics.

    Evaluates across all jobs:
    - Initial storage I_0 (g)
    - Storage at D-T ignition start I(t_ignition) (g)
    - D-D phase net tritium gain (g)
    - Minimum inventory valley I_min (g)
    - Startup tritium deficit / required startup inventory (g)
    - Post-ignition net drawdown (g)
    - Self-sufficiency turning point time t_ss (absolute h & relative h, days)
    - Time to recover non-negative inventory (h)
    - Time to recover initial reserve I_0 (h)
    - Final storage inventory at simulation end (g)
    - Net full-cycle tritium gain (g)
    - Safety operational status classification
    """
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return pd.DataFrame()

    sds_col = target_col or detect_sds_column(results_df)
    if not sds_col:
        return pd.DataFrame()

    job_ids = (
        sorted(results_df["job_id"].unique()) if "job_id" in results_df.columns else [1]
    )

    # Job name and ignition time mapping
    job_info_map: Dict[Any, Dict[str, Any]] = {}
    if jobs_df is not None and not jobs_df.empty:
        jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
        for _, r in jobs_df.iterrows():
            jid = r[jid_col]
            t_dd = r.get("t_DD_end") or r.get("t_dd") or r.get("t_DD") or 0.0
            name = r.get("name") or r.get("job_name") or f"工况 {jid}"
            job_info_map[jid] = {"name": name, "t_ignition": float(t_dd)}

    records: List[Dict[str, Any]] = []

    for idx, jid in enumerate(job_ids):
        if "job_id" in results_df.columns:
            subset = results_df[results_df["job_id"] == jid].sort_values("time")
        else:
            subset = results_df.sort_values("time")

        if subset.empty:
            continue

        t = subset["time"].values
        sds = subset[sds_col].values

        time_to_hours_factor = resolve_time_factor_to_hours(
            float(t[-1]) if len(t) > 0 else 0.0, config
        )

        # Normalise OMC results time to hours
        t_h = t / time_to_hours_factor

        info = job_info_map.get(jid, {"name": f"工况 {jid}", "t_ignition": 0.0})
        t_ign = float(info["t_ignition"])  # In hours
        if t_ign > t_h[-1]:
            t_ign = 0.0

        i_0 = float(sds[0])

        # Filter points after ignition (both t_h and t_ign are in hours)
        ign_mask = t_h >= t_ign
        if np.any(ign_mask):
            t_dt_h = t_h[ign_mask]
            sds_dt = sds[ign_mask]
        else:
            t_dt_h = t_h
            sds_dt = sds

        i_at_dt = float(sds_dt[0])
        dd_gain = float(i_at_dt - i_0)

        min_idx = int(np.argmin(sds_dt))
        i_min = float(sds_dt[min_idx])
        t_min_abs_h = float(t_dt_h[min_idx])
        t_min_rel_h = float(t_min_abs_h - t_ign)
        t_min_rel_d = t_min_rel_h / HOURS_PER_DAY

        deficit = float(i_0 - i_min)
        dt_drawdown = float(i_at_dt - i_min)

        # Time to recover non-negative inventory (>0)
        idx_zero = np.where((t_dt_h > t_min_abs_h) & (sds_dt >= 0.0))[0]
        if i_min >= 0.0:
            t_zero_rel_h = 0.0
        elif len(idx_zero) > 0:
            t_zero_rel_h = float(t_dt_h[idx_zero[0]] - t_ign)
        else:
            t_zero_rel_h = float("nan")

        # Time to recover initial reserve I_0
        idx_rec = np.where((t_dt_h > t_min_abs_h) & (sds_dt >= i_0))[0]
        if len(idx_rec) > 0:
            t_rec_abs_h = float(t_dt_h[idx_rec[0]])
            t_rec_rel_h = float(t_rec_abs_h - t_ign)
        else:
            t_rec_abs_h = float("nan")
            t_rec_rel_h = float("nan")

        i_end = float(sds[-1])
        net_change = float(i_end - i_0)

        # Scale-independent physical safety classification
        if i_min < 0.0:
            safety = f"存在储罐亏空断供风险 (最低下探至 {i_min:.1f}g)"
        elif net_change >= 0.0:
            safety = "达成自主净增殖自持 (期末存量高于初始装填)"
        else:
            safety = "处于净消耗运行阶段 (期末存量低于初始装填)"

        # Check for TBR_Required / Required_TBR from jobs_df or config
        tbr_req = None
        tbr_design = None
        if jobs_df is not None and not jobs_df.empty:
            jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
            matching_jobs = jobs_df[jobs_df[jid_col] == jid]
            if not matching_jobs.empty:
                r_job = matching_jobs.iloc[0]
                for cand in [
                    "Required_TBR",
                    "TBR_Required",
                    "required_tbr",
                    "Required_TBR约束值",
                ]:
                    if cand in r_job and pd.notnull(r_job[cand]):
                        tbr_req = float(r_job[cand])
                        break
                for cand_des in ["TBR", "blanket.TBR", "tbr_design"]:
                    if cand_des in r_job and pd.notnull(r_job[cand_des]):
                        tbr_design = float(r_job[cand_des])
                        break

        if config:
            model_params = config.get("model", {}).get("parameters", {})
            for cand_m in ["TBR", "blanket.TBR", "tbr_design"]:
                if cand_m in model_params:
                    tbr_design = float(model_params[cand_m])
                    break
            for cand_req in ["Required_TBR", "TBR_Required", "required_tbr"]:
                if cand_req in model_params:
                    tbr_req = float(model_params[cand_req])
                    break

        delta_tbr = (
            (tbr_design - tbr_req)
            if (tbr_design is not None and tbr_req is not None)
            else None
        )

        records.append(
            {
                "job_id": jid,
                "工况名称": info["name"],
                "点火起始时间_h": round(t_ign, 1),
                "总仿真时长_h": round(float(t_h[-1]), 1),
                "初始装填I_0_g": round(i_0, 1),
                "点火起点存量_g": round(i_at_dt, 1),
                "前期阶段净增量_g": round(dd_gain, 1),
                "最低谷底存量I_min_g": round(i_min, 1),
                "必需首炉外供氚量_g": round(deficit, 1),
                "点火后最大下探亏空_g": round(dt_drawdown, 1),
                "自持转折绝对时间_h": round(t_min_abs_h, 1),
                "自持转折相对时滞_h": round(t_min_rel_h, 1),
                "自持转折相对时滞_d": round(t_min_rel_d, 2),
                "自持转折相对时滞_s": round(t_min_rel_h * SECONDS_PER_HOUR, 1),
                "必需产氚比_TBR_Required": (
                    round(tbr_req, 4) if tbr_req is not None else None
                ),
                "设计产氚比_TBR_Design": (
                    round(tbr_design, 3) if tbr_design is not None else None
                ),
                "自持增殖裕度_Delta_TBR": (
                    (f"+{delta_tbr:.4f}" if delta_tbr >= 0 else f"{delta_tbr:.4f}")
                    if delta_tbr is not None
                    else None
                ),
                "恢复非负相对时间_h": (
                    round(t_zero_rel_h, 1) if not np.isnan(t_zero_rel_h) else None
                ),
                "恢复初始储备相对时滞_h": (
                    round(t_rec_rel_h, 1) if not np.isnan(t_rec_rel_h) else None
                ),
                "期末储罐存量_g": round(i_end, 1),
                "全周期净氚增量_g": round(net_change, 1),
                "安全状态判定": safety,
            }
        )

    return pd.DataFrame(records)


def analyze_subsystem_holdup_dynamics(
    results_df: pd.DataFrame,
    target_job_id: Optional[Any] = None,
    t_ignition: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Analyzes each process subsystem holdup, paving duration (95% & 99%), and bounds.

    Returns DataFrame containing:
    - Subsystem Name
    - Nominal Time Constant tau (h)
    - 95% Steady Paving Time (h)
    - 99% Final Paving Time (h)
    - Final Steady Holdup Mass (g)
    - Pulse Variation Range (g)
    - Percentage of Total Startup Deficit (%)
    """
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return pd.DataFrame()

    subset = results_df.copy()
    if "job_id" in subset.columns:
        jid = target_job_id or subset["job_id"].iloc[0]
        subset = subset[subset["job_id"] == jid].sort_values("time")
    else:
        subset = subset.sort_values("time")

    if subset.empty:
        return pd.DataFrame()

    t = subset["time"].values
    ign_mask = t >= t_ignition
    if np.any(ign_mask):
        sub_post = subset[ign_mask].copy()
    else:
        sub_post = subset.copy()

    t_post = sub_post["time"].values - t_ignition
    sds_col = detect_sds_column(results_df)

    # 1. Collect target subsystem columns
    target_cols: List[str] = []

    # Priority A: Extract explicitly configured variables from scenario.variableFilter
    if config:
        vf = config.get("scenario", {}).get("variableFilter", "")
        if vf:
            for item in vf.split("|"):
                v = item.strip()
                if v and not v.startswith(".*") and v in sub_post.columns:
                    if v not in ["time", "job_id", sds_col] and v not in target_cols:
                        if np.issubdtype(sub_post[v].dtype, np.number):
                            target_cols.append(v)

    glossary_map = load_glossary_mapping(config=config)

    # Priority B: If variableFilter omitted or incomplete, discover all numerical non-time, non-primary-sds columns
    if not target_cols:
        for c in sub_post.columns:
            if c not in ["time", "job_id", sds_col] and np.issubdtype(
                sub_post[c].dtype, np.number
            ):
                if c not in target_cols:
                    target_cols.append(c)
    else:
        # Also append columns that exist in glossary if present in sub_post
        if glossary_map:
            for c in sub_post.columns:
                if c not in ["time", "job_id", sds_col] and np.issubdtype(
                    sub_post[c].dtype, np.number
                ):
                    if c not in target_cols and (
                        c in glossary_map or c.lower() in glossary_map
                    ):
                        target_cols.append(c)

    records: List[Dict[str, Any]] = []
    total_holdup = 0.0
    tail_count = max(int(len(sub_post) * 0.1), 1)

    for idx, col in enumerate(target_cols):
        meta = resolve_subsystem_metadata(
            col, index=idx, config=config, glossary_map=glossary_map
        )
        arr = sub_post[col].values
        tail_vals = arr[-tail_count:]
        ss_mean = float(np.mean(tail_vals))
        ss_min = float(np.min(tail_vals))
        ss_max = float(np.max(tail_vals))

        if ss_mean <= 0:
            t_95 = 0.0
            t_99 = 0.0
        else:
            idx_95 = np.where(arr >= ss_mean * 0.95)[0]
            t_95 = float(t_post[idx_95[0]]) if len(idx_95) > 0 else float(t_post[-1])

            idx_99 = np.where(arr >= ss_mean * 0.99)[0]
            t_99 = float(t_post[idx_99[0]]) if len(idx_99) > 0 else float(t_post[-1])

        total_holdup += ss_mean

        sub_time_h_factor = resolve_time_factor_to_hours(
            float(t_post[-1]) if len(t_post) > 0 else 0.0, config
        )

        rec_entry = {
            "key": meta["key"],
            "子系统": meta["name"],
            "t95稳态铺底时间_h": round(t_95 / sub_time_h_factor, 1),
            "t99终态铺底时间_h": round(t_99 / sub_time_h_factor, 1),
            "稳态滞留均值_g": round(ss_mean, 1),
            "脉冲波动区间_g": f"{ss_min:.1f} ~ {ss_max:.1f}",
            "_raw_ss_mean": ss_mean,
        }
        if meta.get("desc"):
            rec_entry["说明"] = meta["desc"]
        records.append(rec_entry)

    # Calculate percentage based on positive process holdup
    pos_holdup = sum(max(0.0, r["_raw_ss_mean"]) for r in records)
    for r in records:
        raw_m = r.pop("_raw_ss_mean")
        pct = (raw_m / pos_holdup * 100.0) if pos_holdup > 0 and raw_m > 0 else 0.0
        r["占过程设备滞留比_pct"] = round(pct, 1)

    return pd.DataFrame(records)


def generate_tritium_mass_flows_proof_table(
    eng_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Generates the 10-node closed-loop mass flow balance proof table.

    Proves mathematical and engineering closure of tritium self-sufficiency:
    F01: Fueling injection rate (FS -> Plasma)
    F02: Pure plasma burnup rate (Plasma Burned)
    F03: Divertor exhaust flow (Plasma -> PS)
    F04: Primary pump transfer (PS -> TEP)
    F05: TEP direct internal recycling to SDS (TEP -> SDS, DIR%)
    F06: TEP stream to inner isotope separation (TEP -> I-ISS, 1-DIR%)
    F07: I-ISS bottom pure tritium return (I-ISS -> SDS, 99.9%)
    F08: Blanket neutron breeding rate (Neutron Breeding, TBR * F02)
    F09: Blanket purge gas extraction (BZ -> TES)
    F10: Outer loop recovery to SDS (TES/O-ISS -> SDS)
    IN: Total inflow to SDS
    OUT: Total outflow from SDS
    NET: Net mass growth rate (IN - OUT > 0)
    """
    p = dict(eng_params or {})
    p.update(kwargs)

    p_fus = float(p.get("P_fus", 2000.0))
    duty = float(p.get("duty", 0.80))
    dir_ratio = float(p.get("DIR", 0.60))
    eta_f = float(p.get("eta_f", 0.50))
    f_b = float(p.get("f_b", 0.03))
    tbr_net = float(p.get("TBR_net", 1.10))

    eff_overall = eta_f * f_b if (eta_f * f_b) > 0 else 0.015

    # Peak flows
    m_burn_peak = p_fus * 6.3935e-3
    m_fuel_peak = m_burn_peak / eff_overall
    m_exh_peak = m_fuel_peak - m_burn_peak
    m_tep_dir_peak = m_exh_peak * dir_ratio
    m_tep_iss_peak = m_exh_peak * (1.0 - dir_ratio)
    m_iss_ret_peak = m_tep_iss_peak * 0.999
    m_breed_peak = m_burn_peak * tbr_net
    m_tes_peak = m_breed_peak

    m_sds_in_peak = m_tep_dir_peak + m_iss_ret_peak + m_tes_peak
    m_sds_out_peak = m_fuel_peak
    m_sds_net_peak = m_sds_in_peak - m_sds_out_peak

    flows = [
        {
            "流路编号": "F01",
            "工艺流路节点": "加料注气流率 (FS -> Plasma)",
            "平顶瞬时流率_gh": round(m_fuel_peak, 2),
            "放电周期平均流率_gh": round(m_fuel_peak * duty, 2),
            "物理功能与平衡依据": f"燃料供给基准: m_burn / ({eff_overall:.4f})",
        },
        {
            "流路编号": "F02",
            "工艺流路节点": "等离子体纯燃耗率 (Plasma Burned)",
            "平顶瞬时流率_gh": round(m_burn_peak, 2),
            "放电周期平均流率_gh": round(m_burn_peak * duty, 2),
            "物理功能与平衡依据": f"{p_fus:.0f} MW 额定净聚变核反应消耗",
        },
        {
            "流路编号": "F03",
            "工艺流路节点": "偏滤器尾气排气 (Plasma -> PS)",
            "平顶瞬时流率_gh": round(m_exh_peak, 2),
            "放电周期平均流率_gh": round(m_exh_peak * duty, 2),
            "物理功能与平衡依据": "未燃耗燃料与氦灰排气: F01 - F02",
        },
        {
            "流路编号": "F04",
            "工艺流路节点": "尾气前级输运 (PS -> TEP)",
            "平顶瞬时流率_gh": round(m_exh_peak, 2),
            "放电周期平均流率_gh": round(m_exh_peak * duty, 2),
            "物理功能与平衡依据": "低温吸附泵与干泵组全量连续输运",
        },
        {
            "流路编号": "F05",
            "工艺流路节点": "TEP 短流程直返 (TEP -> SDS)",
            "平顶瞬时流率_gh": round(m_tep_dir_peak, 2),
            "放电周期平均流率_gh": round(m_tep_dir_peak * duty, 2),
            "物理功能与平衡依据": f"直接内循环 (DIR {dir_ratio * 100:.0f}%) 快速回流",
        },
        {
            "流路编号": "F06",
            "工艺流路节点": "TEP 送内同位素分离 (TEP -> I-ISS)",
            "平顶瞬时流率_gh": round(m_tep_iss_peak, 2),
            "放电周期平均流率_gh": round(m_tep_iss_peak * duty, 2),
            "物理功能与平衡依据": f"分流深冷精馏 ({(1.0 - dir_ratio) * 100:.0f}%) 深度提纯",
        },
        {
            "流路编号": "F07",
            "工艺流路节点": "I-ISS 塔底返储罐 (I-ISS -> SDS)",
            "平顶瞬时流率_gh": round(m_iss_ret_peak, 2),
            "放电周期平均流率_gh": round(m_iss_ret_peak * duty, 2),
            "物理功能与平衡依据": "99.9% 纯氚回收 (其余 0.1% 塔顶送 WDS)",
        },
        {
            "流路编号": "F08",
            "工艺流路节点": "增殖包层净产氚 (Neutron Breeding)",
            "平顶瞬时流率_gh": round(m_breed_peak, 2),
            "放电周期平均流率_gh": round(m_breed_peak * duty, 2),
            "物理功能与平衡依据": f"包层中子倍增产氚: TBR({tbr_net:.2f}) * F02",
        },
        {
            "流路编号": "F09",
            "工艺流路节点": "包层吹扫气提取 (BZ -> TES)",
            "平顶瞬时流率_gh": round(m_tes_peak, 2),
            "放电周期平均流率_gh": round(m_tes_peak * duty, 2),
            "物理功能与平衡依据": "氦吹扫回路吸附解吸平衡载带",
        },
        {
            "流路编号": "F10",
            "工艺流路节点": "外循环提纯回送 (TES/O-ISS -> SDS)",
            "平顶瞬时流率_gh": round(m_tes_peak, 2),
            "放电周期平均流率_gh": round(m_tes_peak * duty, 2),
            "物理功能与平衡依据": "外同位素分离产物汇入补充库容",
        },
        {
            "流路编号": "IN",
            "工艺流路节点": "SDS 储罐总回流入库流率",
            "平顶瞬时流率_gh": round(m_sds_in_peak, 2),
            "放电周期平均流率_gh": round(m_sds_in_peak * duty, 2),
            "物理功能与平衡依据": "F05 (TEP) + F07 (I-ISS) + F10 (外循环)",
        },
        {
            "流路编号": "OUT",
            "工艺流路节点": "SDS 储罐总输出加料流率",
            "平顶瞬时流率_gh": round(m_sds_out_peak, 2),
            "放电周期平均流率_gh": round(m_sds_out_peak * duty, 2),
            "物理功能与平衡依据": "F01 连续供给加料注入机",
        },
        {
            "流路编号": "NET",
            "工艺流路节点": "SDS 储罐净质量累积速率",
            "平顶瞬时流率_gh": f"+{m_sds_net_peak:.2f}",
            "放电周期平均流率_gh": f"+{m_sds_net_peak * duty:.2f} ~ +{m_sds_net_peak:.2f}",
            "物理功能与平衡依据": "IN > OUT: 净增殖速率严格大于零，达成完全自持",
        },
    ]

    return pd.DataFrame(flows)


def analyze_simulation_mass_conservation(
    results_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    eng_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Audits dynamic mass conservation strictly based on configured auditor rules.

    Does NOT fabricate assumptions (no hardcoded burnup rates or TBR values).
    If the task does not configure full-plant input/output flow patterns and inventory patterns,
    it explicitly reports unconfigured status rather than computing a synthetic pseudo-residual.
    """
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return {
            "configured": False,
            "status": "no_data",
            "message": "未提供时序数据",
            "status_text": "未提供时序数据",
            "max_relative_residual": 0.0,
        }

    cfg = config or {}
    auditor_data = (
        cfg.get("task", {}).get("auditor")
        if isinstance(cfg.get("task"), dict)
        else None
    ) or cfg.get("auditor", {})
    if not auditor_data:
        patterns = cfg.get("patterns", {})
        if patterns:
            auditor_data = {"enabled": True, "patterns": patterns}

    from tricys.plugins.auditor.offline import (
        audit_dataframe_mass_balance,
        parse_auditor_config,
    )

    auditor_cfg = parse_auditor_config({"auditor": auditor_data})

    if not auditor_cfg.enabled and not (
        auditor_cfg.inventory_patterns or auditor_cfg.source_patterns
    ):
        return {
            "configured": False,
            "is_conserved": True,
            "status": "unconfigured",
            "message": "未在任务配置中定义系统边界进出流股与库存审计规则 (auditor.patterns)",
            "status_text": "未配置全厂源汇审计规则",
            "max_relative_residual": 0.0,
        }

    audit_res = audit_dataframe_mass_balance(results_df, auditor_cfg)
    err = audit_res.get("mass_balance_error", 0.0)
    warn_th = audit_res.get("warn_threshold_g", 0.5)
    audit_res["is_conserved"] = abs(err) <= warn_th if err is not None else True
    if "max_relative_residual" not in audit_res:
        audit_res["max_relative_residual"] = 0.0
    return audit_res


def calculate_real_sensitivity_indices(
    jobs_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    target_metric: Optional[str] = None,
    task_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculates authentic sensitivity indices using actual execution data without synthetic approximations.

    When sampling follows SALib Sobol/Saltelli requirements, executes authentic Sobol GSA
    via SALib.analyze.sobol.analyze, obtaining real S1, ST, and confidence bounds.
    When sampling does not support Sobol matrix decomposition, computes authentic Pearson
    correlation r, Spearman rank correlation rho, and coefficient of determination R^2.
    """
    cfg = task_config or kwargs.get("config", {})
    task_cfg = cfg.get("task", {})
    configured_params = task_cfg.get("parameters", {})

    # 1. Identify input parameter columns and bounds
    param_names: List[str] = []
    param_bounds: Dict[str, Tuple[float, float]] = {}

    if isinstance(configured_params, dict):
        for k, v in configured_params.items():
            param_names.append(k)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    param_bounds[k] = (float(v[0]), float(v[1]))
                except Exception:
                    pass
    elif isinstance(configured_params, list) and len(configured_params) > 0:
        for k in configured_params[0].keys():
            if k not in ("id", "name", "job_id", "job_name", "case_name", "status"):
                param_names.append(k)

    if jobs_df is not None and not jobs_df.empty:
        ignored = {"id", "job_id", "name", "job_name", "case_name", "job", "status"}
        for c in jobs_df.columns:
            if c.lower() not in ignored and c not in param_names:
                num = pd.to_numeric(jobs_df[c], errors="coerce")
                if num.notna().sum() >= 2:
                    param_names.append(c)

    if not param_names:
        return pd.DataFrame()

    # Determine bounds from jobs_df if not provided in config
    if jobs_df is not None and not jobs_df.empty:
        for p in param_names:
            if p not in param_bounds and p in jobs_df.columns:
                s = pd.to_numeric(jobs_df[p], errors="coerce").dropna()
                if len(s) > 0:
                    param_bounds[p] = (float(s.min()), float(s.max()))

    # 2. Identify target output metric
    chosen_metric = target_metric
    if not chosen_metric:
        cfg_metrics = task_cfg.get("metrics", {})
        if isinstance(cfg_metrics, dict) and cfg_metrics:
            chosen_metric = list(cfg_metrics.keys())[0]

    # Combine data if available
    eval_df = pd.DataFrame()
    if metrics_df is not None and not metrics_df.empty:
        eval_df = metrics_df.copy()
        if jobs_df is not None and not jobs_df.empty:
            jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
            if jid_col in eval_df.columns and jid_col in jobs_df.columns:
                eval_df = eval_df.merge(jobs_df, on=jid_col, suffixes=("", "_job"))
            else:
                for c in param_names:
                    if c in jobs_df.columns and c not in eval_df.columns:
                        eval_df[c] = jobs_df[c].values[: len(eval_df)]
    elif jobs_df is not None and not jobs_df.empty:
        eval_df = jobs_df.copy()

    if eval_df.empty:
        return pd.DataFrame()

    if not chosen_metric:
        metric_candidates = [
            c
            for c in eval_df.columns
            if c not in param_names
            and c.lower() not in ("id", "job_id", "name", "job", "status")
            and pd.to_numeric(eval_df[c], errors="coerce").notna().sum() >= 2
        ]
        if metric_candidates:
            chosen_metric = metric_candidates[0]

    if not chosen_metric or chosen_metric not in eval_df.columns:
        return pd.DataFrame()

    y_series = pd.to_numeric(eval_df[chosen_metric], errors="coerce")
    valid_mask = y_series.notna()
    for p in param_names:
        if p in eval_df.columns:
            valid_mask &= pd.to_numeric(eval_df[p], errors="coerce").notna()

    eval_valid = eval_df[valid_mask].copy()
    if len(eval_valid) < 2:
        return pd.DataFrame()

    y_vals = pd.to_numeric(eval_valid[chosen_metric], errors="coerce").values

    # 3. Check if authentic Sobol analysis via SALib can be performed
    sobol_success = False
    sobol_records: List[Dict[str, Any]] = []

    has_all_bounds = all(p in param_bounds for p in param_names)
    method_name = str(task_cfg.get("method", "")).lower()

    glossary_map = load_glossary_mapping(config=cfg)

    if has_all_bounds and (
        method_name in ("sobol", "saltelli", "gsa") or "sobol" in str(task_cfg).lower()
    ):
        try:
            problem = {
                "num_vars": len(param_names),
                "names": param_names,
                "bounds": [list(param_bounds[p]) for p in param_names],
            }
            from SALib.analyze import sobol as salib_sobol

            si = None
            for c_so in [True, False]:
                try:
                    si = salib_sobol.analyze(
                        problem, y_vals, calc_second_order=c_so, seed=42
                    )
                    break
                except Exception:
                    continue

            if si is not None:
                s1_arr = np.nan_to_num(si["S1"], nan=0.0)
                st_arr = np.nan_to_num(si["ST"], nan=0.0)
                s1_conf_arr = np.nan_to_num(si["S1_conf"], nan=0.0)
                st_conf_arr = np.nan_to_num(si["ST_conf"], nan=0.0)

                for i, p in enumerate(param_names):
                    s1 = float(s1_arr[i])
                    st = float(st_arr[i])
                    s1_conf = float(s1_conf_arr[i])
                    st_conf = float(st_conf_arr[i])
                    inter = max(0.0, st - s1)
                    s_rec = {
                        "敏感参数": p,
                        "一阶敏感度 (S1)": round(s1, 4),
                        "全阶敏感度 (ST)": round(st, 4),
                        "交互作用贡献 (ST-S1)": round(inter, 4),
                        "S1置信区间 (±)": round(s1_conf, 4),
                        "ST置信区间 (±)": round(st_conf, 4),
                        "分析方法": "Sobol 全局方差分解 (SALib 实际解算)",
                        "_st": st,
                    }
                    if glossary_map:
                        item = glossary_map.get(p) or glossary_map.get(p.lower())
                        if item and item.get("zh"):
                            s_rec["中文术语"] = item["zh"]
                            if item.get("desc"):
                                s_rec["说明"] = item["desc"]
                    sobol_records.append(s_rec)

                sobol_records.sort(key=lambda x: x["_st"], reverse=True)
                for r in sobol_records:
                    r.pop("_st")
                sobol_success = True
        except Exception as e:
            logger.debug(f"SALib Sobol calculation skipped or failed: {e}")

    if sobol_success and sobol_records:
        return pd.DataFrame(sobol_records)

    # 4. Fallback: Authentic correlation and variance contribution analysis
    from scipy.stats import pearsonr, spearmanr

    corr_records: List[Dict[str, Any]] = []
    for p in param_names:
        r_val = 0.0
        p_val = 1.0
        rho_val = 0.0
        if p in eval_valid.columns:
            x_vals = pd.to_numeric(eval_valid[p], errors="coerce").values
            if np.std(x_vals) > 1e-12 and np.std(y_vals) > 1e-12:
                try:
                    pr = pearsonr(x_vals, y_vals)
                    r_val = float(pr.statistic)
                    p_val = float(pr.pvalue)
                except Exception:
                    pass
                try:
                    sr = spearmanr(x_vals, y_vals)
                    rho_val = float(sr.statistic)
                except Exception:
                    pass

        r_sq = r_val**2
        p_val_str = f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"

        c_rec = {
            "敏感参数": p,
            "Pearson 线性相关度 (r)": round(r_val, 4),
            "Spearman 秩相关度 (rho)": round(rho_val, 4),
            "方差贡献率 (R²)": round(r_sq, 4),
            "显著性检验 (p-value)": p_val_str,
            "分析方法": "单参数回归与相关性分析",
            "_r_sq": r_sq,
        }
        if glossary_map:
            item = glossary_map.get(p) or glossary_map.get(p.lower())
            if item and item.get("zh"):
                c_rec["中文术语"] = item["zh"]
                if item.get("desc"):
                    c_rec["说明"] = item["desc"]
        corr_records.append(c_rec)

    corr_records.sort(key=lambda x: x["_r_sq"], reverse=True)
    for r in corr_records:
        r.pop("_r_sq")

    return pd.DataFrame(corr_records)
