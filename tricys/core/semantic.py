from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ==============================================================================
# TRICYS Standard Fusion Tritium Ontology
# Grounded in example_glossary.csv and Abdou (2021) physics rules
# ==============================================================================
DEFAULT_PHYSICS_ONTOLOGY: Dict[str, List[str]] = {
    # 1. SDS Storage and Central Inventory Dynamics
    "sds_inventory": [
        r".*sds.*inventory.*",
        r".*sds.*I\[1\].*",
        r".*storage.*inventory.*",
        r".*sds_inventory.*",
    ],
    "sds_decay_loss": [
        r".*sds.*decay_loss.*",
        r".*storage.*decay_loss.*",
    ],
    "startup_inventory": [
        r".*startup.*inventory.*",
        r".*initial.*loading.*",
    ],
    # 2. Breeding Blanket and Breeding Ratio (TBR)
    "blanket_inventory": [
        r".*blanket.*inventory.*",
        r".*blanket.*I\[1\].*",
        r".*bz.*inventory.*",
    ],
    "blanket_tbr": [
        r".*blanket.*tbr.*",
        r"^tbr$",
        r".*actual_tbr.*",
    ],
    "blanket_residence_time": [
        r".*blanket.*\.T$",
        r".*blanket.*residence.*",
    ],
    "blanket_to_tes_fraction": [
        r".*blanket.*to_tes.*",
    ],
    "blanket_to_cl_fraction": [
        r".*blanket.*to_cl.*",
    ],
    # 3. Plasma Core and Pulse Fueling Dynamics
    "plasma_burn_fraction": [
        r".*plasma.*\.fb$",
        r".*plasma.*\.f_b$",
        r".*burn_fraction.*",
        r".*burnup.*",
    ],
    "plasma_fueling_eff": [
        r".*plasma.*\.nf$",
        r".*fueling.*efficiency.*",
    ],
    "pulse_amplitude": [
        r".*pulse.*amplitude.*",
        r".*burn_rate.*",
    ],
    "availability_factor": [
        r".*pulse.*width.*",
        r".*duty_factor.*",
        r".*availability.*",
    ],
    # 4. Exhaust Processing, Vacuum Pumping, and Inner-Loop ISS
    "divertor_inventory": [
        r".*div.*inventory.*",
        r".*div.*I\[1\].*",
        r".*divertor.*inventory.*",
    ],
    "pump_inventory": [
        r".*pump.*inventory.*",
        r".*pump.*I\[1\].*",
        r".*vacuum.*inventory.*",
    ],
    "tep_fep_inventory": [
        r".*tep_fep.*inventory.*",
        r".*tep_fep.*I\[1\].*",
    ],
    "tep_fcu_inventory": [
        r".*tep_fcu.*inventory.*",
        r".*tep_fcu.*I\[1\].*",
    ],
    "tep_inventory": [
        r".*tep.*inventory.*",
        r".*tep.*I\[1\].*",
    ],
    "dir_ratio": [
        r".*tep_fep.*to_sds.*",
        r".*dir_fraction.*",
    ],
    "i_iss_inventory": [
        r".*i_iss.*inventory.*",
        r".*i_iss.*I\[1\].*",
    ],
    "i_iss_threshold": [
        r".*i_iss.*threshold.*",
    ],
    # 5. Extraction, Recovery, and Outer-Loop Systems
    "tes_inventory": [
        r".*tes.*inventory.*",
        r".*tes.*I\[1\].*",
    ],
    "tes_threshold": [
        r".*tes.*threshold.*",
    ],
    "o_iss_inventory": [
        r".*o_iss.*inventory.*",
        r".*o_iss.*I\[1\].*",
    ],
    "wds_inventory": [
        r".*wds.*inventory.*",
        r".*wds.*I\[1\].*",
    ],
    # 6. Coolant Loop, Purification, and First Wall
    "coolant_inventory": [
        r".*coolant_pipe.*inventory.*",
        r".*coolant_pipe.*I\[1\].*",
        r".*coolant.*inventory.*",
    ],
    "cps_inventory": [
        r".*cps.*inventory.*",
        r".*cps.*I\[1\].*",
    ],
    "first_wall_inventory": [
        r".*fw.*inventory.*",
        r".*fw.*I\[1\].*",
        r".*first_wall.*inventory.*",
    ],
    # 7. High-Level Scientific Evaluation Metrics
    "required_tbr": [
        r".*required_tbr.*",
        r".*tbr_req.*",
    ],
    "doubling_time": [
        r".*doubling_time.*",
        r"^t_d$",
    ],
    "self_sufficiency_time": [
        r".*self_sufficiency_time.*",
        r".*turnaround_time.*",
    ],
}


class PhysicsSemanticResolver:
    """Universal Fusion Physics Semantic Bus.

    Translates standardized physics concepts (e.g., 'sds_inventory', 'blanket_tbr')
    into concrete Modelica column names across polymorphic plant simulation models.
    Supports user overrides and open-ended runtime extensions.
    """

    def __init__(
        self,
        custom_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Shallow copy default ontology to avoid mutating global dictionary
        self._ontology: Dict[str, List[str]] = {
            k: list(v) for k, v in DEFAULT_PHYSICS_ONTOLOGY.items()
        }

        # Apply user/agent customized overrides
        if custom_mapping:
            for concept, patterns in custom_mapping.items():
                self.add_mapping(concept, patterns, override=True)

    def add_mapping(
        self,
        concept: str,
        patterns: Union[str, List[str]],
        override: bool = False,
    ) -> None:
        """Adds or extends a physics semantic mapping pattern."""
        concept_key = concept.strip().lower()
        new_patterns = [patterns] if isinstance(patterns, str) else list(patterns)

        if override or concept_key not in self._ontology:
            self._ontology[concept_key] = new_patterns
        else:
            # Append non-duplicate patterns
            for p in new_patterns:
                if p not in self._ontology[concept_key]:
                    self._ontology[concept_key].append(p)

    def get_matched_column(
        self,
        columns: Iterable[str],
        concept: str,
    ) -> Optional[str]:
        """Finds the first column in `columns` matching the physics semantic concept."""
        concept_key = concept.strip().lower()
        col_list = list(columns)

        # 1. Check exact column name match first
        for col in col_list:
            if col.lower() == concept_key:
                return col

        patterns = self._ontology.get(concept_key, [])
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for col in col_list:
                if regex.match(col):
                    return col

        # 2. Heuristic fallback: split concept into keywords
        # e.g. "sds_inventory" -> checks if both "sds" and "inventory" appear in col
        parts = concept_key.split("_")
        if len(parts) > 1:
            for col in col_list:
                col_lower = col.lower()
                if all(part in col_lower for part in parts):
                    return col

        return None

    def resolve_variable(
        self,
        df: pd.DataFrame,
        concept: str,
    ) -> Optional[pd.Series]:
        """Resolves and returns the timeseries pd.Series for a physics concept from a DataFrame."""
        if df is None or df.empty:
            return None

        col_name = self.get_matched_column(df.columns, concept)
        if col_name and col_name in df.columns:
            return df[col_name]
        return None

    def get_all_inventories(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """Convenience method: discovers and returns all component inventory timeseries present in the DataFrame."""
        inventories: Dict[str, pd.Series] = {}
        if df is None or df.empty:
            return inventories

        inventory_concepts = [
            "sds_inventory",
            "blanket_inventory",
            "coolant_inventory",
            "divertor_inventory",
            "pump_inventory",
            "tep_fep_inventory",
            "tep_fcu_inventory",
            "tep_inventory",
            "i_iss_inventory",
            "o_iss_inventory",
            "tes_inventory",
            "wds_inventory",
            "cps_inventory",
            "first_wall_inventory",
        ]

        for concept in inventory_concepts:
            series = self.resolve_variable(df, concept)
            if series is not None:
                # Clean key, e.g. "sds_inventory" -> "sds"
                short_name = concept.replace("_inventory", "")
                inventories[short_name] = series

        return inventories

    def resolve_parameter(
        self,
        params: Dict[str, Any],
        concept: str,
    ) -> Optional[Any]:
        """Extracts a parameter value from a dictionary using physics concept resolution."""
        if not params:
            return None

        matched_key = self.get_matched_column(params.keys(), concept)
        if matched_key and matched_key in params:
            return params[matched_key]
        return None
