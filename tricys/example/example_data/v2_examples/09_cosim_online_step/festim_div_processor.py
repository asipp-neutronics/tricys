# -*- coding: utf-8 -*-
"""
TRICYS V2.0 Co-Simulation Processor: Divertor FESTIM Live In-Process FEM Engine.
==============================================================================
Tier 1: 进程内直接 FEM (MonoblockFEMModel)
DOLFINx 有限元求解 · 零 Socket 损耗 · 高精度物理仿真

Couples example_model.mo (Divertor subsystem 'div') with FESTIM v2 Live FEM.
- Simulates hydrogen isotope transport across 1D W/Cu/CuCrZr multi-layer monoblock.
- Resolves 0.85 eV single-vacancy trap saturation dynamics and permeation delay.
- Feeds back 5-element isotope permeation vector to the plant-wide coolant loop.
- Supports closed-loop OnlineGlobalAuditor plant-wide mass conservation audit.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)

logger = logging.getLogger(__name__)

# Check FESTIM availability in current Python environment
try:
    import festim as F

    FESTIM_AVAILABLE = True
except ImportError:
    F = None
    FESTIM_AVAILABLE = False

N_A = 6.02214076e23
M_T = 3.01604928  # Tritium g/mol
M_D = 2.01410178  # Deuterium g/mol
M_H = 1.00782504  # Protium g/mol


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return float(default)
        if isinstance(val, (np.ndarray, list, tuple)):
            if len(val) == 0:
                return float(default)
            return float(val.flat[0]) if hasattr(val, "flat") else float(val[0])
        return float(val)
    except Exception:
        return float(default)


class MonoblockFEMModel:
    """1D Multi-layer Monoblock FESTIM v2 FEM Transport Model (W / Cu / CuCrZr)."""

    def __init__(self, area_m2: float = 140.0, T_K: float = 600.0) -> None:
        if not FESTIM_AVAILABLE or F is None:
            raise RuntimeError(
                "FESTIM is not installed in the current Python environment. "
                "Please run inside the TRICYS Docker container with FESTIM v2."
            )
        self.area_m2 = float(area_m2)
        self.T_K = float(T_K)
        self.build_model()

    def build_model(self) -> None:
        logger.info(
            f"[Tier 1 FEM] Initializing FESTIM v2 1D Monoblock model: "
            f"Area={self.area_m2:.1f} m2, Temperature={self.T_K:.1f} K..."
        )
        # Material properties: W (Armour), Cu (Interlayer), CuCrZr (Heat Sink / Pipe)
        mat_W = F.Material(D_0=4.1e-7, E_D=0.39, name="W", K_S_0=1.87e24, E_K_S=1.04)
        mat_Cu = F.Material(D_0=6.6e-7, E_D=0.39, name="Cu", K_S_0=3.14e24, E_K_S=0.57)
        mat_CuCrZr = F.Material(
            D_0=3.92e-7, E_D=0.42, name="CuCrZr", K_S_0=4.88e24, E_K_S=0.29
        )

        # 1D Non-uniform mesh: 4mm W + 1mm Cu + 1.5mm CuCrZr (Optimized for fast online step demo)
        x_nodes = np.concatenate(
            [
                np.linspace(0.0, 0.004, 15),
                np.linspace(0.004, 0.005, 6)[1:],
                np.linspace(0.005, 0.0065, 10)[1:],
            ]
        )
        mesh = F.Mesh1D(x_nodes)

        self.surf_plasma = F.SurfaceSubdomain1D(id=1, x=0.0)
        self.surf_coolant = F.SurfaceSubdomain1D(id=2, x=0.0065)
        self.vol_W = F.VolumeSubdomain1D(id=1, borders=[0.0, 0.004], material=mat_W)
        self.vol_Cu = F.VolumeSubdomain1D(id=2, borders=[0.004, 0.005], material=mat_Cu)
        self.vol_CuCrZr = F.VolumeSubdomain1D(
            id=3, borders=[0.005, 0.0065], material=mat_CuCrZr
        )

        self.model = F.HydrogenTransportProblem()
        self.model.mesh = mesh
        self.model.subdomains = [
            self.surf_plasma,
            self.surf_coolant,
            self.vol_W,
            self.vol_Cu,
            self.vol_CuCrZr,
        ]

        self.H = F.Species("T")
        # Microscopic 0.85 eV trap in Tungsten armor
        self.trap_W = F.Trap(
            name="trap_W",
            mobile_species=self.H,
            k_0=4.1e-7 / (1.1e-10**2 * 6.3e28),
            E_k=0.39,
            p_0=1e13,
            E_p=0.85,
            n=6.3e24,
            volume=self.vol_W,
        )
        self.model.species = [self.H]
        self.model.traps = [self.trap_W]
        self.model.temperature = self.T_K

        # Boundary conditions
        self.flux_bc = F.ParticleFluxBC(
            subdomain=self.surf_plasma, value=0.0, species="T"
        )
        self.coolant_bc = F.DirichletBC(
            subdomain=self.surf_coolant, value=0.0, species="T"
        )
        self.model.boundary_conditions = [self.flux_bc, self.coolant_bc]

        # Solver settings
        self.model.settings = F.Settings(
            atol=1e-8,
            rtol=1e-8,
            max_iterations=60,
            transient=True,
            stepsize=F.Stepsize(initial_value=1.0),
            final_time=1e10,
        )
        self.model.show_progress_bar = False

        # Derived flux & inventory exports
        self.surf_flux_coolant = F.SurfaceFlux(field=self.H, surface=self.surf_coolant)
        self.tot_vol_mob = F.TotalVolume(field=self.H, volume=self.vol_W)
        self.model.exports = [self.surf_flux_coolant, self.tot_vol_mob]

        self.model.initialise()
        logger.info(
            "[Tier 1 FEM] FESTIM v2 1D Monoblock FEM model initialized successfully."
        )

    def reset(self) -> None:
        """Reset finite element state to initial clean state."""
        self.model.t.value = 0.0
        self.model.u.x.array[:] = 0.0
        self.model.u_n.x.array[:] = 0.0
        if hasattr(self.flux_bc, "value_fenics"):
            self.flux_bc.value_fenics.value = np.array(0.0)

    def step(self, from_plasma_gh: Any, dt_s: float = 1.0) -> dict[str, Any]:
        """Advance FESTIM FEM simulation by dt_s seconds under incoming plasma flux."""
        if (
            isinstance(from_plasma_gh, (list, tuple, np.ndarray))
            and len(from_plasma_gh) > 0
        ):
            t_flow_gh = float(from_plasma_gh[0])
            d_flow_gh = float(from_plasma_gh[1]) if len(from_plasma_gh) > 1 else 0.0
            h_flow_gh = float(from_plasma_gh[2]) if len(from_plasma_gh) > 2 else 0.0
        else:
            t_flow_gh = safe_float(from_plasma_gh)
            d_flow_gh = 0.0
            h_flow_gh = 0.0

        t_flow_gps = max(0.0, t_flow_gh) / 3600.0
        phi_in_m2_s = (t_flow_gps * N_A) / (M_T * self.area_m2)

        if hasattr(self.flux_bc, "value_fenics"):
            self.flux_bc.value_fenics.value = np.array(phi_in_m2_s)
        self.model.dt.value = float(dt_s)

        try:
            self.model.iterate()
        except Exception as e:
            logger.warning(
                f"[Tier 1 FEM] Non-linear solve iteration warning ({e}), soft-stepping with previous state."
            )
            self.model.u.x.array[:] = self.model.u_n.x.array[:]
            self.model.t.value += self.model.dt.value

        # Coolant permeation flux -> mass flow (g/h)
        phi_coolant = abs(safe_float(self.surf_flux_coolant.value))
        to_CL_t_gh = (phi_coolant * self.area_m2 * M_T / N_A) * 3600.0

        # Total retained mass inside Monoblock (g)
        u_arr = self.model.u.x.array
        inv_tot_g = (
            float(np.sum(np.maximum(0.0, u_arr)))
            * (self.area_m2 * M_T / N_A)
            * (0.0065 / max(1, len(u_arr)))
        )

        # 5-element isotope vector matching example_model to_CL[5]
        # [1: Tritium, 2: Deuterium, 3: Protium, 4: He4, 5: Impurities]
        ratio_d = (
            (d_flow_gh / max(1e-12, t_flow_gh)) * (M_D / M_T)
            if t_flow_gh > 1e-12
            else 0.0
        )
        ratio_h = (
            (h_flow_gh / max(1e-12, t_flow_gh)) * (M_H / M_T)
            if t_flow_gh > 1e-12
            else 0.0
        )

        to_cl_vec = [
            float(to_CL_t_gh),
            float(to_CL_t_gh * ratio_d),
            float(to_CL_t_gh * ratio_h),
            0.0,
            0.0,
        ]

        return {
            "to_CL": to_cl_vec,
            "inventory_g": float(inv_tot_g),
            "phi_in": float(phi_in_m2_s),
            "phi_coolant": float(phi_coolant),
        }

    def get_microscopic_profiles(self) -> dict[str, Any]:
        """Extract spatial profiles of mobile and trapped tritium across 1D Monoblock."""
        try:
            x_m = np.array(self.model.mesh.mesh.geometry.x[:, 0])
            sort_idx = np.argsort(x_m)
            x_m = x_m[sort_idx]
            x_mm = x_m * 1e3

            u_mob = np.maximum(0.0, self.model.u.sub(0).collapse().x.array[sort_idx])
            u_trap = np.maximum(0.0, self.model.u.sub(1).collapse().x.array[sort_idx])

            mask_W = x_m <= 0.004001
            mask_Cu = (x_m > 0.004001) & (x_m <= 0.005001)
            mask_CuCrZr = x_m > 0.005001

            n_t = getattr(self.trap_W, "n", 6.3e24)
            theta = np.zeros_like(u_trap)
            theta[mask_W] = np.clip(u_trap[mask_W] / n_t, 0.0, 1.0) * 100.0

            vol_W = self.area_m2 * 0.004
            vol_Cu = self.area_m2 * 0.001
            vol_CuCrZr = self.area_m2 * 0.0015

            inv_trap_W = (
                float(np.mean(u_trap[mask_W])) * (vol_W * M_T / N_A)
                if np.any(mask_W)
                else 0.0
            )
            inv_mob_W = (
                float(np.mean(u_mob[mask_W])) * (vol_W * M_T / N_A)
                if np.any(mask_W)
                else 0.0
            )
            inv_mob_Cu = (
                float(np.mean(u_mob[mask_Cu])) * (vol_Cu * M_T / N_A)
                if np.any(mask_Cu)
                else 0.0
            )
            inv_mob_CuCrZr = (
                float(np.mean(u_mob[mask_CuCrZr])) * (vol_CuCrZr * M_T / N_A)
                if np.any(mask_CuCrZr)
                else 0.0
            )

            return {
                "x_mm": x_mm.tolist(),
                "c_m": u_mob.tolist(),
                "c_t": u_trap.tolist(),
                "theta_pct": theta.tolist(),
                "layers": {
                    "W_trap_g": inv_trap_W,
                    "W_mob_g": inv_mob_W,
                    "Cu_mob_g": inv_mob_Cu,
                    "CuCrZr_mob_g": inv_mob_CuCrZr,
                    "total_g": inv_trap_W + inv_mob_W + inv_mob_Cu + inv_mob_CuCrZr,
                },
            }
        except Exception as e:
            logger.warning(f"[Tier 1 FEM] Failed to extract microscopic profiles: {e}")
            return {}


class FESTIMDivertorProcessor(AbstractTrackProcessor):
    """
    Tier 1 Pure In-Process FESTIM v2 Live FEM Co-Simulation Processor.
    Direct DOLFINx execution with zero socket overhead.
    """

    _latest_cosim_data: dict[str, Any] = {}

    def __init__(
        self,
        area_m2: float = 140.0,
        T_K: float = 600.0,
        decay_const_per_h: float = 6.4e-6,
        **kwargs: Any,
    ) -> None:
        self.area_m2 = float(area_m2)
        self.T_K = float(T_K)
        self.decay_const_per_h = float(decay_const_per_h)

        self.fem_model: MonoblockFEMModel | None = None
        self.live_inventory_g = 0.0
        self.step_count = 0
        self.results_dir: str | None = None
        self.history_records: list[dict[str, float]] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.snapshot_milestones = [0.5, 2.0, 6.67, 20.0, 50.0, 100.0]
        self.captured_milestones: set[float] = set()

    def initialize(self, context: TrackProcessorContext) -> None:
        cfg = context.config.get("params", {})
        if "area_m2" in cfg:
            self.area_m2 = float(cfg["area_m2"])
        if "T_K" in cfg:
            self.T_K = float(cfg["T_K"])
        if "decay_const_per_h" in cfg:
            self.decay_const_per_h = float(cfg["decay_const_per_h"])

        self.results_dir = context.config.get("results_dir")
        self.live_inventory_g = 0.0
        self.step_count = 0
        self.history_records.clear()
        self.snapshots.clear()
        self.captured_milestones.clear()

        logger.info(
            f"[FESTIMDivertorProcessor] Activating Tier 1 Direct In-Process FESTIM FEM "
            f"(Area={self.area_m2}m2, T={self.T_K}K, ResultsDir={self.results_dir})..."
        )
        self.fem_model = MonoblockFEMModel(area_m2=self.area_m2, T_K=self.T_K)
        self.fem_model.reset()
        logger.info(
            "[FESTIMDivertorProcessor] Tier 1 Direct In-Process FESTIM v2 FEM engine fully active."
        )

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        self.step_count += 1
        dt_slow_h = request_vector.dt_slow_h
        dt_s = dt_slow_h * 3600.0
        t_h = float(request_vector.current_time_h)

        # Extract incoming plasma flow from example_model.Cycle (plasma.to_Div -> div.from_plasma)
        from_plasma = request_vector.boundary_inputs.get(
            "from_plasma", [0.0, 0.0, 0.0, 0.0, 0.0]
        )

        if self.fem_model is not None:
            fem_res = self.fem_model.step(from_plasma_gh=from_plasma, dt_s=dt_s)
            self.live_inventory_g = float(fem_res.get("inventory_g", 0.0))
            to_cl_vec = fem_res.get("to_CL", [0.0, 0.0, 0.0, 0.0, 0.0])

            # Record time series history
            t_flow_gh = safe_float(
                from_plasma[0]
                if isinstance(from_plasma, (list, tuple, np.ndarray))
                and len(from_plasma) > 0
                else from_plasma
            )
            self.history_records.append(
                {
                    "time_h": t_h,
                    "inventory_g": self.live_inventory_g,
                    "to_CL_gh": float(to_cl_vec[0]),
                    "plasma_in_gh": t_flow_gh,
                }
            )

            # Check snapshot milestone capture
            for ms in self.snapshot_milestones:
                if ms not in self.captured_milestones and t_h >= (ms - 1e-4):
                    self.captured_milestones.add(ms)
                    prof = self.fem_model.get_microscopic_profiles()
                    if prof:
                        self.snapshots[f"{ms}h"] = prof

            return TrackResult(
                outputs={
                    "to_CL": to_cl_vec,
                }
            )

        # Neutral zero fallback if uninitialized
        return TrackResult(outputs={"to_CL": [0.0, 0.0, 0.0, 0.0, 0.0]})

    def finalize(self) -> None:
        logger.info(
            f"[FESTIMDivertorProcessor] Tier 1 FEM finalize completed after {self.step_count} co-simulation steps."
        )
        if self.fem_model is not None:
            final_prof = self.fem_model.get_microscopic_profiles()
            if final_prof:
                self.snapshots["final"] = final_prof

        data = {
            "history": self.history_records,
            "snapshots": self.snapshots,
            "area_m2": self.area_m2,
            "T_K": self.T_K,
            "step_count": self.step_count,
        }
        FESTIMDivertorProcessor._latest_cosim_data = data

        if self.results_dir and os.path.exists(self.results_dir):
            out_file = os.path.join(self.results_dir, "festim_micro_data.json")
            try:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                logger.info(
                    f"[FESTIMDivertorProcessor] Saved microscopic dataset to {out_file}"
                )
            except Exception as e:
                logger.warning(
                    f"[FESTIMDivertorProcessor] Could not save {out_file}: {e}"
                )

    @classmethod
    def get_latest_cosim_data(cls) -> dict[str, Any]:
        return cls._latest_cosim_data

    def get_mass_inventory(self) -> float:
        """Report microscopic + trapped tritium inventory (g) for plant-wide mass audit."""
        return float(self.live_inventory_g)

    def get_decay_rate(self) -> float:
        """Report radioactive decay rate (g/s) of divertor tritium inventory."""
        inv = self.get_mass_inventory()
        return float(inv * self.decay_const_per_h / 3600.0)

    def get_release_rate(self) -> float:
        """Report environmental release rate (g/s). Divertor is inside containment vacuum vessel."""
        return 0.0


# Backward compatibility alias
CFEDRFESTIMDivertorProcessor = FESTIMDivertorProcessor
