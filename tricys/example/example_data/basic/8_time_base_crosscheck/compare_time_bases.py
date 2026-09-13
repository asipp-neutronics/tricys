#!/usr/bin/env python3
"""Cross-validate hour-base and second-base runs of the example model.

Both runs drive the *same* model source (``example_model.Cycle``) with the same
hour-authored inputs; they differ only in ``simulation.time_base`` and the
model's ``time_scale``:

  * second-base: time_base="second", time_scale=3600  -> ``time`` axis in seconds
  * hour-base:   time_base="hour",   time_scale=1     -> ``time`` axis in hours

Physically they must describe the same trajectory, so

    sds_second(t * 3600)  ==  sds_hour(t)

This script loads both ``results/sweep_results.h5`` files, interpolates the
second-base curve onto the hour-base time grid (scaled by 3600) and reports the
maximum absolute / relative deviation.

Usage:
    python compare_time_bases.py <second_run_dir> <hour_run_dir>
    python compare_time_bases.py            # auto-detect the two newest runs
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLE = "sds.I[1]"
SECONDS_PER_HOUR = 3600.0
REL_TOL = 1e-3
ABS_TOL = 1e-6


def load(run_dir: Path) -> pd.DataFrame:
    h5 = run_dir / "results" / "sweep_results.h5"
    if not h5.exists():
        raise SystemExit(f"missing {h5}")
    df = pd.read_hdf(h5, "/results")
    if VARIABLE not in df.columns:
        raise SystemExit(f"{h5}: missing column {VARIABLE!r} (have {list(df.columns)})")
    return df[["time", VARIABLE]].drop_duplicates(subset="time").sort_values("time")


def auto_detect(root: Path):
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    if len(runs) < 2:
        raise SystemExit("need at least two run directories; pass them explicitly")
    # newest two, oldest first
    return runs[-2], runs[-1]


def main(argv):
    root = Path(__file__).resolve().parent
    if len(argv) == 3:
        sec_dir, hour_dir = Path(argv[1]), Path(argv[2])
    else:
        sec_dir, hour_dir = auto_detect(root)

    sec, hour = load(sec_dir), load(hour_dir)
    sec["time_h"] = sec["time"] / SECONDS_PER_HOUR

    print(f"second-base run: {sec_dir.name}")
    print(f"  time {sec['time'].min():.6g} -> {sec['time'].max():.6g} s "
          f"({sec['time_h'].max():.4g} h), {len(sec)} points")
    print(f"hour-base run:   {hour_dir.name}")
    print(f"  time {hour['time'].min():.6g} -> {hour['time'].max():.6g} h, {len(hour)} points")

    t_h = hour["time"].to_numpy()
    y_hour = hour[VARIABLE].to_numpy()
    y_sec = np.interp(t_h, sec["time_h"].to_numpy(), sec[VARIABLE].to_numpy())

    abs_dev = np.abs(y_sec - y_hour)
    denom = np.maximum(np.abs(y_hour), 1e-12)
    rel_dev = abs_dev / denom
    i_abs, i_rel = int(np.argmax(abs_dev)), int(np.argmax(rel_dev))

    print("\n=== cross-validation (hour grid) ===")
    print(f"  max abs deviation: {abs_dev.max():.6e}  (at t={t_h[i_abs]:.4g} h)")
    print(f"  max rel deviation: {rel_dev.max():.6e}  (at t={t_h[i_rel]:.4g} h)")
    print(f"  hour-base  range : {y_hour.min():.6g} .. {y_hour.max():.6g}")
    print(f"  second-base range: {sec[VARIABLE].min():.6g} .. {sec[VARIABLE].max():.6g}")

    ok = bool(np.allclose(y_sec, y_hour, rtol=REL_TOL, atol=ABS_TOL))
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(rtol={REL_TOL}, atol={ABS_TOL})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
