# Time-base cross-check (hour vs second)

Demonstrates that the **same model source** can be driven with either the
legacy **hour-based** time axis or the new **second-based** axis, and that the
two runs describe the identical physical trajectory.

## How it works

`example_model.Cycle` (see `../../example_model_single/example_model.mo`) exposes an
`inner parameter Real time_scale`:

* `time_scale = 1`    → hour-based axis (legacy behaviour; `time` is read as hours)
* `time_scale = 3600` → second-based axis (seconds-native; `time` is in seconds)

and integrates `der(...) = (RHS) / time_scale`.

The framework declares the model's base via `simulation.time_base` and always
authors `stop_time` / `step_size` in **hours**:

| Config | `time_base` | `time_scale` | FOC | resulting `time` axis |
|--------|-------------|--------------|-----|-----------------------|
| `time_base_second.json` | `"second"` | `3600` | `example_scenario_mix.foc` (`hour_to_second`) | 0 .. 4.32e6 s (= 1200 h) |
| `time_base_hour.json`   | `"hour"`   | `1`    | `example_scenario_mix_hour.foc` (hours)      | 0 .. 1200 h |

## Run

```bash
# from this directory
tricys -c time_base_second.json
tricys -c time_base_hour.json
python compare_time_bases.py          # auto-detects the two newest runs
```

`compare_time_bases.py` interpolates the second-base curve onto the hour-base grid
(`t_s = 3600 * t_h`) and reports the max absolute / relative deviation. It exits
non-zero on mismatch.

Typical result: max relative deviation ~1e-10 → **PASS**.
