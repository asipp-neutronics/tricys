# Upgrading from TRICYS 1.0 to 2.0

In one sentence: **time fields in the configuration are still written in hours; the new `time_base` key tells the framework whether the *model* is authored in seconds or in hours.**

Legacy models (written in hours in the 1.0 era) should declare `time_base: hour`, which reproduces 1.0 behaviour exactly.
New models authored in SI seconds keep the default `second`, and the framework multiplies the hour-valued configuration by 3600 before handing it to OpenModelica.

---

## 1. Time base (the only thing that needs a decision)

| `time_base` | How the model expresses time | stopTime handed to OMC |
| --- | --- | --- |
| `second` (default) | model is in seconds | configuration value (hours) × 3600 |
| `hour` | model is in hours | configuration value (hours), unchanged |

Same `stop_time: 2000`:

- `time_base: second` → OMC receives `7 200 000` (seconds)
- `time_base: hour` → OMC receives `2 000` (hours, identical to 1.0)

> **Since 2.0.1**: a configuration with neither `time_base` nor a `scenario` block (the 1.0 shape) triggers a WARNING about the 3600x risk. It is a warning only — the default stays `second`; the framework will not guess for you.

**Where to put it:**

- 2.0 YAML configurations (`scenario` block): `scenario.time_base`
- 1.0 JSON configurations (`simulation` block): `simulation.time_base`

**How to tell which kind your model is:**

```bash
python -m tricys model review path/to/package.mo example_model.Cycle
```

Look at `time_scale` in the report: `3600` means a seconds-based model, `1` means an hours-based model.
A top-level model without a `time_scale` declaration is reported as `TIME_SCALE_MISSING` (CRITICAL) — that is the 1.0-era shape.
Either configure `time_base: hour` for such models, or add `inner parameter Real time_scale = 3600;` following the 2.0 convention and re-check every `der(...)` term.

---

## 2. Breaking changes

| Change | Impact | Action |
| --- | --- | --- |
| New `time_base`, defaults to `second` | Without the key, a legacy model is treated as seconds-based and results are off by 3600× | Set `time_base: hour` on legacy models |
| `tricys.handlers` archived | Moved to `tricys.example.example_data.v1_examples_legacy.handlers`; a `sys.modules` shim remains in `tricys/__init__.py` | Prefer the new import path; the shim fails silently, do not rely on it |
| Prebuilt FMU / SSP assets removed | `example_model_ssp/` is no longer shipped; the SSP example is archived | Build your own, or use co-simulation examples 08–10 |
| v1 examples archived | Former `basic/` and `analysis/` examples moved under `v1_examples_legacy/` | Update any script or doc that references those paths |
| `tricys.handlers` dropped from the package list | Packaging scripts that list it explicitly will fail | Remove it from the package list |

---

## 3. The 1.0 configuration format still works

`tricys validate` / `tricys run` still accept the 1.0 JSON layout (`paths` + `simulation`).
The 2.0 research DSL (`scenario` + `task`, supporting `simulate / sweep / sensitivity / cosim / optimize`) and explicit pipelines are **optional** paths, not a forced migration.

```bash
python -m tricys validate -c config.json
python -m tricys run -c config.json
```

---

## 4. Migration checklist

- [ ] `python -m tricys validate -c <config>` passes
- [ ] `python -m tricys model review <package.mo> <model>` — `time_scale` and `time_base` agree
- [ ] Legacy models: add `time_base: hour` to the configuration
- [ ] Run once with a small `stop_time` and compare the time axis and final state against the 1.0 result
- [ ] Replace `import tricys.handlers` with the archived module path
- [ ] Replace references to `example_model_ssp` / prebuilt `*.fmu` assets with v2 co-simulation examples

---

## 5. Known issues in 2.0.0

- **Job failures do not reach the exit code**: when some jobs fail in a sweep or a co-simulation, the command still returns `0` and the results file merely contains fewer jobs. Scheduled for 2.0.1; until then, check the job count in the results file.
- **The auditor does not sanity-check its initial mass**: the online mass auditor takes whatever it reads at initialisation (0, NaN, negative) as the baseline. Scheduled for 2.0.1.

---

## 6. If you used the v2.0 branch before the 2.0.0 release

In sequential mode, the hour → second conversion was skipped for 1.0-style configurations, so the simulated time span was 1/3600 of the correct value.
That is fixed in 2.0.0 (sequential and concurrent paths now share a single conversion entry point). **Results produced in sequential mode from 1.0-style configurations before the fix must be considered invalid** and re-run.
