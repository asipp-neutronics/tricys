# Fusion Operation Configuration

TRICYS now supports FOC (Fusion Operation Config) to describe fusion operating schedules such as pulses, dwell periods, staged schedules, and repeated cycles. This allows the operating profile to be managed separately from the Modelica model itself.

The current repository example entry points are:

- `tricys/example/example_data/basic/5_fusion_operation_config/fusion_operation_config.json`
- `tricys/example/example_data/example_foc/example_scenario_mix.foc`

---

## 1. Example Configuration

The recommended file-driven example is:

```json
{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "foc": {
        "foc_path": "../../example_foc/example_scenario_mix.foc",
        "foc_component": "pulseSource"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 1200.0,
        "step_size": 0.5
    }
}
```

The corresponding FOC file example is:

```text
# Mixed operation example
TIME_UNIT hour
TIME_CONVERSION hour_to_second
PULSE 1500 0.9 0.1 300
DWELL 100
BEGIN_SCHEDULE
  PULSE 2000 0.9 0.1 100
  POWER 1000
  BURN 200
  DWELL 100
END_SCHEDULE
REPEAT 2
```

### 1.1. FOC time-header syntax

If the time unit used in the FOC file does not match the model's internal time base, or if you want to declare the current FOC time unit explicitly, add time-header directives at the top of the `.foc` file:

```text
TIME_UNIT <second|hour|day|week|year>
TIME_CONVERSION <factor>
TIME_CONVERSION <source_unit>_to_<target_unit>
```

Common forms are:

- when neither `TIME_UNIT` nor `TIME_CONVERSION` is declared, the FOC time unit defaults to seconds and no conversion is applied
- `TIME_UNIT hour`: only declares that durations in the current FOC file are written in hours; it does not trigger conversion by itself
- the **effective** time base of a FOC (after any `TIME_CONVERSION`) must be **seconds**, because the CFEDR family models advance Modelica's `time` in seconds; a FOC whose effective base is hours/days/years (or a bare numeric factor) triggers a warning and would run at the wrong speed
- `TIME_CONVERSION NONE`: applies no extra time conversion and uses the written FOC time values directly
- `TIME_CONVERSION 3600`: multiplies all subsequent durations by `3600`
- `TIME_CONVERSION second_to_hour`: converts FOC time values written in seconds into hours for a model that uses hours internally; reverse named conversions such as `hour_to_second` are also supported
- conversion is performed only when `TIME_CONVERSION` is declared explicitly

Constraints:

- supported explicit units are `second`, `hour`, `day`, `week`, and `year`
- these header directives must appear before business commands such as `POWER`, `BURN`, `DWELL`, `PULSE`, and `BEGIN_SCHEDULE`
- `TIME_UNIT` and `TIME_CONVERSION` may each be declared only once
- if `TIME_UNIT` and a named `TIME_CONVERSION` are both declared, they must agree on the source unit

Example 1: the FOC is written in hours and explicitly converted to the model's default second-based time axis

```text
TIME_UNIT hour
TIME_CONVERSION hour_to_second
PULSE 1500 0.9 0.1 300
```

Example 1b: the FOC is written in hours but only declares the unit without converting — **wrong for the seconds-based models**

```text
TIME_UNIT hour
PULSE 1500 0.9 0.1 300
```

`TIME_UNIT` declares the unit but does **not** convert. The CFEDR family models advance
Modelica's built-in `time` in **seconds**, so a schedule left in hours would run roughly
3600× too fast. TRICYS emits a warning when a FOC's effective time base is not seconds;
add `TIME_CONVERSION hour_to_second` (Example 1) to fix it. Use bare-`TIME_UNIT` form
only when the durations you write are already in the model's time base (seconds).

Example 2: the FOC is written in seconds while the model uses hours internally

```text
TIME_CONVERSION second_to_hour
PULSE 1500 7200 3600 1
```

---

## 2. Top-Level `foc` Configuration

FOC settings no longer live under `simulation`. They are defined in a dedicated top-level `foc` object.

### 2.1. `foc_path`

- Type: string
- Purpose: path to a `.foc` file
- Recommendation: use a path relative to the config file when possible

### 2.2. `foc_component`

- Type: string
- Purpose: selects which pulse-like child component should be replaced by the FOC schedule
- Requirement: mandatory when FOC is enabled
- Supported selectors: instance name, component path, or component type filter

!!! warning "Always persist foc_component"
    Even if the UI auto-detects a single pulse-like component, the saved task config should still include `foc_component` explicitly.

---

## 3. Input Modes and Entry Points

### 3.1. File mode: `foc_path`

This is the main mode used by the documented examples. It is useful because:

- operating schedules can be versioned independently
- the same model can be tested against multiple operating profiles
- CLI usage, shipped examples, and basic configs all use this path

### 3.2. Inline mode: `foc_content`

`foc_content` is not a general CLI/basic config field. It is a transport field used by backend task payloads and GUI uploads, for example:

```json
{
    "foc": {
        "foc_component": "pulseSource",
        "foc_name": "task_input.foc",
        "foc_content": "PULSE 1500 0.9 0.1 300\nDWELL 100\n"
    }
}
```

This is the mode commonly produced by GUI uploads. The backend materializes `foc_content` into `foc/task_input.foc` inside the task workspace and fills the internal `foc_path`, so the simulation core still follows a single file-based path.

!!! note "CLI/basic still use `foc_path`"
    `tricys -c config.json`, shipped example configs, and basic-run configs currently support the file mode only. In those contexts, keep using `foc.foc_path` instead of writing `foc.foc_content` directly.

---

## 4. Typical Use Cases

- combining short pulse cycles with long dwell periods
- comparing tritium inventory response under different operating schedules
- switching operation schedules without editing the main Modelica model
- using GUI uploads for temporary schedules and example files for curated standard scenarios

---

## 5. Relation to Basic Configuration

Fusion operation config extends [Basic Configuration](basic_configuration.md) with a top-level `foc` block. It does not replace `simulation`.

- `simulation` defines the model, output variables, stop time, and step size
- `foc` defines how a pulse-like component is driven by an operating schedule

If `simulation.stop_time` is shorter than the total FOC schedule duration, TRICYS emits a truncation warning.

---

## 6. Next Step

- To understand why this feature uses a top-level `foc` block plus task-workspace materialization, read [Fusion Operation Principle](../../explanation/tricys_basic/fusion_operation_principle.md).
- To run the example from the shipped example menu, use the entry registered in `tricys/example/example_data/basic/example_runner.json`.