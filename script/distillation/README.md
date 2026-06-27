# Distillation Column Initialization from h2iso

This script uses the `h2iso` flowsheet solver to compute steady-state
column profiles for the ISS (Isotope Separation System) and generates
Modelica initialization scripts (`.mos` files) for tricys.

## Requirements

```bash
pip install "h2iso[solver]>=0.2.0"
```

Or install from source:
```bash
pip install -e "/path/to/h2iso[solver]"
```

## Usage

```bash
python init_from_h2iso.py --config <flowsheet.json> --output <output_dir>
```

### Arguments

| Argument     | Description                               | Default       |
| ------------ | ----------------------------------------- | ------------- |
| `--config`   | Path to flowsheet JSON configuration      | (required)    |
| `--output`   | Output directory for `.mos` files         | `.` (current) |
| `--max-iter` | Max tear stream iterations                | 50            |
| `--tol`      | Convergence tolerance                     | 1e-4          |
| `--strict`   | Abort with error if flowsheet diverges    | off           |

### Example

```bash
# Solve ISS-O and generate init scripts
python script/distillation/init_from_h2iso.py \
    --config tests/fixtures/wang2022/wang2022_isso.json \
    --output output/iss_init/

# Generated files:
#   CD1_init.mos       - CD1 column initialization
#   CD2_init.mos       - CD2 column initialization
#   CD3_init.mos       - CD3 column initialization
#   init_all_columns.mos - Master script loading all columns
#   init_summary.json  - Solve summary (convergence, iterations)
```

## Output Format

Each `<column>_init.mos` file contains `setParameterValue()` calls that
set initial compositions, temperatures, and flow rates for the
corresponding tricys Modelica column model.

The 5-component composition vector maps to tricys convention:
- `x[1]` = H₂
- `x[2]` = HD + HT (lumped)
- `x[3]` = D₂
- `x[4]` = DT
- `x[5]` = T₂

## Integration with tricys

1. Place the generated `.mos` files in the simulation working directory
2. In your Modelica simulation script, call:
   ```
   runScript("init_all_columns.mos");
   ```
3. This sets initial values before the dynamic simulation starts

## Flowsheet JSON Format

The script accepts the same JSON format used by `h2iso flowsheet --config`.
See `h2iso/tests/fixtures/wang2022/wang2022_isso.json` for an example
ISS-O (outer fuel cycle) configuration with 3 columns and 2 recycle streams.
