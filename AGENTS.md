# AGENTS.md — TRICYS Project Guide

## Project Overview

TRICYS (TRitium Integrated CYcle Simulation) is a modular, multi-scale fusion reactor tritium fuel cycle simulator. It enforces plant-wide mass conservation and provides physics-based dynamic closed-loop analysis.

## Project Structure

- `tricys/` — Core simulation engine (physics modeling, Modelica integration, parameter scanning, sensitivity analysis)
- `tricys_backend/` — FastAPI backend service (task queues, WebSocket log streaming, HDF5 data retrieval)
- `tricys_goview/` — Vue 3 GoView-based data dashboard
- `test/` — Test suite (analysis, core, dwsim, handlers, postprocess, utils, visualizer)
- `docs/` — Documentation (mkdocs-based)
- `example/` — Example simulations and demos
- `docker/` — Docker configuration
- `script/` — Utility scripts
- `tools/` — Auxiliary tools

## Tech Stack

- **Language**: Python 3.8+
- **Simulation**: OMPython (OpenModelica), DWSIM
- **Data**: pandas, numpy, HDF5 (tables), SALib (sensitivity analysis)
- **Visualization**: matplotlib, seaborn, plotly, dash
- **AI**: OpenAI LLM integration for report generation
- **Backend**: FastAPI, WebSocket
- **Frontend**: Vue 3, GoView

## Common Commands

```bash
make install        # Install in editable mode
make dev-install    # Install with dev dependencies
make test           # Run tests
make lint           # Check code style
make format         # Auto-format code
make check          # Format then lint
make docs-serve     # Serve docs locally
make deploy         # Interactive deployment wizard
```

## Conventions

- Python code follows PEP 8; use `make format` for auto-formatting.
- Test files live in `test/` mirroring the package structure.
- Tests tagged with `@pytest.mark.build_test` (function decorator or module-level
  `pytestmark = pytest.mark.build_test`) are what the "HDF5 Image Build Test" job
  runs (`pytest -m build_test test` in `.github/workflows/build_test.yml`); keep
  them free of OMC / h2iso / DWSIM dependencies so they stay runnable in that image.
- The OMC-backed hour-vs-second cross-check has its own job in
  `.github/workflows/time_base_crosscheck.yml`.
- Simulation inputs use Modelica `.mo` files; DWSIM uses `.dwxmz`.
- Reports are auto-generated as Markdown.
- HDF5 is used for large simulation data storage.

## Git Remotes

| Remote    | URL |
|-----------|-----|
| origin    | `git@github.com:zxkjack123/tricys.git` (SSH, primary push) |
| upstream  | `https://github.com/asipp-neutronics/tricys.git` (upstream source) |
| asipp     | `https://github.com/asipp-neutronics/tricys.git` (alias of upstream) |
| couuas    | `https://github.com/couuas/tricys.git` (UAS collaborator fork) |
