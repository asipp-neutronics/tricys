# TRICYS: Fusion Reactor Plant-Wide Tritium Fuel Cycle Integrated Simulation Platform

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-All--in--One-2496ED.svg?logo=docker&logoColor=white)](https://github.com/asipp-neutronics/tricys/pkgs/container/tricys)
[![Docs](https://img.shields.io/badge/Docs-Five--Pillars-orange.svg)](https://asipp-neutronics.github.io/tricys/)
[![OpenModelica](https://img.shields.io/badge/OpenModelica-1.27+-red.svg)](https://openmodelica.org/)

[English](README.md) | [简体中文](README_zh.md)

![TRICYS V2.0 Banner](./tricys_v2_banner.png)


## 1. Core Positioning (Overview)

**TRICYS** (**TR**itium **I**ntegrated **CY**cle **S**imulation) is an open-source, modular, multi-scale fusion reactor tritium fuel cycle simulator, designed to provide physics-based dynamic closed-loop analysis, strictly adhering to plant-wide mass conservation principles.

Our goal is to provide researchers and engineers with a flexible and robust platform to explore various tritium management strategies, optimize system designs, and gain deep insights into tritium flow and inventory dynamics in fusion reactor environments.

---

## 2. Docker Quick Start (5 Steps)

Experience the full-stack TRICYS multiphysics simulation and digital twin quickly through the out-of-the-box All-in-One single-container image:

### 2.1. Pull the Official Single-Container Image
```bash
docker pull ghcr.io/asipp-neutronics/tricys:latest
```

### 2.2. Launch the Full-Stack Platform and Enter the Terminal
```bash
docker compose up -d
docker exec -it -u appuser tricys_allinone bash
```
> **Seamless Cross-Platform Compatibility**: `docker-compose.yml` has a built-in intelligent dual-track X11 bridging mechanism that works out of the box on both Windows 11 (WSLg) and native Linux (`/tmp/.X11-unix`) by simply running the single command above.

### 2.3. Open Graphical Modeling Environment OMEdit
Enter the command in the container terminal (runs in the background to keep the terminal available; automatically preloads the `example_model` model library and its 20 sub-models by default):
```bash
OMEdit &
```

### 2.4. Run Benchmark Case 01 and Navigate Directly to Visualization via Link
Execute directly in the container terminal:
```bash
tricys example --run 1
```
Upon completion, the terminal will directly output an interactive jump link for the Visualizer (containing an authentication token, e.g., `http://localhost:8050/hdf5/?token=...`). **Simply copy this link and open it in your host browser to view it**; engineering reports in Markdown, Word, and PDF formats covering multi-condition analyses will also be automatically generated in the artifacts directory.

### 2.5. Open the TRICYS Visualization Platform in Browser
Access the full-stack services directly in your host browser:

| Service Name | Access URL | Core Capability Description |
| :--- | :--- | :--- |
| **Unified Web Console** | [http://localhost:8080](http://localhost:8080) | Full-stack reverse proxy gateway, Vue3 simulation workbench, and system monitoring |
| **Digital Twin Dashboard** | [http://localhost:8080/goview/](http://localhost:8080/goview/) *(or :3020)* | GoView fusion reactor plant-wide tritium cycle visualization dashboard |
| **HDF5 Scientific Interactive Plot** | [http://localhost:8080/hdf5/](http://localhost:8080/hdf5/) *(or :8050)* | Dash/Plotly multi-condition curves and tritium self-sufficiency boundary interactive analysis |
| **FastAPI API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI / Swagger interactive REST API documentation and testing |

---

## 3. Documentation Portal

| Core Pillar | Key Topics Index |
| :--- | :--- |
| **1. Tutorials** | Environment setup, 5-minute quick start, and zero-code modification seamless migration guide from V1 to V2 |
| **2. How-To Guides** | Schema 2.0 configuration, transient dynamics, pulsed scheduling, Sobol sensitivity, co-simulation, and mass audits |
| **3. Theory & Methods** | Physical models, software architecture, and mathematical/algorithmic proofs |
| **4. Verification & Benchmarks** | **Abdou 2021 V&V** and 12 major benchmark cases |
| **5. Reference & Specifications** | CLI machine-readable protocol dictionary, complete Schema 2.0 specification, Python API reference, and 4-tier self-healing tree |

---

## 4. Citation & References

### 4.1. Citing the TRICYS Platform
If you use the TRICYS platform in scientific research or academic publications, please cite the core paper in the following format:

```bibtex
@article{tricys2026,
  title   = {TRICYS: A Multi-Scale Integrated Simulation Platform for Fusion Reactor Tritium Fuel Cycle},
  author  = {},
  journal = {},
  year    = {2026}
}
```

### 4.2. V&V & Benchmark References
The system dynamics theoretical kernel and engineering benchmark cases of TRICYS rigorously benchmark and reproduce fusion self-sufficiency theory and engineering design achievements. Please also refer to and cite the following foundational publications:

- **Abdou 2021 Fusion Fuel Self-Sufficiency Benchmark** (fully reproduces the 292 discrete-point theoretical verification dataset, benchmarking *Nuclear Fusion*, 61, 013001; independently open-sourced and hosted at [`couuas/tricys_abdou_2021`](https://github.com/couuas/tricys_abdou_2021)):
```bibtex
@article{abdou2021physics,
  title = {Physics and technology considerations for the deuterium--tritium fuel cycle and conditions for tritium fuel self-sufficiency},
  author = {Abdou, M. A. and Riva, M. and Day, C. and Loarte, A. and Boccaccini, L. V. and Clark, R. and Henders, M. and others},
  journal = {Nuclear Fusion},
  volume = {61},
  number = {1},
  pages = {013001},
  year = {2021},
  doi = {10.1088/1741-4326/abbf35}
}
```

- **Jin 2026 CFEDR Plant-Wide Tritium Cycle System Simulation Benchmark**:
```bibtex
@article{jin2026cfedr,
  title = {CFEDR tritium fuel cycle model and tritium self-sufficiency analysis},
  author = {Jin, Chuan and Zhang, Xiaokang and Gui, Long and Jiang, Yingwu and Zheng, Shanliang},
  journal = {Fusion Engineering and Design},
  volume = {228},
  pages = {115766},
  year = {2026},
  doi = {10.1016/j.fusengdes.2026.115766}
}
```

---

## 5. License & Contributing

- **Issues & Contributions**: Bug reports and improvement suggestions are welcomed via [GitHub Issues](https://github.com/asipp-neutronics/tricys/issues).
- **License**: This project is open-sourced under the [Apache-2.0 License](./LICENSE). Copyright © 2024–2026 ASIPP.
