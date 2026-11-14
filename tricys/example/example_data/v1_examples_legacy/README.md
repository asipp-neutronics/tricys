# TRICYS V1.0 遗留算例套件 (Legacy Examples)

> [!WARNING]
> **历史版本存档提示 (V1.0 Legacy Archive)**  
> 本目录 (`v1_examples_legacy/`) 下保存的配置文件及算例为 **TRICYS V1.0 版本遗留资产 (Legacy Artifacts)**。  
> 随着 TRICYS 架构全面升级至 **Schema 2.0 (V2.0)**，本目录中的 JSON 配置文件主要用于历史回归测试与向后兼容性验证，**可能与当前版本的核心求解器流水线及前置/后置处理器存在不兼容**。  
> 建议所有新项目与科研仿真工作使用现代化 Schema 2.0 工业级基准算例：请参见上层目录中的 [`v2_examples/`](../v2_examples/)。

---

## 1. 架构差异与兼容性说明

TRICYS V1.0 与 V2.0 存在重大架构演进，主要差异体现在以下几个核心维度：

### 1.1 时间基准机制演进 (`time_scale` / `time_base`)
- **V1.0 历史机制**：
  - 依赖于隐式或局部的 `time_scale` 与 `time_base` 转换逻辑。
  - 不同子模块（如 Basic 仿真、FOC 调度、后处理指标提取）对秒（Second）与小时（Hour）的换算定义较为分散，需要手动在 JSON 中协调单位一致性。
- **V2.0 现代化机制**：
  - 在 Schema 2.0 DSL 的 `scenario` 字段中提供了统一声明式物理时间总线：TRICYS 配置默认一律采用小时（hour，如 `stopTime: 2000.0` 即 2000 小时），并通过 `time_base` 显式告知底层 Modelica 模型的基本时间单位（`second` 或 `hour`），由 TRICYS 自动完成 OMC 求解入参转换与后处理对齐：
    ```yaml
    scenario:
      time_base: "second"  # 告知底层 Modelica 模型基本单位是秒（若为金川等以小时为基准的模型则填 "hour"）
      startTime: 0.0       # TRICYS 默认单位：小时 (hours)
      stopTime: 2000.0     # TRICYS 默认单位：小时 (hours)
      stepSize: 0.5        # TRICYS 默认单位：小时 (hours)
    ```
  - 底层由 `TimeBaseCrosschecker` 与物理语义流水线自动保障跨求解器（Modelica / FMI / Aspen Plus / Python Handler）时间步长的严格守恒与无量纲换算。

### 1.2 工作流与管道拓扑
- **V1.0**：分为独立的 `basic` 和 `analysis` 命令行子系统（分别采用 `basic_configuration.json` 与 `simulation_analysis.json`）。
- **V2.0**：完全统一为声明式流水线架构（`PipelineBuilder`），支持 `simulate`、`sweep`、`sensitivity`（含 Sobol GSA）、`cosim`（含离线查表、在线步进与参数注入）等统一任务原语，并通过 DAG 拓扑图执行 Kahn 自动排序。

---

## 2. 目录结构概览

```text
v1_examples_legacy/
├── README.md               # 本说明文档
├── basic/                  # V1.0 Basic 仿真工况遗留配置文件
│   ├── 1_basic_configuration/
│   ├── 2_parameter_sweep/
│   ├── 3_post_processing_module/
│   ├── 4_co_simulation_module/
│   ├── 5_fusion_operation_config/
│   ├── 6_constrained_buffer/
│   ├── 7_blanket_constraints/
│   ├── 8_offline_auditor/
│   ├── 9_online_cosim/
│   └── example_runner.json
└── analysis/               # V1.0 Analysis 敏感性/不确定性分析遗留配置文件
    ├── 1_baseline_condition_analysis/
    ├── 2_single_parameter_sensitivity_analysis/
    ├── 3_multi_parameter_sensitivity_analysis/
    ├── 4_sobol_global_sensitivity_analysis/
    ├── 5_latin_uncertainty_analysis/
    └── example_runner.json
```

---

## 3. 推荐迁移建议

若您需要复现或使用类似功能，请参考对应迁移至 V2.0 的等价案例：

| V1.0 遗留算例 (`v1_examples_legacy`) | V2.0 现代化算例 (`v2_examples`) | 核心特性 |
| :--- | :--- | :--- |
| `basic/1_basic_configuration` | `02_simulate_pulsed_transient.yaml` | 单堆瞬态仿真 + 质量守恒审查 |
| `basic/2_parameter_sweep` | `01_sweep_startup_self_sufficiency.yaml` | 多工况并行参数扫描 + 氚自持评估 |
| `basic/5_fusion_operation_config` | `05_simulate_foc_pulsed_scenario.yaml` | FOC 聚变混合工况脉冲调度注入 |
| `basic/8_offline_auditor` | `06_simulate_mass_conservation_audit.yaml` | 全系统物理语义质量守恒闭环离线审计 |
| `basic/4_co_simulation_module` | `08_cosim_offline_table.yaml` | 协同仿真：离线查表/两阶段替代模式 |
| `basic/9_online_cosim` | `09_cosim_online_step.yaml` | 协同仿真：FMI/SSP 在线步进双向同步 |
| `analysis/4_sobol_global_sensitivity_analysis` | `03_sensitivity_sobol_gsa.yaml` | 高维参数全局灵敏度与方差分解 (Sobol GSA) |
| - | `04_sweep_pulse_width_required_tbr.yaml` | 脉宽参数扫描与 Required TBR 嵌套二分搜索 |
| - | `07_custom_plugin_pipeline.yaml` | 自定义 Python 插件挂载与物理语义总线扩展 |
| - | `10_cosim_parameter_injection.yaml` | CFETR ISS-I/ISS-O 稳态求解参数快速注入 |

---
*TRICYS 开发组归档于 2026 年*
