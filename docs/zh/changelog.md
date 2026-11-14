---
hide:
  - navigation   
---

TRICYS 遵循[语义化版本规范（Semantic Versioning）](https://semver.org/lang/zh-CN/)：

- **主版本号（Major）**：进行不兼容的 API 变更时递增
- **次版本号（Minor）**：以向后兼容的方式添加新功能时递增
- **修订号（Patch）**：进行向后兼容的问题修正时递增

---

## 2.0.1 ( 未发布 )

### 变更（行为）

* **任一作业失败即返回非零退出码**：此前批量扫描与协同仿真中作业失败只写日志，进程仍返回 `0`，结果文件里只是少了对应的行，与"跑成功"无法区分。现在 `StandardSimulationEngine`（顺序与并发两条路径）与 `OfflineTableEngine` 在任意作业失败时抛出 `JobExecutionError`（携带失败作业 ID 与原因），CLI 退出码为 `1`。**依赖"部分作业失败也返回 0"的脚本需要适配。**

---

## 2.0.0 ( 2026-11-14 )

TRICYS 2.0.0 是一次架构级重构：从面向过程的脚本拼装，转为可插拔流水线 + 模型中心化的工程化底座。

### 新增

* **可插拔工作流引擎**：预处理器 → 仿真引擎 → 后处理器三阶段流水线，阶段之间只通过强类型的 `PipelineContext` 单向流转；支持插件注册与外部扩展。
* **Schema 2.0 双轨配置**：科研 DSL（`simulate` / `sweep` / `sensitivity` / `cosim` / `optimize`）与显式 pipeline 并存；1.0 的 JSON 配置仍可直接解析。
* **时间量纲治理**：新增 `time_base`（`second` / `hour`），配置时间字段恒为小时，由框架换算到模型时基；根除 Issue #99 的 3600× 量纲错位。
* **协同仿真三范式**：离线查表（`offline_table`）、在线步进（`online_step`）、稳态参数注入（`param_inject`）；集成 DWSIM（CAPE-OPEN）与 FESTIM 2.0 偏滤器 FEM 在线步。
* **双层嵌套二分优化引擎**：外层最小启动氚库存寻优 + 内层临界 TBR 逼近，可同时约束在役储备谷底与目标倍增时间。
* **模型自省与审查**：`tricys model inspect` / `tricys model review`，静态审查时间量纲、守恒流与初值，可生成修复补丁；`tricys run --diagnose` 提供结构化失败拦截。
* **可视化与报告**：HDF5 结果看板，报告可导出 Markdown / DOCX / PDF。
* **云原生容器底座**：All-in-One 单镜像集成 OpenModelica、FESTIM、CasADi/H2ISO、FastAPI 后端、Vue3 控制台、GoView 大屏、Dash 看板与 Nginx 网关。
* **算例**：12 个 v2 工业基准算例（01–12）；1.0 算例归档至 `v1_examples_legacy/`。

### 变更（不兼容）

* 新增 `time_base`，默认 `second`。**1.0 时代按小时书写的模型必须显式配置 `time_base: hour`**，否则结果差 3600×。详见《从 TRICYS 1.0 升级到 2.0》。
* `tricys.handlers` 归档至 `tricys.example.example_data.v1_examples_legacy.handlers`，仅保留 `sys.modules` 兼容垫片。
* 移除预编译 FMU / SSP 资产（`example_model_ssp/` 不再随包发布）。
* 包列表中移除 `tricys.handlers`。

### 修复

* **顺序与并发路径的时间基准换算不一致**：1.0 风格配置（时间字段在 `simulation` 下）走顺序路径时不换算，同一配置下发给 OpenModelica 的 stopTime 相差 3600 倍且都报告成功。现在两条路径共用同一换算入口。**修复前用顺序模式跑出的 1.0 存量配置结果视为无效，需重跑。**

### 已知问题

* 作业失败不传导到退出码：批量扫描与协同仿真中部分作业失败时仍返回 `0`，计划在 2.0.1 修复。
* 在线质量审计以初始读数为基准，读数异常（0 / NaN / 负值）不会被拒绝，计划在 2.0.1 修复。

---

## 1.0.0 ( 2025-11-15 )

TRICYS 1.0.0 是项目的首个正式稳定版本，标志着核心功能的完善和生产环境就绪。

* **参数扫描与并发**：系统地研究多个参数对系统性能的影响，支持并发运行和大规模批量仿真。
* **子模块协同仿真**：支持与外部工具（如 Aspen Plus）进行数据交换完成子模块系统集成。
* **自动化报告生成**：自动生成标准化的 Markdown 分析报告，包含图表、统计数据和可视果。
* **参数敏感性分析**：支持系统参数的自定义敏感性分析，并集成SALib（SensitivitAnalysis Library in Python）库量化参数对输出的影响。