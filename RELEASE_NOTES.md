# TRICYS 发布说明

## 2.0.1 — 2026-11-16

基线：`v2.0.0`（2026-11-14）。本次为修复版，无新增功能，但包含 **4 处行为收紧**，升级前请读"升级提示"。

### 修复

1. **多文件 Modelica 包（`package.mo`）无法运行。**
   OMPython 4 会把 `.mo` 复制到自己的临时工作目录再加载，`package.mo` 因此与同目录的其它 `.mo` 分离、包名与临时目录名不匹配，OMC 直接拒绝。仓库内 6 个 v1 存量配置因此完全跑不出结果。现在加载前会先把整个包目录暂存到以包名命名的目录。

2. **在线质量审计的初始基线未校验。**
   初始存量读到 `NaN` / 无穷 / 负值、全部库存变量读取失败、或审计模式未匹配到任何库存变量时，旧代码照单全收，之后所有质量衡算都建立在这个错误起点上。现在初始化阶段直接报错（`AuditorInitializationError`）；部分变量读取失败仍只告警，初始总量恰为 0 只警告。

3. **作业失败不传导到退出码。**
   批量扫描与协同仿真中作业失败只写日志、进程仍返回 `0`，结果文件里只是少了行。现在**任一作业失败即返回非零退出码**（`JobExecutionError`）。

### 行为变更

| 变更 | 影响 | 适配 |
|---|---|---|
| 任一作业失败 → 非零退出码 | 依赖"部分失败也返回 0"的脚本/CI 会变红 | 检查退出码处理；需要容忍部分失败时改判结果文件中的作业数 |
| 审计器初始化拒绝不可信基线 | 配置错误会从"跑出结果"变成"启动即报错" | 核对 `auditor.patterns.inventory` 是否匹配模型变量 |
| v1 形状缺省时基告警 | 仅新增一条 WARNING，默认值仍为 `second` | 无需动作；若要消除告警，显式设 `time_base` |
| `pyproject.toml` 协议声明修正为 Apache-2.0 | 发布元数据与 LICENSE 一致（此前 classifier 写 MIT） | 无需动作 |

### 升级提示

- **1.0 时代按小时书写的模型**：在配置里显式设置 `time_base: "hour"`（v1 配置放在 `simulation` 段，v2 配置放在 `scenario` 段）。否则框架按 `second` 处理，时间跨度会差 3600 倍。详见《从 TRICYS 1.0 升级到 2.0》。
- 用 `python -m tricys model review <package.mo> <model>` 可以确认模型的 `time_scale` 是 `3600`（秒基准）还是 `1`（小时基准）。

### 已知问题

- `v1_examples_legacy/basic/9_online_cosim` 无法直接运行：其 SSP 拓扑来自仓库外的 CFEDR 模型，且运行需要 OMSimulator 与 Aspen。2.0 的对应算例是 `v2_examples/09_cosim_online_step.yaml`。
- `v1_examples_legacy/basic/4_co_simulation_module` 需要 Aspen Plus（`example_aspenbkp/T2-Threetowers4.bkp`）。
- 分析类示例中的 Sobol / Latin / 单参数敏感性算例计算时间较长（分钟级），不属失败。

### 验证

| 项 | 结果 |
|---|---|
| `pytest -m build_test` | 110 passed |
| `pytest -m "not slow"` | 375 passed / 4 skipped / 0 failed |
| `pytest -m slow` | 11 passed / 5 skipped |
| `pytest test/test_example_runner.py` | 5 passed |
| CI（单元测试 / 容器多物理场 / 镜像构建） | 全绿 |
| `mkdocs build --strict` | 0 warning / 0 死链（已锁进部署） |

---

更早版本的变更记录见 [`docs/zh/changelog.md`](docs/zh/changelog.md) 与 [`docs/en/changelog.md`](docs/en/changelog.md)。
