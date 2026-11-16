# AC2 回归验证：v1.0 存量配置兼容性

> 验证时间：2026-11-14｜验证对象：TRICYS 2.0.0（main @ `e740e5e`）｜基线：v1.1.0（`e22f65a`）
> 对应 Stage 5 验收标准 AC2：**现存所有 v1.0 JSON 配置文件无需做任何人工修改，即可被 `tricys` CLI 正常解析并准确计算。**

## 结论

**AC2 未达成。** 仓库内 16 个真实 v1.0 配置（另 2 个 `example_runner.json` 是算例清单，不是配置）：

| 分类 | 数量 | 说明 |
|---|---|---|
| 正常运行（exit 0） | 6 | basic 1/2/3/5，analysis 1/3 |
| 运行失败（exit 1） | 6 | basic 4/6/6-sweep/7/7-sweep/8 |
| 240 s 未跑完（未结论） | 3 | analysis 2/4/5 |
| 解析即失败 | 1 | basic 9（依赖资产已被删除） |

即使运行成功的那些，也只在"随仓升级过的秒基准模型"上成立；**换成用户自带的 v1 小时制模型，同一份配置会算成 3600 倍**（见 §3）。

## 方法

- 环境：本机 `.venv`（Python 3.12.3，OMC 1.27，OMPython 4.x）
- v2 侧：`main` @ `e740e5e`（2.0.0 发布后、2.0.1 B2 已合入）
- v1 基线：独立 worktree @ `v1.1.0`（`e22f65a`），用 `PYTHONPATH` 隔离导入，确认 `tricys.__file__` 指向 worktree
- 配置来源：`tricys/example/example_data/v1_examples_legacy/**`，未做任何修改
- 判据：CLI 退出码 + 产出 H5/CSV 的时间轴与末态

## 1. 解析阶段

`tricys validate -c`：16 个真实配置中 15 个通过。

唯一失败：`basic/9_online_cosim/online_cosim.json`

```
ERROR: File specified in 'paths.package_path' not found:
  .../v1_examples_legacy/example_model_ssp/ssp/example_model.ssp
```

`example_model_ssp/` 是 2.0.0 移除的预编译 SSP 资产（见 `changelog` 的破坏性变更）。这条配置**必须人工修改或删除**，直接违反 AC2 的"无需修改"。

## 2. 运行阶段：多文件包加载失败（v1/v2 同因）

6 个失败的配置全部使用多文件包（`example_model/package.mo`）。失败形态一致：

```
loadFile("/tmp/<uuid>/package.mo")
  Expected the package to have name <uuid>, but got example_model.
```

根因在 OMPython 4.x 的 `ModelicaSystemOMC`：它把 `package.mo` 复制进 OMC 的工作目录 `/tmp/<uuid>/`，OMC 按目录名推断包名，于是包名不匹配直接拒绝加载。仅复制 `package.mo` 也必然会丢失同目录的其它 `.mo`。

**这不是 v2 独有的回归**：用同一套依赖跑 v1.1.0 的代码，错误一字不差。区别只在退出码——v1.1.0 吞掉异常返回 `0`（即 2.0.1 修掉的 B2），v2.0 返回 `1`。

所以对 AC2 的影响是"配置跑不出结果"，但账不能只记在 2.0.0 头上：它与 OMPython 4.x 升级绑定，需要一个独立的修复项。

## 3. 数值阶段：时间基准 3600 倍偏差（最严重）

这是唯一能让正确配置**静默算错**的一类。

对照实验，三次运行使用**同一个模型文件**（v1.1.0 时代的小时制 `example_model_single/example_model.mo`）和**同一份配置**（`basic_configuration.json`，`stop_time=2000`）：

| 运行 | 版本 | 配置改动 | 时间轴 max | 墙钟 |
|---|---|---|---|---|
| A | v1.1.0 | 无 | **2 000** | 8 s |
| B | 2.0.0 | 无（原样） | **7 200 000** | 233 s |
| C | 2.0.0 | 仅加一行 `time_base: "hour"` | **2 000** | 5.3 s |

B 与 A 相差 3600 倍，退出码 0、无任何告警。C 证明**唯一原因就是缺失的 `time_base` 默认为 `second`**：小时制模型被当成秒制，框架把 2000 小时乘成 7 200 000 秒再交给它。

补充对照：仓库自带的那份配置（`v1_examples_legacy` 内）指向的是**已升级为秒基准**的模型（`time_scale = 3600`），此时 2.0.0 产出 7.2e6 秒 = 2000 小时，物理上是正确的。也就是说 AC2 只在"模型也一起升级"的前提下成立，对用户手里的存量模型不成立。

**数值等价性未验证**：v1.1.0 的 CSV/H5 导出在这些配置下只写 `time` 一列（`variableFilter` 里 `sds.I[1]` 的方括号未转义，v2 在 `basic_prepare_config` 里补了转义），拿不到末态存量做逐点比对。时间跨度已足以判定"不准确"，但**末态数值一致性仍属未验证，不可宣称通过**。

## 4. 未结论项

`analysis/2`（单参数敏感性）、`analysis/4`（Sobol）、`analysis/5`（Latin）在 240 s 上限内未跑完（样本数多，属正常耗时），没有观察到失败，但也**不能记为通过**。

## 处置结论（2026-11-15 裁决）

1. **时基默认值保持 `second`（SI），不改为 `hour`。**
   理由：v2 的 12 个算例全部是秒基准模型（`time_scale = 3600`），把默认改回 `hour` 会让它们在默认路径上反向错 3600 倍——等于用一个 bug 换另一个；而且默认值描述的是"模型时基"，不是"配置时间单位"（配置恒为小时）。
   **改为显式告警**：`StandardSimulationEngine` 检测到 v1 形状（有 `simulation` 段、无 `time_base`、无 `scenario`）时打一条 WARNING，指名 3600 倍风险与修复方式；默认值不变。
   **因此 AC2 的措辞应调整为**"v1.0 配置可被解析并运行，缺省时基会给出明确的 3600 倍风险提示"，而不是"无需修改即准确计算"。
2. **`9_online_cosim` 移出 AC2 统计口径。** 该算例的 SSP 拓扑来自仓库外的 CFEDR 模型（配置引用 `CFEDR.I_ISS`，本仓库没有 `package CFEDR`），用仓内 `example_model` 重建只能得到 2 个 FMU（原 SSP 为 18 个组件）；运行还需要 OMSimulator 与 Aspen。标注为"需外部模型与外部运行时"的遗留算例，2.0 的替代算例是 `v2_examples/09_cosim_online_step.yaml`。
3. **多文件包加载缺陷单独立项**：`ModelicaSystemOMC` 把 `package.mo` 复制到 `/tmp/<uuid>/` 导致包名不匹配；用同一套依赖跑 v1.1.0 同样复现，不是 2.0 独有。修法是绕开该复制行为（`load_modelica_package` 已能正确加载）。

按调整后的口径，AC2 仍未达成：剩余缺口是上述第 3 条（多文件包）与第 3 节的时基偏差。

## 5. 2.0.1 修复与复验（2026-11-16）

§2 的多文件包缺陷已修：加载前先把整个包目录暂存到「以包名命名的目录」下，OMPython 计算出的路径与传入路径一致，于是原地加载、同目录的其它 `.mo` 也都在。改动覆盖顺序引擎、顺序扫描与协同仿真三处 `ModelicaSystem` 构造点。

复跑同一批配置（未做任何修改）：

| 配置 | 修复前 | 修复后 |
|---|---|---|
| `basic/6_constrained_buffer` | exit 1 | **exit 0**（35 s） |
| `basic/6_constrained_buffer_sweep` | exit 1 | **exit 0**（358 s） |
| `basic/7_blanket_constraints` | exit 1 | **exit 0**（27 s） |
| `basic/7_blanket_constraints_sweep` | exit 1 | **exit 0**（378 s） |
| `basic/8_offline_auditor` | exit 1 | **exit 0**（28 s） |
| `basic/4_co_simulation_module` | exit 1 | exit 1（Aspen 处理器，与本缺陷无关） |

`7_blanket_constraints` 的输出时间轴 `time_max = 1.8e7 s = 5000 h × 3600`，与配置的 `stop_time=5000`（小时）自洽。

按此更新口径：16 个配置中 **11 绿 / 1 红（Aspen）/ 3 未结论 / 1 解析失败（9 号）**。AC2 仍未达成，剩余缺口是 Aspen 依赖与 9 号算例的外部资产；原"多文件包"缺口已关闭。
