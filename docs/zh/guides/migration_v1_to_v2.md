# 从 TRICYS 1.0 升级到 2.0

一句话：**配置里的时间字段仍然写小时；2.0 新增的 `time_base` 说明「模型是用秒还是小时书写的」。**

老模型（1.0 时代按小时写）请在配置里显式写 `time_base: hour`，行为与 1.0 完全一致；
新模型按 SI 秒书写，用默认的 `second`，框架会把配置里的小时值乘 3600 后下发给 OpenModelica。

---

## 1. 时间基准（升级时唯一必须动脑的地方）

| `time_base` | 模型怎么写时间 | 框架下发给 OMC 的 stopTime |
| --- | --- | --- |
| `second`（默认） | 模型内部按秒 | 配置值（小时）× 3600 |
| `hour` | 模型内部按小时 | 配置值（小时）原样 |

同一个 `stop_time: 2000`：

- `time_base: second` → OMC 收到 `7 200 000`（秒）
- `time_base: hour` → OMC 收到 `2 000`（小时，与 1.0 一致）

> **2.0.1 起**：如果配置里既没有 `time_base` 也没有 `scenario` 段（即 1.0 形状），框架会打一条 WARNING 提示 3600 倍风险。它只是提示，不会替你改默认值——默认仍是 `second`。

**放在哪：**

- 2.0 配置（YAML，`scenario` 块）：`scenario.time_base`
- 1.0 存量配置（JSON，`simulation` 块）：`simulation.time_base`

**怎么判断自己的模型属于哪一种：**

```bash
python -m tricys model review path/to/package.mo example_model.Cycle
```

看输出里的 `time_scale`：`3600` 表示秒基准，`1` 表示小时基准。
如果顶层模型没有声明 `time_scale`，会报 `TIME_SCALE_MISSING`（CRITICAL）—— 这种模型就是 1.0 时代的写法，
升级时应当配 `time_base: hour`，或者按 2.0 规范补 `inner parameter Real time_scale = 3600;` 并核对全部 `der(...)` 项。

---

## 2. 破坏性变更清单

| 变更 | 影响 | 处理方式 |
| --- | --- | --- |
| 新增 `time_base`，默认 `second` | 不写这个键，老模型会被当成秒基准模型，结果差 3600× | 老模型显式写 `time_base: hour` |
| `tricys.handlers` 归档 | 模块迁到 `tricys.example.example_data.v1_examples_legacy.handlers`，`tricys/__init__.py` 保留了 `sys.modules` 兼容垫片 | 新代码直接改用新路径；垫片失败时静默，别依赖它 |
| 预编译 FMU / SSP 资产移除 | `example_model_ssp/` 不再随包发布，SSP 示例归档 | 自行构建，或改用 2.0 的协同仿真算例 08–10 |
| v1 算例归档 | 原 `basic/`、`analysis/` 示例移到 `v1_examples_legacy/` | 引用这些路径的脚本/文档需更新 |
| 包列表移除 `tricys.handlers` | 打包脚本若显式列了这个包会失败 | 从打包清单里去掉 |

---

## 3. 配置格式没有被强制更换

`tricys validate` / `tricys run` 仍然接受 1.0 的 JSON 结构（`paths` + `simulation`）。
2.0 新增的科研 DSL（`scenario` + `task`，支持 `simulate / sweep / sensitivity / cosim / optimize`）和显式 pipeline 是**可选**路径，不是强制迁移项。

```bash
python -m tricys validate -c config.json
python -m tricys run -c config.json
```

---

## 4. 迁移检查清单

- [ ] `python -m tricys validate -c <配置>` 通过
- [ ] `python -m tricys model review <package.mo> <model>`，确认 `time_scale` 与 `time_base` 匹配
- [ ] 老模型：配置里补 `time_base: hour`
- [ ] 用一个小 `stop_time` 跑一次，与 1.0 的同配置结果对照时间轴与末态
- [ ] 若代码里 `import tricys.handlers`，改成归档后的新路径
- [ ] 若引用 `example_model_ssp` / `*.fmu` 预编译资产，改用 v2 协同仿真算例

---

## 5. 2.0.0 已知问题

- **作业失败不传导到退出码**：批量扫描与协同仿真中部分作业失败时，命令仍返回 `0`，结果文件里只是少了对应作业。计划在 2.0.1 修复；在此之前请检查结果文件中的作业数量。
- **审计器初始质量无合理性校验**：在线质量审计以初始读数为基准，读数异常（0、NaN、负值）不会被拒绝。计划在 2.0.1 修复。

---

## 6. 如果你在 2.0.0 正式版之前用过 v2.0 分支

顺序执行模式对 1.0 存量配置漏做了一次时基换算，同样的配置得到的时间跨度是正确值的 1/3600。
该问题已在 2.0.0 中修复（顺序与并发路径现在共用同一换算入口）。**在修复前用顺序模式跑出的 1.0 存量配置结果应视为无效**，需要重跑。
