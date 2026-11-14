# TRICYS V2.0 演示算例 10: 基于 Wang (2022) 真实配置的 h2iso 参数注入协同仿真

## 1. 案例背景与工程意义

在聚变堆氘氚燃料循环中，**氢同位素分离系统（Hydrogen Isotope Separation System, ISS）** 负责从等离子体排灰气（CAP）、中性束注入（NBI）、水脱氚系统（WDS）和包层产氚提取系统（TES）等复杂流股中分离提纯超高纯度 $D_2$ 与 $T_2$。

本算例基于 **Wang et al. (2022)** (*Fusion Engineering and Design*, 184: 113078) 的 CFETR 工业标准级流程配置：
- **内循环系统 (ISS-I)**：包含 4 座低温精馏塔（CD1 ~ CD4）和 2 座同位素交换催化平衡器（EQ1、EQ2），处理高通量排气。
- **外循环系统 (ISS-O)**：包含 3 座低温精馏塔（CD1 ~ CD3）和 1 座同位素催化平衡器（EQ），包含 2 股撕裂循环回收流。

在长周期动态全堆燃料循环仿真（如 5000 小时储氚演化）中，若直接联立求解 7 座低温精馏塔的数百个非线性 MESH 偏微分微分代数方程组（DAE），计算耗时将长达数天甚至发散。

**TRICYS 的参数注入（Parameter Injection）机制**通过在系统仿真前置阶段调用高保真度稳态求解器 **`h2iso`** 求解 MESH 方程，提取塔顶/塔底分离比、摩尔组分、温度分布与水力学滞留时间 $\tau$，并通过 Modelica 0-D 动态代理与 OpenModelica 原生 `-overrideFile` 机制注入系统模型，实现秒级高精度长周期闭环仿真。

---

## 2. 核心架构与多保真协同机制

```
                      +------------------------------------------+
                      |         Wang (2022) 流程配置 JSON        |
                      |   (wang2022_issi.json / wang2022_isso.json)
                      +------------------------------------------+
                                           |
                                           v
[前置外部求解阶段]    +------------------------------------------+
                      |   h2iso MESH 稳态求解器 (run_h2iso_wang2022)   |
                      |     - Wegstein 撕裂流加速算法              |
                      |     - 真实计算 7 座深冷精馏塔与催化平衡器   |
                      +------------------------------------------+
                                           |
                                           v
[参数生成与接口适配]  +------------------------------------------+
                      | 生成 override_cycle.txt 与 Modelica 代理  |
                      |   - Generic_ISS.mo (6-组分分子物理模型)   |
                      |   - Generic_ISS_Adapters.mo (原子/分子适配)|
                      +------------------------------------------+
                                           |
                                           v
[强拓扑校验与模型替换] +------------------------------------------+
                      | TRICYS Submodel Replacer 强拓扑强校验    |
                      |   - 上下游端口因果性、维度与连线拓扑校验   |
                      |   - 无缝替换 i_iss 与 o_iss 子组件        |
                      +------------------------------------------+
                                           |
                                           v
[极速系统长周期仿真]  +------------------------------------------+
                      | OpenModelica 原生极速长周期仿真           |
                      |   - Dassl / Euler 算法 5000h 演化         |
                      |   - 原生 -overrideFile 参数极速注入       |
                      +------------------------------------------+
```

---

## 3. 文件清单与资产说明

本独立文件夹集中管理算例运行所需的所有配置、模型、脚本及基准数据：

| 文件名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| `10_cosim_parameter_injection.yaml` | 配置文件 | TRICYS Schema 2.0 任务配置文件，声明模型、场景、注入任务及指标评估 |
| `run_h2iso_wang2022.py` | 求解脚本 | 任务级 Handler 脚本，真实加载配置并驱动 h2iso 严密求解，输出覆盖参数 |
| `wang2022_issi.json` | 流程配置 | CFETR ISS-I 4塔+2平衡器流程规格定义 (Wang 2022) |
| `wang2022_isso.json` | 流程配置 | CFETR ISS-O 3塔+1平衡器+2撕裂流流程规格定义 (Wang 2022) |
| `example_model.mo` | 系统模型 | 全厂聚变堆燃料循环基准模型 (`example_model.Cycle`) |
| `Generic_ISS.mo` | 物理模型 | 6-组分分子级高保真深冷精馏塔与催化平衡器 0-D 动态代理模型 |
| `Generic_ISS_Adapters.mo` | 适配接口 | 5-维原子同位素流 (H, D, T) $\leftrightarrow$ 6-维分子流适配组件 |
| `override_cycle.txt` | 参数覆盖 | h2iso 求解生成的 OpenModelica 参数覆盖文本 (131 项覆盖参数) |
| `README.md` | 文档说明 | 算例技术规范与操作指南 |

---

## 4. 运行方式

### 4.1 独立案例运行 (推荐)
直接通过 TRICYS CLI 运行本独立目录中的配置文件：
```bash
# 运行 5000h 协同仿真，自动调用 h2iso 求解、校验端口替换模型并生成 Word/PDF 报告
python -m tricys.main run -c tricys/example/example_data/v2_examples/10_cosim_parameter_injection/10_cosim_parameter_injection.yaml
```

### 4.2 独立运行 Handler 求解器
亦可直接运行外部求解脚本，验证 h2iso 对 Wang 2022 流程的计算：
```bash
# 真实运行 h2iso Wegstein 求解并重新生成 override_cycle.txt
python run_h2iso_wang2022.py --mode live --max-iter 10 --tol 0.01
```

---

## 5. 校验与质量保障

- **端口与拓扑强校验**：运行前调用 `validate_port_and_connection_compatibility`，确保替换的 `Generic_ISS_Adapters.ISS_I_Adapter` 与 `Generic_ISS_Adapters.ISS_O_Adapter` 的进出端口与 `example_model.mo` 中原组件完全对齐，杜绝连线悬空。
- **全生命周期产出**：仿真结束后自动在输出目录生成：
  - HDF5 时域轨迹数据 (`simulation_results.h5`)
  - Markdown/Word/PDF 综合工程评估报告
  - 储氚峰值与终态滞留量定量指标
