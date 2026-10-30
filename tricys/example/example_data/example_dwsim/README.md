# TRICYS DWSIM 协同仿真模型库 (example_dwsim)

本目录为 **TRICYS 全厂聚变堆多物理场协同仿真平台** 中关于 **DWSIM（开源 CAPE-OPEN 化工流程模拟器）** 的基准模型与算例资产库。

---

## 1. 定位与优势
* **开源自闭环**：作为商业闭源软件 Aspen Plus（`example_aspenbkp/T2-Threetowers4.bkp`）的开源替代方案。
* **Linux 容器原生支持**：DWSIM 基于 .NET 跨平台核心开发，可通过 `pythonnet` 在 TRICYS All-in-One Linux Docker 容器（Ubuntu 24.04）中无头（Headless）运行，彻底摆脱 Windows COM 依赖。
* **多物理场协同**：配合两阶段离线查表替代模式（`mode: "offline_table"`），与 OpenModelica 全厂宏观动力学模型（`example_model.Cycle`）完成闭环协同与质量守恒审计。

---

## 2. 经典模型配置：同位素分离系统 (ISS) 三塔低温精馏
对应算例：`08_cosim_dwsim_offline_table.yaml`

### 2.1 流股接口定义 (Stream Conventions)
| 流股名称 (Stream) | 流向 (Direction) | 物理含义 (Description) | 对应 Modelica 端口 |
| :--- | :--- | :--- | :--- |
| **FROMTEP** | 进料 (Feed) | 来自托卡马克尾气净化系统 (TEP) 的氢同位素原料流 | `tep_fcu.outflow[1..3]` |
| **WDS** | 出料 (Top Waste) | 塔顶富集轻同位素 (H2/HD) 废氢流，排往水去氚系统 (WDS) | `i_iss.to_WDS[1..3]` |
| **SDSD2** | 出料 (D2 Product) | 第二塔塔顶富集氘气产物流，输往储氢储氚系统 (SDS) | `i_iss.to_SDS[2]` |
| **SDST2** | 出料 (T2 Product) | 第三塔塔底高纯度氚气产物流 (>98% T2)，输往储氢储氚系统 (SDS) | `i_iss.to_SDS[1]` |

### 2.2 组分定义 (Compounds)
* 六组分氢同位素系统：`H2`, `HD`, `D2`, `HT`, `DT`, `T2`
* 热力学物性方法：Peng-Robinson (PR) 或 Lee-Kesler-Plocker (LKP) 低温氢物性修正模型。

---

## 3. 自动化调度方式
在协同仿真执行时，`tricys.handlers.dwsim_iss_handler:run_dwsim_simulation` 会自动：
1. 从阶段一初步仿真提取进料通量 `temp_input_csv`；
2. 计算同位素交换平衡分布并注入 DWSIM 进料流股；
3. 动态推进 DWSIM 解算；
4. 提取出料流股并构建满足 OpenModelica `CombiTimeTable` 格式的重构数据表 `temp_output_csv`。
