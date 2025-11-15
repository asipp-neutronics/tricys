# 氚燃料循环 0 维系统模型

## 1. 氚循环系统介绍

聚变堆氚燃料循环在架构上分为两个高度耦合的系统：内部燃料循环和外部燃料循环。

![cycle.svg](../../assets/cycle_system.png)

### 1.1. 内部燃料循环
这是一个处理未燃燃料的高通量、快速闭环回路 。其物质流路径如下：

1)  **注入 (Fuelling):** 循环始于燃料储存与输送系统 (SDS)。高纯 D-T 燃料经由加料系统 (FS) 注入托卡马克真空室 (Plasma) 。
  
2)  **排出 (Exhaust):** 在等离子体中，仅有少量燃料发生聚变, 大部分未燃燃料、聚变产物（氦灰）及杂质被引导至偏滤器 (Divertor) 区域 。
  
3)  **泵送 (Pumping):** 真空泵系统 (Pump_System) 将这些排出的高温废气抽出 。
  
4)  **净化 (TEP):** 废气被送至托卡马克排气处理系统 (TEP)。TEP 在此进行关键的化学净化，以从杂质（如氚化水 Q2O 和氚化甲烷 CQ4）中回收氢同位素 。
  
5)  **分离 (ISS):** 净化后的氢同位素 (Q2) 混合气流被送至内部同位素分离系统 (I-ISS)，通常通过低温精馏技术进行同位素分离 。
 
6) **回流 (Return):** 最终，分离出的高纯度 D2 和 T2 燃料被送回 SDS，实现燃料的再循环，完成内部闭环 。

### 1.2. 外部燃料循环
这是一个负责生产新燃料以实现氚自持的低通量、慢速回路 。其物质流路径如下：

1)  **增殖 (Breeding):** 聚变产生的高能中子进入增殖包层 (Blanket)，与包层中的锂 (Li) 发生核反应以增殖新的氚 。
  
2)  **提取 (Extraction):** 新生的氚通过增殖剂氚提取系统 (TES) 从包层材料中移出 。
  
3)  **泄漏与净化 (Permeation & Purification):** 与此同时，少量氚会不可避免地渗透进入冷却剂回路 (CL) 。冷却剂净化系统 (CPS) 负责从冷却剂中捕获并回收这部分渗透的氚 。
  
4)  **汇集与分离 (Collection & Separation):** 来自 TES 的富氚流（例如，固态包层的氦气吹扫气或液态包层的渗透器产气 ）与来自 CPS 的回收氚流汇合，一同被输送至外部同位素分离系统 (O-ISS) 进行提纯。
  
5)  **补充 (Replenish):** O-ISS 将氦气及其他杂质去除后，把高纯度氚送入 SDS，补充到主燃料循环中，完成全厂的燃料闭环 。



## 2. Modelica示例模型

TRICYS 的核心是一个氚燃料循环 0 维系统模型，该模型将聚变堆的氚燃料循环系统抽象为一系列相互连接的子系统，每个子系统代表实际工厂中的一个关键功能模块。

0 维模型意味着我们关注的是系统级的物质流动和库存变化，而不是详细的空间分布。这种建模方法特别适合用于：

- 系统级的氚库存分析
- 燃料自持时间评估
- 设计参数优化
- 运行策略研究
- 安全性评估

![cycle.svg](../../assets/cycle.svg)

| 模型缩写 (Abbreviation) | 中文全称 (Chinese Full Name) | 英文全称 (English Full Name) |
| :--- | :--- | :--- |
| `Plasma` | 等离子体 | Plasma |
| `Fueling_System` | 燃料注入系统 | Fueling System |
| `Pump_System` | 真空泵系统 | Vacuum Pumping System |
| `TEP_FEP` | 托卡马克排气处理 - 前端 | Tokamak Exhaust Processing - Front-End Processing |
| `TEP_IP` | 托卡马克排气处理 - 中间处理 | Tokamak Exhaust Processing - Intermediate Processing |
| `TEP_FCU` | 托卡马克排气处理 - 最终净化单元 | Tokamak Exhaust Processing - Final Cleanup Unit |
| `I_ISS` | 内部同位素分离系统 | Inner Isotope Separation System |
| `SDS` | 储存与输送系统 | Storage and Delivery System |
| `Blanket` | 增殖包层 | Breeding Blanket |
| `TES` | 氚提取系统 | Tritium Extraction System |
| `O_ISS` | 外部同位素分离系统 | Outer Isotope Separation System |
| `FW` | 第一壁 | First Wall |
| `DIV` | 偏滤器 | Divertor |
| `Coolant_Pipe` | 冷却剂回路 | Coolant Pipe |
| `CPS` | 冷却剂净化系统 | Coolant Purification System |
| `WDS` | 水去氚系统 | Water Detritiation System |

## 3. 进一步学习

- **[快速开始](../quickstart.md)**：运行您的第一个仿真
- **[基础配置](../tricys_basic/basic_configuration.md)**：了解如何配置模型参数
- **[参数扫描](../tricys_basic/parameter_sweep.md)**：系统地研究参数空间
- **[敏感性分析](../tricys_analysis/single_parameter_sensitivity_analysis.md)**：识别关键参数