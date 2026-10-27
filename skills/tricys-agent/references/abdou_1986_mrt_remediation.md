# Abdou et al. (1986) 集中参数动态微分方程与 Agent 智能修复准则

> **理论参考文献**:  
> *M. A. Abdou, E. L. Vold, C. Y. Gung, M. Z. Youssef, K. Shin, "Deuterium-tritium fuel self-sufficiency in fusion reactors," Fusion Technology, Vol. 9, March 1986, pp. 250–285.*

---

## 📌 1. 文献核心理论地位与物理奠基

Abdou et al. (1986) 是聚变工程界公认的**全厂动态物料平衡与氚自持动力学理论奠基之作**。该文献首次建立了：
1. **集中参数动态微分方程体系 (Lumped Parameter Dynamic ODE System)**：将聚变回路各个复杂化工与真空物理过程抽象为集中参数存量节点；
2. **平均停留时间方法 (Mean Residence Time, MRT)**：通过特征时间常数 $\tau_k$ 与铺底滞留量 $I_{hold, k}$ 描述回路各节点的动态周转；
3. **自持所需增殖比解析理论 ($TBR_{req}$)**：精确建立了 $TBR$ 临界需求与燃烧份额 $f_b$、各回路停留时间 $\tau_k$、储备储备天数 $t_r$ 及非放射性损失 $\epsilon_k$ 之间的定量泛函关系。

**在 TRICYS 中的定位**：TRICYS 平台系统级宏观 Modelica 模型的动力学拓扑与物料平衡方程均源自 Abdou (1986) 的集中参数理论基石。

---

## ⚛️ 2. Abdou (1986) 集中参数动力学控制方程

对全厂任意子系统 $k$（如 PS、TEP、ISS、WDS、SDS 等），其有效在役存量 $I_k(t)$ 遵循宏观物料守恒微分代数方程：

$$\frac{\mathrm{d}I_k(t)}{\mathrm{d}t} = \dot{m}_{in, k}(t) - \dot{m}_{out, k}(t) - \dot{m}_{decay, k}(t) - \dot{m}_{loss, k}(t)$$

### 2.1 MRT 流股本构关系 (Constitutive Flow Relation)
基于平均停留时间 $\tau_k$ 与铺底门槛（Hold-up Threshold）$I_{floor, k}$：

$$\dot{m}_{out, k}(t) = \begin{cases} 
\frac{I_k(t) - I_{floor, k}}{\tau_k}, & I_k(t) > I_{floor, k} \\
0, & I_k(t) \le I_{floor, k}
\end{cases}$$

### 2.2 放射性衰变与非放射性损失
- **放射性衰变项**：$\dot{m}_{decay, k}(t) = \lambda_T I_k(t)$，其中 $\lambda_T = 1.78 \times 10^{-9}\text{ s}^{-1} \approx 6.4 \times 10^{-6}\text{ h}^{-1}$；
- **非放射性损失项**（微漏、非回收流股、废弃滞留）：
  $$\dot{m}_{loss, k}(t) = \epsilon_k \cdot \left( \frac{I_k(t) - I_{floor, k}}{\tau_k} \right) \quad \text{或} \quad \epsilon_k \cdot \frac{I_k(t)}{\tau_k}$$

---

## 🛠️ 3. Agent Model 智能介入修复准则 (Intelligent Remediation Directives)

当面对程序化脚本无法自动处置的未知复杂工况、非标准模型写法或数值发散缺陷时，Agent 必须以 **Abdou (1986)** 为理论指南，严格遵循以下五项智能修复准则：

### 准则一：集中参数动态守恒结构准则 (Lumped-Parameter Dynamic Conservation)
- **物理机理**：任何储存或滞留单元的导数项必须直接对应物理流股进出代数和。
- **Agent 修复准则**：
  若模型中的微分方程缺失衰变流股、时间常数未参数化或流股不平衡，Agent 应将其重构为标准的集中参数一阶形式：
  ```modelica
  der(inventory) = (inflow - (1 + Epsilon)*((inventory - Threshold)/T) - inventory*Decay) / time_scale;
  ```
  严禁随意丢弃衰变项 `inventory*Decay` 或损失项 `Epsilon`。

### 准则二：时基量纲统一与规范化准则 (Time Scale Normalization Rule)
- **物理机理**：Abdou 原始方程中特征时间 $\tau_k$ 多以小时（h）给出，而在现代数值仿真平台（如 OpenModelica）中，积分器时间步长通常以国际标准秒（s）为基准。
- **Agent 修复准则**：
  在所有动态方程中，**导数项必须规范化除以参数 `time_scale`**（秒制仿真时 `time_scale = 3600`，小时制时 `time_scale = 1`）：
  ```modelica
  der(x) = (rhs_in_gh) / time_scale;
  ```
  对于脉冲周期 $T_{period}$、宽度 $T_{width}$ 或事故触发时间 $t_{trigger}$，凡涉及自变量 `time` 的比较，必须统一乘上 `time_scale`：
  ```modelica
  time < startTime * time_scale
  ```
  **严禁**硬编码 `/ Constants.Seconds_per_hour` 或 `/ 3600`，必须保证模型随时可通过改变顶层 `time_scale` 切换时基。

### 准则三：铺底门槛与分段流股平滑准则 (Hold-up Threshold & Discontinuity Handling)
- **物理机理**：未达铺底门槛（$I \le Threshold$）时，储罐不发生主动外送，主输出为 0；达到门槛后流出速率线性释放。
- **Agent 修复准则**：
  若模型在 $I \le Threshold$ 时产生负流出或除以零发散，Agent 必须修复为分段保护逻辑：
  ```modelica
  if inventory > Threshold then
    der(inventory) = (inflow - (1 + Epsilon)*((inventory - Threshold)/T) - inventory*Decay) / time_scale;
    outflow = (inventory - Threshold)/T;
  else
    der(inventory) = (inflow - Epsilon*(inventory/T) - inventory*Decay) / time_scale;
    outflow = 0;
  end if;
  ```
  同时确保下游系统的输入直接挂接 `outflow`，避免在零时刻因下游过载抽空储罐引发负存量发散。

### 准则四：全厂质量守恒与流股连续性准则 (Plant-Wide Closed-Loop Conservation)
- **物理机理**：聚变堆是封闭燃料循环，氚具有极高毒性与稀缺性，全厂任何一个节点流出的质量流必须有明确去向。
- **Agent 修复准则**：
  Agent 在补齐或修复方程时，必须确保以下流股闭环：
  1. 上游组件的 `outflow` 必须完全分配给下游组件的 `inflow`（各分支分配比例之和 $\sum \eta_i = 1$）；
  2. 非放射性泄漏流股必须汇入次级/三级包容（`Secondary_Containment` / `Tertiary_Containment`）或去污系统（`VDS` / `WDS`）；
  3. 最终排向环境的流股必须接入累积监控节点（`Stack_Release` / `Liquid_Release`）。
  **严禁**在代数方程中产生“悬空流股”或凭空产生质量（Phantom Mass Creation）。

### 准则五：状态初值与物理非负有界性准则 (Initial State & Non-Negative Boundedness)
- **物理机理**：实际储罐与管道不能存在负存量，零时刻必须有物理确定的初值以消除代数奇异。
- **Agent 修复准则**：
  所有微分状态变量必须具备确定初值声明 `(start = ..., fixed = true)`。在役储罐主节点（如 `sds.inventory`）初始装料 $I_0$ 必须留有足够的起步安全裕度（通常 $I_0 > \Delta I_{deficit}$），确保系统安全越过起动亏空谷底。
