package Generic_ISS
  "Generic Hydrogen Isotope Separation System (ISS) High-Fidelity Physics Core Library"

  // ========================================================================
  // 1. 基础物理降阶单元 (Base Units - 6-Species Molecular Basis)
  // 组分索引映射: 1:H2, 2:HD, 3:HT, 4:D2, 5:DT, 6:T2
  // ========================================================================

  model Column_0D_6 "6-组分高保真精馏塔 0-D 动态代理单元"
    // 端口定义 (6 维分子质量流量 g/h)
    Modelica.Blocks.Interfaces.RealInput feed[6] "进料质量流率 (g/h)"
      annotation(Placement(transformation(extent={{-120,-10},{-100,10}})));
    Modelica.Blocks.Interfaces.RealOutput top[6] "塔顶产品质量流率 (g/h)"
      annotation(Placement(transformation(extent={{100,30},{120,50}})));
    Modelica.Blocks.Interfaces.RealOutput bottom[6] "塔釜产品质量流率 (g/h)"
      annotation(Placement(transformation(extent={{100,-50},{120,-30}})));

    // 状态输出端口
    Modelica.Blocks.Interfaces.RealOutput inventory_T "塔内总储氚滞留量 (g)"
      annotation(Placement(transformation(extent={{100,-10},{120,10}})));

    // --- 动态注入参数 (由 h2iso 求解结果自动覆盖) ---
    parameter Real m_top_ref[6] = {0,0,0,0,0,0} "h2iso 塔顶 6 组分参考质量流量 (g/h)" annotation(Evaluate=false);
    parameter Real m_bottom_ref[6] = {0,0,0,0,0,0} "h2iso 塔底 6 组分参考质量流量 (g/h)" annotation(Evaluate=false);
    parameter Real tau = 1.0 "塔特征停留时间 (h) = M_total / m_feed" annotation(Evaluate=false);
    parameter Real N_stages = 0 "理论板数" annotation(Evaluate=false);
    parameter Real R = 0 "回流比" annotation(Evaluate=false);
    parameter Real T_top = 0 "塔顶温度 (K)" annotation(Evaluate=false);
    parameter Real T_bottom = 0 "塔釜温度 (K)" annotation(Evaluate=false);

    final parameter Real MW_T = 3.016049;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    // --- 内部状态与代数变量 ---
    Real I[6](start={0,0,0,0,0,0}) "塔内 6 组分动态持液滞留量 (g)";
    Real outflow[6] "总流出流率 (g/h)";
    Real SF_top[6] "各组分塔顶分离系数 (无量纲)";

  equation
    for i in 1:6 loop
      // 1. 动态质量守恒：流入 - 流出 = 滞留量导数
      der(I[i]) = feed[i] - outflow[i];

      // 2. 一阶水力学滞后
      outflow[i] = I[i] / max(tau, 1e-4);

      // 3. 基于 h2iso 稳态高保真分离比 (带除零极小值保护)
      SF_top[i] = if (m_top_ref[i] + m_bottom_ref[i] > 1e-12)
                  then m_top_ref[i] / (m_top_ref[i] + m_bottom_ref[i])
                  else 0.0;

      // 4. 产物动态分流
      top[i] = SF_top[i] * outflow[i];
      bottom[i] = (1.0 - SF_top[i]) * outflow[i];
    end for;

    // 5. 塔内原子氚总滞留量 (g)
    inventory_T = I[3]*(MW_T/MW[3]) + I[5]*(MW_T/MW[5]) + I[6];
  end Column_0D_6;


  model Equilibrator_0D_6 "6-组分催化同位素交换平衡反应器"
    // 分子摩尔质量 (g/mol): [H2, HD, HT, D2, DT, T2]
    final parameter Real MW_T = 3.016049;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};
    parameter Real tau = 0.1 "平衡器停留时间常数 (h)" annotation(Evaluate=false);

    Modelica.Blocks.Interfaces.RealInput feed[6] "进料 (g/h)"
      annotation(Placement(transformation(extent={{-120,-10},{-100,10}})));
    Modelica.Blocks.Interfaces.RealOutput outflow[6] "平衡产物 (g/h)"
      annotation(Placement(transformation(extent={{100,-10},{120,10}})));
    Modelica.Blocks.Interfaces.RealOutput inventory_T "平衡器内总储氚滞留量 (g)";

    Real I[6](start={0,0,0,0,0,0}) "平衡器内滞留量 (g)";
    Real flow_hold[6] "滞留流出流率 (g/h)";
    Real n_in[6] "各分子摩尔流率 (mol/h)";
    Real n_H, n_D, n_T, n_atom_tot "原子摩尔流率与总原子数 (mol/h)";
    Real a_H, a_D, a_T "进料原子分数";
    Real x_eq[6] "化学平衡分子摩尔分数";
    Real n_out_tot "输出总分子摩尔流率 (mol/h)";

  equation
    // 1. 动态持液滞留
    for i in 1:6 loop
      der(I[i]) = feed[i] - flow_hold[i];
      flow_hold[i] = I[i] / max(tau, 1e-4);
      n_in[i] = flow_hold[i] / MW[i];
    end for;

    // 2. 严格原子守恒与原子分数提取
    n_H = 2.0 * n_in[1] + n_in[2] + n_in[3];
    n_D = n_in[2] + 2.0 * n_in[4] + n_in[5];
    n_T = n_in[3] + n_in[5] + 2.0 * n_in[6];
    n_atom_tot = n_H + n_D + n_T;

    a_H = if n_atom_tot > 1e-12 then n_H / n_atom_tot else 0.0;
    a_D = if n_atom_tot > 1e-12 then n_D / n_atom_tot else 0.0;
    a_T = if n_atom_tot > 1e-12 then n_T / n_atom_tot else 0.0;

    // 3. 统计热力学同位素重组平衡分配 (高温统计极限 K_eq=4 近似，满足整体同位素原子守恒)
    x_eq[1] = a_H * a_H;         // H2
    x_eq[2] = 2.0 * a_H * a_D;   // HD
    x_eq[3] = 2.0 * a_H * a_T;   // HT
    x_eq[4] = a_D * a_D;         // D2
    x_eq[5] = 2.0 * a_D * a_T;   // DT
    x_eq[6] = a_T * a_T;         // T2

    // 4. 重构 6 维分子输出质量流量
    n_out_tot = 0.5 * n_atom_tot;
    for i in 1:6 loop
      outflow[i] = n_out_tot * x_eq[i] * MW[i];
    end for;

    // 5. 储氚滞留量
    inventory_T = I[3]*(MW_T/MW[3]) + I[5]*(MW_T/MW[5]) + I[6];
  end Equilibrator_0D_6;


  // ========================================================================
  // 2. 核心拓扑 (Core Topologies - 严格匹配 Wang 2022)
  // ========================================================================

  model ISS_I_Core "ISS-I 内燃料循环物理核心 (4塔 + 2平衡器)"
    parameter Integer N = 6 "分子组分维度";
    final parameter Real MW_T = 3.016049;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    // 纯物理边界端口 (6 维分子质量流 g/h)
    Modelica.Blocks.Interfaces.RealInput feed_TEP[N] "TEP 进料 (进 CD1)";
    Modelica.Blocks.Interfaces.RealInput feed_NBI[N] "NBI 进料 (进 CD2)";

    Modelica.Blocks.Interfaces.RealOutput prod_T2_SDS[N] "高纯 T2 核燃料产物 -> SDS (CD3 塔底)";
    Modelica.Blocks.Interfaces.RealOutput prod_D2_SDS[N] "高纯 D2 产物 -> SDS (CD2 顶 + CD4 底)";
    Modelica.Blocks.Interfaces.RealOutput waste_HD_WDS[N] "含氢废气 -> WDS 排气 (CD4 顶)";
    Modelica.Blocks.Interfaces.RealOutput total_inventory[N] "全系统 6 维分子总动态持液量 (g)";
    Modelica.Blocks.Interfaces.RealOutput inventory_T "ISS-I 动态总储氚滞留量 (g)"
      annotation(Placement(transformation(extent={{100,0},{120,20}})));

    // 实例化物理单元
    Column_0D_6 CD1, CD2, CD3, CD4;
    Equilibrator_0D_6 E1, E2;

  equation
    // 1. CD1: 接收 TEP 进料与 E1 (CD3 顶流催化平衡后) 的回流
    CD1.feed = feed_TEP + E1.outflow;

    // 2. CD2: 接收 NBI 进料
    CD2.feed = feed_NBI;

    // 3. CD3: 接收 CD1 塔釜与 CD2 塔釜重组分
    CD3.feed = CD1.bottom + CD2.bottom;

    // 4. E1 & E2 催化平衡器
    E1.feed = CD3.top; // DT 转化为 D2 + T2 回流至 CD1
    E2.feed = CD1.top; // 杂质流催化重组送 CD4

    // 5. CD4: 接收 E2 出口物料进行脱氢
    CD4.feed = E2.outflow;

    // 6. 系统输出边界汇聚
    prod_T2_SDS = CD3.bottom;
    prod_D2_SDS = CD2.top + CD4.bottom;
    waste_HD_WDS = CD4.top;

    // 7. 提取全系统各分子滞留量总和
    for i in 1:N loop
      total_inventory[i] = CD1.I[i] + CD2.I[i] + CD3.I[i] + CD4.I[i] + E1.I[i] + E2.I[i];
    end for;

    // 8. 全系统原子氚总滞留量 (g)
    inventory_T = total_inventory[3]*(MW_T/MW[3]) + total_inventory[5]*(MW_T/MW[5]) + total_inventory[6];
  end ISS_I_Core;


  model ISS_O_Core "ISS-O 外燃料循环物理核心 (3塔 + 1平衡器)"
    parameter Integer N = 6 "分子组分维度";
    final parameter Real MW_T = 3.016049;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    // 纯物理边界端口 (6 维分子质量流 g/h)
    Modelica.Blocks.Interfaces.RealInput feed_WDS[N] "WDS 进料 (进 CD1)";
    Modelica.Blocks.Interfaces.RealInput feed_TES[N] "TES 进料 (进 CD2)";

    Modelica.Blocks.Interfaces.RealOutput prod_T2_SDS[N] "高纯 T2 浓缩产物 -> SDS (CD3 塔底)";
    Modelica.Blocks.Interfaces.RealOutput waste_HD[N] "洁净废气排空 (CD1 塔顶)";
    Modelica.Blocks.Interfaces.RealOutput total_inventory[N] "全系统 6 维分子总动态持液量 (g)";
    Modelica.Blocks.Interfaces.RealOutput inventory_T "ISS-O 动态总储氚滞留量 (g)"
      annotation(Placement(transformation(extent={{100,0},{120,20}})));

    // 实例化物理单元
    Column_0D_6 CD1, CD2, CD3;
    Equilibrator_0D_6 E;

  equation
    // 1. CD1: 接收 WDS 进料与 CD2 塔顶富氢回流
    CD1.feed = feed_WDS + CD2.top;

    // 2. CD2: 接收 TES 进料、CD1 塔釜富同位素流及 CD3 塔顶回流
    CD2.feed = feed_TES + CD1.bottom + CD3.top;

    // 3. E 催化平衡器: 接收 CD2 塔釜纯 HT 进行重组 (2HT -> H2 + T2)
    E.feed = CD2.bottom;

    // 4. CD3: 接收平衡转化后产物进行最终 T2 提纯
    CD3.feed = E.outflow;

    // 5. 系统输出边界汇聚
    waste_HD = CD1.top;
    prod_T2_SDS = CD3.bottom;

    // 6. 提取全系统各分子滞留量总和
    for i in 1:N loop
      total_inventory[i] = CD1.I[i] + CD2.I[i] + CD3.I[i] + E.I[i];
    end for;

    // 7. 全系统原子氚总滞留量 (g)
    inventory_T = total_inventory[3]*(MW_T/MW[3]) + total_inventory[5]*(MW_T/MW[5]) + total_inventory[6];
  end ISS_O_Core;

end Generic_ISS;
