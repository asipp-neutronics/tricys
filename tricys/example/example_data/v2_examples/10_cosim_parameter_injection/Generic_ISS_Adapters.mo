package Generic_ISS_Adapters
  "Multi-Component and 3LC Adapters for Generic_ISS Hydrogen Isotope Separation System"

  // ========================================================================
  // 1. 多组分 5D/6D 适配器外壳 (Multi-Component 5D/6D Adapter Shells)
  // 将 5 维原子质量流 [T, D, H, He, Imp] (g/h) 映射为 6 维分子流 [H2..T2]
  // 工程假设：He (组分4) 与 Imp (组分5) 假设在前级净化系统已脱除，ISS 内部仅追踪同位素，出口恒置0
  // ========================================================================

  model ISS_I_Adapter "ISS-I 多组分 5D/6D 双向适配器 (兼容 example_model.I_ISS 接口)"
    // 5 维外部接口
    Modelica.Blocks.Interfaces.RealInput from_TEP_FCU[5] "来自 TEP_FCU 进料 [T,D,H,He,Imp] (g/h)"
      annotation(Placement(transformation(extent={{-120,30},{-100,50}})));
    Modelica.Blocks.Interfaces.RealInput from_NBI[5] "来自 NBI 进料 [T,D,H,He,Imp] (g/h)"
      annotation(Placement(transformation(extent={{-120,-50},{-100,-30}})));

    Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到 SDS 系统 (高纯燃料产物) (g/h)"
      annotation(Placement(transformation(extent={{100,30},{120,50}})));
    Modelica.Blocks.Interfaces.RealOutput to_WDS[5] "输出到 WDS 系统 (CD4 塔顶含氢废气) (g/h)"
      annotation(Placement(transformation(extent={{100,-50},{120,-30}})));

    // 细分产物别名端口
    Modelica.Blocks.Interfaces.RealOutput to_SDS_T2[5] "高纯 T2 产物 -> SDS (g/h)";
    Modelica.Blocks.Interfaces.RealOutput to_SDS_D2[5] "高纯 D2 产物 -> SDS (g/h)";
    Modelica.Blocks.Interfaces.RealOutput to_WDS_waste[5] "含氢废气 -> WDS (g/h)";
    Modelica.Blocks.Interfaces.RealOutput total_inventory_5D[5] "全系统 5 组分总盘存 (g)";
    Modelica.Blocks.Interfaces.RealOutput inventory_T "ISS-I 动态总储氚滞留量 (g)"
      annotation(Placement(transformation(extent={{100,-10},{120,10}})));

    // 实例化 6 维物理核心
    Generic_ISS.ISS_I_Core core;

    final parameter Real MW_T = 3.016049;
    final parameter Real MW_D = 2.014102;
    final parameter Real MW_H = 1.007825;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    Real n_tep_T, n_tep_D, n_tep_H, n_tep_tot;
    Real a_tep_T, a_tep_D, a_tep_H;
    Real n_nbi_T, n_nbi_D, n_nbi_H, n_nbi_tot;
    Real a_nbi_T, a_nbi_D, a_nbi_H;

  equation
    // 1. TEP 进料适配 (5D -> 6D 统计平衡分配)
    n_tep_T = from_TEP_FCU[1] / MW_T;
    n_tep_D = from_TEP_FCU[2] / MW_D;
    n_tep_H = from_TEP_FCU[3] / MW_H;
    n_tep_tot = n_tep_T + n_tep_D + n_tep_H;
    a_tep_T = if n_tep_tot > 1e-12 then n_tep_T / n_tep_tot else 0.0;
    a_tep_D = if n_tep_tot > 1e-12 then n_tep_D / n_tep_tot else 0.0;
    a_tep_H = if n_tep_tot > 1e-12 then n_tep_H / n_tep_tot else 0.0;

    core.feed_TEP[1] = 0.5 * n_tep_tot * (a_tep_H * a_tep_H) * MW[1];
    core.feed_TEP[2] = 0.5 * n_tep_tot * (2.0 * a_tep_H * a_tep_D) * MW[2];
    core.feed_TEP[3] = 0.5 * n_tep_tot * (2.0 * a_tep_H * a_tep_T) * MW[3];
    core.feed_TEP[4] = 0.5 * n_tep_tot * (a_tep_D * a_tep_D) * MW[4];
    core.feed_TEP[5] = 0.5 * n_tep_tot * (2.0 * a_tep_D * a_tep_T) * MW[5];
    core.feed_TEP[6] = 0.5 * n_tep_tot * (a_tep_T * a_tep_T) * MW[6];

    // 2. NBI 进料适配 (5D -> 6D)
    n_nbi_T = from_NBI[1] / MW_T;
    n_nbi_D = from_NBI[2] / MW_D;
    n_nbi_H = from_NBI[3] / MW_H;
    n_nbi_tot = n_nbi_T + n_nbi_D + n_nbi_H;
    a_nbi_T = if n_nbi_tot > 1e-12 then n_nbi_T / n_nbi_tot else 0.0;
    a_nbi_D = if n_nbi_tot > 1e-12 then n_nbi_D / n_nbi_tot else 0.0;
    a_nbi_H = if n_nbi_tot > 1e-12 then n_nbi_H / n_nbi_tot else 0.0;

    core.feed_NBI[1] = 0.5 * n_nbi_tot * (a_nbi_H * a_nbi_H) * MW[1];
    core.feed_NBI[2] = 0.5 * n_nbi_tot * (2.0 * a_nbi_H * a_nbi_D) * MW[2];
    core.feed_NBI[3] = 0.5 * n_nbi_tot * (2.0 * a_nbi_H * a_nbi_T) * MW[3];
    core.feed_NBI[4] = 0.5 * n_nbi_tot * (a_nbi_D * a_nbi_D) * MW[4];
    core.feed_NBI[5] = 0.5 * n_nbi_tot * (2.0 * a_nbi_D * a_nbi_T) * MW[5];
    core.feed_NBI[6] = 0.5 * n_nbi_tot * (a_nbi_T * a_nbi_T) * MW[6];

    // 3. 出料适配 (6D -> 5D 精确原子质量加权折算)
    to_SDS_T2[1] = core.prod_T2_SDS[3]*(MW_T/MW[3]) + core.prod_T2_SDS[5]*(MW_T/MW[5]) + core.prod_T2_SDS[6];
    to_SDS_T2[2] = core.prod_T2_SDS[2]*(MW_D/MW[2]) + core.prod_T2_SDS[4] + core.prod_T2_SDS[5]*(MW_D/MW[5]);
    to_SDS_T2[3] = core.prod_T2_SDS[1] + core.prod_T2_SDS[2]*(MW_H/MW[2]) + core.prod_T2_SDS[3]*(MW_H/MW[3]);
    to_SDS_T2[4] = 0.0;
    to_SDS_T2[5] = 0.0;

    to_SDS_D2[1] = core.prod_D2_SDS[3]*(MW_T/MW[3]) + core.prod_D2_SDS[5]*(MW_T/MW[5]) + core.prod_D2_SDS[6];
    to_SDS_D2[2] = core.prod_D2_SDS[2]*(MW_D/MW[2]) + core.prod_D2_SDS[4] + core.prod_D2_SDS[5]*(MW_D/MW[5]);
    to_SDS_D2[3] = core.prod_D2_SDS[1] + core.prod_D2_SDS[2]*(MW_H/MW[2]) + core.prod_D2_SDS[3]*(MW_H/MW[3]);
    to_SDS_D2[4] = 0.0;
    to_SDS_D2[5] = 0.0;

    to_WDS_waste[1] = core.waste_HD_WDS[3]*(MW_T/MW[3]) + core.waste_HD_WDS[5]*(MW_T/MW[5]) + core.waste_HD_WDS[6];
    to_WDS_waste[2] = core.waste_HD_WDS[2]*(MW_D/MW[2]) + core.waste_HD_WDS[4] + core.waste_HD_WDS[5]*(MW_D/MW[5]);
    to_WDS_waste[3] = core.waste_HD_WDS[1] + core.waste_HD_WDS[2]*(MW_H/MW[2]) + core.waste_HD_WDS[3]*(MW_H/MW[3]);
    to_WDS_waste[4] = 0.0;
    to_WDS_waste[5] = 0.0;

    // 对接标准回路端口: to_SDS (T2 + D2 产品总和), to_WDS (含氢排气)
    to_SDS = to_SDS_T2 + to_SDS_D2;
    to_WDS = to_WDS_waste;

    // 4. 盘存映射 (6D -> 5D)
    total_inventory_5D[1] = core.total_inventory[3]*(MW_T/MW[3]) + core.total_inventory[5]*(MW_T/MW[5]) + core.total_inventory[6];
    total_inventory_5D[2] = core.total_inventory[2]*(MW_D/MW[2]) + core.total_inventory[4] + core.total_inventory[5]*(MW_D/MW[5]);
    total_inventory_5D[3] = core.total_inventory[1] + core.total_inventory[2]*(MW_H/MW[2]) + core.total_inventory[3]*(MW_H/MW[3]);
    total_inventory_5D[4] = 0.0;
    total_inventory_5D[5] = 0.0;

    // 5. 动态总储氚滞留量
    inventory_T = total_inventory_5D[1];
  end ISS_I_Adapter;


  model ISS_O_Adapter "ISS-O 多组分 5D/6D 双向适配器 (兼容 example_model.O_ISS 接口)"
    // 5 维外部接口 (严格对应 example_model.O_ISS 端口命名)
    Modelica.Blocks.Interfaces.RealInput from_CPS[5] "来自 CPS 的输入 (g/h)"
      annotation(Placement(transformation(extent={{-120,40},{-100,60}})));
    Modelica.Blocks.Interfaces.RealInput from_WDS[5] "来自 WDS 的输入 (g/h)"
      annotation(Placement(transformation(extent={{-120,0},{-100,20}})));
    Modelica.Blocks.Interfaces.RealInput from_TES[5] "来自 TES 的输入 (g/h)"
      annotation(Placement(transformation(extent={{-120,-40},{-100,-20}})));

    Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到 SDS 系统 (高纯 T2 产物) (g/h)"
      annotation(Placement(transformation(extent={{100,-30},{120,-10}})));
    Modelica.Blocks.Interfaces.RealOutput top_CD1[5] "CD1 塔顶洁净废气排放 (g/h)"
      annotation(Placement(transformation(extent={{100,30},{120,50}})));
    Modelica.Blocks.Interfaces.RealOutput waste_HD[5] "洁净废气排空 (别名端口) (g/h)";
    Modelica.Blocks.Interfaces.RealOutput total_inventory_5D[5] "全系统 5 组分总盘存 (g)";
    Modelica.Blocks.Interfaces.RealOutput inventory_T "ISS-O 动态总储氚滞留量 (g)"
      annotation(Placement(transformation(extent={{100,-10},{120,10}})));

    // 实例化 6 维物理核心
    Generic_ISS.ISS_O_Core core;

    final parameter Real MW_T = 3.016049;
    final parameter Real MW_D = 2.014102;
    final parameter Real MW_H = 1.007825;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    Real n_wds_T, n_wds_D, n_wds_H, n_wds_tot;
    Real a_wds_T, a_wds_D, a_wds_H;
    Real n_tes_T, n_tes_D, n_tes_H, n_tes_tot;
    Real a_tes_T, a_tes_D, a_tes_H;

  equation
    // 1. WDS 进料适配
    n_wds_T = from_WDS[1] / MW_T;
    n_wds_D = from_WDS[2] / MW_D;
    n_wds_H = from_WDS[3] / MW_H;
    n_wds_tot = n_wds_T + n_wds_D + n_wds_H;
    a_wds_T = if n_wds_tot > 1e-12 then n_wds_T / n_wds_tot else 0.0;
    a_wds_D = if n_wds_tot > 1e-12 then n_wds_D / n_wds_tot else 0.0;
    a_wds_H = if n_wds_tot > 1e-12 then n_wds_H / n_wds_tot else 0.0;

    core.feed_WDS[1] = 0.5 * n_wds_tot * (a_wds_H * a_wds_H) * MW[1];
    core.feed_WDS[2] = 0.5 * n_wds_tot * (2.0 * a_wds_H * a_wds_D) * MW[2];
    core.feed_WDS[3] = 0.5 * n_wds_tot * (2.0 * a_wds_H * a_wds_T) * MW[3];
    core.feed_WDS[4] = 0.5 * n_wds_tot * (a_wds_D * a_wds_D) * MW[4];
    core.feed_WDS[5] = 0.5 * n_wds_tot * (2.0 * a_wds_D * a_wds_T) * MW[5];
    core.feed_WDS[6] = 0.5 * n_wds_tot * (a_wds_T * a_wds_T) * MW[6];

    // 2. TES + CPS 进料适配 (CPS 汇入 TES 输入流)
    n_tes_T = (from_TES[1] + from_CPS[1]) / MW_T;
    n_tes_D = (from_TES[2] + from_CPS[2]) / MW_D;
    n_tes_H = (from_TES[3] + from_CPS[3]) / MW_H;
    n_tes_tot = n_tes_T + n_tes_D + n_tes_H;
    a_tes_T = if n_tes_tot > 1e-12 then n_tes_T / n_tes_tot else 0.0;
    a_tes_D = if n_tes_tot > 1e-12 then n_tes_D / n_tes_tot else 0.0;
    a_tes_H = if n_tes_tot > 1e-12 then n_tes_H / n_tes_tot else 0.0;

    core.feed_TES[1] = 0.5 * n_tes_tot * (a_tes_H * a_tes_H) * MW[1];
    core.feed_TES[2] = 0.5 * n_tes_tot * (2.0 * a_tes_H * a_tes_D) * MW[2];
    core.feed_TES[3] = 0.5 * n_tes_tot * (2.0 * a_tes_H * a_tes_T) * MW[3];
    core.feed_TES[4] = 0.5 * n_tes_tot * (a_tes_D * a_tes_D) * MW[4];
    core.feed_TES[5] = 0.5 * n_tes_tot * (2.0 * a_tes_D * a_tes_T) * MW[5];
    core.feed_TES[6] = 0.5 * n_tes_tot * (a_tes_T * a_tes_T) * MW[6];

    // 3. 出料适配
    to_SDS[1] = core.prod_T2_SDS[3]*(MW_T/MW[3]) + core.prod_T2_SDS[5]*(MW_T/MW[5]) + core.prod_T2_SDS[6];
    to_SDS[2] = core.prod_T2_SDS[2]*(MW_D/MW[2]) + core.prod_T2_SDS[4] + core.prod_T2_SDS[5]*(MW_D/MW[5]);
    to_SDS[3] = core.prod_T2_SDS[1] + core.prod_T2_SDS[2]*(MW_H/MW[2]) + core.prod_T2_SDS[3]*(MW_H/MW[3]);
    to_SDS[4] = 0.0;
    to_SDS[5] = 0.0;

    top_CD1[1] = core.waste_HD[3]*(MW_T/MW[3]) + core.waste_HD[5]*(MW_T/MW[5]) + core.waste_HD[6];
    top_CD1[2] = core.waste_HD[2]*(MW_D/MW[2]) + core.waste_HD[4] + core.waste_HD[5]*(MW_D/MW[5]);
    top_CD1[3] = core.waste_HD[1] + core.waste_HD[2]*(MW_H/MW[2]) + core.waste_HD[3]*(MW_H/MW[3]);
    top_CD1[4] = 0.0;
    top_CD1[5] = 0.0;

    waste_HD = top_CD1;

    // 4. 盘存映射
    total_inventory_5D[1] = core.total_inventory[3]*(MW_T/MW[3]) + core.total_inventory[5]*(MW_T/MW[5]) + core.total_inventory[6];
    total_inventory_5D[2] = core.total_inventory[2]*(MW_D/MW[2]) + core.total_inventory[4] + core.total_inventory[5]*(MW_D/MW[5]);
    total_inventory_5D[3] = core.total_inventory[1] + core.total_inventory[2]*(MW_H/MW[2]) + core.total_inventory[3]*(MW_H/MW[3]);
    total_inventory_5D[4] = 0.0;
    total_inventory_5D[5] = 0.0;

    // 5. 动态总储氚滞留量
    inventory_T = total_inventory_5D[1];
  end ISS_O_Adapter;


  // ========================================================================
  // 3-Level Confinement (3LC) 标量氚流适配器 (针对 CFEDR_3LC 系统)
  // 将 1 维标量纯氚质量流量 (g/h) 映射为 6 维分子流 [H2..T2] 并实现三级包容安全审计
  // ========================================================================

  model ISS_I_3LC_Adapter "ISS-I 高保真 0-D 物理核心适配器 (针对 CFEDR_3LC 标量氚流与三级包容系统)"
    // 外部标量氚质量流率端口 (g/h)
    Modelica.Blocks.Interfaces.RealInput from_TEP "来自 TEP 排气净化进料 (g/h 纯氚当量)"
      annotation(Placement(transformation(extent={{-120,-10},{-100,10}})));

    Modelica.Blocks.Interfaces.RealOutput to_SDS "输出至 SDS 储氚系统的高纯燃料 (g/h 纯氚)"
      annotation(Placement(transformation(extent={{100,30},{120,50}})));
    Modelica.Blocks.Interfaces.RealOutput to_WDS "输出至 WDS 的含氢废气微量截留氚 (g/h)"
      annotation(Placement(transformation(extent={{100,0},{120,20}})));
    Modelica.Blocks.Interfaces.RealOutput to_VDS "向 VDS (通风除氚系统) 的微量渗透泄漏 (g/h)"
      annotation(Placement(transformation(extent={{100,-30},{120,-10}})));
    Modelica.Blocks.Interfaces.RealOutput to_Secondary "向二级包容边界 (手套箱/密封室) 的微量泄漏 (g/h)"
      annotation(Placement(transformation(extent={{100,-60},{120,-40}})));

    // 内部宏观状态与核安全审计变量 (与 CFEDR 接口完全对齐)
    Real inventory(unit="g", start = 0) "ISS-I 动态总储氚滞留量 (g)";
    Real inflow(unit="g/h", start = 0) "总进料氚流率 (g/h)";
    Real outflow(unit="g/h", start = 0) "总出料氚流率 (g/h)";
    Real inv_HT(unit="g", start = 0) "HT 分子形态储氚滞留量 (g)";
    Real inv_HTO(unit="g", start = 0) "HTO 分子形态储氚滞留量 (g)";
    Real decay_rate(unit="g/h") "氚放射性衰变损失率 (g/h)";

    // 适配参数 (与 CFEDR 默认兼容)
    parameter Real Epsilon(unit="1") = 1e-4 "微量泄漏率系数";
    parameter Real EtoV(unit="1") = 0.99 "泄漏物分配至 VDS 的比例 (其余进入二级包容)";
    parameter Real Decay(unit="1/h") = 6.42e-6 "氚衰变常数 (1/h)";
    parameter Real T(unit="h") = 6.0 "水力学特征时间常数 (h)";
    parameter Real Threshold(unit="g") = 300 "门限阈值 (兼容 CFEDR)";

    // 实例化 6 组分机理核心
    Generic_ISS.ISS_I_Core core;

    final parameter Real MW_T = 3.016049;
    final parameter Real MW_D = 2.014102;
    final parameter Real MW_H = 1.007825;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    Real n_T, n_tot;

  equation
    // 1. TEP 进料适配 (标量氚流量 -> 6 维分子进料，按等摩尔 D-T 等离子体排气重构)
    n_T = max(0.0, from_TEP) / MW_T;
    n_tot = 2.0 * n_T; // D:T = 1:1

    core.feed_TEP[1] = 0.0;
    core.feed_TEP[2] = 0.0;
    core.feed_TEP[3] = 0.0;
    core.feed_TEP[4] = 0.25 * n_T * MW[4]; // D2 (0.25 n_T moles)
    core.feed_TEP[5] = 0.50 * n_T * MW[5]; // DT (0.50 n_T moles)
    core.feed_TEP[6] = 0.25 * n_T * MW[6]; // T2 (0.25 n_T moles)

    core.feed_NBI = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // 2. 出料适配 (6D -> 1D 纯氚质量流量折算)
    to_SDS = (core.prod_T2_SDS[3]*(MW_T/MW[3]) + core.prod_T2_SDS[5]*(MW_T/MW[5]) + core.prod_T2_SDS[6])
           + (core.prod_D2_SDS[3]*(MW_T/MW[3]) + core.prod_D2_SDS[5]*(MW_T/MW[5]) + core.prod_D2_SDS[6]);
    to_WDS = core.waste_HD_WDS[3]*(MW_T/MW[3]) + core.waste_HD_WDS[5]*(MW_T/MW[5]) + core.waste_HD_WDS[6];

    inflow = from_TEP;
    outflow = to_SDS + to_WDS;

    // 3. 3-Level Confinement 动态泄漏与安全审计
    to_VDS = Epsilon * EtoV * outflow;
    to_Secondary = Epsilon * (1.0 - EtoV) * outflow;

    // 4. 内部动态盘存与物性
    inventory = core.inventory_T;
    inv_HT = core.total_inventory[3]*(MW_T/MW[3]);
    inv_HTO = 0.0;
    decay_rate = inventory * Decay;
  end ISS_I_3LC_Adapter;


  model ISS_O_3LC_Adapter "ISS-O 高保真 0-D 物理核心适配器 (针对 CFEDR_3LC 标量氚流与三级包容系统)"
    // 外部标量氚质量流率端口 (g/h)
    Modelica.Blocks.Interfaces.RealInput from_CPS_PFC "来自 CPS_PFC 进料 (g/h 纯氚当量)"
      annotation(Placement(transformation(extent={{-120,40},{-100,60}})));
    Modelica.Blocks.Interfaces.RealInput from_CPS_BZ "来自 CPS_BZ 进料 (g/h 纯氚当量)"
      annotation(Placement(transformation(extent={{-120,10},{-100,30}})));
    Modelica.Blocks.Interfaces.RealInput from_TES "来自 TES 包层提取进料 (g/h 纯氚当量)"
      annotation(Placement(transformation(extent={{-120,-20},{-100,0}})));
    Modelica.Blocks.Interfaces.RealInput from_WDS "来自 WDS 水除氚浓缩进料 (g/h 纯氚当量)"
      annotation(Placement(transformation(extent={{-120,-50},{-100,-30}})));

    Modelica.Blocks.Interfaces.RealOutput to_SDS "输出至 SDS 储氚系统的高纯燃料 (g/h 纯氚)"
      annotation(Placement(transformation(extent={{100,30},{120,50}})));
    Modelica.Blocks.Interfaces.RealOutput to_VDS "向 VDS (通风除氚系统) 的微量渗透泄漏 (g/h)"
      annotation(Placement(transformation(extent={{100,-10},{120,10}})));
    Modelica.Blocks.Interfaces.RealOutput to_Secondary "向二级包容边界 (手套箱/密封室) 的微量泄漏 (g/h)"
      annotation(Placement(transformation(extent={{100,-50},{120,-30}})));

    // 内部宏观状态与核安全审计变量 (与 CFEDR 接口完全对齐)
    Real inventory(unit="g", start = 0) "ISS-O 动态总储氚滞留量 (g)";
    Real inflow(unit="g/h", start = 0) "总进料氚流率 (g/h)";
    Real outflow(unit="g/h", start = 0) "总出料氚流率 (g/h)";
    Real inv_HT(unit="g", start = 0) "HT 分子形态储氚滞留量 (g)";
    Real inv_HTO(unit="g", start = 0) "HTO 分子形态储氚滞留量 (g)";
    Real decay_rate(unit="g/h") "氚放射性衰变损失率 (g/h)";

    // 适配参数 (与 CFEDR 默认兼容)
    parameter Real Epsilon(unit="1") = 1e-4 "微量泄漏率系数";
    parameter Real EtoV(unit="1") = 0.95 "泄漏物分配至 VDS 的比例 (其余进入二级包容)";
    parameter Real Decay(unit="1/h") = 6.42e-6 "氚衰变常数 (1/h)";
    parameter Real T(unit="h") = 6.0 "水力学特征时间常数 (h)";
    parameter Real Threshold(unit="g") = 20 "门限阈值 (兼容 CFEDR)";

    // 实例化 6 组分机理核心
    Generic_ISS.ISS_O_Core core;

    final parameter Real MW_T = 3.016049;
    final parameter Real MW_D = 2.014102;
    final parameter Real MW_H = 1.007825;
    final parameter Real MW[6] = {2.01588, 3.02204, 4.02399, 4.02820, 5.03015, 6.03210};

    Real m_T_wds, n_wds_T, n_wds_tot;
    Real m_T_tes, n_tes_T, n_tes_tot;

  equation
    // 1. WDS 进料适配 (进 CD1): 氢同位素载体中微量氚提取 (a_H ~ 0.99, a_T ~ 0.01 工程假设)
    m_T_wds = max(0.0, from_WDS);
    n_wds_T = m_T_wds / MW_T;
    n_wds_tot = 100.0 * n_wds_T;

    core.feed_WDS[1] = 0.5 * n_wds_tot * (0.99 * 0.99) * MW[1];
    core.feed_WDS[2] = 0.0;
    core.feed_WDS[3] = n_wds_tot * (0.99 * 0.01) * MW[3];
    core.feed_WDS[4] = 0.0;
    core.feed_WDS[5] = 0.0;
    core.feed_WDS[6] = 0.5 * n_wds_tot * (0.01 * 0.01) * MW[6];

    // 2. TES + CPS 进料适配 (进 CD2): TES 载气提纯
    m_T_tes = max(0.0, from_TES + from_CPS_PFC + from_CPS_BZ);
    n_tes_T = m_T_tes / MW_T;
    n_tes_tot = 100.0 * n_tes_T;

    core.feed_TES[1] = 0.5 * n_tes_tot * (0.99 * 0.99) * MW[1];
    core.feed_TES[2] = 0.0;
    core.feed_TES[3] = n_tes_tot * (0.99 * 0.01) * MW[3];
    core.feed_TES[4] = 0.0;
    core.feed_TES[5] = 0.0;
    core.feed_TES[6] = 0.5 * n_tes_tot * (0.01 * 0.01) * MW[6];

    // 3. 出料适配 (6D -> 1D 纯氚质量流量折算)
    to_SDS = core.prod_T2_SDS[3]*(MW_T/MW[3]) + core.prod_T2_SDS[5]*(MW_T/MW[5]) + core.prod_T2_SDS[6];

    inflow = from_CPS_PFC + from_CPS_BZ + from_TES + from_WDS;
    outflow = to_SDS;

    // 4. 3-Level Confinement 动态泄漏与安全审计
    to_VDS = Epsilon * EtoV * outflow;
    to_Secondary = Epsilon * (1.0 - EtoV) * outflow;

    // 5. 内部动态盘存与物性
    inventory = core.inventory_T;
    inv_HT = core.total_inventory[3]*(MW_T/MW[3]);
    inv_HTO = 0.0;
    decay_rate = inventory * Decay;
  end ISS_O_3LC_Adapter;

end Generic_ISS_Adapters;
