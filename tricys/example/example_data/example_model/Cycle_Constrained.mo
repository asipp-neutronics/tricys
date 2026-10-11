within example_model;
model Cycle_Constrained
  "系统顶层模型：使用 ConstrainedBuffer 替换 Blanket，验证容量/速率约束"

  inner parameter Real time_scale = 3600 "Time-base factor: 1 = hour-based axis, 3600 = second-based axis";
  // 实例化脉冲信号模块
  Modelica.Blocks.Sources.Pulse pulseSource(amplitude = 9.60984, period = 500 * time_scale, width = 100) annotation(
    Placement(transformation(origin = {-120, -20}, extent = {{-60, 20}, {-40, 40}})));

  // 氚增殖源（pulseSource * TBR → from_Upstream[1]）
  parameter Real TBR = 1.1 "Tritium Breeding Ratio";
  Modelica.Blocks.Sources.RealExpression breeding_source(y = pulseSource.y * TBR)
    "氚增殖率 = 脉冲 × TBR";
  Modelica.Blocks.Sources.Constant zero_signal[5](each k = 0) "零信号源";

  // 实例化子系统
  Plasma plasma annotation(
    Placement(transformation(origin = {-140, 10}, extent = {{0, -10}, {20, 10}})));
  Fueling_System FS annotation(
    Placement(transformation(origin = {-70, 10}, extent = {{-10, -10}, {10, 10}})));
  SDS sds annotation(
    Placement(transformation(origin = {-10, 10}, extent = {{-10, -10}, {10, 10}})));
  Pump_System pump_System annotation(
    Placement(transformation(origin = {-130, -30}, extent = {{-10, -10}, {10, 10}})));
  TEP_FEP tep_fep annotation(
    Placement(transformation(origin = {-90, -50}, extent = {{-10, -10}, {10, 10}})));
  TEP_IP tep_ip annotation(
    Placement(transformation(origin = {-30, -50}, extent = {{-10, -10}, {10, 10}})));
  TEP_FCU tep_fcu annotation(
    Placement(transformation(origin = {30, -50}, extent = {{-10, -10}, {10, 10}})));
  I_ISS i_iss annotation(
    Placement(transformation(origin = {70, -10}, extent = {{-10, -10}, {10, 10}})));
  WDS wds annotation(
    Placement(transformation(origin = {110, 50}, extent = {{-10, -10}, {10, 10}})));
  O_ISS o_iss annotation(
    Placement(transformation(origin = {50, 50}, extent = {{-10, -10}, {10, 10}})));
  FW fw annotation(
    Placement(transformation(origin = {-90, 170}, extent = {{-10, -10}, {10, 10}})));
  DIV div annotation(
    Placement(transformation(origin = {-90, 90}, extent = {{-10, -10}, {10, 10}})));
  CPS cps annotation(
    Placement(transformation(origin = {-30, 50}, extent = {{-10, -10}, {10, 10}})));
  TES tes annotation(
    Placement(transformation(origin = {50, 130}, extent = {{-10, -10}, {10, 10}})));
  Coolant_Pipe coolant_pipe annotation(
    Placement(transformation(origin = {-30, 130}, extent = {{-10, -10}, {10, 10}})));

  // 约束型包层（替换原 Blanket）
  ConstrainedBuffer blanket_c(
    T = 24,
    capacity_max = 1e9,
    rate_max = 1e9,
    softness = 0.02,
    to_Down_Fraction = 0.01,
    decay_loss = {6.4e-6, 0, 0, 0, 0},
    nonradio_loss = {0, 0, 0, 0, 0}
  ) annotation(
    Placement(transformation(origin = {10, 130}, extent = {{-10, -10}, {10, 10}})));

equation
  // === 连接脉冲信号 ===
  connect(pulseSource.y, plasma.pulseInput) annotation(
    Line(points = {{-159, 10}, {-142, 10}}, color = {255, 0, 0}, pattern = LinePattern.Dash, thickness = 1));

  // === 燃料系统连接 ===
  connect(FS.to_Plasma, plasma.from_Fueling_System) annotation(
    Line(points = {{-81.4, 10}, {-118.4, 10}}, color = {0, 0, 127}, thickness = 0.5));
  connect(sds.to_FS, FS.from_SDS) annotation(
    Line(points = {{-21.4, 10}, {-59.4, 10}}, color = {0, 0, 127}, thickness = 0.5));

  // === 约束包层连接 ===
  // 氚增殖作为 from_Upstream[1]，其余为 0
  blanket_c.from_Upstream[1] = breeding_source.y;
  blanket_c.from_Upstream[2] = 0;
  blanket_c.from_Upstream[3] = 0;
  blanket_c.from_Upstream[4] = 0;
  blanket_c.from_Upstream[5] = 0;

  // TES 回流作为 from_Recycle
  connect(tes.to_BZ, blanket_c.from_Recycle) annotation(
    Line(points = {{38.6, 124}, {20.6, 124}}, color = {0, 0, 127}, thickness = 0.5));

  // blanket_c.to_Downstream → coolant_pipe（相当于原 to_CL）
  connect(blanket_c.to_Downstream, coolant_pipe.from_BZ) annotation(
    Line(points = {{-1.4, 130}, {-18.4, 130}}, color = {0, 0, 127}, thickness = 0.5));

  // blanket_c 主出流的 (1-to_Down_Fraction) 部分已经进入 overflow_out
  // 需要将 TES 的入流从 overflow_out 取出（模拟原 to_TES 逻辑）
  // 这里调整设计：to_Down_Fraction = 0.01 对应原 to_CL_Fraction
  // overflow_out 中包含 (1-0.01)*outflow + 容量溢出 → 接 TES
  connect(blanket_c.overflow_out, tes.from_BZ) annotation(
    Line(points = {{21.4, 136}, {39.4, 136}}, color = {0, 0, 127}, thickness = 0.5));

  // rate_clip_out 接地（在无约束时为 0；有约束时可接 SDS/安全系统）
  // 此处不连接，Modelica 允许 output 不接

  // === 冷却管路连接 ===
  connect(coolant_pipe.from_FW, fw.to_CL) annotation(
    Line(points = {{-41.4, 138}, {-56.8, 138}, {-56.8, 176}, {-79, 176}}, color = {0, 0, 127}, thickness = 0.5));
  connect(coolant_pipe.to_FW, fw.from_CL) annotation(
    Line(points = {{-41.4, 134}, {-64.8, 134}, {-64.8, 164}, {-79, 164}}, color = {0, 0, 127}, thickness = 0.5));
  connect(cps.to_FW, fw.from_CPS) annotation(
    Line(points = {{-41.4, 56}, {-60.4, 56}, {-60.4, 130}, {-90, 150}, {-90, 159}}, color = {0, 0, 127}, thickness = 0.5));
  connect(cps.to_ISS_O, o_iss.from_CPS) annotation(
    Line(points = {{-18.6, 50}, {38.4, 50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(coolant_pipe.to_CPS, cps.from_CL) annotation(
    Line(points = {{-30, 118.6}, {-30, 62.6}}, color = {0, 0, 127}, thickness = 0.5));

  // === TES 和 ISS 连接 ===
  connect(tes.to_O_ISS, o_iss.from_TES) annotation(
    Line(points = {{50, 118.6}, {50, 62.6}}, color = {0, 0, 127}, thickness = 0.5));
  connect(coolant_pipe.to_WDS, wds.from_CL) annotation(
    Line(points = {{-30, 141.4}, {-30, 159.4}, {110, 159.4}, {110, 61}}, color = {0, 0, 127}, thickness = 0.5));
  connect(wds.to_O_ISS, o_iss.from_WDS) annotation(
    Line(points = {{99, 50}, {62.6, 50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(o_iss.to_SDS, sds.from_O_ISS) annotation(
    Line(points = {{50, 38}, {50, 16}, {2, 16}}, color = {0, 0, 127}, thickness = 0.5));

  // === 等离子体排出 → 泵 → TEP ===
  connect(plasma.to_Pump, pump_System.from_Plasma) annotation(
    Line(points = {{-130, -2}, {-130, -18}}, color = {0, 0, 127}, thickness = 0.5));
  connect(pump_System.to_TEP_FEP, tep_fep.from_pump) annotation(
    Line(points = {{-130, -42}, {-130, -50}, {-101, -50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tep_fep.to_TEP_IP, tep_ip.from_TEP_FEP) annotation(
    Line(points = {{-78.6, -50}, {-41, -50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tep_ip.to_TEP_FCU, tep_fcu.from_TEP_IP) annotation(
    Line(points = {{-19, -50}, {17.4, -50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tep_fcu.to_I_ISS, i_iss.from_TEP_FCU) annotation(
    Line(points = {{41, -50}, {70, -50}, {70, -22}}, color = {0, 0, 127}, thickness = 0.5));
  connect(i_iss.to_SDS, sds.from_I_ISS) annotation(
    Line(points = {{58, -10}, {30, -10}, {30, 4}, {2, 4}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tep_fep.to_SDS, sds.from_TEP_FEP) annotation(
    Line(points = {{-90, -38}, {-90, -20}, {-10, -20}, {-10, -2}}, color = {0, 0, 127}, thickness = 0.5));

  // === DIV 连接 ===
  connect(coolant_pipe.to_DIV, div.from_CL) annotation(
    Line(points = {{-42, 126}, {-66, 126}, {-66, 96}, {-78, 96}}, color = {0, 0, 127}, thickness = 0.5));
  connect(div.to_CL, coolant_pipe.from_DIV) annotation(
    Line(points = {{-78, 84}, {-54, 84}, {-54, 122}, {-42, 122}}, color = {0, 0, 127}, thickness = 0.5));
  connect(plasma.to_Div, div.from_plasma) annotation(
    Line(points = {{-124, 22}, {-124, 90}, {-102, 90}}, color = {0, 0, 127}, thickness = 0.5));
  connect(fw.from_plasma, plasma.to_FW) annotation(
    Line(points = {{-102, 170}, {-136, 170}, {-136, 22}}, color = {0, 0, 127}, thickness = 0.5));
  connect(i_iss.to_WDS, wds.from_I_ISS) annotation(
    Line(points = {{82, -10}, {110, -10}, {110, 38}}, color = {0, 0, 127}, thickness = 0.5));
  connect(cps.to_DIV, div.from_CPS) annotation(
    Line(points = {{-42, 44}, {-90, 44}, {-90, 78}}, color = {0, 0, 127}, thickness = 0.5));

annotation(
    Diagram(coordinateSystem(extent = {{-200, 200}, {140, -80}})),
    Icon(graphics = {
      Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-100, 100}, {100, -100}}),
      Text(extent = {{-80, 80}, {80, -80}}, textString = "Cycle_C", fontName = "Arial")}),
    uses(Modelica(version = "4.0.0")));
end Cycle_Constrained;
