within example_model;
model Blanket

  // 定义输入端口
  Modelica.Blocks.Interfaces.RealInput pulseInput "等离子体脉冲控制信号" annotation(
    Placement(transformation(origin = {-122, 90}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));

  // 输入端口：来自TES的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_TES[5] "来自TES的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, -60}, extent = {{-10, 10}, {10, -10}}, rotation = -180)));

  // 输出端口：输出到Coolant_Loop系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_CL[5] "输出到Coolant_Loop系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{10, 10}, {-10, -10}})));

  // 输出端口：输出到TES系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_TES[5] "输出到TES系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 60}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 新增端口：容量超限溢流（守恒，默认悬空）
  Modelica.Blocks.Interfaces.RealOutput overflow_out[5] "容量超限溢流（守恒）" annotation(
    Placement(transformation(origin = {110, -40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 20}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 新增端口：速率超限截流（守恒，默认悬空）
  Modelica.Blocks.Interfaces.RealOutput rate_clip_out[5] "速率超限截流（守恒）" annotation(
    Placement(transformation(origin = {110, -80}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, -20}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5](start = {0, 0, 0, 0, 0}) "总输出流（约束后）";

  // 新增中间变量
  Real I_total "总盘存量";
  Real outflow_nominal[5] "未约束的名义出流";
  Real outflow_total_nominal "名义总出流";
  Real inflow_total[5] "合计入流（含 TBR 增殖）";
  Real rate_scale "速率约束缩放因子 (0,1]";
  Real admit_scale "容量约束准入因子 [0,1]";

  // 参数定义
  parameter Real T = 24 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0, 0, 0, 0, 0} "非放射性损失";
  parameter Real TBR = 1.1 "Tritium Breeding Ratio (TBR), range: 1.05-1.15";
  parameter Real to_CL_Fraction = 0.01 "输出到CL的比例";
  parameter Real to_TES_Fraction = 1 - to_CL_Fraction "输出到TES的比例";

  // 新增约束参数（默认 1e9 = 等价无约束）
  parameter Real capacity_max(unit = "g") = 1e9
    "最大盘存量上限（总氚当量）。默认 1e9 = 无约束";
  parameter Real rate_max(unit = "g/h") = 1e9
    "最大出流速率上限（总氚当量）。默认 1e9 = 无约束";
  parameter Real softness = 0.02
    "sigmoid 平滑因子：0=硬约束（带事件），0.01-0.1=软约束";


    Real decay_rate[5] "衰变速率";
    Real leak_rate[5] "泄漏速率";
    Real cumulative_breed[5](start = {0, 0, 0, 0, 0}) "累计增殖";
equation
  // --- 聚合计算 ---
  I_total = sum(I);
  for i in 1:5 loop
    inflow_total[i] = (if i == 1 then pulseInput * TBR else 0) + from_TES[i];
    outflow_nominal[i] = I[i] / T;
    end for;
  outflow_total_nominal = sum(outflow_nominal);

  // --- 速率约束 ---
  if softness <= 0 then
    rate_scale = min(1.0, rate_max / max(outflow_total_nominal, 1e-30));
  else
    rate_scale = (rate_max / max(outflow_total_nominal, 1e-30))
               + (1.0 - rate_max / max(outflow_total_nominal, 1e-30))
                 / (1.0 + exp((outflow_total_nominal - rate_max) / (softness * rate_max + 1e-30)));
  end if;

  // --- 容量约束 ---
  if softness <= 0 then
    admit_scale = if I_total >= capacity_max then 0.0 else 1.0;
  else
    admit_scale = 1.0 / (1.0 + exp((I_total - capacity_max) / (softness * capacity_max + 1e-30)));
  end if;

  // --- 逐同位素质量平衡 ---
  for i in 1:5 loop
    outflow[i] = rate_scale * outflow_nominal[i];
    der(I[i]) = admit_scale * inflow_total[i]
              - (1 + nonradio_loss[i]) * outflow[i]
              - decay_loss[i] * I[i];
    to_TES[i] = to_TES_Fraction * outflow[i];
    to_CL[i] = to_CL_Fraction * outflow[i];
    overflow_out[i] = (1.0 - admit_scale) * inflow_total[i];
    rate_clip_out[i] = (1.0 - rate_scale) * outflow_nominal[i];
      decay_rate[i] = decay_loss[i]*I[i];
      leak_rate[i] = nonradio_loss[i]*I[i]/T;
      der(cumulative_breed[i]) = if i == 1 then pulseInput*TBR else 0;
    end for;

annotation(
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {255, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "Blanket", fontName = "Arial")}
    ),
    uses(Modelica(version = "4.0.0"))
);

end Blanket;
