within CFEDR;
model CPS

  // 输入端口：来自Coolant_Loop的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_CP[5] "来自Coolant_Loop的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, 114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));

  // 输出端口：输出到ISS_O系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_O_ISS[5] "输出到ISS_O系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 输出端口：输出到FW系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_FW[5] "输出到FW系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 60}, extent = {{10, -10}, {-10, 10}})));

  // 输出端口：输出到DIV系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_DIV[5] "输出到DIV系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, -60}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 48 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0.0001, 0.0001, 0.0001} "非放射性损失";
  parameter Real threshold = 200 "铺底量";
  parameter Real to_O_ISS_Fraction = 0.99 "0.95Abdou 输出到ISS_O的比例";
  parameter Real to_FW_Fraction = 0.06 "输出到FW的比例";

  // 辅助变量：计算I的总和
  Real I_total "I的5个分量之和";
equation
  // 计算I的总和
  I_total = sum(I);
  
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    // 根据储存量是否超过铺底量，分为两种情况
    if I_total > threshold then
      der(I[i]) = from_CP[i] - (1 + nonradio_loss[i]) * (I[i] - threshold * I[i] / I_total) / T  - decay_loss[i] * I[i];
      outflow[i] = (I[i] - threshold * I[i] / I_total) / T;
    else
      der(I[i]) = from_CP[i] - nonradio_loss[i] * I[i]/T  - decay_loss[i] * I[i];
      outflow[i] = 0;
    end if;

    // 输出流分配到ISS_O、FW、DIV
    to_O_ISS[i] = to_O_ISS_Fraction * outflow[i];
    to_FW[i] = to_FW_Fraction * (1 - to_O_ISS_Fraction) * outflow[i];
    to_DIV[i] = (1 - to_FW_Fraction) * (1 - to_O_ISS_Fraction) * outflow[i];
  end for;

annotation(
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {255, 85, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {-3, 3}, extent = {{-75, 35}, {75, -35}}, textString = "CPS", fontName = "Arial")}
    ),
    uses(Modelica(version = "4.0.0"))
);

end CPS;
