within CFEDR;
model Fueling_System

  // 输入端口：输出到plasma（5维）
  Modelica.Blocks.Interfaces.RealInput to_Plasma[5] "输出到等离子体的燃料" annotation(
    Placement(transformation(origin = {116, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));

  // 输出端口：来自SDS（5维）
  Modelica.Blocks.Interfaces.RealOutput from_SDS[5] "来自SDS的输入（燃料输入）" annotation(
    Placement(transformation(origin = {-114, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 0.5 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  //parameter Real nonradio_loss[5] (each unit="1") = {0, 0, 0, 0, 0} "非放射性损失";

equation
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    der(I[i]) = from_SDS[i] - 1 * I[i] / T  - decay_loss[i] * I[i];
    outflow[i] = I[i]/T;
    to_Plasma[i] = outflow[i];
  end for;

  annotation(
    Diagram,
    Icon(graphics = {Rectangle(fillColor = {255, 255, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Fueling
System", fontName = "Arial")}),
    uses(Modelica(version = "4.0.0")));
end Fueling_System;
