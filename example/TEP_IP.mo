within example;
model TEP_IP

  // 输入端口：来自TEP_FEP的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_TEP_FEP[5] "来自TEP_FEP的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{-10, 10}, {10, -10}}, rotation = -0)));

  // 输出端口：输出到TEP_FCU系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_TEP_FCU[5] "输出到TEP_FCU系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}})));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 0.5 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";

equation
  // 计算每个维度流体的动态变化
  for i in 1:5 loop
    der(I[i]) = from_TEP_FEP[i] - (1 + nonradio_loss[i]) * I[i] / T  - decay_loss[i] * I[i];
    // 输出流分配到TEP_FCU
    outflow[i] = I[i]/T;
    to_TEP_FCU[i] = I[i] / T;
  end for;

annotation(
    Diagram,
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {0, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {0, 2}, extent = {{-78, 40}, {78, -40}}, textString = "TEP_IP", fontName = "Arial", textStyle = {TextStyle.Italic})}
    ),
    uses(Modelica(version = "4.0.0"))
);

end TEP_IP;
