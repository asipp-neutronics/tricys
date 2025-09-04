within example_model;
model TEP_FEP

  // 输入端口：来自Pump_System的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_pump[5] "来自Pump_System的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{-10, 10}, {10, -10}}, rotation = -0)));

  // 输出端口：输出到SDS系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到SDS系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, 114}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));

  // 输出端口：输出到TEP_IP系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_TEP_IP[5] "输出到TEP_IP系统" annotation(
    Placement(transformation(origin = {110, -20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 0.1 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  //DIR比例，逻辑不对，因为DT不一定一样多，需要改
  parameter Real to_SDS_Fraction[5] = {0.5, 0.5, 0, 0, 0} "输出到SDS的比例";

equation
  // 计算每个维度流体的动态变化
  for i in 1:5 loop
    der(I[i]) = from_pump[i] - (1 + nonradio_loss[i]) * I[i] / T  - decay_loss[i] * I[i];
    outflow[i] = I[i]/T;
    // 输出流分配到SDS和TEP_IP
    to_SDS[i] = outflow[i] * to_SDS_Fraction[i];
    to_TEP_IP[i] = outflow[i] * (1 - to_SDS_Fraction[i]);
  end for;

annotation(
    Diagram,
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {170, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {0, 2}, extent = {{-78, 40}, {78, -40}}, textString = "TEP_FEP", fontName = "Arial", textStyle = {TextStyle.Italic})}
    ),
    uses(Modelica(version = "4.0.0"))
);

end TEP_FEP;
