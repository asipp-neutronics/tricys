within FFCAS;
model DIV

  // 输入端口：来自Plasma的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_plasma[5] "来自Plasma的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 输入端口：来自Coolant_Loop的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_CL[5] "来自Coolant_Loop的输入" annotation(
    Placement(transformation(origin = {-120, 60}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 60}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));

  // 输入端口：来自CPS的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_CPS[5] "来自CPS的输入" annotation(
    Placement(transformation(origin = {-120, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));

  // 输出端口：输出到Coolant_Loop（5维）
  Modelica.Blocks.Interfaces.RealOutput to_CL[5] "输出到Coolant_Loop" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, -60}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 0.28 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0, 0, 0, 0, 0} "非放射性损失";

equation
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    der(I[i]) = from_plasma[i] + from_CL[i] + from_CPS[i] - (1 + nonradio_loss[i]) * I[i] / T  - decay_loss[i] * I[i];
    outflow[i] = I[i] / T;
    to_CL[i] = outflow[i];
  end for;

annotation(
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {255, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "DIV", fontName = "Arial")}
    ),
    uses(Modelica(version = "4.0.0"))
);

end DIV;
