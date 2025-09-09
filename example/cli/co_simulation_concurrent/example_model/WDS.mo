within example_model;
model WDS

  // 输入端口：来自I_ISS的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_I_ISS[5] "来自I_ISS的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));

  // 输入端口：来自CL的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_CL[5] "来自CL的输入" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, 114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));

  // 输出端口：输出到O_ISS系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_O_ISS[5] "输出到O_ISS系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 24 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  parameter Real threshold = 1000 "铺底量";

  // 辅助变量：计算I的总和
  Real I_total "I的5个分量之和";
equation
  // 计算I的总和
  I_total = sum(I);
  
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    // 根据储存量是否超过阈值，分为两种情况
    if I_total > threshold then
      der(I[i]) = from_I_ISS[i] + from_CL[i] - (1 + nonradio_loss[i]) * (I[i] - threshold) / T  - decay_loss[i] * I[i];
      outflow[i] = (I[i] - threshold)/T;
    else
      der(I[i]) = from_I_ISS[i] + from_CL[i] -  nonradio_loss[i] * I[i]/T  - decay_loss[i] * I[i];
      outflow[i] = 0;
    end if;
    // 输出流分配到O_ISS
    to_O_ISS[i] = outflow[i];
  end for;

annotation(
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {85, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "WDS", fontName = "Arial")}
    ),
    Diagram(coordinateSystem(extent = {{-100, -100}, {100, 100}})),
    uses(Modelica(version = "4.0.0")),
    version = ""
);

end WDS;
