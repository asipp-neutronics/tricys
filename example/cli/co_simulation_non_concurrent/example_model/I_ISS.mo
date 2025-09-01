within example_model;
model I_ISS

  // 输入端口：来自TEP_FCU的输入
  Modelica.Blocks.Interfaces.RealInput from_TEP_FCU[5] "来自TEP_FCU的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));

  // 输出端口：输出到SDS系统
  Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到SDS系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));

  // 输出端口：输出到WDS系统
  Modelica.Blocks.Interfaces.RealOutput to_WDS[5] "输出到WDS系统" annotation(
    Placement(transformation(origin = {110, -20}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

  // 参数定义
  parameter Real T = 4 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  parameter Real threshold = 300 "铺底量";
  parameter Real to_WDS_Fraction = 1e-8 "输出到WDS的比例";
  parameter Real to_SDS_Fraction = 1 - to_WDS_Fraction "输出到SDS的比例";
  
  // 辅助变量：计算I的总和
  Real I_total "I的5个分量之和";
equation
  // 计算I的总和
  I_total = sum(I);
  
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    // 根据储存量是否超过阈值，分为两种情况
    if I_total > threshold then
      der(I[i]) = from_TEP_FCU[i] - (1 + nonradio_loss[i]) * (I[i] - threshold) / T  - decay_loss[i] * I[i];
      outflow[i] = (I[i] - threshold)/T;
    else
      der(I[i]) = from_TEP_FCU[i] - nonradio_loss[i] * I[i]/T  - decay_loss[i] * I[i];
      outflow[i] = 0;
    end if;
    // 输出流分配到SDS和WDS
    to_WDS[i] = to_WDS_Fraction * outflow[i];
    to_SDS[i] = to_SDS_Fraction * outflow[i];
  end for;

annotation(
    Icon(graphics = {
      Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}),
      Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}),
      Rectangle(fillColor = {255, 85, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {4, 3}, extent = {{-72, 33}, {72, -33}}, textString = "I-ISS", fontName = "Arial")}
    ),
    uses(Modelica(version = "4.0.0"))
);

end I_ISS;
