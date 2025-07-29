within example;
model Coolant_Pipe

  // 输入端口：来自FW的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_FW[5] "来自FW的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 80}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 输入端口：来自DIV的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_DIV[5] "来自DIV的输入" annotation(
    Placement(transformation(origin = {-120, 60}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, -80}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));

  // 输入端口：来自BZ的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_BZ[5] "来自BZ的输入" annotation(
    Placement(transformation(origin = {-120, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, 10}, {-10, -10}})));

  // 输出端口：输出到CPS系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_CPS[5] "输出到CPS系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));

  // 输出端口：输出到FW系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_FW[5] "输出到FW系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 40}, extent = {{10, -10}, {-10, 10}})));

  // 输出端口：输出到DIV系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_DIV[5] "输出到DIV系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, -40}, extent = {{10, -10}, {-10, 10}})));

  // 输出端口：输出到WDS系统（5维）
  Modelica.Blocks.Interfaces.RealOutput to_WDS[5] "输出到WDS系统" annotation(
    Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, 114}, extent = {{10, -10}, {-10, 10}}, rotation = 270)));

  // 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5](start = {0, 0, 0, 0, 0}) "总输出流";

  // 参数定义
  parameter Real T = 24 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  parameter Real to_WDS_Fraction = 1e-4;
  parameter Real to_CPS_Fraction = 1e-2;
  parameter Real to_FW_Fraction = 0.6;

equation
  // 计算每种物质的动态变化和输出
  for i in 1:5 loop
    der(I[i]) = from_DIV[i] + from_FW[i] + from_BZ[i] - (1 + nonradio_loss[i]) * I[i] / T  - decay_loss[i] * I[i];
    outflow[i] = I[i] / T;
    to_WDS[i] = to_WDS_Fraction * outflow[i];
    to_CPS[i] = to_CPS_Fraction * (1 - to_WDS_Fraction) * outflow[i];
    to_FW[i] = to_FW_Fraction * (1 - to_CPS_Fraction) * (1 - to_WDS_Fraction) * outflow[i];
    to_DIV[i] = (1 - to_FW_Fraction) * (1 - to_CPS_Fraction) * (1 - to_WDS_Fraction) * outflow[i];

  end for;
  annotation(
    Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "Coolant
Loop", fontName = "Arial")}),
    uses(Modelica(version = "4.0.0")));
end Coolant_Pipe;
