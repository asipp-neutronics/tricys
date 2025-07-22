within CFEDR;
model SDS

  // 输入端口：来自I_ISS的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_I_ISS[5] "来自I_ISS 的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, -60}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));

  // 输入端口：来自O_ISS的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_O_ISS[5] "来自O_ISS 的输出" annotation(
    Placement(transformation(origin = {-120, -40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 60}, extent = {{-10, 10}, {10, -10}}, rotation = 180)));

  // 输入端口：来自TEP_FEP的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_TEP[5] "来自TEP 的输入" annotation(
    Placement(transformation(origin = {-120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));

  // 输出端口：输出到Fueling_System（5维）
  Modelica.Blocks.Interfaces.RealInput to_FS[5] "输出到Fueling_System" annotation(
    Placement(transformation(origin = {120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));

  // 状态变量：SDS 内部的氚存储量
  Real I[5](start = {3000, 3000, 3000, 0, 0}) "SDS 内的氚存储量";

  // 参数定义
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";

equation
  for i in 1:5 loop
    // 只对 T, D, H 进行计算
    der(I[i]) = from_I_ISS[i] + from_O_ISS[i] + from_TEP[i] - to_FS[i]  - decay_loss[i] * I[i];
  end for;

annotation(
    Diagram,
    Icon(graphics = {Rectangle( fillColor = {85, 255, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "SDS", fontName = "Arial")}),
    uses(Modelica(version = "4.0.0")));
end SDS;
