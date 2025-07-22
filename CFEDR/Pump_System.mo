within CFEDR;
model Pump_System

// 输入端口：来自Plasma的输入（5维）
  Modelica.Blocks.Interfaces.RealInput from_Plasma[5] "来自Plasma的输入" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{-10, 10}, {10, -10}}, rotation = -0)));

// 输出端口：输出到TEP_FEP（5维）
  Modelica.Blocks.Interfaces.RealOutput to_TEP[5] "输出到TEP系统" annotation(
    Placement(transformation(origin = {120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));

// 状态变量：系统中5种物质的储存量
  Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
  Real outflow[5] "总输出流";

// 参数定义
  parameter Real T = 0.17 "平均滞留时间 (mean residence time)";
  parameter Real decay_loss[5] (each unit="1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
  parameter Real nonradio_loss[5] (each unit="1") = {0.0001, 0.0001, 0.0001, 0.0001, 0.0001} "非放射性损失";
  parameter Real threshold = 640 "铺底量";

// 辅助变量：计算I的总和
  Real I_total "I的5个分量之和";
equation
// 计算I的总和
  I_total = sum(I);

// 计算每种物质的动态变化和输出
for i in 1:5 loop
// 根据I的总和是否超过阈值，分为两种情况
    if I_total > threshold then
      der(I[i]) = from_Plasma[i] - (1 + nonradio_loss[i]) * (I[i] - threshold * I[i] / I_total) / T - decay_loss[i] * I[i];
      outflow[i] = (I[i] - threshold * I[i] / I_total) / T;
    else
      der(I[i]) = from_Plasma[i] - nonradio_loss[i] * I[i] / T - decay_loss[i] * I[i];
      outflow[i] = 0;
    end if;
// 输出流分配到TEP_FEP
    to_TEP[i] = outflow[i];
  end for;

annotation(
    Diagram,
    Icon(graphics = {
      Rectangle(fillColor = {255, 255, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(extent = {{-80, 80}, {80, -80}}, textString = "Pump\nSystem", fontName = "Arial")}
    ),
    uses(Modelica(version = "4.0.0")));
end Pump_System;
