within example_model;
model Plasma

  // 定义输入端口
  Modelica.Blocks.Interfaces.RealInput pulseInput "等离子体脉冲控制信号，单位: g/h" annotation(
    Placement(transformation(origin = {-122, 90}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{-10, -10}, {10, 10}})));

  // 输出端口：来自Fueling_System的输入（5维）
  Modelica.Blocks.Interfaces.RealOutput from_Fueling_System[5] "来自Fueling_System的输入" annotation(
    Placement(transformation(origin = {-110, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));

  // 输出端口：输出到FW（5维）
  Modelica.Blocks.Interfaces.RealOutput to_FW[5] "输出到FW" annotation(
    Placement(transformation(origin = {114, -40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-60, 116}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));

  // 输出端口：输出到Div（5维）
  Modelica.Blocks.Interfaces.RealOutput to_Div[5] "输出到Div" annotation(
    Placement(transformation(origin = {114, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {60, 114}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));

  // 输出端口：输出到Pump_System（5维）
  Modelica.Blocks.Interfaces.RealOutput to_Pump[5] "输出到Pump_System" annotation(
    Placement(transformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));

  // 参数部分
  Real He_generated "聚变反应生成的氦，单位: g/h";
  parameter Real fb = 0.06 "燃烧分数 (氚和氘的燃烧比例，仅在脉冲开启时有效，无单位)";
  parameter Real nf = 0.5 "燃料效率 (氚和氘进入等离子体的比例，无单位)";
  Real H_injection "氕的背景注入速率，单位: g/h";
  //parameter Real Imp_injection = 0.001 "杂质的背景注入速率，单位: g/h";
  parameter Real to_Div_fraction[5] = {1e-4, 1e-4, 1e-4, 1e-1, 1e-2} "流向偏滤器的比例，无单位";
  parameter Real to_FW_fraction[5] = {1e-4, 1e-4, 1e-4, 1e-1, 1e-2} "流向第一壁的比例，无单位";
  parameter Real He_yield = 4.002602 / 3.01693 "氦产额 (每单位质量氚+氘生成的氦质量，无单位)";

equation
  // 计算氦生成（仅在脉冲开启时生成）
  He_generated = if pulseInput > 0 then He_yield * pulseInput else 0;
  H_injection = (pulseInput * (1.00784 / 3.01693))/(fb * nf) * 2 / 99;

  // 使用 for 循环定义每种物质的计算逻辑
  for i in 1:5 loop

    // 燃料注入逻辑（氕和杂质的注入与脉冲信号绑定）
    from_Fueling_System[i] = if i == 1 then pulseInput /(fb * nf) // 氚的注入
                            else if i == 2 then (pulseInput * (2.01409 / 3.01693))/(fb * nf) // 氘的注入
                            else if i == 3 then H_injection // 氕的背景注入
                            else 0; // 氦无外部注入

    // 流出逻辑（包括未燃烧物质和滞留物质的输运）
    to_FW[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0)) * to_FW_fraction[i];
    to_Div[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0)) * to_Div_fraction[i];
    to_Pump[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0)) * (1 - to_Div_fraction[i] - to_FW_fraction[i] - fb * nf);
  end for;

annotation(
    Diagram,
    Icon(graphics = {Rectangle(fillColor = {255, 0, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Plasma", fontName = "Arial")}),
    uses(Modelica(version = "4.0.0")));
end Plasma;
