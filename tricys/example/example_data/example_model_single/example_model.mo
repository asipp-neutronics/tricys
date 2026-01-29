package example_model
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
        der(I[i]) = from_I_ISS[i] + from_CL[i] - (1 + nonradio_loss[i]) * (1 - threshold / I_total) * I[i] / T  - decay_loss[i] * I[i];
        outflow[i] = (1 - threshold / I_total) * I[i] / T;
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
  
  model TES
    // 输入端口：来自BZ的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_BZ[5] "来自BZ的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 60}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
    // 输出端口：输出到O_ISS系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_O_ISS[5] "输出到O_ISS系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));
    // 输出端口：输出到BZ系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_BZ[5] "输出到BZ系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, -60}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 12 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
    parameter Real threshold = 200 "铺底量";
    parameter Real to_O_ISS_Fraction = 0.95;
    parameter Real to_BZ_Fraction = 1 - to_O_ISS_Fraction;
    // 辅助变量：计算I的总和
    Real I_total "I的5个分量之和";
  equation
// 计算I的总和
    I_total = sum(I);
  // 计算每种物质的动态变化和输出
for i in 1:5 loop
// 根据储存量是否超过阈值，分为两种情况
      if I_total > threshold then
        der(I[i]) = from_BZ[i] - (1 + nonradio_loss[i])*(1 - threshold/I_total)*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = (1 - threshold/I_total)*I[i]/T;
      else
        der(I[i]) = from_BZ[i] - nonradio_loss[i]*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = 0;
      end if;
// 输出流分配到ISS_O
      to_O_ISS[i] = to_O_ISS_Fraction*outflow[i];
      to_BZ[i] = to_BZ_Fraction*outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 85, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "TES", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end TES;
  
  model TEP_IP
    // 输入端口：来自TEP_FEP的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_TEP_FEP[5] "来自TEP_FEP的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{-10, 10}, {10, -10}}, rotation = -0)));
    // 输出端口：输出到TEP_FCU系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_TEP_FCU[5] "输出到TEP_FCU系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}})));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.5 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  equation
// 计算每个维度流体的动态变化
    for i in 1:5 loop
      der(I[i]) = from_TEP_FEP[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
// 输出流分配到TEP_FCU
      outflow[i] = I[i]/T;
      to_TEP_FCU[i] = I[i]/T;
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {0, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {0, 2}, extent = {{-78, 40}, {78, -40}}, textString = "TEP_IP", fontName = "Arial", textStyle = {TextStyle.Italic})}),
      uses(Modelica(version = "4.0.0")));
  end TEP_IP;
  
  model TEP_FEP
    // 输入端口：来自Pump_System的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_pump[5] "来自Pump_System的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{-10, 10}, {10, -10}}, rotation = -0)));
    // 输出端口：输出到SDS系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到SDS系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, 114}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));
    // 输出端口：输出到TEP_IP系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_TEP_IP[5] "输出到TEP_IP系统" annotation(
      Placement(transformation(origin = {110, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -0)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.1 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
    //DIR比例，逻辑不对，因为DT不一定一样多，需要改
    parameter Real to_SDS_Fraction[5] = {0.5, 0.5, 0, 0, 0} "输出到SDS的比例";
  equation
// 计算每个维度流体的动态变化
    for i in 1:5 loop
      der(I[i]) = from_pump[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
// 输出流分配到SDS和TEP_IP
      to_SDS[i] = outflow[i]*to_SDS_Fraction[i];
      to_TEP_IP[i] = outflow[i]*(1 - to_SDS_Fraction[i]);
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {170, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {0, 2}, extent = {{-78, 40}, {78, -40}}, textString = "TEP_FEP", fontName = "Arial", textStyle = {TextStyle.Italic})}),
      uses(Modelica(version = "4.0.0")));
  end TEP_FEP;
  
  model TEP_FCU
    // 输入端口：来自TEP_IP的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_TEP_IP[5] "来自TEP_IP的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 输出端口：输出到I_ISS系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_I_ISS[5] "输出到I_ISS系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.1 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
  equation
// 计算每个维度流体的动态变化
    for i in 1:5 loop
      der(I[i]) = from_TEP_IP[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
// 输出流分配到I_ISS
      outflow[i] = I[i]/T;
      to_I_ISS[i] = I[i]/T;
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {0, 255, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {0, 2}, extent = {{-78, 40}, {78, -40}}, textString = "TEP_FCU", fontName = "Arial", textStyle = {TextStyle.Italic})}),
      uses(Modelica(version = "4.0.0")));
  end TEP_FCU;
  
  model SDS
    // 输入端口：来自I_ISS的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_I_ISS[5] "来自I_ISS 的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, -60}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));
    // 输入端口：来自O_ISS的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_O_ISS[5] "来自O_ISS 的输出" annotation(
      Placement(transformation(origin = {-120, -40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 60}, extent = {{-10, 10}, {10, -10}}, rotation = 180)));
    // 输入端口：来自TEP_FEP的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_TEP_FEP[5] "来自TEP/FEP 的输入" annotation(
      Placement(transformation(origin = {-120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
    // 输出端口：输出到Fueling_System（5维）
    Modelica.Blocks.Interfaces.RealInput to_FS[5] "输出到Fueling_System" annotation(
      Placement(transformation(origin = {120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = -0)));
    // 状态变量：SDS 内部的氚存储量
    Real I[5](start = {3000, 3000, 0, 0, 0}) "SDS 内的氚存储量";
    // 参数定义
    // parameter Real T = 0.5 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0, 0, 0, 0, 0} "非放射性损失";
  equation
    for i in 1:5 loop
// 只对 T, D, H 进行计算
      der(I[i]) = from_I_ISS[i] + from_O_ISS[i] + from_TEP_FEP[i] - (1 + nonradio_loss[i])*to_FS[i] - decay_loss[i]*I[i];
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Rectangle(fillColor = {85, 255, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "SDS", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end SDS;
  
  model Pump_System
    // 输入端口：来自Plasma的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_Plasma[5] "来自Plasma的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, 114}, extent = {{10, 10}, {-10, -10}}, rotation = 90)));
    // 输出端口：输出到TEP_FEP（5维）
    Modelica.Blocks.Interfaces.RealOutput to_TEP_FEP[5] "输出到TEP_FEP系统" annotation(
      Placement(transformation(origin = {120, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.17 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
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
        der(I[i]) = from_Plasma[i] - (1 + nonradio_loss[i])*(1 - threshold/I_total)*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = (1 - threshold/I_total)*I[i]/T;
      else
        der(I[i]) = from_Plasma[i] - nonradio_loss[i]*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = 0;
      end if;
// 输出流分配到TEP_FEP
      to_TEP_FEP[i] = outflow[i];
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Rectangle(fillColor = {255, 255, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Pump\nSystem", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end Pump_System;
  
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
    parameter Real He_yield = 4.002602/3.01693 "氦产额 (每单位质量氚+氘生成的氦质量，无单位)";
  equation
// 计算氦生成（仅在脉冲开启时生成）
    He_generated = if pulseInput > 0 then He_yield*pulseInput else 0;
    H_injection = (pulseInput*(1.00784/3.01693))/(fb*nf)*2/99;
  // 使用 for 循环定义每种物质的计算逻辑
for i in 1:5 loop
// 燃料注入逻辑（氕和杂质的注入与脉冲信号绑定）
      from_Fueling_System[i] = if i == 1 then pulseInput/(fb*nf)// 氚的注入
       else if i == 2 then (pulseInput*(2.01409/3.01693))/(fb*nf)// 氘的注入
       else if i == 3 then H_injection// 氕的背景注入
       else 0;
// 氦无外部注入
// 流出逻辑（包括未燃烧物质和滞留物质的输运）
      to_FW[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0))*to_FW_fraction[i];
      to_Div[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0))*to_Div_fraction[i];
      to_Pump[i] = (from_Fueling_System[i] + (if i == 4 then He_generated else 0))*(1 - to_Div_fraction[i] - to_FW_fraction[i] - fb*nf);
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Rectangle(fillColor = {255, 0, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Plasma", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end Plasma;
  
  model O_ISS
    // 输入端口：来自CPS的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_CPS[5] "来自CPS的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 输入端口：来自WDS的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_WDS[5] "来自WDS的输入" annotation(
      Placement(transformation(origin = {-120, 60}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));
    // 输入端口：来自TES的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_TES[5] "来自TES的输入" annotation(
      Placement(transformation(origin = {-120, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, 114}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
    // 输出端口：输出到SDS系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到SDS系统" annotation(
      Placement(transformation(origin = {110, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 12 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
    parameter Real threshold = 20 "铺底量";
    // 辅助变量：计算I的总和
    Real I_total "I的5个分量之和";
  equation
// 计算I的总和
    I_total = sum(I);
  // 计算每种物质的动态变化和输出
for i in 1:5 loop
// 根据储存量是否超过阈值，分为两种情况
      if I_total > threshold then
        der(I[i]) = from_CPS[i] + from_TES[i] + from_WDS[i] - (1 + nonradio_loss[i])*(1 - threshold/I_total)*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = (1 - threshold/I_total)*I[i]/T;
      else
        der(I[i]) = from_CPS[i] + from_TES[i] + from_WDS[i] - nonradio_loss[i]*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = 0;
      end if;
// 输出流分配到SDS
      to_SDS[i] = outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 85, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {5, 7}, extent = {{-75, 33}, {75, -33}}, textString = "O-ISS", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end O_ISS;
  
  model I_ISS
    // 输入端口：来自TEP_FCU的输入
    Modelica.Blocks.Interfaces.RealInput from_TEP_FCU[5] "来自TEP_FCU的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));
    // 输出端口：输出到SDS系统
    Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "输出到SDS系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
    // 输出端口：输出到WDS系统
    Modelica.Blocks.Interfaces.RealOutput to_WDS[5] "输出到WDS系统" annotation(
      Placement(transformation(origin = {110, -20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 4 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
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
        der(I[i]) = from_TEP_FCU[i] - (1 + nonradio_loss[i])*(1 - threshold/I_total)*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = (1 - threshold/I_total)*I[i]/T;
      else
        der(I[i]) = from_TEP_FCU[i] - nonradio_loss[i]*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = 0;
      end if;
// 输出流分配到SDS和WDS
      to_WDS[i] = to_WDS_Fraction*outflow[i];
      to_SDS[i] = to_SDS_Fraction*outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 85, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {4, 3}, extent = {{-72, 33}, {72, -33}}, textString = "I-ISS", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end I_ISS;
  
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
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    //parameter Real nonradio_loss[5] (each unit="1") = {0, 0, 0, 0, 0} "非放射性损失";
  equation
// 计算每种物质的动态变化和输出
    for i in 1:5 loop
      der(I[i]) = from_SDS[i] - 1*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
      to_Plasma[i] = outflow[i];
    end for;
    annotation(
      Diagram,
      Icon(graphics = {Rectangle(fillColor = {255, 255, 0}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Fueling
  System", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end Fueling_System;
  
  model FW
    // 输入端口：来自plasma（5维）
    Modelica.Blocks.Interfaces.RealInput from_plasma[5] annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 输入端口：来自Coolant_Loop（5维）
    Modelica.Blocks.Interfaces.RealInput from_CL[5] annotation(
      Placement(transformation(origin = {-120, 60}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, -60}, extent = {{-10, 10}, {10, -10}}, rotation = 180)));
    // 输入端口：来自CPS（5维）
    Modelica.Blocks.Interfaces.RealInput from_CPS[5] annotation(
      Placement(transformation(origin = {-120, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));
    // 输出端口：输出到Coolant_Loop（5维）
    Modelica.Blocks.Interfaces.RealOutput to_CL[5] annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 60}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.28 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0, 0, 0, 0, 0} "非放射性损失";
  equation
// 计算每种物质的动态变化和输出
    for i in 1:5 loop
      der(I[i]) = from_plasma[i] + from_CL[i] + from_CPS[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
      to_CL[i] = outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "FW", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end FW;
  
  model DIV
    // 输入端口：来自Plasma的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_plasma[5] "来自Plasma的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 输入端口：来自Coolant_Loop的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_CL[5] "来自Coolant_Loop的输入" annotation(
      Placement(transformation(origin = {-120, 60}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 60}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));
    // 输入端口：来自CPS的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_CPS[5] "来自CPS的输入" annotation(
      Placement(transformation(origin = {-120, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));
    // 输出端口：输出到Coolant_Loop（5维）
    Modelica.Blocks.Interfaces.RealOutput to_CL[5] "输出到Coolant_Loop" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, -60}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 0.28 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0, 0, 0, 0, 0} "非放射性损失";
  equation
// 计算每种物质的动态变化和输出
    for i in 1:5 loop
      der(I[i]) = from_plasma[i] + from_CL[i] + from_CPS[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
      to_CL[i] = outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "DIV", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end DIV;
  
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
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
    parameter Real to_WDS_Fraction = 1e-4;
    parameter Real to_CPS_Fraction = 1e-2;
    parameter Real to_FW_Fraction = 0.6;
  equation
// 计算每种物质的动态变化和输出
    for i in 1:5 loop
      der(I[i]) = from_DIV[i] + from_FW[i] + from_BZ[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
      to_WDS[i] = to_WDS_Fraction*outflow[i];
      to_CPS[i] = to_CPS_Fraction*(1 - to_WDS_Fraction)*outflow[i];
      to_FW[i] = to_FW_Fraction*(1 - to_CPS_Fraction)*(1 - to_WDS_Fraction)*outflow[i];
      to_DIV[i] = (1 - to_FW_Fraction)*(1 - to_CPS_Fraction)*(1 - to_WDS_Fraction)*outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 255, 127}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "Coolant
  Loop", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end Coolant_Pipe;
  
  model CPS
    // 输入端口：来自Coolant_Loop的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_CL[5] "来自Coolant_Loop的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, 114}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));
    // 输出端口：输出到ISS_O系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_ISS_O[5] "输出到ISS_O系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 0}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 输出端口：输出到FW系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_FW[5] "输出到FW系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 60}, extent = {{10, -10}, {-10, 10}})));
    // 输出端口：输出到DIV系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_DIV[5] "输出到DIV系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, -60}, extent = {{-10, -10}, {10, 10}}, rotation = -180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5] "总输出流";
    // 参数定义
    parameter Real T = 48 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0.0001, 0.0001, 0, 0, 0} "非放射性损失";
    parameter Real threshold = 200 "铺底量";
    parameter Real to_ISS_O_Fraction = 0.99 "0.95Abdou 输出到ISS_O的比例";
    parameter Real to_FW_Fraction = 0.06 "输出到FW的比例";
    // 辅助变量：计算I的总和
    Real I_total "I的5个分量之和";
  equation
// 计算I的总和
    I_total = sum(I);
  // 计算每种物质的动态变化和输出
for i in 1:5 loop
// 根据储存量是否超过铺底量，分为两种情况
      if I_total > threshold then
        der(I[i]) = from_CL[i] - (1 + nonradio_loss[i])*(1 - threshold/I_total)*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = (1 - threshold/I_total)*I[i]/T;
      else
        der(I[i]) = from_CL[i] - nonradio_loss[i]*I[i]/T - decay_loss[i]*I[i];
        outflow[i] = 0;
      end if;
// 输出流分配到ISS_O、FW、DIV
      to_ISS_O[i] = to_ISS_O_Fraction*outflow[i];
      to_FW[i] = to_FW_Fraction*(1 - to_ISS_O_Fraction)*outflow[i];
      to_DIV[i] = (1 - to_FW_Fraction)*(1 - to_ISS_O_Fraction)*outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 85, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {-3, 3}, extent = {{-75, 35}, {75, -35}}, textString = "CPS", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end CPS;
  
  model Blanket
    // 定义输入端口
    Modelica.Blocks.Interfaces.RealInput pulseInput "等离子体脉冲控制信号" annotation(
      Placement(transformation(origin = {-122, 90}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {0, -114}, extent = {{10, 10}, {-10, -10}}, rotation = -90)));
    // 输入端口：来自TES的输入（5维）
    Modelica.Blocks.Interfaces.RealInput from_TES[5] "来自TES的输入" annotation(
      Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, -60}, extent = {{-10, 10}, {10, -10}}, rotation = -180)));
    // 输出端口：输出到Coolant_Loop系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_CL[5] "输出到Coolant_Loop系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-114, 0}, extent = {{10, 10}, {-10, -10}})));
    // 输出端口：输出到TES系统（5维）
    Modelica.Blocks.Interfaces.RealOutput to_TES[5] "输出到TES系统" annotation(
      Placement(transformation(origin = {110, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {114, 60}, extent = {{10, -10}, {-10, 10}}, rotation = 180)));
    // 状态变量：系统中5种物质的储存量
    Real I[5](start = {0, 0, 0, 0, 0}) "系统中5种物质的储存量";
    Real outflow[5](start = {0, 0, 0, 0, 0}) "总输出流";
    // 参数定义
    parameter Real T = 24 "平均滞留时间 (mean residence time)";
    parameter Real decay_loss[5](each unit = "1/h") = {6.4e-6, 0, 0, 0, 0} "Tritium decay loss for 5 materials (放射性衰变损失)";
    parameter Real nonradio_loss[5](each unit = "1") = {0, 0, 0, 0, 0} "非放射性损失";
    parameter Real TBR = 1.1 "Tritium Breeding Ratio (TBR), range: 1.05-1.15";
    parameter Real to_CL_Fraction = 0.01 "输出到CL的比例";
    parameter Real to_TES_Fraction = 1 - to_CL_Fraction "输出到TES的比例";
  equation
// 计算每种物质的动态变化和输出
    for i in 1:5 loop
      der(I[i]) = (if i == 1 then pulseInput*TBR else 0) + from_TES[i] - (1 + nonradio_loss[i])*I[i]/T - decay_loss[i]*I[i];
      outflow[i] = I[i]/T;
      to_TES[i] = to_TES_Fraction*outflow[i];
      to_CL[i] = to_CL_Fraction*outflow[i];
    end for;
    annotation(
      Icon(graphics = {Line(origin = {-100, -100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, 100}, points = {{0, 0}, {200, 0}}, color = {0, 0, 127}), Line(origin = {-100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Line(origin = {100, -100}, points = {{0, 0}, {0, 200}}, color = {0, 0, 127}), Rectangle(fillColor = {255, 170, 255}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}), Text(origin = {1, 1}, extent = {{-73, 37}, {73, -37}}, textString = "Blanket", fontName = "Arial")}),
      uses(Modelica(version = "4.0.0")));
  end Blanket;
  
  model Cycle
    // 实例化脉冲信号模块
    Modelica.Blocks.Sources.Pulse pulseSource(amplitude = 9.60984, period = 10, width = 90) annotation(
      Placement(transformation(origin = {-120, -20}, extent = {{-60, 20}, {-40, 40}})));
    // 实例化 plasma 模型
    Plasma plasma annotation(
      Placement(transformation(origin = {-140, 10}, extent = {{0, -10}, {20, 10}})));
    Fueling_System FS annotation(
      Placement(transformation(origin = {-70, 10}, extent = {{-10, -10}, {10, 10}})));
    SDS sds annotation(
      Placement(transformation(origin = {-10, 10}, extent = {{-10, -10}, {10, 10}})));
    Pump_System pump_System annotation(
      Placement(transformation(origin = {-130, -30}, extent = {{-10, -10}, {10, 10}})));
    TEP_FEP tep_fep annotation(
      Placement(transformation(origin = {-90, -50}, extent = {{-10, -10}, {10, 10}})));
    TEP_IP tep_ip annotation(
      Placement(transformation(origin = {-30, -50}, extent = {{-10, -10}, {10, 10}})));
    TEP_FCU tep_fcu annotation(
      Placement(transformation(origin = {30, -50}, extent = {{-10, -10}, {10, 10}})));
    I_ISS i_iss annotation(
      Placement(transformation(origin = {70, -10}, extent = {{-10, -10}, {10, 10}})));
    WDS wds annotation(
      Placement(transformation(origin = {110, 50}, extent = {{-10, -10}, {10, 10}})));
    O_ISS o_iss annotation(
      Placement(transformation(origin = {50, 50}, extent = {{-10, -10}, {10, 10}})));
    FW fw annotation(
      Placement(transformation(origin = {-90, 170}, extent = {{-10, -10}, {10, 10}})));
    DIV div annotation(
      Placement(transformation(origin = {-90, 90}, extent = {{-10, -10}, {10, 10}})));
    CPS cps annotation(
      Placement(transformation(origin = {-30, 50}, extent = {{-10, -10}, {10, 10}})));
    TES tes annotation(
      Placement(transformation(origin = {50, 130}, extent = {{-10, -10}, {10, 10}})));
    Blanket blanket annotation(
      Placement(transformation(origin = {10, 130}, extent = {{-10, -10}, {10, 10}})));
    Coolant_Pipe coolant_pipe annotation(
      Placement(transformation(origin = {-30, 130}, extent = {{-10, -10}, {10, 10}})));
  equation
// 将脉冲信号的输出连接到 plasma 模型的输入端口
    connect(pulseSource.y, plasma.pulseInput) annotation(
      Line(points = {{-159, 10}, {-142, 10}}, color = {255, 0, 0}, pattern = LinePattern.Dash, thickness = 1, arrow = {Arrow.None, Arrow.Open}, smooth = Smooth.Bezier));
    connect(FS.to_Plasma, plasma.from_Fueling_System) annotation(
      Line(points = {{-81.4, 10}, {-118.4, 10}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(sds.to_FS, FS.from_SDS) annotation(
      Line(points = {{-21.4, 10}, {-59.4, 10}}, color = {0, 0, 127}, thickness = 0.5));
    connect(coolant_pipe.from_FW, fw.to_CL) annotation(
      Line(points = {{-41.4, 138}, {-56.8, 138}, {-56.8, 176}, {-79, 176}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(coolant_pipe.to_FW, fw.from_CL) annotation(
      Line(points = {{-41.4, 134}, {-64.8, 134}, {-64.8, 164}, {-79, 164}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(cps.to_FW, fw.from_CPS) annotation(
      Line(points = {{-41.4, 56}, {-60.4, 56}, {-60.4, 130}, {-90, 150}, {-90, 159}}, color = {0, 0, 127}, pattern = LinePattern.Dash, thickness = 0.5, smooth = Smooth.Bezier));
    connect(cps.to_ISS_O, o_iss.from_CPS) annotation(
      Line(points = {{-18.6, 50}, {38.4, 50}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(coolant_pipe.to_CPS, cps.from_CL) annotation(
      Line(points = {{-30, 118.6}, {-30, 62.6}}, color = {0, 0, 127}, thickness = 0.5));
    connect(blanket.to_CL, coolant_pipe.from_BZ) annotation(
      Line(points = {{-1.4, 130}, {-18.4, 130}}, color = {0, 0, 127}, thickness = 0.5));
    connect(blanket.to_TES, tes.from_BZ) annotation(
      Line(points = {{21.4, 136}, {39.4, 136}}, color = {0, 0, 127}, thickness = 0.5));
    connect(tes.to_BZ, blanket.from_TES) annotation(
      Line(points = {{38.6, 124}, {20.6, 124}}, color = {0, 0, 127}, thickness = 0.5));
    connect(tes.to_O_ISS, o_iss.from_TES) annotation(
      Line(points = {{50, 118.6}, {50, 62.6}}, color = {0, 0, 127}, thickness = 0.5));
    connect(coolant_pipe.to_WDS, wds.from_CL) annotation(
      Line(points = {{-30, 141.4}, {-30, 159.4}, {110, 159.4}, {110, 61}}, color = {0, 0, 127}, thickness = 0.5));
    connect(wds.to_O_ISS, o_iss.from_WDS) annotation(
      Line(points = {{99, 50}, {62.6, 50}}, color = {0, 0, 127}, thickness = 0.5));
    connect(o_iss.to_SDS, sds.from_O_ISS) annotation(
      Line(points = {{50, 38}, {50, 16}, {2, 16}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(plasma.to_Pump, pump_System.from_Plasma) annotation(
      Line(points = {{-130, -2}, {-130, -18}}, color = {0, 0, 127}, thickness = 0.5));
    connect(pump_System.to_TEP_FEP, tep_fep.from_pump) annotation(
      Line(points = {{-130, -42}, {-130, -50}, {-101, -50}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(tep_fep.to_TEP_IP, tep_ip.from_TEP_FEP) annotation(
      Line(points = {{-78.6, -50}, {-41, -50}}, color = {0, 0, 127}, thickness = 0.5));
    connect(tep_ip.to_TEP_FCU, tep_fcu.from_TEP_IP) annotation(
      Line(points = {{-19, -50}, {17.4, -50}}, color = {0, 0, 127}, thickness = 0.5));
    connect(tep_fcu.to_I_ISS, i_iss.from_TEP_FCU) annotation(
      Line(points = {{41, -50}, {70, -50}, {70, -22}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(i_iss.to_SDS, sds.from_I_ISS) annotation(
      Line(points = {{58, -10}, {30, -10}, {30, 4}, {2, 4}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(tep_fep.to_SDS, sds.from_TEP_FEP) annotation(
      Line(points = {{-90, -38}, {-90, -20}, {-10, -20}, {-10, -2}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(pulseSource.y, blanket.pulseInput) annotation(
      Line(points = {{-159, 10}, {-159, 66}, {9.75, 66}, {9.75, 118}, {10, 118}}, color = {255, 0, 0}, pattern = LinePattern.Dash, thickness = 1, arrow = {Arrow.None, Arrow.Open}));
    connect(coolant_pipe.to_DIV, div.from_CL) annotation(
      Line(points = {{-42, 126}, {-66, 126}, {-66, 96}, {-78, 96}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(div.to_CL, coolant_pipe.from_DIV) annotation(
      Line(points = {{-78, 84}, {-54, 84}, {-54, 122}, {-42, 122}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(plasma.to_Div, div.from_plasma) annotation(
      Line(points = {{-124, 22}, {-124, 90}, {-102, 90}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(fw.from_plasma, plasma.to_FW) annotation(
      Line(points = {{-102, 170}, {-136, 170}, {-136, 22}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(i_iss.to_WDS, wds.from_I_ISS) annotation(
      Line(points = {{82, -10}, {110, -10}, {110, 38}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    connect(cps.to_DIV, div.from_CPS) annotation(
      Line(points = {{-42, 44}, {-90, 44}, {-90, 78}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
    annotation(
      Diagram(coordinateSystem(extent = {{-200, 200}, {140, -80}})),
      Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Cycle", fontName = "Arial")}),
      version = "",
      uses);
  end Cycle;
end example_model;
