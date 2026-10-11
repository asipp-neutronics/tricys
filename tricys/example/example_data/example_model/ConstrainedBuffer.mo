within example_model;
model ConstrainedBuffer
  "通用约束型缓冲组件：支持容量上限与处理速率上限，超限部分守恒导出"

  // ===== 输入端口 =====
  Modelica.Blocks.Interfaces.RealInput from_Upstream[5] "上游主入流" annotation(
    Placement(transformation(origin = {-120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, 60}, extent = {{-10, -10}, {10, 10}})));

  Modelica.Blocks.Interfaces.RealInput from_Recycle[5] "回流入口（无回流时接 0）" annotation(
    Placement(transformation(origin = {-120, -40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {-114, -60}, extent = {{-10, -10}, {10, 10}})));

  // ===== 输出端口 =====
  Modelica.Blocks.Interfaces.RealOutput to_Downstream[5] "下游正常出流" annotation(
    Placement(transformation(origin = {120, 40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 60}, extent = {{-10, -10}, {10, 10}})));

  Modelica.Blocks.Interfaces.RealOutput overflow_out[5] "容量超限溢流（守恒）" annotation(
    Placement(transformation(origin = {120, 0}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, 0}, extent = {{-10, -10}, {10, 10}})));

  Modelica.Blocks.Interfaces.RealOutput rate_clip_out[5] "速率超限截流（守恒）" annotation(
    Placement(transformation(origin = {120, -40}, extent = {{-10, -10}, {10, 10}}),
              iconTransformation(origin = {114, -60}, extent = {{-10, -10}, {10, 10}})));

  // ===== 参数定义 =====
  parameter Real T(unit = "h") = 12 "平均滞留时间 (mean residence time)";
  parameter Real capacity_max(unit = "g") = 1e9
    "最大盘存量上限（总氚当量）。设 1e9 禁用约束";
  parameter Real rate_max(unit = "g/h") = 1e9
    "最大出流速率上限（总氚当量）。设 1e9 禁用约束";
  parameter Real softness = 0.02
    "平滑因子：0=硬约束（带事件），0.01-0.1=sigmoid 软约束";
  parameter Real decay_loss[5] (each unit = "1/h") = {6.4e-6, 0, 0, 0, 0}
    "放射性衰变损失率";
  parameter Real nonradio_loss[5] (each unit = "1") = {0, 0, 0, 0, 0}
    "非放射性损失因子";
  parameter Real to_Down_Fraction = 1.0
    "正常出流中导向下游端口的比例（余量入 overflow）";

  // ===== 状态变量 =====
  Real I[5](start = {0, 0, 0, 0, 0}) "各同位素盘存量";
  Real I_total "总盘存量";
  Real outflow_nominal[5] "未约束的名义出流";
  Real outflow_total_nominal "名义总出流";
  Real rate_scale "速率约束缩放因子 (0,1]";
  Real admit_scale "容量约束准入因子 [0,1]";
  Real outflow[5] "实际出流（约束后）";
  Real inflow_total[5] "合计入流";
  Real inflow_admit[5] "准入后的入流";

equation
  // --- 聚合计算 ---
  I_total = sum(I);
  for i in 1:5 loop
    inflow_total[i] = from_Upstream[i] + from_Recycle[i];
    outflow_nominal[i] = I[i] / T;
  end for;
  outflow_total_nominal = sum(outflow_nominal);

  // --- 速率约束：限制总出流不超过 rate_max ---
  if softness <= 0 then
    // 硬约束：直接截断
    rate_scale = min(1.0, rate_max / max(outflow_total_nominal, 1e-30));
  else
    // 软约束：sigmoid 平滑过渡
    // 当 outflow_total_nominal << rate_max 时 rate_scale ≈ 1
    // 当 outflow_total_nominal >> rate_max 时 rate_scale ≈ rate_max/outflow_total_nominal
    rate_scale = (rate_max / max(outflow_total_nominal, 1e-30))
               + (1.0 - rate_max / max(outflow_total_nominal, 1e-30))
                 / (1.0 + exp((outflow_total_nominal - rate_max) / (softness * rate_max + 1e-30)));
  end if;

  // --- 容量约束：限制入流使盘存不超过 capacity_max ---
  if softness <= 0 then
    admit_scale = if I_total >= capacity_max then 0.0 else 1.0;
  else
    admit_scale = 1.0 / (1.0 + exp((I_total - capacity_max) / (softness * capacity_max + 1e-30)));
  end if;

  // --- 逐同位素质量平衡 ---
  for i in 1:5 loop
    outflow[i] = rate_scale * outflow_nominal[i];
    inflow_admit[i] = admit_scale * inflow_total[i];
    der(I[i]) = (inflow_admit[i]
              - (1 + nonradio_loss[i]) * outflow[i]
              - decay_loss[i] * I[i]) / time_scale;
    to_Downstream[i] = to_Down_Fraction * outflow[i];
    overflow_out[i] = (1.0 - admit_scale) * inflow_total[i]
                    + (1.0 - to_Down_Fraction) * outflow[i];
    rate_clip_out[i] = (1.0 - rate_scale) * outflow_nominal[i];
  end for;

annotation(
    Icon(graphics = {
      Rectangle(fillColor = {170, 255, 170}, fillPattern = FillPattern.Solid, extent = {{-100, 100}, {100, -100}}),
      Text(origin = {0, 20}, extent = {{-80, 30}, {80, -10}}, textString = "Constrained", fontName = "Arial"),
      Text(origin = {0, -20}, extent = {{-80, 10}, {80, -30}}, textString = "Buffer", fontName = "Arial"),
      Line(origin = {0, 60}, points = {{-60, 0}, {60, 0}}, color = {255, 0, 0}, thickness = 2),
      Line(origin = {0, -60}, points = {{-60, 0}, {60, 0}}, color = {255, 0, 0}, thickness = 2)}),
    Documentation(info = "<html>
      <p>通用约束型缓冲组件，适用于聚变氚燃料循环系统仿真。</p>
      <h4>功能</h4>
      <ul>
        <li><b>容量约束</b>: 当总盘存量接近 capacity_max 时，准入因子 admit_scale 趋向 0，多余入流转入 overflow_out 端口</li>
        <li><b>速率约束</b>: 当总出流接近 rate_max 时，rate_scale 趋向饱和值，超出部分转入 rate_clip_out 端口</li>
        <li><b>质量守恒</b>: 所有被约束截断的物质流均通过专用端口导出，不凭空消失</li>
      </ul>
      <h4>参数</h4>
      <ul>
        <li>capacity_max = 1e9: 设为极大值时退化为无约束行为</li>
        <li>rate_max = 1e9: 设为极大值时退化为标准 I/T 出流</li>
        <li>softness = 0.02: sigmoid 平滑因子，避免求解器事件风暴</li>
      </ul>
    </html>"),
    uses(Modelica(version = "4.0.0")));
end ConstrainedBuffer;
