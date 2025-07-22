within CFEDR;

block pulseFusion "Generate two identical pulse signals of type Real"
  // 参数定义
  parameter Real amplitude = 9.60984 "脉冲幅度";
  parameter Real width(final min = 1e-10, final max = 100) = 100 "脉冲宽度，占周期的百分比";
  parameter Real period(final min = 1e-10, start = 480) = 240"一个周期的时间（小时）";
  parameter Integer nperiod = -1 "周期数（< 0 表示无限周期）";
  parameter Real startTime = 0 "第一个脉冲的开始时间（秒）";
  parameter Real offset = 0 "输出信号的偏移量";
  // 输出定义，添加 Placement 注解以在组件视图中显示端口
  Modelica.Blocks.Interfaces.RealOutput y1 annotation(
    Placement(transformation(origin = {108, 20}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {112, 40}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput y2 annotation(
    Placement(transformation(origin = {106, -40}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {112, -40}, extent = {{-10, -10}, {10, 10}})));
protected
  // 内部变量
  Real T_width = period*width/100 "脉冲宽度时间（秒）";
  Real T_start "当前周期的开始时间（秒）";
  Integer count "周期计数";
initial algorithm
// 初始化周期计数和当前周期开始时间
  count := integer((time - startTime)/period);
  T_start := startTime + count*period;
equation
// 当进入新周期时，更新计数和开始时间
  when integer((time - startTime)/period) > pre(count) then
    count = pre(count) + 1;
    T_start = time;
  end when;
// 计算输出 y1
  y1 = offset + (if (time < startTime or nperiod == 0 or (nperiod > 0 and count >= nperiod)) then 0 else if time < T_start + T_width then amplitude else 0);
// 确保 y2 与 y1 相同
  y2 = y1;
  annotation(
    Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Line(points = {{-80, 68}, {-80, -80}}, color = {192, 192, 192}), Polygon(lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid, points = {{-80, 90}, {-88, 68}, {-72, 68}, {-80, 90}}), Line(points = {{-90, -70}, {82, -70}}, color = {192, 192, 192}), Polygon(lineColor = {192, 192, 192}, fillColor = {192, 192, 192}, fillPattern = FillPattern.Solid, points = {{90, -70}, {68, -62}, {68, -78}, {90, -70}}), Line(points = {{-80, -70}, {-40, -70}, {-40, 44}, {0, 44}, {0, -70}, {40, -70}, {40, 44}, {79, 44}}), Text(extent = {{-147, -152}, {153, -112}}, textString = "period=%period"), Text(textColor = {0, 0, 255}, extent = {{70, 60}, {100, 30}}, textString = "y1"), Text(textColor = {0, 0, 255}, extent = {{70, 30}, {100, 0}}, textString = "y2"), Rectangle(extent = {{-100, 100}, {100, -100}})}),
    Documentation(info = "<html>
<p>
实数输出 y1 和 y2 是相同的脉冲信号。
</p>

<p>
此块生成两个相同的脉冲信号，具有相同的幅度、宽度、周期和偏移量，由参数定义。时间参数（period 和 startTime）以秒为单位。此块不依赖 Modelica 标准库，旨在避免在模型中重复使用 Pulse 块，当需要两个相同脉冲信号时使用。
</p>
</html>"),
    uses(Modelica(version = "4.0.0")));
end pulseFusion;
