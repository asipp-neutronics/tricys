within CFEDR;
model Cycle
  Plasma plasma annotation(
    Placement(transformation(origin = {-140, 10}, extent = {{0, -10}, {20, 10}})));
  Fueling_System fs annotation(
    Placement(transformation(origin = {-70, 10}, extent = {{-10, -10}, {10, 10}})));
  SDS sds annotation(
    Placement(transformation(origin = {-10, 10}, extent = {{-10, -10}, {10, 10}})));
  Pump_System ps annotation(
    Placement(transformation(origin = {-90, -50}, extent = {{-10, -10}, {10, 10}})));
  I_ISS i_iss annotation(
    Placement(transformation(origin = {70, -10}, extent = {{-10, -10}, {10, 10}})));
  WDS wds annotation(
    Placement(transformation(origin = {110, 50}, extent = {{-10, -10}, {10, 10}})));
  O_ISS o_iss annotation(
    Placement(transformation(origin = {50, 50}, extent = {{-10, -10}, {10, 10}})));
  FW fw annotation(
    Placement(transformation(origin = {-90, 150}, extent = {{-10, -10}, {10, 10}})));
  DIV div(T = 250)  annotation(
    Placement(transformation(origin = {-90, 110}, extent = {{-10, -10}, {10, 10}})));
  CPS cps annotation(
    Placement(transformation(origin = {-30, 50}, extent = {{-10, -10}, {10, 10}})));
  TES tes annotation(
    Placement(transformation(origin = {50, 130}, extent = {{-10, -10}, {10, 10}})));
  Blanket blanket annotation(
    Placement(transformation(origin = {10, 130}, extent = {{-10, -10}, {10, 10}})));
  Coolant_Pipe cp annotation(
    Placement(transformation(origin = {-30, 130}, extent = {{-10, -10}, {10, 10}})));
  TEP tep annotation(
    Placement(transformation(origin = {-10, -50}, extent = {{-10, -10}, {10, 10}})));
  pulseFusion pulse annotation(
    Placement(transformation(origin = {-170, 70}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(fs.to_Plasma, plasma.from_Fueling_System) annotation(
    Line(points = {{-81.4, 10}, {-118.4, 10}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(sds.to_FS, fs.from_SDS) annotation(
    Line(points = {{-21.4, 10}, {-59.4, 10}}, color = {0, 0, 127}, thickness = 0.5));
  connect(cp.from_FW, fw.to_CP) annotation(
    Line(points = {{-41.4, 138}, {-56.8, 138}, {-56.8, 156}, {-79, 156}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cp.to_FW, fw.from_CP) annotation(
    Line(points = {{-41.4, 134}, {-64.8, 134}, {-64.8, 144}, {-79, 144}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cps.to_FW, fw.from_CPS) annotation(
    Line(points = {{-41.4, 56}, {-60.4, 56}, {-60.4, 112}, {-90, 130}, {-90, 139}}, color = {0, 0, 127}, pattern = LinePattern.Dash, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cps.to_O_ISS, o_iss.from_CPS) annotation(
    Line(points = {{-18.6, 50}, {38.4, 50}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(blanket.to_CP, cp.from_BZ) annotation(
    Line(points = {{-1.4, 130}, {-18.4, 130}}, color = {0, 0, 127}, thickness = 0.5));
  connect(blanket.to_TES, tes.from_BZ) annotation(
    Line(points = {{21.4, 136}, {39.4, 136}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tes.to_BZ, blanket.from_TES) annotation(
    Line(points = {{38.6, 124}, {20.6, 124}}, color = {0, 0, 127}, thickness = 0.5));
  connect(tes.to_O_ISS, o_iss.from_TES) annotation(
    Line(points = {{50, 118.6}, {50, 62.6}}, color = {0, 0, 127}, thickness = 0.5));
  connect(cp.to_WDS, wds.from_CP) annotation(
    Line(points = {{-30, 141.4}, {-30, 159.4}, {110, 159.4}, {110, 61}}, color = {0, 0, 127}, thickness = 0.5));
  connect(wds.to_O_ISS, o_iss.from_WDS) annotation(
    Line(points = {{99, 50}, {62.6, 50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(o_iss.to_SDS, sds.from_O_ISS) annotation(
    Line(points = {{50, 38}, {50, 16}, {2, 16}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(div.to_CP, cp.from_DIV) annotation(
    Line(points = {{-79, 104}, {-54, 104}, {-54, 122}, {-42, 122}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cps.to_DIV, div.from_CPS) annotation(
    Line(points = {{-42, 44}, {-90, 47}, {-90, 99}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cp.to_DIV, div.from_CP) annotation(
    Line(points = {{-42, 126}, {-62, 126}, {-62, 116}, {-79, 116}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(pulse.y2, plasma.pulseInput) annotation(
    Line(points = {{-158, 66}, {-152, 66}, {-152, 10}, {-142, 10}}, color = {255, 0, 0}, pattern = LinePattern.Dash, thickness = 1));
  connect(pulse.y1, blanket.pulseInput) annotation(
    Line(points = {{-158, 74}, {10, 74}, {10, 118}}, color = {255, 0, 0}, pattern = LinePattern.Dash, thickness = 1));
  connect(sds.from_TEP, tep.to_SDS) annotation(
    Line(points = {{-10, -2}, {-10, -38}}, color = {0, 0, 127}, thickness = 0.5));
  connect(i_iss.from_TEP, tep.to_I_ISS) annotation(
    Line(points = {{70, -22}, {70, -50}, {2, -50}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(ps.from_Plasma, plasma.to_Pump) annotation(
    Line(points = {{-102, -50}, {-130, -50}, {-130, -2}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(ps.to_TEP, tep.from_Pump) annotation(
    Line(points = {{-78, -50}, {-22, -50}}, color = {0, 0, 127}, thickness = 0.5));
  connect(i_iss.to_SDS, sds.from_I_ISS) annotation(
    Line(points = {{58, -10}, {30, -10}, {30, 4}, {2, 4}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(i_iss.to_WDS, wds.from_I_ISS) annotation(
    Line(points = {{82, -10}, {110, -10}, {110, 38}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(plasma.to_FW, fw.from_plasma) annotation(
    Line(points = {{-136, 22}, {-136, 150}, {-102, 150}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(plasma.to_Div, div.from_plasma) annotation(
    Line(points = {{-124, 22}, {-124, 110}, {-102, 110}}, color = {0, 0, 127}, thickness = 0.5, smooth = Smooth.Bezier));
  connect(cp.to_CPS, cps.from_CP) annotation(
    Line(points = {{-30, 118}, {-30, 62}}, color = {0, 0, 127}, thickness = 0.5));
  annotation(
    Diagram(coordinateSystem(extent = {{-200, 200}, {140, -80}})), 
    Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-100, 100}, {100, -100}}), Text(extent = {{-80, 80}, {80, -80}}, textString = "Cycle", fontName = "Arial")}), 
    version = "", 
    uses,experiment(Algorithm=Dassl,InlineIntegrator=false,InlineStepSize=false,Interval=0.05,StartTime=0,StopTime=24000,Tolerance=1e-06),__MWORKS(ContinueSimConfig(SaveContinueFile="false",SaveBeforeStop="false",NumberBeforeStop=1,FixedContinueInterval="false",ContinueIntervalLength=24000,ContinueTimeVector)));
end Cycle;
