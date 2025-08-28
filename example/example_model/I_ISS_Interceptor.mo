within example_model;

model I_ISS_Interceptor
  Modelica.Blocks.Interfaces.RealInput physical_to_SDS[5] "Received from I_ISS";
  Modelica.Blocks.Interfaces.RealInput physical_to_WDS[5] "Received from I_ISS";

  Modelica.Blocks.Interfaces.RealOutput final_to_SDS[5] "Final output";
  Modelica.Blocks.Interfaces.RealOutput final_to_WDS[5] "Final output";

protected
  parameter String fileName = "D:/Administrator/Documents/Github/tricys/example/temp/i_iss_outputs.csv" "Path to the CSV file";
  parameter Integer columns_to_SDS[6] = {1,2,3,4,1,1} "Column mapping for to_SDS: {time, y1, y2, ...}. Use 1 for pass-through";
  parameter Integer columns_to_WDS[6] = {1,5,6,7,1,1} "Column mapping for to_WDS: {time, y1, y2, ...}. Use 1 for pass-through";


  Modelica.Blocks.Sources.CombiTimeTable table_to_SDS(
    tableName="csv_data_to_SDS",
    fileName=fileName,
    columns=columns_to_SDS,
    tableOnFile = true
  ) annotation(HideResult=true);

  Modelica.Blocks.Sources.CombiTimeTable table_to_WDS(
    tableName="csv_data_to_WDS",
    fileName=fileName,
    columns=columns_to_WDS,
    tableOnFile = true
  ) annotation(HideResult=true);

equation
  // Element-wise connection for to_SDS
  for i in 1:5 loop
    final_to_SDS[i] = if columns_to_SDS[i+1] <> 1 then table_to_SDS.y[i] else physical_to_SDS[i];
  end for;
  // Element-wise connection for to_WDS
  for i in 1:5 loop
    final_to_WDS[i] = if columns_to_WDS[i+1] <> 1 then table_to_WDS.y[i] else physical_to_WDS[i];
  end for;

annotation(
  Icon(graphics = {
    Rectangle(fillColor = {255, 255, 180}, extent = {{-100, 100}, {100, -100}}),
    Text(extent = {{-80, 40}, {80, -40}}, textString = "I_ISS\nInterceptor")
  }
));
end I_ISS_Interceptor;