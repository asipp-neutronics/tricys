within example_model;

model DIV_Interceptor
  Modelica.Blocks.Interfaces.RealInput physical_to_CL[5] "Received from DIV";

  Modelica.Blocks.Interfaces.RealOutput final_to_CL[5] "Final output";

protected
  parameter String fileName = "D:/Administrator/Documents/Github/tricys/example/temp/div_outputs.csv" "Path to the CSV file";
  parameter Integer columns_to_CL[6] = {1,2,3,4,5,6} "Column mapping for to_CL: {time, y1, y2, ...}. Use 1 for pass-through";


  Modelica.Blocks.Sources.CombiTimeTable table_to_CL(
    tableName="csv_data_to_CL",
    fileName=fileName,
    columns=columns_to_CL,
    tableOnFile = true
  ) annotation(HideResult=true);

equation
  // Element-wise connection for to_CL
  for i in 1:5 loop
    final_to_CL[i] = if columns_to_CL[i+1] <> 1 then table_to_CL.y[i] else physical_to_CL[i];
  end for;

annotation(
  Icon(graphics = {
    Rectangle(fillColor = {255, 255, 180}, extent = {{-100, 100}, {100, -100}}),
    Text(extent = {{-80, 40}, {80, -40}}, textString = "DIV\nInterceptor")
  }
));
end DIV_Interceptor;