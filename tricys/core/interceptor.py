import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from OMPython import OMCSessionZMQ

logger = logging.getLogger(__name__)


def _generate_interceptor(
    submodel_name: str,
    output_ports: list[Dict[str, Any]],
    csv_file: str,
    add_within_clause: bool = True,
) -> tuple[str, str]:
    """Generates the Modelica code for an interceptor model.

    The interceptor wraps an existing model, allowing its output ports to be
    overridden by data from a CSV file.

    Args:
        submodel_name: The full name of the submodel to be intercepted (e.g., 'MyPackage.MyModel').
        output_ports: A list of dictionaries, where each dictionary describes an output port.
            Each dictionary should have 'name', 'dim', and 'default_column' keys.
        csv_file: The path to the CSV file to be used for data injection.
        add_within_clause: Whether to add the 'within <package_name>;' clause. Defaults to True.

    Returns:
        A tuple containing (interceptor_model_name, generated_modelica_code).

    Note:
        The generated interceptor uses CombiTimeTable for each output port to allow
        selective override of output signals. Column mapping is configurable via
        parameters, where column index 1 indicates pass-through mode.
    """
    package_name = submodel_name.split(".")[0]
    original_model_name = submodel_name.split(".")[-1]
    interceptor_name = f"{original_model_name}_Interceptor"

    inputs_code = ""
    outputs_code = ""
    parameters_code = (
        f'  parameter String fileName = "{csv_file}" "Path to the CSV file";\n'
    )
    protected_declarations_code = ""
    equation_code = "equation\n"

    logger.info(
        "Generating interceptor code for ports",
        extra={"ports": [p["name"] for p in output_ports]},
    )
    for port in output_ports:
        dim_str = f'[{port["dim"]}]' if port["dim"] > 1 else ""
        port_name = port["name"]

        # 1. Generate Input and Output port declarations (no change)
        inputs_code += f'  Modelica.Blocks.Interfaces.RealInput physical_{port_name}{dim_str} "Received from {original_model_name}";\n'
        outputs_code += f'  Modelica.Blocks.Interfaces.RealOutput final_{port_name}{dim_str} "Final output";\n'

        # 2. Generate a configurable 'columns' parameter for each port
        parameters_code += f'  parameter Integer columns_{port_name}[{port["dim"] + 1}] = {port["default_column"]} "Column mapping for {port_name}: {{time, y1, y2, ...}}. Use 1 for pass-through";\n'

        # 3. Generate the CombiTimeTable instance in the 'protected' section
        table_name = f"table_{port_name}"
        protected_declarations_code += f"""
  Modelica.Blocks.Sources.CombiTimeTable {table_name}(
    tableName="csv_data_{port_name}",
    fileName=fileName,
    columns=columns_{port_name},
    tableOnFile = true
  ) annotation(HideResult=true);
"""

        # 4. Generate the equation logic with element-by-element control (removed useCSV)
        if port["dim"] > 1:
            # Vector port: Use a 'for' loop for granular control
            equation_code += (
                f"  // Element-wise connection for {port_name}\n"
                f"  for i in 1:{port['dim']} loop\n"
                f"    final_{port_name}[i] = if columns_{port_name}[i+1] <> 1 then {table_name}.y[i] else physical_{port_name}[i];\n"
                f"  end for;\n"
            )
        else:
            # Scalar port: Use a simpler if-statement
            equation_code += (
                f"  // Connection for {port_name}\n"
                f"  final_{port_name} = if columns_{port_name}[2] <> 1 then {table_name}.y[1] else physical_{port_name};\n"
            )

        # Assemble the final model string

        within_clause = f"within {package_name};\n\n" if add_within_clause else ""

        model_template = f"""

    {within_clause}model {interceptor_name}

    {inputs_code}

    {outputs_code}

    protected

    {parameters_code}

    {protected_declarations_code}

    {equation_code}

    annotation(

      Icon(graphics = {{

        Rectangle(fillColor = {{255, 255, 180}}, extent = {{{{-100, 100}}, {{100, -100}}}}),

        Text(extent = {{{{-80, 40}}, {{80, -40}}}}, textString = \"{original_model_name}\\nInterceptor\")

      }}));

    end {interceptor_name};

    """
    return interceptor_name, model_template.strip()


def _integrate_interceptor_single_file(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handles the model interception process for a single-file Modelica package.

    This function reads the entire package code, generates interceptor models,
    and modifies the system model in-memory before writing the result to a
    new `_intercepted.mo` file.

    Args:
        package_path: The path to the single `.mo` file.
        model_name: The full name of the system model to be modified.
        interception_configs: A list of dictionaries for interception tasks.
            Each dict should contain 'submodel_name', 'csv_uri', 'instance_name',
            and 'output_placeholder' keys.

    Returns:
        A dictionary containing:
            - interceptor_model_paths: List of interceptor model paths (empty for single-file)
            - system_model_path: Path to the modified system model file

    Note:
        The original package model is preserved. A new '_Intercepted' variant is created
        with interceptor models embedded in the same file. The output file has suffix
        '_intercepted.mo'. Uses OMCSessionZMQ to inspect model components.
    """
    omc = None
    logger.info(
        "Starting model processing for single-file package",
        extra={
            "package_path": package_path,
            "num_interception_tasks": len(interception_configs),
        },
    )
    try:
        with open(package_path, "r", encoding="utf-8") as f:
            original_package_code = f.read()

        omc = OMCSessionZMQ()
        omc.sendExpression(f'loadFile("{Path(package_path).as_posix()}")')

        output_dir = os.path.dirname(package_path)

        # Part 1: Generate all individual interceptor model codes as strings
        generated_interceptor_codes = []
        for config in interception_configs:
            submodel_name = config["submodel_name"]
            csv_uri = config["csv_uri"]
            column_config = config["output_placeholder"]

            logger.info(
                "Identifying output ports", extra={"submodel_name": submodel_name}
            )
            components = omc.sendExpression(f"getComponents({submodel_name})")
            output_ports = []
            for comp in components:
                if comp[0] == "Modelica.Blocks.Interfaces.RealOutput":
                    dim = int(comp[11][0]) if comp[11] else 1
                    output_ports.append(
                        {
                            "name": comp[1],
                            "type": comp[0],
                            "dim": dim,
                            "comment": comp[2],
                            "default_column": column_config.get(comp[1], ""),
                        }
                    )

            if not output_ports:
                raise ValueError(f"No RealOutput ports found in model {submodel_name}.")

            config["output_ports"] = output_ports
            logger.info(
                "Identified output ports",
                extra={"output_ports": [p["name"] for p in output_ports]},
            )
            interceptor_name, interceptor_code = _generate_interceptor(
                submodel_name, output_ports, csv_uri, add_within_clause=False
            )
            generated_interceptor_codes.append(interceptor_code)
            # Store the generated name for rewriting the main model
            config["interceptor_name"] = interceptor_name

    finally:
        if omc:
            omc.sendExpression("quit()")

    # Part 2: Isolate and modify the system model code from the package string
    model_short_name = model_name.split(".")[-1]
    system_model_pattern = re.compile(
        r"(model\s+"
        + re.escape(model_short_name)
        + r".*?end\s+"
        + re.escape(model_short_name)
        + r"\s*;)",
        re.DOTALL,
    )
    match = system_model_pattern.search(original_package_code)
    if not match:
        raise ValueError(
            f"Could not find system model '{model_short_name}' in the provided file."
        )

    original_system_code = match.group(1)
    modified_system_code = original_system_code

    all_interceptor_declarations = ""

    for config in interception_configs:
        instance_name_in_system = config["instance_name"]
        output_ports = config["output_ports"]
        package_name = config["submodel_name"].split(".")[0]
        interceptor_name = config["interceptor_name"]
        interceptor_instance_name = f"{instance_name_in_system}_interceptor"

        for port in output_ports:
            port_name = port["name"]
            pattern = re.compile(
                r"(connect\s*\(\s*"
                + re.escape(instance_name_in_system)
                + r"\."
                + re.escape(port_name)
                + r"\s*,\s*)(.*?)\s*\)(.*?;)",
                re.IGNORECASE | re.DOTALL,
            )
            replacement = (
                f"connect({instance_name_in_system}.{port_name}, {interceptor_instance_name}.physical_{port_name});\n"
                f"    connect({interceptor_instance_name}.final_{port_name}, \\2)\\3"
            )
            modified_system_code, num_subs = pattern.subn(
                replacement, modified_system_code
            )
            if num_subs > 0:
                logger.info(
                    f"Successfully rewired port '{port_name}' for instance '{instance_name_in_system}'."
                )
            else:
                logger.warning(
                    "Could not find a connection for port",
                    extra={
                        "port_name": port_name,
                        "instance_name": instance_name_in_system,
                    },
                )

        all_interceptor_declarations += (
            f"  {package_name}.{interceptor_name} {interceptor_instance_name};\n"
        )

    # Part 3: Insert declarations and rename the system model block
    final_system_code_block, num_subs = re.subn(
        r"(equation)",
        all_interceptor_declarations + r"\n\1",
        modified_system_code,
        count=1,
        flags=re.IGNORECASE,
    )
    if num_subs == 0:
        final_system_code_block = modified_system_code.replace(
            f"end {model_short_name};",
            f"{all_interceptor_declarations}end {model_short_name};",
        )

    intercepted_system_name = f"{model_short_name}_Intercepted"
    final_system_code_block = re.sub(
        r"(\bmodel\s+)" + re.escape(model_short_name),
        r"\1" + intercepted_system_name,
        final_system_code_block,
        count=1,
    )
    final_system_code_block = re.sub(
        r"(\bend\s+)" + re.escape(model_short_name) + r"(\s*;)",
        r"\1" + intercepted_system_name + r"\2",
        final_system_code_block,
    )

    # Part 4: Assemble the final package code string
    # 4a. Keep the original system model unchanged
    # 4b. Add the new Intercepted system model and interceptor models before package end

    # Collect all new models to add
    new_models_code = []

    # Add all interceptor models
    new_models_code.extend(generated_interceptor_codes)

    # Add the new Intercepted system model
    new_models_code.append(final_system_code_block)

    # Combine all new models
    all_new_models_code = "\n\n".join(new_models_code)

    # Insert before the package end statement
    package_name = model_name.split(".")[0]
    final_package_code = original_package_code.replace(
        f"end {package_name};", f"{all_new_models_code}\n\nend {package_name};"
    )

    logger.info(
        "Created new Intercepted system model while preserving original",
        extra={
            "original_model": model_short_name,
            "new_model": intercepted_system_name,
        },
    )

    # Part 5: Save the new single-file package
    original_filename = os.path.basename(package_path)
    modified_filename = original_filename.replace(".mo", "_intercepted.mo")
    modified_system_file_path = os.path.join(output_dir, modified_filename)

    with open(modified_system_file_path, "w", encoding="utf-8") as f:
        f.write(final_package_code)

    logger.info("Automated modification complete")

    return {
        "interceptor_model_paths": [],
        "system_model_path": modified_system_file_path,
    }


def _integrate_interceptor_multi_file(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handles the model interception process for a multi-file Modelica package.

    This function creates new `.mo` files for each interceptor model and generates
    a modified version of the main system model with re-routed connections.

    Args:
        package_path: The path to the main `package.mo` file.
        model_name: The full name of the system model to be modified.
        interception_configs: A list of dictionaries for interception tasks.
            Each dict should contain 'submodel_name', 'csv_uri', 'instance_name',
            and 'output_placeholder' keys.

    Returns:
        A dictionary containing:
            - interceptor_model_paths: Paths to the generated interceptor model files
            - system_model_path: Path to the modified system model file

    Note:
        Each interceptor is written as a separate .mo file in the package directory.
        The system model is renamed with '_Intercepted' suffix. Connection statements
        are rewritten to route through interceptor instances using regex pattern matching.
    """
    omc = None
    logger.info(
        "Starting model processing for multi-file package",
        extra={
            "num_interception_tasks": len(interception_configs),
        },
    )
    try:
        omc = OMCSessionZMQ()
        omc.sendExpression(f'loadFile("{Path(package_path).as_posix()}")')

        logger.info("Proceeding with multi-interceptor model generation")
        package_dir = os.path.dirname(package_path)
        model_short_name = model_name.split(".")[-1]
        system_model_path = os.path.join(package_dir, f"{model_short_name}.mo")

        if not os.path.exists(system_model_path):
            raise FileNotFoundError(
                f"Inferred system model path does not exist: {system_model_path}"
            )

        output_dir = os.path.dirname(system_model_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Proceeding with multi-interceptor model generation")
        package_dir = os.path.dirname(package_path)
        model_short_name = model_name.split(".")[-1]
        system_model_path = os.path.join(package_dir, f"{model_short_name}.mo")

        if not os.path.exists(system_model_path):
            raise FileNotFoundError(
                f"Inferred system model path does not exist: {system_model_path}"
            )

        generated_interceptor_files = []

        # Part 1: Generate all individual interceptor model files
        for config in interception_configs:
            submodel_name = config["submodel_name"]
            csv_uri = config["csv_uri"]
            column_config = config["output_placeholder"]

            logger.info(
                "Identifying output ports", extra={"submodel_name": submodel_name}
            )
            components = omc.sendExpression(f"getComponents({submodel_name})")
            output_ports = []
            for comp in components:
                if comp[0] == "Modelica.Blocks.Interfaces.RealOutput":
                    dim = int(comp[11][0]) if comp[11] else 1
                    output_ports.append(
                        {
                            "name": comp[1],
                            "type": comp[0],
                            "dim": dim,
                            "comment": comp[2],
                            "default_column": column_config.get(comp[1], ""),
                        }
                    )

            if not output_ports:
                raise ValueError(f"No RealOutput ports found in model {submodel_name}.")

            config["output_ports"] = output_ports
            logger.info(
                "Identified output ports",
                extra={"output_ports": [p["name"] for p in output_ports]},
            )

            package_name = submodel_name.split(".")[0]
            original_model_short_name = submodel_name.split(".")[-1]
            interceptor_name, interceptor_code = _generate_interceptor(
                submodel_name, output_ports, csv_uri
            )

            interceptor_file_path = os.path.join(output_dir, f"{interceptor_name}.mo")
            with open(interceptor_file_path, "w", encoding="utf-8") as f:
                f.write(interceptor_code)
            logger.info(
                "Generated interceptor model file",
                extra={"file_path": interceptor_file_path},
            )
            generated_interceptor_files.append(interceptor_file_path)

    finally:
        if omc:
            omc.sendExpression("quit()")

    # Part 2: Modify the system model to include all interceptors
    with open(system_model_path, "r", encoding="utf-8") as f:
        modified_system_code = f.read()

    all_interceptor_declarations = ""

    for config in interception_configs:
        instance_name_in_system = config["instance_name"]
        output_ports = config["output_ports"]

        package_name = config["submodel_name"].split(".")[0]
        original_model_short_name = config["submodel_name"].split(".")[-1]
        interceptor_name = f"{original_model_short_name}_Interceptor"
        interceptor_instance_name = f"{instance_name_in_system}_interceptor"

        for port in output_ports:
            port_name = port["name"]
            pattern = re.compile(
                r"(connect\s*\(\s*"
                + re.escape(instance_name_in_system)
                + r"\."
                + re.escape(port_name)
                + r"\s*,\s*)(.*?)\s*\)(.*?;)",
                re.IGNORECASE | re.DOTALL,
            )
            replacement = (
                f"connect({instance_name_in_system}.{port_name}, {interceptor_instance_name}.physical_{port_name});\n"
                f"    connect({interceptor_instance_name}.final_{port_name}, \\2)\\3"
            )
            modified_system_code, num_subs = pattern.subn(
                replacement, modified_system_code
            )
            if num_subs > 0:
                logger.info(
                    "Successfully rewired port",
                    extra={
                        "port_name": port_name,
                        "instance_name": instance_name_in_system,
                    },
                )
            else:
                logger.warning(
                    "Could not find a connection for port",
                    extra={
                        "port_name": port_name,
                        "instance_name": instance_name_in_system,
                    },
                )

        all_interceptor_declarations += (
            f"  {package_name}.{interceptor_name} {interceptor_instance_name};\n"
        )

    # Part 3: Insert all declarations and save the final model
    final_system_code, num_subs = re.subn(
        r"(equation)",
        all_interceptor_declarations + r"\n\1",
        modified_system_code,
        count=1,
        flags=re.IGNORECASE,
    )
    if num_subs == 0:
        model_name_from_path = os.path.basename(system_model_path).replace(".mo", "")
        final_system_code = modified_system_code.replace(
            f"end {model_name_from_path};",
            f"{all_interceptor_declarations}end {model_name_from_path};",
        )

    original_system_name = os.path.basename(system_model_path).replace(".mo", "")
    intercepted_system_name = f"{original_system_name}_Intercepted"
    modified_system_filename = f"{intercepted_system_name}.mo"
    modified_system_file_path = os.path.join(output_dir, modified_system_filename)

    final_system_code = re.sub(
        r"(\bmodel\s+)" + re.escape(original_system_name),
        r"\1" + intercepted_system_name,
        final_system_code,
        count=1,
    )
    final_system_code = re.sub(
        r"(\bend\s+)" + re.escape(original_system_name) + r"(\s*;)",
        r"\1" + intercepted_system_name + r"\2",
        final_system_code,
    )

    with open(modified_system_file_path, "w", encoding="utf-8") as f:
        f.write(final_system_code)

    logger.info(
        "Generated modified main model file",
        extra={
            "file_path": modified_system_file_path,
        },
    )
    logger.info("Automated modification complete")

    return {
        "interceptor_model_paths": generated_interceptor_files,
        "system_model_path": modified_system_file_path,
    }


def _replace_submodel_with_csv(
    submodel_path: str,
    output_ports: list[Dict[str, Any]],
    csv_file: str,
    backup_suffix: str = "_backup",
) -> Dict[str, Any]:
    """Replaces a submodel's internal equations with CSV data source.

    This function directly modifies the submodel by:
    1. Backing up the original model file.
    2. Keeping all input/output port declarations unchanged.
    3. Removing all equations and internal declarations.
    4. Adding CombiTimeTable components to read CSV data.
    5. Creating simple mapping equations: output = table.y

    Args:
        submodel_path: Path to the submodel .mo file to be replaced.
        output_ports: A list of dictionaries describing output ports.
            Each dict should have 'name', 'dim', and 'default_column' keys.
        csv_file: Path to the CSV file containing replacement data.
        backup_suffix: Suffix for the backup file. Defaults to "_backup".

    Returns:
        A dictionary containing:
            - original_path: Path to the original model file
            - backup_path: Path to the backup file (.bak extension)
            - modified_path: Path to the modified model file

    Note:
        The backup uses .bak extension regardless of backup_suffix parameter.
        Conditional logic is added to output 0.0 when column mapping is 1.
        All annotations and port declarations are preserved from the original model.
    """
    if not os.path.exists(submodel_path):
        raise FileNotFoundError(f"Submodel file not found: {submodel_path}")

    logger.info(
        "Starting submodel replacement with CSV data",
        extra={"submodel_path": submodel_path, "csv_file": csv_file},
    )

    # Step 1: Backup the original file - use .bak extension
    backup_path = submodel_path.replace(".mo", ".bak")
    shutil.copy2(submodel_path, backup_path)
    logger.info("Created backup", extra={"backup_path": backup_path})

    # Step 2: Read the original model
    with open(submodel_path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Step 3: Extract model structure components
    # Extract model name
    model_name_match = re.search(r"\bmodel\s+(\w+)", original_code)
    if not model_name_match:
        raise ValueError("Could not find model name in the file")
    model_name = model_name_match.group(1)

    # Extract within clause if exists
    within_match = re.search(r"^within\s+[^;]+;", original_code, re.MULTILINE)
    within_clause = within_match.group(0) + "\n" if within_match else ""

    # Extract all Input/Output port declarations
    # We need to extract port declarations more carefully to handle complex annotations
    ports = []

    # Split by lines and find port declarations
    lines = original_code.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line starts a port declaration
        if (
            "Modelica.Blocks.Interfaces.RealInput" in line
            or "Modelica.Blocks.Interfaces.RealOutput" in line
        ):
            # Collect the complete declaration (might span multiple lines)
            port_decl = line

            # If line doesn't end with semicolon, continue collecting
            while not port_decl.rstrip().endswith(";") and i + 1 < len(lines):
                i += 1
                port_decl += "\n" + lines[i]

            ports.append(port_decl)

        i += 1

    logger.info(
        "Extracted port declarations",
        extra={"num_ports": len(ports), "model_name": model_name},
    )

    # Step 4: Generate new model code
    new_code_parts = []

    # Add within clause
    if within_clause:
        new_code_parts.append(within_clause)

    # Start model definition
    new_code_parts.append(f"model {model_name}")
    new_code_parts.append("")

    # Add all original port declarations (preserve formatting)
    for port in ports:
        # Add proper indentation
        port_lines = port.split("\n")
        for port_line in port_lines:
            if port_line.strip():
                # Ensure proper indentation
                if not port_line.startswith("  "):
                    new_code_parts.append(f"  {port_line.strip()}")
                else:
                    new_code_parts.append(port_line.rstrip())
    new_code_parts.append("")

    # Add parameters section
    new_code_parts.append("protected")
    new_code_parts.append(
        f'  parameter String fileName = "{csv_file}" "Path to the CSV file";'
    )

    # Add CombiTimeTable for each output port
    for port_info in output_ports:
        port_name = port_info["name"]
        default_column = port_info["default_column"]
        table_name = f"table_{port_name}"

        # Convert Python list to Modelica array syntax: [1,2,3] -> {1,2,3}
        if isinstance(default_column, list):
            column_str = "{" + ", ".join(str(c) for c in default_column) + "}"
        else:
            column_str = str(default_column)

        new_code_parts.append("")
        new_code_parts.append(f"  // CSV data source for {port_name}")
        new_code_parts.append(f"  Modelica.Blocks.Sources.CombiTimeTable {table_name}(")
        new_code_parts.append(f'    tableName="csv_data_{port_name}",')
        new_code_parts.append("    fileName=fileName,")
        new_code_parts.append(f"    columns={column_str},")
        new_code_parts.append("    tableOnFile=true")
        new_code_parts.append("  );")

    # Also store columns as parameters for conditional logic
    for port_info in output_ports:
        port_name = port_info["name"]
        default_column = port_info["default_column"]

        # Convert to Modelica array syntax
        if isinstance(default_column, list):
            column_str = "{" + ", ".join(str(c) for c in default_column) + "}"
        else:
            column_str = str(default_column)

        new_code_parts.append(
            f"  parameter Integer columns_{port_name}[{port_info['dim'] + 1}] = {column_str};"
        )

    new_code_parts.append("")

    # Add equation section
    new_code_parts.append("equation")
    new_code_parts.append("  // Map CSV data to output ports")
    new_code_parts.append("  // If columns[i+1] = 1, output 0 instead of CSV data")

    for port_info in output_ports:
        port_name = port_info["name"]
        dim = port_info["dim"]
        table_name = f"table_{port_name}"

        if dim > 1:
            # Vector port with conditional logic
            new_code_parts.append(f"  // Vector port: {port_name}[{dim}]")
            new_code_parts.append(f"  for i in 1:{dim} loop")
            new_code_parts.append(
                f"    {port_name}[i] = if columns_{port_name}[i+1] == 1 then 0.0 else {table_name}.y[i];"
            )
            new_code_parts.append("  end for;")
        else:
            # Scalar port with conditional logic
            new_code_parts.append(
                f"  {port_name} = if columns_{port_name}[2] == 1 then 0.0 else {table_name}.y[1];"
            )

    new_code_parts.append("")

    # Extract and preserve annotation if exists
    annotation_match = re.search(
        r"annotation\s*\([^)]*(?:\([^)]*(?:\([^)]*\))*[^)]*\))*[^)]*\);",
        original_code,
        re.DOTALL,
    )
    if annotation_match:
        new_code_parts.append(f"  {annotation_match.group(0)}")
        new_code_parts.append("")

    # End model
    new_code_parts.append(f"end {model_name};")

    # Step 5: Write the modified model
    new_code = "\n".join(new_code_parts)

    with open(submodel_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    logger.info(
        "Submodel replaced successfully",
        extra={
            "modified_path": submodel_path,
            "num_output_ports": len(output_ports),
        },
    )

    return {
        "original_path": submodel_path,
        "backup_path": backup_path,
        "modified_path": submodel_path,
    }


def replace_submodels_with_csv(
    package_path: str,
    replacement_configs: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Replaces multiple submodels with CSV data sources.

    Args:
        package_path: Path to the Modelica package directory or package.mo file.
        replacement_configs: A list of dictionaries, each defining a replacement task:
            - submodel_name: Full name of the submodel (e.g., 'MyPackage.MyModel')
            - output_ports: List of output port definitions
            - csv_file: Path to the CSV file

    Returns:
        A dictionary containing:
            - replaced_models: List of results from each replacement
            - package_path: Original package path

    Note:
        Automatically determines if package is directory-based or single-file.
        Continues processing remaining models if one fails, but re-raises the error
        after logging. Each submodel file is identified by splitting the full name
        and locating {ModelName}.mo in the package directory.
    """
    logger.info(
        "Starting batch submodel replacement",
        extra={"num_configs": len(replacement_configs)},
    )

    # Determine package directory
    if os.path.isdir(package_path):
        package_dir = package_path
    elif os.path.isfile(package_path) and package_path.endswith("package.mo"):
        package_dir = os.path.dirname(package_path)
    elif os.path.isfile(package_path):
        package_dir = os.path.dirname(package_path)
    else:
        raise FileNotFoundError(f"Invalid package path: {package_path}")

    replaced_models = []

    for config in replacement_configs:
        submodel_name = config["submodel_name"]
        output_ports = config["output_ports"]
        csv_file = config["csv_file"]

        # Construct submodel file path
        model_simple_name = submodel_name.split(".")[-1]
        submodel_file = os.path.join(package_dir, f"{model_simple_name}.mo")

        if not os.path.exists(submodel_file):
            logger.warning(
                "Submodel file not found, skipping",
                extra={"submodel_name": submodel_name, "expected_path": submodel_file},
            )
            continue

        try:
            result = _replace_submodel_with_csv(
                submodel_path=submodel_file,
                output_ports=output_ports,
                csv_file=csv_file,
            )
            result["submodel_name"] = submodel_name
            replaced_models.append(result)

            logger.info(
                "Successfully replaced submodel",
                extra={"submodel_name": submodel_name},
            )

        except Exception as e:
            logger.error(
                "Failed to replace submodel",
                extra={"submodel_name": submodel_name, "error": str(e)},
            )
            raise

    logger.info(
        "Batch replacement completed",
        extra={"num_replaced": len(replaced_models)},
    )

    return {
        "replaced_models": replaced_models,
        "package_path": package_path,
    }


def _integrate_replacement(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Integrates direct replacement mode by modifying submodels internally.

    This function processes interception_configs and directly modifies the
    submodel files (or submodels within a single file) to replace their
    internal equations with CSV data sources.

    Args:
        package_path: Path to the Modelica package.
        model_name: Full name of the system model (used for context/logging).
        interception_configs: List of configurations from co-simulation.
            Each dict should contain 'submodel_name', 'csv_uri', 'instance_name',
            and 'output_placeholder' keys.

    Returns:
        A dictionary containing:
            - interceptor_model_paths: Empty list (for compatibility)
            - system_model_path: Original package path (no system model modification needed)
            - replaced_models: List of replaced model info

    Note:
        Automatically detects single-file vs multi-file package format based on
        the package_path. Routes to appropriate handler function. Unlike interceptor
        mode, no connection rewiring is performed as submodels are modified in-place.
    """
    logger.info(
        "Starting direct replacement integration",
        extra={
            "package_path": package_path,
            "model_name": model_name,
            "num_configs": len(interception_configs),
        },
    )

    # Check if it's a single-file package
    is_single_file = False
    if os.path.isfile(package_path) and not package_path.endswith("package.mo"):
        is_single_file = True
        logger.info("Detected single-file package format")

    if is_single_file:
        # Handle single-file package
        return _integrate_replacement_single_file(
            package_path, model_name, interception_configs
        )
    else:
        # Handle multi-file package
        return _integrate_replacement_multi_file(
            package_path, model_name, interception_configs
        )


def _integrate_replacement_multi_file(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handles direct replacement for multi-file packages.

    Each submodel is in a separate .mo file.

    Args:
        package_path: Path to the Modelica package directory or package.mo file.
        model_name: Full name of the system model (used for context/logging).
        interception_configs: List of configurations from co-simulation.
            Each dict should contain 'submodel_name', 'csv_uri', 'instance_name',
            and 'output_placeholder' keys.

    Returns:
        A dictionary containing:
            - interceptor_model_paths: Empty list (no interceptors created)
            - system_model_path: Original package path (unchanged)
            - replaced_models: List of replacement results for each submodel

    Note:
        Parses output_placeholder column specifications in format "{1,2,3,4,5,6}".
        Skips submodels if corresponding .mo file is not found in package directory.
        Creates .bak backup files before modification.
    """
    # Determine package directory
    if os.path.isdir(package_path):
        package_dir = package_path
    elif os.path.isfile(package_path) and package_path.endswith("package.mo"):
        package_dir = os.path.dirname(package_path)
    elif os.path.isfile(package_path):
        package_dir = os.path.dirname(package_path)
    else:
        raise FileNotFoundError(f"Invalid package path: {package_path}")

    replaced_models = []

    for config in interception_configs:
        submodel_name = config["submodel_name"]
        csv_uri = config["csv_uri"]
        output_placeholder = config["output_placeholder"]

        # Parse output ports from placeholder
        output_ports = []
        for port_name, column_spec in output_placeholder.items():
            # Parse column spec like "{1,2,3,4,5,6}"
            if isinstance(column_spec, str) and column_spec.startswith("{"):
                columns_str = column_spec.strip("{}").split(",")
                columns = [int(c) for c in columns_str]
                dim = len(columns) - 1  # First column is time
                output_ports.append(
                    {
                        "name": port_name,
                        "dim": dim,
                        "default_column": columns,
                    }
                )
            else:
                logger.warning(
                    "Unexpected column spec format",
                    extra={
                        "submodel_name": submodel_name,
                        "port_name": port_name,
                        "column_spec": column_spec,
                    },
                )
                continue

        # Construct submodel file path
        model_simple_name = submodel_name.split(".")[-1]
        submodel_file = os.path.join(package_dir, f"{model_simple_name}.mo")

        if not os.path.exists(submodel_file):
            logger.warning(
                "Submodel file not found for direct replacement",
                extra={
                    "submodel_name": submodel_name,
                    "expected_path": submodel_file,
                },
            )
            continue

        try:
            result = _replace_submodel_with_csv(
                submodel_path=submodel_file,
                output_ports=output_ports,
                csv_file=csv_uri,
            )
            result["submodel_name"] = submodel_name
            result["instance_name"] = config.get("instance_name", "")
            replaced_models.append(result)

            logger.info(
                "Successfully applied direct replacement",
                extra={
                    "submodel_name": submodel_name,
                    "csv_file": csv_uri,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to apply direct replacement",
                extra={
                    "submodel_name": submodel_name,
                    "error": str(e),
                },
            )
            raise

    logger.info(
        "Direct replacement integration completed",
        extra={"num_replaced": len(replaced_models)},
    )

    # For direct replacement, we don't modify the system model
    # Return the original package path as system_model_path
    return {
        "interceptor_model_paths": [],  # No interceptors created
        "system_model_path": package_path,  # Original package unchanged
        "replaced_models": replaced_models,
    }


def _integrate_replacement_single_file(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handles direct replacement for single-file packages.

    All submodels are embedded in one .mo file.

    Args:
        package_path: Path to the single Modelica package file.
        model_name: Full name of the system model (used for context/logging).
        interception_configs: List of configurations from co-simulation.
            Each dict should contain 'submodel_name', 'csv_uri', 'instance_name',
            and 'output_placeholder' keys.

    Returns:
        A dictionary containing:
            - interceptor_model_paths: Empty list
            - system_model_path: Original package path
            - replaced_models: List of replacement results for each submodel

    Note:
        Creates a single .bak backup of the entire package file. Modifies all
        specified submodels within the package using regex pattern matching.
        The entire modified package is written back to the original file.
    """
    logger.info(
        "Processing single-file package for direct replacement",
        extra={"package_path": package_path},
    )

    # Read the entire package file
    with open(package_path, "r", encoding="utf-8") as f:
        original_package_code = f.read()

    # Backup the original file
    backup_path = package_path.replace(".mo", ".bak")
    shutil.copy2(package_path, backup_path)
    logger.info("Created backup", extra={"backup_path": backup_path})

    modified_package_code = original_package_code
    replaced_models = []

    for config in interception_configs:
        submodel_name = config["submodel_name"]
        csv_uri = config["csv_uri"]
        output_placeholder = config["output_placeholder"]

        # Parse output ports from placeholder
        output_ports = []
        for port_name, column_spec in output_placeholder.items():
            if isinstance(column_spec, str) and column_spec.startswith("{"):
                columns_str = column_spec.strip("{}").split(",")
                columns = [int(c) for c in columns_str]
                dim = len(columns) - 1
                output_ports.append(
                    {
                        "name": port_name,
                        "dim": dim,
                        "default_column": columns,
                    }
                )

        # Extract the submodel from the package code
        model_short_name = submodel_name.split(".")[-1]

        # Find the model block in the package
        model_pattern = re.compile(
            r"(model\s+"
            + re.escape(model_short_name)
            + r".*?end\s+"
            + re.escape(model_short_name)
            + r"\s*;)",
            re.DOTALL,
        )
        match = model_pattern.search(modified_package_code)

        if not match:
            logger.warning(
                "Submodel not found in single-file package",
                extra={"submodel_name": submodel_name},
            )
            continue

        original_model_code = match.group(1)

        # Generate replacement model code
        try:
            replaced_model_code = _generate_replaced_model_code(
                original_model_code,
                model_short_name,
                output_ports,
                csv_uri,
            )

            # Replace the model in the package code
            modified_package_code = modified_package_code.replace(
                original_model_code,
                replaced_model_code,
            )

            replaced_models.append(
                {
                    "submodel_name": submodel_name,
                    "instance_name": config.get("instance_name", ""),
                    "original_path": package_path,
                    "backup_path": backup_path,
                    "modified_path": package_path,
                }
            )

            logger.info(
                "Successfully replaced submodel in single file",
                extra={"submodel_name": submodel_name},
            )

        except Exception as e:
            logger.error(
                "Failed to replace submodel in single file",
                extra={"submodel_name": submodel_name, "error": str(e)},
            )
            raise

    # Write the modified package back to the file
    with open(package_path, "w", encoding="utf-8") as f:
        f.write(modified_package_code)

    logger.info(
        "Single-file direct replacement completed",
        extra={"num_replaced": len(replaced_models)},
    )

    return {
        "interceptor_model_paths": [],
        "system_model_path": package_path,
        "replaced_models": replaced_models,
    }


def _generate_replaced_model_code(
    original_model_code: str,
    model_name: str,
    output_ports: list[Dict[str, Any]],
    csv_file: str,
) -> str:
    """Generates the replacement model code from original model code.

    This is similar to _replace_submodel_with_csv but works on code strings
    instead of files, suitable for single-file package processing.

    Args:
        original_model_code: The original Modelica model code as string.
        model_name: The name of the model being replaced.
        output_ports: A list of dictionaries describing output ports.
            Each dict should have 'name', 'dim', and 'default_column' keys.
        csv_file: Path to the CSV file containing replacement data.

    Returns:
        The generated Modelica model code as a string.

    Note:
        Extracts and preserves port declarations and annotations from original code.
        Generates CombiTimeTable instances for CSV data loading. Outputs 0.0 when
        column index is 1 (pass-through/disabled mode).
    """
    # Extract port declarations
    ports = []
    lines = original_model_code.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if (
            "Modelica.Blocks.Interfaces.RealInput" in line
            or "Modelica.Blocks.Interfaces.RealOutput" in line
        ):
            port_decl = line

            while not port_decl.rstrip().endswith(";") and i + 1 < len(lines):
                i += 1
                port_decl += "\n" + lines[i]

            ports.append(port_decl)

        i += 1

    # Generate new model code
    new_code_parts = []

    new_code_parts.append(f"model {model_name}")
    new_code_parts.append("")

    # Add all original port declarations
    for port in ports:
        port_lines = port.split("\n")
        for port_line in port_lines:
            if port_line.strip():
                if not port_line.startswith("  "):
                    new_code_parts.append(f"  {port_line.strip()}")
                else:
                    new_code_parts.append(port_line.rstrip())
    new_code_parts.append("")

    # Add protected section
    new_code_parts.append("protected")
    new_code_parts.append(
        f'  parameter String fileName = "{csv_file}" "Path to the CSV file";'
    )

    # Add CombiTimeTable for each output port
    for port_info in output_ports:
        port_name = port_info["name"]
        default_column = port_info["default_column"]
        table_name = f"table_{port_name}"

        # Convert to Modelica array syntax
        if isinstance(default_column, list):
            column_str = "{" + ", ".join(str(c) for c in default_column) + "}"
        else:
            column_str = str(default_column)

        new_code_parts.append("")
        new_code_parts.append(f"  // CSV data source for {port_name}")
        new_code_parts.append(f"  Modelica.Blocks.Sources.CombiTimeTable {table_name}(")
        new_code_parts.append(f'    tableName="csv_data_{port_name}",')
        new_code_parts.append("    fileName=fileName,")
        new_code_parts.append(f"    columns={column_str},")
        new_code_parts.append("    tableOnFile=true")
        new_code_parts.append("  );")

    # Store columns as parameters
    for port_info in output_ports:
        port_name = port_info["name"]
        default_column = port_info["default_column"]

        if isinstance(default_column, list):
            column_str = "{" + ", ".join(str(c) for c in default_column) + "}"
        else:
            column_str = str(default_column)

        new_code_parts.append(
            f"  parameter Integer columns_{port_name}[{port_info['dim'] + 1}] = {column_str};"
        )

    new_code_parts.append("")

    # Add equation section
    new_code_parts.append("equation")
    new_code_parts.append("  // Map CSV data to output ports")
    new_code_parts.append("  // If columns[i+1] = 1, output 0 instead of CSV data")

    for port_info in output_ports:
        port_name = port_info["name"]
        dim = port_info["dim"]
        table_name = f"table_{port_name}"

        if dim > 1:
            new_code_parts.append(f"  // Vector port: {port_name}[{dim}]")
            new_code_parts.append(f"  for i in 1:{dim} loop")
            new_code_parts.append(
                f"    {port_name}[i] = if columns_{port_name}[i+1] == 1 then 0.0 else {table_name}.y[i];"
            )
            new_code_parts.append("  end for;")
        else:
            new_code_parts.append(
                f"  {port_name} = if columns_{port_name}[2] == 1 then 0.0 else {table_name}.y[1];"
            )

    new_code_parts.append("")

    # Extract and preserve annotation if exists
    annotation_match = re.search(
        r"annotation\s*\([^)]*(?:\([^)]*(?:\([^)]*\))*[^)]*\))*[^)]*\);",
        original_model_code,
        re.DOTALL,
    )
    if annotation_match:
        new_code_parts.append(f"  {annotation_match.group(0)}")
        new_code_parts.append("")

    new_code_parts.append(f"end {model_name};")

    return "\n".join(new_code_parts)


def integrate_interceptor_model(
    package_path: str, model_name: str, interception_configs: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Integrates CSV data replacement into a system model.

    This function supports two modes (all handlers must use the same mode):
    1. "interceptor" (default): Creates interceptor models between submodels and system.
    2. "replacement": Directly modifies submodels to use CSV data.

    Args:
        package_path: The file path to the Modelica package. For multi-file packages,
            this should be the path to `package.mo`. For single-file packages,
            this should be the path to the `.mo` file containing the package.
        model_name: The full name of the system model to be modified.
        interception_configs: A list of dictionaries, each defining an interception task.
            All configs must have the same 'mode' field. Each dict should contain
            'submodel_name', 'csv_uri', 'instance_name', 'output_placeholder', and
            optionally 'mode' (defaults to 'interceptor').

    Returns:
        A dictionary containing the paths to modified models. Structure varies by mode:
            - interceptor mode: interceptor_model_paths, system_model_path
            - replacement mode: replaced_models, system_model_path

    Raises:
        ValueError: If interception_configs is empty or if mixed modes are detected.
        FileNotFoundError: If package_path is invalid or package.mo not found.

    Note:
        Mode is determined from the first config's 'mode' field. All configs must use
        the same mode or ValueError is raised. Automatically detects single-file vs
        multi-file package structure and routes to appropriate handler.
    """
    if not interception_configs:
        raise ValueError("interception_configs cannot be empty")

    # Get mode from first config (all should be the same)
    mode = interception_configs[0].get("mode", "interceptor")

    # Validate that all configs use the same mode
    for config in interception_configs:
        config_mode = config.get("mode", "interceptor")
        if config_mode != mode:
            raise ValueError(
                f"Mixed modes are not supported. All handlers must use the same mode. "
                f"Expected '{mode}', but found '{config_mode}' in config for '{config.get('submodel_name')}'"
            )

    logger.info(
        "Integrating CSV data replacement",
        extra={
            "mode": mode,
            "num_submodels": len(interception_configs),
        },
    )

    # Route to appropriate handler based on mode
    if mode == "replacement":
        return _integrate_replacement(package_path, model_name, interception_configs)
    else:  # mode == "interceptor"
        # Determine package type and route to appropriate handler
        if os.path.isdir(package_path):
            package_file = os.path.join(package_path, "package.mo")
            if os.path.exists(package_file):
                return _integrate_interceptor_multi_file(
                    package_file, model_name, interception_configs
                )
            else:
                raise FileNotFoundError(
                    f"No package.mo found in directory: {package_path}"
                )
        elif os.path.isfile(package_path) and package_path.endswith("package.mo"):
            return _integrate_interceptor_multi_file(
                package_path, model_name, interception_configs
            )
        elif os.path.isfile(package_path):
            return _integrate_interceptor_single_file(
                package_path, model_name, interception_configs
            )
        else:
            raise FileNotFoundError(f"Invalid package path: {package_path}")
