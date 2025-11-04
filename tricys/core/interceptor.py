import logging
import os
import re
from pathlib import Path

from OMPython import OMCSessionZMQ

logger = logging.getLogger(__name__)


def _generate_interceptor(
    submodel_name: str,
    output_ports: list,
    csv_file: str,
    add_within_clause: bool = True,
) -> tuple[str, str]:
    """
    Generates the Modelica code for an interceptor model.

    The interceptor wraps an existing model, allowing its output ports to be
    overridden by data from a CSV file.

    Args:
        submodel_name: The full name of the submodel to be intercepted (e.g., 'MyPackage.MyModel').
        output_ports: A list of dictionaries, where each dictionary describes an output port.
                      Each dictionary should have 'name', 'dim', and 'default_column' keys.
        csv_file: The path to the CSV file to be used for data injection.
        add_within_clause: Whether to add the 'within <package_name>;' clause.

    Returns:
        A tuple containing the interceptor model name and the generated Modelica code.
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
        f"Generating interceptor code for ports: '{[p['name'] for p in output_ports]}'"
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
    package_path: str, model_name: str, interception_configs: list
):
    omc = None
    logger.info(
        f"--- Starting model processing for single-file package, {len(interception_configs)} interception tasks in total ---"
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

            logger.info(f"Identifying output ports for '{submodel_name}'...")
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
            logger.info(f"Identified output ports: {[p['name'] for p in output_ports]}")

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
                    f"Could not find a connection for port '{port_name}' of instance '{instance_name_in_system}'."
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
    # 4a. Replace the original system model block with the modified one
    final_package_code = original_package_code.replace(
        original_system_code, final_system_code_block
    )

    # 4b. Inject the new interceptor models at the end of the package
    all_interceptors_code = "\n\n".join(generated_interceptor_codes)
    package_name = model_name.split(".")[0]
    final_package_code = final_package_code.replace(
        f"end {package_name};", f"{all_interceptors_code}\n\nend {package_name};"
    )

    # Part 5: Save the new single-file package
    original_filename = os.path.basename(package_path)
    modified_filename = original_filename.replace(".mo", "_intercepted.mo")
    modified_system_file_path = os.path.join(output_dir, modified_filename)

    with open(modified_system_file_path, "w", encoding="utf-8") as f:
        f.write(final_package_code)

    logger.info(
        f"\nGenerated modified single-file package: {modified_system_file_path}"
    )
    logger.info("\n--- Automated modification complete! ---")

    return {
        "interceptor_model_paths": [],
        "system_model_path": modified_system_file_path,
    }


def _integrate_interceptor_multi_file(
    package_path: str, model_name: str, interception_configs: list
):
    omc = None
    logger.info(
        f"--- Starting model processing, {len(interception_configs)} interception tasks in total ---"
    )
    try:
        omc = OMCSessionZMQ()
        omc.sendExpression(f'loadFile("{Path(package_path).as_posix()}")')

        logger.info("Proceeding with multi-interceptor model generation.")
        package_dir = os.path.dirname(package_path)
        model_short_name = model_name.split(".")[-1]
        system_model_path = os.path.join(package_dir, f"{model_short_name}.mo")

        if not os.path.exists(system_model_path):
            raise FileNotFoundError(
                f"Inferred system model path does not exist: {system_model_path}"
            )

        output_dir = os.path.dirname(system_model_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Proceeding with multi-interceptor model generation.")
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

            logger.info(f"Identifying output ports for '{submodel_name}'...")
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
            logger.info(f"Identified output ports: {[p['name'] for p in output_ports]}")

            package_name = submodel_name.split(".")[0]
            original_model_short_name = submodel_name.split(".")[-1]
            interceptor_name, interceptor_code = _generate_interceptor(
                submodel_name, output_ports, csv_uri
            )

            interceptor_file_path = os.path.join(output_dir, f"{interceptor_name}.mo")
            with open(interceptor_file_path, "w", encoding="utf-8") as f:
                f.write(interceptor_code)
            logger.info(f"Generated interceptor model file: {interceptor_file_path}")
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
                    f"Successfully rewired port '{port_name}' for instance '{instance_name_in_system}'."
                )
            else:
                logger.warning(
                    f"Could not find a connection for port '{port_name}' of instance '{instance_name_in_system}'."
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

    logger.info(f"\nGenerated modified main model file: {modified_system_file_path}")
    logger.info("\n--- Automated modification complete! ---")

    return {
        "interceptor_model_paths": generated_interceptor_files,
        "system_model_path": modified_system_file_path,
    }


def integrate_interceptor_model(
    package_path: str, model_name: str, interception_configs: list
):
    """
    Integrates one or more interceptor models into a system model.

    This function performs a multi-step process:
    1. Generates individual interceptor models based on the configuration.
    2. Modifies the main system model to instantiate these interceptors.
    3. Re-routes the connections of the original components through the interceptors.
    4. Saves the modified system model to a new file.

    Args:
        package_path: The file path to the Modelica package. For multi-file packages,
                      this should be the path to `package.mo`. For single-file packages,
                      this should be the path to the `.mo` file containing the package.
        model_name: The full name of the system model to be modified.
        interception_configs: A list of dictionaries, each defining an interception task.
    """
    # For multi-file packages, package_path is the path to the `package.mo` file.
    # The directory of this file is the package directory.
    # For single-file packages, package_path is the path to the file itself.
    if os.path.isdir(package_path):
        # Handle cases where a directory is passed, find package.mo
        package_file = os.path.join(package_path, "package.mo")
        if os.path.exists(package_file):
            return _integrate_interceptor_multi_file(
                package_file, model_name, interception_configs
            )
        else:
            raise FileNotFoundError(
                f"No package.mo found in the directory: {package_path}"
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
        raise FileNotFoundError(f"Invalid package path provided: {package_path}")
