"""Utilities for interacting with OpenModelica via OMPython.

This module provides a set of functions to manage an OpenModelica session,
load models, retrieve parameter details, and format parameter values for
simulation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from OMPython import OMCSessionZMQ

logger = logging.getLogger(__name__)


def get_om_session() -> OMCSessionZMQ:
    """Initializes and returns a new OMCSessionZMQ session.

    Returns:
        OMCSessionZMQ: An active OpenModelica session object.
    """
    logger.debug("Initializing new OMCSessionZMQ session.")
    return OMCSessionZMQ()


def load_modelica_package(omc: OMCSessionZMQ, package_path: str) -> bool:
    """Loads a Modelica package into the OpenModelica session.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        package_path (str): The file path to the Modelica package (`package.mo`).

    Returns:
        bool: True if the package was loaded successfully, False otherwise.
    """
    logger.info(f"Loading package: {package_path}")
    load_result = omc.sendExpression(f'loadFile("{package_path}")')
    if not load_result:
        logger.error(f"Failed to load package: {package_path}")
        return False
    return True


def get_model_parameter_names(omc: OMCSessionZMQ, model_name: str) -> List[str]:
    """Parses and returns all subcomponent parameter names for a given model.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        model_name (str): The full name of the model (e.g., 'example.Cycle').

    Returns:
        List[str]: A list of all available parameter names (e.g., ['blanket.TBR']).
    """
    logger.info(f"Getting parameter names for model '{model_name}'")
    all_params = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.warning(f"Model '{model_name}' not found in package.")
            return []

        components = omc.sendExpression(f"getComponents({model_name})")
        if not components:
            logger.warning(f"No components found for {model_name}")
            return []

        for comp in components:
            comp_type, comp_name = comp[0], comp[1]
            if comp_type.startswith(model_name.split(".")[0]):
                params = omc.sendExpression(f"getParameterNames({comp_type})")
                for param in params:
                    full_param = f"{comp_name}.{param}"
                    if full_param not in all_params:
                        all_params.append(full_param)

        logger.info(f"Found {len(all_params)} parameter names.")
        return all_params

    except Exception as e:
        logger.error(f"Failed to get parameter names: {e}", exc_info=True)
        return []


def _recursive_get_parameters(
    omc: OMCSessionZMQ, class_name: str, path_prefix: str, params_list: list
):
    """A private helper function to recursively traverse a model and collect parameters.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        class_name (str): The name of the class/model to inspect.
        path_prefix (str): The hierarchical path prefix for the current component.
        params_list (list): The list to which parameter details are appended.
    """
    logger.debug(f"Recursively exploring: {class_name} with prefix: '{path_prefix}'")
    components = omc.sendExpression(f"getComponents({class_name})")
    if not components:
        return

    for comp in components:
        comp_type, comp_name, comp_comment = comp[0], comp[1], comp[2]
        comp_variability = comp[8]
        comp_dimensions = str(comp[11])  # Extract dimensions

        full_name = f"{path_prefix}.{comp_name}" if path_prefix else comp_name

        if comp_variability == "parameter":
            logger.debug(f"Found parameter: {full_name} of type {comp_type}")
            param_value = omc.sendExpression(
                f'getParameterValue(stringTypeName("{class_name}"), "{comp_name}")'
            )
            params_list.append(
                {
                    "name": full_name,
                    "type": comp_type,
                    "defaultValue": param_value,
                    "comment": comp_comment,
                    "dimensions": comp_dimensions,  # Add dimensions to the dictionary
                }
            )
        elif comp_variability != "parameter" and omc.sendExpression(
            f"isModel({comp_type})"
        ):
            if comp_type.startswith(class_name.split(".")[0]):
                _recursive_get_parameters(omc, comp_type, full_name, params_list)
            else:
                logger.debug(
                    f"Skipping non-example component: {full_name} ({comp_type})"
                )


def get_all_parameters_details(
    omc: OMCSessionZMQ, model_name: str
) -> List[Dict[str, Any]]:
    """Recursively retrieves detailed information for all parameters in a given model.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        model_name (str): The full name of the model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains the detailed information of a single parameter.
    """
    logger.info(f"Getting detailed parameters for model '{model_name}' via recursion.")
    all_params_details = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.error(f"Model '{model_name}' not found in package.")
            return []
        _recursive_get_parameters(omc, model_name, "", all_params_details)
        logger.info(
            f"Successfully found details for {len(all_params_details)} parameters."
        )
        return all_params_details
    except Exception as e:
        logger.error(
            f"Failed to get detailed parameters via recursion: {e}", exc_info=True
        )
        return []


def format_parameter_value(name: str, value: Any) -> str:
    """Formats a parameter value into a string recognized by OpenModelica.

    Args:
        name (str): The name of the parameter.
        value (Any): The value of the parameter.

    Returns:
        str: A formatted string for use in simulation overrides (e.g., "p=1.0").
    """
    if isinstance(value, list):
        # Format lists as {v1,v2,...}
        return f"{name}={{{','.join(map(str, value))}}}"
    elif isinstance(value, str):
        # Format strings as "value"
        return f'{name}="{value}"'
    # For numbers and booleans, direct string conversion is fine
    return f"{name}={value}"


def _generate_interceptor(
    submodel_name: str, output_ports: list, csv_file: str
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
    model_template = f"""
within {package_name};

model {interceptor_name}
{inputs_code}
{outputs_code}
protected
{parameters_code}
{protected_declarations_code}
{equation_code}
annotation(
  Icon(graphics = {{
    Rectangle(fillColor = {{255, 255, 180}}, extent = {{{{-100, 100}}, {{100, -100}}}}),
    Text(extent = {{{{-80, 40}}, {{80, -40}}}}, textString = "{original_model_name}\\nInterceptor")
  }}
));
end {interceptor_name};
"""
    return interceptor_name, model_template.strip()


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
        package_path: The file path to the Modelica package (`package.mo`).
        model_name: The full name of the system model to be modified.
        interception_configs: A list of dictionaries, each defining an interception task.
    """
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
