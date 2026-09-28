import logging
import re
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_FOC_STRATEGIES = {"table", "array"}
SUPPORTED_TIME_UNITS = {
    "second": 1.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "year": 31536000.0,
}


class FOCParseError(ValueError):
    """Raised when a .foc script violates the supported DSL."""


def _build_output_declarations(output_names):
    return "\n".join(
        f"  Modelica.Blocks.Interfaces.RealOutput {output_name};"
        for output_name in output_names
    )


def _build_output_equations(output_names, source_name):
    return "\n".join(
        f"  {output_name} = {source_name};" for output_name in output_names
    )


def _build_table_pulse_block(output_names):
    outputs = _build_output_declarations(output_names)
    equations = _build_output_equations(output_names, "source_value")
    return f"""block FOC_TablePulse
    "External FOC table driven pulse source"
    parameter Real conversion_factor = 6.3935e-3 "Tritium consumption rate for 1MW";
    parameter Real amplitude_scale = 1.0 "Global pulse amplitude / power scale factor";
    parameter String fileName = "foc_table.txt" "Python generated table file path";

{outputs}

protected
    Modelica.Blocks.Sources.CombiTimeTable table(
        tableOnFile = true,
        tableName = "FOC_Data",
        fileName = fileName,
        extrapolation = Modelica.Blocks.Types.Extrapolation.HoldLastPoint,
        smoothness = Modelica.Blocks.Types.Smoothness.LinearSegments
    );
    Real source_value;

equation
    source_value = table.y[1] * conversion_factor * amplitude_scale;
{equations}
end FOC_TablePulse;
"""


def _build_array_pulse_block(output_names):
    outputs = _build_output_declarations(output_names)
    equations = _build_output_equations(output_names, "source_value")
    return f"""block FOC_ArrayPulse
    "Array driven discrete pulse source"
    parameter Real conversion_factor = 6.3935e-3 "Tritium consumption rate for 1MW";
    parameter Real amplitude_scale = 1.0 "Global pulse amplitude / power scale factor";
    parameter Real amplitudes[:] = {{0.0}};
    parameter Real durations[:] = {{1.0}};

{outputs}

protected
    parameter Integer n_steps = size(amplitudes, 1);
    parameter Real cumulative_times[n_steps + 1] = cat(1, {{0.0}}, {{sum(durations[1:k]) for k in 1:n_steps}});
    Real source_value;

equation
    source_value = conversion_factor * amplitude_scale * sum({{if time >= cumulative_times[i] and time < cumulative_times[i + 1] then amplitudes[i] else 0.0 for i in 1:n_steps}});
{equations}
end FOC_ArrayPulse;
"""


def _build_foc_block_definition(strategy, output_names):
    if strategy == "table":
        return _build_table_pulse_block(output_names)
    return _build_array_pulse_block(output_names)


def _normalize_line(raw_line):
    return raw_line.split("#", 1)[0].strip()


def _require_arg_count(parts, expected, line_no, cmd):
    if len(parts) != expected:
        raise FOCParseError(
            f"Line {line_no}: {cmd} expects {expected - 1} argument(s), got {len(parts) - 1}."
        )


def _parse_non_negative_float(raw_value, line_no, field_name):
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise FOCParseError(
            f"Line {line_no}: {field_name} must be a number, got {raw_value!r}."
        ) from exc
    if value < 0:
        raise FOCParseError(
            f"Line {line_no}: {field_name} must be non-negative, got {value}."
        )
    return value


def _parse_positive_float(raw_value, line_no, field_name):
    value = _parse_non_negative_float(raw_value, line_no, field_name)
    if value <= 0:
        raise FOCParseError(
            f"Line {line_no}: {field_name} must be greater than zero, got {value}."
        )
    return value


def _parse_positive_integer(raw_value, line_no, field_name):
    try:
        numeric_value = float(raw_value)
    except ValueError as exc:
        raise FOCParseError(
            f"Line {line_no}: {field_name} must be an integer, got {raw_value!r}."
        ) from exc

    if not numeric_value.is_integer() or numeric_value <= 0:
        raise FOCParseError(
            f"Line {line_no}: {field_name} must be a positive integer, got {raw_value!r}."
        )
    return int(numeric_value)


def _parse_time_unit(raw_value, line_no):
    normalized_value = raw_value.strip().lower()
    if normalized_value not in SUPPORTED_TIME_UNITS:
        supported_units = ", ".join(SUPPORTED_TIME_UNITS)
        raise FOCParseError(
            f"Line {line_no}: unsupported TIME_UNIT {raw_value!r}. Supported values are {supported_units}."
        )
    return normalized_value


def _build_time_conversion_factor(source_unit, target_unit):
    return SUPPORTED_TIME_UNITS[source_unit] / SUPPORTED_TIME_UNITS[target_unit]


def _parse_named_time_conversion(raw_value, line_no):
    match = re.fullmatch(
        r"([A-Za-z]+)_to_([A-Za-z]+)", raw_value.strip(), flags=re.IGNORECASE
    )
    if not match:
        return None

    source_unit = _parse_time_unit(match.group(1), line_no)
    target_unit = _parse_time_unit(match.group(2), line_no)
    return (
        _build_time_conversion_factor(source_unit, target_unit),
        source_unit,
        target_unit,
    )


def _parse_time_conversion(raw_value, line_no):
    normalized_value = raw_value.strip().upper()
    if normalized_value in {"NONE", "OFF", "IDENTITY"}:
        return 1.0, None, None

    named_conversion = _parse_named_time_conversion(raw_value, line_no)
    if named_conversion is not None:
        return named_conversion

    return (
        _parse_positive_float(raw_value, line_no, "time conversion factor"),
        None,
        None,
    )


def _validate_step_series(amplitudes, durations):
    if len(amplitudes) != len(durations):
        raise ValueError("amplitudes and durations must have the same length")
    if not amplitudes:
        raise ValueError("FOC compilation produced no steps")
    if any(duration <= 0 for duration in durations):
        raise ValueError("all durations must be greater than zero")


def _build_time_power_rows(amplitudes, durations):
    _validate_step_series(amplitudes, durations)

    rows = [(0.0, float(amplitudes[0]))]
    current_time = 0.0

    for index, duration in enumerate(durations):
        current_time += float(duration)
        current_amplitude = float(amplitudes[index])
        rows.append((current_time, current_amplitude))

        if index < len(amplitudes) - 1:
            next_amplitude = float(amplitudes[index + 1])
            if next_amplitude != current_amplitude:
                rows.append((current_time, next_amplitude))
        elif current_amplitude != 0.0:
            rows.append((current_time, 0.0))

    return rows


def _parse_foc_lines(lines):
    amplitudes, durations = [], []
    current_power = None
    in_schedule = False
    schedule_amps, schedule_durs = [], []
    completed_schedule = None
    foc_time_unit = "second"
    time_unit_explicit = False
    time_conversion_factor = 1.0
    time_conversion_explicit = False
    time_conversion_source_unit = None
    saw_non_header_command = False

    def append_step(amplitude, duration):
        target_amps = schedule_amps if in_schedule else amplitudes
        target_durs = schedule_durs if in_schedule else durations
        target_amps.append(amplitude)
        target_durs.append(duration * time_conversion_factor)

    for line_no, raw_line in enumerate(lines, start=1):
        line = _normalize_line(raw_line)
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].upper()

        if cmd == "TIME_UNIT":
            _require_arg_count(parts, 2, line_no, cmd)
            if saw_non_header_command:
                raise FOCParseError(
                    f"Line {line_no}: TIME_UNIT must appear before all other FOC commands."
                )
            if time_unit_explicit:
                raise FOCParseError(
                    f"Line {line_no}: TIME_UNIT may only be declared once."
                )

            foc_time_unit = _parse_time_unit(parts[1], line_no)
            time_unit_explicit = True

            if (
                time_conversion_source_unit
                and time_conversion_source_unit != foc_time_unit
            ):
                raise FOCParseError(
                    f"Line {line_no}: TIME_UNIT {foc_time_unit!r} does not match TIME_CONVERSION source unit {time_conversion_source_unit!r}."
                )
            continue

        if cmd == "TIME_CONVERSION":
            _require_arg_count(parts, 2, line_no, cmd)
            if saw_non_header_command:
                raise FOCParseError(
                    f"Line {line_no}: TIME_CONVERSION must appear before all other FOC commands."
                )
            if time_conversion_explicit:
                raise FOCParseError(
                    f"Line {line_no}: TIME_CONVERSION may only be declared once."
                )
            (
                time_conversion_factor,
                time_conversion_source_unit,
                _,
            ) = _parse_time_conversion(parts[1], line_no)
            if (
                time_unit_explicit
                and time_conversion_source_unit
                and time_conversion_source_unit != foc_time_unit
            ):
                raise FOCParseError(
                    f"Line {line_no}: TIME_CONVERSION source unit {time_conversion_source_unit!r} does not match TIME_UNIT {foc_time_unit!r}."
                )
            time_conversion_explicit = True
            continue

        saw_non_header_command = True

        if cmd == "POWER":
            _require_arg_count(parts, 2, line_no, cmd)
            current_power = _parse_non_negative_float(parts[1], line_no, "power")
        elif cmd == "BURN":
            _require_arg_count(parts, 2, line_no, cmd)
            if current_power is None:
                raise FOCParseError(
                    f"Line {line_no}: BURN requires a POWER to be set first."
                )
            append_step(
                current_power, _parse_positive_float(parts[1], line_no, "burn time")
            )
        elif cmd == "DWELL":
            _require_arg_count(parts, 2, line_no, cmd)
            append_step(0.0, _parse_positive_float(parts[1], line_no, "dwell time"))
        elif cmd == "PULSE":
            _require_arg_count(parts, 5, line_no, cmd)
            pulse_power = _parse_non_negative_float(parts[1], line_no, "pulse power")
            pulse_burn = _parse_positive_float(parts[2], line_no, "pulse burn time")
            pulse_dwell = _parse_non_negative_float(
                parts[3], line_no, "pulse dwell time"
            )
            pulse_cycles = _parse_positive_integer(parts[4], line_no, "pulse cycles")

            for _ in range(pulse_cycles):
                append_step(pulse_power, pulse_burn)
                if pulse_dwell > 0:
                    append_step(0.0, pulse_dwell)
        elif cmd == "BEGIN_SCHEDULE":
            _require_arg_count(parts, 1, line_no, cmd)
            if in_schedule:
                raise FOCParseError(
                    f"Line {line_no}: nested schedules are not supported."
                )
            in_schedule = True
            schedule_amps.clear()
            schedule_durs.clear()
            completed_schedule = None
        elif cmd == "END_SCHEDULE":
            _require_arg_count(parts, 1, line_no, cmd)
            if not in_schedule:
                raise FOCParseError(
                    f"Line {line_no}: END_SCHEDULE without BEGIN_SCHEDULE."
                )
            if not schedule_amps:
                raise FOCParseError(f"Line {line_no}: schedule block cannot be empty.")
            in_schedule = False
            completed_schedule = (schedule_amps.copy(), schedule_durs.copy())
        elif cmd == "REPEAT":
            _require_arg_count(parts, 2, line_no, cmd)
            if in_schedule:
                raise FOCParseError(
                    f"Line {line_no}: REPEAT must appear after END_SCHEDULE."
                )
            if completed_schedule is None:
                raise FOCParseError(
                    f"Line {line_no}: REPEAT requires a completed schedule block."
                )
            repeats = _parse_positive_integer(parts[1], line_no, "repeat count")
            amplitudes.extend(completed_schedule[0] * repeats)
            durations.extend(completed_schedule[1] * repeats)
            completed_schedule = None
        else:
            raise FOCParseError(f"Line {line_no}: unsupported command {cmd!r}.")

    if in_schedule:
        raise FOCParseError("Schedule block was not closed with END_SCHEDULE.")

    _validate_step_series(amplitudes, durations)
    return amplitudes, durations


def parse_foc_content(content):
    return _parse_foc_lines(content.splitlines())


def build_foc_preview(content):
    amplitudes, durations = parse_foc_content(content)
    rows = _build_time_power_rows(amplitudes, durations)
    return {
        "amplitudes": amplitudes,
        "durations": durations,
        "rows": rows,
        "schedule_duration": float(sum(durations)),
        "step_count": len(amplitudes),
    }


def _insert_block_definition(content, block_definition, block_name):
    if f"block {block_name}" in content:
        return content

    within_match = re.match(r"(\s*within\s+[^;]+;\s*\n)", content)
    if within_match:
        return (
            content[: within_match.end()]
            + "\n"
            + block_definition
            + "\n"
            + content[within_match.end() :]
        )

    package_match = re.match(r"(\s*package\s+\w+[^\n]*\n)", content)
    if package_match:
        return (
            content[: package_match.end()]
            + "\n"
            + block_definition
            + "\n"
            + content[package_match.end() :]
        )

    return block_definition + "\n" + content


def _write_package_helper_block(
    package_dir, package_name, block_name, block_definition
):
    helper_path = Path(package_dir) / f"{block_name}.mo"
    helper_content = f"within {package_name};\n" + block_definition + "\n"
    helper_path.write_text(helper_content, encoding="utf-8")
    return helper_path


def _update_package_order(package_dir, block_name, before_name=None):
    package_order_path = Path(package_dir) / "package.order"
    if not package_order_path.exists():
        return

    lines = package_order_path.read_text(encoding="utf-8").splitlines()
    stripped_lines = [line.strip() for line in lines]
    if block_name in stripped_lines:
        return

    insert_index = len(lines)
    if before_name and before_name in stripped_lines:
        insert_index = stripped_lines.index(before_name)

    lines.insert(insert_index, block_name)
    package_order_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_modelica_array(values):
    return "{" + ", ".join(str(float(value)) for value in values) + "}"


def _validate_strategy(strategy):
    if strategy not in SUPPORTED_FOC_STRATEGIES:
        raise ValueError(
            f"Unsupported FOC strategy: {strategy!r}. Expected one of {sorted(SUPPORTED_FOC_STRATEGIES)}"
        )


def _build_model_block_pattern(model_name):
    model_leaf_name = model_name.split(".")[-1]
    return rf"((?:model|block)\s+{re.escape(model_leaf_name)}\b.*?end\s+{re.escape(model_leaf_name)}\s*;)"


def _extract_model_block(content, model_name):
    match = re.search(_build_model_block_pattern(model_name), content, flags=re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Failed to locate model block {model_name!r} in the Modelica source"
        )
    return match.group(1)


def _is_pulse_like_type(type_name):
    type_leaf = type_name.split(".")[-1]
    return type_name in {"FOC_TablePulse", "FOC_ArrayPulse"} or type_leaf.endswith(
        "Pulse"
    )


def _build_component_pattern():
    return r"((?:\w+\.)*\w+)" r"\s+(\w+)(?:\s*\([^)]*\))?\s*annotation\s*\(([^;]+)\);"


def _parse_component_selector(target_component, model_name):
    if not target_component:
        return {"kind": "default", "raw": None}

    selector = target_component.strip()
    normalized_model_name = model_name.strip()
    selector_prefix = selector.split(":", 1)[0].lower()

    if selector_prefix == "path" and ":" in selector:
        path_value = selector.split(":", 1)[1].strip()
        path_parts = [part for part in path_value.split(".") if part]
        if len(path_parts) < 2:
            raise ValueError(
                "foc.foc_component path selectors must include a component name, for example 'path:Cycle.pulse'."
            )
        component_name = path_parts[-1]
        model_path = ".".join(path_parts[:-1])
        if not (
            normalized_model_name == model_path
            or normalized_model_name.endswith(f".{model_path}")
        ):
            raise ValueError(
                f"foc.foc_component path {path_value!r} does not match target model {normalized_model_name!r}."
            )
        return {
            "kind": "path",
            "raw": selector,
            "component_name": component_name,
            "model_path": model_path,
        }

    if selector_prefix == "type" and ":" in selector:
        type_value = selector.split(":", 1)[1].strip()
        if not type_value:
            raise ValueError(
                "foc.foc_component type selectors must include a type name, for example 'type:Pulse'."
            )
        return {
            "kind": "type",
            "raw": selector,
            "type_name": type_value,
            "type_leaf": type_value.split(".")[-1],
        }

    if "." in selector:
        selector_parts = [part for part in selector.split(".") if part]
        if selector_parts[-1][:1].isupper():
            return {
                "kind": "type",
                "raw": selector,
                "type_name": selector,
                "type_leaf": selector_parts[-1],
            }
        component_name = selector_parts[-1]
        model_path = ".".join(selector_parts[:-1])
        if not (
            normalized_model_name == model_path
            or normalized_model_name.endswith(f".{model_path}")
        ):
            raise ValueError(
                f"foc.foc_component path {selector!r} does not match target model {normalized_model_name!r}."
            )
        return {
            "kind": "path",
            "raw": selector,
            "component_name": component_name,
            "model_path": model_path,
        }

    if selector[:1].isupper():
        return {
            "kind": "type",
            "raw": selector,
            "type_name": selector,
            "type_leaf": selector,
        }

    return {"kind": "instance", "raw": selector, "component_name": selector}


def _component_matches_selector(type_name, component_name, selector):
    if selector["kind"] == "default":
        return type_name == "Modelica.Blocks.Sources.Pulse" or type_name in {
            "FOC_TablePulse",
            "FOC_ArrayPulse",
        }

    if selector["kind"] in {"instance", "path"}:
        return component_name == selector["component_name"] and _is_pulse_like_type(
            type_name
        )

    if selector["kind"] == "type":
        type_leaf = type_name.split(".")[-1]
        return type_name == selector["type_name"] or type_leaf == selector["type_leaf"]

    return False


def _collect_target_component_names(block_content, selector):
    pattern = _build_component_pattern()
    component_names = []

    for match in re.finditer(pattern, block_content):
        type_name = match.group(1)
        component_name = match.group(2)
        if _component_matches_selector(type_name, component_name, selector):
            component_names.append(component_name)

    return component_names


def _extract_output_names_from_connects(block_content, component_names):
    connect_pattern = re.compile(
        r"connect\s*\(\s*(\w+)\.(\w+)\s*,\s*(\w+)\.(\w+)\s*\)",
        flags=re.DOTALL,
    )
    ordered_outputs = []
    seen_outputs = set()
    target_names = set(component_names)

    for match in connect_pattern.finditer(block_content):
        left_component, left_port, right_component, right_port = match.groups()
        if left_component in target_names and left_port not in seen_outputs:
            ordered_outputs.append(left_port)
            seen_outputs.add(left_port)
        if right_component in target_names and right_port not in seen_outputs:
            ordered_outputs.append(right_port)
            seen_outputs.add(right_port)

    return ordered_outputs


def _resolve_foc_output_names(content, model_name, target_component):
    block_content = _extract_model_block(content, model_name)
    selector = _parse_component_selector(target_component, model_name)
    component_names = _collect_target_component_names(block_content, selector)

    if not component_names:
        if target_component:
            raise RuntimeError(
                f"Failed to find FOC component {target_component!r} in model {model_name!r}"
            )
        raise RuntimeError(f"Failed to find a Pulse component in model {model_name!r}")

    output_names = _extract_output_names_from_connects(block_content, component_names)
    return output_names or ["y"]


def validate_foc_component_replacement(package_path, model_name, foc_component):
    if not isinstance(foc_component, str) or not foc_component.strip():
        raise ValueError("foc.foc_component is required when FOC is enabled.")

    modelica_target = _resolve_modelica_target_file(package_path, model_name)
    content = Path(modelica_target).read_text(encoding="utf-8")
    block_content = _extract_model_block(content, model_name)
    selector = _parse_component_selector(foc_component, model_name)
    component_names = _collect_target_component_names(block_content, selector)

    if not component_names:
        raise ValueError(
            f"foc.foc_component {foc_component!r} does not match a replaceable pulse-like component in model {model_name!r}."
        )

    return {
        "modelica_target": str(modelica_target),
        "component_names": component_names,
        "output_names": _extract_output_names_from_connects(
            block_content, component_names
        )
        or ["y"],
    }


def _replace_component_declarations(
    block_content, pattern, selector, replacement_factory
):
    replacement_count = 0

    def replacer(match):
        nonlocal replacement_count
        type_name = match.group(1)
        component_name = match.group(2)
        annotation = match.group(3)

        if not _component_matches_selector(type_name, component_name, selector):
            return match.group(0)

        replacement_count += 1
        return replacement_factory(component_name, annotation)

    updated_block = re.sub(pattern, replacer, block_content)
    return updated_block, replacement_count


def _replace_in_model_block(content, model_name, block_replacer):
    def replacer(match):
        return block_replacer(match.group(1))

    updated_content, replacements = re.subn(
        _build_model_block_pattern(model_name),
        replacer,
        content,
        count=1,
        flags=re.DOTALL,
    )
    if replacements == 0:
        raise RuntimeError(
            f"Failed to locate model block {model_name!r} in the Modelica source"
        )
    return updated_content


def _update_modelica_pulse_source(
    modelica_path,
    model_name,
    strategy,
    table_filename=None,
    amplitudes=None,
    durations=None,
    inline_block_definition=True,
    target_component=None,
):
    content = Path(modelica_path).read_text(encoding="utf-8")
    _validate_strategy(strategy)
    output_names = _resolve_foc_output_names(content, model_name, target_component)
    block_name = "FOC_TablePulse" if strategy == "table" else "FOC_ArrayPulse"
    block_definition = _build_foc_block_definition(strategy, output_names)

    if inline_block_definition:
        content = _insert_block_definition(content, block_definition, block_name)

    selector = _parse_component_selector(target_component, model_name)
    pattern = _build_component_pattern()

    def replacement_factory(var_name, annotation):
        if strategy == "table":
            normalized_filename = table_filename.replace("\\", "/")
            return (
                f'FOC_TablePulse {var_name}(fileName="{normalized_filename}") '
                f"annotation({annotation});"
            )

        amplitudes_literal = _format_modelica_array(amplitudes)
        durations_literal = _format_modelica_array(durations)
        return (
            f"FOC_ArrayPulse {var_name}(amplitudes={amplitudes_literal}, "
            f"durations={durations_literal}) annotation({annotation});"
        )

    replacements = 0

    def block_replacer(block_content):
        nonlocal replacements
        updated_block, replacements = _replace_component_declarations(
            block_content,
            pattern,
            selector,
            replacement_factory,
        )
        return updated_block

    updated_content = _replace_in_model_block(content, model_name, block_replacer)
    if replacements == 0:
        if target_component:
            raise RuntimeError(
                f"Failed to find FOC component {target_component!r} in model {model_name!r} within {modelica_path}"
            )
        raise RuntimeError(
            f"Failed to find a Pulse component in model {model_name!r} within {modelica_path}"
        )

    Path(modelica_path).write_text(updated_content, encoding="utf-8")
    return replacements


def _resolve_modelica_target_file(package_path, model_name):
    package_path = Path(package_path)
    model_leaf_name = model_name.split(".")[-1]

    if package_path.is_file() and package_path.name != "package.mo":
        return package_path

    package_dir = package_path if package_path.is_dir() else package_path.parent
    candidate_file = package_dir / f"{model_leaf_name}.mo"
    if candidate_file.exists():
        return candidate_file

    if package_path.is_file():
        return package_path

    package_file = package_dir / "package.mo"
    if package_file.exists():
        return package_file

    raise FileNotFoundError(f"Could not locate Modelica source for model {model_name}")


def parse_foc_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file_handle:
        return _parse_foc_lines(file_handle.readlines())


def export_for_combitimetable(amplitudes, durations, filename="foc_table.txt"):
    rows = _build_time_power_rows(amplitudes, durations)
    df = pd.DataFrame(rows, columns=["time", "power"])
    table_path = Path(filename)

    with open(table_path, "w", encoding="utf-8", newline="") as file_handle:
        file_handle.write(f"#1\nfloat FOC_Data({len(df)}, 2)\n")
        df.to_csv(file_handle, sep="\t", index=False, header=False)

    csv_path = table_path.with_suffix(".csv")
    df.to_csv(csv_path, sep=",", index=False, header=["Time(s)", "Power(MW)"])
    return csv_path


def prepare_foc_simulation_package(
    package_path,
    model_name,
    foc_path,
    workspace_dir,
    strategy="table",
    foc_component=None,
):
    package_path = Path(package_path).resolve()
    foc_path = Path(foc_path).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    _validate_strategy(strategy)

    if not foc_path.exists():
        raise FileNotFoundError(f"FOC file not found: {foc_path}")
    if not package_path.exists():
        raise FileNotFoundError(f"Modelica package not found: {package_path}")
    if not isinstance(foc_component, str) or not foc_component.strip():
        raise ValueError("foc.foc_component is required when FOC is enabled.")

    workspace_dir.mkdir(parents=True, exist_ok=True)

    amplitudes, durations = parse_foc_file(foc_path)

    if package_path.is_file() and package_path.name != "package.mo":
        generated_package_path = workspace_dir / package_path.name
        shutil.copy2(package_path, generated_package_path)
    else:
        source_dir = package_path if package_path.is_dir() else package_path.parent
        generated_dir = workspace_dir / source_dir.name
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        shutil.copytree(source_dir, generated_dir)
        generated_package_path = (
            generated_dir / package_path.name
            if package_path.is_file()
            else generated_dir
        )

    modelica_target = _resolve_modelica_target_file(generated_package_path, model_name)
    validate_foc_component_replacement(modelica_target, model_name, foc_component)
    package_entry_path = (
        generated_package_path / "package.mo"
        if Path(generated_package_path).is_dir()
        else Path(generated_package_path)
    )
    package_mode = package_entry_path.name == "package.mo"
    inline_block_definition = True

    if package_mode:
        package_dir = package_entry_path.parent
        package_name = model_name.split(".")[0]
        model_leaf_name = model_name.split(".")[-1]
        block_name = "FOC_TablePulse" if strategy == "table" else "FOC_ArrayPulse"
        source_content = Path(modelica_target).read_text(encoding="utf-8")
        output_names = _resolve_foc_output_names(
            source_content, model_name, foc_component
        )
        block_definition = _build_foc_block_definition(strategy, output_names)
        _write_package_helper_block(
            package_dir, package_name, block_name, block_definition
        )
        _update_package_order(package_dir, block_name, before_name=model_leaf_name)
        inline_block_definition = False

    table_path = None
    if strategy == "table":
        table_path = workspace_dir / "foc_table.txt"
        export_for_combitimetable(amplitudes, durations, table_path)

    replacements = _update_modelica_pulse_source(
        modelica_target,
        model_name,
        strategy,
        table_filename=table_path.as_posix() if table_path else None,
        amplitudes=amplitudes,
        durations=durations,
        inline_block_definition=inline_block_definition,
        target_component=foc_component,
    )

    logger.info(
        "Prepared FOC-enabled Modelica package",
        extra={
            "package_path": str(generated_package_path),
            "modelica_target": str(modelica_target),
            "foc_path": str(foc_path),
            "table_path": str(table_path) if table_path else None,
            "strategy": strategy,
            "foc_component": foc_component,
            "pulse_replacements": replacements,
            "schedule_duration": float(sum(durations)),
        },
    )

    return {
        "package_path": str(package_entry_path),
        "modelica_target": str(modelica_target),
        "table_path": str(table_path) if table_path else None,
        "schedule_duration": float(sum(durations)),
        "step_count": len(amplitudes),
        "strategy": strategy,
        "foc_component": foc_component,
    }
