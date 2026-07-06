import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

TEMPLATE = """\"\"\"
Auto-generated strongly typed processor base class for {class_name}.
Do not edit manually. Re-run generate_processors.py to update.
\"\"\"

from dataclasses import dataclass, field
from typing import List, Optional
from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import UnifiedStateVector, TrackResult, TrackProcessorContext

@dataclass
class {class_name}InputData:
{input_fields}

@dataclass
class {class_name}OutputData:
{output_fields}

class {class_name}ProcessorBase(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.on_initialize(context)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        inputs = {class_name}InputData(
{input_assignments}
        )
        outputs = {class_name}OutputData()
        extra_state = request_vector.extra_state or {{}}

        self.on_step(request_vector.current_time_h, request_vector.dt_slow_h, inputs, outputs, extra_state)

        out_dict = {{}}
{output_dict_assignments}

        return TrackResult(outputs=out_dict)

    def finalize(self) -> None:
        self.on_finalize()

    def on_initialize(self, context: TrackProcessorContext) -> None:
        pass

    def on_step(self, current_time_h: float, dt_slow_h: float, inputs: {class_name}InputData, outputs: {class_name}OutputData, extra_state: dict) -> None:
        raise NotImplementedError("Subclasses must implement on_step")

    def on_finalize(self) -> None:
        pass

    def get_mass_inventory(self) -> float:
        \"\"\"
        Report the total mass (in grams) currently buffered, delayed, or
        held inside this processor and its external surrogate.
        Returns 0.0 by default.
        \"\"\"
        return 0.0

    def get_decay_rate(self) -> float:
        \"\"\"
        Report the radioactive decay rate for Tritium (in g/s).
        Returns 0.0 by default.
        \"\"\"
        return 0.0

    def get_release_rate(self) -> float:
        \"\"\"
        Report the environmental release rate for Tritium (in g/s).
        Returns 0.0 by default.
        \"\"\"
        return 0.0
"""


def generate_from_dict(schema: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # create __init__.py
    with open(os.path.join(output_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("# Auto-generated typed base processors\n")

    generated_count = 0
    for model_name, data in schema.items():
        if model_name == "Cycle":
            continue  # Top-level system, not a component

        connectors = data.get("connectors", {})

        inputs = {}
        outputs = {}

        for name, info in connectors.items():
            ctype = info.get("type", "")
            if "RealInput" in ctype:
                inputs[name] = info
            elif "RealOutput" in ctype:
                outputs[name] = info

        # If no connectors, maybe it's not a component we need a processor for, but let's generate it anyway
        class_name = model_name

        # Build input fields
        input_fields_lines = []
        input_assign_lines = []
        if not inputs:
            input_fields_lines.append("    pass")
        for name, info in inputs.items():
            dim = info.get("dimension")
            if dim is None or not dim:
                # scalar
                input_fields_lines.append(f"    {name}: float = 0.0")
                input_assign_lines.append(
                    f"            {name}=request_vector.boundary_inputs.get('{name}', 0.0),"
                )
            else:
                # list
                input_fields_lines.append(
                    f"    {name}: List[float] = field(default_factory=list)"
                )
                input_assign_lines.append(
                    f"            {name}=request_vector.boundary_inputs.get('{name}', []),"
                )

        # Build output fields
        output_fields_lines = []
        output_dict_lines = []
        if not outputs:
            output_fields_lines.append("    pass")
        for name, info in outputs.items():
            dim = info.get("dimension")
            if dim is None or not dim:
                output_fields_lines.append(f"    {name}: Optional[float] = None")
            else:
                output_fields_lines.append(f"    {name}: Optional[List[float]] = None")
            output_dict_lines.append(
                f"        if outputs.{name} is not None:\n            out_dict['{name}'] = outputs.{name}"
            )

        input_fields_str = (
            "\n".join(input_fields_lines) if input_fields_lines else "    pass"
        )
        input_assign_str = "\n".join(input_assign_lines)
        output_fields_str = (
            "\n".join(output_fields_lines) if output_fields_lines else "    pass"
        )
        output_dict_str = "\n".join(output_dict_lines)

        content = TEMPLATE.format(
            class_name=class_name,
            input_fields=input_fields_str,
            output_fields=output_fields_str,
            input_assignments=input_assign_str,
            output_dict_assignments=output_dict_str,
        )

        file_name = f"{model_name.lower()}_processor_base.py"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated processor base for {model_name} at {file_path}")
        generated_count += 1

    return generated_count


def generate(schema_path, output_dir):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    count = generate_from_dict(schema, output_dir)
    print(f"Generated {count} processor base files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strongly-typed processor bases from schema."
    )
    parser.add_argument("--schema", required=True, help="Path to schema.json")
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save typed_base processors"
    )
    args = parser.parse_args()
    generate(args.schema, args.out_dir)
