from pathlib import Path

from tricys.core.foc import (
    _build_array_pulse_block,
    _build_table_pulse_block,
    prepare_foc_simulation_package,
)
from tricys.preprocessing.foc import FOCInjectorPlugin
from tricys.workflow.context import PipelineContext


def test_foc_table_pulse_block_includes_amplitude_scale():
    block_str = _build_table_pulse_block(["y"])
    assert "parameter Real amplitude_scale = 1.0" in block_str
    assert (
        "source_value = table.y[1] * conversion_factor * amplitude_scale;" in block_str
    )


def test_foc_array_pulse_block_includes_amplitude_scale():
    block_str = _build_array_pulse_block(["y"])
    assert "parameter Real amplitude_scale = 1.0" in block_str
    assert "source_value = conversion_factor * amplitude_scale * sum(" in block_str


def test_foc_conflict_detection_warns_on_width_and_period():
    plugin = FOCInjectorPlugin()
    context = PipelineContext.from_config(
        {
            "simulation_parameters": {
                "pulseSource.width": [50, 90],
                "plasma.nf": [0.1, 0.2],
            },
            "sensitivity_analysis": {
                "analysis_cases": [
                    {
                        "case_name": "sobol_test",
                        "independent_variable": ["pulseSource.period", "blanket.TBR"],
                    }
                ]
            },
        }
    )
    context.jobs = [{"pulseSource.width": 50}, {"pulseSource.width": 90}]

    warnings = plugin._check_foc_parameter_conflicts(context, "pulseSource")
    assert len(warnings) == 2
    assert any("pulseSource.width" in w for w in warnings)
    assert any("pulseSource.period" in w for w in warnings)
    assert context.get_metadata("foc_parameter_warnings") == warnings


def test_foc_conflict_detection_permits_amplitude_scale():
    plugin = FOCInjectorPlugin()
    context = PipelineContext.from_config(
        {
            "simulation_parameters": {
                "pulseSource.amplitude_scale": [0.8, 1.0, 1.2],
                "blanket.TBR": [1.05, 1.15],
            },
        }
    )
    context.jobs = [{"pulseSource.amplitude_scale": 0.8}]

    warnings = plugin._check_foc_parameter_conflicts(context, "pulseSource")
    assert len(warnings) == 0
    assert context.get_metadata("foc_parameter_warnings") is None


def test_prepare_foc_package_embeds_amplitude_scale(tmp_path):
    model_path = tmp_path / "Example.mo"
    model_path.write_text(
        """
package Example
model Cycle
  Modelica.Blocks.Sources.Pulse pulseSource(amplitude = 1, period = 10, width = 90) annotation(
    Placement(transformation(origin = {-90, 10}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(pulseSource.y, plasma.pulseInput);
end Cycle;
end Example;
""".strip(),
        encoding="utf-8",
    )
    foc_path = tmp_path / "scenario.foc"
    foc_path.write_text("PULSE 1000 10 5 2\n", encoding="utf-8")

    # 1. Table strategy
    result_table = prepare_foc_simulation_package(
        model_path,
        "Example.Cycle",
        foc_path,
        tmp_path / "prepared_table",
        strategy="table",
        foc_component="pulseSource",
    )
    table_text = Path(result_table["package_path"]).read_text(encoding="utf-8")
    assert "parameter Real amplitude_scale = 1.0" in table_text
    assert (
        "source_value = table.y[1] * conversion_factor * amplitude_scale;" in table_text
    )

    # 2. Array strategy
    result_array = prepare_foc_simulation_package(
        model_path,
        "Example.Cycle",
        foc_path,
        tmp_path / "prepared_array",
        strategy="array",
        foc_component="pulseSource",
    )
    array_text = Path(result_array["package_path"]).read_text(encoding="utf-8")
    assert "parameter Real amplitude_scale = 1.0" in array_text
    assert "source_value = conversion_factor * amplitude_scale * sum(" in array_text
