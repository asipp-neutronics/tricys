import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest

from tricys.core.foc import (
    FOCParseError,
    export_for_combitimetable,
    parse_foc_file,
    prepare_foc_simulation_package,
)
from tricys.core.modelica import get_om_session

if find_spec("OMPython") is not None:
    OMPYTHON_AVAILABLE = True
else:
    OMPYTHON_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "tricys" / "example" / "example_data" / "example_model_single"


def _write_mixed_schedule(foc_path: Path) -> None:
    foc_path.write_text(
        """
PULSE 1500 400 100 3
POWER 1000
BURN 1000
DWELL 500
PULSE 1200 300 100 2
POWER 800
BURN 3400
DWELL 300
PULSE 900 700 0 3
PULSE 1500 400 100 2
POWER 500
BURN 1000
DWELL 500
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_parse_foc_file_supports_mixed_schedule(tmp_path):
    foc_path = tmp_path / "test_scenario.foc"
    _write_mixed_schedule(foc_path)

    amplitudes, durations = parse_foc_file(str(foc_path))

    assert len(amplitudes) == 23
    assert len(durations) == 23
    assert amplitudes[:6] == [1500.0, 0.0, 1500.0, 0.0, 1500.0, 0.0]
    assert durations[-6:] == [400.0, 100.0, 400.0, 100.0, 1000.0, 500.0]
    assert sum(durations) == 12100.0


def test_parse_foc_file_applies_top_level_time_conversion(tmp_path):
    foc_path = tmp_path / "time_conversion.foc"
    foc_path.write_text(
        "TIME_CONVERSION 3600\nPULSE 1500 0.5 0.25 2\n",
        encoding="utf-8",
    )

    amplitudes, durations = parse_foc_file(str(foc_path))

    assert amplitudes == [1500.0, 0.0, 1500.0, 0.0]
    assert durations == [1800.0, 900.0, 1800.0, 900.0]


def test_parse_foc_file_declares_time_unit_without_implicit_conversion(tmp_path):
    foc_path = tmp_path / "time_unit_hour.foc"
    foc_path.write_text(
        "TIME_UNIT hour\nPULSE 1500 0.5 0.25 1\n",
        encoding="utf-8",
    )

    amplitudes, durations = parse_foc_file(str(foc_path))

    assert amplitudes == [1500.0, 0.0]
    assert durations == [0.5, 0.25]


def test_parse_foc_file_accepts_named_time_conversion_alias(tmp_path):
    foc_path = tmp_path / "time_conversion_alias.foc"
    foc_path.write_text(
        "TIME_CONVERSION second_to_hour\nPULSE 1500 7200 3600 1\n",
        encoding="utf-8",
    )

    amplitudes, durations = parse_foc_file(str(foc_path))

    assert amplitudes == [1500.0, 0.0]
    assert durations == [2.0, 1.0]


def test_parse_foc_file_accepts_explicit_no_time_conversion(tmp_path):
    foc_path = tmp_path / "time_conversion_none.foc"
    foc_path.write_text(
        "TIME_CONVERSION NONE\nPULSE 1500 10 5 1\n",
        encoding="utf-8",
    )

    _, durations = parse_foc_file(str(foc_path))

    assert durations == [10.0, 5.0]


def test_parse_foc_file_accepts_time_unit_with_explicit_no_conversion(tmp_path):
    foc_path = tmp_path / "time_unit_none.foc"
    foc_path.write_text(
        "TIME_UNIT hour\nTIME_CONVERSION NONE\nPULSE 1500 10 5 1\n",
        encoding="utf-8",
    )

    _, durations = parse_foc_file(str(foc_path))

    assert durations == [10.0, 5.0]


@pytest.mark.parametrize(
    ("content", "message_fragment"),
    [
        ("POWRE 1000\nBURN 10\n", "unsupported command"),
        ("BURN 10\n", "requires a POWER"),
        ("PULSE 1000 10 5 2.9\n", "positive integer"),
        ("REPEAT 3\n", "requires a completed schedule block"),
        ("BEGIN_SCHEDULE\nPOWER 1000\n", "not closed with END_SCHEDULE"),
        ("TIME_UNIT month\nPULSE 1000 10 5 1\n", "unsupported TIME_UNIT"),
        (
            "POWER 1000\nTIME_CONVERSION 3600\nBURN 10\n",
            "must appear before all other FOC commands",
        ),
        (
            "TIME_CONVERSION second_to_hour\nTIME_UNIT hour\nPULSE 1000 10 5 1\n",
            "does not match TIME_CONVERSION source unit",
        ),
        ("TIME_CONVERSION 0\nPULSE 1000 10 5 1\n", "greater than zero"),
    ],
)
def test_parse_foc_file_rejects_invalid_dsl(tmp_path, content, message_fragment):
    foc_path = tmp_path / "invalid.foc"
    foc_path.write_text(content, encoding="utf-8")

    with pytest.raises(FOCParseError, match=message_fragment):
        parse_foc_file(str(foc_path))


def test_export_for_combitimetable_uses_tab_delimiter_and_terminal_zero(
    tmp_path,
):
    table_path = tmp_path / "foc_table.txt"

    export_for_combitimetable([1000.0], [100.0], str(table_path))

    lines = table_path.read_text(encoding="utf-8").splitlines()
    assert lines[:5] == [
        "#1",
        "float FOC_Data(3, 2)",
        "0.0\t1000.0",
        "100.0\t1000.0",
        "100.0\t0.0",
    ]

    csv_lines = table_path.with_suffix(".csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "Time(s),Power(MW)"


def test_prepare_foc_simulation_package_array_strategy_embeds_parameter_arrays(
    tmp_path,
):
    model_copy = tmp_path / "example_model.mo"
    shutil.copy2(EXAMPLE_DIR / "example_model.mo", model_copy)

    foc_path = tmp_path / "scenario.foc"
    foc_path.write_text("PULSE 1000 10 5 2\n", encoding="utf-8")

    result = prepare_foc_simulation_package(
        model_copy,
        "example_model.Cycle",
        foc_path,
        tmp_path / "prepared",
        strategy="array",
        foc_component="pulseSource",
    )

    prepared_text = Path(result["package_path"]).read_text(encoding="utf-8")
    assert "block FOC_ArrayPulse" in prepared_text
    assert (
        "FOC_ArrayPulse pulseSource(amplitudes={1000.0, 0.0, 1000.0, 0.0}, "
        "durations={10.0, 5.0, 10.0, 5.0})"
    ) in prepared_text


def test_prepare_foc_simulation_package_array_strategy_applies_time_conversion(
    tmp_path,
):
    model_copy = tmp_path / "example_model.mo"
    shutil.copy2(EXAMPLE_DIR / "example_model.mo", model_copy)

    foc_path = tmp_path / "scenario.foc"
    foc_path.write_text(
        "TIME_CONVERSION 3600\nPULSE 1000 0.5 0.25 2\n",
        encoding="utf-8",
    )

    result = prepare_foc_simulation_package(
        model_copy,
        "example_model.Cycle",
        foc_path,
        tmp_path / "prepared",
        strategy="array",
        foc_component="pulseSource",
    )

    prepared_text = Path(result["package_path"]).read_text(encoding="utf-8")
    assert (
        "FOC_ArrayPulse pulseSource(amplitudes={1000.0, 0.0, 1000.0, 0.0}, "
        "durations={1800.0, 900.0, 1800.0, 900.0})"
    ) in prepared_text


def test_prepare_foc_simulation_package_array_strategy_supports_named_time_conversion(
    tmp_path,
):
    model_copy = tmp_path / "example_model.mo"
    shutil.copy2(EXAMPLE_DIR / "example_model.mo", model_copy)

    foc_path = tmp_path / "scenario.foc"
    foc_path.write_text(
        "TIME_CONVERSION second_to_hour\nPULSE 1000 7200 3600 1\n",
        encoding="utf-8",
    )

    result = prepare_foc_simulation_package(
        model_copy,
        "example_model.Cycle",
        foc_path,
        tmp_path / "prepared",
        strategy="array",
        foc_component="pulseSource",
    )

    prepared_text = Path(result["package_path"]).read_text(encoding="utf-8")
    assert (
        "FOC_ArrayPulse pulseSource(amplitudes={1000.0, 0.0}, " "durations={2.0, 1.0})"
    ) in prepared_text


@pytest.mark.skipif(not OMPYTHON_AVAILABLE, reason="OMPython is not available")
def test_table_pulse_integration_builds_in_openmodelica(tmp_path):
    model_copy = tmp_path / "example_model.mo"
    shutil.copy2(EXAMPLE_DIR / "example_model.mo", model_copy)

    foc_path = tmp_path / "scenario.foc"
    foc_path.write_text("PULSE 1000 10 5 2\n", encoding="utf-8")

    result = prepare_foc_simulation_package(
        model_copy,
        "example_model.Cycle",
        foc_path,
        tmp_path / "prepared",
        strategy="table",
        foc_component="pulseSource",
    )

    prepared_model = Path(result["package_path"])
    prepared_dir = prepared_model.parent
    cleanup_roots = [Path.cwd(), prepared_dir]

    omc = get_om_session()
    try:
        assert omc.sendExpression(f'cd("{prepared_dir.as_posix()}")')
        assert omc.sendExpression("loadModel(Modelica)")
        assert omc.sendExpression(f'loadFile("{prepared_model.as_posix()}")')

        build_result = omc.sendExpression("buildModel(example_model.Cycle)")
        assert build_result
        assert build_result[0]
    finally:
        omc.sendExpression("quit()")
        for cleanup_root in cleanup_roots:
            for artifact in cleanup_root.glob("example_model.Cycle*"):
                if artifact.is_file():
                    artifact.unlink()
