#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for TRICYS V2.0 Example Runner and Legacy V1 Archive."""

from pathlib import Path

import pytest

from tricys.example.example_runner import (
    TricysAllTestRunner,
    TricysAnaTestRunner,
    TricysTestRunner,
    TricysV2ExampleRunner,
)


def test_v1_legacy_directory_structure():
    """Verify that basic and analysis examples are properly archived in v1_examples_legacy."""
    repo_root = Path(__file__).resolve().parent.parent
    legacy_dir = (
        repo_root / "tricys" / "example" / "example_data" / "v1_examples_legacy"
    )

    assert legacy_dir.exists(), f"Legacy dir {legacy_dir} must exist"
    assert (
        legacy_dir / "README.md"
    ).exists(), "v1_examples_legacy/README.md must exist"

    basic_dir = legacy_dir / "basic"
    analysis_dir = legacy_dir / "analysis"

    assert basic_dir.exists(), "v1_examples_legacy/basic must exist"
    assert (
        basic_dir / "example_runner.json"
    ).exists(), "basic/example_runner.json must exist"
    assert analysis_dir.exists(), "v1_examples_legacy/analysis must exist"
    assert (
        analysis_dir / "example_runner.json"
    ).exists(), "analysis/example_runner.json must exist"

    # Verify that README mentions V1.0, time_scale, and time_base
    readme_content = (legacy_dir / "README.md").read_text(encoding="utf-8")
    assert "V1.0" in readme_content
    assert "time_scale" in readme_content
    assert "time_base" in readme_content


def test_v2_runner_discovery():
    """Verify that TricysV2ExampleRunner discovers all V2 examples."""
    runner = TricysV2ExampleRunner()
    assert (
        len(runner.examples) >= 10
    ), f"Expected at least 10 V2 examples, found {len(runner.examples)}"

    # Ensure all required task types are represented
    task_types = {ex["task_type"] for ex in runner.examples.values()}
    assert "SWEEP" in task_types
    assert "SIMULATE" in task_types
    assert "SENSITIVITY" in task_types
    assert "COSIM" in task_types

    # Ensure backward compatibility aliases exist and point to V2 runner
    assert TricysAllTestRunner is TricysV2ExampleRunner
    assert TricysTestRunner is TricysV2ExampleRunner
    assert TricysAnaTestRunner is TricysV2ExampleRunner


def test_v2_runner_validation_and_compilation():
    """Verify that every discovered V2 example passes validation and DAG compilation."""
    runner = TricysV2ExampleRunner()
    for ex_id, ex in runner.examples.items():
        v_ok = runner.validate_example(ex)
        assert v_ok, f"Validation failed for example {ex_id}: {ex['name']}"

        c_ok = runner.compile_example(ex)
        assert c_ok, f"Compilation failed for example {ex_id}: {ex['name']}"


def test_v2_runner_dry_run_copy():
    """Verify copy_example and dry_run execution."""
    runner = TricysV2ExampleRunner()
    ex = runner.examples["1"]

    copied = runner.copy_example(ex, no_vis=True)
    assert copied, "copy_example should succeed"

    dest_cfg = runner.test_example_v2_dir / ex["config"]
    assert dest_cfg.exists(), f"Destination config {dest_cfg} should exist"

    # Test run_command in dry_run mode
    dry_ok = runner.run_command(ex, dry_run=True, no_vis=True)
    assert dry_ok, "Dry-run execution should return True"


def test_v2_runner_cli_flags():
    """Test CLI flags --list, --compile, and --validate via main()."""
    from tricys.example.example_runner.runner import main as runner_main

    # Test --list exits with 0
    with pytest.raises(SystemExit) as exc_info:
        runner_main(["--list"])
    assert exc_info.value.code == 0

    # Test --validate 1 exits with 0
    with pytest.raises(SystemExit) as exc_info:
        runner_main(["--validate", "1"])
    assert exc_info.value.code == 0

    # Test --compile 1 exits with 0
    with pytest.raises(SystemExit) as exc_info:
        runner_main(["--compile", "1"])
    assert exc_info.value.code == 0
