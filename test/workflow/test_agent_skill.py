"""Unit tests for Agent Skill plugin metadata contract and AST security sandboxing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from tricys.workflow.base import (
    PostprocessPlugin,
    SecurityValidationError,
)
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import PluginRegistry, validate_script_ast_safety

VALID_EXTERNAL_PLUGIN_SCRIPT = """
from tricys.workflow.base import PostprocessPlugin, PluginMetadata
from tricys.workflow.context import PipelineContext

class CustomAIAnalyticPlugin(PostprocessPlugin):
    name = "custom_ai_analytic"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name=cls.name,
            category="postprocess",
            description="External dynamic AI analytics plugin for tritium inventory.",
            version="1.0.0",
            author="Agent Researcher",
            input_requirements=["trajectory"],
            output_artifacts=["ai_insight.json"],
        )

    def run(self, context: PipelineContext) -> None:
        context.metadata["ai_analyzed"] = True
"""

MALICIOUS_SUBPROCESS_SCRIPT = """
import subprocess
from tricys.workflow.base import PostprocessPlugin

class ExploitPlugin(PostprocessPlugin):
    def run(self, context):
        subprocess.run(["echo", "hacked"])
"""

MALICIOUS_OS_SYSTEM_SCRIPT = """
import os
from tricys.workflow.base import PostprocessPlugin

class ExploitPlugin(PostprocessPlugin):
    def run(self, context):
        os.system("echo hacked")
"""

MALICIOUS_EVAL_SCRIPT = """
from tricys.workflow.base import PostprocessPlugin

class ExploitPlugin(PostprocessPlugin):
    def run(self, context):
        eval("1 + 1")
"""


def test_plugin_registry_introspect_and_info():
    """Tests introspection and query API of PluginRegistry."""
    data = PluginRegistry.introspect()
    assert "preprocess" in data
    assert "postprocess" in data
    assert "engine" in data

    # Verify safety plugin metadata
    safety_meta = PluginRegistry.get_plugin_info("safety")
    assert safety_meta is not None
    assert safety_meta["name"] == "safety"
    assert safety_meta["category"] == "postprocess"
    assert "audit_report.json" in safety_meta["output_artifacts"]

    # Verify engine metadata
    std_meta = PluginRegistry.get_plugin_info("standard")
    assert std_meta is not None
    assert std_meta["category"] == "engine"


def test_valid_external_plugin_dynamic_loading():
    """Tests that a compliant external plugin script is securely loaded and executable."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(VALID_EXTERNAL_PLUGIN_SCRIPT)
        script_path = f.name

    try:
        plugin_cls = PluginRegistry.load_class_from_source(
            script_path, expected_base=PostprocessPlugin
        )
        assert issubclass(plugin_cls, PostprocessPlugin)

        plugin_instance = plugin_cls()
        ctx = PipelineContext(config={})
        plugin_instance.run(ctx)
        assert ctx.metadata.get("ai_analyzed") is True

        meta = plugin_cls.get_metadata()
        assert meta.name == "custom_ai_analytic"
        assert meta.author == "Agent Researcher"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def test_ast_sandbox_blocks_malicious_scripts():
    """Tests that malicious scripts with subprocess, os.system, or eval are blocked by AST guard."""
    # Test 1: Subprocess import
    with pytest.raises(SecurityValidationError) as exc1:
        validate_script_ast_safety(MALICIOUS_SUBPROCESS_SCRIPT)
    assert "import of 'subprocess' is strictly forbidden" in str(exc1.value)

    # Test 2: os.system call
    with pytest.raises(SecurityValidationError) as exc2:
        validate_script_ast_safety(MALICIOUS_OS_SYSTEM_SCRIPT)
    assert "call to 'os.system()' is strictly forbidden" in str(exc2.value)

    # Test 3: eval call
    with pytest.raises(SecurityValidationError) as exc3:
        validate_script_ast_safety(MALICIOUS_EVAL_SCRIPT)
    assert "call to 'eval()' is strictly forbidden" in str(exc3.value)


def test_load_class_from_source_blocks_malicious_file():
    """Tests that load_class_from_source halts before executing malicious code."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(MALICIOUS_OS_SYSTEM_SCRIPT)
        script_path = f.name

    try:
        with pytest.raises(SecurityValidationError):
            PluginRegistry.load_class_from_source(
                script_path, expected_base=PostprocessPlugin
            )
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def test_cli_plugin_list_and_info():
    """Tests CLI commands python -m tricys.main plugin list / info."""
    # 1. Test plugin list --json
    cmd_list = [sys.executable, "-m", "tricys.main", "plugin", "list", "--json"]
    res_list = subprocess.run(cmd_list, capture_output=True, text=True)
    assert res_list.returncode == 0
    data = json.loads(res_list.stdout)
    assert "preprocess" in data
    assert "postprocess" in data
    assert "engine" in data

    # 2. Test plugin info safety --json
    cmd_info = [
        sys.executable,
        "-m",
        "tricys.main",
        "plugin",
        "info",
        "safety",
        "--json",
    ]
    res_info = subprocess.run(cmd_info, capture_output=True, text=True)
    assert res_info.returncode == 0
    info = json.loads(res_info.stdout)
    assert info["name"] == "safety"
    assert info["category"] == "postprocess"


def test_skills_tricys_agent_structure_and_templates():
    """Validates skills/tricys-agent specification integrity and YAML templates."""
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parent.parent.parent
    skill_dir = repo_root / "skills" / "tricys-agent"

    # 1. SKILL.md
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.exists(), "skills/tricys-agent/SKILL.md must exist"
    content = skill_md.read_text(encoding="utf-8")
    assert "name: tricys-agent" in content
    assert "七步法" in content or "7-Step" in content
    assert "Abdou" in content

    # 2. abdou_2021_physics_rules.md & abdou_1986_mrt_remediation.md
    physics_rules = skill_dir / "references" / "abdou_2021_physics_rules.md"
    assert physics_rules.exists()
    pr_content = physics_rules.read_text(encoding="utf-8")
    assert "Abdou" in pr_content
    assert "Grade A" in pr_content

    mrt_rules = skill_dir / "references" / "abdou_1986_mrt_remediation.md"
    assert mrt_rules.exists()
    mrt_content = mrt_rules.read_text(encoding="utf-8")
    assert "Abdou" in mrt_content
    assert "MRT" in mrt_content

    # 3. templates (6 items)
    templates_dir = skill_dir / "templates"
    assert templates_dir.exists()
    expected_templates = [
        "01_sweep_startup_template.yaml",
        "02_simulate_pulsed_bisection_template.yaml",
        "03_sensitivity_sobol_template.yaml",
        "04_pulse_width_sweep_template.yaml",
        "05_simulate_foc_template.yaml",
        "06_mass_balance_audit_template.yaml",
    ]
    for tmpl_name in expected_templates:
        tmpl_path = templates_dir / tmpl_name
        assert tmpl_path.exists(), f"Template {tmpl_name} must exist"
        raw_text = tmpl_path.read_text(encoding="utf-8")

        # Mock placeholders to test parsing as valid YAML dict
        filled_text = (
            raw_text.replace("${MODEL_FILE_PATH}", "mock.mo")
            .replace("${MODEL_NAME}", "Mock.Cycle")
            .replace("${STORAGE_NODE}", "sds.I[1]")
            .replace("${TBR_PARAM}", "blanket.TBR")
            .replace("${PULSE_WIDTH_PARAM}", "pulseSource.width")
            .replace("${FOC_FILE_PATH}", "test.foc")
            .replace("${PULSE_COMPONENT}", "pulseSource")
            .replace("${PARAM_1}", "param1")
            .replace("${PARAM_2}", "param2")
        )
        parsed = yaml.safe_load(filled_text)
        assert isinstance(parsed, dict)
        assert parsed.get("version") == "2.0"
        assert "model" in parsed
        assert "scenario" in parsed
        assert "task" in parsed
