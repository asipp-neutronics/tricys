"""回归测试：``script/cosim/4_export_ssp.py`` 不得删除 ``--fmu_dir`` 内既有的用户文件。

背景（PR #101 评审项 C1）：原实现在 auto-build 的 ``finally`` 块中遍历
``--fmu_dir`` 并删除所有非 ``.fmu`` 条目（``os.remove`` / ``shutil.rmtree``）。
由于 ``--fmu_dir`` 由用户指定，一旦该目录里存在与构建无关的其它文件或子目录，
就会被**不可逆删除**。修复后只清理本次构建新产生的中间产物，
预置内容一律不动。

本测试不依赖真实 OMC/OMPython：在 ``sys.modules`` 中注入桩件后加载被测脚本，
并以「哨兵文件 / 哨兵目录 / 既有 .fmu」作为判据。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.build_test

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "script" / "cosim" / "4_export_ssp.py"

# 伪 OMC 在 buildModelFMU 时生成的中间产物
JUNK_FILES = ("Pulse.c", "Pulse.o", "Pulse_info.json")
JUNK_DIR = "TempBuildDir"


def _load_script_with_stubs(monkeypatch: pytest.MonkeyPatch, fmu_dir: Path):
    """注入 OMPython / OMSimulator 桩件并加载被测脚本模块。"""

    class FakeOMC:
        def sendExpression(self, expr, parsed=False):  # noqa: ARG002
            e = expr.strip()
            if e.startswith("loadFile"):
                return True
            if e == "cd()":
                return ""  # 假值 → 跳过 cwd 还原分支
            if e.startswith("cd("):
                return True
            if e.startswith("buildModelFMU"):
                for name in JUNK_FILES:
                    (fmu_dir / name).write_text("junk", encoding="utf-8")
                (fmu_dir / JUNK_DIR).mkdir(exist_ok=True)
                (fmu_dir / "Pulse.fmu").write_text("fmu", encoding="utf-8")
                return "Pulse.fmu"
            if e == "getErrorString()":
                return ""
            return True

    class FakeSSP:
        def addResource(self, *args, **kwargs):  # noqa: ARG002
            pass

        def addComponent(self, *args, **kwargs):  # noqa: ARG002
            pass

        def export(self, path):
            Path(path).write_text("ssp", encoding="utf-8")

    ompython = types.ModuleType("OMPython")
    ompython.OMCSessionZMQ = FakeOMC
    omsimulator = types.ModuleType("OMSimulator")
    omsimulator.SSP = FakeSSP
    omsimulator.CRef = lambda *args, **kwargs: ("cref", args)

    monkeypatch.setitem(sys.modules, "OMPython", ompython)
    monkeypatch.setitem(sys.modules, "OMSimulator", omsimulator)

    spec = importlib.util.spec_from_file_location(
        "_ssp_export_under_test", TARGET_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_ssp_never_deletes_preexisting_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fmu_dir = tmp_path / "Pulse_fmus"
    (fmu_dir / "user_data" / "nested").mkdir(parents=True)
    (fmu_dir / "USER_IMPORTANT.txt").write_text("irreplaceable", encoding="utf-8")
    (fmu_dir / "user_data" / "nested" / "child.txt").write_text(
        "nested user data", encoding="utf-8"
    )
    (fmu_dir / "OtherModel.fmu").write_text("pre-existing fmu", encoding="utf-8")
    mo_file = tmp_path / "model.mo"
    mo_file.write_text("model", encoding="utf-8")

    module = _load_script_with_stubs(monkeypatch, fmu_dir)

    component = module.ModelComponent(
        class_name="Pulse",
        instance_name="pulse",
        full_class_name="Modelica.Blocks.Sources.Pulse",
    )
    monkeypatch.setattr(
        module, "extract_model_topology", lambda omc, name: ([component], [])
    )
    monkeypatch.setattr(
        module, "build_component_type_map", lambda comps: {"pulse": "Pulse"}
    )
    monkeypatch.setattr(module, "add_topology_connection", lambda *a, **k: None)

    module.export_ssp(
        str(mo_file), "M", str(fmu_dir), str(tmp_path / "out" / "x.ssp")
    )

    # 修复点：预置于 --fmu_dir 的内容必须原样存活
    assert (fmu_dir / "USER_IMPORTANT.txt").exists()
    assert (fmu_dir / "user_data" / "nested" / "child.txt").exists()
    assert (fmu_dir / "OtherModel.fmu").exists()

    # 原有行为保留：本次构建产物与中间物仍按预期处理
    assert (fmu_dir / "Pulse.fmu").exists()
    for name in JUNK_FILES:
        assert not (fmu_dir / name).exists(), f"中间产物 {name} 应被清理"
    assert not (fmu_dir / JUNK_DIR).exists(), "中间目录应被清理"
