"""Multi-file Modelica packages must survive OMPython's work-directory copy.

OMPython 4 (``ModelicaSystemOMC``) copies the given ``.mo`` file into its own work
directory and loads that copy. For a single-file model that is harmless; for a
``package.mo`` the copy lands alone in ``/tmp/<uuid>/``, and OMC then refuses it:

    Expected the package to have name <uuid>, but got <package name>

``prepare_modelica_source`` stages the whole package directory under a directory named
after the package, so the path OMPython computes equals the file it was handed and the
package is loaded in place with its sibling ``.mo`` files next to it.

The staging contract is pure filesystem work (runs in the CI gate); the end-to-end
``ModelicaSystem`` construction needs OpenModelica and is marked ``slow``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tricys.core.modelica import ModelicaSystem, prepare_modelica_source

try:
    from OMPython import OMCSessionZMQ  # noqa: F401

    OMPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without OMPython
    OMPYTHON_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PACKAGE = (
    REPO_ROOT
    / "tricys"
    / "example"
    / "example_data"
    / "example_model"
    / "package.mo"
)


def _make_package(root: Path, name: str = "demo_pkg") -> Path:
    package_dir = root / name
    package_dir.mkdir(parents=True)
    (package_dir / "package.mo").write_text(
        f"package {name}\nend {name};\n", encoding="utf-8"
    )
    (package_dir / "Part.mo").write_text("model Part\nend Part;\n", encoding="utf-8")
    (package_dir / "package.order").write_text("Part\n", encoding="utf-8")
    return package_dir


def test_multifile_package_is_staged_under_matching_directory(tmp_path):
    package_dir = _make_package(tmp_path / "src")

    model_file, build_dir = prepare_modelica_source(
        str(package_dir / "package.mo"), str(tmp_path / "work")
    )

    model_path = Path(model_file)
    assert model_path.name == "package.mo"
    # OMC infers the package name from the containing directory: it must match.
    assert model_path.parent.name == package_dir.name
    assert model_path.is_file()
    assert (model_path.parent / "Part.mo").is_file()
    assert (model_path.parent / "package.order").is_file()
    assert build_dir == str(model_path.parent)


def test_staging_leaves_the_source_package_untouched(tmp_path):
    package_dir = _make_package(tmp_path / "src")
    before = sorted(p.name for p in package_dir.iterdir())

    prepare_modelica_source(str(package_dir / "package.mo"), str(tmp_path / "work"))

    assert sorted(p.name for p in package_dir.iterdir()) == before


def test_staging_is_idempotent(tmp_path):
    package_dir = _make_package(tmp_path / "src")
    work = tmp_path / "work"

    first = prepare_modelica_source(str(package_dir / "package.mo"), str(work))
    second = prepare_modelica_source(str(package_dir / "package.mo"), str(work))

    assert first == second


def test_single_file_model_is_used_in_place(tmp_path):
    solo = tmp_path / "solo.mo"
    solo.write_text("model Solo\nend Solo;\n", encoding="utf-8")

    model_file, build_dir = prepare_modelica_source(str(solo), str(tmp_path / "work"))

    assert model_file == str(solo)
    assert build_dir is None


@pytest.mark.slow
def test_modelica_system_loads_multifile_package_through_staging(tmp_path):
    """End-to-end: the exact constructor call the engine makes must not raise."""
    if not OMPYTHON_AVAILABLE or not EXAMPLE_PACKAGE.is_file() or not shutil.which("omc"):
        pytest.skip("OpenModelica / OMPython not available")

    model_file, build_dir = prepare_modelica_source(
        str(EXAMPLE_PACKAGE), str(tmp_path / "work")
    )
    mod = ModelicaSystem(
        fileName=model_file,
        modelName="example_model.Cycle",
        customBuildDirectory=build_dir,
    )

    parameters = mod.getParameters()
    assert isinstance(parameters, dict)
    assert parameters, "package loaded but no parameters were reported"
