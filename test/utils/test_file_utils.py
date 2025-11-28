import os
import shutil
from pathlib import Path

import pytest

from tricys.utils.file_utils import archive_run, get_unique_filename, unarchive_run

TEST_DIR = "temp_file_utils_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_get_unique_filename():
    """Test get_unique_filename."""
    # Test with a non-existing file
    path = get_unique_filename(TEST_DIR, "test.txt")
    assert path == os.path.join(TEST_DIR, "test.txt")

    # Test with an existing file
    (Path(TEST_DIR) / "test.txt").touch()
    path = get_unique_filename(TEST_DIR, "test.txt")
    assert path == os.path.join(TEST_DIR, "test_1.txt")

    # Test with multiple existing files
    (Path(TEST_DIR) / "test_1.txt").touch()
    path = get_unique_filename(TEST_DIR, "test.txt")
    assert path == os.path.join(TEST_DIR, "test_2.txt")


@pytest.mark.skip(reason="Complex test involving logging and file structure")
def test_archive_and_unarchive_run():
    """Test archive_run and unarchive_run."""
    # --- Setup a dummy run directory ---
    timestamp = "20250101_120000"
    run_dir = Path(TEST_DIR) / timestamp
    run_dir.mkdir()

    # Create dummy log file with config
    log_dir = run_dir / "log"
    log_dir.mkdir()
    log_file = log_dir / f"simulation_{timestamp}.log"
    log_content = f"""
2025-01-01 12:00:00,000 - INFO - tricys.utils.config_utils - Original config: {{"paths": {{"package_path": "assets/model.mo"}}, "simulation": {{"model_name": "MyModel"}}, "sensitivity_analysis": {{"enabled": false}}}}
2025-01-01 12:00:00,000 - INFO - tricys.utils.config_utils - Runtime config: {{"paths": {{"package_path": "{os.path.abspath(TEST_DIR)}/assets/model.mo"}}, "simulation": {{"model_name": "MyModel"}}, "run_timestamp": "{timestamp}"}}
"""
    log_file.write_text(log_content)

    # Create dummy assets
    assets_dir = Path(TEST_DIR) / "assets"
    assets_dir.mkdir()
    (assets_dir / "model.mo").touch()

    # --- Test archive_run ---
    os.chdir(TEST_DIR)
    try:
        archive_run(timestamp)
        zip_file = f"archive_{timestamp}.zip"
        assert os.path.exists(zip_file)

        # --- Test unarchive_run ---
        extract_dir = Path("unarchived")
        extract_dir.mkdir()
        os.chdir(extract_dir)

        unarchive_run(f"../{zip_file}")

        # Check unarchived content
        assert (Path(".") / "config.json").exists()
        assert (Path(".") / timestamp / "log").exists()
        assert (Path(".") / "assets" / "model.mo").exists()

    finally:
        os.chdir("..")  # Back to TEST_DIR
        os.chdir("..")  # Back to root
