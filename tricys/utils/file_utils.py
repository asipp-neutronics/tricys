"""Utility functions for file and directory management.

This module provides helper functions for creating unique filenames and managing log
file rotation.
"""

import json
import logging
import os
import shutil
import sys
import zipfile

from tricys.utils.log_utils import restore_configs_from_log

logger = logging.getLogger(__name__)


def get_unique_filename(base_path: str, filename: str) -> str:
    """Generates a unique filename by appending a counter if the file already exists.

    Args:
        base_path: The directory path where the file will be saved.
        filename: The desired filename, including the extension.

    Returns:
        A unique, non-existing file path.

    Note:
        Appends _1, _2, etc. before the extension until a non-existing filename is found.
        Example: if "data.csv" exists, returns "data_1.csv", then "data_2.csv", etc.
    """
    base_name, ext = os.path.splitext(filename)
    counter = 0
    new_filename = filename
    new_filepath = os.path.join(base_path, new_filename)

    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{base_name}_{counter}{ext}"
        new_filepath = os.path.join(base_path, new_filename)

    return new_filepath


def archive_run(timestamp: str) -> None:
    """Archives a run (simulation or analysis) based on its configuration.

    Args:
        timestamp: The timestamp directory name of the run to archive.

    Note:
        Determines run type (analysis vs simulation) from configuration. Delegates
        to _archive_run() with appropriate run_type. Extracts configuration from
        log files using restore_configs_from_log().
    """

    configs = restore_configs_from_log(timestamp)
    if not configs:
        return
    runtime_config, original_config = configs
    logger.info("Successfully extracted both runtime and original configurations.")

    is_analysis = "sensitivity_analysis" in original_config and original_config.get(
        "sensitivity_analysis", {}
    ).get("enabled", False)

    if is_analysis:
        _archive_run(timestamp, "analysis")
    else:
        _archive_run(timestamp, "simulation")


def _archive_run(timestamp: str, run_type: str) -> None:
    """Internal implementation to archive a run by collecting all necessary files.

    Args:
        timestamp: The timestamp directory name to archive.
        run_type: Type of run ("analysis" or "simulation").

    Note:
        Creates temporary 'archive' directory, copies assets and workspace, updates
        paths in configuration, and creates zip archive. For analysis runs, ignores
        temp and *.tmp files. Cleans up temporary directory after completion.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting archive for {run_type} run: {timestamp}")

    if not os.path.isdir(timestamp):
        logger.error(f"Timestamp directory not found: {timestamp}")
        sys.exit(1)

    archive_root = "archive"
    if os.path.exists(archive_root):
        shutil.rmtree(archive_root)
    os.makedirs(archive_root)
    logger.info(f"Created temporary archive directory: {archive_root}")

    try:
        # 1. Extract configs from log
        configs = restore_configs_from_log(timestamp)
        if not configs:
            return
        runtime_config, original_config = configs
        logger.info("Successfully extracted both runtime and original configurations.")

        # 2. Copy assets and update paths
        final_config = json.loads(json.dumps(original_config))
        _copy_and_update_paths(runtime_config, final_config, archive_root, logger)
        logger.info("Copied external assets and updated paths in final configuration.")

        # 3. Copy workspace
        ignore_patterns = (
            shutil.ignore_patterns("temp", "*.tmp")
            if run_type == "analysis"
            else shutil.ignore_patterns("temp")
        )
        dest_workspace_path = os.path.join(archive_root, timestamp)
        shutil.copytree(timestamp, dest_workspace_path, ignore=ignore_patterns)
        logger.info(f"Copied workspace '{timestamp}' to archive, ignoring temp files.")

        # 4. Save final config
        final_config_path = os.path.join(archive_root, "config.json")
        with open(final_config_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved modified original configuration to {final_config_path}")

        # 5. Create zip archive
        archive_filename_prefix = (
            "archive_ana_" if run_type == "analysis" else "archive_"
        )
        archive_filename = f"{archive_filename_prefix}{timestamp}"
        shutil.make_archive(archive_filename, "zip", archive_root)
        logger.info(f"Successfully created archive: {archive_filename}.zip")

    finally:
        if os.path.exists(archive_root):
            shutil.rmtree(archive_root)
            logger.info(f"Cleaned up temporary archive directory: {archive_root}")


def unarchive_run(zip_file: str) -> None:
    """Unarchives a simulation run from a zip file.

    Args:
        zip_file: Path to the zip file to extract.

    Raises:
        SystemExit: If zip file not found or extraction fails.

    Note:
        Extracts to current directory if empty, otherwise creates new directory
        named after the zip file. Sets up basic logging for the unarchive process.
        Handles BadZipFile exceptions gracefully.
    """
    # Basic logging setup for unarchive command
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    if not os.path.isfile(zip_file):
        logger.error(f"Archive file not found: {zip_file}")
        sys.exit(1)

    target_dir = "."
    if os.listdir("."):  # If the list of CWD contents is not empty
        dir_name = os.path.splitext(os.path.basename(zip_file))[0]
        target_dir = dir_name
        logger.info(
            f"Current directory is not empty. Extracting to new directory: {target_dir}"
        )
        os.makedirs(target_dir, exist_ok=True)
    else:
        logger.info("Current directory is empty. Extracting to current directory.")

    # Unzip the file
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(
            f"Successfully unarchived '{zip_file}' to '{os.path.abspath(target_dir)}'"
        )
    except zipfile.BadZipFile:
        logger.error(f"Error: '{zip_file}' is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during unarchiving: {e}")
        sys.exit(1)


def _copy_and_update_paths(runtime_node, final_node, archive_root, logger) -> None:
    """Recursively traverses nodes to find asset paths and copy them to archive.

    Args:
        runtime_node: Source configuration node with absolute paths.
        final_node: Target configuration node to update with relative paths.
        archive_root: Root directory of the archive.
        logger: Logger instance for logging operations.

    Note:
        Handles special case for independent_variable_sampling when independent_variable
        is "file". Copies files/directories for path keys (ending with '_path' or in
        path_keys list). Updates final_node with relative paths. Processes package_path
        differently for single files vs directories.
    """
    if not isinstance(runtime_node, type(final_node)):
        return

    if isinstance(runtime_node, dict):
        # Handle special case: independent_variable_sampling when independent_variable is "file"
        if (
            runtime_node.get("independent_variable") == "file"
            and "independent_variable_sampling" in runtime_node
            and isinstance(runtime_node["independent_variable_sampling"], str)
            and os.path.isfile(runtime_node["independent_variable_sampling"])
        ):
            original_path = runtime_node["independent_variable_sampling"]
            base_name = os.path.basename(original_path)
            dest_path = os.path.join(archive_root, base_name)
            if not os.path.exists(dest_path):
                shutil.copy(original_path, dest_path)
                logger.info(f"Copied asset: {original_path} -> {dest_path}")
            final_node["independent_variable_sampling"] = base_name

        for key, runtime_value in runtime_node.items():
            if key not in final_node:
                continue

            if isinstance(runtime_value, str):
                path_keys = ["package_path", "db_path", "glossary_path"]
                is_path_key = key.endswith("_path") or key in path_keys

                if is_path_key and os.path.exists(runtime_value):
                    new_relative_path = ""
                    if key == "package_path":
                        if os.path.isfile(runtime_value) and not runtime_value.endswith(
                            "package.mo"
                        ):
                            base_name = os.path.basename(runtime_value)
                            dest_path = os.path.join(archive_root, base_name)
                            if not os.path.exists(dest_path):
                                shutil.copy(runtime_value, dest_path)
                            new_relative_path = base_name
                        else:
                            src_dir = (
                                os.path.dirname(runtime_value)
                                if os.path.isfile(runtime_value)
                                else runtime_value
                            )
                            dir_name = os.path.basename(src_dir)
                            dest_dir = os.path.join(archive_root, dir_name)
                            if not os.path.exists(dest_dir):
                                shutil.copytree(src_dir, dest_dir)

                            if os.path.isfile(runtime_value):
                                new_relative_path = os.path.join(
                                    dir_name, os.path.basename(runtime_value)
                                ).replace("\\", "/")
                            else:
                                new_relative_path = dir_name.replace("\\", "/")
                    else:
                        if os.path.isfile(runtime_value):
                            base_name = os.path.basename(runtime_value)
                            dest_path = os.path.join(archive_root, base_name)
                            if not os.path.exists(dest_path):
                                shutil.copy(runtime_value, dest_path)
                            new_relative_path = base_name
                        elif os.path.isdir(runtime_value):
                            dir_name = os.path.basename(runtime_value)
                            dest_dir = os.path.join(archive_root, dir_name)
                            if not os.path.exists(dest_dir):
                                shutil.copytree(runtime_value, dest_dir)
                            new_relative_path = dir_name

                    if new_relative_path:
                        final_node[key] = new_relative_path
                        logger.info(
                            f"Copied and updated path for '{key}': {runtime_value} -> {new_relative_path}"
                        )

            elif isinstance(runtime_value, (dict, list)):
                _copy_and_update_paths(
                    runtime_value, final_node[key], archive_root, logger
                )

    elif isinstance(runtime_node, list):
        if len(runtime_node) != len(final_node):
            return
        for i in range(len(runtime_node)):
            _copy_and_update_paths(runtime_node[i], final_node[i], archive_root, logger)
