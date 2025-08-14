"""Utility functions for file and directory management.

This module provides helper functions for creating unique filenames and managing log
file rotation.
"""

import os


def get_unique_filename(base_path: str, filename: str) -> str:
    """Generates a unique filename by appending a counter if the file already exists.

    Args:
        base_path (str): The directory path where the file will be saved.
        filename (str): The desired filename, including the extension.

    Returns:
        str: A unique, non-existing file path.
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


def delete_old_logs(log_path: str, max_files: int):
    """Deletes the oldest log files in a directory to meet a specified limit.

    Checks the number of `.log` files in the given directory and removes the
    oldest ones based on modification time until the file count matches the
    `max_files` limit.

    Args:
        log_path (str): The path to the directory containing log files.
        max_files (int): The maximum number of `.log` files to retain.
    """
    log_files = [
        os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith(".log")
    ]

    if len(log_files) > max_files:
        # Sort by modification time, oldest first
        log_files.sort(key=os.path.getmtime)

        # Calculate how many files to delete
        files_to_delete_count = len(log_files) - max_files

        # Delete the oldest files
        for i in range(files_to_delete_count):
            os.remove(log_files[i])
