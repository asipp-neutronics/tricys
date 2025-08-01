"""本模块提供文件操作的实用功能。"""

import os


def get_unique_filename(base_path: str, filename: str) -> str:
    """
    如果文件已存在，则通过附加计数器来生成唯一的文件名。

    参数:
        base_path (str): 将保存文件的目录路径。
        filename (str): 所需的文件名（包括扩展名）。

    返回:
        str: 一个不存在的唯一文件路径。
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
    """
    检查日志目录中的.log文件数量，并删除最旧的文件，直到文件数量达到指定的限制。

    参数:
        log_path (str): 日志文件的目录路径。
        max_files (int): 要保留的最大.log文件数量。
    """
    log_files = [
        os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith(".log")
    ]

    if len(log_files) > max_files:
        # 按修改时间排序，最旧的在前
        log_files.sort(key=os.path.getmtime)

        # 计算要删除的文件数量
        files_to_delete_count = len(log_files) - max_files

        # 删除最旧的文件
        for i in range(files_to_delete_count):
            os.remove(log_files[i])
