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
