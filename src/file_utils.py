import os

def get_unique_filename(base_path: str, filename: str) -> str:
    """
    Generate a unique filename by appending a counter if the file already exists.

    Args:
        base_path (str): Directory path where the file will be saved.
        filename (str): Desired filename (including extension).

    Returns:
        str: A unique file path that does not already exist.
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