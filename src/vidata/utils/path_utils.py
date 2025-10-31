from pathlib import Path


def copy_folder_structure(src: str | Path, dst: str | Path) -> None:
    """Copies the folder structure from src to dst without copying files.

    Parameters:
        src (str): The source directory path.
        dst (str): The destination directory path.
    """
    src = Path(src)
    dst = Path(dst)

    for src_dir in src.rglob("*"):
        if src_dir.is_dir():
            # Create the corresponding directory in the destination
            dst_dir = dst / src_dir.relative_to(src)
            dst_dir.mkdir(parents=True, exist_ok=True)
