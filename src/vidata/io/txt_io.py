from pathlib import Path


def load_txt(txt_file: str | Path) -> list[str]:
    """Load lines from a text file.

    Args:
        txt_file (str): Path to the text file.

    Returns:
        list[str]: list of lines read from the text file (including newline characters).
    """
    with open(txt_file, encoding="utf-8") as f:
        lines = f.readlines()
    return lines


def save_txt(
    lines: list[str], txt_file: str | Path, append: bool = False, newline: str | None = "\n"
) -> None:
    """Write lines to a text file.

    Args:
        lines (list[str]): list of strings to write. Newline characters are added if not present.
        txt_file (str): Path to the text file.
        append (bool): Whether to append to the file instead of overwriting.
        newline (Optional[str]): Newline character to add at the end of each line, if not already present.
    """
    mode = "a" if append else "w"
    with open(txt_file, mode, encoding="utf-8") as f:
        for line in lines:
            if not line.endswith("\n") and newline:
                line += newline
            f.write(line)
