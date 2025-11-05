import gzip
import json
from pathlib import Path
from typing import Any


def load_json(json_file: str | Path, encoding: str | None = None) -> Any:
    """Load data from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(json_file, encoding=encoding) as f:
        data = json.load(f)
    return data


def save_json(
    data: Any, json_file: str | Path, indent: int = 4, encoding: str | None = None
) -> None:
    """Write data to a JSON file.

    Args:
        json_file (str): Path to the JSON file.
        data (Any): The data to be written to the JSON file.
        indent (int): Indent level.
    """
    with open(json_file, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def load_jsongz(json_file: str | Path, encoding="utf-8") -> Any:
    """Load data from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with gzip.open(json_file, "rt", encoding=encoding) as f:
        data = json.load(f)
    return data


def save_jsongz(data: Any, json_file: str | Path, encoding="utf-8") -> None:
    """Write data to a JSON file.

    Args:
        json_file (str): Path to the JSON file.
        data (Any): The data to be written to the JSON file.
        indent (int): Indent level.
    """
    with gzip.open(json_file, "wt", encoding=encoding) as f:
        json.dump(data, f)  # =, separators=(",", ":"))

    # with open(json_file, "w") as f:
    #     json.dump(data, f, indent=indent)
