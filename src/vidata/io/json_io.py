import json
from typing import Any


def load_json(json_file: str) -> Any:
    """Load data from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def save_json(data: Any, json_file: str, indent: int = 4) -> None:
    """Write data to a JSON file.

    Args:
        json_file (str): Path to the JSON file.
        data (Any): The data to be written to the JSON file.
        indent (int): Indent level.
    """
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)
