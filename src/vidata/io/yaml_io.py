from pathlib import Path
from typing import Any

import yaml  # type: ignore


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file and return the data as a Python object.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Any: Parsed contents of the YAML file (usually a dict or list).
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path, sort_keys: bool = False) -> None:
    """Save a Python object to a YAML file.

    Args:
        data (Any): The Python object to serialize (typically a dict or list).
        path (str): Destination file path.
        sort_keys (bool, optional): Whether to sort dictionary keys. Defaults to False.
    """
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=sort_keys, default_flow_style=False)
