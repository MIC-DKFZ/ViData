import pickle  # nosec B403: pickle import is intentional; only used for trusted inputs.
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """Load a Python object from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Any: The deserialized Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Save a Python object to a pickle file.

    Args:
        data (Any): The Python object to serialize.
        path (str): Destination file path.
        protocol (int, optional): Pickle protocol version. Defaults to highest available.
    """
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)
