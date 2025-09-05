from pathlib import Path
from typing import Union

import numpy as np

from vidata.registry import register_loader, register_writer


@register_loader("image", ".npy", backend="numpy")
@register_loader("mask", ".npy", backend="numpy")
def load_npy(path: str) -> np.ndarray:
    """Load a NumPy array from a .npy file.

    Args:
        path (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    return np.load(path, allow_pickle=False), {}


@register_writer("image", ".npy", backend="numpy")
@register_writer("mask", ".npy", backend="numpy")
def save_npy(array: np.ndarray, path: Union[str, Path], *args, **kwargs) -> None:
    """Save a NumPy array to a .npy file.

    Args:
        array (np.ndarray): NumPy array to save.
        path (str): Output file path.
    """
    np.save(path, array)


@register_loader("image", ".npz", backend="numpy")
@register_loader("mask", ".npz", backend="numpy")
def load_npz(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load multiple arrays from a .npz file into a dictionary.

    Args:
        path (str): Path to the .npz file.

    Returns:
        dict[str, np.ndarray]: dictionary mapping keys to arrays.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}, {}


@register_writer("image", ".npz", backend="numpy")
@register_writer("mask", ".npz", backend="numpy")
def save_npz(
    data_dict: dict[str, np.ndarray], path: str, compress: bool = True, *args, **kwargs
) -> None:
    """Save multiple NumPy arrays to a .npz file.

    Args:
        data_dict (dict[str, np.ndarray]): dictionary of arrays to save.
        path (str): Output file path.
        compress (bool, optional): Whether to use compressed format. Defaults to True.
    """
    if compress:
        if isinstance(data_dict, dict):
            np.savez_compressed(path, **data_dict)
        else:
            np.savez_compressed(path, data_dict)
    else:
        if isinstance(data_dict, dict):
            np.savez(path, **data_dict)
        else:
            np.savez(path, data_dict)
