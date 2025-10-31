from pathlib import Path

import numpy as np

from vidata.registry import register_loader, register_writer


@register_loader("image", ".npy", backend="numpy")
@register_loader("mask", ".npy", backend="numpy")
def load_npy(file: str | Path) -> np.ndarray:
    """Load a NumPy array from a .npy file.

    Args:
        file (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    return np.load(file, allow_pickle=False), {}


@register_writer("image", ".npy", backend="numpy")
@register_writer("mask", ".npy", backend="numpy")
def save_npy(array: np.ndarray, file: str | Path, *args, **kwargs) -> list[str]:
    """Save a NumPy array to a .npy file.

    Args:
        array (np.ndarray): NumPy array to save.
        file (str): Output file file.
    """
    np.save(file, array)
    return [str(file)]


@register_loader("image", ".npz", backend="numpy")
@register_loader("mask", ".npz", backend="numpy")
def load_npz(file: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load multiple arrays from a .npz file into a dictionary.

    Args:
        file (str): Path to the .npz file.

    Returns:
        dict[str, np.ndarray]: dictionary mapping keys to arrays.
    """
    with np.load(file) as data:
        return {key: data[key] for key in data.files}, {}


@register_writer("image", ".npz", backend="numpy")
@register_writer("mask", ".npz", backend="numpy")
def save_npz(
    data_dict: dict[str, np.ndarray], file: str | Path, compress: bool = True, *args, **kwargs
) -> list[str]:
    """Save multiple NumPy arrays to a .npz file.

    Args:
        data_dict (dict[str, np.ndarray]): dictionary of arrays to save.
        file (str): Output file file.
        compress (bool, optional): Whether to use compressed format. Defaults to True.
    """
    if compress:
        if isinstance(data_dict, dict):
            np.savez_compressed(file, **data_dict)
        else:
            np.savez_compressed(file, data_dict)
    else:
        if isinstance(data_dict, dict):
            np.savez(file, **data_dict)
        else:
            np.savez(file, data_dict)
    return [str(file)]
