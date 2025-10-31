from pathlib import Path

import numpy as np
import tifffile

from vidata.registry import register_loader, register_writer


@register_loader("image", ".tif", ".tiff", backend="tifffile")
@register_loader("mask", ".tif", ".tiff", backend="tifffile")
def load_tif(path: str | Path) -> np.ndarray:
    """Load a TIFF (.tif) file into a NumPy array.

    Args:
        path (str): Path to the TIFF file.

    Returns:
        np.ndarray: Loaded image data as a NumPy array.
    """
    return tifffile.imread(str(path)), {}


@register_writer("image", ".tif", ".tiff", backend="tifffile")
@register_writer("mask", ".tif", ".tiff", backend="tifffile")
def save_tif(
    data: np.ndarray,
    file: str | Path,
    tile_size: int = 256,
    compression: str = "zlib",
) -> list[str]:
    """Save a NumPy array as a tiled, compressed TIFF (.tif) file.

    Args:
        data (np.ndarray): Image data to save.
        file (Union[str, Path]): Output file path.
        tile_size (int, optional): Size of the TIFF tiles. Defaults to 256.
        compression (str, optional): Compression algorithm (e.g., "zlib", "jpeg", "lzma"). Defaults to "zlib".
    """
    options = {
        "tile": (tile_size, tile_size),
        "compression": compression,
    }
    tifffile.imwrite(file, data, **options)
    return [str(file)]
