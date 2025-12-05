from pathlib import Path

import nrrd
import numpy as np

from vidata.registry import register_loader, register_writer
from vidata.utils.affine import build_affine


@register_writer("image", ".nrrd", backend="nrrd")
@register_writer("mask", ".nrrd", backend="nrrd")
def save_nrrd(data: np.ndarray, file: str | Path, metadata: dict | None = None) -> list[str]:
    """Save a NumPy array as a nrrd using pynrrd.

    Args:
        data (np.ndarray): Image data array (z, y, x) or (y, x).
        file (Union[str, Path]): Output file path (.nrrd).
        metadata (Optional[dict]): Optional metadata dictionary containing:
            - "spacing" (list or np.ndarray): Physical spacing per axis.
            - "origin" (list or np.ndarray): Origin of the image.
            - "direction" (list or np.ndarray): Flattened or matrix-form direction.
    """
    ndims = data.ndim

    if ndims == 3:
        data = data.transpose(2, 1, 0)  # (X,Y,Z) → (Z,Y,X)
    elif ndims == 2:
        data = data.transpose(1, 0)

    # if metadata is not None:
    if metadata is not None and "spacing" in metadata:
        spacing = np.array(metadata["spacing"][::-1])
    else:
        spacing = np.ones(ndims)

    if metadata is not None and "origin" in metadata:
        origin = np.array(metadata["origin"][::-1])
    else:
        origin = np.zeros(ndims)

    if metadata is not None and "direction" in metadata:
        direction = np.array(metadata["direction"]).flatten()[::-1].reshape(ndims, ndims)
    else:
        direction = np.eye(ndims)

    space_dirs = direction * spacing[:, None]

    header = {
        "type": "short",
        "dimension": ndims,
        "space origin": origin.tolist(),
        "space directions": space_dirs.tolist(),
        "encoding": "gzip",  # compressed NRRD
    }
    nrrd.write(str(file), data, header)
    return [str(file)]


@register_loader("image", ".nrrd", backend="nrrd")
@register_loader("mask", ".nrrd", backend="nrrd")
def load_nrrd(file: str | Path) -> tuple[np.ndarray, dict]:
    """Load a nrrd file using pynrrd and return data and metadata. This function is consistent with
    vidata.io.sitk_io.load_sitk

    Args:
        file (Union[str, Path]): Path to the medical image file (.nrrd).

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: A tuple containing:
            - Image data as a NumPy array (z, y, x or y, x).
            - Metadata dictionary with:
                - "spacing": voxel spacing (np.ndarray)
                - "origin": image origin (np.ndarray)
                - "direction": orientation matrix (np.ndarray)
                - "affine": computed affine matrix (np.ndarray)
    """
    array, header = nrrd.read(str(file))
    ndims = array.ndim

    # Convert the array to z,y,x axis ordering (numpy)
    if ndims == 3:
        array = array.transpose(2, 1, 0)  # (X,Y,Z) → (Z,Y,X)
    elif ndims == 2:
        array = array.transpose(1, 0)  # (X,Y) → (Y,X)

    # Extract the metadata
    origin = np.array(header["space origin"]) if "space origin" in header else np.zeros(ndims)
    spacing = (
        np.linalg.norm(np.array(header["space directions"]), axis=1)
        if "space directions" in header
        else np.ones(ndims)
    )

    if "space directions" in header:
        norms = np.linalg.norm(header["space directions"], axis=1, keepdims=True)
        direction = header["space directions"] / (norms + 1e-12)
    else:
        direction = np.eye(ndims)
    # NRRD uses COLUMN vectors → SITK uses ROW vectors --> .T
    direction = direction.T.flatten()  # Flattened SITK style

    # Convert Metadata from (X,Y,Z) → (Z,Y,X)
    spacing = spacing[::-1]
    origin = origin[::-1]
    direction = np.array(direction[::-1]).reshape(ndims, ndims)

    affine = build_affine(ndims, spacing, origin, direction)

    metadata = {
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "affine": affine,
        "header": header,
    }

    return array, metadata
