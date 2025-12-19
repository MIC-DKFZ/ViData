import locale
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from vidata.registry import register_loader, register_writer
from vidata.utils.affine import build_affine


@contextmanager
def temporary_c_locale():
    # Save current LC_NUMERIC
    old_locale = locale.setlocale(locale.LC_NUMERIC, None)

    try:
        # Switch to safe C locale
        locale.setlocale(locale.LC_NUMERIC, "C")
        yield
    finally:
        # Restore original locale
        locale.setlocale(locale.LC_NUMERIC, old_locale)


@register_writer("image", ".nii.gz", ".nii", ".mha", ".nrrd", backend="sitk")
@register_writer("mask", ".nii.gz", ".nii", ".mha", ".nrrd", backend="sitk")
def save_sitk(data: np.ndarray, file: str | Path, metadata: dict | None = None) -> list[str]:
    """Save a NumPy array as a medical image file using SimpleITK.

    Args:
        data (np.ndarray): Image data array (z, y, x) or (y, x).
        file (Union[str, Path]): Output file path (.nii.gz, .mha, .nrrd, etc.).
        metadata (Optional[dict]): Optional metadata dictionary containing:
            - "spacing" (list or np.ndarray): Physical spacing per axis.
            - "origin" (list or np.ndarray): Origin of the image.
            - "direction" (list or np.ndarray): Flattened or matrix-form direction.
    """
    image_sitk = sitk.GetImageFromArray(data)
    if metadata is not None:
        if "spacing" in metadata:
            spacing = metadata["spacing"]
            spacing = spacing.tolist() if isinstance(spacing, np.ndarray) else spacing
            image_sitk.SetSpacing(spacing[::-1])
        if "origin" in metadata:
            origin = metadata["origin"]
            origin = origin.tolist() if isinstance(origin, np.ndarray) else origin
            image_sitk.SetOrigin(origin[::-1])
        if "direction" in metadata:
            direction = metadata["direction"]
            direction = np.array(direction) if not isinstance(direction, np.ndarray) else direction
            image_sitk.SetDirection(direction.flatten().tolist()[::-1])

    sitk.WriteImage(image_sitk, str(file), useCompression=True)
    return [str(file)]


@register_loader("image", ".nii.gz", ".nii", ".mha", ".nrrd", backend="sitk")
@register_loader("mask", ".nii.gz", ".nii", ".mha", ".nrrd", backend="sitk")
def load_sitk(file: str | Path) -> tuple[np.ndarray, dict]:
    """Load a medical image file using SimpleITK and return data and metadata.

    Args:
        file (Union[str, Path]): Path to the medical image file (.nii.gz, .mha, .nrrd, etc.).

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: A tuple containing:
            - Image data as a NumPy array (z, y, x or y, x).
            - Metadata dictionary with:
                - "spacing": voxel spacing (np.ndarray)
                - "origin": image origin (np.ndarray)
                - "direction": orientation matrix (np.ndarray)
                - "affine": computed affine matrix (np.ndarray)
    """
    with temporary_c_locale():
        image = sitk.ReadImage(file)

    array = sitk.GetArrayFromImage(image)
    ndims = len(array.shape)

    spacing = np.array(image.GetSpacing()[::-1])
    origin = np.array(image.GetOrigin()[::-1])
    direction = np.array(image.GetDirection()[::-1]).reshape(ndims, ndims)

    affine = build_affine(ndims, spacing, origin, direction)

    metadata = {
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "affine": affine,
    }

    return array, metadata
