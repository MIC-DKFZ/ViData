from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from vidata.registry import register_loader, register_writer
from vidata.utils.affine import build_affine, decompose_affine


@register_writer("image", ".nii.gz", ".nii", backend="nibabel")
@register_writer("mask", ".nii.gz", ".nii", backend="nibabel")
def save_nib(data, file: str | Path, metadata: dict | None = None) -> list[str]:
    """
    Save a NumPy array and SITK-style metadata to a NIfTI file using nibabel.

    This function expects input data and metadata in **SITK-style** conventions
    (LPS world coordinates, axis order `(z, y, x)` for 3D or `(y, x)` for 2D).
    It converts both the array and affine to **NIfTI-style** conventions
    (RAS+ coordinates, axis order `(x, y, z)` or `(x, y)`), then writes the file
    if metadata is provided in the metadata it is used, otherwise it is constructed from spacing, origin, direction, shear.

    Args:
        data (np.ndarray): Image data array (z, y, x) or (y, x).
        file (Union[str, Path]): Output file path (.nii.gz, .mha, .nrrd, etc.).
        metadata (Optional[dict]): Optional metadata dictionary containing:
            - "spacing" (list or np.ndarray): Physical spacing per axis.
            - "origin" (list or np.ndarray): Origin of the image.
            - "direction" (list or np.ndarray): Flattened or matrix-form direction.
            - "shear" : optional shear parameters.
            - "affine" : optional (ndim+1, ndim+1) affine matrix in SITK ordering.
              If provided, overrides ``spacing``, ``origin``, ``direction``, and ``shear``.
    """
    ndim = data.ndim
    metadata = metadata if metadata is not None else {}
    if "affine" in metadata:
        affine = metadata["affine"]
    else:
        affine = build_affine(
            ndim,
            metadata.get("spacing"),
            metadata.get("origin"),
            metadata.get("direction"),
            metadata.get("shear"),
        )
    if ndim == 3:
        data = data.transpose(2, 1, 0)
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  # swap
    elif ndim == 2:
        data = data.transpose(1, 0)
        P = np.array([[0, 1], [1, 0]], dtype=float)  # swap
    else:
        raise ValueError(f"Expected 2D or 3D volume, got shape {data.shape}")

    affine_nib = np.eye(4)
    affine_nib[:ndim, :ndim] = P @ affine[:ndim, :ndim] @ P.T
    affine_nib[:ndim, -1] = P @ affine[:ndim, ndim]
    affine_nib = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_nib

    image_nib = nib.Nifti1Image(data, affine=affine_nib)
    nib.save(image_nib, str(file))
    return [str(file)]


@register_loader("image", ".nii.gz", ".nii", backend="nibabel")
@register_loader("mask", ".nii.gz", ".nii", backend="nibabel")
def load_nib(file: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a NIfTI image with nibabel and return the data plus **SITK-style** metadata.

    This performs three things to match your SimpleITK conventions:
      1) **Axis order:** converts arrays from nibabel's (x, y, z) to (z, y, x),
         or from (x, y) to (y, x) for 2D.
      2) **World convention:** converts the affine from RAS+ (nibabel) to LPS (ITK/SITK)
         by flipping X and Y in world space.
      3) **Affine shape:** returns a homogeneous affine of shape (ndim+1, ndim+1).

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
    image = nib.load(file)
    array = image.get_fdata()
    affine = image.affine
    ndim = array.ndim
    affine_sitk = np.eye(ndim + 1)

    # (x, y, z) -> (z, y, x) or (x,y) -> (y,x)
    if ndim == 3:
        array = array.transpose(2, 1, 0)
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  # swap
    elif ndim == 2:
        array = array.transpose(1, 0)
        P = np.array([[0, 1], [1, 0]], dtype=float)  # swap
    else:
        raise ValueError(f"Expected 2D or 3D volume, got shape {array.shape}")
    # --- Convert affine: RAS+ -> LPS, then reorder axes to match array ---
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine  # world flip X,Y
    # linear part: (ndim x ndim) block; translation: last column
    affine_sitk[:ndim, :ndim] = P @ affine[:ndim, :ndim] @ P.T
    affine_sitk[:ndim, ndim] = P @ affine[:ndim, 3]

    # --- Decompose to spacing/origin/direction (napari-style) ---
    scale, translate, R, shear = decompose_affine(affine_sitk)

    metadata = {
        "spacing": scale,
        "origin": translate,
        "direction": R,
        "shear": shear,
        "affine": affine_sitk,
    }
    return array, metadata


@register_writer("image", ".nii.gz", ".nii", backend="nibabelRO")
@register_writer("mask", ".nii.gz", ".nii", backend="nibabelRO")
def save_nibRO(data, file: str | Path, metadata: dict | None = None) -> list[str]:
    """
    Save a NumPy array and SITK-style metadata to a NIfTI file using nibabel,
    performing reorientation to a consistent voxel order.

    This function expects input data and metadata in **SITK-style** conventions
    (LPS world coordinates, axis order `(z, y, x)` for 3D or `(y, x)` for 2D).
    It converts both the array and affine to **NIfTI-style** conventions
    (RAS+ coordinates, axis order `(x, y, z)` or `(x, y)`), then writes the file
    if metadata is provided in the metadata it is used, otherwise it is constructed from spacing, origin, direction, shear.
    If `metadata["affine_original"]` is provided, it is used for reorientation.

    Args:
        data (np.ndarray): Image data array (z, y, x) or (y, x).
        file (Union[str, Path]): Output file path (.nii.gz, .mha, .nrrd, etc.).
        metadata (Optional[dict]): Optional metadata dictionary containing:
            - "spacing" (list or np.ndarray): Physical spacing per axis.
            - "origin" (list or np.ndarray): Origin of the image.
            - "direction" (list or np.ndarray): Flattened or matrix-form direction.
            - "shear" : optional shear parameters.
            - "affine" : optional (ndim+1, ndim+1) affine matrix in SITK ordering.
              If provided, overrides ``spacing``, ``origin``, ``direction``, and ``shear``.
    """
    ndim = data.ndim
    metadata = metadata if metadata is not None else {}
    if "affine" in metadata:
        affine = metadata["affine"]
    else:
        affine = build_affine(
            ndim,
            metadata.get("spacing"),
            metadata.get("origin"),
            metadata.get("direction"),
            metadata.get("shear"),
        )
    if ndim == 3:
        data = data.transpose(2, 1, 0)
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  # swap
    elif ndim == 2:
        data = data.transpose(1, 0)
        P = np.array([[0, 1], [1, 0]], dtype=float)  # swap
    else:
        raise ValueError(f"Expected 2D or 3D volume, got shape {data.shape}")

    affine_nib = np.eye(4)
    affine_nib[:ndim, :ndim] = P @ affine[:ndim, :ndim] @ P.T
    affine_nib[:ndim, -1] = P @ affine[:ndim, ndim]
    affine_nib = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_nib

    image_nib = nib.Nifti1Image(data, affine=affine_nib)
    # Reorient
    if "affine_original" in metadata:
        # affine_original = metadata.get("affine_original",np.eye(4))
        affine_original = metadata["affine_original"]
        img_ornt = io_orientation(affine_original)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        image_nib = image_nib.as_reoriented(from_canonical)

    nib.save(image_nib, str(file))
    return [str(file)]


@register_loader("image", ".nii.gz", ".nii", backend="nibabelRO")
@register_loader("mask", ".nii.gz", ".nii", backend="nibabelRO")
def load_nibRO(file: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a NIfTI image with nibabel and return the data plus **SITK-style** metadata,
    performing reorientation to a consistent voxel order.

    This performs three things to match your SimpleITK conventions:
      1) **Axis order:** converts arrays from nibabel's (x, y, z) to (z, y, x),
         or from (x, y) to (y, x) for 2D.
      2) **World convention:** converts the affine from RAS+ (nibabel) to LPS (ITK/SITK)
         by flipping X and Y in world space.
      3) **Affine shape:** returns a homogeneous affine of shape (ndim+1, ndim+1).

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
    image = nib.load(file)
    affine = image.affine

    image_ro = image.as_reoriented(io_orientation(affine))
    affine_ro = image_ro.affine

    array = image_ro.get_fdata()
    ndim = array.ndim
    affine_sitk = np.eye(ndim + 1)

    # (x, y, z) -> (z, y, x) or (x,y) -> (y,x)
    if ndim == 3:
        array = array.transpose(2, 1, 0)
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  # swap
    elif ndim == 2:
        array = array.transpose(1, 0)
        P = np.array([[0, 1], [1, 0]], dtype=float)  # swap
    else:
        raise ValueError(f"Expected 2D or 3D volume, got shape {array.shape}")
    # --- Convert affine: RAS+ -> LPS, then reorder axes to match array ---
    affine_ro = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_ro  # world flip X,Y
    # linear part: (ndim x ndim) block; translation: last column
    affine_sitk[:ndim, :ndim] = P @ affine_ro[:ndim, :ndim] @ P.T
    affine_sitk[:ndim, ndim] = P @ affine_ro[:ndim, 3]

    # --- Decompose to spacing/origin/direction (napari-style) ---
    scale, translate, R, shear = decompose_affine(affine_sitk)

    metadata = {
        "spacing": scale,
        "origin": translate,
        "direction": R,
        "shear": shear,
        "affine": affine_sitk,
        "affine_original": affine,
    }
    return array, metadata
