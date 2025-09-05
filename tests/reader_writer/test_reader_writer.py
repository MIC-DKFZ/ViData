from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from vidata.loaders import (
    ImageLoader,
    ImageStackLoader,
    MultilabelLoader,
    MultilabelStackedLoader,
    SemSegLoader,
)
from vidata.task_manager import (
    ImageManager,
    MultiLabelSegmentationManager,
    SemanticSegmentationManager,
)
from vidata.writers import (
    ImageStackWriter,
    ImageWriter,
    MultilabelStackedWriter,
    MultilabelWriter,
    SemSegWriter,
)


class FileSpec(NamedTuple):
    lossy: bool  # lossy compression
    float: bool  # supports float
    metadata: bool  # supports metadata
    mmapf: Callable | None  # function to unpack if mmap


FILETYPES = {
    ".png": FileSpec(False, False, False, None),
    ".jpg": FileSpec(True, False, False, None),
    ".jpeg": FileSpec(True, False, False, None),
    ".bmp": FileSpec(False, False, False, None),
    ".tif": FileSpec(False, True, False, None),
    ".tiff": FileSpec(False, True, False, None),
    ".nii.gz": FileSpec(False, True, True, None),
    ".nii": FileSpec(False, True, True, None),
    ".mha": FileSpec(False, True, True, None),
    ".nrrd": FileSpec(False, True, True, None),
    ".b2nd": FileSpec(False, True, True, lambda x: x[...]),
    ".npy": FileSpec(False, True, False, None),
    ".npz": FileSpec(False, True, False, lambda x: x["arr_0"]),
}


#######################################################################
# --- Image: Single Channel 2D Images [int|float] ---
#######################################################################
@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("size", [(100, 80)])
@pytest.mark.parametrize(
    "metadata",
    [None, {"spacing": [0.1, 0.2], "origin": [10, 20], "direction": [[1.0, 0], [0, -1.0]]}],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".png", "imageio"),
        (".jpg", "imageio"),
        (".bmp", "imageio"),
        (".jpeg", "imageio"),
        (".tif", "tifffile"),
        (".tiff", "tifffile"),
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_img_2d(dtype, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = ImageManager.random(size=size, dtype=dtype)
    loader = ImageLoader(ftype=file_type, backend=backend)
    writer = ImageWriter(ftype=file_type, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Image: RGB 2D Images ---
#######################################################################
@pytest.mark.parametrize("dtype", ["int"])
@pytest.mark.parametrize("size", [(100, 80, 3)])
@pytest.mark.parametrize(
    "metadata",
    [None, {"spacing": [0.1, 0.2], "origin": [10, 20], "direction": [[1.0, 0], [0, -1.0]]}],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".png", "imageio"),
        (".jpg", "imageio"),
        (".bmp", "imageio"),
        (".jpeg", "imageio"),
        (".tif", "tifffile"),
        (".tiff", "tifffile"),
    ],
)
def test_img_rgb(dtype, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = ImageManager.random(size=size, dtype=dtype)
    loader = ImageLoader(ftype=file_type, backend=backend)
    writer = ImageWriter(ftype=file_type, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Image: Single Channel 3D Images [int|float] ---
#######################################################################
@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_img_3d(dtype, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = ImageManager.random(size=size, dtype=dtype)
    loader = ImageLoader(ftype=file_type, backend=backend)
    writer = ImageWriter(ftype=file_type, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Image: Multi Channel ndim Images [int|float] ---
#######################################################################
@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_img_nd_multichannel(dtype, channels, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = ImageManager.random(size=(*size, channels), dtype=dtype)
    loader = ImageLoader(ftype=file_type, channels=channels, backend=backend)
    writer = ImageWriter(ftype=file_type, channels=channels, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Image: Multi Channel Stacked 2D Images [int|float] ---
#######################################################################
@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("size", [(100, 80)])
@pytest.mark.parametrize(
    "metadata",
    [None, {"spacing": [0.1, 0.2], "origin": [10, 20], "direction": [[1.0, 0], [0, -1.0]]}],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".png", "imageio"),
        (".jpg", "imageio"),
        (".bmp", "imageio"),
        (".jpeg", "imageio"),
        (".tif", "tifffile"),
        (".tiff", "tifffile"),
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
    ],
)
def test_img_2d_multichannel_stacked(dtype, channels, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath("test_")

    data = ImageManager.random(size=(*size, channels), dtype=dtype)
    loader = ImageStackLoader(ftype=file_type, channels=channels, backend=backend)
    writer = ImageStackWriter(ftype=file_type, channels=channels, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Image: Multi Channel Stacked 3D Images [int|float] ---
#######################################################################
@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
    ],
)
def test_img_3d_multichannel_stacked(dtype, channels, size, metadata, file_type, backend, tmp_path):
    if dtype == "float" and not FILETYPES[file_type].float:
        pytest.skip(f"{file_type} does not support float")

    file = Path(tmp_path).joinpath("test_")

    data = ImageManager.random(size=(*size, channels), dtype=dtype)
    loader = ImageStackLoader(ftype=file_type, channels=channels, backend=backend)
    writer = ImageStackWriter(ftype=file_type, channels=channels, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    if not FILETYPES[file_type].lossy:
        assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Semantic Segmentation: 2D Mask ---
#######################################################################
@pytest.mark.parametrize("size", [(100, 80)])
@pytest.mark.parametrize("classes", [7])
@pytest.mark.parametrize(
    "metadata",
    [None, {"spacing": [0.1, 0.2], "origin": [10, 20], "direction": [[1.0, 0], [0, -1.0]]}],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".png", "imageio"),
        (".bmp", "imageio"),
        (".tif", "tifffile"),
        (".tiff", "tifffile"),
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_semseg_2d(size, classes, metadata, file_type, backend, tmp_path):
    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = SemanticSegmentationManager.random(size=size, num_classes=classes)
    loader = SemSegLoader(ftype=file_type, backend=backend)
    writer = SemSegWriter(ftype=file_type, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- Semantic Segmentation: 3D Mask ---
#######################################################################
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize("classes", [7])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_semseg_3d(size, classes, metadata, file_type, backend, tmp_path):
    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = SemanticSegmentationManager.random(size=size, num_classes=classes)
    loader = SemSegLoader(ftype=file_type, backend=backend)
    writer = SemSegWriter(ftype=file_type, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- MultiLabel: Stacked 2D Masks [int] ---
#######################################################################
@pytest.mark.parametrize("classes", [1, 3])
@pytest.mark.parametrize("size", [(100, 80)])
@pytest.mark.parametrize(
    "metadata",
    [None, {"spacing": [0.1, 0.2], "origin": [10, 20], "direction": [[1.0, 0], [0, -1.0]]}],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".png", "imageio"),
        (".bmp", "imageio"),
        (".tif", "tifffile"),
        (".tiff", "tifffile"),
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
    ],
)
def test_multilabel_2d_stacked(classes, size, metadata, file_type, backend, tmp_path):
    file = Path(tmp_path).joinpath("test_")

    data = MultiLabelSegmentationManager.random(size=size, num_classes=classes)
    loader = MultilabelStackedLoader(ftype=file_type, num_classes=classes, backend=backend)
    writer = MultilabelStackedWriter(ftype=file_type, num_classes=classes, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- MultiLabel: Stacked 3D Masks [int] ---
#######################################################################
@pytest.mark.parametrize("classes", [1, 3])
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".nii.gz", "sitk"),
        (".nii.gz", "nibabel"),
        (".nrrd", "sitk"),
        (".nii", "sitk"),
        (".nii", "nibabel"),
        (".mha", "sitk"),
    ],
)
def test_multilabel_3d_stacked(classes, size, metadata, file_type, backend, tmp_path):
    file = Path(tmp_path).joinpath("test_")

    data = MultiLabelSegmentationManager.random(size=size, num_classes=classes)
    loader = MultilabelStackedLoader(ftype=file_type, num_classes=classes, backend=backend)
    writer = MultilabelStackedWriter(ftype=file_type, num_classes=classes, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))


#######################################################################
# --- MultiLabel: ndim Masks (one file) ---
#######################################################################
@pytest.mark.parametrize("classes", [1, 3])
@pytest.mark.parametrize("size", [(100, 80, 50)])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "spacing": [1, 0.5, 0.25],
            "origin": [10, 20, 30],
            "direction": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        },
    ],
)
@pytest.mark.parametrize(
    "file_type, backend",
    [
        (".b2nd", "blosc2"),
        (".b2nd", "blosc2pkl"),
        (".npy", None),
        (".npz", None),
    ],
)
def test_multilabel_ndim(classes, size, metadata, file_type, backend, tmp_path):
    file = Path(tmp_path).joinpath(f"test_{file_type}")

    data = MultiLabelSegmentationManager.random(size=size, num_classes=classes)
    loader = MultilabelLoader(ftype=file_type, num_classes=classes, backend=backend)
    writer = MultilabelWriter(ftype=file_type, num_classes=classes, backend=backend)

    args = {"data": data, "file": file}
    if FILETYPES[file_type].metadata:
        args["metadata"] = metadata
    if file_type == ".b2nd":
        args["patch_size"] = (20, 30) if len(size) == 2 else (20, 30, 40)

    writer.save(**args)

    data_l, metadata_l = loader.load(file=file)

    if FILETYPES[file_type].mmapf is not None:
        data_l = FILETYPES[file_type].mmapf(data_l)

    assert np.array_equal(data_l, data)

    if FILETYPES[file_type].metadata and metadata is not None:
        for key in metadata:
            assert np.allclose(np.array(metadata_l[key]), np.array(metadata[key]))
