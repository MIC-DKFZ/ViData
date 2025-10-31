from pathlib import Path

import numpy as np
import pytest


def dummy_data(size, dtype="float"):
    if dtype == "float":
        return np.random.rand(*size).astype(np.float32)
    elif dtype == "int":
        return np.random.randint(0, 255, size=size, dtype=np.uint8)


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("file_ending", [".nii.gz", ".mha", ".nrrd"])
@pytest.mark.parametrize(
    "size, metadata",
    [
        ((100, 100, 100), None),
        ((100, 100), None),
        (
            (110, 90, 100),
            {
                "spacing": [1, 0.5, 0.25],
                "origin": [10, 20, 30],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            },
        ),
        (
            (110, 90, 100),
            {
                "spacing": np.array([1, 0.5, 0.25]),
                "origin": np.array([10, 20, 30]),
                "direction": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            },
        ),
        (
            (110, 90),
            {
                "spacing": [1, 0.5],
                "origin": [10, 20],
                "direction": [[1.0, 0.0], [0.0, -1.0]],
            },
        ),
        ((120, 100, 80), {"spacing": [1, 0.5, 0.25]}),
        ((120, 100), {"spacing": [1, 0.5]}),
    ],
)
def test_sitk(dtype, file_ending, size, metadata, tmp_path):
    from vidata.io import load_sitk, save_sitk

    file = Path(tmp_path).joinpath(f"sitk_{dtype}{file_ending}")

    data = dummy_data(size, dtype)

    save_sitk(data, file, metadata)
    data_l, metadata_l = load_sitk(file)

    assert np.array_equal(data, data_l)
    if metadata is not None:
        for key in metadata:
            assert np.all(metadata_l[key] == metadata[key])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("file_ending", [".nii.gz", ".nii"])
@pytest.mark.parametrize(
    "size, metadata",
    [
        ((100, 100, 100), None),
        ((100, 100), None),
        (
            (110, 90, 100),
            {
                "spacing": [1, 0.5, 0.25],
                "origin": [10, 20, 30],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            },
        ),
        (
            (110, 90, 100),
            {
                "spacing": np.array([1, 0.5, 0.25]),
                "origin": np.array([10, 20, 30]),
                "direction": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            },
        ),
        (
            (110, 90),
            {
                "spacing": [1, 0.5],
                "origin": [10, 20],
                "direction": [[1.0, 0.0], [0.0, -1.0]],
            },
        ),
        ((120, 100, 80), {"spacing": [1, 0.5, 0.25]}),
        ((120, 100), {"spacing": [1, 0.5]}),
    ],
)
def test_nib(dtype, file_ending, size, metadata, tmp_path):
    from vidata.io import load_nib, save_nib

    file = Path(tmp_path).joinpath(f"sitk_{dtype}{file_ending}")

    data = dummy_data(size, dtype)

    save_nib(data, file, metadata)
    data_l, metadata_l = load_nib(file)

    assert np.array_equal(data, data_l)
    if metadata is not None:
        for key in metadata:
            assert np.all(metadata_l[key] == metadata[key])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("file_ending", [".nii.gz", ".nii"])
@pytest.mark.parametrize(
    "size, metadata",
    [
        (
            (110, 90, 100),
            {
                "spacing": [1, 0.5, 0.25],
                "origin": [10, 20, 30],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            },
        ),
        (
            (110, 90, 100),
            {
                "spacing": np.array([1, 0.5, 0.25]),
                "origin": np.array([10, 20, 30]),
                "direction": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            },
        ),
        (
            (120, 100, 80),
            {
                "spacing": [1, 0.5, 0.25],
                "direction": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            },
        ),
    ],
)
def test_nibRO(dtype, file_ending, size, metadata, tmp_path):
    from vidata.io import load_nibRO, save_nibRO

    file = Path(tmp_path).joinpath(f"sitk_{dtype}{file_ending}")

    data = dummy_data(size, dtype)

    save_nibRO(data, file, metadata)
    data_l, metadata_l = load_nibRO(file)
    print("D1", data)
    print("D2", data_l)
    print(metadata_l)
    assert np.array_equal(data, data_l)
    if metadata is not None:
        for key in metadata:
            assert np.all(metadata_l[key] == metadata[key])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize(
    "size, metadata",
    [
        ((100, 100, 100), None),
        ((100, 100), None),
        (
            (110, 90, 100),
            {
                "spacing": [1, 0.5, 0.25],
                "origin": [10, 20, 30],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            },
        ),
        (
            (110, 90),
            {
                "spacing": [1, 0.5],
                "origin": [10, 20],
                "direction": [[1.0, 0.0], [0.0, -1.0]],
            },
        ),
        ((120, 100, 80), {"spacing": [1, 0.5, 0.25]}),
        ((120, 100), {"spacing": [1, 0.5]}),
    ],
)
def test_blosc2(dtype, size, metadata, tmp_path):
    from vidata.io import load_blosc2, save_blosc2

    file = Path(tmp_path).joinpath("data.b2nd")

    data = dummy_data(size, dtype)

    save_blosc2(data, file, (80, 70, 60) if len(data) == 3 else (80, 70), metadata=metadata)
    data_l, metadata_l = load_blosc2(file)
    data_l = data_l[...]

    assert np.array_equal(data, data_l)
    if metadata is not None:
        for key in metadata:
            assert np.all(metadata_l[key] == metadata[key])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize(
    "size, metadata",
    [
        ((100, 100, 100), None),
        ((100, 100), None),
        (
            (110, 90, 100),
            {
                "spacing": [1, 0.5, 0.25],
                "origin": [10, 20, 30],
                "direction": [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
            },
        ),
        (
            (110, 90),
            {
                "spacing": [1, 0.5],
                "origin": [10, 20],
                "direction": [[1.0, 0.0], [0.0, -1.0]],
            },
        ),
        ((120, 100, 80), {"spacing": [1, 0.5, 0.25]}),
        ((120, 100), {"spacing": [1, 0.5]}),
    ],
)
def test_blosc2pkl(dtype, size, metadata, tmp_path):
    from vidata.io import load_blosc2pkl, save_blosc2pkl

    file = Path(tmp_path).joinpath("data.b2nd")

    data = dummy_data(size, dtype)

    save_blosc2pkl(data, file, (80, 70, 60), metadata=metadata)
    data_l, metadata_l = load_blosc2pkl(file)
    data_l = data_l[...]

    assert np.array_equal(data, data_l)
    if metadata is not None:
        for key in metadata:
            assert np.all(metadata_l[key] == metadata[key])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("file_ending", [".tif", ".tiff"])
@pytest.mark.parametrize(
    "size",
    [
        (100, 100, 100),
        (100, 100),
        (110, 90, 100),
        (110, 90),
    ],
)
def test_tiff(dtype, file_ending, size, tmp_path):
    from vidata.io import load_tif, save_tif

    file = Path(tmp_path).joinpath(f"tif_{dtype}{file_ending}")

    data = dummy_data(size, dtype)

    save_tif(data, file)
    data_l, _ = load_tif(file)

    assert np.array_equal(data, data_l)


@pytest.mark.parametrize("dtype", ["int"])
@pytest.mark.parametrize("file_ending", [".png", ".jpg"])
@pytest.mark.parametrize(
    "size",
    [
        ((100, 100)),
        ((100, 100, 3)),
    ],
)
def test_image(dtype, file_ending, size, tmp_path):
    from vidata.io import load_image, save_image

    file = Path(tmp_path).joinpath(f"image_{dtype}{file_ending}")

    data = dummy_data(size, dtype)
    save_image(data, file)
    data_l, _ = load_image(file)
    assert data.shape == data_l.shape
    if file_ending != ".jpg":  # jpg is lossy:(
        assert np.array_equal(data, data_l)


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize(
    "size",
    [
        (100, 100, 100),
        (100, 100),
        (110, 90, 100),
        (110, 90),
    ],
)
def test_numpy(dtype, size, tmp_path):
    from vidata.io import load_npy, load_npz, save_npy, save_npz

    file = str(Path(tmp_path).joinpath(f"numpy_{dtype}"))

    data = dummy_data(size, dtype)

    save_npy(data, file + ".npy")
    data_l, _ = load_npy(file + ".npy")
    assert np.array_equal(data, data_l)

    # compressed array
    save_npz(data, file + ".npz", False)
    data_l, _ = load_npz(file + ".npz")
    data_l = data_l["arr_0"]
    assert np.array_equal(data, data_l)

    save_npz(data, file + ".npz", True)
    data_l, _ = load_npz(file + ".npz")
    data_l = data_l["arr_0"]

    assert np.array_equal(data, data_l)
    # compressed with dict
    save_npz({"data": data}, file + ".npz", False)
    data_l, _ = load_npz(file + ".npz")
    data_l = data_l["data"]
    assert np.array_equal(data, data_l)

    save_npz({"data": data}, file + ".npz", True)
    data_l, _ = load_npz(file + ".npz")
    data_l = data_l["data"]
    assert np.array_equal(data, data_l)
