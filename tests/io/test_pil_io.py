from pathlib import Path

import numpy as np

from vidata.loaders import ImageLoader, SemSegLoader
from vidata.writers import ImageWriter, SemSegWriter


def test_pil_rgb_roundtrip_function_api(tmp_path):
    from vidata.io import load_pilRGB, save_pilRGB

    file = Path(tmp_path).joinpath("rgb.png")
    data = np.random.randint(0, 256, size=(37, 23, 3), dtype=np.uint8)

    save_pilRGB(data, file)
    loaded, metadata = load_pilRGB(file)

    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)
    assert metadata == {}


def test_pil_mask_roundtrip_loader_writer_api(tmp_path):
    file = Path(tmp_path).joinpath("mask.png")
    data = np.random.randint(0, 8, size=(41, 27), dtype=np.uint8)

    writer = SemSegWriter(ftype=".png", backend="pil")
    loader = SemSegLoader(ftype=".png", backend="pil")

    writer.save(data, file)
    loaded, metadata = loader.load(file)

    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)
    assert metadata == {}


def test_pil_image_roundtrip_loader_writer_api(tmp_path):
    file = Path(tmp_path).joinpath("image.jpg")
    data = np.random.randint(0, 256, size=(31, 29, 3), dtype=np.uint8)

    writer = ImageWriter(ftype=".jpg", backend="pilRGB")
    loader = ImageLoader(ftype=".jpg", backend="pilRGB")

    writer.save(data, file)
    loaded, metadata = loader.load(file)

    assert loaded.shape == data.shape
    assert metadata == {}
