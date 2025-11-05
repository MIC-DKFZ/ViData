from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vidata.registry import register_loader, register_writer


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_loader("mask", ".png", ".bmp", backend="imageio")
def load_image(file: str | Path):
    data = iio.imread(file)  # automatically handles RGB, grayscale, masks
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageioRGB")
@register_writer("mask", ".png", ".bmp", backend="imageio")
def save_image(data: np.ndarray, file: str | Path) -> list[str]:
    iio.imwrite(file, data)
    return [str(file)]


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageioRGB")
def load_imageRGB(file: str | Path):
    data = iio.imread(file)  # automatically handles RGB, grayscale, masks
    if data.ndim == 2:
        # Grayscale → RGB
        data = np.stack([data] * 3, axis=-1)
    return data, {}
