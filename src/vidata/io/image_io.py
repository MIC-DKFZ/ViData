import imageio.v3 as iio
import numpy as np

from vidata.registry import register_loader, register_writer


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_loader("mask", ".png", ".bmp", backend="imageio")
def load_image(file: str):
    data = iio.imread(file)  # automatically handles RGB, grayscale, masks
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_writer("mask", ".png", ".bmp", backend="imageio")
def save_image(data: np.ndarray, file: str) -> list[str]:
    iio.imwrite(file, data)
    return [file]
