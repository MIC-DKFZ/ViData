from pathlib import Path

import numpy as np
from PIL import Image

from vidata.registry import register_loader, register_writer


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="pil")
@register_loader("mask", ".png", ".bmp", backend="pil")
def load_pil(file: str | Path):
    with Image.open(file) as image:
        data = np.asarray(image)
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="pil")
@register_writer("mask", ".png", ".bmp", backend="pil")
def save_pil(data: np.ndarray, file: str | Path) -> list[str]:
    image = Image.fromarray(data)
    image.save(file)
    return [str(file)]


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="pilRGB")
def load_pilRGB(file: str | Path):
    with Image.open(file) as image:
        data = np.asarray(image.convert("RGB"))
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="pilRGB")
def save_pilRGB(data: np.ndarray, file: str | Path) -> list[str]:
    image = Image.fromarray(data).convert("RGB")
    image.save(file)
    return [str(file)]
