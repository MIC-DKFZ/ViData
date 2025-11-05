from pathlib import Path

import cv2
import numpy as np

from vidata.registry import register_loader, register_writer

cv2.setNumThreads(0)


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="cv2")
@register_loader("mask", ".png", ".bmp", backend="cv2")
def load_cv2(file: str | Path):
    data = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="cv2")
@register_writer("mask", ".png", ".bmp", backend="cv2")
def save_cv2(data: np.ndarray, file: str | Path) -> list[str]:
    cv2.imwrite(file, data)
    return [str(file)]


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="cv2RGB")
def save_cv2RGB(data: np.ndarray, file: str | Path) -> list[str]:
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file, data)
    return [str(file)]


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="cv2RGB")
def load_cv2RGB(file: str | Path):
    data = cv2.imread(file, cv2.IMREAD_COLOR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    return data, {}
