import numpy as np

from vidata.registry import register_task


@register_task("image")
class ImageManager:
    @staticmethod
    def random(size: tuple[int, ...], dtype="float") -> np.ndarray:
        if dtype == "float":
            return np.random.rand(*size).astype(np.float32)
        if dtype == "int":
            return np.random.randint(0, 255, size=size, dtype=np.uint8)

    @staticmethod
    def empty(size: tuple[int, ...], dtype="float") -> np.ndarray:
        if dtype == "float":
            return np.zeros(*size, dtype=np.float32)
        if dtype == "int":
            return np.zeros(*size, dtype=np.uint8)
