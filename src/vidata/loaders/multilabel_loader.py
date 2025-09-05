from typing import Any

import numpy as np

from vidata.loaders.base_loader import BaseLoader


class MultilabelLoader(BaseLoader):
    """Loader for single multilabel mask files."""

    role: str = "mask"


class MultilabelStackedLoader(BaseLoader):
    """Loader for stacked multilabel mask data.

    Loads multiple single-class mask files and stacks them into a
    multi-class mask array along the first axis.
    """

    role: str = "mask"

    def __init__(
        self, ftype: str, num_classes: int, backend: str | None = None, zero_padding: int = 4
    ):
        super().__init__(ftype, backend)
        self.num_classes = num_classes
        self.zero_padding = zero_padding

    def load(self, file: str) -> tuple[Any, dict[str, Any]]:
        """Load a stacked multilabel mask.

        Args:
            file (str): Base file path without class index or extension.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - A NumPy array containing the stacked masks with shape
                  `(num_classes, H, W, ...)`.
                - A metadata dictionary from the first class file.
        """
        files = [
            f"{file}_{str(i).zfill(self.zero_padding)}{self.ftype}" for i in range(self.num_classes)
        ]

        data = [self.load_file(f) for f in files]
        image, meta = zip(*data, strict=False)

        return np.stack(image, axis=0), meta[0]
