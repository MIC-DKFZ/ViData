from typing import Any

import numpy as np

from vidata.loaders.base_loader import BaseLoader


class ImageLoader(BaseLoader):
    """Loader for single image files."""

    role: str = "image"


class ImageStackLoader(BaseLoader):
    """Loader for stacked multichannel image data.

    Loads multiple single-channel image files and stacks them into a
    multichannel array along the last axis.
    """

    role: str = "image"

    def __init__(
        self, ftype: str, channels: int, backend: str | None = None, zero_padding: int = 4
    ) -> None:
        super().__init__(ftype, backend)
        self.channels = channels
        self.zero_padding = zero_padding

    def load(self, file: str) -> tuple[Any, dict[str, Any]]:
        """Load a multichannel image stack.

        Args:
            file (str): Base file path without channel index or extension.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - A NumPy array containing the stacked image data with shape
                  `(H, W, channels)` or similar, depending on the file content.
                - A metadata dictionary from the first channel file.
        """
        files = [
            f"{file}_{str(i).zfill(self.zero_padding)}{self.ftype}" for i in range(self.channels)
        ]

        data = [self.load_file(f) for f in files]
        image, meta = zip(*data, strict=False)

        return np.stack(image, axis=-1), meta[0]
