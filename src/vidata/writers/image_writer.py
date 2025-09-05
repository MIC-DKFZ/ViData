from pathlib import Path
from typing import Any, Union

from vidata.writers.base_writer import BaseWriter

PathLike = Union[str, Path]


class ImageWriter(BaseWriter):
    """Writer for single-file image outputs.

    This class picks a concrete writer function from the registry based on
    ``role='image'`` and the provided file type (e.g., ``'.nii.gz'`` or ``'.png'``).
    """

    role: str = "image"


class ImageStackWriter(BaseWriter):
    """Writer for channel-stacked images (one file per channel).

    Files are written as:
    ``{file}_{idx:0{zero_padding}d}{ftype}``, where ``file`` is a *base path without
    extension*, ``idx`` is the channel index, and ``ftype`` is the registered file type.

    Example:
        If ``file='out/scan'``, ``zero_padding=4`` and ``ftype='.nii.gz'``,
        channels 0..3 will be saved to:
        - ``out/scan_0000.nii.gz``
        - ``out/scan_0001.nii.gz``
        - ``out/scan_0002.nii.gz``
        - ``out/scan_0003.nii.gz``

    Args:
        ftype: File extension / dtype key registered in the writer registry
            (e.g., ``'.nii.gz'``, ``'.png'``).
        channels: Number of channels expected on the last axis of ``data``.
        backend: Optional backend name to select when multiple are registered.
        zero_padding: Width of the channel index zero-padding in filenames.
    """

    role: str = "image"

    def __init__(
        self, ftype: str, channels: int, backend: str | None = None, zero_padding: int = 4
    ):
        super().__init__(ftype, backend)
        self.channels = channels
        self.zero_padding = zero_padding

    def save(self, data: Any, file: PathLike, *args, **kwargs) -> None:
        files = [
            f"{file}_{str(i).zfill(self.zero_padding)}{self.ftype}" for i in range(self.channels)
        ]
        assert len(files) == data.shape[-1]  # number of channels shout match dimension of data
        for i, file in enumerate(files):
            self.save_file(data[..., i], file, *args, **kwargs)
