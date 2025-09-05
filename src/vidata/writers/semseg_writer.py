from vidata.writers.base_writer import BaseWriter


class SemSegWriter(BaseWriter):
    """Writer for semantic segmentation masks."""

    role: str = "mask"
