from vidata.loaders.base_loader import BaseLoader


class SemSegLoader(BaseLoader):
    """Loader for single semantic segmentation mask files."""

    role: str = "mask"
