import warnings
from collections.abc import Callable
from typing import Any

from vidata.registry import LOADER_REGISTRY


class BaseLoader:
    """Base class for file loaders.

    This class provides a common interface for loading files of a specific role and type.
    Subclasses must define the `role` attribute and register their loaders in `LOADER_REGISTRY`.

    Attributes:
        role (str): The data role associated with the loader (e.g., `"image"`, `"mask"`).
        load_file (Callable[[str], Any]): A function that loads a file from a given path.
        ftype (str): The file type/extension handled by this loader (e.g., `".png"`, `".nii.gz"`).
    """

    role: str
    load_file: Callable[[str], Any]

    def __init__(self, ftype: str, backend: str | None = None, *args, **kwargs):
        """Initialize a base loader.

        Args:
            ftype (str): File type/extension handled by the loader.
            backend (Optional[str], optional): Specific backend to use for loading.
                If `None`, the first registered backend is used.
            *args: Additional positional arguments for subclass initialization.
            **kwargs: Additional keyword arguments for subclass initialization.

        Raises:
            NotImplementedError: If the subclass does not define the `role` attribute.
            ValueError: If no loader is registered for the given role and file type.
        """
        self.ftype = ftype
        if not hasattr(self, "role"):
            raise NotImplementedError("Subclasses must define `role`.")

        loaders = LOADER_REGISTRY[self.role].get(self.ftype)
        if loaders is None:
            raise ValueError(f"No loader registered for {self.role} type: '{self.ftype}'")

        if backend is not None and backend not in loaders:
            warnings.warn(
                f"Loader Backend {backend} is not registered for {self.role} type: {self.ftype}, fallback to {list(loaders.keys())[0]}",
                stacklevel=2,
            )
            backend = None

        self.load_file = loaders[backend] if backend is not None else next(iter(loaders.values()))

    def load(self, file: str) -> tuple[Any, dict[str, Any]]:
        """Load a file.

        Args:
            file (str): Path to the file to load.

        Returns:
            tuple[Any, dict]: The loaded file data and metadata as dict.
        """
        return self.load_file(file)

    def __call__(self, file: str) -> tuple[Any, dict[str, Any]]:
        """Allow the loader instance to be called like a function."""
        return self.load(file)
