import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

from vidata.registry import WRITER_REGISTRY

PathLike = Union[str, Path]


class BaseWriter:
    """Base class for writers that save data to disk via a registered backend.

    Subclasses must define the class attribute ``role`` (e.g., ``"image"`` or ``"mask"``),
    which determines the writer registry slot to look up.

    Writers are registered per (role, file type) and may provide multiple backends
    (e.g., ``"nibabel"``, ``"sitk"``). If a backend is not specified, the first
    available backend for that (role, file type) is used.

    Args:
        ftype: File extension / dtype key registered for this writer (e.g., ``".nii.gz"``,
            ``".png"``).
        backend: Optional backend name to select among multiple registered backends for
            the given (role, file type). If the given backend is not available, a warning
            is emitted and the default backend is chosen.
        *args: Ignored by the base class, reserved for subclasses.
        **kwargs: Ignored by the base class, reserved for subclasses.

    Raises:
        NotImplementedError: If a subclass does not define ``role``.
        ValueError: If no writer is registered for ``(role, ftype)``.
    """

    role: str
    save_file: Callable[..., Any]

    def __init__(self, ftype: str, backend: str | None = None, *args, **kwargs):
        self.ftype = ftype
        if not hasattr(self, "role"):
            raise NotImplementedError("Subclasses must define `role`.")

        writers = WRITER_REGISTRY[self.role].get(self.ftype)
        if writers is None:
            raise ValueError(f"No writer registered for {self.role} type: '{self.ftype}'")

        if backend is not None and backend not in writers:
            warnings.warn(
                f"Writer Backend {backend} is not registered for {self.role} type: {self.ftype}, fallback to {list(writers.keys())[0]}",
                stacklevel=2,
            )
            backend = None

        self.save_file = writers[backend] if backend is not None else next(iter(writers.values()))

    def save(self, data: Any, file: PathLike, *args, **kwargs) -> None:
        """Save ``data`` to ``file`` using the selected backend writer.

        This method simply forwards all arguments to the concrete backend function
        selected from the registry during initialization.

        Args:
            data: The in-memory object to be written (e.g., ``np.ndarray``).
            file: Destination path.
            *args: Additional positional arguments forwarded to the backend writer.
            **kwargs: Additional keyword arguments forwarded to the backend writer.

        Returns:
            Backend-dependent return value (usually ``None``).
        """
        return self.save_file(data, file, *args, **kwargs)

    def __call__(self, data: Any, file: PathLike, *args, **kwargs) -> Any:
        """Allow the writer instance to be called like a function."""
        return self.save(data, file, *args, **kwargs)
