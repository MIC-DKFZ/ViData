from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal

Target = Literal["image", "mask"]

dict[Target, dict[str, dict[str, Callable]]]

LOADER_REGISTRY: dict[str, dict[str, dict[str, Callable[..., Any]]]] = defaultdict(
    lambda: defaultdict(dict)
)
WRITER_REGISTRY: dict[str, dict[str, dict[str, Callable[..., Any]]]] = defaultdict(
    lambda: defaultdict(dict)
)


def register_loader(target: Target, *dtypes: str, backend: str = "default") -> Callable:
    """
    Decorator to register a loader function for specific dtypes and target role.

    Example:
        @register_loader("image", ".nii.gz", ".nrrd")
        def load_image(...): ...
    """

    def decorator(func: Callable) -> Callable:
        for dtype in dtypes:
            LOADER_REGISTRY[target][dtype][backend] = func
        return func

    return decorator


def register_writer(
    target: Target,
    *dtypes: str,
    backend: str = "default",
) -> Callable[[Callable], Callable]:
    """
    Decorator to register a writer function for specific dtypes and target role.

    Example:
        @register_writer("image", ".png", ".tif")
        def save_image(...): ...
    """

    def decorator(func: Callable) -> Callable:
        for dtype in dtypes:
            WRITER_REGISTRY[target][dtype][backend] = func
        return func

    return decorator


# --- Trigger backend imports ---
# This must come LAST so the above decorators exist before data_io modules import them
import vidata.io  # noqa
