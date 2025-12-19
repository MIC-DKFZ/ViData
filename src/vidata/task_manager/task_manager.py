from abc import ABC, abstractmethod

import numpy as np


class TaskManager(ABC):
    """
    Abstract base class for task-specific operations on target data.
    """

    @staticmethod
    @abstractmethod
    def random(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        """Generate a random label array with given shape and number of classes."""

    @staticmethod
    @abstractmethod
    def empty(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        """Return an empty label array (typically all zeros or a sentinel class)."""

    @staticmethod
    @abstractmethod
    def class_ids(
        data: np.ndarray, return_counts: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return a sorted array of unique class IDs present in the data."""

    @staticmethod
    @abstractmethod
    def class_count(data: np.ndarray, class_id: int) -> int:
        """Return the count of pixels/voxels that belong to a given class ID."""

    @staticmethod
    @abstractmethod
    def class_location(
        data: np.ndarray, class_id: int, return_mask: bool = False
    ) -> tuple[np.ndarray, ...] | np.ndarray:
        """Return the indices where the given class ID occurs."""

    @staticmethod
    @abstractmethod
    def spatial_dims(shape: np.ndarray) -> np.ndarray:
        """Return the spatial dimensions of the given shape."""

    @staticmethod
    @abstractmethod
    def has_background():
        """if the task has a dedicated background class --> is class 0 the bg class?"""
