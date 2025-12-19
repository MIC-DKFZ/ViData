import numpy as np

from vidata.registry import register_task
from vidata.task_manager.task_manager import TaskManager


@register_task("multilabel")
class MultiLabelSegmentationManager(TaskManager):
    @staticmethod
    def random(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        return np.random.randint(0, 2, size=(num_classes, *size), dtype=np.uint8)

    @staticmethod
    def empty(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        return np.zeros((num_classes, *size), dtype=np.uint8)

    @staticmethod
    def class_ids(
        data: np.ndarray, return_counts: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return class indices that are present in the mask (i.e., where at least one pixel is non-zero).
        """
        class_ids = np.flatnonzero(data.reshape(data.shape[0], -1).any(axis=1))

        if return_counts:
            axes = tuple(range(1, data.ndim))
            counts = np.count_nonzero(data, axis=axes)
            return class_ids, counts[class_ids]

        return class_ids

    @staticmethod
    def class_count(data: np.ndarray, class_id: int) -> int:
        """
        Return number of pixels/voxels labeled for the given class.
        """
        return int(np.sum(data[class_id]))

    @staticmethod
    def class_location(
        data: np.ndarray, class_id: int, return_mask: bool = False
    ) -> tuple[np.ndarray, ...] | np.ndarray:
        """
        Return indices where the given class is active (non-zero).
        """
        if return_mask:
            return data[class_id]
        return np.where(data[class_id] > 0)

    @staticmethod
    def spatial_dims(shape: np.ndarray) -> np.ndarray:
        """Return the spatial dimensions of the given shape."""
        return shape[1:]

    @staticmethod
    def has_background():
        """if the task has a dedicated background class --> is class 0 the bg class?"""
        return False
