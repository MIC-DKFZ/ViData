import numpy as np

from vidata.task_manager.task_manager import TaskManager


class SemanticSegmentationManager(TaskManager):
    @staticmethod
    def random(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        return np.random.randint(0, num_classes, size=size, dtype=np.uint8)

    @staticmethod
    def empty(size: tuple[int, ...], num_classes: int) -> np.ndarray:
        return np.zeros(size, dtype=np.uint8)

    @staticmethod
    def class_ids(data: np.ndarray) -> np.ndarray:
        return np.unique(data)

    @staticmethod
    def class_count(data: np.ndarray, class_id: int) -> int:
        return int(np.sum(data == class_id))

    @staticmethod
    def class_location(data: np.ndarray, class_id: int) -> tuple[np.ndarray, ...]:
        return np.where(data == class_id)

    @staticmethod
    def spatial_dims(shape: np.ndarray) -> np.ndarray:
        """Return the spatial dimensions of the given shape."""
        return shape

    @staticmethod
    def has_background():
        """if the task has a dedicated background class --> is class 0 the bg class?"""
        return True
