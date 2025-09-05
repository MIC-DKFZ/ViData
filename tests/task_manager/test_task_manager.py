import numpy as np
import pytest

from vidata.task_manager.multilabel_segmentation_manager import MultiLabelSegmentationManager
from vidata.task_manager.semantic_segmentation_manager import SemanticSegmentationManager


def dummy_sem_seg_data(size, num_classes):
    return np.random.randint(0, num_classes + 1, size=size, dtype=np.uint8)


@pytest.mark.parametrize(
    "size",
    [
        (100, 100, 100),
        (500, 500),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 8])
def test_sem_seg(size, num_classes):
    tmanager = SemanticSegmentationManager()

    # Empty Data
    data = tmanager.empty(size, num_classes)
    assert not np.any(data)
    assert data.shape == size

    cid = tmanager.class_ids(data)
    assert cid.tolist() == [0]

    ccnt = tmanager.class_count(data, 0)
    assert ccnt == np.prod(size)

    cloc = tmanager.class_location(data, 0)
    assert len(cloc) == len(data.shape)
    for cl in cloc:
        assert len(cl) == np.prod(size)

    cloc = tmanager.class_location(data, 1)
    assert len(cloc) == len(data.shape)
    for cl in cloc:
        assert cl.tolist() == []

    # Random Data
    data = tmanager.random(size, num_classes)
    assert data.shape == size

    cid = tmanager.class_ids(data)
    assert 0 <= len(cid) <= num_classes

    ccnt = tmanager.class_count(data, 1)
    assert 0 <= ccnt <= np.prod(size)

    ccnt = [tmanager.class_count(data, ci) for ci in cid]
    assert sum(ccnt) == np.prod(size)

    cloc = tmanager.class_location(data, 1)
    assert len(cloc) == len(data.shape)


@pytest.mark.parametrize(
    "size",
    [
        (100, 100, 100),
        (500, 500),
    ],
)
@pytest.mark.parametrize("num_classes", [2, 8])
def test_multi_seg(size, num_classes):
    tmanager = MultiLabelSegmentationManager()

    # Empty Data
    data = tmanager.empty(size, num_classes)
    assert not np.any(data)
    assert data.shape == (num_classes, *size)

    cid = tmanager.class_ids(data)
    assert cid.tolist() == []
    #
    for cn in range(num_classes):
        ccnt = tmanager.class_count(data, cn)
        assert ccnt == 0

    cloc = tmanager.class_location(data, 0)
    assert len(cloc) == len(data.shape) - 1
    for cl in cloc:
        assert cl.tolist() == []

    # Random Data
    data = tmanager.random(size, num_classes)
    assert data.shape == (num_classes, *size)
    #
    cid = tmanager.class_ids(data)
    assert 0 <= len(cid) <= num_classes
    #
    ccnt = tmanager.class_count(data, 1)
    assert 0 <= ccnt <= np.prod(size)
    #
    ccnt = [tmanager.class_count(data, ci) for ci in range(num_classes)]
    for cc in ccnt:
        assert 0 <= cc <= np.prod(size)
    #
    cloc = tmanager.class_location(data, 1)
    assert len(cloc) == len(data.shape) - 1
