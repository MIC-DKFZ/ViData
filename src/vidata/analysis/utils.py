from itertools import combinations

import numpy as np
from pandas import DataFrame


def get_occurrence_matrix(n_classes, class_occ, norm=True):
    co_mat = np.zeros((n_classes, n_classes), dtype=int)
    for row in class_occ:
        # include “self-pairs” so diagonal counts single-class occurrences
        for i, j in combinations(row, 2):
            if i in range(n_classes) and j in range(n_classes):
                co_mat[i, j] += 1
                co_mat[j, i] += 1  # keep symmetric
        # diagonal: count each class occurring onk its own in this image
        for i in row:
            if i in range(n_classes):
                co_mat[i, i] += 1
    if norm:
        diag = co_mat.diagonal()
        co_mat = np.divide(co_mat, diag, out=np.zeros_like(co_mat, dtype=float), where=diag != 0)
    return co_mat.T


def gather_shape_stats(spatial_dims: DataFrame):
    shape_counts = spatial_dims.value_counts()
    shapes = np.array(shape_counts.index.tolist())  # list of tuples
    counts = np.array(shape_counts.values)

    shape_arr = np.array(spatial_dims.tolist())
    shape_min = np.min(shape_arr, axis=0)
    shape_max = np.max(shape_arr, axis=0)
    shape_mean = np.mean(shape_arr, axis=0)
    shape_median = np.median(shape_arr, axis=0)
    return {
        "min": shape_min.tolist(),
        "max": shape_max.tolist(),
        "mean": np.round(shape_mean).astype(int).tolist(),
        "median": shape_median.tolist(),
        "unique": shapes.tolist(),
        "counts": counts.tolist(),
    }
