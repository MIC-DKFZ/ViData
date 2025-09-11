from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike


def combine_scale(s1: ArrayLike, s2: ArrayLike) -> np.ndarray:
    """
    Elementwise combine scaling vectors.
    s_total = s2 * s1  (order doesn't matter)
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    if s1.shape != s2.shape:
        raise ValueError(f"scale shapes must match, got {s1.shape} vs {s2.shape}")
    return s2 * s1


def combine_translation(t1: ArrayLike, t2: ArrayLike) -> np.ndarray:
    """
    Add translations.
    t_total = t2 + t1  (order doesn't matter for pure translations)
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    if t1.shape != t2.shape:
        raise ValueError(f"translation shapes must match, got {t1.shape} vs {t2.shape}")
    return t2 + t1


def combine_rotation(R1: ArrayLike, R2: ArrayLike) -> np.ndarray:
    """
    Compose direction/rotation matrices.
    R_total = R2 @ R1  (apply R1, then R2)
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    if R1.shape != R2.shape or R1.ndim != 2 or R1.shape[0] != R1.shape[1]:
        raise ValueError(
            f"rotation matrices must be same square shape, got {R1.shape} vs {R2.shape}"
        )
    return R2 @ R1


def combine_shear(H1: ArrayLike, H2: ArrayLike) -> np.ndarray:
    """
    Compose shear matrices.
    H_total = H2 @ H1  (apply H1, then H2)
    """

    def combine_shear_vectors_2d(s1, s2):
        """2D shear: vectors are length-1: (s_xy,). H_total = H2 @ H1."""
        s1 = float(np.asarray(s1).ravel()[0])
        s2 = float(np.asarray(s2).ravel()[0])
        return np.array([s2 + s1], dtype=float)

    def combine_shear_vectors_3d(s1, s2):
        """3D shear: vectors ordered (s_xy, s_xz, s_yz). H_total = H2 @ H1."""
        s1 = np.asarray(s1, dtype=float).ravel()
        s2 = np.asarray(s2, dtype=float).ravel()
        if s1.size != 3 or s2.size != 3:
            raise ValueError("3D shear vectors must have length 3: (s_xy, s_xz, s_yz).")

        a1, b1, c1 = s1  # (xy, xz, yz) for first shear
        a2, b2, c2 = s2  # (xy, xz, yz) for second shear

        # totals when H_total = H2 @ H1
        a = a2 + a1  # s_xy
        c = c2 + c1  # s_yz
        b = b2 + b1 + a2 * c1  # s_xz (cross-term!)
        return np.array([a, b, c], dtype=float)

    s1 = np.asarray(H1).ravel()
    if s1.size == 1:
        return combine_shear_vectors_2d(H1, H2)
    if s1.size == 3:
        return combine_shear_vectors_3d(H1, H2)
    raise ValueError("Only 2D (len=1) and 3D (len=3) shear vectors are supported here.")


def decompose_affine(
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose an affine transformation matrix into scale, translation, rotation, and shear.

    This function uses QR decomposition (`np.linalg.qr`) on the linear part
    of the affine to obtain:
        - An orthonormal rotation matrix.
        - An upper-triangular matrix encoding scale and shear.

    Steps:
    1. Extract the linear transform (top-left block) and translation vector (last column).
    2. Apply QR decomposition to get rotation and upper-triangular factors.
    3. Normalize so that scale values are non-negative, absorbing signs into `rotate`.
    4. Extract shear values from the strictly upper-triangular part of the normalized matrix.

    Args:
        affine (np.ndarray): An (N+1, N+1) affine transformation matrix.

    Returns:
        tuple:
            - scale (np.ndarray): Shape (N,), positive scale factors.
            - translate (np.ndarray): Shape (N,), translation vector.
            - rotate (np.ndarray): Shape (N, N), orthonormal rotation matrix.
            - shear (np.ndarray): Shape (N*(N-1)//2,), upper-triangular shear values.

    Notes:
        - This matches the napari-style decomposition order: rotate @ shear @ scale.
        - Shear values follow the order of flattening the strictly upper-triangular
          elements row-wise.
        - For exact inverses, `build_affine` must use the same shear convention.
    """
    ndims = affine.shape[0] - 1
    matrix = affine[:ndims, :ndims]
    translate = affine[:ndims, -1].copy()

    n = matrix.shape[0]

    # Upper Triangular
    # rotate, tri = scipy.linalg.qr(matrix)
    rotate, tri = np.linalg.qr(matrix)

    scale_with_sign = np.diag(tri).copy()
    scale = np.abs(scale_with_sign)
    normalize = scale / scale_with_sign

    tri *= normalize.reshape((-1, 1))
    rotate *= normalize

    # Take any reflection into account
    tri_normalized = tri @ np.linalg.inv(np.diag(scale))

    # Upper Triangular
    shear = tri_normalized[np.triu(np.ones((n, n)), 1).astype(bool)]

    return scale, translate, rotate, shear


def build_affine(
    ndims: int,
    scale: np.ndarray | None = None,
    translate: np.ndarray | None = None,
    rotate: np.ndarray | None = None,
    shear: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build an affine transformation matrix from scale, translation, rotation, and shear.

    Args:
        ndims (int): Number of spatial dimensions.
        scale (np.ndarray, optional): Shape (N,), voxel spacing or scaling factors.
        translate (np.ndarray, optional): Shape (N,), translation vector.
        rotate (np.ndarray, optional): Shape (N, N), rotation matrix.
        shear (np.ndarray, optional): Shape (N*(N-1)//2,), Upper-triangular shear parameter.

    Returns:
        np.ndarray: (N+1, N+1) affine transformation matrix.

    Notes:
        scale ~ spacing
        translate ~ origin
        rotate ~ direction

    """
    # Replace with default transformation if not given
    scale = np.ones(ndims) if scale is None else scale
    translate = np.ones(ndims) if translate is None else translate
    rotate = np.eye(ndims) if rotate is None else rotate
    shear = np.zeros(ndims) if shear is None else shear

    # Scaling
    scale_matrix = np.diag(scale)

    # Shearing
    shear_matrix = np.eye(ndims)
    upper_triangle_indices = list(combinations(range(ndims), 2))  # (i, j) pairs
    for shear_value, (i, j) in zip(shear, upper_triangle_indices, strict=False):
        shear_matrix[i, j] = shear_value  # Shear in one direction
        # shear_matrix[j, i] = shear_value  # Ensure symmetry for 2D and above

    # Build affine
    affine = np.eye(ndims + 1)
    # compose linear part (napari order)
    # affine[:ndims, :ndims] = rotate @ shear_matrix @ scale_matrix
    affine[:ndims, :ndims] = rotate @ scale_matrix @ shear_matrix
    affine[:ndims, ndims] = translate

    return affine
