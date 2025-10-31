"""
Credits go to FabianIsensee and Karol-G
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/dataloading/nnunet_dataset.py
https://gist.github.com/Karol-G/68bc6cf6cbe4f1a21993a988533be0b3
"""

import math
from copy import deepcopy
from pathlib import Path
from typing import Union

import blosc2
import numpy as np

from vidata.io.pickle_io import load_pickle, save_pickle
from vidata.registry import register_loader, register_writer


@register_writer("image", ".b2nd", backend="blosc2")
@register_writer("mask", ".b2nd", backend="blosc2")
def save_blosc2(
    data: np.ndarray,
    file: str | Path,
    patch_size: Union[tuple[int, int], tuple[int, int, int]] | None = None,
    clevel: int = 8,
    nthreads: int = 8,
    codec: blosc2.Codec = blosc2.Codec.ZSTD,
    metadata: dict | None = None,
) -> list[str]:
    """Saves a NumPy array to a Blosc2 file with specified compression parameters.

    Args:
        data (np.ndarray): The input array to compress and store.
        file (str): Output path for the Blosc2 file.
        patch_size (Union[tuple[int, int], tuple[int, int, int]]): Desired patch size for each dimension.
        clevel (int, optional): Compression level (0–9). Defaults to 8.
        nthreads (int, optional): Number of threads to use for compression. Defaults to 8.
        codec (blosc2.Codec, optional): Compression codec. Defaults to blosc2.Codec.ZSTD.
        metadata (Optional[dict], optional): Optional dictionary of metadata to attach. Defaults to None.
    """

    if patch_size is None:
        _is_float = np.issubdtype(data.dtype.type, np.floating)
        _is_2d = data.ndim == 2

        base_patch_size = (512 if _is_float else 1024) if _is_2d else (64 if _is_float else 96)
        patch_size = tuple([min(s, base_patch_size) for s in data.shape])

    blocks, chunks = comp_blosc2_params(data.shape, patch_size, data.itemsize)
    blosc2.set_nthreads(nthreads)
    blosc2.asarray(
        array=np.ascontiguousarray(data),
        urlpath=file,
        cparams={"codec": codec, "clevel": clevel, "nthreads": nthreads},
        chunks=chunks,
        blocks=blocks,
        mmap_mode="w+",
        meta=metadata,
    )
    return [str(file)]


@register_loader("image", ".b2nd", backend="blosc2")
@register_loader("mask", ".b2nd", backend="blosc2")
def load_blosc2(file: str | Path, nthreads: int = 1) -> tuple[blosc2.NDArray, dict]:
    """Reads a Blosc2 file and returns the data and metadata.

    Args:
        file (str): Path to the Blosc2 file.
        nthreads (int, optional): Number of threads to use for decompression. Defaults to 1.

    Returns:
        tuple[blosc2.NDArray, dict]: The compressed data array and associated metadata.
    """
    blosc2.set_nthreads(nthreads)
    data = blosc2.open(urlpath=file, mode="r", dparams={"nthreads": nthreads}, mmap_mode="r")
    metadata = dict(data.schunk.meta)
    del metadata["b2nd"]
    return data, metadata


@register_writer("image", ".b2nd", backend="blosc2pkl")
@register_writer("mask", ".b2nd", backend="blosc2pkl")
def save_blosc2pkl(
    data: np.ndarray,
    file: str | Path,
    patch_size: Union[tuple[int, int], tuple[int, int, int]] | None = None,
    clevel: int = 8,
    nthreads: int = 8,
    codec: blosc2.Codec = blosc2.Codec.ZSTD,
    metadata: dict | None = None,
):
    """Saves a NumPy array to a Blosc2 file with specified compression parameters.

    Args:
        data (np.ndarray): The input array to compress and store.
        file (str): Output path for the Blosc2 file.
        patch_size (Union[tuple[int, int], tuple[int, int, int]]): Desired patch size for each dimension.
        clevel (int, optional): Compression level (0–9). Defaults to 8.
        nthreads (int, optional): Number of threads to use for compression. Defaults to 8.
        codec (blosc2.Codec, optional): Compression codec. Defaults to blosc2.Codec.ZSTD.
        metadata (Optional[dict], optional): Optional dictionary of metadata to attach. Defaults to None.
    """
    save_blosc2(data, file, patch_size=patch_size, clevel=clevel, nthreads=nthreads, codec=codec)
    file_pkl = str(file).replace(".b2nd", ".pkl")
    save_pickle(metadata, file_pkl)
    return [file, file_pkl]


@register_loader("image", ".b2nd", backend="blosc2pkl")
@register_loader("mask", ".b2nd", backend="blosc2pkl")
def load_blosc2pkl(file: str | Path, nthreads: int = 1) -> tuple[blosc2.NDArray, dict]:
    """Reads a Blosc2 file and returns the data and metadata.

    Args:
        file (str): Path to the Blosc2 file.
        nthreads (int, optional): Number of threads to use for decompression. Defaults to 1.

    Returns:
        tuple[blosc2.NDArray, dict]: The compressed data array and associated metadata.
    """
    data, _ = load_blosc2(file, nthreads)
    file_pkl = str(file).replace(".b2nd", ".pkl")
    metadata = load_pickle(file_pkl)
    return data, metadata


def comp_blosc2_params(
    image_size: tuple[int, int, int, int],
    patch_size: Union[tuple[int, int], tuple[int, int, int]],
    bytes_per_pixel: int = 4,  # 4 byte are float32
    l1_cache_size_per_core_in_bytes: int = 32768,
    # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
    l3_cache_size_per_core_in_bytes: int = 1441792,
    # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
    safety_factor: float = 0.8,
    # we dont will the caches to the brim. 0.8 means we target 80% of the caches
):
    """
    Computes a recommended block and chunk size for saving arrays with Blosc v2.
    Blosc2 NDIM documentation:
    "Having a second partition allows for greater flexibility in fitting different partitions to different CPU cache levels.
    Typically, the first partition (also known as chunks) should be sized to fit within the L3 cache,
    while the second partition (also known as blocks) should be sized to fit within the L2 or L1 caches,
    depending on whether the priority is compression ratio or speed."
    (Source: https://www.blosc.org/posts/blosc2-ndim-intro/)
    Our approach is not fully optimized for this yet.
    Currently, we aim to fit the uncompressed block within the L1 cache, accepting that it might occasionally spill over into L2, which we consider acceptable.
    Note: This configuration is specifically optimized for nnU-Net data loading, where each read operation is performed by a single core, so multi-threading is not an option.
    The default cache values are based on an older Intel 4110 CPU with 32KB L1, 128KB L2, and 1408KB L3 cache per core.
    We haven't further optimized for modern CPUs with larger caches, as our data must still be compatible with the older systems.
    Args:
        image_size (tuple[int, int, int, int]): The size of the image.
        patch_size (Union[tuple[int, int], tuple[int, int, int]]): Patch size, containing the spatial dimensions (x, y) or (x, y, z).
        bytes_per_pixel (int, optional): Number of bytes per element. Defaults to 4 (for float32).
        l1_cache_size_per_core_in_bytes (int, optional): Size of the L1 cache per core in bytes. Defaults to 32768.
        l3_cache_size_per_core_in_bytes (int, optional): Size of the L3 cache exclusively accessible by each core in bytes. Defaults to 1441792.
        safety_factor (float, optional): Safety factor to avoid filling caches completely. Defaults to 0.8.
    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: Recommended block size and chunk size.
    """

    num_squeezes = 0

    if len(image_size) == 2:
        image_size = (1, 1, *image_size)
        num_squeezes = 2

    if len(image_size) == 3:
        image_size = (1, *image_size)
        num_squeezes = 1

    if len(image_size) != 4:
        raise RuntimeError("Image size must be 4D.")

    if not (len(patch_size) == 2 or len(patch_size) == 3):
        raise RuntimeError("Patch size must be 2D or 3D.")

    num_channels = image_size[0]
    if len(patch_size) == 2:
        patch_size = (1, *patch_size)
    patch_size = np.array(patch_size)
    block_size = np.array(
        (
            num_channels,
            *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size],
        )
    )

    # shrink the block size until it fits in L1
    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        # pick largest deviation from patch_size that is not 1
        axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        # now reduce that axis to the next lowest power of 2
        block_size[picked_axis + 1] = 2 ** (
            max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
        )
        block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

    block_size = np.array([min(i, j) for i, j in zip(image_size, block_size, strict=False)])

    # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
    # load data into L3
    # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
    chunk_size = deepcopy(block_size)
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        if patch_size[0] == 1 and all(
            i == j for i, j in zip(chunk_size[2:], image_size[2:], strict=False)
        ):
            break
        if all(i == j for i, j in zip(chunk_size, image_size, strict=False)):
            break
        # find axis that deviates from block_size the most
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while (
            chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
            or patch_size[picked_axis] == 1
        ):
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size, strict=False)]) > 1.5:
            # chunk size should not exceed patch size * 1.5 on average
            chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
            break
    # better safe than sorry
    chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size, strict=False)]

    block_size = block_size[num_squeezes:]
    chunk_size = chunk_size[num_squeezes:]

    return tuple(block_size), tuple(chunk_size)
