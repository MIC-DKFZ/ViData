# ViData: A Unified Toolkit for 2D & 3D Vision Data I/O, Management, and Processing

> A unified Python toolkit for managing, processing, and analyzing 2D and 3D vision data —
> from raw files to task-aware analytics.
> Designed to streamline computer vision and medical imaging workflows and data pipelines.
>
> - **Unified I/O:** Load and save 2D images, 3D volumes, n-dim arrays, and configs with consistent `load_xxx` / `save_xxx` API.
> - **Supported Formats:** PNG, JPG, TIFF, BMP, NIfTI, NRRD, MHA, NumPy, Blosc2, JSON, YAML, Pickle, Text.
> - **Loaders & Writers:** Task-aware data handling for images, semantic segmentation, and multilabel masks (single-file or stacked).
> - **File Management:** Collect files with patterns, filters, and split definitions.
> - **Task-Aware Handling:** Built-in support for semantic and multilabel segmentation masks with clear shape and axis conventions.
> - **Flexible Dataset Configs:** Define datasets in YAML with layers, modalities, labels, and optional splits/folds.
> - **Analysis & Inspection Tools:** CLI for dataset statistics and a Napari plugin for visual inspection.

# Quick Start

> From raw files to full dataset pipelines — in just a few lines.

```py
from vidata.io import load_image, save_image, load_sitk, save_sitk
from vidata.loaders import ImageLoader, SemSegLoader, MultilabelLoader
from vidata.writers import ImageWriter, SemSegWriter, MultilabelWriter
from vidata.file_manager import FileManager
from vidata import ConfigManager

# --- Raw IO (direct file access) ---
data, meta = load_image("file_in.png")      # Load a 2D image
save_image(data, "file_out.png", meta)      # Save a 2D image
data, meta = load_sitk("file_in.nii.gz")    # Load a 3D volume
save_sitk(data, "file_out.nii.gz", meta)    # Save a 3D volume
# Supports PNG, TIFF, NIfTI, NRRD, MHA, NumPy, Blosc2, JSON, YAML, Pickle, and more

# --- Manage, load, and save image data ---
img_fm = FileManager(path=".../images", file_type=".png") # Collect files (.png, .tif, .nii.gz, .b2nd, ...)
img_lo = ImageLoader(ftype=".png")                        # Define how to load data
img_wr = ImageWriter(ftype=".png")                        # Define how to save data
data, meta = img_lo.load(img_fm[0])                       # Load a collected file
img_wr.save(data, ".../out/file.png", meta)               # Save the processed data

# --- Manage, load, and save label data (semantic or multilabel) ---
lbl_fm = FileManager(path=".../labels", file_type=".nii.gz")
lbl_lo = SemSegLoader( ftype=".nii.gz", backend="nibabel")  # or: MultilabelLoader(".nii.gz")
lbl_wr = SemSegWriter( ftype=".nii.gz", backend="sitk")     # or: MultilabelWriter(ftype=".nii.gz")
data, meta = lbl_lo.load(lbl_fm[0])
lbl_wr.save(data, ".../out/file.nii.gz", meta)

# --- Build everything from a YAML config ---
cm = ConfigManager("path/to/my/dataset.yaml")    # Parse dataset config
img_layer = cm["MyImageLayer"]                   # Access image layer by user-defined layer name
lbl_layer = cm["MyLabelLayer"]                   # Access label layer by user-defined layer name

img_fm, img_lo, img_wr = img_layer.file_manager(), img_layer.data_loader(), img_layer.data_writer()
lbl_fm, lbl_lo, lbl_wr = lbl_layer.file_manager(), lbl_layer.data_loader(), lbl_layer.data_writer()
```

# Installation

```bash
pip install vidata
# or latest dev version
git clone https://github.com/MIC-DKFZ/vidata.git
cd vidata
pip install -e ./

# (optional - for visual inspections)
pip install napari-data-inspection
```

# Module Overview

| Module                                | Purpose                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`io`](#io)                           | Provides low-level reading and writing support for multiple file types.                                 |
| [`loader`](#loaderswriters)           | Wraps `io` with task-specific logic to load data in correct format.                                     |
| [`writers`](#loaderswriters)          | Wraps `io` with task-specific logic to write data in correct format.                                    |
| [`task_manager`](#task-manager)       | Encapsulates logic for handling semantic and multi-label segmentation data.                             |
| [`file_manager`](#file-manager)       | Handles file selection and organization using paths, patterns and splits.                               |
| [`config_manager`](#config-manager)   | Parses dataset configuration files and instantiates loaders, writers, file managers, and task managers. |
| [`data_analyzer`](#data-analysis)     | Computes and visualizes dataset statistics.                                                             |
| [`data_inspection`](#data-inspection) | Inspect your data (Requires napari_data_inspection to be installed)                                     |

# IO

**TL;DR;**

- Use `load_xxx` / `save_xxx` for all supported formats.
- **Images/arrays:** PNG/JPG/TIFF (`imageio`, `tifffile`), NIfTI/NRRD/MHA (`sitk`, `nibabel`), Blosc2, NumPy (`.npy`, `.npz`).
- **Configs/metadata:** JSON, YAML, Pickle, TXT.
- **Extendable** by custom _load_ and _save_ functions.
- Functions always follow the same pattern:

```python
data, meta = load_xxx("file.ext")
save_xxx(data, "out.ext", meta)
```

<details>
<summary> Expand for Full Details </summary>

### Imaging and Array Data

| Module      | Extension(s)                       | Backend(s)             | Notes                                                                 |
| ----------- | ---------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| `image_io`  | `.png`, `.jpg`, `.jpeg`, `.bmp`    | `imageio` `imageioRGB` | Standard 2D image formats (RGB ensures 3 color channels)              |
| `cv2_io`    | `.png`, `.jpg`, `.jpeg`, `.bmp`    | `cv2` `cv2RGB`         | Standard 2D image formats using opencv (RGB ensures 3 color channels) |
| `tif_io`    | `.tif`, `.tiff`                    | `tifffile`             | Multipage TIFF, high bit-depths supported                             |
| `sitk_io`   | `.nii.gz`, `.nii`, `.mha`, `.nrrd` | `sitk`                 | Medical image formats (3D volumes)                                    |
| `nib_io`    | `.nii.gz`, `.nii`,                 | `nibabel`              | Alternative medical imaging backend                                   |
| `nrrd_io`   | `.nrrd`                            | `nrrd`                 | Alternative backend for .nrrd files                                   |
| `blosc2_io` | `.b2nd`                            | `blosc2`,              | Compressed N-dimensional arrays.                                      |
| `blosc2_io` | `.b2nd`                            | `blosc2pkl`            | Compressed N-dimensional arrays with metadata in a separate pkl file. |
| `numpy_io`  | `.npy`                             | `numpy`                | Single NumPy array                                                    |
| `numpy_io`  | `.npz`                             | `numpy`                | Dictionary of arrays                                                  |

```py
from vidata.io import (
    load_image, save_image,
    load_imageRGB,
    load_cv2, save_cv2,
    load_cv2RGB, save_cv2RGB,
    load_tif, save_tif,
    load_sitk, save_sitk,
    load_nib, save_nib,
    load_nrrd, save_nrrd,
    load_blosc2, save_blosc2,
    load_blosc2pkl, save_blosc2pkl,
    load_npy, save_npy,
    load_npz, save_npz,
)

# Standard image formats (PNG, JPG, BMP, …)
img, meta = load_image("example.png")
save_image(img, "out.png", meta)

# TIFF (supports multipage / high bit depth)
img, meta = load_tif("example.tif")
save_tif(img, "out.tif", meta)

# Medical imaging (NIfTI, MHA, NRRD) with SimpleITK
vol, meta = load_sitk("example.nii.gz")
save_sitk(vol, "out_sitk.nii.gz", meta)

# Medical imaging with Nibabel
vol, meta = load_nib("example.nii.gz")
save_nib(vol, "out_nib.nii.gz", meta)

# Medical imaging with PyNRRD
vol, meta = load_nrrd("example.nrrd")
save_nrrd(vol, "out_nib.nrrd", meta)

# Blosc2 compressed array
arr, meta = load_blosc2("example.b2nd")
save_blosc2(arr, "out.b2nd", meta)

# Blosc2 compressed array but with metadata in a separate pickle file (with the same name)
arr, meta = load_blosc2pkl("example.b2nd")
save_blosc2pkl(arr, "out.b2nd", meta)

# NumPy arrays
arr, _ = load_npy("example.npy")
save_npy(arr, "out.npy")

arrs, _ = load_npz("example.npz")
save_npz(arrs, "out.npz")
```

### Configuration and Metadata

| Module      | Extension(s)    | Notes                 |
| ----------- | --------------- | --------------------- |
| `json_io`   | `.json`         | JSON metadata/configs |
| `yaml_io`   | `.yaml`, `.yml` | YAML metadata/configs |
| `pickle_io` | `.pkl`          | Python pickles        |
| `txt_io`    | `.txt`          | Plain text files      |

```py
from vidata.io import (
    load_json, save_json,
    load_yaml, save_yaml,
    load_pickle, save_pickle,
    load_txt, save_txt,
)

# JSON
obj = load_json("config.json")
save_json(obj, "out.json")

# YAML
obj = load_yaml("config.yaml")
save_yaml(obj, "out.yaml")

# Pickle (Python objects)
obj = load_pickle("config.pkl")
save_pickle(obj, "out.pkl")

# Plain text
obj = load_txt("config.txt")
save_txt(obj, "out.txt")
```

### Custom Load and Save Functions

- Register a reader / writer with a decorator.
- Reader must return (numpy_array, metadata_dict).
- Writer must return list\[str\] of the file(s) it wrote.
- Registration happens at import time—make sure this module is imported (e.g., from your package’s __init__.py).
- See [here](https://github.com/MIC-DKFZ/ViData/blob/main/src/vidata/io/image_io.py) for an example.

```py
# custom_io_template.py  — fill in the TODOs and import this module somewhere at startup.
import numpy as np
from typing import Tuple, Dict, List

# TODO: import your backend library (e.g., imageio, tifffile, nibabel, SimpleITK, ...)
# import imageio.v3 as iio

from vidata.registry import register_loader, register_writer

# --------------------------- READER ------------------------------------------
# Replace file extension and backend name to your custom function
@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")  # To Register Image Loading
@register_loader("mask", ".png", ".bmp", backend="imageio") # To Register Label Loading
def load_custom(file: str) -> tuple[np.ndarray, dict]:
    """
    Load a file and return (data, metadata).
    metadata can be empty or include keys like: spacing, origin, direction, shear, dtype, etc.
    """
    # data = iio.imread(file)  # example for imageio
    data = ...  # TODO: replace
    meta = {}   # TODO: replace
    return data, meta
    # return meta # for metadata files like json or yaml

# --------------------------- WRITER ------------------------------------------
# Replace file extension and backend name to your custom function
@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio") # To Register Image Writer
@register_writer("mask", ".png", ".bmp", backend="imageio") # To Register Label Writer
def save_custom(data: np.ndarray, file: str) -> list[str]:
    """
    Save array to `file`. Return all created paths (include sidecars if any).
    """
    # TODO: write using your backend
    # iio.imwrite(file, data)
    return [file]
```

</details>

# Loaders/Writers

**TL;DR;**

- Use `ImageLoader/Writer`, `SemSegLoader/Writer`, `MultilabelLoader/Writer` for **single-file** data.
- Use `ImageStackLoader/Writer`, `MultilabelStackedLoader/Writer` when **channels/classes are split across files** (`*_0000`, `*_0001`, …).
- Some formats support multiple backends (e.g., `.nii.gz` → `sitk`, `nibabel`).

```python
from vidata.loaders import ImageLoader
from vidata.writers import ImageWriter

# Minimal example: load an image and save it again
loader = ImageLoader(ftype=".png", channels=3)
writer = ImageWriter(ftype=".png")

image, meta = loader.load("example.png")
writer.save(image, "out.png", meta)
```

<details>
<summary> Expand for Full Details </summary>

## Single-file loaders/writers

Use these loaders when each image or label is stored in a **single file**.

```py
from vidata.loaders import ImageLoader, SemSegLoader, MultilabelLoader
from vidata.writers import ImageWriter, SemSegWriter, MultilabelWriter

# Image Loader/Writer
loader = ImageLoader(ftype=".png")
writer = ImageWriter(ftype=".png")
image, meta = loader.load("path/to/file.png")
writer.save(image,"path/to/output.png",meta)

# Semantic Segmentation Loader/Writer
loader = SemSegLoader(ftype=".png")
writer = SemSegWriter(ftype=".png")
mask, meta = loader.load("path/to/file.png")
writer.save(mask,"path/to/output.png",meta)

# Multilabel Segmentation Loader/Writer
loader = MultilabelLoader(ftype=".png")
writer = MultilabelWriter(ftype=".png")
mask, meta = loader.load("path/to/file.png")
writer.save(mask,"path/to/output.png",meta)
```

## Stacked loaders/writers

Use these loaders when each channel (image) or class (labels) is stored in a separate file.
Files must follow a numeric suffix convention: file_0000.png, file_0001.png, etc.
Only exists for Image and Multilabel Loaders

```py
from vidata.loaders import ImageStackLoader, MultilabelStackedLoader
from vidata.writers import ImageStackWriter, MultilabelStackedWriter

# Image Loader
#       expects multiple files with numeric suffixes: file_0000.png, file_0001.png, file_0002.png
loader = ImageStackLoader(ftype=".png", channels=3)           # channels=3 --> 3 files are expected
writer = ImageStackWriter(ftype=".png", channels=3)           # channels=3 --> 3 files are expected
image, meta = loader.load("path/to/file")                     # Pass only the base path without suffix and file_type
writer.save(image,"path/to/output.png",meta)                  # each channel will be saved in a separate file

# Multilabel Segmentation Loader
#       expects file names: file_0000.png,... , file_0006.png
loader = MultilabelStackedLoader(ftype=".png", classes=7)     # classes=7 --> 7 files are expected
writer = MultilabelStackedWriter(ftype=".png", classes=7)     # classes=7 --> 7 files are expected
mask, meta = loader.load("path/to/file")                      # Pass only the base path without suffix and file_type
writer.save(mask,"path/to/output.png",meta)                   # each class will be saved in a separate file
```

## Backend selection

Some file types support multiple backends. You can explicitly select one (Analogue for all Loaders and Writers):

```py
from vidata.loaders import ImageLoader

# Explicit backend (SimpleITK)
loader=ImageLoader(ftype=".nii.gz",channels=1, backend="sitk")
image, meta = loader.load("path/to/file.nii.gz")

# Explicit backend (Nibabel)
loader=ImageLoader(ftype=".nii.gz",channels=1, backend="nibabel")
image, meta = loader.load("path/to/file.nii.gz")

# If not specified the first one from the table below is chosen (SimpleITK in this case)
loader=ImageLoader(ftype=".nii.gz",channels=1)
image, meta = loader.load("path/to/file.nii.gz")
```

| Data Type      | Extension(s)                    | Available Backends    |
| -------------- | ------------------------------- | --------------------- |
| **Image/Mask** | `.png`, `.jpg`, `.jpeg`, `.bmp` | `imageio`             |
| **Image/Mask** | `.nii.gz`, `.mha`, `.nrrd`      | `sitk`, `nibabel`     |
| **Image/Mask** | `.tif`, `.tiff`                 | `tifffile`            |
| **Image/Mask** | `.b2nd`                         | `blosc2`, `blosc2pkl` |
| **Image/Mask** | `.npy`                          | `numpy`               |
| **Image/Mask** | `.npz`                          | `numpy`               |

</details>

# Shape and Axis Conventions

**TL;DR;**

- **Images** → (H, W, C) (2D) or (D, H, W, C) (3D).
- **Semantic masks** → (H, W) (2D) or (D, H, W) (3D).
- **Multilabel masks** → (N, H, W) (2D) or (N, D, H, W) (3D).
- **Axes**: Z=Depth, Y=Height, X=Width, C=Channels.

<details>
<summary> Expand for Full Details </summary>

> **Legend**: \
> `D`, `H`, `W`: Spatial Dims \
> `C`: Number of Channels \
> `N`: Number of Classes \
> `B`: Batch size

## Expected Data Shapes

| Data Type                         | Expected Shapes 2D      | Expected Shapes 3D            |
| --------------------------------- | ----------------------- | ----------------------------- |
| **Images**                        | `(H, W)` or `(H, W, C)` | `(D, H, W)` or `(D, H, W, C)` |
| **Semantic Segmentation Masks**   | `(H, W)`                | `(D, H, W)`                   |
| **Multilabel Segmentation Masks** | `(N, H, W)`             | `(N, D, H, W)`                |

## Axis Definition

| **Axis**       | **Spatial Dimension** | **Anatomical Plane** | **Direction** | **NumPy Axis**   | **SimpleITK Axis** | **PyTorch Axis**    |
| -------------- | --------------------- | -------------------- | ------------- | ---------------- | ------------------ | ------------------- |
| Z *(if 3D)*    | Depth `(D)`           | Axial                | Bottom ↔ Top  | 0                | 2                  | 1 *(after channel)* |
| Y              | Height `(H)`          | Coronal              | Back ↔ Front  | 1 *(or 0 in 2D)* | 1                  | 2                   |
| X              | Width `(W)`           | Sagittal             | Left ↔ Right  | 2 *(or 1 in 2D)* | 0                  | 3                   |
| C *(optional)* | Channel `(C)`         | -                    | -             | 3 *(if present)* | Not used           | 0                   |

## Batch Shape Conventions

| **Framework** | **2D Shape**   | **3D Shape**      |
| ------------- | -------------- | ----------------- |
| **NumPy**     | `(B, H, W, C)` | `(B, D, H, W, C)` |
|               | `(B, Y, X, C)` | `(B, Z, Y, X, C)` |
| **PyTorch**   | `(B, C, H, W)` | `(B, C, D, H, W)` |
|               | `(B, C, Y, X)` | `(B, C, Z, Y, X)` |

</details>

# Task Manager

**TL;DR;**

- Use `SemanticSegmentationManager` for **single-label masks** (each voxel has one class ID).
- Use `MultiLabelSegmentationManager` for **multi-label masks** (channel-first one-hot).
- Provides utilities to generate dummy masks, list class IDs, count pixels, and locate classes.

```python
from vidata.task_manager import SemanticSegmentationManager

tm = SemanticSegmentationManager()
mask = tm.random((128, 128), num_classes=3)
print(tm.class_ids(mask))  # -> e.g. [0, 1, 2]
```

<details>
<summary> Expand for Full Details </summary>

The task managers provide a unified interface for **semantic segmentation** and **multilabel segmentation** labels, with utilities to generate dummy data, inspect class distributions, and query spatial properties.

```py
from vidata.task_manager import (
    SemanticSegmentationManager,
    MultiLabelSegmentationManager,
)

# Semantic segmentation (2D example)
ssm = SemanticSegmentationManager()

mask = ssm.random(size, num_classes)       # shape: (H, W) or (D, H, W), dtype=int, range: [0:num_classes]
empty = ssm.empty(size, num_classes)       # zeros by default
ids = ssm.class_ids(mask)                  # sorted unique class IDs (e.g., [0,1,2,3,4])
count_bg = ssm.class_count(mask, 1)        # number of pixels/voxels with class 1
coords = ssm.class_location(mask, 3)       # indices (tuple of arrays) where class==3
spatial = ssm.spatial_dims(mask.shape)     # (H, W) or (D, H, W)
has_bg = ssm.has_background()              # True (class 0 is background)

# Multilabel segmentation (3D example)
mlm = MultiLabelSegmentationManager()
ml_mask = mlm.random(size, num_classes)    # shape: (N, H, W) or (N, D, H, W), channel-first one-hot
ml_empty = mlm.empty(size, num_classes)    # all zeros
ml_ids = mlm.class_ids(ml_mask)            # sorted unique class IDs (e.g., [0,1,2,3,4])
ml_count = mlm.class_count(ml_mask, 2)     # number of pixels/voxels where channel 2 is active
ml_coords = mlm.class_location(ml_mask, 6) # indices (tuple of arrays) where channel 6 is active
ml_spatial = mlm.spatial_dims(ml_mask.shape) # (H, W) or (D, H, W)
mlm.has_background()                       # False (class 0 is not background)
```

</details>

# File Manager

**TL;DR;**

- Use `FileManager` to collect **single-file samples** (`*.png`, `*.nii.gz`, …).
- Use `FileManagerStacked` when **samples are split across multiple files** (`*_0000`, `*_0001`, …).
- Supports `pattern`, `include_names`, and `exclude_names` filters.
- Returns a list-like object (`len()`, indexing).

```python
from vidata.file_manager import FileManager

fm = FileManager(path="data/images", file_type=".png")
print(len(fm), "files found")
print(fm[0])  # -> Path('data/images/example.png')
```

<details>
<summary> Expand for Full Details </summary>

The `file_manager` module collects files from a root directory with optional
**pattern**, **include**/**exclude** filters, and a variant for **stacked** files
(e.g., `*_0000.*`, `*_0001.*`).

```py
from vidata.file_manager import FileManager

# Single-file data (e.g., images or masks stored one-per-file)
fm = FileManager(
    path="data/images",
    file_type=".png",
    pattern="*_image",            # optional; glob-like,
    include_names=["case_","sample_"], # optional; keep only if any token is in file name
    exclude_names=["corrupt"],   # optional; drop if any token is in file name
)
print(len(fm), "files")           # > 30 files | how many files are found
print(fm[0])                      # > data/images/case_1_image.png
```

Use FileManagerStacked when each sample is stored across multiple files with a
numeric suffix (e.g., channels or classes): file_0000.ext, file_0001.ext, …

```py
from vidata.file_manager import FileManagerStacked

# Single-file data (e.g., images or masks stored one-per-file)
fm = FileManagerStacked(
    path="data/stacked_images",
    file_type=".png",
    pattern="*_image",            # optional; glob-like,
    include_names=["case_","sample_"], # optional; keep only if any token is in file name
    exclude_names=["corrupt"],   # optional; drop if any token is in file name
)
print(len(fm), "files")           # > 30 files | how many files are found (number of base path)
print(fm[0])                      # > data/stacked_images/case_1_image | gives the base path without suffix and dtype
```

</details>

# Config Template

**TL;DR;**

- Config is a YAML **template for datasets**: always starts with a `name`.
- Add one or more **layers** (e.g., `image`, `semseg`, `multilabel`) with fields like `path`, `file_type`, `channels`/`classes`.
- **Optional splits**: define `train/val/test` overrides or point to a `splits_file`.
- See [`examples/template.yaml`](examples/template.yaml) and/or run `vidata_template` for a template

<details>
<summary> Expand for Full Details </summary>

A config is a YAML file describing the dataset name, layers, and optional
split definitions.

```yaml
name: DatasetName

# Define an arbitrary number of layers
layers:
  - name: SomeImages          # unique layer name
    type: image               # 'image' for image data and for label data one of: 'semseg' | 'multilabel'
    path: some/path/to/data   # directory
    file_type: .png           # required extension incl. dot (e.g., .png | .nii.gz | .b2nd)
    pattern:                  # optional regex to filter files relative to 'path' without file_type
    backend:                  # optional backend to load/write the data (e.g., sitk | nib) for .nii.gz, etc.
    channels: 3               # number of channels, required for image
    file_stack: False         # True if each channel is a separate file: *_0000, *_0001, ...

  - name: SomeLabels
    type: semseg              # semseg --> Semantic Segmentation | multilabel --> Multilabel Segmentation
    path: some/path/to/data
    file_type: .png|.nii.gz|.b2nd|...
    pattern:
    backend:
    classes: 19               # number of classes, required for semseg/multilabel
    file_stack: False         # for multilabel: true if each class is a separate file: *_0000, *_0001, ...
    ignore_bg: null           # optional bool for labels; if true, class 0 is ignored in metrics and analysis
    ignore_index: 255         # optional int for labels; label value to ignore in loss/eval

# Splitting (optional)
splits:
  splits_file: some/path/splits_final.json # optional: use a file to define splits, content can be:
                              # 1) object: {"train": [...], "val": [...], "test": [...]}
                              # 2) list of folds: [{"train": [...], "val": [...]}, {...}, ...]
  # optional: Per-layer overrides for paths/patterns by split.
  train:
    SomeImages:               # empty -> use defaults
    SomeLabels:               # empty -> use defaults
  val:
    SomeImages:
      pattern: some_pattern   # optional: example override for pattern parameter for layer and split
      path: some/path/to/data # optional: example override for path parameter for layer and split
    SomeLabels:
      pattern: some_pattern
      path: some/path/to/data
  test:                       # empty --> split is not defined
```

</details>

# Config Manager

**TL;DR;**

- Central entry point: validates configs and builds `FileManager`, `Loader`, and `TaskManager` for each dataset layer.
- Config is a YAML file with `layers` (images/labels) and optional `splits`.
- Access layers by name, get file managers for splits/folds, and construct loaders/task managers automatically.

```python
from vidata.config_manager import ConfigManager
from vidata.io import load_yaml

cfg = load_yaml("dataset.yaml")
cm = ConfigManager(cfg)

print(cm.layer_names())  # -> ['SomeImages', 'SomeLabels']
image_layer = cm.layer("SomeImages")

fm = image_layer.file_manager()  # file discovery for this split/fold
loader = image_layer.data_loader()  # ready-to-use loader
arr, meta = loader.load(str(fm[0]))
```

<details>
<summary> Expand for Full Details </summary>

The `config_manager` module ties everything together: it validates dataset
configs, builds the appropriate **FileManager**, **Loader**, and **TaskManager**
for each dataset layer, and applies split definitions (train/val/test or folds).

```py
from vidata.config_manager import ConfigManager
from vidata.io import load_yaml

# Load a dataset config
cfg = load_yaml("dataset.yaml")
cm = ConfigManager(cfg)

# Access layers
print(cm.layer_names())                 # ['SomeImages', 'SomeLabels']
image_layer = cm.layer("SomeImages")

# Get a file manager for train split
fm = image_layer.file_manager(split="train", fold=0)
print(len(fm), "files")

# Build a loader
loader = image_layer.data_loader()
arr, meta = loader.load(str(fm[0]))

# Get the task manager for labels
label_layer = cm.layer("SomeLabels")
tm = label_layer.task_manager()
print("Classes in labels:", tm.class_ids(arr))
```

</details>

# Data Analysis

The `vidata_analyze` CLI computes dataset statistics and writes them to the
specified output directory. Results include:

- **Image statistics**: sizes, resolutions, intensity distributions
- **Label statistics**: class counts, frequencies, co-occurrence
- **Split summaries**: optional per-split analysis

```bash
vidata_analyze -c path/to/datasets/*.yaml  -o <outputdir>
# Analyze a specific split/fold
vidata_analyze -c path/to/datasets/*.yaml  -o <outputdir> -s <split> -f <fold>
```

# Data Inspection

The data_inspections CLI provides an interactive viewer (via Napari) to
browse images, labels, and splits defined in your dataset config

```bash
pip install napari-data-inspection[all]
```

Run the following

```bash
data_inspection -c path/to/datasets/*.yaml
# Inspect a specific split/fold
data_inspection -c path/to/datasets/*.yaml  -s <split> -f <fold>
```

# Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.

This repository was generated with [copier] using the [napari-plugin-template].

[copier]: https://copier.readthedocs.io/en/stable/
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
