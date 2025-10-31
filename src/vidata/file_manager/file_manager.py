from pathlib import Path
from typing import Union

import numpy as np
from natsort import natsorted

PathLike = Union[str, Path]


class FileManager:
    """
    Flexible file collector with optional patterns and name based filtering.

    Parameters
    ----------
    path : str | Path
        Root directory to search.
    file_type : str
        File extension (e.g., ".nii.gz", ".png").
    pattern : str  | None
        Glob-like pattern (e.g., "*_image", "_0000")
    include_names: list[str] | None
        Keep files whose RELATIVE path contains ANY of these substrings.
    exclude_names: list[str] | None
        Drop files whose RELATIVE path contains ANY of these substrings. (Exclude wins.)
    recursive: bool
        Whether to recursively search subdirectories.
    """

    def __init__(
        self,
        path: PathLike,
        file_type: str,
        pattern: str | None = None,
        include_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
        recursive: bool = False,
    ):
        self.path = path
        self.file_type = file_type
        self.pattern = pattern
        self.include_names = include_names
        self.exclude_names = exclude_names
        self.recursive = recursive
        self.collect_files()
        self.filter_files()

    def filter_files(self):
        if self.include_names is not None:
            _files_re = [str(_file.relative_to(self.path)) for _file in self.files]
            self.files = [
                _file
                for _file, rel in zip(list(self.files), _files_re, strict=False)
                if any(_token in rel for _token in self.include_names)
            ]

        if self.exclude_names is not None:
            _files_re = [str(_file.relative_to(self.path)) for _file in self.files]
            self.files = [
                _file
                for _file, rel in zip(list(self.files), _files_re, strict=False)
                if not any(_token in rel for _token in self.exclude_names)
            ]

    def collect_files(self):
        if self.file_type == "" or self.path == "":
            self.files = []
            return

        if self.pattern is None:
            pattern = "*"
        elif "*" not in self.pattern:
            pattern = "*" + self.pattern
        else:
            pattern = self.pattern

        if self.recursive:
            files = list(Path(self.path).rglob(pattern + self.file_type))
        else:
            files = list(Path(self.path).glob(pattern + self.file_type))
        self.files = natsorted(files, key=lambda p: p.name)

    def get_name(self, file: str | int, with_file_type=True) -> str:
        """Just keep this for backwards compatibility"""
        return self.name_from_path(file, with_file_type)

    def name_from_path(self, file: str | int, include_ext: bool = True) -> str:
        if isinstance(file, int):
            file = str(self.files[file])
        name = str(Path(file).relative_to(self.path))
        if not include_ext:
            name = name.replace(self.file_type, "")
        return name

    def path_from_name(self, name: str | Path, include_ext=True):
        rel = Path(name)
        if include_ext and rel.suffix != self.file_type:
            rel = rel.with_suffix(self.file_type)
        return (self.path / rel).resolve()

    def __getitem__(self, item: int):
        return self.files[item]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)


class FileManagerStacked(FileManager):
    """
    Expect stacks in the following format like this :
    path
        file1_0000.nii.gz
        file1_0001.nii.gz
        ...
    Returns file1
    ...
    """

    def collect_files(self):
        super().collect_files()
        if self.files != []:
            files = [file.with_name(file.stem.rsplit("_", 1)[0]) for file in self.files]
            files = np.unique(files)

            self.files = natsorted(files, key=lambda p: p.name)
