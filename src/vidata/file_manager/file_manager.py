from pathlib import Path
from typing import Union

import numpy as np
from natsort import natsorted

from vidata.io import load_json

PathLike = Union[str, Path]


class FileManager:
    """
    Flexible file collector with optional patterns and name based filtering.
    Also supports lazy loading (useful for multiprocessing).

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
    lazy_init : bool
        If True, defer file collection until the first access (default: False).
    """

    def __init__(
        self,
        path: PathLike,
        file_type: str,
        pattern: str | None = None,
        include_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
        recursive: bool = False,
        lazy_init: bool = False,
    ):
        self.path = Path(path)
        self.file_type = file_type
        self.pattern = pattern
        self.include_names = include_names
        self.exclude_names = exclude_names
        self.recursive = recursive

        self._files: list[Path] | None
        if not lazy_init:
            self.refresh()
        else:
            self._files = None

    def refresh(self):
        """
        (Re)collect and filter files immediately.

        This method rebuilds the internal file list by scanning the directory and
        applying inclusion/exclusion filters.
        """
        self._files = self.collect_files(self.path, self.file_type, self.pattern, self.recursive)
        self._files = self.filter_files(
            self._files, self.path, self.include_names, self.exclude_names
        )

    @property
    def files(self) -> list[Path]:
        """
        Lazily returns the collected file list.

        If `lazy_init=True` was set and the files have not yet been collected,
        this property will automatically trigger a collection.
        """
        if self._files is None:  # Lazy loading
            self.refresh()
            assert self._files is not None
        return self._files

    @files.setter
    def files(self, value: list[Path]):
        """Directly override the internal file list (advanced use only)."""
        self._files = value

    @staticmethod
    def filter_files(
        files: list[Path],
        path: Path,
        include_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> list[Path]:
        """
        Filter a list of files based on inclusion or exclusion substrings.

        Parameters
        ----------
        files : list[Path]
            Input file list.
        path : Path
            Root path used to compute relative paths for filtering.
        include_names : list[str] | None
            Substrings; keep files containing any of these in their relative path.
        exclude_names : list[str] | None
            Substrings; remove files containing any of these in their relative path.

        Returns
        -------
        list[Path]
            Filtered file list.
        """
        if include_names is not None:
            _files_re = [str(_file.relative_to(path)) for _file in files]
            files = [
                _file
                for _file, rel in zip(list(files), _files_re, strict=False)
                if any(_token in rel for _token in include_names)
            ]

        if exclude_names is not None:
            _files_re = [str(_file.relative_to(path)) for _file in files]
            files = [
                _file
                for _file, rel in zip(list(files), _files_re, strict=False)
                if not any(_token in rel for _token in exclude_names)
            ]
        return files

    @staticmethod
    def collect_files(
        path: Path, file_type: str, pattern: str | None, recursive: bool = False
    ) -> list[Path]:
        """
        Collect files under the given directory according to a pattern and extension.

        Parameters
        ----------
        path : Path
            Root directory to search or path to a json file which contains a list of absolute paths.
        file_type : str
            File extension to match (e.g., ".png").
        pattern : str | None
            Glob-like pattern (e.g., "*_image").
        recursive : bool, optional
            Whether to recursively search subdirectories.

        Returns
        -------
        list[Path]
            Naturally sorted list of file paths.
        """
        if file_type == "" or path == "":
            return []

        if str(path).endswith(".json"):
            if Path(path).is_file():
                files = load_json(path)
                files = [Path(file) for file in files]
            else:
                raise FileNotFoundError(path)
        else:
            if pattern is None:
                pattern = "*"
            elif "*" not in pattern:
                pattern = "*" + pattern

            if recursive:
                files = list(Path(path).rglob(pattern + file_type))
            else:
                files = list(Path(path).glob(pattern + file_type))
                files = natsorted(files, key=lambda p: p.name)
        return files

    def get_name(self, file: str | int, with_file_type=True) -> str:
        """Legacy alias for :meth:`name_from_path` (kept for backward compatibility)."""
        return self.name_from_path(file, with_file_type)

    def name_from_path(self, file: Path | str | int, include_ext: bool = True) -> str:
        """
        Get the relative name of a file (e.g., 'subdir/sample.png').

        Parameters
        ----------
        file : str | int
            File path or index into the internal file list.
        include_ext : bool
            Whether to keep the file extension.

        Returns
        -------
        str
            Relative file name.
        """
        if isinstance(file, int):
            file = self.files[file]
        if not isinstance(file, Path):
            file = Path(file)

        name_pl = file.relative_to(self.path) if self.path.suffix != ".json" else file
        name = name_pl.as_posix()

        if not include_ext and name.endswith(self.file_type):
            name = name[: -len(self.file_type)]

        return name

    def path_from_name(self, name: str | Path, include_ext=True):
        """
        Convert a relative name (as from :meth:`name_from_path`) to an absolute path.
        """
        rel = Path(name)
        if include_ext and rel.suffix != self.file_type:
            rel = rel.with_suffix(self.file_type)
        if self.path.suffix == ".json":
            return rel
        else:
            return (self.path / rel).resolve()

    def __getitem__(self, item: int):
        return self.files[item]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __getstate__(self):
        """
        Make the object lightweight for pickling.

        The file list is omitted to reduce memory footprint when the object is
        sent to subprocesses. Workers can rebuild it lazily on first access.
        """
        return {
            "path": str(self.path),
            "file_type": self.file_type,
            "pattern": self.pattern,
            "include_names": self.include_names,
            "exclude_names": self.exclude_names,
            "recursive": self.recursive,
            "_files": None,
        }

    def __setstate__(self, state):
        """
        Restore object state after unpickling (used in multiprocessing).
        The file list will be lazily rebuilt on first access.
        """
        self.path = Path(state["path"])
        self.file_type = state["file_type"]
        self.pattern = state["pattern"]
        self.include_names = state["include_names"]
        self.exclude_names = state["exclude_names"]
        self.recursive = state["recursive"]
        self._files = state.get("_files", None)


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

    @staticmethod
    def collect_files(
        path: Path, file_type: str, pattern: str | None, recursive: bool = False
    ) -> list[Path]:
        # def collect_files(self):
        files = FileManager.collect_files(path, file_type, pattern, recursive)
        if files != []:
            files = [file.with_name(file.stem.rsplit("_", 1)[0]) for file in files]
            files = np.unique(files)

            files = natsorted(files, key=lambda p: p.name)
        return files
