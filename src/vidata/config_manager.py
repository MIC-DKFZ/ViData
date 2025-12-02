from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from vidata.file_manager import FileManager, FileManagerStacked
from vidata.io import load_json
from vidata.loaders import (
    BaseLoader,
    ImageLoader,
    ImageStackLoader,
    MultilabelLoader,
    MultilabelStackedLoader,
    SemSegLoader,
)
from vidata.registry import TASK_REGISTRY
from vidata.task_manager import (
    TaskManager,
)
from vidata.writers import (
    BaseWriter,
    ImageStackWriter,
    ImageWriter,
    MultilabelStackedWriter,
    MultilabelWriter,
    SemSegWriter,
)

_VALID_SPLITS = ["train", "val", "test"]
_IMAGE_LAYERS = {"image"}
_LABEL_LAYERS = {"semseg", "multilabel"}
_LOADER_MAPPING: dict[str, type[BaseLoader]] = {
    "image": ImageLoader,
    "semseg": SemSegLoader,
    "multilabel": MultilabelLoader,
}
_STACKED_LOADER_MAPPING: dict[str, type[BaseLoader]] = {
    "image": ImageStackLoader,
    "multilabel": MultilabelStackedLoader,
}
_WRITER_MAPPING: dict[str, type[BaseWriter]] = {
    "image": ImageWriter,
    "semseg": SemSegWriter,
    "multilabel": MultilabelWriter,
}
_STACKED_WRITER_MAPPING: dict[str, type[BaseWriter]] = {
    "image": ImageStackWriter,
    "multilabel": MultilabelStackedWriter,
}


class LayerConfigManager:
    def __init__(self, layer_config, split_config=None, splits_file=None, strict: bool = True):
        self.layer_config = layer_config
        self.split_config = split_config if split_config is not None else {}

        # Check if Config is valid
        req_all = ["name", "type", "path", "file_type"]
        req_img = ["channels"]
        req_lbl = ["classes"]
        errs = []
        for req in req_all:
            if req not in self.layer_config:
                errs.append(
                    f"Missing required field '{req}' for layer '{self.layer_config.get('name')}'"
                )

        if (
            not isinstance(self.layer_config.get("name"), str)
            or self.layer_config.get("name") == ""
        ):
            errs.append(f"name entry '{self.layer_config.get('name')}' must be a not empty string")

        if not isinstance(self.layer_config.get("path"), str):
            errs.append(f"name entry '{self.layer_config.get('path')}' must be a string")

        if (
            not isinstance(self.layer_config.get("file_type"), str)
            or self.layer_config.get("file_type") == ""
        ):
            errs.append(
                f"file_type entry '{self.layer_config.get('file_type')}' must be a not empty string"
            )

        # Check Types and Type Specific Requirements
        if self.layer_config.get("type").lower() in _IMAGE_LAYERS:
            for req in req_img:
                if req not in self.layer_config:
                    errs.append(
                        f"Missing required field '{req}' for layer '{self.layer_config.get('name')}'"
                    )

            if not isinstance(self.layer_config.get("channels"), int):
                errs.append(
                    f"Channels entry '{self.layer_config.get('channels')}' must be an integer."
                )

        elif self.layer_config.get("type").lower() in _LABEL_LAYERS:
            for req in req_lbl:
                if req not in self.layer_config:
                    errs.append(
                        f"Missing required field '{req}' for layer '{self.layer_config.get('name')}'"
                    )

                if not isinstance(self.layer_config.get("classes"), int):
                    errs.append(
                        f"Classes entry '{self.layer_config.get('classes')}' must be an integer."
                    )
        else:
            errs.append(
                f"Unknown layer type '{self.layer_config['type']}' for layer '{self.layer_config.get('name')}'"
            )

        if errs != [] and strict:
            for err in errs:
                print(err)
            raise ValueError(f"Config for layer '{self.layer_config.get('name')}' is invalid")

        # Check if splits_file is valid
        if splits_file is not None and not Path(splits_file).exists() and strict:
            raise FileNotFoundError(f"splits_file not found: {splits_file}")
        self.splits_file = splits_file

    @property
    def name(self):
        return self.layer_config["name"]

    @name.setter
    def name(self, value: str):
        self.layer_config["name"] = value

    @property
    def type(self):
        return self.layer_config["type"]

    @type.setter
    def type(self, value: str):
        self.layer_config["type"] = value

    @property
    def file_type(self):
        return self.layer_config["file_type"]

    @file_type.setter
    def file_type(self, value: str):
        self.layer_config["file_type"] = value

    @property
    def file_stack(self):
        return self.layer_config.get("file_stack", False)

    @file_stack.setter
    def file_stack(self, value: bool):
        self.layer_config["file_stack"] = bool(value)

    @property
    def backend(self):
        return self.layer_config.get("backend")

    @backend.setter
    def backend(self, value: str):
        self.layer_config["backend"] = value

    @property
    def classes(self):
        if self.type.lower() in _LABEL_LAYERS:
            return self.layer_config["classes"]
        else:
            raise KeyError(f"Layer type {self.type} does not have a class attribute")

    @classes.setter
    def classes(self, value: int):
        if self.type.lower() in _LABEL_LAYERS:
            self.layer_config["classes"] = int(value)
        else:
            raise KeyError(f"Layer type {self.type} does not accept 'classes'")

    @property
    def channels(self):
        if self.type.lower() in _IMAGE_LAYERS:
            return self.layer_config["channels"]
        else:
            raise KeyError(f"Layer type {self.type} does not have a channels attribute")

    @channels.setter
    def channels(self, value: int):
        if self.type.lower() in _IMAGE_LAYERS:
            self.layer_config["channels"] = int(value)
        else:
            raise KeyError(f"Layer type {self.type} does not accept 'channels'")

    @property
    def ignore_bg(self):
        if self.type.lower() in _LABEL_LAYERS:
            return self.layer_config.get("ignore_bg", True)
        else:
            raise KeyError(f"Layer type {self.type} does not have a ignore_bg attribute")

    @ignore_bg.setter
    def ignore_bg(self, value: bool):
        if self.type.lower() in _LABEL_LAYERS:
            self.layer_config["ignore_bg"] = value
        else:
            raise KeyError(f"Layer type {self.type} does not accept 'ignore_bg'")

    @property
    def ignore_index(self):
        if self.type.lower() in _LABEL_LAYERS:
            return self.layer_config.get("ignore_index")
        else:
            raise KeyError(f"Layer type {self.type} does not have a ignore_index attribute")

    @ignore_index.setter
    def ignore_index(self, value: int):
        if self.type.lower() in _LABEL_LAYERS:
            self.layer_config["ignore_index"] = value
        else:
            raise KeyError(f"Layer type {self.type} does not accept 'ignore_index'")

    def config(self, split: str | None = None, fold: int | None = None):
        _cfg = self.layer_config.copy()

        if split is None:
            return _cfg

        if self.split_config.get(split) is None:
            if self.splits_file is not None:
                return _cfg
            else:
                raise ValueError(
                    f"split {split} is not defined for {self.name} in neither the config nor a split file"
                )
        else:
            _cfg.update(self.split_config[split])
        if fold is not None and split and self.splits_file is not None:
            _cfg["include_names"] = self.resolve_splits_file(split, fold)
        return _cfg

    def resolve_splits_file(self, split: str, fold: int | None = None) -> list[str]:
        if self.splits_file is None:
            raise ValueError(f"no splits file defined for {self.name}")
        splits = load_json(self.splits_file)

        if isinstance(splits, list):
            if fold is None:
                raise ValueError(
                    "splits_index/fold is required if your splits_file contains a list"
                )
            if not (0 <= fold < len(splits)):
                raise ValueError(
                    f"splits_index/fold {fold} is not in range of your splits file with len {len(splits)}"
                )
            splits = splits[fold]

        if split not in splits:
            raise ValueError(f"split {split} is not in splits_file with keys {list(splits.keys())}")

        resolved = splits[split]
        assert isinstance(resolved, list)  # Should be a list of files
        return resolved

    def file_manager(self, split: str | None = None, fold: int | None = None) -> FileManager:
        _cfg = self.config(split=split)

        manager_cls = FileManagerStacked if self.file_stack else FileManager

        include_names = None
        if self.splits_file is not None and split is not None:
            include_names = self.resolve_splits_file(split, fold)
        return manager_cls(
            path=_cfg["path"],
            file_type=_cfg["file_type"],
            pattern=_cfg.get("pattern"),
            include_names=include_names,
        )

    def data_loader(self) -> BaseLoader:
        reg = _STACKED_LOADER_MAPPING if self.file_stack else _LOADER_MAPPING
        try:
            loader_cls = reg[self.type.lower()]
        except KeyError as err:
            raise ValueError(f"type {self.type} is not supported for layer {self.name}") from err

        args = {
            "ftype": self.file_type,
            "backend": self.backend,
        }

        if self.type.lower() in _IMAGE_LAYERS:
            args["channels"] = self.channels
        elif self.type.lower() in _LABEL_LAYERS:
            args["num_classes"] = self.classes

        return loader_cls(**args)

    def data_writer(self) -> BaseWriter:
        reg = _STACKED_WRITER_MAPPING if self.file_stack else _WRITER_MAPPING
        try:
            writer_cls = reg[self.type.lower()]
        except KeyError as err:
            raise ValueError(f"type {self.type} is not supported for layer {self.name}") from err

        args = {
            "ftype": self.file_type,
            "backend": self.backend,
        }

        if self.type.lower() in _IMAGE_LAYERS:
            args["channels"] = self.channels
        elif self.type.lower() in _LABEL_LAYERS:
            args["num_classes"] = self.classes

        return writer_cls(**args)

    def task_manager(self) -> TaskManager:
        if self.type.lower() in TASK_REGISTRY:
            return TASK_REGISTRY[self.type.lower()]
        else:
            raise ValueError(f"No Task manager defined for layer {self.name} and type {self.type}")


class ConfigManager:
    def __init__(self, config: dict | DictConfig | str | Path, strict: bool = True):
        if isinstance(config, (str | Path)):
            self.config = OmegaConf.load(config)
        else:
            self.config = config
        self.layers = []

        split_cfg = self.config.get("splits", {})

        layers_cfg = self.config.get("layers", [])
        if isinstance(layers_cfg, (dict | DictConfig)):
            layer_list = []
            for layer_name, layer_cfg in layers_cfg.items():
                layer_cfg = dict(layer_cfg)
                layer_cfg["name"] = layer_name
                layer_list.append(layer_cfg)
        elif isinstance(layers_cfg, (list | ListConfig)):
            layer_list = layers_cfg
        else:
            raise ValueError(f"Invalid type for layers: {type(layers_cfg)}. Must be dict or list.")

        for layer_cfg in layer_list:
            layer_split = {}

            for k in _VALID_SPLITS:
                if split_cfg.get(k, {}) is not None and layer_cfg["name"] in split_cfg.get(k, {}):
                    ovrds = split_cfg[k][layer_cfg["name"]]
                    layer_split[k] = ovrds if ovrds is not None else {}
            lcm = LayerConfigManager(
                layer_cfg, layer_split, split_cfg.get("splits_file"), strict=strict
            )

            self.layers.append(lcm)

    @property
    def name(self):
        return self.config["name"]

    def layer(self, layer_name: str) -> LayerConfigManager:
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        raise ValueError(f"No layer found with name '{layer_name}'")

    def layer_names(self):
        return [layer.name for layer in self.layers]

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, layer_name: str) -> LayerConfigManager:
        return self.layer(layer_name)
