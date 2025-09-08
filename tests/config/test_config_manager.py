from pathlib import Path

import pytest

from vidata import ConfigManager, LayerConfigManager
from vidata.file_manager import FileManager
from vidata.io import save_json
from vidata.loaders import ImageLoader, MultilabelStackedLoader, SemSegLoader
from vidata.writers import ImageWriter, MultilabelStackedWriter, SemSegWriter

LEN_VAL = 4
LEN_TRAIN = 6


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


@pytest.fixture
def simple_config(tmp_path):
    Path(tmp_path / "images").parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_path / "labels").parent.mkdir(parents=True, exist_ok=True)

    for i in range(LEN_TRAIN):
        _touch(Path(tmp_path / "images" / f"train_{i}.png"))
        _touch(Path(tmp_path / "labels" / f"train_{i}.png"))
        for j in range(5):
            _touch(Path(tmp_path / "mllabels" / f"train_{i}_000{j}.png"))
    for i in range(LEN_VAL):
        _touch(Path(tmp_path / "images" / f"val_{i}.png"))
        _touch(Path(tmp_path / "labels" / f"val_{i}.png"))
        for j in range(5):
            _touch(Path(tmp_path / "mllabels" / f"val_{i}_000{j}.png"))

    return {
        "name": "MyDs",
        "root": tmp_path,
        "layers": [
            {
                "name": "Images",
                "type": "image",
                "path": str(tmp_path / "images"),
                "pattern": "train_*",
                "file_type": ".png",
                "channels": 3,
                "file_stack": False,
                "backend": None,
            },
            {
                "name": "Labels",
                "type": "semseg",
                "path": str(tmp_path / "labels"),
                "pattern": "train_*",
                "file_type": ".png",
                "classes": 5,
                "backend": None,
            },
            {
                "name": "MLLabels",
                "type": "multilabel",
                "path": str(tmp_path / "mllabels"),
                "pattern": "train_*",
                "file_stack": True,
                "file_type": ".png",
                "classes": 5,
                "backend": None,
            },
        ],
        "splits": {
            "train": {"Images": None, "MLLabels": None, "Labels": None},
            "val": {
                "Images": {"pattern": "val_*"},
                "MLLabels": {"pattern": "val_*"},
                "Labels": {"pattern": "val_*"},
            },
        },
    }


def test_config_manager(simple_config):
    cm = ConfigManager(simple_config)

    assert cm.name == simple_config["name"]
    assert len(cm) == 3

    assert cm.layer_names() == ["Images", "Labels", "MLLabels"]

    for i, layer in enumerate(cm.layers):
        l_conf = simple_config["layers"][i]
        assert isinstance(layer, LayerConfigManager)
        assert layer.name == l_conf["name"]
        assert layer.type == l_conf["type"]
        assert layer.file_type == l_conf["file_type"]
        assert layer.file_stack == l_conf.get("file_stack", False)
        assert layer.backend == l_conf["backend"]

    for i, (ln, loader) in enumerate(
        [("Images", ImageLoader), ("Labels", SemSegLoader), ("MLLabels", MultilabelStackedLoader)]
    ):
        layer = cm.layer(ln)

        dl = layer.data_loader()
        assert isinstance(dl, loader)

        l_conf = layer.config()
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager()
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN

        l_conf = layer.config("train")
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager("train")
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN

        l_conf = layer.config("val")
        l_conf["pattern"] = simple_config["splits"]["val"][layer.name]["pattern"]
        fm = layer.file_manager("val")
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_VAL


def test_config_manager_splitfile(simple_config):
    splits_file = {"train": ["_0", "_2", "_4"], "val": ["_0", "_2"]}
    save_json(splits_file, Path(simple_config["root"]) / "splits.json")
    simple_config["splits"]["splits_file"] = Path(simple_config["root"]) / "splits.json"
    cm = ConfigManager(simple_config)

    assert cm.name == simple_config["name"]
    assert len(cm) == 3

    assert cm.layer_names() == ["Images", "Labels", "MLLabels"]

    for i, layer in enumerate(cm.layers):
        l_conf = simple_config["layers"][i]
        assert isinstance(layer, LayerConfigManager)
        assert layer.name == l_conf["name"]
        assert layer.type == l_conf["type"]
        assert layer.file_type == l_conf["file_type"]
        assert layer.file_stack == l_conf.get("file_stack", False)
        assert layer.backend == l_conf["backend"]

    for i, (ln, loader, writer) in enumerate(
        [
            ("Images", ImageLoader, ImageWriter),
            ("Labels", SemSegLoader, SemSegWriter),
            ("MLLabels", MultilabelStackedLoader, MultilabelStackedWriter),
        ]
    ):
        layer = cm.layer(ln)

        dl = layer.data_loader()
        assert isinstance(dl, loader)
        df = layer.data_writer()
        assert isinstance(df, writer)

        l_conf = layer.config()
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager()
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN

        l_conf = layer.config("train")
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager("train")
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN // 2

        l_conf = layer.config("val")
        l_conf["pattern"] = simple_config["splits"]["val"][layer.name]["pattern"]
        fm = layer.file_manager("val")
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_VAL // 2


def test_config_manager_splitfile_fold(simple_config):
    splits_file = [{"train": ["_0", "_2", "_4"], "val": ["_0", "_2"]}]
    save_json(splits_file, Path(simple_config["root"]) / "splits.json")
    simple_config["splits"]["splits_file"] = Path(simple_config["root"]) / "splits.json"
    cm = ConfigManager(simple_config)

    assert cm.name == simple_config["name"]
    assert len(cm) == 3

    assert cm.layer_names() == ["Images", "Labels", "MLLabels"]

    for i, layer in enumerate(cm.layers):
        l_conf = simple_config["layers"][i]
        assert isinstance(layer, LayerConfigManager)
        assert layer.name == l_conf["name"]
        assert layer.type == l_conf["type"]
        assert layer.file_type == l_conf["file_type"]
        assert layer.file_stack == l_conf.get("file_stack", False)
        assert layer.backend == l_conf["backend"]

    for i, (ln, loader, writer) in enumerate(
        [
            ("Images", ImageLoader, ImageWriter),
            ("Labels", SemSegLoader, SemSegWriter),
            ("MLLabels", MultilabelStackedLoader, MultilabelStackedWriter),
        ]
    ):
        layer = cm.layer(ln)

        dl = layer.data_loader()
        assert isinstance(dl, loader)
        df = layer.data_writer()
        assert isinstance(df, writer)

        l_conf = layer.config()
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager()
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN

        l_conf = layer.config("train", 0)
        l_conf["pattern"] = simple_config["layers"][i]["pattern"]
        fm = layer.file_manager("train", 0)
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_TRAIN // 2

        l_conf = layer.config("val", 0)
        l_conf["pattern"] = simple_config["splits"]["val"][layer.name]["pattern"]
        fm = layer.file_manager("val", 0)
        assert isinstance(fm, FileManager)
        assert len(fm) == LEN_VAL // 2
