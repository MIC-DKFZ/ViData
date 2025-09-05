import argparse
from pathlib import Path

from omegaconf import OmegaConf

from vidata.io import load_json
from vidata.workflows import build_file_manager

_IMAGE_TYPES = ["Image"]
_LABEL_TYPES = ["Labels", "SemSeg", "MultiLabel"]
_VALID_SPLITS = ["train", "val", "test"]


def apply_split(cfg: OmegaConf, split: str):
    layers = []
    for layer in cfg.layers:
        layer_split = apply_split_layer(layer, split, cfg.get("split", {}))
        layers.extend(layer_split)
    cfg.layers = layers
    return cfg


def apply_split_layer(layer, target_split, split_cfg):
    _cfg = split_cfg.get(target_split, {}).get(layer.name)
    if _cfg is None:
        return None

    target_layer = layer.copy()
    for k, v in _cfg.items():
        target_layer[k] = v

    return target_layer


def verify_layer(layer_cfg):
    req_all = ["name", "type", "path", "file_type"]
    req_img = ["channels"]
    req_lbl = ["classes"]
    errs = []

    for req in req_all:
        if req not in layer_cfg:
            errs.append(f"Missing required field '{req}' for layer '{layer_cfg.get('name')}'")

    if not isinstance(layer_cfg.get("name"), str) or layer_cfg.get("name") == "":
        errs.append(f"name entry '{layer_cfg.get('name')}' must be a not empty string")

    if not isinstance(layer_cfg.get("path"), str):
        errs.append(f"name entry '{layer_cfg.get('path')}' must be a string")

    if not isinstance(layer_cfg.get("file_type"), str) or layer_cfg.get("file_type") == "":
        errs.append(f"file_type entry '{layer_cfg.get('file_type')}' must be a not empty string")

    # Check Types and Type Specific Requirements
    if layer_cfg.get("type") in _IMAGE_TYPES:
        for req in req_img:
            if req not in layer_cfg:
                errs.append(f"Missing required field '{req}' for layer '{layer_cfg.get('name')}'")

        if not isinstance(layer_cfg.get("channels"), int):
            errs.append(f"Channels entry '{layer_cfg.get('channels')}' must be an integer.")

    elif layer_cfg.get("type") in _LABEL_TYPES:
        for req in req_lbl:
            if req not in layer_cfg:
                errs.append(f"Missing required field '{req}' for layer '{layer_cfg.get('name')}'")

            if not isinstance(layer_cfg.get("classes"), int):
                errs.append(f"Classes entry '{layer_cfg.get('classes')}' must be an integer.")
    else:
        errs.append(f"Unknown layer type '{layer_cfg['type']}' for layer '{layer_cfg.get('name')}'")

    if (
        "pattern" in layer_cfg
        and layer_cfg.get("pattern") is not None
        and not isinstance(layer_cfg.get("pattern"), str)
    ):
        errs.append(f"pattern entry '{layer_cfg.get('pattern')}' must be a string")

    return errs == [], errs


def verify_config(cfg):
    layer_is_valid = {}
    for layer in cfg.layers:
        print(f"--- Verify Layer '{layer.get('name')}' ---")
        is_valid, errs = verify_layer(layer)
        if is_valid:
            print(f"Layer {layer.name} is valid")
            fm = build_file_manager(
                path=layer.path,
                file_type=layer.file_type,
                pattern=layer.get("pattern", None),
                file_stack=layer.get("file_stack", False),
            )
            print(f"Layer {layer.name} contains {len(fm)} files")
        else:
            print(f"Layer {layer.name} is not valid")
            for err in errs:
                print(f"    {err}")
        layer_is_valid[layer.name] = is_valid
    if "split" not in cfg:
        quit()

    split_file = cfg.split.get("split_file", None)
    if split_file is not None:
        _splits = load_json(split_file)
        if not isinstance(_splits, list):
            _splits = [_splits]
        folds = len(_splits)
    else:
        _splits = None
        folds = 1

    for fold in range(folds):
        if folds > 1:
            print(f"--- Verify Splits Fold {fold} ---")
        else:
            print("--- Verify Splits ---")
        for valid_split in _VALID_SPLITS:
            for layer in cfg.layers:
                if not layer_is_valid[layer.name]:
                    continue

                layer_split = layer.copy()
                # layer_split = apply_split_layer(layer, valid_split, cfg.get("split", {}))
                split_dict = layer.get(valid_split, {}).get(layer.name, {})
                layer_split.update(split_dict)
                if split_dict == {} and (_splits is None or valid_split not in _splits[fold]):
                    print(f"{valid_split}: For Layer {layer.name} exists no explicit split")
                    continue

                layer_split = layer_split if layer_split is not None else layer

                fm = build_file_manager(
                    path=layer_split.path,
                    file_type=layer_split.file_type,
                    pattern=layer_split.get("pattern", None),
                    file_stack=layer_split.get("file_stack", False),
                    split=valid_split,
                    splits_file=split_file,
                    splits_index=fold,
                )
                # print(f"Layer {layer.name} contains {len(fm)} files for split '{valid_split}'")
                print(f"{valid_split}: Layer {layer.name} contains {len(fm)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify your Dataset configuration")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_file = args.config
    cfg = OmegaConf.load(config_file)

    verify_config(cfg)
