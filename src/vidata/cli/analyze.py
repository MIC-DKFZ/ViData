import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from vidata.analysis.base_analyzer import Analyzer
from vidata.analysis.image_analyzer import ImageAnalyzer
from vidata.analysis.label_analyzer import LabelAnalyzer
from vidata.config_manager import ConfigManager


def run_analysis(
    conf_manager: ConfigManager,
    output_dir: str | Path,
    split: str | None = None,
    fold: int | None = None,
    p: int = 16,
    verbose: bool = False,
    layer_name: str | None = None,
):
    for layer in conf_manager.layers:
        name = layer.name

        if layer_name is not None and name != str(layer_name):
            print(f"Skipping Layer {name}")
            continue

        if split is not None:
            name += f"_{split}"
        if fold is not None:
            name += f"_{fold}"

        file_manager = layer.file_manager(split, fold)
        loader = layer.data_loader()

        analyzer: Analyzer
        if layer.type.lower() == "image":
            analyzer = ImageAnalyzer(loader, file_manager, layer.channels)
        elif layer.type.lower() in ["labels", "semseg", "multilabel"]:
            analyzer = LabelAnalyzer(
                loader, file_manager, layer.task_manager(), layer.classes, layer.ignore_bg
            )
        else:
            raise ValueError(f"Layer type {layer.type} cannot be analyzed")

        _ = analyzer.run(n_processes=p, verbose=verbose)
        analyzer.save(Path(output_dir) / f"{name}.csv")
        analyzer.load(Path(output_dir) / f"{name}.csv")

        analyzer.aggregate(Path(output_dir) / f"{name}.json")
        analyzer.plot(output_dir, name=name)


def main():
    parser = argparse.ArgumentParser(description="Analyze your Data")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("-l", "--layer", type=Path, default=None, help="Use a specific layer")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to output directory")
    parser.add_argument(
        "-p", "--processes", type=int, default=16, help="Number of worker processes"
    )
    parser.add_argument(
        "-s", "--split", default=None, help="Split tag for filenames (e.g. train/val/test)"
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=None,
        help="Which split to use (fold index). Only needed when split_file is provided in the config.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose analyzer output")
    args = parser.parse_args()

    config_file = args.config
    p = args.processes
    verbose = args.verbose
    split = args.split
    fold = args.fold
    layer_name = args.layer

    cfg = OmegaConf.load(config_file)

    output_dir = (
        args.output if args.output is not None else Path(Path.cwd() / "stats" / cfg["name"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    shutil.copy(config_file, str(output_dir / config_file.name))

    conf_manager = ConfigManager(cfg)

    run_analysis(conf_manager, output_dir, split, fold, p, verbose, layer_name)


if __name__ == "__main__":
    main()
