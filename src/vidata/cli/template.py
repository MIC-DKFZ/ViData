from pathlib import Path

from omegaconf import OmegaConf


def main():
    print("=== YAML Template Creator ===")
    project_name = input("Project Name: ")
    if project_name == "":
        raise Exception("Project Name cannot be empty")

    output_path = Path.cwd() / (project_name + ".yaml")
    if output_path.exists() and not input("Project exists, overwrite? (Y/N)").lower() == "y":
        raise FileExistsError(f"Output path already exists: {output_path}")

    n_ilayers = input("Number of Image layers: ")
    n_ilayers = None if n_ilayers == "" else int(n_ilayers)

    n_llayers = input("Number of Label layers: ")
    n_llayers = None if n_llayers == "" else int(n_llayers)

    f_type = input("File Type (e.g. .nii.gz, .png): ")
    f_type = None if f_type == "" else f_type
    if n_ilayers is not None and n_ilayers > 0:
        n_channels = input("Number of Image Channels: ")
        n_channels = None if n_channels == "" else int(n_channels)
    else:
        n_channels = "TODO"
    if n_llayers is not None and n_llayers > 0:
        n_classes = input("Number of Label Classes: ")
        n_classes = None if n_classes == "" else int(n_classes)
        task = input("Semantic Segmentation(S)/MultilabelSegmentation(M): ")
        if task.lower() == "s":
            task = "semseg"
        elif task.lower() == "m":
            task = "multilabel"
        else:
            task = "TODO - semseg|multilabel"
    else:
        n_classes = "TODO"
        task = "TODO - semseg|multilabel"

    split = input("Create Split Template (Y/N): ")
    split = split.lower() == "y"

    config = {"name": project_name}
    layers_i = [
        {
            "name": f"ImageLayer{i + 1}",
            "type": "image",  # change to "labels"/"points" if needed
            "path": "TODO",
            "file_type": f_type,
            "pattern": None,
            "backend": None,
            "channel": n_channels,  # optional
            "file_stack": False,
        }
        for i in range(n_ilayers)
    ]
    layers_l = [
        {
            "name": f"LabelLayer{i + 1}",
            "type": task,
            "path": "TODO",
            "file_type": f_type,
            "pattern": None,
            "backend": None,
            "classes": n_classes,
            "file_stack": False,
            "ignore_bg": None,
            "ignore_index": None,
        }
        for i in range(n_llayers)
    ]
    config["layers"] = layers_i + layers_l

    if split:
        config["split"] = {"splits_file": None, "train": None, "val": None, "test": None}
        layer_names = {}
        for layer in config["layers"]:
            layer_names[layer["name"]] = None
        config["split"]["train"] = layer_names
        config["split"]["val"] = layer_names
        config["split"]["test"] = layer_names

    OmegaConf.save(config, output_path)
    print(f"✔ Wrote template to: {output_path}")
    print(" - Fill out all 'TODO'")
    print(" - Optional - rename the layers")
    print(" - Optional - 'null' entries are optional, you can change or delete them")


if __name__ == "__main__":
    main()
