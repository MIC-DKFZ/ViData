import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vidata.analysis.base_analyzer import Analyzer
from vidata.analysis.utils import gather_shape_stats, get_occurrence_matrix
from vidata.analysis.viz_utils import adjust_layout, save_figure
from vidata.file_manager import FileManager
from vidata.io import save_json
from vidata.loaders import BaseLoader
from vidata.task_manager import TaskManager
from vidata.utils.color import get_colormap
from vidata.utils.multiprocess import multiprocess_iter


class LabelAnalyzer(Analyzer):
    def __init__(
        self,
        data_loader: BaseLoader,
        file_manager: FileManager,
        task_manager: TaskManager,
        n_classes: int,
        ignore_bg: bool,
    ):
        self.data_loader = data_loader
        self.file_manager = file_manager
        self.task_manager = task_manager
        self.n_classes = n_classes
        self.ignore_bg = ignore_bg

    def analyze_case(self, file, verbose=False):
        # file = self.file_manager[index]
        data, meta = self.data_loader.load(file)
        data = data[...]  # To resolve memmap dtypes
        data = data.astype(np.uint8)

        stats = {
            "name": file.name,
            "dtype": str(data.dtype),
            "shape": data.shape,
            "class_ids": self.task_manager.class_ids(data).tolist(),
        }

        for n in range(self.n_classes):
            cnt = self.task_manager.class_count(data, n)
            stats[f"class_{n}_cnt"] = cnt

        if meta is not None and "spacing" in meta:
            stats["spacing"] = meta["spacing"].tolist()

        if verbose:
            print(stats)
        return stats

    def run(self, n_processes=8, progressbar=True, verbose=False):
        stats = multiprocess_iter(
            self.analyze_case,
            iterables={"file": self.file_manager},
            const={"verbose": verbose},
            p=n_processes,
            progressbar=progressbar,
            desc="Analyzing Label Data",
        )
        self.stats = pd.DataFrame(stats)

        return self.stats

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stats.to_csv(path)

    def load(self, path):
        self.stats = pd.read_csv(path)
        self.stats["shape"] = self.stats["shape"].apply(ast.literal_eval)
        self.stats["class_ids"] = self.stats["class_ids"].apply(ast.literal_eval)
        if "spacing" in self.stats.columns:
            self.stats["spacing"] = self.stats["spacing"].apply(ast.literal_eval)
        return self.stats

    def aggregate(self, path=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        global_stats = {}

        stats = self.stats.copy()

        class_cnt = [(stats[f"class_{n}_cnt"] > 0).sum() for n in range(self.n_classes)]
        class_size = [
            stats.loc[stats[f"class_{n}_cnt"] > 0, f"class_{n}_cnt"].mean()
            for n in range(self.n_classes)
        ]
        global_stats["labels"] = {
            "count": np.array(class_cnt).tolist(),
            "size": np.array(class_size).tolist(),
        }

        stats["spatial_dims"] = stats["shape"].apply(lambda s: self.task_manager.spatial_dims(s))
        global_stats["shape"] = gather_shape_stats(stats["spatial_dims"])

        global_stats["co-occurrence"] = get_occurrence_matrix(
            self.n_classes, stats["class_ids"], norm=False
        ).tolist()

        if path is not None:
            save_json(global_stats, path)

        return global_stats

    def plot(self, path, name=""):
        Path(path).mkdir(parents=True, exist_ok=True)
        class_cnt = [(self.stats[f"class_{n}_cnt"] > 0).sum() for n in range(self.n_classes)]
        class_size = [
            self.stats.loc[self.stats[f"class_{n}_cnt"] > 0, f"class_{n}_cnt"].mean()
            for n in range(self.n_classes)
        ]
        categories = [f"Class {n}" for n in range(self.n_classes)]
        if self.ignore_bg and self.task_manager.has_background():
            class_cnt = class_cnt[1:]
            class_size = class_size[1:]
            categories = categories[1:]

        # --- Frequency Plots --- #
        class_pct = np.array(class_cnt) / len(self.stats)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=categories,
                y=class_cnt,
                marker={
                    "color": class_pct,
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": 1,
                    "colorbar": {
                        "title": "% Images",
                        "tickformat": ".0%",
                    },
                },
                customdata=[f"{p:.0%}" for p in class_pct],
                hovertemplate="Count: %{y}<br>Presence: %{customdata}",
            )
        )
        adjust_layout(
            fig,
            "Class Frequency",
            "Classes",
            "Number Images",
            width=2000 if self.n_classes > 50 else 1000,
        )
        save_figure(fig, Path(path) / f"{name}_Class_Frequency")

        # --- Size Plots --- #

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=categories,
                y=class_size,
                marker_color="#0059a0",
            )
        )
        adjust_layout(
            fig,
            "Average Class Size",
            "Classes",
            "Number Pixels/Voxels",
            width=2000 if self.n_classes > 50 else 1000,
        )
        save_figure(fig, Path(path) / f"{name}_Avg_Class_Size")

        # --- Size - Frequency Plot --- #
        colors = get_colormap("tab10", len(class_cnt), as_uint=True)
        fig = go.Figure()
        for cnt, size, legend_name, col in zip(
            class_cnt, class_size, categories, colors, strict=False
        ):
            fig.add_trace(
                go.Scatter(
                    x=[cnt],
                    y=[size],
                    mode="markers",
                    name=legend_name,  # ← legend label
                    marker={
                        "size": 15,
                        "color": f"rgb{col}",
                        "line": {
                            "width": 1,
                            "color": "black",
                        },
                    },
                    hovertemplate=(
                        f"{name}<br>" + "Frequency: %{x}<br>Avg size: %{y}<extra></extra>"
                    ),
                )
            )
        adjust_layout(
            fig,
            "Class Frequency vs. Mean Voxel/Pixel Count",
            "Number of Images Containing Class",
            "Mean Voxel/Pixel Count",
        )
        save_figure(fig, Path(path) / f"{name}_Class_Frequency_Size_Ratio")

        # --- Co-Occurrence --- #
        co_mat_norm = get_occurrence_matrix(self.n_classes, self.stats["class_ids"])

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=co_mat_norm,
                x=[str(i) for i in range(self.n_classes)],
                y=[str(i) for i in range(self.n_classes)],
                colorscale="Viridis",
                colorbar={"title": "Relative co-occurrence"},
                hovertemplate="Classes %{y} & %{x}<br>Probability: %{z}<extra></extra>",
                zmin=0,
                zmax=1,
            )
        )
        adjust_layout(
            fig,
            "Class-to-Class Co-occurrence Matrix",
            "Class ID",
            "Class ID",
            height=2000 if self.n_classes > 50 else 1000,
            width=2000 if self.n_classes > 50 else 1000,
        )
        fig.update_layout(
            yaxis={"autorange": "reversed", "scaleanchor": "x", "constrain": "domain"},
            xaxis={"scaleanchor": "y", "constrain": "domain"},
        )
        save_figure(fig, Path(path) / f"{name}_Matrix")
