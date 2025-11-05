import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vidata.analysis.base_analyzer import Analyzer
from vidata.analysis.utils import gather_shape_stats
from vidata.analysis.viz_utils import adjust_layout, save_figure
from vidata.file_manager import FileManager
from vidata.io import save_json
from vidata.loaders import BaseLoader
from vidata.utils.color import get_colormap
from vidata.utils.multiprocess import multiprocess_iter


def get_spatial_dims(shape: tuple[int, ...], num_channels: int = 1):
    """
    Return only the spatial dimensions (no channel axis).

    Works for shapes
      (H, W)                2-D grayscale
      (H, W, C)             2-D multi-channel
      (D, H, W)             3-D grayscale
      (D, H, W, C)          3-D multi-channel
    and handles the singleton-channel ambiguity when `num_channels == 1`.
    """
    if shape[-1] == num_channels:
        dims = shape[:-1]
    elif num_channels == 1:
        dims = shape
    else:
        raise ValueError(f"Shape {shape} and num_channels {num_channels} mismatch")
    return dims


class ImageAnalyzer(Analyzer):
    def __init__(self, data_loader: BaseLoader, file_manager: FileManager, nchannels: int):
        self.data_loader = data_loader
        self.file_manager = file_manager
        self.nchannels = nchannels
        self.stats = None
        self.global_stats = None

    def analyze_case(self, file, verbose=False):
        # file = self.file_manager[index]
        data, meta = self.data_loader.load(file)
        data = data[...]  # To resolve memmap dtypes
        stats = {
            "name": file.name,
            "dtype": str(data.dtype),
            "shape": data.shape,
        }
        # Add Dummy dimension if needed
        if (
            self.nchannels is None
            or self.nchannels == 0
            or (self.nchannels == 1 and data.shape[-1] != 1)
        ):
            data = np.expand_dims(data, axis=-1)

        stats_axes = tuple(range(data.ndim - 1))
        stats["min"] = np.min(data, axis=stats_axes).tolist()
        stats["max"] = np.max(data, axis=stats_axes).tolist()
        stats["mean"] = np.mean(data, axis=stats_axes, dtype=np.float64).tolist()
        stats["std"] = np.std(data, axis=stats_axes, dtype=np.float64).tolist()
        stats["median"] = np.median(data, axis=stats_axes).tolist()

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
            desc="Analyzing Image Data",
        )
        self.stats = pd.DataFrame(stats)

        return self.stats

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stats.to_csv(path)

    def load(self, path):
        self.stats = pd.read_csv(path)

        self.stats["shape"] = self.stats["shape"].apply(ast.literal_eval)
        self.stats["min"] = self.stats["min"].apply(ast.literal_eval)
        self.stats["max"] = self.stats["max"].apply(ast.literal_eval)
        self.stats["mean"] = self.stats["mean"].apply(ast.literal_eval)
        self.stats["std"] = self.stats["std"].apply(ast.literal_eval)
        self.stats["median"] = self.stats["median"].apply(ast.literal_eval)
        if "spacing" in self.stats.columns:
            self.stats["spacing"] = self.stats["spacing"].apply(ast.literal_eval)
        return self.stats

    def aggregate(self, path=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        global_stats = {}

        stats = self.stats.copy()
        stats["spatial_dims"] = stats["shape"].apply(lambda s: get_spatial_dims(s, self.nchannels))
        stats["voxels_per_channel"] = stats["spatial_dims"].apply(lambda s: np.prod(s))

        m = np.stack(stats["mean"].to_numpy())  # shape: (n_images, C)
        s = np.stack(stats["std"].to_numpy())  # shape: (n_images, C)
        N = np.stack(stats["voxels_per_channel"].to_numpy())  # shape: (n_images, C)
        N = np.expand_dims(N, axis=1)

        # ---------- overall (dataset-level) MEAN ----------
        overall_mean = (m * N).sum(axis=0) / N.sum(axis=0)  # shape: (C,)
        # ---------- overall (dataset-level) STD ----------
        # 1. pooled sum of squares of deviations within each image (= (N_i - 1)*s_i²)
        within_var = ((N - 1) * (s**2)).sum(axis=0)

        # 2. between-image variance of their means
        between_var = (N * (m - overall_mean) ** 2).sum(axis=0)

        # 3. pooled variance, then √ → std
        pooled_var = (within_var + between_var) / (N.sum(axis=0) - 1)
        overall_std = np.sqrt(pooled_var)  # shape: (C,)
        global_stats["intensity"] = {
            "mean": overall_mean.tolist(),
            "std": overall_std.tolist(),
        }

        global_stats["shape"] = gather_shape_stats(stats["spatial_dims"])

        if "spacing" in stats.columns:
            # stats["spacing"] = stats["spacing"].apply(ast.literal_eval)
            spacing_array = np.array(stats["spacing"].tolist())  # shape: (N, D)
            global_stats["spacing"] = {"mean": spacing_array.mean(axis=0).tolist()}

        if path is not None:
            save_json(global_stats, path)

        return global_stats

    def plot(self, path, name=""):
        Path(path).mkdir(parents=True, exist_ok=True)
        stats = self.stats.copy()
        names = stats["name"].to_numpy()

        means_df = pd.DataFrame(stats["mean"].tolist())
        means_df.columns = [f"Channel {i}" for i in means_df.columns]

        std_df = pd.DataFrame(stats["std"].tolist())
        std_df.columns = [f"Channel {i}" for i in std_df.columns]
        # --- Intensity Plots --- #
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Per-Image Channel Means", "Per-Image Channel Standard Deviations"),
        )

        channels = means_df.columns  # e.g. ['Channel 0', …]
        color = get_colormap("tab10", len(channels), as_uint=True)

        for i, ch in enumerate(means_df.columns):
            fig.add_trace(
                go.Box(
                    y=means_df[ch],
                    name=ch,
                    customdata=np.expand_dims(names, axis=1),
                    marker_color=f"rgb{color[i]}",
                    # boxpoints="outliers",  # show individual points
                    boxpoints="all",  # show individual points
                    jitter=0.3,
                    pointpos=-1.8,
                    boxmean=True,
                    hovertemplate=("Mean: %{y:.2f}<br>" + "Image: %{customdata[0]}"),
                ),
                row=1,
                col=1,
            )

        for i, ch in enumerate(std_df.columns):
            fig.add_trace(
                go.Box(
                    y=std_df[ch],
                    name=ch,
                    customdata=np.expand_dims(names, axis=1),
                    marker_color=f"rgb{color[i]}",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    boxmean=True,
                    hovertemplate=("Std: %{y:.2f}<br>" + "Image: %{customdata[0]}<extra></extra>"),
                ),
                row=1,
                col=2,
            )

        adjust_layout(fig, "Distribution of Per-Image Channel Intensity Statistics", subplots=True)
        fig.update_layout(
            showlegend=False,
        )
        fig.update_yaxes(title_text="Mean Intensity", row=1, col=1)
        fig.update_yaxes(title_text="Std. Dev.", row=1, col=2)
        save_figure(fig, Path(path) / f"{name}_Intensity_Distribution")

        # --- Shape Plots --- #
        stats["spatial_dims"] = stats["shape"].apply(lambda s: get_spatial_dims(s, self.nchannels))
        shape_counts = stats["spatial_dims"].value_counts()
        shapes = np.array(shape_counts.index.tolist())  # list of tuples
        counts = np.array(shape_counts.values)
        counts_ptc = counts / len(stats)
        is_3d = len(shapes[0]) == 3
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene" if is_3d else "xy"}, {"type": "xy"}]],
            subplot_titles=["Shape Distribution (Scatter)", "Shape Distribution(Bar)"],
            column_widths=[0.4, 0.6] if len(shapes) > 50 else None,
        )

        marker_size = np.log(counts) * 11
        marker_size[marker_size == 0] = 5
        if not is_3d:
            fig.add_trace(
                go.Scatter(
                    x=shapes[:, 1],
                    y=shapes[:, 0],
                    mode="markers",
                    marker_size=marker_size,
                    showlegend=False,
                    marker={
                        "size": [3 + f * 2 for f in counts],  # scale size visually
                        "color": counts_ptc,
                        "colorscale": "Viridis",
                        "cmin": 0,
                        "cmax": 1,
                        "showscale": True,
                        "colorbar": {"title": "% Images"},
                        # "opacity": 0.8,
                    },
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="Height", row=1, col=1)
            fig.update_xaxes(title_text="Width", row=1, col=1)

        else:
            # (D, H, W)
            fig.add_trace(
                go.Scatter3d(
                    x=shapes[:, 2],
                    y=shapes[:, 1],
                    z=shapes[:, 0],
                    mode="markers",
                    marker_size=marker_size,
                    showlegend=False,
                    marker={
                        "size": [3 + f * 2 for f in counts],  # scale size visually
                        "color": counts_ptc,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "cmin": 0,
                        "cmax": 1,
                        "colorbar": {"title": "% Images"},
                        "opacity": 0.8,
                    },
                ),
                row=1,
                col=1,
            )
            fig.update_layout(
                scene={
                    "xaxis_title": "Width",
                    "yaxis_title": "Height",
                    "zaxis_title": "Depth",
                    # "aspectmode": "data",
                }
            )

        fig.add_trace(
            go.Bar(
                x=[str(tuple([int(si) for si in s])) for s in shapes],
                y=counts,
                showlegend=False,
                marker={
                    "color": counts_ptc,
                    "cmin": 0,
                    "cmax": 1,
                    "colorscale": "Viridis",
                },
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Number Images", row=1, col=2)
        fig.update_xaxes(title_text="Shape(s)", row=1, col=2)

        adjust_layout(
            fig, "Shape Distribution", subplots=True, width=2000 if len(counts) > 50 else 1000
        )

        save_figure(fig, Path(path) / f"{name}_Shape_Distribution")

        # --- Spacing Plots (Optional)--- #
        if "spacing" in stats.columns:
            # stats["spacing"] = stats["spacing"].apply(ast.literal_eval)
            spacing_df = pd.DataFrame(stats["spacing"].tolist())
            spacing_df.columns = [f"Dimension {i}" for i in spacing_df.columns]
            fig = go.Figure()
            for ch in spacing_df.columns:
                fig.add_trace(
                    go.Box(
                        y=spacing_df[ch],
                        name=ch,
                        marker_color="#0059a0",
                        customdata=np.expand_dims(names, axis=1),
                        boxpoints="all",  # show individual points
                        jitter=0.3,
                        pointpos=-1.8,
                        boxmean=True,
                        showlegend=False,
                        hovertemplate=("Spacing: %{y:.2f}<br>" + "Image: %{customdata[0]}"),
                    )
                )
            adjust_layout(fig, "Spacing Distribution")
            save_figure(fig, Path(path) / f"{name}_Spacing_Distribution")
