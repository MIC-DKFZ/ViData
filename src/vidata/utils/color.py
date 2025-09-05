import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.color import rgb2lab
from sklearn.metrics import pairwise_distances


def rgb_to_lab(rgb_colors):
    """Convert RGB colors in [0, 1] to Lab."""
    rgb_colors = np.array(rgb_colors).reshape(-1, 1, 3)
    lab_colors = rgb2lab(rgb_colors).reshape(-1, 3)
    return lab_colors


def extend_palette(
    base_colors,
    target_size,
    candidate_pool_size=10000,
    seed=42,
    min_saturation=0.15,
    max_saturation=0.85,
    min_lum=0.1,
    max_lum=0.9,
):
    """
    Extend a list of RGB base_colors with perceptually distinct and vivid colors.

    Parameters:
        base_colors (List of tuples): Initial RGB values [(r, g, b), ...] in [0,1]
        target_size (int): Total number of colors desired
        candidate_pool_size (int): Number of candidate colors to sample
        seed (int): Random seed for reproducibility
        min_saturation (float): Minimum std deviation across RGB to consider colorful
        min_lum (float): Minimum brightness
        max_lum (float): Maximum brightness

    Returns:
        List of RGB tuples with length `target_size`
    """
    np.random.seed(seed)
    base = np.array(base_colors)
    import skimage

    # Step 1: Generate a pool of candidate RGB values
    candidates = np.random.rand(candidate_pool_size, 3)

    # Step 2: Filter candidates by saturation and luminance
    sat_mask = (candidates.std(axis=1) > min_saturation) & (candidates.std(axis=1) < max_saturation)
    lum = candidates @ [0.299, 0.587, 0.114]  # Perceptual luminance
    lum_mask = (lum > min_lum) & (lum < max_lum)

    candidates = candidates[sat_mask & lum_mask]
    if len(candidates) == 0:
        raise ValueError("No candidates passed the saturation/luminance filters.")
    # Step 3: Greedily add the most distinct colors until target_size is reached
    while len(base) < target_size:
        dist = pairwise_distances(
            skimage.color.rgb2lab(candidates), skimage.color.rgb2lab(base), metric="euclidean"
        )
        min_dist = dist.min(axis=1)
        idx = np.argmax(min_dist)
        base = np.vstack([base, candidates[idx]])
        candidates = np.delete(candidates, idx, axis=0)
        if len(candidates) == 0:
            raise ValueError("Not enough candidates are available.")

    return [tuple(map(float, c)) for c in base]


def get_colormap(name: str, n_colors: int, as_uint: bool = False):
    """
    Returns a list of RGB color tuples from a colormap name, supporting both discrete (qualitative)
    and continuous palettes. If the base palette has fewer colors than `n_colors`, it will be
    extended with perceptually distinct colors.

    Parameters:
        name (str): Name of the seaborn or matplotlib colormap (e.g. "tab10", "viridis", "pastel").
        n_colors (int): Desired number of colors.

    Returns:
        List[Tuple[float, float, float]]: List of RGB color tuples in [0, 1] range.

    Behavior:
        - If the colormap is qualitative (categorical):
            - If it has more colors than requested, it's truncated.
            - If it has fewer colors than requested, it is extended using `extend_palette`.
        - If the colormap is continuous, it is sampled evenly to get `n_colors` values.
    """
    # Check if name is from one of seaborn's qualitative palettes
    if name in sns.palettes.QUAL_PALETTES:
        colors = sns.color_palette(name)
        if len(colors) > n_colors:
            colors = colors[:n_colors]
        elif len(colors) < n_colors:
            # import glasbey
            # colors=glasbey.extend_palette(colors, palette_size=n_colors)
            colors = extend_palette(colors, n_colors)
    else:
        # Treat as continuous: sample evenly n_colors values from the palette
        colors = sns.color_palette(name, n_colors)
    if as_uint:
        colors = [tuple(int(255 * c) for c in col) for col in colors]

    return colors


def viz_colormap(colors):
    n = len(colors)
    fig, ax = plt.subplots(figsize=(3, n * 0.4))  # taller for more colors
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)

    # Plot each color as a horizontal bar
    for i, color in enumerate(colors):
        ax.axhspan(i, i + 1, color=color)
        ax.text(-0.05, i + 0.5, str(i), va="center", ha="right", fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Colormap Preview", fontsize=12)
    ax.invert_yaxis()  # Optional: top-down indexing
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    colors = get_colormap("tab10", 25)
    viz_colormap(colors)
