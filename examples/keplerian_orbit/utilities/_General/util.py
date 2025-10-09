import numpy as np
import matplotlib.pyplot as plt


def compute_volume(bounds):
    """
    Compute the volume of an axis-aligned hyper-rectangle
    """
    # Compute the length of the interval in each dimension.
    side_lengths = bounds[:, 1] - bounds[:, 0]
    
    # Volume is the product of all side lengths.
    volume = np.prod(side_lengths)
    return volume


def get_valid_target_bounds(domain_bounds, target_bounds):
    """
    Compute the intersection of target_bounds and domain_bounds.
    """
    # Compute the valid lower and upper bounds per dimension
    valid_lower = np.maximum(domain_bounds[:, 0], target_bounds[:, 0])
    valid_upper = np.minimum(domain_bounds[:, 1], target_bounds[:, 1])
    # Check for a valid intersection in each dimension
    if np.any(valid_lower > valid_upper):
        return None
    # Stack the lower and upper bounds to form a valid bounds array.
    valid_target_bounds = np.stack([valid_lower, valid_upper], axis=1)
    return valid_target_bounds


def save_metrics_npz(metrics, path):
    # convert lists to float arrays; keep 't' as-is if already np.ndarray
    out = {}
    for k, v in metrics.items():
        if k == "t":
            out[k] = np.asarray(v)
        else:
            out[k] = np.asarray(v, dtype=float)
    # sanity: all series (except t) should match len(t)
    n = len(out["t"])
    for k, v in out.items():
        if k != "t":
            assert len(v) == n, f"Length mismatch for {k}: {len(v)} vs t={n}"
    np.savez(path, **out)


def load_metrics_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def set_publication_plot_style(font_family='Times New Roman', font_size=18):
    """
    Publication-ready Matplotlib defaults with safe spacing to prevent label/tick overlap.
    """
    plt.rcParams.update({
        # Typography
        'font.family': font_family,
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
        'lines.linewidth': 2,

        # Spacing (the important bits)
        'figure.constrained_layout.use': True,     # auto-avoid overlaps
        'figure.constrained_layout.h_pad': 0.05,   # inch padding between rows
        'figure.constrained_layout.w_pad': 0.05,   # inch padding between cols
        'figure.constrained_layout.hspace': 0.10,  # additional height space
        'figure.constrained_layout.wspace': 0.10,  # additional width space

        # Extra padding around text/ticks
        'axes.labelpad': 8,        # space between axis and its label (pts)
        'axes.titlepad': 10,       # space between axes and title (pts)
        'xtick.major.pad': 6,      # tick label padding (pts)
        'ytick.major.pad': 6,

        # When saving, keep the tight layout
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
