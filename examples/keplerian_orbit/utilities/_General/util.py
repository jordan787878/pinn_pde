import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys, time


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


def p_rel_worst_error(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    max_p2 =  np.max(p2).item()
    if(max_p2 > 0):
        max_diff = np.max(np.abs(p1-p2)).item()
        # print(max_diff, max_p2)
        rel_error = max_diff / max_p2
        return 100.*rel_error
    else:
        return np.NaN
    

def p_total_variation(p1, p2, vol_est, verbose=False, eps=np.finfo(np.float32).tiny):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    diff = np.abs(p1-p2)
    diff[diff < eps] = 0.0 # Set values below threshold to zero
    tv = 0.5 * np.mean(diff) * vol_est
    return 100. *tv.item()


colors_6set = sns.color_palette([
    "#000000",  
    "#8C00FF",  
    "#00FF1E",  # orange
    "#FF008C",  # purple
    "#00FBFF",  # green
    "#FF8400",  # brown
    "#999999",  # gray
])


def plot_training_history(model_paths: dict, palette: str = "husl", model="p_net.pth"):
    """
    model_paths: dict like {"PNet_XL_V1": "output/.../best.pt", ...}
    palette: seaborn palette name (e.g., 'mako', 'rocket', 'flare', 'viridis', ...)
    """
    set_publication_plot_style()
    colors = sns.color_palette(palette, n_colors=len(model_paths))

    for (label, path), color in zip(model_paths.items(), colors):
        y = np.asarray(torch.load(path+"/"+model, map_location="cpu")["loss_history"], dtype=float)
        x = np.arange(len(y))
        plt.plot(x, y, label=label, color=color, alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.show()


def save_config_human(config, model_name):
    out = Path(config["save_path"]); out.mkdir(parents=True, exist_ok=True)
    # Plain text (ultra-readable)
    txt_path  = out / f"{model_name}-config.txt"
    with open(txt_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")


class RunLogger:
    """
    Redirects stdout/stderr to a file. If tee=True, also keeps console output.
    Usage:
        with RunLogger("output/pinn-xl/train.log", tee=True):
            ... your code ...
    Or:
        lg = RunLogger("output/pinn-xl/train.log", tee=True); lg.start()
        ... your code ...
        lg.stop()
    """
    def __init__(self, log_path, tee: bool = True, header: bool = True):
        self.log_path = Path(log_path)
        self.tee = bool(tee)
        self.header = bool(header)
        self._old_out = None
        self._old_err = None
        self._fh = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered file
        self._fh = open(self.log_path, "a", buffering=1)
        if self.header:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"\n===== logging start {ts} =====\n")

        self._old_out, self._old_err = sys.stdout, sys.stderr

        if self.tee:
            class _Tee:
                def write(_, s):
                    self._old_out.write(s)
                    self._fh.write(s)
                def flush(_):
                    self._old_out.flush()
                    self._fh.flush()
            sys.stdout = sys.stderr = _Tee()
        else:
            sys.stdout = sys.stderr = self._fh

    def stop(self):
        if self._fh is None:
            return
        if self.header:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"===== logging end   {ts} =====\n")
        # restore
        if self._old_out is not None: sys.stdout = self._old_out
        if self._old_err is not None: sys.stderr = self._old_err
        self._fh.close()
        self._fh = None

    # context-manager API
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.stop()
        # don't suppress exceptions
        return False

