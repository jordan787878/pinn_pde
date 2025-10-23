import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from pathlib import Path
import sys, time
from cycler import cycler


colors_6set = sns.color_palette([
    "#000000",  
    "#8C00FF",  
    "#00FF1E",  # orange
    "#FF008C",  # purple
    "#00FBFF",  # green
    "#FF8400",  # brown
    "#999999",  # gray
])

# 6 high-contrast linestyle/marker pairs (index-bound)
linestyles_6set = [
    (0, (5, 2)),        # custom medium dashes
    (0, (7, 2, 3, 2)),  # custom long–short dash
    "--",               # dashed
    "-.",               # dash-dot
    "-",                # solid
    ":",                # dotted
]
markers_6set = ['o', 's', '^', 'D', 'None', 'X']  # circle, square, up-tri, diamond, down-tri, bold X


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


def set_publication_plot_style(font_family='Times New Roman', font_size=18,
                               sci_power=(-2, 2), tick_pad=6,
                               legend_loc='upper left', legend_frame=True,
                               pair_line_marker_cycle=False):
    """
    Publication-ready Matplotlib defaults with consistent tick formatting.
    """
    mpl.rcdefaults()       # reset rcParams to built-in defaults
    plt.style.use('default')  # ensure default style (no external style lingering)
    
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
        'lines.markersize': 7,
        'lines.markeredgewidth': 1.5,
        'lines.markerfacecolor': 'none',   # hollow markers = great contrast

        # Tick appearance & alignment
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.pad': tick_pad,
        'ytick.major.pad': tick_pad,
        'xtick.top': False,     # corner/pair plots usually cleaner without top/right ticks
        'ytick.right': False,

        # ---------- Legend defaults ----------
        'legend.loc': legend_loc,
        'legend.frameon': legend_frame,
        'legend.borderaxespad': 0.4,   # padding between legend and axes
        'legend.borderpad': 0.3,       # padding inside the legend box
        'legend.handlelength': 1.8,    # line handle length
        'legend.handletextpad': 0.6,   # space between handle and text
        'legend.columnspacing': 0.8,
        'legend.labelspacing': 0.4,
        'legend.markerscale': 2.0,     # scale marker size in legend
        # 'legend.numpoints': 1,        # uncomment for single-point line legends

        # Scientific notation like the helper used
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': sci_power,  # switch to sci notation outside these powers
        'axes.formatter.useoffset': False,   # avoid confusing 1eN offsets on axes
        'axes.unicode_minus': False,         # minus sign renders consistently with mathtext

        # Spacing / layout
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.05,
        'figure.constrained_layout.w_pad': 0.05,
        'figure.constrained_layout.hspace': 0.10,
        'figure.constrained_layout.wspace': 0.10,
        'axes.labelpad': 8,
        'axes.titlepad': 10,

        # ---------- Grid defaults ----------
        'axes.grid': True,            # turn grid on by default
        'axes.grid.axis': 'both',     # x and y
        'axes.grid.which': 'major',   # grid for major ticks (change to 'both' if desired)
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,

        # Save tight
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Figure size
        'figure.figsize': (10, 8),
    })

    if pair_line_marker_cycle:
        plt.rc('axes', prop_cycle=cycler(linestyle=linestyles_6set) + cycler(marker=markers_6set))


def apply_default_locators(ax_or_fig, max_ticks=4):
    """
    Apply MaxNLocator(max_ticks) + ScalarFormatter(useMathText, powerlimits from rcParams)
    to all axes in a Figure or a single Axes. Call once after plotting.
    """
    axes = []
    if hasattr(ax_or_fig, 'get_axes'):  # Figure
        axes = [a for a in ax_or_fig.get_axes() if a.name == 'axes']
    else:  # single Axes
        axes = [ax_or_fig]

    for ax in axes:
        # Major tick density
        ax.xaxis.set_major_locator(mticker.MaxNLocator(max_ticks))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(max_ticks))

        # Formatter consistent with rcParams (mathtext + powerlimits)
        xf = mticker.ScalarFormatter(useMathText=plt.rcParams['axes.formatter.use_mathtext'])
        yf = mticker.ScalarFormatter(useMathText=plt.rcParams['axes.formatter.use_mathtext'])
        xf.set_powerlimits(plt.rcParams['axes.formatter.limits'])
        yf.set_powerlimits(plt.rcParams['axes.formatter.limits'])
        xf.set_useOffset(plt.rcParams['axes.formatter.useoffset'])
        yf.set_useOffset(plt.rcParams['axes.formatter.useoffset'])
        ax.xaxis.set_major_formatter(xf)
        ax.yaxis.set_major_formatter(yf)


def custom_save_plot(save_plot, save_path):
    if(save_plot):
        plt.savefig(save_path, format="pdf", dpi=300)
        print("[info] save fig to path: ", save_path)


def tidy_corner_axes(fig, axes, labels=None, max_ticks=4, labelsize=9, pad=2, sci_power=(-2, 2)):
    """
    Make axes and labels/ticks line up nicely on a corner/pair plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axes : 2D numpy array of Axes (e.g., from plt.subplots / PairGrid / corner.corner)
    labels : list[str] or None
        If given, sets x-labels on bottom row and y-labels on left column.
    max_ticks : int
        Max number of ticks on each axis.
    labelsize : int
        Tick label size.
    pad : float
        Tick label pad.
    sci_power : tuple[int, int]
        Power limits for switching to scientific notation (ScalarFormatter).
    """
    n, m = axes.shape

    # Share tick formatters/locators
    for i in range(n):
        for j in range(m):
            ax = axes[i, j]
            if ax is None:
                continue

            # show tick labels only on outer axes
            ax.label_outer()

            # consistent tick density & style
            ax.xaxis.set_major_locator(mticker.MaxNLocator(max_ticks))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(max_ticks))
            xf = mticker.ScalarFormatter(useMathText=True)
            yf = mticker.ScalarFormatter(useMathText=True)
            xf.set_powerlimits(sci_power)
            yf.set_powerlimits(sci_power)
            ax.xaxis.set_major_formatter(xf)
            ax.yaxis.set_major_formatter(yf)

            ax.tick_params(axis="both", direction="in", pad=pad, labelsize=labelsize)

    # put labels only where they align visually
    if labels:
        for j in range(m):
            axes[-1, j].set_xlabel(labels[j])
        for i in range(n):
            axes[i, 0].set_ylabel(labels[i])

    # tighten spacing & align label baselines
    try:
        fig.align_labels()        # matplotlib ≥3.4 aligns across subplots
    except Exception:
        pass
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    fig.canvas.draw_idle()


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

