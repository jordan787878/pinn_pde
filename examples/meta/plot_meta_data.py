import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

DATA_FOLDERS = ["data_1dnonlinear"]
SYSTEM_LABELS = ['1D-NL']
N_dataset = 1
width = 0.01
DATA_POINST = [31]
PLOT_LOSS_THRESHOLD = [0.02]
palette = sns.color_palette("viridis", N_dataset)


def plot_a1_data(axs, data_folder, number_of_data, plot_color, plot_threshold, sys_label):
    # load data
    x_display = []
    y_display = []
    has_labeled = False
    for i in range(1, number_of_data+1):
        data_i = np.load(data_folder+'/a1_data_'+str(i)+'.npz')
        loss = data_i['loss']
        if(loss < plot_threshold):
            a1_data   = data_i['a1_data']
            axs.scatter(loss, np.max(a1_data), s=15, color=plot_color)
            x_center = loss
            x_l = x_center/(10**width)
            x_r = x_center*(10**width)
            axs.plot([x_l, x_r], [np.mean(a1_data), np.mean(a1_data)], color=plot_color, linewidth=1)
            if(has_labeled == False):
                axs.plot([x_l, x_r], [np.mean(a1_data), np.mean(a1_data)], color=plot_color, linewidth=1, label=sys_label)
                has_labeled = True
            axs.add_patch(Rectangle((x_l, np.mean(a1_data)-np.std(a1_data)), x_r-x_l, 2*np.std(a1_data),
                edgecolor = plot_color,
                facecolor = 'none',
                fill=False,
                lw=1))
            x_display.append(x_center)
            y_display.append(np.mean(a1_data))
    axs.plot(x_display, y_display, color=plot_color, linestyle=":", linewidth=1.0)


def main():
    # Get the last 3 colors from the "hls" palette with 8 colors
    # colors = sns.color_palette("hls", 8)[-3:]  # Indexes 5, 6, 7 (last three)   

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 20,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 20,              # Title font size
    "axes.labelsize": 20,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 20,             # Legend text size
    "legend.title_fontsize": 20        # Legend title size (if you use legend titles)
    })

    fig, axs = plt.subplots(1, 1, figsize=(10, 7))

    for i in range(N_dataset):
        plot_a1_data(axs, DATA_FOLDERS[i], DATA_POINST[i], palette[i], PLOT_LOSS_THRESHOLD[i], SYSTEM_LABELS[i])

    # axs.set_xlim([0.0, 0.1])
    axs.set_ylim([0, 5])
    axs.set_xscale('log')
    axs.legend(loc="upper right", framealpha=0.3)
    axs.set_xlabel(r'$\hat{e}_1$'+' Train loss')
    axs.set_ylabel(r'$\alpha_1$' )
    axs.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid

    # Directly set x-ticks to powers of 10
    xticks = [10**(-3), 10**(-2)]  # Example: 10^-3, 10^-2, 10^-1, 10^0

    # Set the x-ticks to these values
    plt.xticks(xticks)

    # Use FuncFormatter to format the tick labels as LaTeX-style scientific notation
    def scientific_formatter(x, pos):
        return f'$10^{{{int(np.log10(x))}}}$'  # Format as $10^{-3}$, $10^{-2}$, ...

    plt.gca().xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('meta_plot.pdf', format='pdf', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
