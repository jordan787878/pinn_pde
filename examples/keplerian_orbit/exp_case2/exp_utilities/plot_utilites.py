import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import seaborn as sns
import torch
from scipy.interpolate import griddata


def set_publication_plot_style(font_family='Times New Roman', font_size=18):
    """
    Update Matplotlib settings to use publication-ready fonts.

    Parameters:
        font_family (str): Font family to be used for all texts.
        font_size (int): Base font size for labels, titles, legends, and ticks.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = font_size
    plt.rcParams['lines.linewidth'] = 2


def visual_phat_trainings(constants, p_net, data_foler, save_plots=False, save_plot_path=None):
    """
    visualize phat training results using final or intermediate saved model
    """

    set_publication_plot_style()

    # Create a figure with a black background
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Adjust the axis panes and tick labels to be visible on a black background
    ax.xaxis.pane.set_facecolor('black')
    ax.yaxis.pane.set_facecolor('black')
    ax.zaxis.pane.set_facecolor('black')
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # t_span = simple_interpolate(constants.T_PRIME_SPAN)
    t_span = constants.T_PRIME_SPAN
    for t_prime in t_span:
    # for t_prime in np.linspace(constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1], num=6):
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(data_foler+"x1s.npy")
        x2s = np.load(data_foler+"x2s.npy")
        x3s = np.load(data_foler+"x3s.npy")
        x4s = np.load(data_foler+"x4s.npy")

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx1 = x1s[1] - x1s[0] # dr'
        dx2 = x2s[1] - x2s[0] # dphi'
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # load pdf_monte on the domain
        t_label = t_prime
        if(t_prime in constants.T_PRIME_SPAN):
            pdf_monte = np.load(data_foler+"pdf_t{:.3f}.npy".format(t_prime))
            pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        if(t_label in constants.T_PRIME_SPAN):
            pdf_mo_rphi_data = np.empty((0,4))
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                if(t_label in constants.T_PRIME_SPAN):
                    pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        if(t_label in constants.T_PRIME_SPAN):
            z_mo = pdf_mo_rphi_data[:, 3]/(r) 

        # _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        # __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        # ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        # print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 50
        num_strides = 5
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), 
                                    np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        if(t_label in constants.T_PRIME_SPAN):
            grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == t_span[0]):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8, label="NN")
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-", label="MC")
                x_text = (grid_x.min() + grid_x.max()) / 2
                y_text = (grid_y.min() + grid_y.max()) / 2
                ax.text(x_text, y_text, 1.1*np.max(z_nn), f"t={t_label:.2f}", color="white", fontsize=14)

        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8)
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-")
                x_text = (grid_x.min() + grid_x.max()) / 2
                y_text = (grid_y.min() + grid_y.max()) / 2
                ax.text(x_text, y_text, 1.1*np.max(z_nn), f"t={t_label:.2f}", color="white", fontsize=14)
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Optional: Add a color bar
    # Labels and title
    ax.legend()
    ax.view_init(40, -133) # viewing angle
    ax.set_xlabel('X, m', color='white', labelpad=10)
    ax.set_ylabel('Y, m', color='white', labelpad=10)
    ax.set_zlabel('PDF', color='white', labelpad=10)
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))

    # Save the figure as a PDF, with minimal extra margins and no clipping of labels
    plt.tight_layout(pad=0.3)
    if(save_plots):
        if(save_plot_path is not None):
            print("Save plot to: ", save_plot_path)
            fig.savefig(save_plot_path, format='pdf')
    else:
        plt.show()


def visual_e1hat_training(constants, networks, data_foler, save_plots=False, save_plot_path=None):
    """
    visualize e1 networks training results using final or intermediate saved model
    NOTE: select the t_span
    """
    p_net, e1_net_seq1, e1_net_seq2 = networks

    set_publication_plot_style()

    x1s = np.load(data_foler+"x1s.npy")
    x2s = np.load(data_foler+"x2s.npy")
    x3s = np.load(data_foler+"x3s.npy")
    x4s = np.load(data_foler+"x4s.npy")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    ymin = [-0.025, -0.070]
    ymax = [0.019,  0.042]

    t_span = [constants.T_PRIME_SPAN[2], constants.T_PRIME_SPAN[-1]]
    for i in range(len(t_span)):
        t_prime = t_span[i]
        pdf_true = np.load(data_foler+"pdf_t{:.3f}.npy".format(t_prime))

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        if(t_prime >= 0.5*constants.TF/constants.T):
            e1_net = e1_net_seq1
        else:
            e1_net = e1_net_seq2
        e1 = pdf_true - pdf_nn
        e1_nn  = e1_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
        
        e1_vec = e1.reshape(-1)
        e1_nn_vec = e1_nn.reshape(-1)
        a1 = np.max(np.abs(e1_vec - e1_nn_vec)) / np.max(np.abs(e1_nn_vec))
        print("a1 (t=", np.round(t_prime,2),"): ", np.round(a1,3))
        # print(E_x1, E_x2, E_x3, E_x4)
        max_e1 = np.max(np.abs(e1_vec))
        max_e1_nn = np.max(np.abs(e1_nn_vec))
        # print(max_e1, max_e1_nn)
        B1 = 2.0 * max_e1_nn

        gap = 1
        # Generate an array for the x-axis based on indices
        x = np.arange(len(e1_vec))[::gap]
        # Select every n-th element from the data array for the y-axis
        y1 = e1_vec[::gap]
        # y2 = e1_nn_vec[::gap]
        axs[i].plot(x, y1, "black", linewidth=0.5, rasterized=True, label="MC")
        # axs[i].plot(x, y2, "blue",  linewidth=0.5, rasterized=True, label="NN")
        # axs[i].fill_between(x, y1=0.0*x+B1, y2=0.0*0-B1, 
        #                     color="green", edgecolor="none", alpha=0.1, label=r"$B_1$")
        axs[i].set_ylabel("Error")
        axs[i].grid(True)
        axs[i].text(0.02, 0.98, f"t={t_prime:.2f}T", transform=axs[i].transAxes,
                    ha='left', va='top', color='black', fontsize=18,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        axs[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        # axs[i].set_ylim([ymin[i], ymax[i]])
        # ymin, ymax = axs[i].get_ylim()
        # print(f"Subplot {i}: ymin = {ymin}, ymax = {ymax}")

    axs[1].set_xlabel("4D state idx")
    axs[0].legend(loc="lower left", ncol=3)
    # fig.subplots_adjust(left=0.1)
    plt.tight_layout(pad=0.2)
    if(save_plots):
        if(save_plot_path is not None):
            print("Save plot to: ", save_plot_path)
            fig.savefig(save_plot_path, format='pdf')
    else:
        plt.show()


def plot_app1_onlymc(constants, data_foler, N_mc=4, save_plots=False, save_plot_path=None):
    """
    compare the Prob using different samples of M.C.
    """
    set_publication_plot_style()

    pr_mcs = np.load(data_foler)
    plot_labels = ["1e+8", "1e+7", "1e+6", "1e+5"]
    # pr_nn_data_labels = ["data/app1/pr_nn_Nd50_onlyphat.npy", "data/app1/pr_nn_Nd50_phat+B.npy", "data/app1/pr_nn_Nd50_LP.npy", "data/app1/pr_nn_Nd50_gmm7.npy"]
    # plot_labels = [r"$\hat{p}$",r"$\int_{X^{'}} \hat{p}+B_1 dx$", r"LP($\hat{p},B_1$)", r"F.O.($\hat{p},B_1$)"]

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", pr_mcs.shape[1])

    for j in range(N_mc):
        t_span = pr_mcs[:,0]
        pr_base = pr_mcs[:,1] 
        pr      = pr_mcs[:,j+1]

        # Create a mask to filter out NaN values.
        mask_nan = ~np.isnan(pr)
        mask_nan_based = ~np.isnan(pr_base)
        # Create a mask that selects only the time points in t_plot_list.
        mask_time = np.isin(t_span, constants.T_PRIME_SPAN)
        # Combine the two masks.
        mask = mask_nan & mask_nan_based & mask_time

        plt.plot(t_span[mask], pr[mask] - pr_base[mask], color=colors[j], linestyle="--", marker="d", label="M.C. samples="+plot_labels[j])

    plt.grid(True)
    plt.ylabel(r"$\Delta Pr(x \in X^{'})$")
    plt.xlabel("t")
    plt.legend(loc="lower right")
    # Get current axes, and then obtain and reformat the xticks.
    plt.tight_layout(pad=0.2)
    # Define the tick positions.
    ticks = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    # Create corresponding labels with "T" appended.
    tick_labels = [f"{tick:.2f}T" for tick in ticks]

    # Get the current axis.
    ax = plt.gca()
    # Set the tick positions.
    ax.set_xticks(ticks)
    # Set the tick labels.
    ax.set_xticklabels(tick_labels)
    if(save_plots):
        if(save_plot_path is not None):
            print("Save plot to: ", save_plot_path)
            fig.savefig(save_plot_path, format='pdf')
    else:
        plt.show()


def plot_app1(N_mc=1, save_plots=False, save_plot_path=None):
    set_publication_plot_style()

    parent_folder = "data/app1/tar1/prob/"
    pr_mcs = np.load(parent_folder+"pr_mcs.npy")
    
    pr_nn_data_labels = [parent_folder+"pr_nn_Nd50_onlyphat.npy", 
                         parent_folder+"pr_nn_Nd50_phat+B.npy", 
                         parent_folder+"pr_nn_Nd50_LP.npy", 
                        #  parent_folder+"pr_nn_Nd50_FOx128(new).npy",
                        #  parent_folder+"pr_nn_Nd50_gmmx64(iter-10k).npy", # weight of region_loss 1e-2
                        #  parent_folder+"pr_nn_Nd50_gmmx64(iter-20k).npy", # weight of region_loss 1e-2
                         parent_folder+"diaggmmx64.npy", # weight of region_loss 1e-1 with half random samples
                         parent_folder+"diaggmmx64(aug_vio).npy",
                         ]
    plot_labels = [r"$\hat{p}$",
                   r"$\int_{X^{'}} \hat{p}+B_1 dx$", 
                   r"LP($\hat{p},B_1$)", 
                #    r"FO($\hat{p},B_1$) RBFx256(new)",
                #    r"FO($\hat{p},B_1$) GMMx64(10k det.)",
                #    r"FO($\hat{p},B_1$) GMMx64(20k det.)",
                   r"FO($\hat{p},B_1$) GMMx64(20k)",
                   r"FO($\hat{p},B_1$) GMMx64(20k aug.)",
                   ]
    plot_fills  = [False, False, True, True, False]
    plot_style =  ["--", "-", "-", "-", "-"]

    pr_nn_data = []
    for j in range(len(pr_nn_data_labels)):
        pr_nn_data.append(np.load(pr_nn_data_labels[j]))

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", len(pr_nn_data_labels))

    for j in range(N_mc):
        t_span = pr_mcs[:,0]
        pr = pr_mcs[:,j+2] # should change back to pr = pr_mcs[:,j+1] 
        mask = ~np.isnan(pr)
        plt.plot(t_span[mask], pr[mask], color="black", linestyle="", marker="o", markersize=4, label="MC")
        # print(t_span[mask], pr[mask])

    for j in range(len(pr_nn_data_labels)):
        pr_nn_data_i = pr_nn_data[j]
        t_span = pr_nn_data_i[:,0]
        plt.plot(t_span, pr_nn_data_i[:,1], color=colors[j], linestyle=plot_style[j], label=plot_labels[j])
        if(plot_fills[j]): 
            plt.fill_between(t_span, y1=0.0*t_span, y2=pr_nn_data_i[:,1],
                             color=colors[j], edgecolor="none", alpha=0.1)

    plt.grid(True)
    plt.ylabel(r"$Pr(x \in X^{'})$")
    plt.xlabel("t")
    plt.legend(loc="upper left", ncol=2)
    plt.ylim([-0.05, 1.5])
    # Get current axes, and then obtain and reformat the xticks.
    plt.tight_layout(pad=0.2)
    # Define the tick positions.
    ticks = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    # Create corresponding labels with "T" appended.
    tick_labels = [f"{tick:.2f}T" for tick in ticks]

    # Get the current axis.
    ax = plt.gca()
    # Set the tick positions.
    ax.set_xticks(ticks)
    # Set the tick labels.
    ax.set_xticklabels(tick_labels)
    if(save_plots):
        if(save_plot_path is not None):
            print("Save plot to: ", save_plot_path)
            fig.savefig(save_plot_path, format='pdf')
    else:
        plt.show()


def plot_target(constants, p_net, target_r, target_ph, data_foler):
    # Create a figure with a black background
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    z_max = -np.inf

    # Adjust the axis panes and tick labels to be visible on a black background
    ax.xaxis.pane.set_facecolor('black')
    ax.yaxis.pane.set_facecolor('black')
    ax.zaxis.pane.set_facecolor('black')
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # t_span = simple_interpolate(constants.T_PRIME_SPAN)
    t_span = constants.T_PRIME_SPAN
    for t_prime in t_span:
    # for t_prime in np.linspace(constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1], num=6):
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(data_foler+"x1s.npy")
        x2s = np.load(data_foler+"x2s.npy")
        x3s = np.load(data_foler+"x3s.npy")
        x4s = np.load(data_foler+"x4s.npy")

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx1 = x1s[1] - x1s[0] # dr'
        dx2 = x2s[1] - x2s[0] # dphi'
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # load pdf_monte on the domain
        t_label = t_prime
        if(t_prime in constants.T_PRIME_SPAN):
            pdf_monte = np.load(data_foler+"pdf_t{:.3f}.npy".format(t_prime))
            pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        if(t_label in constants.T_PRIME_SPAN):
            pdf_mo_rphi_data = np.empty((0,4))
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                if(t_label in constants.T_PRIME_SPAN):
                    pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        if(t_label in constants.T_PRIME_SPAN):
            z_mo = pdf_mo_rphi_data[:, 3]/(r) 
            max_z_t_prime = np.max(z_mo)
            z_max = max(z_max, max_z_t_prime)

        # _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        # __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        # ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        # print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 50
        num_strides = 5
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), 
                                    np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        if(t_label in constants.T_PRIME_SPAN):
            grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == t_span[0]):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8, label="NN")
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-", label="Monte")
        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8)
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-")
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Optional: Add a color bar
    
    # Plot the target volume.
    print("z_max: ", z_max)
    target_plot = plot_target_volume(ax, target_r, target_ph, z_max)

    # Labels and title
    ax.legend()
    ax.view_init(40, -133) # viewing angle
    ax.set_xlabel('X, m', color='white')
    ax.set_ylabel('Y, m', color='white')
    ax.set_zlabel('PDF Value', color='white')
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
    # Save the figure as a PDF, with minimal extra margins and no clipping of labels
    plt.tight_layout(pad=0.3)
    # fig.savefig("figs/target_X.pdf", format='pdf')
    plt.show()


def plot_target_volume(ax, target_r, target_ph, z_max, grid_resolution=30, 
                       color='red', alpha=0.3, edge_color='gray', edge_width=2.5):
    """
    Plot an extruded 3D volume for a target region defined in spherical coordinates
    and display its edges.

    The volume is created by extruding the target region, defined by:
      - target_r: an array-like of two elements (min, max) for the radial coordinate,
      - target_ph: an array-like of two elements (min, max) for the angular coordinate (ϕ, in radians).

    The extrusion is performed from z = 0 up to z = z_max.

    Parameters:
      ax             : A matplotlib 3D axis object.
      target_r       : Array-like with two elements specifying the min and max radial limits.
      target_ph      : Array-like with two elements specifying the min and max angular limits (in radians).
      z_max          : The maximum z-value (height) for the extruded volume.
      grid_resolution: Number of grid points to use along each dimension (default: 50).
      color          : Face color for the volume surfaces (default: 'red').
      alpha          : Transparency of the volume surfaces (default: 1.0).
      edge_color     : Color to use for the edge lines (default: black).
      edge_width     : Line width for the edge lines (default: 2).

    Returns:
      surfaces: A dictionary containing:
          - 'top': top surface,
          - 'bottom': bottom surface,
          - 'side1', 'side2', 'side3', 'side4': side surfaces,
          - 'edges': a dictionary of edge line objects ('top_edge', 'bottom_edge', 'vertical_edges').
    """
    # Create grid arrays for r and ϕ.
    r_vals = np.linspace(np.min(target_r), np.max(target_r), grid_resolution)
    phi_vals = np.linspace(np.min(target_ph), np.max(target_ph), grid_resolution)
    
    # Create mesh grid for the top and bottom surfaces.
    grid_tar_r, grid_tar_ph = np.meshgrid(r_vals, phi_vals, indexing='ij')
    x_tar = grid_tar_r * np.cos(grid_tar_ph)
    y_tar = grid_tar_r * np.sin(grid_tar_ph)
    
    # Plot the top surface (z = z_max) and bottom surface (z = 0)
    surf_top = ax.plot_surface(x_tar, y_tar, np.full_like(x_tar, z_max),
                               facecolor=color, edgecolor='none', alpha=alpha)
    surf_bottom = ax.plot_surface(x_tar, y_tar, np.zeros_like(x_tar),
                                  facecolor=color, edgecolor='none', alpha=alpha)
    
    # Define vertical grid along z for the sides.
    z_side = np.linspace(0, z_max, grid_resolution)
    
    # Side 1: r = target_r.min() (vary ϕ)
    x_side1 = np.min(target_r) * np.cos(phi_vals)
    y_side1 = np.min(target_r) * np.sin(phi_vals)
    X_side1, Z_side1 = np.meshgrid(x_side1, z_side, indexing='ij')
    Y_side1, _       = np.meshgrid(y_side1, z_side, indexing='ij')
    surf_side1 = ax.plot_surface(X_side1, Y_side1, Z_side1,
                                facecolor=color, edgecolor='none', alpha=alpha, label="target region")
    
    # Side 2: r = target_r.max() (vary ϕ)
    x_side2 = np.max(target_r) * np.cos(phi_vals)
    y_side2 = np.max(target_r) * np.sin(phi_vals)
    X_side2, Z_side2 = np.meshgrid(x_side2, z_side, indexing='ij')
    Y_side2, _       = np.meshgrid(y_side2, z_side, indexing='ij')
    surf_side2 = ax.plot_surface(X_side2, Y_side2, Z_side2,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # Side 3: ϕ = target_ph.min() (vary r)
    x_side3 = r_vals * np.cos(np.min(target_ph))
    y_side3 = r_vals * np.sin(np.min(target_ph))
    X_side3, Z_side3 = np.meshgrid(x_side3, z_side, indexing='ij')
    Y_side3, _       = np.meshgrid(y_side3, z_side, indexing='ij')
    surf_side3 = ax.plot_surface(X_side3, Y_side3, Z_side3,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # Side 4: ϕ = target_ph.max() (vary r)
    x_side4 = r_vals * np.cos(np.max(target_ph))
    y_side4 = r_vals * np.sin(np.max(target_ph))
    X_side4, Z_side4 = np.meshgrid(x_side4, z_side, indexing='ij')
    Y_side4, _       = np.meshgrid(y_side4, z_side, indexing='ij')
    surf_side4 = ax.plot_surface(X_side4, Y_side4, Z_side4,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # --- Add edge lines ---
    # Compute corner coordinates in the top and bottom planes.
    r_min = np.min(target_r)
    r_max = np.max(target_r)
    phi_min = np.min(target_ph)
    phi_max = np.max(target_ph)
    
    # Four corners (completing the loop by repeating the first point)
    top_corners = np.array([
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), z_max],
        [r_min * np.cos(phi_max), r_min * np.sin(phi_max), z_max],
        [r_max * np.cos(phi_max), r_max * np.sin(phi_max), z_max],
        [r_max * np.cos(phi_min), r_max * np.sin(phi_min), z_max],
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), z_max]
    ])
    bottom_corners = np.array([
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), 0],
        [r_min * np.cos(phi_max), r_min * np.sin(phi_max), 0],
        [r_max * np.cos(phi_max), r_max * np.sin(phi_max), 0],
        [r_max * np.cos(phi_min), r_max * np.sin(phi_min), 0],
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), 0]
    ])
    
    # Draw top and bottom edges.
    edge_top, = ax.plot(top_corners[:, 0], top_corners[:, 1], top_corners[:, 2],
                        color=edge_color, lw=edge_width)
    edge_bottom, = ax.plot(bottom_corners[:, 0], bottom_corners[:, 1], bottom_corners[:, 2],
                           color=edge_color, lw=edge_width)
    
    # Draw vertical edges connecting top and bottom corners.
    vertical_edges = []
    for i in range(4):
        ve, = ax.plot([top_corners[i, 0], bottom_corners[i, 0]],
                      [top_corners[i, 1], bottom_corners[i, 1]],
                      [top_corners[i, 2], bottom_corners[i, 2]],
                      color=edge_color, lw=edge_width)
        vertical_edges.append(ve)
    
    surfaces = {
        'top': surf_top,
        'bottom': surf_bottom,
        'side1': surf_side1,
        'side2': surf_side2,
        'side3': surf_side3,
        'side4': surf_side4,
    }
    return surfaces


# --- Haven't updated ---
# def visual_fo_rbf(t, p_net, model, target_region, N_discret=50):
#     x1s = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], N_discret)
#     x2s = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], N_discret)
#     x3s = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], N_discret)
#     x4s = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], N_discret)
#     x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
#     grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
#     dx1 = x1s[1] - x1s[0]
#     dx2 = x2s[1] - x2s[0]
#     dx3 = x3s[1] - x3s[0]
#     dx4 = x4s[1] - x4s[0]
#     # obtain pdf(nn)
#     grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
#     t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
#     p0 = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
#     # plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     p0_2D =  np.sum(p0, axis=(2,3)) * dx3 * dx4
#     p_rbf = model(grid_points_tensor).detach().numpy().reshape(x1_grid.shape)
#     p_rbf_2D = np.sum(p_rbf, axis=(2,3)) * dx3 * dx4
#     X, Y =  np.meshgrid(x1s, x2s, indexing="ij")
#     ax.plot_surface(X, Y, p0_2D, color="none", rstride=2, cstride=2, 
#                     edgecolor='black', linewidth=0.5, label=r"$\hat{p}$")
#     ax.plot_surface(X, Y, p_rbf_2D, color="none", rstride=2, cstride=2, 
#                     edgecolor='blue', linestyle="--", linewidth=0.5, label=r"$p_{FO}$")
#     # Create the patch for the target region.
#     # Assume target_region is a 2x2 array:
#     #   target_region[0] = [r_min, r_max]
#     #   target_region[1] = [phi_min, phi_max]
#     r_bounds = target_region[0, :]    # [r_min, r_max]
#     phi_bounds = target_region[1, :]  # [phi_min, phi_max]
#     # Define the corners of the rectangular patch at z = 0.
#     patch_vertices = [
#         [r_bounds[0], phi_bounds[0], 0],
#         [r_bounds[1], phi_bounds[0], 0],
#         [r_bounds[1], phi_bounds[1], 0],
#         [r_bounds[0], phi_bounds[1], 0],
#     ]
#     # Create a Poly3DCollection and add it to the 3D axis.
#     patch = Poly3DCollection([patch_vertices], facecolor='red', alpha=0.5, edgecolor='k', label=r"$X^'_{tar}$")
#     ax.add_collection3d(patch)
#     ax.set_xlabel(r"$r'$")
#     ax.set_ylabel(r"$\phi'$")
#     ax.set_zlabel("PDF")
#     plt.legend()
#     plt.show()