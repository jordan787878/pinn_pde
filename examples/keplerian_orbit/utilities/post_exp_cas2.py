import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import norm, multivariate_normal


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


def p_init_better(x, mean, cov):
    """
    x is numpy array of shape (N x 4), N is the sample size
    """
    pdf_func = multivariate_normal(mean=mean, cov=cov)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


def load_trained_model(net, path, method="old"):
    print("[load model from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if(method == "new"):
        loss_history = np.array(checkpoint['loss_history'])
        print("best epoch: ", epoch, ", min loss:", np.min(loss_history), ", train time:", checkpoint['train_time'])
    else:
        print("best epoch: ", epoch, ", min loss:", checkpoint['loss'], ", train time:", checkpoint['train_time'])
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("pnet loss")
    # plt.tight_layout()
    # plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    # plt.close()
    return net


# Paper
def plot_train_loss(path):
    set_publication_plot_style()
    print("[load pnet model from: "+ path)
    checkpoint = torch.load(path)
    loss_history = np.array(checkpoint['loss_history'])
    # Create a figure and plot the loss history.
    fig = plt.figure(figsize=(8, 6))
    plt.plot(loss_history, color='black')

    # Define epochs at which to scatter points and their corresponding labels
    # scatter_epochs = np.array([9, 4287, 17264, 49579]) # phat
    # scatter_epochs = np.array([92, 261, 2181, 49219]) # e1hat_seq1
    scatter_epochs = np.array([51, 299, 4540, 47055]) # e1hat_seq2
    scatter_labels = ['a', 'b', 'c', 'd']

    # For this example, we'll assume loss_history has enough entries;
    # in practice, ensure that your loss_history length exceeds the maximum epoch in scatter_epochs.
    # Extract the loss values at these epochs
    scatter_losses = loss_history[scatter_epochs]

    # Scatter the points using red markers
    plt.scatter(scatter_epochs, scatter_losses, color='blue', s=50, zorder=5)

    # Annotate each scatter point with its label (offset the text to avoid overlap)
    for epoch, loss_val, label in zip(scatter_epochs, scatter_losses, scatter_labels):
        plt.annotate(label, (epoch, loss_val), textcoords="offset points", xytext=(-12,-12),
                    fontsize=18, color='blue')

    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log10 scale)")
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    fig.savefig("figs/v2e1hat_seq2_loss.pdf", format='pdf')
    # Display the plot.
    plt.show()


# Paper
def check_pdf_Nrphi(p_net, constants, DATA_FOLDER):
    """
    marginalize the pdf of normalize spherical to [r,phi]
    """
    set_publication_plot_style()
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(NN) marginalized to rphi at t=", t_prime)
        print(DATA_FOLDER)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        # pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
        # print("[check] x1s, pdf(monte) data type: ", x1s.dtype, pdf_monte.dtype)
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        samples = np.load(DATA_FOLDER+"samples_t{:.3f}.npy".format(t_prime))
        r_samples = samples[:,0]
        phi_samples = samples[:,2]

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # pdf_monte_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4
        pdf_nn_Nrphi = np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important

        # Plotting the contour plot
        fig = plt.figure(figsize=(8, 6))
        cp = plt.contourf(x1_grid, x2_grid, pdf_nn_Nrphi, levels=30, cmap="viridis", alpha=0.8)
        # Adding color bar
        plt.colorbar(cp)
        # scatter samples of (r, phi) on to the plot
        plt.scatter(r_samples, phi_samples, s=30, c='white', linewidths=0.5, edgecolor='black', alpha=1.0, label='Samples')
        # Adding labels and title
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$\phi'$")
        # plt.title(r"$p(r',\phi')$"+ "from NN and 200 Samples at t="+str(np.round(t_prime,2))+"T")
        plt.legend()
        plt.tight_layout(pad=0.2)
        fig.savefig("figs/v2phat_idx3_nrphi"+str(np.round(t_prime,3))+".pdf", format='pdf')
        plt.show()


def check_pdfnn_cartesian_wrt_monte(p_net, constants, DATA_FOLDER):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        # load pdf_monte on the domain
        pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))

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
        pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        pdf_mo_rphi_data = np.empty((0,4))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        z_mo = pdf_mo_rphi_data[:, 3]/(r) 

        _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 70
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == 0.0):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--", label="NN")
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5, label="Monte")
        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--")
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5)
        # Optional: Add a color bar
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # Labels and title
    ax.legend()
    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.set_zlabel('PDF Value')
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
    # Show the plot
    plt.show()


def compute_total_variation(t, data_1, data_2, p_net):
    x1s = np.load(data_1+"x1s.npy")
    x2s = np.load(data_1+"x2s.npy")
    x3s = np.load(data_1+"x3s.npy")
    x4s = np.load(data_1+"x4s.npy")
    dx1 = x1s[1] - x1s[0] # dr'
    dx2 = x2s[1] - x2s[0] # dphi'
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dv = dx1 * dx2 * dx3 * dx4
    pdf_1 = np.load(data_1+"pdf_t{:.3f}.npy".format(t)).ravel()
    pdf_2 = np.load(data_2+"pdf_t{:.3f}.npy".format(t)).ravel()
    tv = np.sum(np.abs(pdf_1 - pdf_2)) * dv

    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    pdf_3 = p_net(grid_points_tensor, t_tensor).detach().numpy().ravel()
    tv_nn = np.sum(np.abs(pdf_1 - pdf_3)) * dv

    idx_plot = np.arange(1, 1+len(pdf_1))
    plt.figure
    plt.plot(idx_plot, np.abs(pdf_1- pdf_2), "r:", linewidth=0.5)
    plt.plot(idx_plot, np.abs(pdf_1- pdf_3), "b--", linewidth=0.5)
    plt.show()
    return tv, tv_nn


def check_train_results(e1_net, p_net, t, data_folder, constants):
    """
    """
    # load p(monte)
    x1s = np.load(data_folder+"x1s.npy")
    x2s = np.load(data_folder+"x2s.npy")
    x3s = np.load(data_folder+"x3s.npy")
    x4s = np.load(data_folder+"x4s.npy")
    pdf_true = np.load(data_folder+"pdf_t{:.3f}.npy".format(t))
    # print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    if(t == 0.0):
        pdf_true = p_init_better(grid_points, constants.N_MEAN_I, constants.N_COV_I).reshape(x1_grid.shape) # obtain analytical p(true)
        # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

    # Grid spacings (assumed uniform)
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dV = dx1 * dx2 * dx3 * dx4
    E_x1 = np.sum(x1_grid * pdf_true) * dV
    E_x2 = np.sum(x2_grid * pdf_true) * dV
    E_x3 = np.sum(x3_grid * pdf_true) * dV
    E_x4 = np.sum(x4_grid * pdf_true) * dV

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    # if(t == 0): # [test]
    #     pdf_nn = p_init_perturb(grid_points).reshape(x1_grid.shape)
    # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    e1_nn  = e1_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    e1 = pdf_true - pdf_nn
    
    e1_vec = e1.reshape(-1)
    e1_nn_vec = e1_nn.reshape(-1)
    a1 = np.max(np.abs(e1_vec - e1_nn_vec)) / np.max(np.abs(e1_nn_vec))
    print("a1 (t=", np.round(t,2),"): ", np.round(a1,3))
    # print(E_x1, E_x2, E_x3, E_x4)
    max_e1 = np.max(np.abs(e1_vec))
    max_e1_nn = np.max(np.abs(e1_nn_vec))
    print(max_e1, max_e1_nn)
    B1 = 2.0 * max_e1_nn 

    idx_plot = np.arange(1, 1+len(e1_vec))
    plt.figure
    plt.plot(idx_plot, e1_vec, "black", linewidth=2.0)
    plt.plot(idx_plot, e1_nn_vec, "b--", linewidth=0.2)
    plt.fill_between(idx_plot, y1=0.0*idx_plot+B1, y2=0.0*idx_plot-B1, 
                         color="green", alpha=0.2, label=r"$B$")
    plt.show()

    return max_e1


def get_max_e1_init(p_net, data_folder, constants):
    """
    """
    # load p(monte)
    t = constants.TI
    x1s = np.load(data_folder+"x1s.npy")
    x2s = np.load(data_folder+"x2s.npy")
    x3s = np.load(data_folder+"x3s.npy")
    x4s = np.load(data_folder+"x4s.npy")
    pdf_true = np.load(data_folder+"pdf_t{:.3f}.npy".format(t))
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    pdf_true = p_init_better(grid_points, constants.N_MEAN_I, constants.N_COV_I).reshape(x1_grid.shape) # obtain analytical p(true)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    e1 = pdf_true - pdf_nn
    e1_vec = e1.reshape(-1)
    max_e1 = np.max(np.abs(e1_vec))

    return max_e1


# def check_pdfnn_marginalize(p_net, t=0.0):
#     """
#     marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
#     the true analytical pdf is obtained by 2 ways (but equivalent)
#     1. using the univariate normal
#     2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
#     """
#     global constants
#     print("[result] pdf_nn vs monte (or analytical at t=TI)")

#     # load p(monte)
#     x1s = np.load(DATA_FOLDER+"x1s.npy")
#     x2s = np.load(DATA_FOLDER+"x2s.npy")
#     x3s = np.load(DATA_FOLDER+"x3s.npy")
#     x4s = np.load(DATA_FOLDER+"x4s.npy")
#     pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t))
#     print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)

#     # prepare grid points 
#     x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
#     grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
#     print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    
#     # if t == 0.0, obtain analytical p(true)
#     if(t == 0.0):
#         pdf_true = p_init(grid_points).reshape(x1_grid.shape)
#         print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

#     # obtain pdf(nn)
#     grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
#     t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t).to(device)
#     print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
#     pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
#     print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    
#     dx1 = x1s[1] - x1s[0]
#     dx2 = x2s[1] - x2s[0]
#     dx3 = x3s[1] - x3s[0]
#     dx4 = x4s[1] - x4s[0]

#     # create figure
#     fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#     for j in range(4):
#         if(j == 0):
#             ax = axs[0,0]
#             x_axis = x1s
#             marginalize_pdf_monte = np.sum(pdf_monte, axis=(1,2,3)) * dx2 * dx3 * dx4
#             marginalize_pdf_nn = np.sum(pdf_nn, axis=(1,2,3)) * dx2 * dx3 * dx4
#             ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
#             ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
#             ax.set_ylim(0.0, constants.MAX_PX1)
#             ax.set_ylabel(r"$r$")
#             if(t == 0.0):
#                 # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
#                 marginalize_pdf_true = np.sum(pdf_true, axis=(1,2,3)) * dx2 * dx3 * dx4
#                 ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
#         elif(j == 1):
#             ax = axs[0,1]
#             x_axis = x2s
#             marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,2,3)) * dx1 * dx3 * dx4
#             marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,2,3)) * dx1 * dx3 * dx4
#             ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
#             ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
#             ax.set_ylim(0.0, constants.MAX_PX2)
#             ax.set_ylabel(r"$\phi$")
#             if(t == 0.0):
#                     # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
#                     marginalize_pdf_true = np.sum(pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
#                     ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
#         elif(j == 2):
#             ax = axs[1,0]
#             x_axis = x3s
#             marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,3)) * dx1 * dx2 * dx4
#             marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,3)) * dx1 * dx2 * dx4
#             ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
#             ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
#             ax.set_ylim(0.0, constants.MAX_PX3)
#             ax.set_ylabel(r"$v_r$")
#             if(t == 0.0):
#                     # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
#                     marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
#                     ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
#         else:
#             ax = axs[1,1]
#             x_axis = x4s
#             marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,2)) * dx1 * dx2 * dx3
#             marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,2)) * dx1 * dx2 * dx3
#             ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
#             ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
#             ax.set_ylim(0.0, constants.MAX_PX4)
#             ax.set_ylabel(r"$v_\phi$")
#             if(t == 0.0):
#                     # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
#                     marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
#                     ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
#         ax.legend()
#     plt.show()


# def test_nn_cartesian_pdf_xy(p_net):
    # """
    # convert the normalize spherical pdf_nn to pdf_nn(x,y)
    # the contour plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    # """
    # global constants
    # # Create the contour plot
    # plt.figure(figsize=(8, 6))
    # for t_prime in constants.T_PRIME_SPAN:
    #     print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
    #     x1s = np.load(DATA_FOLDER+"x1s.npy")
    #     x2s = np.load(DATA_FOLDER+"x2s.npy")
    #     x3s = np.load(DATA_FOLDER+"x3s.npy")
    #     x4s = np.load(DATA_FOLDER+"x4s.npy")

    #     # obtain pdf_nn on the domain
    #     x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    #     grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    #     grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    #     t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime).to(device)
    #     pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

    #     dx3 = x3s[1] - x3s[0]
    #     dx4 = x4s[1] - x4s[0]

    #     # marginalize to spherical position (r, phi)
    #     pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4

    #     # convert pdf
    #     pdf_nn_rphi_data = np.empty((0,4))
    #     x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
    #     for i in range(len(x1_grid)):
    #         for j in range(len(x2_grid)):
    #             # extract [r', phi', p(r',phi')]
    #             Nr = x1_grid[i,j]
    #             Nphi = x2_grid[i,j]
    #             r = Nr*constants.R
    #             phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
    #             t = constants.T*t_prime
    #             pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI)])))

    #     # convert pdf(r,phi) to pdf(x,y)
    #     x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1])  
    #     y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
    #     r = pdf_nn_rphi_data[:, 0]
    #     z_nn = pdf_nn_rphi_data[:, 3]/(r**2)  
    #     # Define the grid where you want to plot the contours
    #     grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),  # Adjust 100 to get finer resolution
    #                                  np.linspace(y.min(), y.max(), 100))
    #     # Interpolate scattered data onto the grid
    #     grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
    #     cp = plt.contourf(grid_x, grid_y, grid_z_nn, levels=15, cmap="viridis", alpha=1.0)  # Filled contour plot
    #     # plt.colorbar(cp)  # Add a colorbar to indicate the values
    # # Labels and title
    # plt.axis('equal')
    # plt.xlabel('X, m')
    # plt.ylabel('Y, m')
    # plt.title('pdf_nn(x,y) over T='+str(constants.TF)+' sec.')
    # plt.show()