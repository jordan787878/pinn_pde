import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns
import torch
from scipy.interpolate import griddata
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle


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


def check_pdf_Nrphi(constants, mc_folder=None, p_net=None):
    """
    marginalize the pdf of normalize spherical to [r,phi]
    """
    set_publication_plot_style()
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(NN) marginalized to rphi at t=", t_prime)
        x1s = np.load("data/grids/x1s.npy")
        x2s = np.load("data/grids/x2s.npy")
        x3s = np.load("data/grids/x3s.npy")
        x4s = np.load("data/grids/x4s.npy")
        # pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
        # print("[check] x1s, pdf(monte) data type: ", x1s.dtype, pdf_monte.dtype)
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        
        if(mc_folder is None):
            pdf = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
        else:
            pdf = np.load(mc_folder+"pdf_t{:.3f}.npy".format(t_prime))

        samples = np.load("data/samples/samples_t{:.3f}.npy".format(t_prime))
        r_samples = samples[:,0]
        phi_samples = samples[:,2]

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]
        # pdf_monte_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4
        pdf_nn_Nrphi = np.sum(pdf, axis=(2,3)) * dx3 * dx4
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important

        # Plotting the contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cp = plt.contourf(x1_grid, x2_grid, pdf_nn_Nrphi, levels=30, cmap="viridis", alpha=0.8)
        # Adding color bar
        plt.colorbar(cp)
        # scatter samples of (r, phi) on to the plot
        # plt.scatter(r_samples, phi_samples, s=30, c='white', linewidths=0.5, edgecolor='black', alpha=1.0, label='Samples')
        ax.text(
            0.01, 0.99,                   # near top-left
            f"t = {t_prime:.3f}\np estimated by 1e+5 samples",
            transform=ax.transAxes,       # use axes coords
            fontsize=32,                  # big text
            color='white',
            va='top', ha='left'           # align text box
        )
        # Adding labels and title
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$\phi'$")
        # plt.title(r"$p(r',\phi')$"+ "from NN and 200 Samples at t="+str(np.round(t_prime,2))+"T")
        # plt.legend(loc='upper right')
        plt.tight_layout(pad=0.2)
        fig.savefig("figs/pdf_monte_10x5_t{:.3f}.pdf".format(t_prime), format='pdf'); plt.close()
        # plt.show()


def check_pdf_cartesian_wrt_samples(constants, mc_folder=None, p_net=None):
    """
    convert the normalize spherical pdf to pdf(x,y)
    the contour plot is not exact, since we use interpolation to create x,y grid and p(x,y) on this grid
    """
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf marginalized to xy at t=", t_prime)

        x1s = np.load("data/grids/x1s.npy")
        x2s = np.load("data/grids/x2s.npy")
        x3s = np.load("data/grids/x3s.npy")
        x4s = np.load("data/grids/x4s.npy")
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)

        if(p_net is None):
            pdf = np.load(mc_folder+"pdf_t{:.3f}.npy".format(t_prime))
        else:
            pdf = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        print("[check] x1s, pdf data type: ", x1s.dtype, pdf.dtype)
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        pdf_monte_Nrphi =  np.sum(pdf, axis=(2,3)) * dx3 * dx4
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        pdf_rphi_data = np.empty((0,4))
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                pdf_Nrphi = pdf_monte_Nrphi[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                pdf_rphi = (pdf_Nrphi/(constants.R*constants.PHI))/r**2
                t = constants.T*t_prime
                pdf_rphi_data = np.vstack((pdf_rphi_data, np.array([r, phi, t, pdf_rphi])))

        # Coordinates (x and y) and values (z)
        x = pdf_rphi_data[:, 0] * np.cos(pdf_rphi_data[:, 1])  # x-coordinates (1st column)
        y = pdf_rphi_data[:, 0] * np.sin(pdf_rphi_data[:, 1])  # y-coordinates (2nd column)
        z = pdf_rphi_data[:, 3]  # values to plot (4th column)
        print(np.max(z))
        # Define the grid where you want to plot the contours
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),  # Adjust 100 to get finer resolution
                                     np.linspace(y.min(), y.max(), 100))
        # Interpolate scattered data onto the grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap="viridis", alpha=0.9)  # Filled contour plot
        # plt.colorbar(cp)  # Add a colorbar to indicate the values

        samples = np.load("data/samples/samples_t{:.3f}.npy".format(t_prime))
        r_samples = samples[1:30,0]*constants.R
        phi_samples = samples[1:30,2]*constants.PHI + constants.W*constants.T*t_prime
        x_samples = r_samples * np.cos(phi_samples)
        y_samples = r_samples * np.sin(phi_samples)
        if(t_prime == 0.0):
            plt.scatter(x_samples, y_samples, s=3, c='white', linewidths=0.2, edgecolor='red', alpha=0.9, label='Samples')
        else:
            plt.scatter(x_samples, y_samples, s=3, c='white', linewidths=0.2, edgecolor='red', alpha=0.9)
    # Labels and title
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title('p(x,y) v.s. Samples over 0.2T')
    plt.legend()
    plt.savefig("figs/figure1.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def check_error_flatten(constants, p_init_func, e1_net, p_net, t, data_folder):
    """
    """    
    set_publication_plot_style(font_size=16)
    # load p(monte)
    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")
    pdf_true = np.load(data_folder+"pdf_t{:.3f}.npy".format(t))
    # print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    
    if(t == 0.0):
        pdf_true = p_init_func(constants, grid_points).reshape(x1_grid.shape) # obtain analytical p(true)
        print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

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
    if(e1_net is not None):
        e1_nn  = e1_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    e1 = pdf_true - pdf_nn
    
    e1_vec = e1.reshape(-1)
    if(e1_net is not None):
        e1_nn_vec = e1_nn.reshape(-1)
        a1 = np.max(np.abs(e1_vec - e1_nn_vec)) / np.max(np.abs(e1_nn_vec))
        print("a1 (t=", np.round(t,2),"): ", np.round(a1,3))
    # print(E_x1, E_x2, E_x3, E_x4)
    max_e1 = np.max(np.abs(e1_vec))
    print("[test] max(p_mc - p_nn) at t={:.3f}: {:.4f}".format(t, max_e1))
    if(e1_net is not None):
        max_e1_nn = np.max(np.abs(e1_nn_vec))
        print("[test] max(p_mc - p_nn), max(e1_nn) at t={:.3f}: {:.4f} vs {:.4f}".format(t, max_e1, max_e1_nn))
        B1 = 2.0 * max_e1_nn 
        idx_plot = np.arange(1, 1+len(e1_vec))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(idx_plot, e1_vec, "black", linewidth=2.0, label=r"$e_1$")
        ax.plot(idx_plot, e1_nn_vec, "b--", linewidth=0.2, label=r"$\hat{e}_1$")
        ax.fill_between(idx_plot, y1=0.0*idx_plot+B1, y2=0.0*idx_plot-B1, 
                            color="green", alpha=0.2, label=r"$B_1$")
        ax.legend(loc='upper right')
        ax.text(
            0.01, 0.99,                   # near top-left
            f"t = {t:.3f}",
            transform=ax.transAxes,       # use axes coords
            fontsize=20,                  # big text
            color='black',
            va='top', ha='left'           # align text box
        )
        ax.set_ylim([-1.5*B1, 1.5*B1])
        ax.set_ylabel("Error")
        ax.set_xlabel("4D state idx")
        plt.tight_layout(pad=0.2)
        plt.show()


def check_pdfnn_cartesian_wrt_monte(constants, p_net, mc_folder):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    set_publication_plot_style()

    # Create a figure
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.xaxis.pane.set_facecolor('black')
    ax.yaxis.pane.set_facecolor('black')
    ax.zaxis.pane.set_facecolor('black')
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    for t_prime in constants.T_PRIME_SPAN:
        if(t_prime < constants.T_PRIME_SPAN[3] or t_prime > constants.T_PRIME_SPAN[3]):
            continue

        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load("data/grids/x1s.npy")
        x2s = np.load("data/grids/x2s.npy")
        x3s = np.load("data/grids/x3s.npy")
        x4s = np.load("data/grids/x4s.npy")
        # load pdf_monte on the domain
        pdf_monte = np.load(mc_folder+"pdf_t{:.3f}.npy".format(t_prime))

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

        surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, 
                                edgecolor='white',  
                                linewidth=0.5, linestyle="-", 
                                label=r"MC $p(x,y)$"+", t={:.3f}T".format(t_prime))
        surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, cmap=cm.viridis, rstride=3, cstride=3, edgecolor=None, 
                                label=r"PINN $\hat{p}(x,y)$" + ", t={:.3f}T".format(t_prime))

        # # 2) project the NN surface down onto the floor
        # ax.contourf(
        #     grid_x, grid_y, grid_z_nn,
        #     zdir='x',                
        #     offset=grid_x.min(),            
        #     alpha=1.0,           
        # )
        # X_off = np.full_like(grid_x, grid_x.min())   # same shape as grid_x
        # # ax.plot_wireframe(
        # #     X_off,          # constant‐x plane
        # #     grid_y, 
        # #     grid_z_mo,
        # #     rstride=3, 
        # #     cstride=3,
        # #     color='white',
        # #     linewidth=0.5,
        # #     linestyle='solid'
        # # )
        # ax.plot_wireframe(
        #     X_off,          # constant‐x plane
        #     grid_y, 
        #     grid_z_nn,
        #     rstride=3, 
        #     cstride=3,
        #     color='blue',
        #     linewidth=0.5,
        #     linestyle='solid'
        # )

        # ax.contourf(
        #     grid_x, grid_y, grid_z_nn,
        #     zdir='y',                # drop along the z‐axis
        #     offset=grid_y.max(),            # onto the z=z_min plane
        #     cmap=cm.viridis,  
        #     alpha=.7,           # or any colormap you like
        # )
        # Y_off = np.full_like(grid_y, grid_y.max())   # same shape as grid_x
        # ax.plot_wireframe(
        #     grid_x,          # constant‐x plane
        #     Y_off, 
        #     grid_z_mo,
        #     rstride=3, 
        #     cstride=3,
        #     color='white',
        #     linewidth=0.5,
        #     linestyle='solid'
        # )

        ax.view_init(25, -40)
        ax.legend()
        ax.set_xlabel('\n X, m', color='white')
        ax.set_ylabel('\n Y, m', color='white')
        ax.set_zlabel('\n PDF Value', color='white')
        plt.tight_layout(pad=0.1)
        # fig.savefig("figs/case2_pinn_vs_monte_t{:.3f}.pdf".format(t_prime), format='pdf'); plt.close()
        plt.show()


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


def visual_e1hat_training(constants, networks, data_foler, save_plot_path=None):
    """
    visualize e1 networks training results using final or intermediate saved model
    NOTE: select the t_span
    """
    p_net, e1_net_seq1, e1_net_seq2 = networks

    set_publication_plot_style()

    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")

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
        y2 = e1_nn_vec[::gap]
        axs[i].plot(x, y1, "black", linewidth=0.5, rasterized=True, label="MC")
        axs[i].plot(x, y2, "blue",  linewidth=0.5, rasterized=True, label="NN")
        axs[i].fill_between(x, y1=0.0*x+B1, y2=0.0*0-B1, 
                            color="green", edgecolor="none", alpha=0.1, label="Error bound")
        axs[i].set_ylabel("Error")
        axs[i].grid(True)
        axs[i].text(0.02, 0.98, f"t={t_prime:.2f}T", transform=axs[i].transAxes,
                    ha='left', va='top', color='black', fontsize=18,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        axs[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        # axs[i].set_ylim([-1.5*B1, 1.5*B1])
        # ymin, ymax = axs[i].get_ylim()
        # print(f"Subplot {i}: ymin = {ymin}, ymax = {ymax}")

    axs[1].set_xlabel("4D state idx")
    axs[0].legend(loc="lower left", ncol=3)
    # fig.subplots_adjust(left=0.1)
    plt.tight_layout(pad=0.2)
    if(save_plot_path is not None):
        print("Save plot to: ", save_plot_path)
        fig.savefig(save_plot_path, format='png', dpi=300)
    else:
        plt.show()


def plot_train_loss(path):
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
    # fig.savefig("figs/v2e1hat_seq2_loss.pdf", format='pdf')
    # Display the plot.
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


def plot_app1(target, save_plot_path=None):
    set_publication_plot_style()
    parent_folder = "data/app1/"+target+"/prob/"
    def set_plots():
        if(target == "tar1"):
            pr_mcs = np.load(parent_folder+"mc.npy")
            pr_nn_data_labels = [
                                parent_folder+"est_nres50.npy", 
                                parent_folder+"ni_nres50.npy", 
                                parent_folder+"lp_nres50.npy", 
                                parent_folder+"fo_gmmx16_nres50_100k.npy",
                                parent_folder+"fo_gmmx32_nres50_100k.npy",
                                parent_folder+"fo_gmmx48_nres50_100k.npy",
                                parent_folder+"fo_gmmx64_nres50_100k.npy",
                                parent_folder+"fo_gmmx80_nres50_100k.npy",
                                ]
            plot_labels = [
                        r"Only $\hat{p}$",
                        r"NI$_{50}$", 
                        r"LP$_{50}$", 
                        r"FO$_{16}$",
                        r"FO$_{32}$",
                        r"FO$_{48}$",
                        r"FO$_{64}$",
                        r"FO$_{80}$",
                        ]
            plot_fills  = [False, False, False, False, False, False, False, False]
            plot_style =  ["--", "-", "-", "-", "-", "-", "-", "-"]
            marker = ["", ">", "d", "", "", "", "", ""]

        elif(target == "tar2"):
            pr_mcs = np.load(parent_folder+"mc.npy")
            pr_nn_data_labels = [
                                parent_folder+"est_nres50.npy", 
                                parent_folder+"ni_nres50.npy", 
                                parent_folder+"lp_nres50.npy", 
                                parent_folder+"fo_gmmx16_nres50_100k.npy",
                                parent_folder+"fo_gmmx32_nres50_100k.npy",
                                parent_folder+"fo_gmmx48_nres50_100k.npy",
                                parent_folder+"fo_gmmx64_nres50_100k.npy",
                                parent_folder+"fo_gmmx80_nres50_100k.npy",
                                ]
            plot_labels = [
                        r"Only $\hat{p}$",
                        r"NI$_{50}$", 
                        r"LP$_{50}$", 
                        r"FO$_{16}$",
                        r"FO$_{32}$",
                        r"FO$_{48}$",
                        r"FO$_{64}$",
                        r"FO$_{80}$",
                        ]
            plot_fills  = [False, False, False, False, False, False, False, False]
            plot_style =  ["--", "-", "-", "-", "-", "-", "-", "-"]
            marker = ["", ">", "d", "", "", "", "", ""]
        else:
            raise("this target is not runned")
        return pr_mcs, pr_nn_data_labels, plot_labels, plot_fills, plot_style, marker

    pr_mcs, pr_nn_data_labels, plot_labels, plot_fills, plot_style, marker = set_plots()
    # --- load Pr_opt data ---
    pr_nn_data = []
    for j in range(len(pr_nn_data_labels)):
        pr_nn_data.append(np.load(pr_nn_data_labels[j]))

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", len(pr_nn_data_labels)+1)

    t_span_mc = pr_mcs[:,0]
    pr = pr_mcs[:,-1]
    mask = ~np.isnan(pr)
    plt.plot(t_span_mc[mask], pr[mask], color="black", linestyle="", marker="o", markersize=4, 
             label=r"$\mathbb{P}$ by MC")
    print("[check] Prob. by MC (time, Prob., Prob.)")
    print(np.round(pr_mcs,4))

    print("[check] Prob. by PINN and Error Bound (time, Prob.)")
    for j in range(len(pr_nn_data_labels)):
        pr_nn_data_i = pr_nn_data[j]
        t_span = pr_nn_data_i[:,0]
        print(np.round(pr_nn_data_i,4))
        plt.plot(t_span, pr_nn_data_i[:,1], color=colors[j], 
                 linestyle=plot_style[j], 
                 linewidth = 1.5,
                 marker=marker[j], label=plot_labels[j])
        # if(plot_fills[j]): 
        #     plt.fill_between(t_span, y1=0.0*t_span, y2=pr_nn_data_i[:,1],
        #                      color=colors[j], edgecolor="none", alpha=0.1)

    # plt.grid(True)
    plt.ylabel(r"$\mathbb{P}(X'_{tar})$")
    plt.xlabel("t")
    plt.legend(loc="upper right", ncol=3)
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.2])
    ticks = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    tick_labels = [f"{tick:.2f}T" for tick in ticks]
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    # create the inset axes in the lower-right corner
    ax = plt.gca()
    # 1) draw a red rectangle on the main plot showing the zoom window
    zoom_rect = Rectangle((0.03, 0.0),        # lower-left corner
                        0.07-0.03,          # width
                        0.04-0.0,           # height
                        linewidth=2.0,
                        edgecolor='yellow',
                        facecolor='none',
                        linestyle='--')
    ax.add_patch(zoom_rect)

    # 2) create the inset as before
    axins = inset_axes(ax,
                    width="80%", 
                    height="65%", 
                    loc="lower right",
                    bbox_to_anchor=(0.35, 0.1, 0.65, 0.90),
                    bbox_transform=ax.transAxes)
    # re-plot into the inset
    axins.plot(t_span_mc[mask], pr[mask], color="black", linestyle="", marker="o", markersize=4)
    for j in range(len(pr_nn_data_labels)):
        data = pr_nn_data[j]
        axins.plot(data[:,0], data[:,1],
                color=colors[j],
                linestyle=plot_style[j],
                linewidth=1.5,
                marker=marker[j])
        if plot_fills[j]:
            axins.fill_between(data[:,0], 0, data[:,1], color=colors[j], alpha=0.1)
    # set zoom limits
    axins.set_xlim(0.03, 0.07)
    axins.set_ylim(-0.001, 0.04)
    axins.grid(True)
    # 3) make the inset’s border bold and red
    for spine in axins.spines.values():
        spine.set_edgecolor('yellow')
        spine.set_linewidth(2.0)
    # draw connector lines (you can also switch these to red if you like)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout(pad=0.2)
    if(save_plot_path is not None):
        print("Save plot to: ", save_plot_path)
        fig.savefig(save_plot_path+".pdf", format='pdf')
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
        x1s = np.load("data/grids/x1s.npy")
        x2s = np.load("data/grids/x2s.npy")
        x3s = np.load("data/grids/x3s.npy")
        x4s = np.load("data/grids/x4s.npy")

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


def densify(V, N_degree=1):
    """
    Given a 1D array V of length N, returns an array of length 2N−1
    with V’s entries interleaved with the midpoints of each adjacent pair.
    """
    V = np.asarray(V, dtype=float)

    for i in range(N_degree):
        N = V.shape[0]
        # new length = original + (N−1) midpoints
        new_len = 2*N - 1
        
        new_V = np.empty(new_len, dtype=V.dtype)
        # place original values at even indices
        new_V[0::2] = V
        # compute midpoints and place at odd indices
        new_V[1::2] = (V[:-1] + V[1:]) / 2.0
        V = new_V

    return V


def marginal_gmm_in_2d(X, Y, p_gmm):
    w_np, m_np, cov_np = p_gmm.get_gmm_paramters()
    K, D = m_np.shape

    # 2) flatten grid, build full-D evaluation points
    M, N = X.shape
    fixed_vals = None 
    pts12 = np.stack([X.ravel(), Y.ravel()], axis=1)  # [M*N, 2]
    if fixed_vals is None:
        fixed_vals = np.zeros(D - 2, dtype=float)
    else:
        fixed_vals = np.asarray(fixed_vals, dtype=float)
        assert fixed_vals.shape == (D - 2,)
    # tiled [M*N, D-2]
    pts_rest = np.tile(fixed_vals[None, :], (M*N, 1))
    pts_full = np.concatenate([pts12, pts_rest], axis=1)  # [M*N, D]
    # 3) define a helper for 2-D diagonal‐Gaussian PDF
    def gaussian2d_pdf(xy, mu, var):
        # xy: [P,2], mu: [2,], var: [2,]
        diff = xy - mu[None, :]
        inv_var = 1.0 / var[None, :]
        exp_term = -0.5 * np.sum(diff * diff * inv_var, axis=1)
        norm = 1.0 / (2 * np.pi * np.sqrt(var[0] * var[1]))
        return norm * np.exp(exp_term)
    # 4) accumulate weighted marginals
    pdf_vals = np.zeros(M * N, dtype=float)
    for k in range(K):
        mu_k = m_np[k, :2]
        var_k = cov_np[k, :2]
        pdf_k = gaussian2d_pdf(pts12, mu_k, var_k)
        pdf_vals += w_np[k] * pdf_k
    Z = pdf_vals.reshape(M, N)
    return Z


def plot_pdf_gmm_wrt_pinn(constants, p_net, p_gmm, target_r, target_phi, t, save_plot_path=None):
    set_publication_plot_style()
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")

    # obtain pdf_nn on the domain
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    dx1 = x1s[1] - x1s[0] # dr'
    dx2 = x2s[1] - x2s[0] # dphi'
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    with torch.no_grad():
        pdf_pinn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

    # marginalize to spherical position (r, phi)
    X, Y =  np.meshgrid(x1s, x2s, indexing="ij")
    num_stride = 3
    pdf_pinn_marginal =  np.sum(pdf_pinn, axis=(2,3)) * dx3 * dx4
    surf1 = ax.plot_surface(
        X, Y, pdf_pinn_marginal,
        rstride=num_stride,
        cstride=num_stride,
        # cmap=cm.Blues,        # choose your colormap
        linewidth=0.5,            # no grid lines
        antialiased=True,
        edgecolor='blue', 
        facecolor='blue', 
        alpha=0.1,
        label=r"PINN $\hat{p}$"
    )
 
    if(p_gmm is not None):
        N_dense = 1
        X_dense, Y_dense = np.meshgrid(densify(x1s, N_degree=N_dense), 
                                       densify(x2s, N_degree=N_dense), indexing="ij")
        Z = marginal_gmm_in_2d(X_dense, Y_dense, p_gmm)
        # pdf_gmm = p_gmm(grid_points_tensor).detach().numpy().reshape(x1_grid.shape)
        # pdf_gmm_marginal =  np.sum(pdf_gmm, axis=(2,3)) * dx3 * dx4
        surf2 = ax.plot_surface(
            X_dense, Y_dense, Z,
            rstride=num_stride,
            cstride=num_stride,
            # cmap=cm.Greens,        # choose your colormap
            linewidth=0.5,            # no grid lines
            antialiased=True,
            edgecolor='green',  
            facecolor='green',
            alpha=0.1,
            label=r"FO $\phi$"
        )

    # get normalized target bound
    r_bounds = target_r/constants.R    # [r_min, r_max]
    phi_bounds = (target_phi-constants.W*constants.T*t)/constants.PHI  # [phi_min, phi_max]
    patch_vertices = [
        [r_bounds[0], phi_bounds[0], 0],
        [r_bounds[1], phi_bounds[0], 0],
        [r_bounds[1], phi_bounds[1], 0],
        [r_bounds[0], phi_bounds[1], 0],
    ]
    # Create a Poly3DCollection and add it to the 3D axis.
    patch = Poly3DCollection([patch_vertices], facecolor='red', 
                             alpha=1.0, edgecolor='k', 
                             linewidths=2,
                             zorder=10,
                             label=r"$X^'_{tar}$")
    ax.add_collection3d(patch)
    ax.legend()
    ax.set_xlabel(r"$r'$")
    ax.set_ylabel(r"$\phi'$")
    ax.set_zlabel('PDF Value')
    # ax.set_title('3D Surface Plot of PDF at t= {:.3f}'.format(t))
    ax.view_init(33, -122)
    plt.tight_layout(pad=0.1)
    if(save_plot_path is None):
        plt.show()
    else:
        fig.savefig(save_plot_path+".pdf", format='pdf'); plt.close()
   