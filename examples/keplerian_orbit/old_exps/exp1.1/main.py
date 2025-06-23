import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal
from sklearn.manifold import SpectralEmbedding
import argparse


from monte import DATA_FOLDER
from monte import x1_low, x1_hig, x2_low, x2_hig, x3_low, x3_hig, x4_low, x4_hig, xy_scaled, v_scaled, mean_scaled, cov_scaled
from monte import ti, tf, max_p0
from monte import pdf_xy
print(x1_low, x1_hig)
print(x2_low, x2_hig)
print(x3_low, x3_hig)
print(x4_low, x4_hig)

device = "cpu"
TRAIN_FLAG = False
n_d = 4


def p_init(x):
    pdf_func = multivariate_normal(mean_scaled, cov_scaled)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1)
    return pdf_eval


# def p_init_torch(x):
#     # Ensure input x is a torch tensor
#     if not isinstance(x, torch.Tensor):
#         raise ValueError("Input x must be a torch tensor")
#     mu_0_torch = torch.tensor(mean_scaled, dtype=torch.float32)
#     cov_0_torch = torch.tensor(cov_scaled, dtype=torch.float32)
#     # Define the multivariate normal distribution
#     m = torch.distributions.MultivariateNormal(loc=mu_0_torch, covariance_matrix=cov_0_torch)
#     # Evaluate the PDF at each point in x
#     pdf_eval = m.log_prob(x).exp().reshape(-1, 1)  # Convert log-prob to prob
#     return pdf_eval
# def res_func(x, t, p_net, verbose=False):
#     p = p_net(x,t)
#     p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#     p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#     # p_x1 = p_x[:,0].view(-1,1)
#     p_x2 = p_x[:,1].view(-1,1)
#     x1 = x[:,0].view(-1, 1)
#     # x2 = x[:,1].view(-1, 1)
#     mean_motion = torch.sqrt(mu_gravity/x1**3)
#     Lp = p_x2*mean_motion
#     residual = p_t + Lp
#     if(verbose):
#       print(x1[0,:].item(), t[0,:].item(), p[0,:].item(), p_x2[0,:].item(), mean_motion[0,:].item(), Lp[0,:].item(), p_t[0,:].item())
#     #   print("residual: ", residual, residual.shape)
#     return residual


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # self.delta_layer1 = (nn.Linear(2, neurons))
        # self.delta_layer2 = (nn.Linear(neurons, 1))
    def forward(self, x, t):
        # linear transform based on the center of the domain (in essence, scaled to -1~1)
        _x1 = (x[:,0].view(-1,1)-0.5*(x1_low+x1_hig))/(0.5*(x1_hig-x1_low))
        _x2 = (x[:,1].view(-1,1)-0.5*(x2_low+x2_hig))/(0.5*(x2_hig-x2_low))
        _x3 = (x[:,2].view(-1,1)-0.5*(x3_low+x3_hig))/(0.5*(x3_hig-x3_low))
        _x4 = (x[:,3].view(-1,1)-0.5*(x4_low+x4_hig))/(0.5*(x4_hig-x4_low))
        _t =  t/tf
        inputs = torch.cat([_x1, _x2, _x3, _x4, _t], dim=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus(self.output_layer(layer5_out))
        # inputs2 = torch.cat([_r, _t], dim=1)
        # delta1_out = F.softplus(self.delta_layer1(inputs2))
        # delta2_out = self.delta_layer2(delta1_out)
        # delta_out =  torch.exp(10.0*delta2_out)/(1+torch.exp(10.0*delta2_out))
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, ti, tf
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale
    batch_size = 2000

    # Define the mean and covariance matrix
    _mean = torch.tensor(mean_scaled)
    _covariance_matrix = torch.tensor(cov_scaled)
    _mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    _x1_bc = (torch.rand(batch_size, 1) * (x1_hig - x1_low) + x1_low).to(device)
    _x2_bc = (torch.rand(batch_size, 1) * (x2_hig - x2_low) + x2_low).to(device)
    _x3_bc = (torch.rand(batch_size, 1) * (x3_hig - x3_low) + x3_low).to(device)
    _x4_bc = (torch.rand(batch_size, 1) * (x4_hig - x4_low) + x4_low).to(device)
    _x_bc = torch.cat((_x1_bc, _x2_bc, _x3_bc, _x4_bc), dim=1)
    _x_bc_normal = _mvn.sample((batch_size,))
    x_bc = torch.cat((_x_bc, _x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    print("[debug] ", torch.min(x_bc[:,0]), torch.max(x_bc[:,0]))
    print("[debug] ", torch.min(x_bc[:,1]), torch.max(x_bc[:,1]))
    print("[debug] ", torch.min(x_bc[:,2]), torch.max(x_bc[:,2]))
    print("[debug] ", torch.min(x_bc[:,3]), torch.max(x_bc[:,3]))

    # RAR
    S = 30000
    FLAG = False
    
    PATH = "output/p_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        # u_bc = p_init_torch(x_bc).detach()
        # print(max(u_bc))
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss Function
        loss = mse_u
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, 
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            x1_RAR = (torch.rand(S, 1) * (x1_hig - x1_low) + x1_low).to(device)
            x2_RAR = (torch.rand(S, 1) * (x2_hig - x2_low) + x2_low).to(device)
            x3_RAR = (torch.rand(S, 1) * (x3_hig - x3_low) + x3_low).to(device)
            x4_RAR = (torch.rand(S, 1) * (x4_hig - x4_low) + x4_low).to(device)
            _x_RAR = torch.cat((x1_RAR, x2_RAR, x3_RAR, x4_RAR), dim=1)
            _x_RAR_normal = _mvn.sample((S,))
            x_RAR = torch.cat((_x_RAR, _x_RAR_normal), dim=0)
            t0_RAR = (torch.ones(len(x_RAR), 1) * ti).to(device)
            p_bc_RAR = p_init(x_RAR.detach().numpy())
            p_bc_RAR = Variable(torch.from_numpy(p_bc_RAR).float(), requires_grad=False).to(device)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/normalize
            # print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > 0.0):
                max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].detach()
                t_max = t0_RAR[max_index].detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", x_max[0,:].data, t_max[0].item(), max_abs_ic[0].item(), max_ic_error.item())
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save("output/p_net_train_loss.npy", np.array(loss_history))


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return p_net


# def get_pnet_output(p_net, x1_grid, x2_grid, t1):
#     x1s = torch.from_numpy(x1_grid.reshape(-1,1)).float()
#     x2s = torch.from_numpy(x2_grid.reshape(-1,1)).float()
#     Nx = len(x1_grid)
#     xs = torch.cat((x1s, x2s), dim=1)
#     ts = x1s*0.0 + t1
#     phat = p_net(xs, ts).detach().numpy().reshape(Nx, Nx)
#     return phat
# def compare_pdf(x1_grid, x2_grid, pdf, pdf_nn):
#     # Determine the color limits
#     all_data = np.concatenate((pdf.ravel(), pdf_nn.ravel()))
#     vmin = all_data.min()
#     vmax = all_data.max()
#     fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
#     # First subplot for PDF
#     ax1 = axs[0]
#     ax1.plot_surface(x1_grid, x2_grid, pdf, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)
#     ax1.set_xlabel('X1')
#     ax1.set_ylabel('X2')
#     ax1.set_zlabel('PDF')
#     ax1.set_title('Monte Carlo')
#     ax1.set_xlim([x1_low, x1_hig])
#     ax1.set_ylim([x2_low, x2_hig])
#     ax1.set_zlim([vmin, vmax])
#     ax1.view_init(elev=0, azim=-90)
#     # Second subplot for pdf_nn
#     ax2 = axs[1]
#     ax2.plot_surface(x1_grid, x2_grid, pdf_nn, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)  # Change cmap as needed
#     ax2.set_xlabel('X1')
#     ax2.set_ylabel('X2')
#     ax2.set_zlabel('PDF')
#     ax2.set_title('Neural Network')
#     ax2.set_xlim([x1_low, x1_hig])
#     ax2.set_ylim([x2_low, x2_hig])
#     ax2.set_zlim([vmin, vmax])
#     ax2.view_init(elev=0, azim=-90)
#     plt.show()
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x1_grid, x2_grid, pdf-pdf_nn, cmap='viridis', edgecolor='none')
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('Error')
#     ax.set_title('Approximation Error')
#     # ax.view_init(elev=30, azim=-145)
#     plt.show()


def main():
    mse_cost_function = torch.nn.MSELoss()

    p_net = Net().to(device)
    p_net.apply(init_weights_He)
    p_net.scale = max_p0
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()
    print("[load pnet model from: "+ "output/p_net.pth]")

    t1s = [0.0]
    Nx = 50
    x1s = torch.from_numpy(np.linspace(x1_low, x1_hig, num=Nx, endpoint=True)).float()
    x2s = torch.from_numpy(np.linspace(x2_low, x2_hig, num=Nx, endpoint=True)).float()
    x3s = torch.from_numpy(np.linspace(x3_low, x3_hig, num=Nx, endpoint=True)).float()
    x4s = torch.from_numpy(np.linspace(x4_low, x4_hig, num=Nx, endpoint=True)).float()
    xs = torch.cartesian_prod(x1s, x2s, x3s, x4s)
    p0_xs = p_init(xs).reshape(Nx, Nx, Nx, Nx)

    x1s = np.linspace(x1_low, x1_hig, num=Nx, endpoint=True)
    x2s = np.linspace(x2_low, x2_hig, num=Nx, endpoint=True)
    x3s = np.linspace(x3_low, x3_hig, num=Nx, endpoint=True)
    x4s = np.linspace(x4_low, x4_hig, num=Nx, endpoint=True)
    X1, X2 = np.meshgrid(x1s, x2s)

    for t1 in t1s:
        phat_xs = p_net(xs, t1 + 0.0*xs[:,0].view(-1,1)).detach().numpy().reshape(Nx, Nx, Nx, Nx)
        # Marginalize to pdf(X,Y)
        # Plot and save pdf(X,Y): approx
        P_XY = np.zeros_like(X1)
        for i in range(phat_xs.shape[0]):
            for j in range(phat_xs.shape[1]):
                p_xy = pdf_xy(i, j, phat_xs, x3s, x4s)
                P_XY[i, j] = p_xy
        print("debug max P(X,Y) (approx): ", np.max(P_XY))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vmax = np.max(P_XY)
        ax.plot_surface(X1, X2, P_XY.T, cmap='viridis', vmin=0.0, vmax=vmax)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('P(X1, X2)')
        plt.title('3D Surface Plot of P(X1, X2)')
        fig.savefig('figs/phat_xy_t'+str(t1)+".pdf", format='pdf', dpi=300)
        plt.close()

        # Plot and save pdf(X,Y): true
        P_XY = np.zeros_like(X1)
        for i in range(p0_xs.shape[0]):
            for j in range(p0_xs.shape[1]):
                p_xy = pdf_xy(i, j, p0_xs, x3s, x4s)
                P_XY[i, j] = p_xy
        # print("debug max P(X,Y) (true): ", np.max(P_XY))
        _pdf_xy_func = multivariate_normal(mean_scaled[0:2], cov_scaled[0:2, 0:2])
        _pdf_xy_max = _pdf_xy_func.pdf(mean_scaled[0:2]).reshape(-1,1)
        print("debug max P(X,Y) (true): ", np.max(_pdf_xy_max))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vmax = np.max(P_XY)
        ax.plot_surface(X1, X2, P_XY.T, cmap='viridis', vmin=0.0, vmax=vmax)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('P(X1, X2)')
        plt.title('3D Surface Plot of P(X1, X2)')
        fig.savefig('figs/p_xy_t'+str(t1)+".pdf", format='pdf', dpi=300)
        plt.close()

        fig = plt.figure()
        plt.plot(p0_xs.reshape(-1,)[312400:312600], "black")
        plt.plot(phat_xs.reshape(-1,)[312400:312600], "r--")
        plt.show()


    if(TRAIN_FLAG == False):
        print("[complete 2d nonlinear, with pre-trained models]")
    else:
        print("[complete 2d nonlinear]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()