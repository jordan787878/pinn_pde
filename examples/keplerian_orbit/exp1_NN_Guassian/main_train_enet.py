import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal
import argparse
from monte import pi, n_d, a_0, lam_0, mu_0, cov_0, x1_low, x1_hig, x2_low, x2_hig, ti, tf, t1s, mu_gravity, t_orbit
from main import Net, p_init, p_init_torch, p_multi_normal, res_func, init_weights_He, pos_p_net_train, get_pnet_output
from main import DATA_FOLDER, compare_pdf


TRAIN_FLAG = False
device = "cpu"
torch.set_default_dtype(torch.float32)


class ENet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(ENet, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x1 = (x[:,0].view(-1, 1)-0.5*(x1_low+x1_hig))/(0.5*(x1_hig-x1_low))
        _x2 = (x[:,1].view(-1, 1)-0.5*(x2_low+x2_hig))/(0.5*(x2_hig-x2_low))
        _t  = t/t_orbit
        inputs = torch.cat([_x1, _x2, _t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = output * self.scale
        return output
    

def diff_enet(x, t, e_net, verbose=False):
    output = e_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x2 = output_x[:,1].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    mean_motion = torch.sqrt(mu_gravity/x1**3)
    L_output = output_x2*mean_motion
    residual = output_t + L_output
    # if(verbose):
    #   print(x1[0,:].item(), t[0,:].item(), p[0,:].item(), p_x2[0,:].item(), mean_motion[0,:].item(), Lp[0,:].item(), p_t[0,:].item())
    # #   print("residual: ", residual, residual.shape)
    return residual


def train_e_net(x1_grid, x2_grid, e_net, p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global ti, tf
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e_net.scale
    print("[debug] normalize e_net: ", normalize)

    # Define the mean and covariance matrix
    _mean = torch.tensor([a_0, lam_0], dtype=torch.float32)
    _covariance_matrix = torch.tensor([[100.0**2, 0.0], [0.0, (pi/10)**2]], dtype=torch.float32)
    _mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    _x1_bc = (torch.rand(2000, 1) * (x1_hig - x1_low) + x1_low).to(device)
    _x2_bc = (torch.rand(2000, 1) * (x2_hig - x2_low) + x2_low).to(device)
    _x_bc = torch.cat((_x1_bc, _x2_bc), dim=1)
    _x_bc_normal = _mvn.sample((2000,))
    x1s = torch.from_numpy(x1_grid.reshape(-1,1)).float()
    x2s = torch.from_numpy(x2_grid.reshape(-1,1)).float()
    _x_bc_grid = torch.cat((x1s, x2s), dim=1)
    x_bc = torch.cat((_x_bc_grid, _x_bc, _x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)

    # _x1 = (torch.rand(1000, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
    # _x2 = (torch.rand(1000, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
    # _x = torch.cat((_x1, _x2), dim=1)
    # _x_normal = _mvn.sample((3000,))
    # x = torch.cat((_x, _x_normal), dim=0)
    # t = (torch.rand(len(x), 1, requires_grad=True)  * (tf - ti) + ti).to(device)

    # RAR
    S = 30000
    FLAG = False
    
    PATH = "output/e_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        u_bc = p_init_torch(x_bc).detach()
        u_net_bc = p_net(x_bc, t_bc).to(device)
        e_bc = (u_bc - u_net_bc).detach()
        e_net_bc = e_net(x_bc, t_bc)
        mse_ic = mse_cost_function(e_net_bc/normalize, e_bc/normalize)

        # [warm start]
        # max_ei_train = max(abs(e_bc))
        # while(max_ei_train < 0.5*normalize):
        #     print("[debug] add more samples...")
        #     _x1_RAR = (torch.rand(S, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
        #     _x2_RAR = (torch.rand(S, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
        #     x_RAR = torch.cat((_x1_RAR, _x2_RAR), dim=1)
        #     t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
        #     t0_RAR = 0.0*t_RAR + ti
        #     p_bc_RAR = p_init_torch(x_RAR)
        #     phat_bc_RAR = p_net(x_RAR, t0_RAR)
        #     e_bc_RAR = p_bc_RAR - phat_bc_RAR
        #     max_e_bc_RAR, max_index = torch.topk(torch.abs(e_bc_RAR.squeeze()), 10)
        #     if(max_e_bc_RAR[0] >= 0.5*normalize):
        #         x_max = x_RAR[max_index,:].clone().detach()
        #         t_max = t0_RAR[max_index].clone().detach()
        #         x_bc = torch.cat((x_bc, x_max), dim=0)
        #         t_bc = torch.cat((t_bc, t_max), dim=0)
        #         print("[debug] add at e_bc: ", max_e_bc_RAR.data)
        #         u_bc = p_init_torch(x_bc).detach()
        #         u_net_bc = p_net(x_bc, t_bc).to(device)
        #         e_bc = (u_bc - u_net_bc).detach()
        #         max_ei_train =  max(abs(e_bc))

        #[debug] visualizing the initial distribution samples
        if(epoch%500 == 0):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            print("max e_bc (true): ", np.max(np.abs(e_bc.numpy())))
            ax.scatter(x_bc[:,0].view(-1,1).numpy(), x_bc[:,1].view(-1,1).numpy(), e_bc.numpy(), c='b', marker='o', s=1)
            ax.scatter(x_bc[:,0].view(-1,1).numpy(), x_bc[:,1].view(-1,1).numpy(), e_net_bc.detach().numpy(), c='r', marker='x', s=1)
            ax.set_xlabel('X Axis Label')
            ax.set_ylabel('Y Axis Label')
            ax.set_zlabel('Z Axis Label')
            ax.set_title('3D Scatter Plot')
            plt.show()

        # Loss based on PDE
        # diff_e = diff_enet(x, t, e_net)
        # target_diff_e = -res_func(x, t, p_net, verbose=False).detach()
        # mse_res = mse_cost_function(diff_e/normalize, target_diff_e/normalize)

        # Frequnecy Loss
        # res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
        # mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_ic #+ t_orbit*(mse_res)
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_ic.data, 
                #   ",res:", mse_res.data,
                #   ",res_freq:", mse_norm_res_input.data
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            _x1_RAR = (torch.rand(S, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
            _x2_RAR = (torch.rand(S, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
            x_RAR = torch.cat((_x1_RAR, _x2_RAR), dim=1)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            e_bc_RAR = p_bc_RAR - phat_bc_RAR
            ehat_bc_RAR = e_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(e_bc_RAR - ehat_bc_RAR))/normalize
            if(max_ic_error > 5e-3):
                _, max_index = torch.topk(torch.abs(e_bc_RAR.squeeze() - ehat_bc_RAR.squeeze()), 10)
                x_max = x_RAR[max_index,:].clone().detach()
                t_max = t0_RAR[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", x_max[0,:].data, t_max[0].item(), ", normalized error: {:.4f}".format(max_ic_error.item()))
            # res_RAR = diff_enet(x_RAR, t_RAR, e_net) - res_func(x_RAR, t_RAR, p_net)
            # res_RAR = res_RAR/normalize
            # max_res_RAR = torch.max(torch.abs(res_RAR))
            # if(max_res_RAR > 0.0):
            #     _ , max_index = torch.topk(torch.abs(res_RAR.squeeze()), 10)
            #     x_max = x_RAR[max_index,:].clone()
            #     t_max = t_RAR[max_index].clone()
            #     x = torch.cat((x, x_max), dim=0)
            #     t = torch.cat((t, t_max), dim=0)
            #     print("... RAR Res, add: ", x_max[0,:].data, t_max[0].item(),", normalized error: {:.4f}".format(max_res_RAR.item()))
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save("output/e_net_train_loss.npy", np.array(loss_history))


def pos_enet_train(e_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("enet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
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
    plt.ylabel("enet loss")
    plt.tight_layout()
    plt.savefig("figs/enet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    print("[load enet model from: "+PATH)
    return e_net


def get_enet_output(e_net, x1_grid, x2_grid, t1):
    x1s = torch.from_numpy(x1_grid.reshape(-1,1)).float()
    x2s = torch.from_numpy(x2_grid.reshape(-1,1)).float()
    Nx = len(x1_grid)
    xs = torch.cat((x1s, x2s), dim=1)
    ts = x1s*0.0 + t1
    ehat = e_net(xs, ts).detach().numpy().reshape(Nx, Nx)
    # p_net.verbose = True
    # _mu_torch = torch.from_numpy(mu_0).view(1,2)
    # phat_max = p_net(_mu_torch, (_mu_torch[0]*0.0 + t1).view(-1,1)).detach().numpy()
    # [debug] x1_grid is float64...
    # print(x1_grid.dtype, x1s.dtype)
    # diff_x1_points = x1_grid.reshape(-1,1) - x1s.detach().numpy()
    # plt.figure()
    # plt.plot(np.arange(0,len(diff_x1_points)), diff_x1_points)
    # plt.show()
    return ehat


def main():
    # # [load p_net]
    p_net = Net().to(device)
    max_pi = 0.0016 # set this before training
    p_net.scale = max_pi
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()
    print("[load pnet model from: "+ "output/p_net.pth]")

    # # [checking if p_net is successfully loaded]
    for t1 in t1s:
        # load data
        x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
        x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
        pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
        pdf_nn = get_pnet_output(p_net, x1_grid, x2_grid, t1)
        if(t1 == 0.0):
            max_ei = max(pdf.flatten() - pdf_nn.flatten())
            print("get max ei: ", max_ei)
    #     print("### pdf_nn ###")
    #     mean_NN, cov_NN = p_net.get_mean_cov(torch.tensor(t1).view(-1,1))
    #     print("mean: {:.3f}, {:.3f}".format(mean_NN[0], mean_NN[1]))
    #     print("cov: ")
    #     for row in cov_NN:
    #         print("  ".join("{:.3f}".format(_r) for _r in row))
    #     compare_pdf(x1_grid, x2_grid, t1, pdf, pdf_nn, pdf_nn)

    # # [training]
    e_net = ENet().to(device)
    e_net.apply(init_weights_He)
    e_net.scale = max_ei
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_e_net(x1_grid, x2_grid, e_net, p_net, optimizer, scheduler, mse_cost_function, iterations=100000); print("e_net train complete")
    e_net = pos_enet_train(e_net, PATH="output/e_net.pth", PATH_LOSS="output/e_net_train_loss.npy"); e_net.eval()

    # # [visualize e_net]
    # t1s = [0.0]
    for t1 in t1s:
        print("\nt1: ", t1)
        # load data
        x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
        x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
        pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
        pdf_nn = get_pnet_output(p_net, x1_grid, x2_grid, t1)
        e = (pdf - pdf_nn)
        e_nn =   get_enet_output(e_net, x1_grid ,x2_grid, t1)
    #     print("### pdf_nn ###")
    #     mean_NN, cov_NN = p_net.get_mean_cov(torch.tensor(t1).view(-1,1))
    #     print("mean: {:.3f}, {:.3f}".format(mean_NN[0], mean_NN[1]))
    #     print("cov: ")
    #     for row in cov_NN:
    #         print("  ".join("{:.3f}".format(_r) for _r in row))
        # compare_pdf(x1_grid, x2_grid, t1, pdf, pdf_nn, pdf_nn)
        compare_pdf(x1_grid, x2_grid, t1, e, e_nn, e_nn)

    # # [final printout]
    if(TRAIN_FLAG == False):
        print("[complete simple 2BP, with pretrained models]")
    else:
        print("[complete simple 2BP]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()