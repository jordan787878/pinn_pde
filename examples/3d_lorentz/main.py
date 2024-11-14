import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from constants import MyConstantsLorentz

MONTE_FLAG = True
TRAIN_FLAG = False
constants = MyConstantsLorentz()
# Set a fixed seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)


def dyn_f1(x):
    global constants
    return constants.SIGMA*(x[:,1] - x[:,0])

def dyn_f2(x):
    global constants
    return x[:,0]*(constants.RHO - x[:,2]) - x[:,1]

def dyn_f3(x):
    global constants
    return x[:,0]*x[:,1] - constants.BETA*x[:,2]


def diff_opt(x, t, net, verbose=False):
    output = net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f1_x = torch.autograd.grad(f1, x, grad_outputs=torch.ones_like(f1), create_graph=True)[0]
    f1_x1 = f1_x[:,0].view(-1,1)
    f2_x = torch.autograd.grad(f2, x, grad_outputs=torch.ones_like(f2), create_graph=True)[0]
    f2_x2 = f2_x[:,1].view(-1,1)
    f3_x = torch.autograd.grad(f3, x, grad_outputs=torch.ones_like(f3), create_graph=True)[0]
    f3_x3 = f3_x[:,2].view(-1,1)
    residual = output_t + output_x1*f1 + f1_x1*output \
                        + output_x2*f2 + f2_x2*output \
                        + output_x3*f3 + f3_x3*output
    
    # (if noise is present) Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(output_x.size(1)):
        grad2 = torch.autograd.grad(output_x[:, i], x, grad_outputs=torch.ones_like(output_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    output_xx = torch.stack(hessian, dim=-1)
    output_x1x1 = output_xx[:, 0, 0].view(-1, 1)
    output_x2x2 = output_xx[:, 1, 1].view(-1, 1)
    output_x3x3 = output_xx[:, 2, 2].view(-1, 1)
    residual = residual - 0.5*(constants.L_TENSOR[0,0]*constants.L_TENSOR[0,0]*output_x1x1 + \
                               constants.L_TENSOR[1,1]*constants.L_TENSOR[1,1]*output_x2x2 + \
                               constants.L_TENSOR[2,2]*constants.L_TENSOR[2,2]*output_x3x3)

    if(verbose):
        print(residual.dtype, residual.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)


# p_net
class Net(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(constants.DIM+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X_RANGE[0,0]+constants.X_RANGE[0,1]))/(0.5*(constants.X_RANGE[0,1]-constants.X_RANGE[0,0]))
        _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X_RANGE[1,0]+constants.X_RANGE[1,1]))/(0.5*(constants.X_RANGE[1,1]-constants.X_RANGE[1,0]))
        _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X_RANGE[2,0]+constants.X_RANGE[2,1]))/(0.5*(constants.X_RANGE[2,1]-constants.X_RANGE[2,0]))
        inputs = torch.cat([_x1, _x2, _x3, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output
    

# e1 net
class E1Net(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 100
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(constants.DIM+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = self.scale * output
        return output
    

def get_ini_samples(num_samples=400):
    global constants
    _x_bc_normal = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=num_samples).astype(np.float32)
    _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
    _x_bc = np.column_stack([
        np.random.uniform(constants.X_RANGE[0,0], constants.X_RANGE[0,1], num_samples),
        np.random.uniform(constants.X_RANGE[1,0], constants.X_RANGE[1,1], num_samples),
        np.random.uniform(constants.X_RANGE[2,0], constants.X_RANGE[2,1], num_samples),
    ])
    _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
    x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * constants.TI)
    print("x_bc shape type: ", x_bc.shape, x_bc.dtype)
    print("t_bc shape type: ", t_bc.shape, t_bc.dtype)
    return x_bc, t_bc


def get_res_samples(num_samples=400):
    global constants
    _x_normal = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=num_samples).astype(np.float32)
    _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
    _x = np.column_stack([
        np.random.uniform(constants.X_RANGE[0,0], constants.X_RANGE[0,1], num_samples),
        np.random.uniform(constants.X_RANGE[1,0], constants.X_RANGE[1,1], num_samples),
        np.random.uniform(constants.X_RANGE[2,0], constants.X_RANGE[2,1], num_samples),
    ])
    _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
    x = torch.cat((_x_normal, _x), dim=0)
    t = np.random.uniform(constants.TI, constants.TF, len(x)),
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
    print("x shape type: ", x.shape, x.dtype)
    print("t shape type: ", t.shape, t.dtype)
    return x, t
    

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale
    N_samples = 2000

    # samples of initial condition
    x_bc, t_bc = get_ini_samples(num_samples=N_samples)
    # samples of residual
    x, t = get_res_samples(num_samples=N_samples)

    # RAR
    # S = 30000
    # FLAG = False
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = constants.p_init(x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i/normalize, p_i/normalize)

        # Loss based on PDE
        # res_p = diff_opt(x, t, p_net)/normalize
        # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False)
        # mse_res = mse_cost_function(res_p, all_zeros)

        # Loss Function
        loss = mse_u #+ constants.TF*mse_res
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:  {:.5f}".format(mse_u.item()), 
                #    ",res: {:.5f}".format(mse_res.item())
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time,
                    }, constants._PATH_PNET)
            min_loss = loss.data
            # FLAG = True

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(constants._PATH_PNET_LOSS, np.array(loss_history))
    

# def train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
#     global constants
#     min_loss = np.inf
#     iterations_per_decay = 1000
#     loss_history = []
#     normalize = e1_net.scale
#     N_samples = 1000

#     # samples of initial condition
#     x_bc, t_bc = get_ini_samples(num_samples=N_samples)
#     # samples of residual
#     x, t = get_res_samples(num_samples=N_samples)

#     # RAR
#     # S = 30000
#     # FLAG = False
    
#     start_time = time.time()
#     for epoch in range(iterations):
#         optimizer.zero_grad()

#         # Loss based on boundary conditions
#         p_i = constants.p_init(x_bc.detach().numpy())
#         p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
#         phat_i = p_net(x_bc, t_bc)
#         e_i = p_i - phat_i
#         ehat_i = e1_net(x_bc, t_bc)
#         mse_u = mse_cost_function(ehat_i/normalize, e_i/normalize)

#         # Loss based on PDE
#         res_e1 = diff_opt(x, t, e1_net)
#         res_p  = diff_opt(x, t, p_net).detach()
#         mse_res = mse_cost_function(res_e1/normalize, -res_p/normalize)

#         # Loss Function
#         loss = mse_u + constants.TF*mse_res
#         loss_history.append(loss.data)

#         # Save the min loss model
#         if(loss.data < 0.95*min_loss):
#             train_time = time.time() - start_time
#             print("save epoch:", epoch, ",loss:", loss.data, 
#                   ",ic:  {:.5f}".format(mse_u.item()), 
#                   ",res: {:.5f}".format(mse_res.item())
#                    )
#             torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': e1_net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': loss.data,
#                     'train_time': train_time,
#                     }, constants._PATH_E1NET)
#             min_loss = loss.data
#             # FLAG = True

#         loss.backward(retain_graph=True) 
#         optimizer.step()
#         # Exponential learning rate decay
#         if (epoch + 1) % iterations_per_decay == 0:
#             scheduler.step()
#     np.save(constants._PATH_E1NET_LOSS, np.array(loss_history))


def pos_p_net_train(p_net):
    print("[load pnet model from: "+ constants._PATH_PNET)
    checkpoint = torch.load(constants._PATH_PNET)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    loss_history = np.load(constants._PATH_PNET_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig(constants._FOLDER+"figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return p_net


def pos_e1_net_train(e1_net):
    print("[load pnet model from: "+ constants._PATH_E1NET)
    checkpoint = torch.load(constants._PATH_E1NET)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("e1net best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    loss_history = np.load(constants._PATH_E1NET_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig(constants._FOLDER+"figs/e1net_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return e1_net


def check_pnn_result(p_net):
    global constants
    grid_points_struct = constants.load_gridpoints_from_monte()
    grid_points = grid_points_struct[-1]
    dx1 = grid_points_struct[0][1] - grid_points_struct[0][0]
    dx2 = grid_points_struct[2][1] - grid_points_struct[2][0]
    dx3 = grid_points_struct[4][1] - grid_points_struct[4][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]
    x3_grid = grid_points_struct[5]
    print("[check] x1_grid x2_grid x3_grid shape: ", x1_grid.shape, x1_grid.dtype)
    for t in constants.T_SPAN:
        # load true pdf(t)
        pdf_true = constants.load_p_sol_monte(t)
        print("[check] pdf_true dtype: ", pdf_true.dtype)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))
        ax = axs[0]
        ax.plot(grid_points_struct[0], np.sum(pdf_true.reshape(x1_grid.shape), axis=(1,2))*dx2*dx3, "black")
        ax.plot(grid_points_struct[0], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(1,2))*dx2*dx3, "r--")
        ax.set_xlim(constants.X_RANGE[0,0], constants.X_RANGE[0,1])

        ax = axs[1]
        ax.plot(grid_points_struct[2], np.sum(pdf_true.reshape(x1_grid.shape), axis=(0,2))*dx1*dx3, "black")
        ax.plot(grid_points_struct[2], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(0,2))*dx1*dx3, "r--")
        ax.set_xlim(constants.X_RANGE[1,0], constants.X_RANGE[1,1])

        ax = axs[2]
        ax.plot(grid_points_struct[4], np.sum(pdf_true.reshape(x1_grid.shape), axis=(0,1))*dx1*dx2, "black")
        ax.plot(grid_points_struct[4], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(0,1))*dx1*dx2, "r--")
        ax.set_xlim(constants.X_RANGE[2,0], constants.X_RANGE[2,1])

        plt.show()

        # # NOTE
        # if(t > 0.0):
        #     return
        # # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # # compute p(true)
        # # pdf_true = constants.p_sol(grid_points, t)
        # pdf_true = constants.p_init(grid_points)
        # # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
        # # obtain pdf(nn)
        # grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        # t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        # pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
        # # print-out
        # error_init = pdf_true - pdf_nn
        # print(" =result= p_nn(t={:.1f}) normalized error: {:.4f}".format(t, np.max(np.abs(error_init))/p_net.scale))
        
        # # visualization (marginalized to 2 cooridnates)
        # fig, axs = plt.subplots(1, 3, figsize=(10, 6), subplot_kw={'projection': '3d'})
        # ax = axs[0]
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(2))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('x'); ax.set_ylabel('y')
        # ax.view_init(45, -135)

        # ax = axs[1]
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('y'); ax.set_ylabel('z')
        # ax.view_init(45, -135)

        # ax = axs[2]
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('x'); ax.set_ylabel('z')
        # ax.view_init(45, -135)

        # plt.savefig(constants._FOLDER+'figs/p_vs_pnn.pdf', format='pdf', dpi=300)
        # plt.close()



# def check_e1nn_result(e1_net, p_net):
#     global constants
#     grid_points_struct = constants.prepare_gridpoints()
#     grid_points = grid_points_struct[-1]
#     dx = grid_points_struct[0][1] - grid_points_struct[0][0]
#     x1_grid = grid_points_struct[1]
#     x2_grid = grid_points_struct[3]
#     x3_grid = grid_points_struct[5]
#     for t in constants.T_SPAN:
#         # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
#         # compute p(true)
#         pdf_true = constants.p_sol(grid_points, t)
#         # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
#         # obtain pdf(nn)
#         grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
#         t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
#         # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
#         pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
#         # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
#         e1 = pdf_true - pdf_nn
#         e1_nn = e1_net(grid_points_tensor, t_tensor).detach().numpy()
#         e2 = e1 - e1_nn
#         eS = 2.0*np.max(np.abs(e1_nn))
#         a1 = np.max(np.abs(e2))/np.max(np.abs(e1_nn))
#         print( " =result= e_nn(t={:.1f}) e1: {:.4f}, eS: {:.4f}, a1: {:.3f}".format(t, np.max(np.abs(e1)), eS, a1) )

#         # visualization (marginalized to 2 cooridnates)
#         # E1net plot
#         fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw={'projection': '3d'})
#         ax = axs[0]
#         ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
#                           color="black", linewidth=0.5, alpha=0.7, label=r"e")
#         ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
#                           color="red", linewidth=0.2, alpha=0.7, linestyle="--", label=r"$\hat{e}_1$")
#         ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Error')
#         ax.legend()
#         ax.view_init(20, -135)

#         ax = axs[1]
#         ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.7)
#         ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
#         ax.set_xlabel('y'); ax.set_ylabel('z'); ax.set_zlabel('Error')
#         ax.view_init(20, -135)
#         ax = axs[2]
#         ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.7)
#         ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
#         ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('Error')
#         ax.view_init(20, -135)

#         # Reduce white space between subplots and around the figure
#         plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
#         # Add a text to the top-left corner of the entire figure
#         fig.text(
#             0.05, 0.9, "t={:.1f}".format(t), 
#             fontsize=16, color='black',
#         )
#         plt.show()

#         # Error bound plot
#         fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw={'projection': '3d'})
#         ax = axs[0]
#         ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
#                           color="black", linewidth=0.5, alpha=0.7, label=r"e")
#         ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
#                           color="red", linewidth=0.2   , alpha=0.7, linestyle="--", label=r"$\hat{e}_1$")
#         ax.plot_surface(x1_grid[:, :, 0], x2_grid[:, :, 0], x1_grid[:, :, 0]*0.0 + 2.0*eS, color="green", alpha=0.2, label=r"$e_S$")
#         ax.plot_surface(x1_grid[:, :, 0], x2_grid[:, :, 0], x1_grid[:, :, 0]*0.0 - 2.0*eS, color="green", alpha=0.2)
#         ax.legend()
#         ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Error')
#         ax.view_init(15, -135)

#         ax = axs[1]
#         ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.7)
#         ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
#         ax.plot_surface(x2_grid[0, :, :], x3_grid[0, :, :], x2_grid[0, :, :]*0.0 + 2.0*eS, color="green", alpha=0.2)
#         ax.plot_surface(x2_grid[0, :, :], x3_grid[0, :, :], x2_grid[0, :, :]*0.0 - 2.0*eS, color="green", alpha=0.2)
#         ax.set_xlabel('y'); ax.set_ylabel('z'); ax.set_zlabel('Error')
#         ax.view_init(15, -135)

#         ax = axs[2]
#         ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.7)
#         ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
#         ax.plot_surface(x1_grid[:, 0, :], x3_grid[:, 0, :], x1_grid[:, 0, :]*0.0 + 2.0*eS, color="green", alpha=0.2)
#         ax.plot_surface(x1_grid[:, 0, :], x3_grid[:, 0, :], x1_grid[:, 0, :]*0.0 - 2.0*eS, color="green", alpha=0.2)
#         ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('Error')
#         ax.view_init(15, -135)

#         # Reduce white space between subplots and around the figure
#         plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
#         # Add a text to the top-left corner of the entire figure
#         fig.text(
#             0.05, 0.9, "t={:.1f}".format(t), 
#             fontsize=16, color='black',
#         )
#         plt.show()


# def get_e1init_max(p_net):
#     global constants
#     grid_points_struct = constants.prepare_gridpoints()
#     grid_points = grid_points_struct[-1]
#     t = constants.TI
#     # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
#     # compute p(true)
#     pdf_true = constants.p_sol(grid_points, t)
#     # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
#     # obtain pdf(nn)
#     grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
#     t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
#     # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
#     pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
#     # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
#     # print-out
#     error_init = pdf_true - pdf_nn
#     error_init_max = np.max(np.abs(error_init))
#     print("[check] e1(t0) max: {:.5f}".format(error_init_max))
#     return error_init_max


def main():
    global constants
    if(MONTE_FLAG):
        constants.construct_p_sol()

    p_net = Net(scale=constants.get_pinit_max())
    e1_net = E1Net()
    p_net.apply(init_weights)
    e1_net.apply(init_weights_xavier)

    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=10000); print("p_net train complete")
    p_net = pos_p_net_train(p_net); p_net.eval()
    check_pnn_result(p_net)

    # e1_net.scale = get_e1init_max(p_net)
    # mse_cost_function = torch.nn.MSELoss() # Mean squared error
    # optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # if(TRAIN_FLAG):
    #     train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=1800); print("e1_net train complete")
    # e1_net = pos_e1_net_train(e1_net); e1_net.eval()
    # check_e1nn_result(e1_net, p_net)



if __name__ == "__main__":
    main()