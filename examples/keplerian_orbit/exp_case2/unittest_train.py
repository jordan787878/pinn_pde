"""
The first trial of keplerian orbit using rotating spherical coordiante and dynamics.
It is similar to the Case 2 of the paper: Uncertainty propagation in orbital mechanics via tensor decompostion.
Difference: do not have process noise, and J2 perturbation.
This case is a circular orbit on plannar motion. Hence, we reduce the dynamics to 4D.
Compared to exp2_sphere_4d (baseline):
    1) the final time is increased to 0.2*T.
    2) the solution domain is increased to ensure sum(p) ~= 1.0
Current p_net: save epoch: 98453 ,loss: tensor(0.0001) ,ic: tensor(2.6113e-05) ,res: tensor(0.0005) ,res g: tensor(2.6750)

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from constants import Case2_4D_Constants
from monte import p_init, get_p_init_max


DATA_FOLDER = "data/"
PNET_PATH = "output/p_net_reg.pth"
device = "cpu"
TRAIN_FLAG = False
constants = Case2_4D_Constants()

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def dyn_f1(x):
    return x[:,2]

def dyn_f2(x):
    return x[:,3]

def dyn_f3(x):
    global constants
    return constants.T**2 *x[:,0] *(constants.W +constants.PHI *x[:,3]/constants.T)**2 -constants.T**2 *constants.MU_EARTH/(constants.R**3 * x[:,0]**2)

def dyn_f4(x):
    return -2*constants.T*x[:,2]*(constants.W + constants.PHI * x[:,3]/constants.T)/(x[:,0]*constants.PHI)

def diff_opt_p(x, t, p_net, verbose=False):
    output = p_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    output_x4 = output_x[:,3].view(-1,1)
    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f4 = dyn_f4(x).view(-1,1)
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4), create_graph=True)[0]
    f4_x4 = f4_x[:,3].view(-1,1)
    residual = output_t + output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + f4_x4*output
    # print(residual.dtype)
    return residual

def init_weights_He(m):
    if isinstance(m, nn.Linear):
        # init.kaiming_normal_(m.weight)
        init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

class PNet(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 8
        self.scale = scale
        super(PNet, self).__init__()
        self.hidden_layer1 = (nn.Linear(5,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
        _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
        _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
        _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
        _t  = t/(constants.TF/constants.T)
        inputs = torch.cat([_x1, _x2, _x3, _x4, _t],axis=1)
        layer1_out = F.tanh((self.hidden_layer1(inputs)))
        layer2_out = F.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = F.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = F.tanh((self.hidden_layer4(layer3_out)))
        layer5_out = F.tanh((self.hidden_layer5(layer4_out + layer2_out)))
        output = self.scale * F.softplus(self.output_layer(layer5_out + layer1_out))
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    N0_samples = 1000
    Nr_samples = 1000
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)
    
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(phat_i/normalize, p_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net)/normalize
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p, all_zeros)
        # Define a small scale for perturbation
        # epsilon = 1e-1
        # N_jj = 10
        # for j in range(N_jj):
        #     z = x + torch.randn_like(x) * epsilon
        #     res_p = res_p + diff_opt_p(z, t, p_net)/normalize
        # mse_res = mse_cost_function(res_p/(N_jj+1), all_zeros)

        # # 1) Compute the gradient of res_p with respect to x.
        res_grad = torch.autograd.grad(
            outputs=res_p,
            inputs=x,
            grad_outputs=torch.ones_like(res_p),  # ensure proper broadcasting
            create_graph=True
        )[0]  # res_grad will be a tensor of shape (N, 4)
        grad_norm = torch.sqrt((res_grad ** 2).sum(dim=1, keepdim=True))  # shape: (N, 1)
        tv_loss = grad_norm.mean()

        # # Loss Function
        w_reg = 1e-1
        loss = mse_u + mse_res + w_reg*tv_loss

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, ",res:", mse_res.data,
                  ",res g:", tv_loss.data
                   )
            min_loss = loss.data

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()



def load_trained_model(net, path, method="old"):
    print("[load pnet model from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if(method == "new"):
        loss_history = np.array(checkpoint['loss_history'])
        print("pnet best epoch: ", epoch, ", min loss:", np.min(loss_history), ", train time:", checkpoint['train_time'])
    else:
        print("pnet best epoch: ", epoch, ", min loss:", checkpoint['loss'], ", train time:", checkpoint['train_time'])
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


def main():
    global constants
    p_net = PNet().to(device)
    p_net.apply(init_weights_He)
    p_net.scale = get_p_init_max()

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=50000); print("p_net_reg train complete")


if __name__ == "__main__":
    main()
