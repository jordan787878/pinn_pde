import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from tqdm import tqdm
import warnings
import time


FOLDER = "exp1/run-PE/"
DATA_FOLDER = "exp1/data/"

device = "cpu"
print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

n_d = 1
mu = -2.0
std = 0.5
const_a = -0.1
const_b = 0.1
const_c = 0.5
const_d = 0.5
const_e = 0.8
x_low = -6
x_hig = 6

t0 = 0
T_end = 5
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1/"]
batch_size = 1000
S = 10000


# def f_sde(x):
#     with warnings.catch_warnings():
#         warnings.filterwarnings('error')
#         try:
#             result = const_a*np.power(x, 3) + const_b*np.power(x, 2) + const_c*x + const_d
#         except RuntimeWarning as e:
#             print(f"Warning occurred at x = {x}: {e}")
#             result = np.nan  # or handle the error in another way
#     return result


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


# def p_init_torch(x):
#     return torch.exp(-0.5*((x-mu)/std)**2) / (std*torch.sqrt(2*torch.pi*torch.ones((len(x), 1))))


# def test_p_init():
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    p = p_init(x)
    return max(abs(p))[0]


# def p_sol_monte(t1=T_end, linespace_num=100, stat_sample=10000):
    n_d = 1
    dtt = 0.01
    t_span = np.arange(t0, t1, dtt)
    num_steps = len(t_span)  
     # Initialize arrays
    X_last = np.random.normal(mu, std, stat_sample)
    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)
    # Vectorized simulation of the SDE
    for step in tqdm(range(1, num_steps + 1), desc="Simulating samples"):
        dW = np.random.normal(0, np.sqrt(dtt), stat_sample)
        X_new = X_last + f_sde(X_last) * dtt + const_e * dW
        X_last = X_new
    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)
    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X_last, bins_x1) - 1
    # Initialize the frequency array
    frequency = np.zeros((len(bins_x1) - 1, 1))
    # Count the occurrences in each 2D bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        if 0 <= bin_indices_x1[i] < frequency.shape[0]:
            frequency[bin_indices_x1[i], :] += 1
    # Normalize the frequency to get the proportion
    frequency = frequency / stat_sample
    dx = bins_x1[1]-bins_x1[0]
    frequency = frequency/(dx**n_d)
    # Calculate the midpoints for bins
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    return midpoints_x1, frequency


def p_res_func(x, t, out, verbose=False):
    p = out[:,0].reshape(-1,1)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*const_a*x*x + 2*const_b*x + const_c)*p \
                   + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*p_x \
                   - 0.5*const_e*const_e*p_xx
    if(verbose):
        print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual


def e_res_func(x, t, out, verbose=False):
    e1_out = out[:,1].reshape(-1,1)
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    residual = e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
                   + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*e_x \
                   - 0.5*const_e*const_e*e_xx
    if(verbose):
        print(e_x.shape, e_t.shape, e_xx.shape, residual.shape)
    return residual


# def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# class PENet(nn.Module):
#     def __init__(self, scale=1.0): 
#         neurons = 50
#         self.scale = scale
#         super(PENet, self).__init__()
#         self.hidden_layer1 = (nn.Linear(64,neurons))
#         self.hidden_layer2 = (nn.Linear(neurons,neurons))
#         self.hidden_layer3 = (nn.Linear(neurons,neurons))
#         self.hidden_layer4 = (nn.Linear(neurons,neurons))
#         self.hidden_layer5 = (nn.Linear(neurons,2*neurons))
#         self.p_layer1 =  (nn.Linear(neurons,neurons))
#         self.p_layer2 =  (nn.Linear(neurons,neurons))
#         self.p_out =  (nn.Linear(neurons,1))
#         self.e_layer1 =  (nn.Linear(neurons,neurons))
#         self.e_layer2 =  (nn.Linear(neurons,neurons))
#         self.e_out =  (nn.Linear(neurons,1))
#         self.activation = nn.Tanh()
#         self.softplus = nn.Softplus()
#     def forward(self, x, t):
#         inputs = torch.cat([x,t], axis=1)
#         d = 32; n = 100
#         for i in range(int(d/2)):
#             w_i = (1/pow(n, 2*i/d))
#             x_sin_i = torch.sin(w_i*x)
#             x_cos_i = torch.cos(w_i*x)
#             inputs = torch.cat([inputs, x_sin_i, x_cos_i], axis=1)
#         for i in range(int(d/2)):
#             w_i = (1/pow(n, 2*i/d))
#             t_sin_i = torch.sin(w_i*t)
#             t_cos_i = torch.cos(w_i*t)
#             inputs = torch.cat([inputs, t_sin_i, t_cos_i], axis=1)
#         inputs = inputs[:, 2:]
#         layer1_out = self.activation(self.hidden_layer1(inputs))
#         layer2_out = self.activation(self.hidden_layer2(layer1_out))
#         layer3_out = self.activation(self.hidden_layer3(layer2_out))
#         layer4_out = self.activation(self.hidden_layer4(layer3_out))
#         layer5_out = self.activation(self.hidden_layer5(layer4_out))
#         p_layer1_out = self.activation(self.p_layer1(layer5_out[:,:50]))
#         p_layer2_out = self.activation(self.p_layer2(p_layer1_out))
#         p_out = F.softplus(self.p_out(p_layer2_out))
#         e_layer1_out = self.activation(self.e_layer1(layer5_out[:,50:]))
#         e_layer2_out = self.activation(self.e_layer2(e_layer1_out))
#         e_out = (self.e_out(e_layer2_out))
#         output = torch.cat([p_out, e_out], axis=1)
#         return output



class PENet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(PENet, self).__init__()
        self.hidden_layer1 = (nn.Linear(66,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer = (nn.Linear(neurons,2))
    def forward(self, x, t):
        inputs = torch.cat([x,t], axis=1)
        d = 32; n = 100
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            x_sin_i = torch.sin(w_i*x)
            x_cos_i = torch.cos(w_i*x)
            inputs = torch.cat([inputs, x_sin_i, x_cos_i], axis=1)
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            t_sin_i = torch.sin(w_i*t)
            t_cos_i = torch.cos(w_i*t)
            inputs = torch.cat([inputs, t_sin_i, t_cos_i], axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output =     (self.output_layer(layer5_out))
        return output  



def train_model(net, optimizer, scheduler, mse_cost_function, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/net.pt"
    PATH_LOSS = FOLDER+"output/net_train_loss.npy"
    iterations_per_decay = 1000

    margin_t = 0.05*5
    margin_x = 0.05*12
    x_bc = (torch.rand(batch_size, 1)*(x_hig-x_low+2*margin_x)+x_low-margin_x).to(device)
    x_bc = torch.clamp(x_bc, min=x_low, max=x_hig)
    t_bc = (torch.ones(batch_size, 1) * t0).to(device)

    t = (torch.rand(batch_size, 1, requires_grad=True)*(T_end-t0+2*margin_t)    +t0-margin_t).to(device)
    t = torch.clamp(t, min=t0, max=T_end)
    x = (torch.rand(len(t),     1, requires_grad=True)*(x_hig-x_low+2*margin_x) +x_low-margin_x).to(device)
    x = torch.clamp(x, min=x_low, max=x_hig)

    FLAG = 0

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # p init loss
        p0 = p_init(x_bc.detach().numpy())
        p0 = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        net_out_0 = net(x_bc, t_bc)
        p0_hat = net_out_0[:, 0].reshape(-1,1)
        mse_ic_p = mse_cost_function(p0_hat, p0)

        # p residual loss
        net_out = net(x, t)
        p_res = p_res_func(x, t, net_out)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res_p = mse_cost_function(p_res, all_zeros)
        res_x = torch.autograd.grad((p_res), x, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        res_t = torch.autograd.grad((p_res), t, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        mse_res_p_g = torch.mean(res_x**2 + res_t**2)

        # e init loss
        normalize = torch.max(torch.abs(p0-p0_hat), dim=0)[0].detach()

        e0_hat = net_out_0[:, 1].reshape(-1,1)
        e0_true = (p0 - p0_hat).detach().numpy()
        e0_true = Variable(torch.from_numpy(e0_true).float(), requires_grad=False).to(device)
        mse_ic_e = mse_cost_function(e0_hat/normalize, e0_true/normalize)

        e_res = (e_res_func(x, t, net_out) + p_res)/normalize
        mse_res_e = torch.mean((e_res)**2)
        _res_x = torch.autograd.grad((e_res), x, grad_outputs=torch.ones_like(e_res), create_graph=True)[0]
        _res_t = torch.autograd.grad((e_res), t, grad_outputs=torch.ones_like(e_res), create_graph=True)[0]
        mse_res_e_g = torch.mean(_res_x**2 + _res_t**2)

        w_regular = 1e-3
        loss = (mse_ic_p + mse_res_p + w_regular*mse_res_p_g) + 3.0*(mse_ic_e + mse_res_e + w_regular*mse_res_e_g)
        loss_history.append(loss.data)

        if ((epoch+1)%1000 == 0):
            print(epoch+1," Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("epoch:", epoch, ",loss:", loss.data, 
                  ",ic:",mse_ic_p.data, 
                  ",res:", mse_res_p.data,
                  ",res g:", mse_res_p_g.data,
                  "  ,ic e:",mse_ic_e.data, 
                  ",res e:", mse_res_e.data,
                  ",res g e:", mse_res_e_g.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
            FLAG = FLAG + 1

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # RAR
        if ((epoch+1)%500==0 and FLAG >= 3):
            # sample points
            print("Sample N (p,e): ", x.shape)
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            _t_init_RAR = (torch.ones(100, 1, requires_grad=True) * t0).to(device)
            t_RAR = torch.cat((t_RAR, _t_init_RAR), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            random_number = int(np.random.uniform(20, 50))
            _tt = np.linspace(t0, T_end, num=random_number)
            _xx = np.linspace(x_low, x_hig, num=random_number)
            _t, _x = np.meshgrid(_tt, _xx)
            _t = _t.reshape(-1,1)
            _x = _x.reshape(-1,1)
            _t_quad_RAR = Variable(torch.from_numpy(_t).float(), requires_grad=True).to(device)
            _x_quad_RAR = Variable(torch.from_numpy(_x).float(), requires_grad=True).to(device)
            t_RAR = torch.cat((t_RAR, _t_quad_RAR), dim=0)
            x_RAR = torch.cat((x_RAR, _x_quad_RAR), dim=0)
            
            net_out_RAR = net(x_RAR, t_RAR)
            p_res_RAR = p_res_func(x_RAR, t_RAR, net_out_RAR)
            e_res_RAR = e_res_func(x_RAR, t_RAR, net_out_RAR) + p_res_RAR

            # add to p residual samples
            max_abs_res, max_index = torch.topk(torch.abs(p_res_RAR), 3, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max[0].data, t_max[0].data, ". max res value: ", max_abs_res[0].data)
            
            # add to e residual samples
            max_abs_res, max_index = torch.topk(torch.abs(e_res_RAR), 3, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max.data[0], t_max[0].data, ". max res value: ", max_abs_res[0].data)

            FLAG = 0


def pos_p_net_train(net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data)
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    # print(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/net_loss_history.png")
    plt.close()
    return net


def show_net_results(net):
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    x = x_monte
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    limit_margin = 0.1
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_monte_list.append(p_monte)
        current_min = p_monte.min()
        current_max = p_monte.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        net_out = net(pt_x, pt_t1).data.cpu().numpy()
        phat = net_out[:,0].reshape(-1,1)
        ehat = net_out[:,1].reshape(-1,1)
        p_hat_list.append(phat)
        e1 = p_monte - phat
        e1_list.append(e1)
        e1_hat_list.append(ehat)
    
    limit_margin = 0.1

    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
        if i == 1:
            ax1 = axs[1,0]
        if i == 2:
            ax1 = axs[2,0]
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
        eL = 2.0 * np.max(np.abs(e1_hat))
        ax1.plot(x, p_monte, "blue", label=r"$p$")
        ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, color="green", alpha=0.2)
        ax1.set_ylim(global_min-limit_margin, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/net_result_1.png")
    plt.close()

    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i, (e1, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        eL = 2.0 * np.max(np.abs(e1_hat))
        if i == 0:
            ax1 = axs[0,0]
        if i == 1:
            ax1 = axs[1,0]
        if i == 2:
            ax1 = axs[2,0]
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
        print("eL: ", np.round(eL,3))
        print("a1: ", np.max(np.abs(e1-e1_hat))/np.max(np.abs(e1_hat)))
        if i == 0:
            ax1.plot(x, e1, "blue", label=r"$e$")
            ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+eL, y2=0*x.reshape(-1)-eL, color="green", alpha=0.2)
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, e1, "blue")
            ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+eL, y2=0*x.reshape(-1)-eL, color="green", alpha=0.2)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/net_result_2.png")
    plt.close()
    

def plot_p_monte():
    markers = ['o', 's', 'D', '^', 'v', '<']
    plt.figure(figsize=(8,6))
    for i in range(len(t1s)):
        t1 = t1s[i]
        x_sim = np.load(DATA_FOLDER+ datas[0] + "xsim.npy")
        p_sim = np.load(DATA_FOLDER+ datas[0] + "psim_t"+str(t1)+".npy")
        line, = plt.plot(x_sim, p_sim, linewidth=1.0)
        
        # Plot markers at specific intervals
        interval = 5  # Marker interval
        plt.plot(x_sim[::interval], p_sim[::interval], markers[i], markersize=3, color=line.get_color(), label="t="+str(t1))  # 'o' marker style


    plt.grid(linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/p_sol_monte.png")
    print("save fig to "+FOLDER+"figs/p_sol_monte.png")
    plt.close()
        

def main():
    plot_p_monte()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    model = PENet().to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    start_time = time.time()
    train_model(model, optimizer, scheduler, mse_cost_function, iterations=300000); print("[p_net train complete]")
    end_time = time.time()
    time_train = end_time - start_time
    model = pos_p_net_train(model, PATH=FOLDER+"output/net.pt", PATH_LOSS=FOLDER+"output/net_train_loss.npy"); model.eval()
    show_net_results(model)
    print(f"train net time: {time_train:.4f} seconds")


if __name__ == "__main__":
    main()
