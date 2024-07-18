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
import warnings


FOLDER = "exp1/run-8.1/"
DATA_FOLDER = "exp1/data/"
DATA_FOLDER_TOSAVE = "data1/"
device = "cpu"; print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

n_d = 1
mu = -2
std = 0.5
a = -0.1
b = 0.1
c = 0.5
d = 0.5
e = 0.8
x_low = -6
x_hig = 6

t0 = 0
T_end = 5
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1.0/", "data1.1/", "data1.2/", "data1.3/", "data1.4/", "data1.5/", "data1.6/", "data1.7/", "data1.8/", "data1.9/"]
labels = ["", "", "", "", "", "", "", "", "", ""]


def p_init(x):
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def test_p_init():
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    p = p_init(x)
    return max(abs(p))[0]


def res_func(x,t, net, verbose=False):
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*a*x*x + 2*b*x + c)*p + (a*x*x*x + b*x*x + c*x + d)*p_x - 0.5*e*e*p_xx
    return residual


# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# p_net
class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 30
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.hidden_layer8 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        # normalization to [0-1]
        # _x = 2.0*(x-x_low)/(x_hig-x_low) - 1.0
        # _t = 2.0*(t-t0)/(T_end-t0) - 1.0
        # inputs = torch.cat([_x,_t],axis=1)
        inputs = torch.cat([x, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
        layer7_out = F.softplus((self.hidden_layer7(layer6_out)))
        layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
        output = F.softplus( self.output_layer(layer8_out) )
        return output


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return p_net


def show_p_net_results(p_net):
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    x = x_monte
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x*0+t0).float(), requires_grad=True).to(device)
    p     = p_init(x)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1 = p - p_hat
    max_abs_e1_ti = max(abs(e1))[0]
    return max_abs_e1_ti

    
class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 30
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(18,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.hidden_layer8 = (nn.Linear(neurons,neurons))
        self.hidden_layer9 = (nn.Linear(neurons,neurons))
        self.hidden_layer10 = (nn.Linear(neurons,neurons))
        self.hidden_layer11 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
    def forward(self, x, t):
        ### Transformer position encodint ###
        # position embedding d=8, n=100
        # 0 <= i < d/2
        w1 = (1/pow(100, 2*0/8)) # power(n, (2*i)/d)
        w2 = (1/pow(100, 2*1/8)) # power(n, (2*(i+1))/d)
        w3 = (1/pow(100, 2*2/8))
        w4 = (1/pow(100, 2*3/8))

        x_p1 = torch.sin(w1*x)
        x_p2 = torch.cos(w1*x)
        x_p3 = torch.sin(w2*x)
        x_p4 = torch.cos(w2*x)
        x_p5 = torch.sin(w3*x)
        x_p6 = torch.cos(w3*x)
        x_p7 = torch.sin(w4*x)
        x_p8 = torch.cos(w4*x)

        t_p1 = torch.sin(w1*t)
        t_p2 = torch.cos(w1*t)
        t_p3 = torch.sin(w2*t)
        t_p4 = torch.cos(w2*t)
        t_p5 = torch.sin(w3*t)
        t_p6 = torch.cos(w3*t)
        t_p7 = torch.sin(w4*t)
        t_p8 = torch.cos(w4*t)

        inputs = torch.cat([x, t, 
                            x_p1, x_p2, x_p3, x_p4, x_p5, x_p6, x_p7, x_p8, 
                            t_p1, t_p2, t_p3, t_p4, t_p5, t_p6, t_p7, t_p8],axis=1)
        # inputs = torch.cat([x,t],axis=1)

        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        layer7_out = self.activation((self.hidden_layer7(layer6_out)))
        layer8_out = self.activation((self.hidden_layer8(layer7_out)))
        layer9_out = self.activation((self.hidden_layer9(layer8_out)))
        layer10_out = self.activation((self.hidden_layer10(layer9_out)))
        layer11_out = self.activation((self.hidden_layer11(layer10_out)))
        output = self.output_layer(layer11_out)
        output = self.scale * output
        return output
    

def e1_res_func(x, t, e1_net, p_net, verbose=False):
    net_out = e1_net(x,t)
    net_out_x = torch.autograd.grad(net_out, x, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_t = torch.autograd.grad(net_out, t, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_xx = torch.autograd.grad(net_out_x, x, grad_outputs=torch.ones_like(net_out_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = net_out_t + (3*a*x*x + 2*b*x + c)*net_out + (a*x*x*x + b*x*x + c*x + d)*net_out_x - 0.5*e*e*net_out_xx + p_res
    return residual


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return e1_net


def show_p_net_results_detail(p_net, e1_net):
    t1 = 4.0

    fig, axs = plt.subplots(1,1, figsize=(8,6))
    for k in range(len(datas)):
        x = np.load(DATA_FOLDER+datas[k]+"xsim.npy").reshape(-1,1)
        p_t1_monte = np.load(DATA_FOLDER+datas[k]+"psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        p_t1_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1_monte = p_t1_monte - p_t1_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        print( max(abs(e1_monte-e1_hat))/max(abs(e1_hat)))
        axs.plot(x, e1_monte, "blue", linewidth=0.5, alpha=0.2, label=datas[k]+labels[k])

    # sum of monte
    x = np.load(DATA_FOLDER+datas[0]+"xsim.npy").reshape(-1,1)
    p_t1_monte_sum = 0.0*x
    for k in range(len(datas)):
        p_t1_monte = np.load(DATA_FOLDER+datas[k]+"psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_t1_monte_sum = p_t1_monte_sum + (1/len(datas))*p_t1_monte
        np.save(DATA_FOLDER + DATA_FOLDER_TOSAVE + "psim_t"+str(t1)+".npy", p_t1_monte_sum)
    e1_monte_sum = p_t1_monte_sum - p_t1_hat
    axs.plot(x, e1_monte_sum, "black", label="monte sum")
    print( max(abs(e1_monte_sum-e1_hat))/max(abs(e1_hat)))
    
    # plot exact if t1 = ti
    if(t1 == 0.0):
        p_ti = p_init(x).reshape(-1,1)
        p_ti_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
        axs.plot(x, p_ti-p_ti_hat, linewidth=1.0, label="e1_true")

    # e1_hat plot
    axs.plot(x, e1_hat, "black", linestyle="--", linewidth=2.0, label="e1_hat")

    # reference plot
    axs.plot(x, x*0.0, "black", linestyle=":", linewidth=0.5)

    plt.legend()
    plt.grid()
    plt.show()


def main():
    p_net = Net().to(device)
    p_net.apply(init_weights)
    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    
    max_abs_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,0):", max_abs_e1_ti)
    e1_net = E1Net(scale=max_abs_e1_ti).to(device)
    e1_net.apply(init_weights)
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()

    show_p_net_results_detail(p_net, e1_net)


if __name__ == "__main__":
    main()
