import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = "cpu"
print(device)

mu = -2
std = 1
a = -0.1
b = 0.1
c = 0.0
d = 1
e = 1
x_low = -5
x_hig = 5

t0 = 0
T_end = 2
dt = 0.1

X_sim = np.load("data/exp1/X_sim.npy")
P_sim_t01 = np.load("data/exp1/P_sim_t01.npy")
P_sim_t05 = np.load("data/exp1/P_sim_t05.npy")
P_sim_t10 = np.load("data/exp1/P_sim_t10.npy")
P_sim_t11 = np.load("data/exp1/P_sim_t11.npy")
P_sim_t15 = np.load("data/exp1/P_sim_t15.npy")
P_sim_t20 = np.load("data/exp1/P_sim_t20.npy")


# def f_sde(x):
#   global a, b, c, d
#   return a*(x**3) + b*(x**2) + c*x + d


def p_init(x):
  global mu, std
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def res_func(x,t, net, verbose=False):
    global a, b, c, d, e
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*a*x*x + 2*b*x + c)*p + (a*x*x*x + b*x*x + c*x + d)*p_x - 0.5*e*e*p_xx
    return residual


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 32
        super(Net, self).__init__()
        self.hidden_layer1 = spectral_norm(nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = nn.functional.softplus(self.hidden_layer1(inputs))
        layer2_out = nn.functional.softplus(self.hidden_layer2(layer1_out))
        layer3_out = nn.functional.softplus(self.hidden_layer3(layer2_out))
        layer4_out = nn.functional.softplus(self.hidden_layer4(layer3_out))
        layer5_out = nn.functional.softplus(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        # output = torch.square(output)
        output = nn.functional.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data)
    # see training result
    keys = p_net.state_dict().keys()
    for k in keys:
        l2_norm = torch.norm(p_net.state_dict()[k], p=2)
        print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    # loss_history = np.load(PATH_LOSS)
    return p_net


def show_p_net_results(p_net):
    global x_low, x_hig, X_sim, P_sim_t20, t0
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 2
    P_sim = P_sim_t20
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    p_exact1 = P_sim

    plt.figure()
    plt.plot(x, p_approx0, "b--", label="NN t")
    plt.plot(x, p_exact0,  "b:", linewidth=5, label="True t: ")
    plt.plot(x, p_approx1, "r--", label="NN t")
    plt.plot(X_sim, p_exact1,  "r:", linewidth=5, label="True t: ")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.show()

    res_0 = res_func(pt_x, pt_T0, p_net)
    res_1 = res_func(pt_x, pt_T1, p_net)
    plt.figure()
    plt.plot(x, res_0.detach().numpy(), "b--", label="res t0")
    plt.plot(x, res_1.detach().numpy(), "r--", label="res t1")
    plt.plot([x_low, x_hig], [0,0], "black")
    plt.legend()
    plt.show()

    max_abs_e1_x_0 = max(abs(p_exact0-p_approx0))
    print("max abs e1(x,0):", max_abs_e1_x_0[0])

    return max_abs_e1_x_0[0]


class E1Net(nn.Module):
    def __init__(self):
        neurons = 32
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh((self.hidden_layer1(inputs)))
        layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = torch.tanh((self.hidden_layer4(layer3_out)))
        layer5_out = torch.tanh((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def e1_res_func(x, t, e1_net, p_net, max_abs_e1_x_0, verbose=False):
    global a, b, c, d, e
    net_out = e1_net(x,t)*max_abs_e1_x_0
    net_out_x = torch.autograd.grad(net_out, x, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_t = torch.autograd.grad(net_out, t, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_xx = torch.autograd.grad(net_out_x, x, grad_outputs=torch.ones_like(net_out_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = net_out_t + (3*a*x*x + 2*b*x + c)*net_out + (a*x*x*x + b*x*x + c*x + d)*net_out_x - 0.5*e*e*net_out_xx + p_res
    return residual


# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best epoch: ", epoch, ", loss:", loss.data)
    # see training result
    keys = e1_net.state_dict().keys()
    for k in keys:
        l2_norm = torch.norm(e1_net.state_dict()[k], p=2)
        print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    # loss_history = np.load(PATH_LOSS)
    # min_loss = min(loss_history)
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.savefig("figs/e1net_loss_history.png")
    # plt.close()
    return e1_net


def show_e1_net(p_net, e1_net, max_abs_e1_x_0):
    global x_low, x_hig, t0, X_sim, P_sim_t01, P_sim_t05, P_sim_t10, P_sim_t15, P_sim_t20
    x = np.arange(x_low, x_hig+0.1, 0.1).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    x_sim_ = X_sim
    t1s = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    plt.figure(figsize=(8,6))
    for t1 in t1s:
        if t1 == 0.1:
            p_exact_t = P_sim_t01
        if t1 == 0.5:
            p_exact_t = P_sim_t05
        if t1 == 1.0:
            p_exact_t = P_sim_t10
        if t1 == 1.5:
            p_exact_t = P_sim_t15
        if t1 == 2.0:
            p_exact_t = P_sim_t20
        T1 = 0*x + t1
        pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        e1 = max_abs_e1_x_0*e1_net(pt_x, pt_T1).data.cpu().numpy()
        e_bound = max(abs(e1))*2
        pt_x_sim = Variable(torch.from_numpy(x_sim_.reshape(-1,1)).float(), requires_grad=False).to(device)
        pt_t_sim = 0*pt_x_sim + t1
        error_t = p_exact_t.reshape(-1,1) - p_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
        line1, = plt.plot(x, e1, linestyle="--", label=str(t1))
        plt.plot(x_sim_, error_t, color=line1.get_color(), linewidth=0.5, label=str(t1))
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,6))
    for t1 in t1s:
        if t1 == 0.1:
            p_exact_t = P_sim_t01
        if t1 == 0.5:
            p_exact_t = P_sim_t05
        if t1 == 1.0:
            p_exact_t = P_sim_t10
        if t1 == 1.5:
            p_exact_t = P_sim_t15
        if t1 == 2.0:
            p_exact_t = P_sim_t20
        pt_x_sim = Variable(torch.from_numpy(x_sim_.reshape(-1,1)).float(), requires_grad=False).to(device)
        pt_t_sim = 0*pt_x_sim + t1
        error_t = p_exact_t.reshape(-1,1) - p_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
        e1 = max_abs_e1_x_0*e1_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
        line1, = plt.plot(x_sim_, error_t-e1, linestyle=":", label=str(t1))
        plt.plot(x_sim_, error_t, color=line1.get_color(), linestyle="--", linewidth=0.5, label=str(t1))
        plt.plot(x_sim_, e1, color=line1.get_color(),  linestyle="-", linewidth=0.5, label=str(t1))
        print("alpha1("+str(t1)+"):", max(abs(error_t-e1))/max(abs(e1)))
    plt.legend()
    plt.show()

    plt.figure()
    t = np.linspace(0.0, 2.0, 100).reshape(-1,1)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    for x_eval in [-5, 0, 5]:
        x = x_eval *np.ones((100,1)).reshape(-1,1)
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        e1_res_x = e1_res_func(pt_x, pt_t, e1_net, p_net, max_abs_e1_x_0).data.cpu().numpy()
        cum_sum = np.cumsum(e1_res_x)
        plt.plot(t, cum_sum, label="x="+str(x_eval))
    plt.title("time integral of r2(x,t) at x")
    plt.xlabel('t')
    plt.legend()
    plt.show()


def show_uniform_bound(p_net, e1_net, max_abs_e1_x_0):
    global x_low, x_hig, t0, X_sim, P_sim_t10, P_sim_t15, P_sim_t20
    x = np.arange(x_low, x_hig+0.1, 0.1).reshape(-1,1)
    x_sim_ = X_sim
    t1 = 0.1
    P_sim = P_sim_t01
    T0 = 0*x + t0
    T1 = 0*x + t1
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
    p_i = p_init(x)
    p_f = P_sim_t20

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact1 = P_sim
    e1_exact_0 = p_i - p_approx0

    # e1_net
    e1_0 = max_abs_e1_x_0*e1_net(pt_x, pt_T0).data.cpu().numpy()
    e1_1 = max_abs_e1_x_0*e1_net(pt_x, pt_T1).data.cpu().numpy()
    e_bound_1 = max(abs(e1_1))*2
    print("Error bound:", e_bound_1)
    print("error(t0)", max(abs(e1_exact_0-e1_0))/ max(abs(e1_exact_0)))
    print("alpha(t0)", max(abs(e1_exact_0-e1_0))/ max(abs(e1_0)))
    print("debug", pt_x[0], pt_T0[0], e1_0[0])

    # plot unform error bound in p space at t
    plt.figure(figsize=(8,6))
    plt.plot(x, p_i, "black", linewidth=0.5, alpha=0.5, label="$p_i$")
    plt.plot(x_sim_, p_f, "red", linewidth=0.5, alpha=0.5, label="$p_f$ [Monte]")
    plt.plot(x, p_approx1, "b", linewidth=0.5, label="p_net(t)")
    plt.scatter(x_sim_, p_exact1, s=2, color="blue", marker="*", label="$p(t)$ [Monte]")
    plt.fill_between(x.reshape(-1), y1=p_approx1.reshape(-1)+e_bound_1, y2=p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound(t)")
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('y')
    textstr = 't='+str(t1) + '\nError Bound='+str(np.round(e_bound_1[0],3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.show()
    # plt.savefig("figs/uniform_error_t.png")
    # plt.close()

    # plot unform error bound in error space at t
    plt.figure(figsize=(8,6))
    x_sim_ = x_sim_.reshape(-1,1)
    pt_x_sim = Variable(torch.from_numpy(x_sim_).float(), requires_grad=False).to(device)
    pt_t_sim = 0*pt_x_sim + t1
    e1_t = max_abs_e1_x_0*e1_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
    error_t = p_exact1.reshape(-1,1) - p_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
    alpha1_t = max(abs(error_t - e1_t))/ max(abs(e1_t))
    print("alpha1(t):", alpha1_t)
    plt.fill_between(x.reshape(-1), y1=0*p_approx1.reshape(-1)+e_bound_1, y2=0*p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound")
    plt.scatter(x_sim_, error_t, s=5, color="blue", marker="*", label="$e_1(t)$ [Monte]")
    plt.plot(x, e1_1, "b--", label="$e_1(t)$ Approx")
    plt.plot([x_low, x_hig], [-e_bound_1, -e_bound_1], "black")
    plt.plot([x_low, x_hig], [e_bound_1, e_bound_1], "black")
    plt.plot([x_low, x_hig], [0, 0], "k:", linewidth=0.5)
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('error')
    plt.show()
    # plt.savefig("figs/uniform_error_space_t.png")
    # plt.close()

    # plot e1 approximation
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, e1_exact_0, "b--")
    plt.plot(x, e1_0, "b")
    plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    plt.subplot(2,1,2)
    plt.plot(x, e1_1, "r")
    plt.scatter(x_sim_, error_t, s=5, color="r", marker="*", label="$e_1(t)$ [Monte]")
    plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    plt.show()
    # plt.savefig("figs/e1net_approx_ti_t")
    # plt.close()


def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    p_net = pos_p_net_train(p_net, PATH="output/exp1/p_net.pt", PATH_LOSS="output/exp1/p_net_train_loss.npy")
    p_net.eval()
    max_abs_e1_x_0 = show_p_net_results(p_net)

    print("\n========\n")
    # create e1_net
    e1_net = E1Net()
    e1_net = e1_net.to(device)
    e1_net = pos_e1_net_train(e1_net, PATH="output/exp1/e1_net.pt", PATH_LOSS="output/exp1/e1_net_train_loss.npy")
    e1_net.eval()
    show_e1_net(p_net, e1_net, max_abs_e1_x_0)
    show_uniform_bound(p_net, e1_net, max_abs_e1_x_0)

    print("[complete rational nonlinear]")

if __name__ == "__main__":
    main()