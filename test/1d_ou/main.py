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

x0 = 1
x_low = -6
x_hig = 6
alpha = 0.2
D = 0.2

t0 = 1
T_end = 3
dt = 0.1

def p_exact(x,t):
  return np.sqrt(alpha/(2*np.pi*D*(1-np.exp(-2*alpha*t)))) * np.exp(-1*alpha*(x-x0*np.exp(-alpha*t))**2/(2*D*(1-np.exp(-2*alpha*t))))

def p_init(x):
  return p_exact(x, t0)

def test_p_init():
  x = np.arange(x_low,x_hig,0.01).reshape(-1,1)
  p0 = p_init(x)
  print(np.sum(p0)*0.01)
  plt.plot(x,p0, "b", label="t0")
  p1 = p_exact(x, T_end)
  print(np.sum(p1)*0.01)
  plt.plot(x,p1, "r", label="T_end")
  plt.xlabel("x")
  plt.ylabel("pdf")
  plt.legend()


# def f_sde(x):
#   global a, b, c, d
#   return a*(x**3) + b*(x**2) + c*x + d


def res_func(x,t, net, verbose=False):
    global D
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t - alpha*(p_x*x + p) - D*p_xx
    if(verbose):
      print(p)
      print(residual)
      print(residual.shape)
    return residual


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 64
        super(Net, self).__init__()
        self.hidden_layer1 = spectral_norm(nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        output = torch.square(output)
        # output = nn.functional.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def train_p_net(p_net, optimizer, mse_cost_function):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 3000
    min_loss = np.inf
    loss_history = []
    PATH = "output/p_net.pt"
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        u_bc = p_init(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out, all_zeros)

        loss = mse_u + mse_f

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        # Save the min loss model
        if(loss.data < min_loss):
            print("save epoch:", epoch, ", loss:", loss.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
            loss_history.append(loss.data)
        # with torch.autograd.no_grad():
        #     if (epoch%4000 == 0):
        #         print(epoch,"Traning Loss:",loss.data)
    np.save("output/p_net_train_loss.npy", np.array(loss_history))


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
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figs/pnet_loss_history.png")
    plt.close()
    return p_net


def show_p_net_results(p_net):
    global x_low, x_hig, X_sim, P_sim_t20, t0
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 2
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    p_exact1 = p_exact(x, T1)

    plt.figure()
    plt.plot(x, p_approx0, "b--", linewidth=0.5, label="Approx t")
    plt.plot(x, p_exact0,  "b", linewidth=0.5, label="True t: ")
    plt.plot(x, p_approx1, "r--", linewidth=0.5, label="Approx t")
    plt.plot(x, p_exact1,  "r", linewidth=0.5, label="True t: ")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.savefig("figs/pnet_approx.png")
    plt.close()

    res_0 = res_func(pt_x, pt_T0, p_net)
    res_1 = res_func(pt_x, pt_T1, p_net)
    plt.figure()
    plt.plot(x, res_0.detach().numpy(), "blue", alpha=0.3, label="res t0")
    plt.plot(x, res_1.detach().numpy(), "red", alpha=0.3, label="res t1")
    plt.plot([x_low, x_hig], [0,0], "black")
    plt.legend()
    plt.savefig("figs/pnet_resdiual.png")
    plt.close()

    max_abs_e1_x_0 = max(abs(p_exact0-p_approx0))
    print(max_abs_e1_x_0[0])

    return max_abs_e1_x_0[0]


class E1Net(nn.Module):
    def __init__(self):
        neurons = 32
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        # self.layer_norm1 = nn.LayerNorm(neurons)
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        # self.layer_norm2 = nn.LayerNorm(neurons)
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        # self.layer_norm3 = nn.LayerNorm(neurons)
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh((self.hidden_layer1(inputs)))
        layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
        output = self.output_layer(layer3_out)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def e1_res_func(x, t, e1_net, p_net, verbose=False):
    global D
    e = e1_net(x,t)
    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = e_t - alpha*(e_x*x + e) - D*e_xx + p_res
    return residual


# def test_e1_res(e1_net, p_net):
#     batch_size = 5
#     x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
#     t_collocation = T_end*np.ones((batch_size,1))
#     all_zeros = np.zeros((batch_size,1))
#     pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
#     pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
#     f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)


def train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 40000
    min_loss = np.inf
    loss_history = []
    PATH = "output/e1_net.pt"
    
    # dx_train = 0.05
    # dt_step = 0.01
    # dt_train = 0.01
    x_mar = 0
    t_mar = 0

    reg_mse_res_t0 = 0.0
    reg_mse_res    = 0.5 #(maybe increasing this further)
    reg_alpha1_t0  = 0.0 #(the order of targeted loss of ic)

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        # x_quad = torch.arange(x_low-x_mar, x_hig+dx_train+x_mar, dx_train, requires_grad=True).view(-1, 1).to(device)
        x_bc = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        t_bc = 0*x_bc + t0
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out, u_bc)
        mse_u = mse_u/max_abs_e1_x_0

        # Loss based on PDE
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0 + 2*t_mar) + t0-t_mar).to(device)
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = e1_res_func(x, t, e1_net, p_net)
        mse_res = mse_cost_function(f_out, all_zeros)/max_abs_e1_x_0

        # Loss based on alpha_1(t0)
        # e1hat_t0 = e1_net(x_quad, 0*x_quad + t0)
        # p_t0 = p_init_torch(x_quad)
        # phat_t0 = p_net(x_quad, 0*x_quad + t0)
        # e1_t0 = p_t0 - phat_t0
        # alpha1_t0 = max(abs(e1_t0-e1hat_t0))/max(abs(e1hat_t0))
        
        # Combining the loss functions
        loss = (1-reg_alpha1_t0-reg_mse_res_t0-reg_mse_res)*mse_u  + reg_mse_res*mse_res # + reg_alpha1_t0*alpha1_t0 + reg_mse_res_t0*mse_res_t0 + reg_mse_res*mse_res
        loss.backward() 
        optimizer.step()

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data) # , ",a1(t0):", alpha1_t0.data, ",ic loss:", mse_u.data, ",res:" , mse_res_t0.data, mse_res.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
            loss_history.append(loss.data)
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
    np.save("output/e1_net_train_loss.npy", np.array(loss_history))


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
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figs/e1net_loss_history.png")
    plt.close()
    return e1_net


def show_uniform_bound(p_net, e1_net):
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 3.0
    T0 = 0*x + t0
    T1 = 0*x + t1
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
    p_i = p_init(x)
    p_f = p_exact(x, T_end)
    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_i
    p_exact1 = p_exact(x, t1)
    e1_exact_0 = p_exact0 - p_approx0
    e1_exact_1 = p_exact1 - p_approx1
    # e1_net
    e1_0 = e1_net(pt_x, pt_T0).data.cpu().numpy()
    e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
    e_bound_1 = max(abs(e1_1))*2
    print(e_bound_1)
    # plot unform error bound in p space at t
    plt.figure(figsize=(8,6))
    plt.plot(x, p_i, "black", linewidth=0.5, alpha=0.5, label="$p_i$")
    plt.plot(x, p_f, "red", linewidth=0.5, alpha=0.5, label="$p_f$")
    plt.plot(x, p_exact1, "b", linewidth=0.5, label="p (t)")
    plt.plot(x, p_approx1, "b--", linewidth=0.5, label="p_net(t)")
    plt.fill_between(x.reshape(-1), y1=p_approx1.reshape(-1)+e_bound_1, y2=p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound(t)")
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('y')
    textstr = 't='+str(t1) + '\nError Bound='+str(np.round(e_bound_1[0],3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.savefig("figs/uniform_error_t.png")
    plt.close()
    # plot unform error bound in error space at t
    plt.figure(figsize=(8,6))
    plt.fill_between(x.reshape(-1), y1=0*p_approx1.reshape(-1)+e_bound_1, y2=0*p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound")
    plt.scatter(x, e1_exact_1, s=5, color="blue", marker="*", label="$e_1(t)$")
    plt.plot(x, e1_1, "b--", label="e1 Approx(t)")
    plt.plot([x_low, x_hig], [-e_bound_1, -e_bound_1], "black")
    plt.plot([x_low, x_hig], [e_bound_1, e_bound_1], "black")
    plt.plot([x_low, x_hig], [0,0], "black", linewidth=0.5, linestyle=":")
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('error')
    plt.savefig("figs/uniform_error_space_t.png")
    plt.close()
    # plot e1 approximation
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, e1_exact_0, "b")
    plt.plot(x, e1_0, "b--")
    plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    plt.subplot(2,1,2)
    plt.plot(x, e1_exact_1, "r")
    plt.plot(x, e1_1, "r--")
    plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    plt.savefig("figs/e1net_approx_ti_t")
    plt.close()


def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters())
    # train_p_net(p_net, optimizer, mse_cost_function); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pt", PATH_LOSS="output/p_net_train_loss.npy")
    show_p_net_results(p_net)
    max_abs_e1_x_0 = show_p_net_results(p_net)

    # create e1_net
    e1_net = E1Net()
    e1_net = e1_net.to(device)
    optimizer = torch.optim.Adam(e1_net.parameters())
    # test_e1_res(e1_net, p_net)
    # train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH="output/e1_net.pt", PATH_LOSS="output/e1_net_train_loss.npy")
    show_uniform_bound(p_net, e1_net)



    print("[complete 1d OU]")

if __name__ == "__main__":
    main()