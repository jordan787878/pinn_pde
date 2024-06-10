import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = "cpu"
print(device)

x_low = -1
x_hig = 1

t0 = 0
T_end = 1

def p_init(x):
    return -np.sin(np.pi*x)


def p_sol(x,t):
    return -np.sin(np.pi*x)*np.exp(-np.pi**2*t)


def p_init_torch(x):
    pi = torch.tensor(torch.pi, device=device)  # Ensure this is a tensor on the same device as x
    return -torch.sin(pi*x)


def res_func(x,t, net, verbose=False):
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    Lp = -p_xx
    residual = p_t + Lp
    return residual.to(device)


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
        # output = nn.functional.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def train_p_net(p_net, optimizer, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    PATH = "output/p_net.pt"
    reg_bc = 1

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        u_bc = p_init_torch(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out, all_zeros)

        x_bc = (torch.ones(batch_size, 1) * x_low).to(device)
        net_bc_out = p_net(x_bc, t)
        mse_bc1 = mse_cost_function(net_bc_out, all_zeros)
        x_bc = (torch.ones(batch_size, 1) * x_hig).to(device)
        net_bc_out = p_net(x_bc, t)
        mse_bc2 = mse_cost_function(net_bc_out, all_zeros)
        mse_bc = 0.5*(mse_bc1 + mse_bc2)

        loss = mse_u + mse_f + reg_bc*mse_bc

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
        # with torch.autograd.no_grad():
        #     if (epoch%4000 == 0):
        #         print(epoch,"Traning Loss:",loss.data)
        loss_history.append(loss.data)
        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
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
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 0.2
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    p_exact1 = p_sol(x,T1)

    e1 = p_exact0 - p_approx0
    max_abs_e1_t0 = max(abs(p_exact0 - p_approx0))[0]
    print(max_abs_e1_t0)

    plt.figure()
    plt.plot(x, p_exact0,  "k--", label=r"$p(t_i)$")
    plt.plot(x, p_approx0, "k", label=r"$\hat{p}(t_i)$")
    plt.plot(x, p_exact1,  "r--", label=r"$p(t_f)$")
    plt.plot(x, p_approx1, "r", label=r"$\hat{p}(t_f)$")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.savefig("figs/pnet_approx.png")
    plt.close()

    res_0 = res_func(pt_x, pt_T0, p_net)
    res_1 = res_func(pt_x, pt_T1, p_net)
    plt.figure()
    plt.plot(x, res_0.detach().numpy(), "black", label=r"$r_1(t_i)$")
    plt.plot(x, res_1.detach().numpy(), "red", label=r"$r_1(t_f)$")
    plt.plot([x_low, x_hig], [0,0], "black")
    plt.legend()
    plt.savefig("figs/pnet_resdiual.png")
    plt.close()
    
    return max_abs_e1_t0


class E1Net(nn.Module):
    def __init__(self, scale=1.0):
        neurons = 100
        self.scale = scale
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
        output = output*self.scale
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def e1_res_func(x, t, e1_net, p_net, verbose=False):
    e1 = e1_net(x,t)
    e_x = torch.autograd.grad(e1, x, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e_t = torch.autograd.grad(e1, t, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    Le = -e_xx
    p_res = res_func(x, t, p_net)
    residual = e_t + Le + p_res
    if(verbose):
      print("e1net output: ", e1)
      print(residual)
      print(p_res)
    return residual


def test_e1_res(e1_net, p_net):
    batch_size = 5
    x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
    t_collocation = T_end*np.ones((batch_size,1))
    all_zeros = np.zeros((batch_size,1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_t0, iterations=40000):
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    reg_bc = 1.0
    iterations_per_decay = 1000
    PATH = "output/e1_net.pt"

    # dt_train = 0.1
    # k = 5
    # reg_alpha1_t0 = 1e-4

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        p_bc = p_init_torch(x_bc)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Loss based on boundary conditions: randomly between -1 and 1
        choices = torch.tensor([x_low, x_hig])
        rand_indices = torch.randint(0, 2, (batch_size, 1))
        x_bc = choices[rand_indices]
        t_bc = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        p_bc = 0*t_bc # p(x,t) = 0, x in boundary, known values
        p_net_bc = p_net(x_bc, t_bc)
        e1_bc = p_bc - p_net_bc
        e1_net_bc = e1_net(x_bc, t_bc)
        mse_bc = mse_cost_function(e1_net_bc, e1_bc)

        # for i in range(1,k+1):
        #     pt_t_k = t + (i)*dt_train
        #     f_out_k = e1_res_func(x, t, e1_net, p_net)
        #     mse_f_k = mse_cost_function(f_out_k, all_zeros)
        #     mse_f = mse_f_k + (mse_f_k/k)
        # # Loss based on alpha_1(t0)
        # x = torch.arange(x_low, x_hig, 0.01).view(-1, 1).to(device)
        # t = 0*x
        # e1hat_t0 = e1_net(x, t)
        # p_t0 = p_init_torch(x)
        # phat_t0 = p_net(x,t)
        # e1_t0 = p_t0 - phat_t0
        # alpha1_t0 = max(abs(e1_t0-e1hat_t0))/max(abs(e1hat_t0))
        
        # Combining the loss functions
        loss = (mse_u + mse_res + reg_bc*mse_bc)/max_abs_e1_t0

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:",mse_u.data/max_abs_e1_t0, 
                  ",res:", mse_res.data/max_abs_e1_t0, 
                  ",bc:", mse_bc.data/max_abs_e1_t0)
             #, ", alpha1(t0):", alpha1_t0.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
        loss_history.append(loss.data)
        loss.backward() 
        optimizer.step()
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
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


def show_e1_results(p_net, e1_net):
    plt.figure(figsize=(12,10))
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1s = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    j = 1
    for t1 in t1s:
        plt.subplot(3,2,j)
        T1 = 0*x + t1
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
        p_exact1 = p_sol(x, t1)
        e1_exact_1 = p_exact1 - p_approx1   
        e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
        alpha1 = max(abs(e1_exact_1-e1_1))/max(abs(e1_1))
        plt.plot(x, e1_exact_1, linestyle="--", color="blue", label=r"$e_1$")
        plt.plot(x, e1_1, color="blue", label=r"$\hat{e}_1$")
        plt.plot(x, x*0, color="black", linestyle=":", linewidth=0.5)
        plt.text(0.98, 0.02, "t="+str(t1)+ r"$, \alpha_1=$"+str(np.round(alpha1,2)), 
                 transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right')
        if(j == 1):
            plt.legend(loc="upper right")
        j = j + 1
    plt.savefig("figs/e1net_result.png")
    plt.close()


def show_uniform_bound(p_net, e1_net):
    plt.figure(figsize=(12,10))
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1s = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    j = 1
    for t1 in t1s:
        plt.subplot(3,2,j)
        T1 = 0*x + t1
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
        p_exact1 = p_sol(x, t1)
        # e1_exact_1 = p_exact1 - p_approx1   
        e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
        error_bound = max(abs(e1_1))*2
        plt.plot(x, p_exact1, color="blue", linestyle="--", label="$p$")
        plt.plot(x, p_approx1, "b", linewidth=0.5, label=r"$\hat{p}$")
        plt.fill_between(x.reshape(-1), y1=p_approx1.reshape(-1)+error_bound, y2=p_approx1.reshape(-1)-error_bound, 
                         color="blue", alpha=0.1, label="Error Bound(t)")
        textstr = 't='+str(t1) + '\nError Bound='+str(np.round(error_bound[0],3))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
        if(j == 1):
            plt.legend(loc="upper right")
        j = j + 1
    plt.savefig("figs/uniform_error_t.png")
    plt.close()


def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters())
    # train p_net
    # train_p_net(p_net, optimizer, mse_cost_function, iterations=20000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pt", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_t0 = show_p_net_results(p_net)
    print("max(abs(e1(x,0))):", max_abs_e1_t0)

    # create e1_net
    e1_net = E1Net(scale=max_abs_e1_t0)
    e1_net = e1_net.to(device)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3);   # test_e1_res(e1_net, p_net)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # train e1_net
    train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_t0, iterations=80000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH="output/e1_net.pt", PATH_LOSS="output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_results(p_net, e1_net)
    show_uniform_bound(p_net, e1_net)

    print("[complete rational nonlinear]")


if __name__ == "__main__":
    main()