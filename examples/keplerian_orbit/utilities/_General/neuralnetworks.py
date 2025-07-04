import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self, constants, scale=1.0): 
        super(PNet, self).__init__()
        neurons = 50
        self.scale = scale
        self.constants = constants
        self.hidden_layer1 = (nn.Linear(5,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = normalize_inputs(x, t, self.constants)
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = ((self.hidden_layer4(layer3_out)))
        output = F.softplus(self.output_layer(layer4_out + layer1_out)) * self.scale
        return output


class E1Net(nn.Module):
    def __init__(self, constants, scale=1.0): 
        super(E1Net, self).__init__()
        neurons = 50
        self.constants = constants
        self.scale = scale
        self.hidden_layer1 = (nn.Linear(5,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = normalize_inputs(x, t, self.constants)
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.gelu((self.hidden_layer2(layer1_out)))   
        layer3_out = ((self.hidden_layer3(layer2_out)))
        layer4_out = F.gelu((self.hidden_layer4(layer3_out)))
        layer5_out = ((self.hidden_layer5(layer4_out))) 
        layer6_out = F.gelu((self.hidden_layer6(layer5_out)))
        layer7_out = ((self.hidden_layer7(layer6_out)))
        output = (self.output_layer(layer7_out + layer1_out)) * self.scale
        return output
    

# helper functions
def normalize_inputs(x, t, constants):
    _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
    _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
    _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
    _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
    _t  = t/(constants.TF/constants.T) #[0,1]
    inputs = torch.cat([_x1, _x2, _x3, _x4, _t],axis=1)
    return inputs


def load_trained_model(net, path, method="new"):
    print("[load model] from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if(method == "new"):
        loss_history = np.array(checkpoint['loss_history'])
        print("best epoch: ", epoch, ", min loss:", np.min(loss_history), ", train time:", checkpoint['train_time'])
    else:
        print("best epoch: ", epoch, ", min loss:", checkpoint['loss'], ", train time:", checkpoint['train_time'])
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


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def visual_pinn(model):
    print(model)
    pass