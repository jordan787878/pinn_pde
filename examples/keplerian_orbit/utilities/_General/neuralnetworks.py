import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PNet(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=5): 
        super(PNet, self).__init__()
        neurons = 50
        self.scale = scale
        self.constants = constants
        self.input_feature = input_feature
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        if(self.input_feature == 5):
            inputs = normalize_inputs(x, t, self.constants)
        elif(self.input_feature == 7):
            inputs = normalize_inputs_6d(x, t, self.constants)
        else:
            raise("this PINN is not yet implemented for input feature: {self.input_feature}")
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = ((self.hidden_layer4(layer3_out)))
        output = F.softplus(self.output_layer(layer4_out + layer1_out)) * self.scale
        return output
    

class PNet_Scaled(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=5): 
        super(PNet_Scaled, self).__init__()
        neurons = 50
        self.scale = scale
        self.constants = constants
        self.input_feature = input_feature
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        if(self.input_feature == 5):
            inputs = normalize_inputs(x, t, self.constants)
        elif(self.input_feature == 7):
            inputs = normalize_inputs_6d_scaled(x, t, self.constants)
        else:
            raise("this PINN is not yet implemented for input feature: {self.input_feature}")
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = ((self.hidden_layer4(layer3_out)))
        output = F.softplus(self.output_layer(layer4_out + layer1_out)) * self.scale
        return output
    

# ---- small helpers ----
def get_activation(name: str):
    name = name.lower()
    if name in ["silu", "swish"]:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "softplus":
        # softer than default beta=1 to avoid stiff grads
        return nn.Softplus(beta=1.0)
    raise ValueError(f"Unknown activation: {name}")

class FourierFeatures(nn.Module):
    """
    Positional encoding for a 1D scalar (here: time).
    Produces: [sin(2^k * pi * t), cos(2^k * pi * t)] for k=0..num_frequencies-1
    """
    def __init__(self, num_frequencies=8, include_input=True):
        super().__init__()
        self.num_f = int(num_frequencies)
        self.include_input = include_input
        # fixed frequencies: powers of 2 * pi
        self.register_buffer("freqs", (2.0 ** torch.arange(self.num_f)) * math.pi)

    def forward(self, t):  # t shape: (N,1) or (N,)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # (N, num_f)
        wt = t * self.freqs
        feats = [torch.sin(wt), torch.cos(wt)]
        out = torch.cat(feats, dim=1)  # (N, 2*num_f)
        if self.include_input:
            out = torch.cat([t, out], dim=1)  # (N, 1+2*num_f)
        return out

class ResidualBlock(nn.Module):
    """
    Simple 2-layer pre-activation residual block for MLPs.
    x -> LN(optional) -> Act -> Linear -> Act -> Linear -> + skip
    """
    def __init__(self, width, act="silu", use_layernorm=False):
        super().__init__()
        self.use_ln = use_layernorm
        self.ln1 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.act = get_activation(act)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

        # Kaiming init tailored to SiLU/RELU-like activations
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        h = self.ln1(x)
        h = self.act(h)
        h = self.fc1(h)

        h = self.ln2(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h

# ---- Bigger, deeper PINN head ----
class PNet_XL(nn.Module):
    def __init__(
        self,
        constants,
        scale=1.0,
        input_feature=5,
        width=64,
        depth=8,
        act="silu",
        use_layernorm=False,
        fourier_t=False,
        fourier_f=8,
        input_skip_at=4
    ):
        super().__init__()
        self.constants = constants
        self.scale = scale
        self.input_feature = input_feature
        self.width = int(width)
        self.depth = int(depth)
        self.use_layernorm = use_layernorm
        self.input_skip_at = input_skip_at if (0 < input_skip_at < depth) else -1

        # Time encoding
        self.use_fourier_t = bool(fourier_t)
        if self.use_fourier_t:
            self.time_embed = FourierFeatures(num_frequencies=fourier_f, include_input=True)
            self.t_dim = 1 + 2 * fourier_f
        else:
            self.time_embed = nn.Identity()
            self.t_dim = 1

        in_dim = int(input_feature + (self.t_dim - 1))
        self.fc_in = nn.Linear(in_dim, self.width)

        # Residual trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(self.width, act=act, use_layernorm=use_layernorm)
            for _ in range(self.depth)
        ])

        if self.input_skip_at > 0:
            self.proj_after_skip = nn.Linear(self.width + in_dim, self.width)
        else:
            self.proj_after_skip = None

        self.fc_out = nn.Linear(self.width, 1)
        self.act = get_activation(act)
        self.softplus_out = nn.Softplus(beta=1.0)

        # **Initialize weights after defining all layers**
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization:
          - Kaiming init for hidden/residual layers (good for SiLU/GELU)
          - Xavier init for output for smoother initial pdf scaling
          - Zero bias where appropriate
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _build_inputs(self, x, t):
        if self.input_feature == 5:
            inputs = normalize_inputs(x, t, self.constants)
        elif self.input_feature == 7:
            inputs = normalize_inputs_6d_scaled(x, t, self.constants)
        else:
            raise RuntimeError(f"PINN not yet implemented for input feature: {self.input_feature}")

        if self.use_fourier_t:
            t_enc = self.time_embed(t if t.dim() == 2 else t.view(-1, 1))
            inputs = torch.cat([inputs[:, :-1], t_enc], dim=1)
        return inputs

    def forward(self, x, t):
        inputs = self._build_inputs(x, t)
        h0 = self.fc_in(inputs)
        h = h0
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == self.input_skip_at and self.proj_after_skip is not None:
                h = torch.cat([h, inputs], dim=1)
                h = self.proj_after_skip(h)

        out_hidden = h + h0
        y = self.fc_out(out_hidden)
        return self.softplus_out(y) * self.scale


##### E1 Nueral Networks #####


class E1Net(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=5): 
        super(E1Net, self).__init__()
        neurons = 50
        self.constants = constants
        self.scale = scale
        self.input_feature = input_feature
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        if(self.input_feature == 5):
            inputs = normalize_inputs(x, t, self.constants)
        elif(self.input_feature == 7):
            inputs = normalize_inputs_6d(x, t, self.constants)
        else:
            raise("this PINN is not yet implemented for input feature: {self.input_feature}")
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.gelu((self.hidden_layer2(layer1_out)))   
        layer3_out = ((self.hidden_layer3(layer2_out)))
        layer4_out = F.gelu((self.hidden_layer4(layer3_out)))
        layer5_out = ((self.hidden_layer5(layer4_out))) 
        layer6_out = F.gelu((self.hidden_layer6(layer5_out)))
        layer7_out = ((self.hidden_layer7(layer6_out)))
        output = (self.output_layer(layer7_out + layer1_out)) * self.scale
        return output
    

class E1Net_Equin(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=5, p_net=None): 
        super(E1Net_Equin, self).__init__()
        neurons = 80
        self.scale = scale
        self.constants = constants
        self.input_feature = input_feature
        self.p_net = p_net
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        if(self.input_feature == 5):
            inputs = normalize_inputs(x, t, self.constants)
        elif(self.input_feature == 7):
            inputs = normalize_inputs_6d(x, t, self.constants)
        else:
            raise("this PINN is not yet implemented for input feature: {self.input_feature}")
        layer1_out = F.tanh((self.hidden_layer1(inputs)))
        layer2_out = F.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = F.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = F.tanh((self.hidden_layer4(layer3_out)))
        output = (self.output_layer(layer4_out)) * self.p_net.scale
        p_net_out = self.p_net(x, t)
        output = output - p_net_out
        return output
    

class E1Net_Scaled(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=7): 
        super(E1Net_Scaled, self).__init__()
        neurons = 80
        self.scale = scale
        self.constants = constants
        self.input_feature = input_feature
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = normalize_inputs_6d_scaled(x, t, self.constants)
        layer1_out = F.tanh((self.hidden_layer1(inputs)))
        layer2_out = F.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = F.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = F.tanh((self.hidden_layer4(layer3_out)))
        output = (self.output_layer(layer4_out))
        return output * self.p_net.scale - self.p_net(x, t)
    def set_p_net(self, p_net):
        self.p_net = p_net
        self.p_net.eval()
        for p in self.p_net.parameters():
            p.requires_grad_(False)
    

# ---- Bigger, deeper PINN head ----
class E1Net_XL(nn.Module):
    def __init__(
        self,
        constants,
        p_net=None,
        scale=1.0,
        normalize=1.0,
        input_feature=7,
        width=64,
        depth=8,
        act="silu",
        use_layernorm=True,
        fourier_t=False,
        fourier_f=8,
        input_skip_at=4,
    ):
        super().__init__()
        self.constants = constants
        self.scale = scale
        self.p_net = None
        self.normalize = normalize
        self.input_feature = input_feature
        self.width = int(width)
        self.depth = int(depth)
        self.use_layernorm = use_layernorm
        self.input_skip_at = input_skip_at if (0 < input_skip_at < depth) else -1
        if(p_net is not None):
            self.p_net = p_net
            self.p_net.eval()
            for p in self.p_net.parameters():
                p.requires_grad_(False)
        else:
            # directly learning the error
            self.scale = self.normalize

        # Time encoding
        self.use_fourier_t = bool(fourier_t)
        if self.use_fourier_t:
            self.time_embed = FourierFeatures(num_frequencies=fourier_f, include_input=True)
            self.t_dim = 1 + 2 * fourier_f
        else:
            self.time_embed = nn.Identity()
            self.t_dim = 1

        in_dim = int(input_feature + (self.t_dim - 1))
        self.fc_in = nn.Linear(in_dim, self.width)

        # Residual trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(self.width, act=act, use_layernorm=use_layernorm)
            for _ in range(self.depth)
        ])

        if self.input_skip_at > 0:
            self.proj_after_skip = nn.Linear(self.width + in_dim, self.width)
        else:
            self.proj_after_skip = None

        self.fc_out = nn.Linear(self.width, 1)
        self.act = get_activation(act)
        self.softplus_out = nn.Softplus(beta=1.0)

        # **Initialize weights after defining all layers**
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization:
          - Kaiming init for hidden/residual layers (good for SiLU/GELU)
          - Xavier init for output for smoother initial pdf scaling
          - Zero bias where appropriate
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc_out:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _build_inputs(self, x, t):
        if self.input_feature == 5:
            inputs = normalize_inputs(x, t, self.constants)
        elif self.input_feature == 7:
            inputs = normalize_inputs_6d_scaled(x, t, self.constants)
        else:
            raise RuntimeError(f"PINN not yet implemented for input feature: {self.input_feature}")

        if self.use_fourier_t:
            t_enc = self.time_embed(t if t.dim() == 2 else t.view(-1, 1))
            inputs = torch.cat([inputs[:, :-1], t_enc], dim=1)
        return inputs

    def forward(self, x, t):
        inputs = self._build_inputs(x, t)
        h0 = self.fc_in(inputs)
        h = h0
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == self.input_skip_at and self.proj_after_skip is not None:
                h = torch.cat([h, inputs], dim=1)
                h = self.proj_after_skip(h)

        out_hidden = h + h0
        y = self.fc_out(out_hidden)
        if(self.p_net is not None):
            p_net_out = self.p_net(x, t)
            return y * self.scale - p_net_out
        else:
            return y * self.scale


# helper functions
def normalize_inputs(x, t, constants):
    _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
    _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
    _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
    _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
    _t  = t/(constants.TF/constants.T) #[0,1]
    inputs = torch.cat([_x1, _x2, _x3, _x4, _t],axis=1)
    return inputs


def normalize_inputs_6d(x, t, constants):
    _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
    _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
    _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
    _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
    _x5 = (x[:,4].view(-1, 1) - 0.5*(constants.X5_RANGE[1]+constants.X5_RANGE[0]))/(0.5*(constants.X5_RANGE[1]-constants.X5_RANGE[0]))
    _x6 = (x[:,5].view(-1, 1) - 0.5*(constants.X6_RANGE[1]+constants.X6_RANGE[0]))/(0.5*(constants.X6_RANGE[1]-constants.X6_RANGE[0]))
    _t  = t/(constants.TF/constants.T) #[0,1]
    inputs = torch.cat([_x1, _x2, _x3, _x4, _x5, _x6, _t],axis=1)
    return inputs


def normalize_inputs_6d_scaled(x, t, constants):
    _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.N_X1_RANGE[1]+constants.N_X1_RANGE[0]))/(0.5*(constants.N_X1_RANGE[1]-constants.N_X1_RANGE[0]))
    _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.N_X2_RANGE[1]+constants.N_X2_RANGE[0]))/(0.5*(constants.N_X2_RANGE[1]-constants.N_X2_RANGE[0]))
    _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.N_X3_RANGE[1]+constants.N_X3_RANGE[0]))/(0.5*(constants.N_X3_RANGE[1]-constants.N_X3_RANGE[0]))
    _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.N_X4_RANGE[1]+constants.N_X4_RANGE[0]))/(0.5*(constants.N_X4_RANGE[1]-constants.N_X4_RANGE[0]))
    _x5 = (x[:,4].view(-1, 1) - 0.5*(constants.N_X5_RANGE[1]+constants.N_X5_RANGE[0]))/(0.5*(constants.N_X5_RANGE[1]-constants.N_X5_RANGE[0]))
    _x6 = (x[:,5].view(-1, 1) - 0.5*(constants.N_X6_RANGE[1]+constants.N_X6_RANGE[0]))/(0.5*(constants.N_X6_RANGE[1]-constants.N_X6_RANGE[0]))
    _t  = t/(constants.TF/constants.T) #[0,1]
    inputs = torch.cat([_x1, _x2, _x3, _x4, _x5, _x6, _t],axis=1)
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