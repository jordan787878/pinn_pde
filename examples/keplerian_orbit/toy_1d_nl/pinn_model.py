import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

##### PNet #####


# --- Base (Does not work well with Base training; Works very well with V0 training) ---
class PNet(nn.Module):
    def __init__(self, scale=1.0, neurons=50): 
        super().__init__()
        self.scale = scale
        self.neurons = neurons
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self._init_weights()
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        output = F.softplus( self.output_layer(layer3_out) )
        return output
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Example: Xavier uniform initialization for weights
                nn.init.xavier_uniform_(m.weight)
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---- Bigger Deeper Model (Does not work well with Base PNet training) ---
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

class PNet_XL(nn.Module):
    def __init__(
        self,
        scale=1.0,
        input_feature=2,
        width=64,
        depth=8,
        act="silu",
        use_layernorm=False,
        fourier_t=False,
        fourier_f=8,
        input_skip_at=4
    ):
        super().__init__()
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
        inputs = torch.cat([x, t], axis=1)
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


################


##### ENet #####

# --- Prior ---
class E1Net_Prior(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super().__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.GELU()
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        output = self.output_layer(layer6_out)
        output = self.scale * output
        return output

# --- Base ---
class E1Net(nn.Module):
    """
    use Pytorch default initialization strategy
    """
    def __init__(self, scale=1.0, normalize=1.0, neurons=64): 
        super().__init__()
        self.scale = scale
        self.normalize = normalize
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
        # self._init_weights()
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        output = self.output_layer(layer3_out)
        output = self.scale * output
        return output
    