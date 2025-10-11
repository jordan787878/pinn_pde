import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


### Exp Case1 Equin ###

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
        self.D = int(input_feature - 1)
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

class ENet_XL(nn.Module):
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
        self.D = int(input_feature - 1)
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
 
### End of Exp Case1 Equin ###
    


class PNet(nn.Module):
    def __init__(self, constants, scale=1.0, input_feature=5): 
        super().__init__()
        neurons = 50
        self.scale = scale
        self.constants = constants
        self.input_feature = input_feature
        self.hidden_layer1 = (nn.Linear(input_feature, neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self._init_weights()
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
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # kaiming normal initialization for weights
                nn.init.kaiming_normal_(m.weight)
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
     
class PNet_XL_Sphere(nn.Module):
    def __init__(
        self,
        constants,
        scale=1.0,
        input_feature=7,
        width=64,
        depth=8,
        act="silu",
        use_layernorm=False,
        fourier_t=False,
        fourier_f=8,
        input_skip_at=4,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.constants = constants
        # register as BUFFER so it's saved/loaded but not trainable
        scale_t = torch.as_tensor(scale, dtype=dtype, device=device)
        self.register_buffer("scale", scale_t, persistent=True)
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
            inputs = normalize_inputs_6d(x, t, self.constants)
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



##### These are for Equinoctial in scaled dimension #####

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
# class PNet_XL(nn.Module):
#     def __init__(
#         self,
#         constants,
#         scale=1.0,
#         input_feature=5,
#         width=64,
#         depth=8,
#         act="silu",
#         use_layernorm=False,
#         fourier_t=False,
#         fourier_f=8,
#         input_skip_at=4
#     ):
#         super().__init__()
#         self.constants = constants
#         self.scale = scale
#         self.input_feature = input_feature
#         self.width = int(width)
#         self.depth = int(depth)
#         self.use_layernorm = use_layernorm
#         self.input_skip_at = input_skip_at if (0 < input_skip_at < depth) else -1

#         # Time encoding
#         self.use_fourier_t = bool(fourier_t)
#         if self.use_fourier_t:
#             self.time_embed = FourierFeatures(num_frequencies=fourier_f, include_input=True)
#             self.t_dim = 1 + 2 * fourier_f
#         else:
#             self.time_embed = nn.Identity()
#             self.t_dim = 1

#         in_dim = int(input_feature + (self.t_dim - 1))
#         self.fc_in = nn.Linear(in_dim, self.width)

#         # Residual trunk
#         self.blocks = nn.ModuleList([
#             ResidualBlock(self.width, act=act, use_layernorm=use_layernorm)
#             for _ in range(self.depth)
#         ])

#         if self.input_skip_at > 0:
#             self.proj_after_skip = nn.Linear(self.width + in_dim, self.width)
#         else:
#             self.proj_after_skip = None

#         self.fc_out = nn.Linear(self.width, 1)
#         self.act = get_activation(act)
#         self.softplus_out = nn.Softplus(beta=1.0)

#         # **Initialize weights after defining all layers**
#         self._init_weights()

#     def _init_weights(self):
#         """
#         Custom weight initialization:
#           - Kaiming init for hidden/residual layers (good for SiLU/GELU)
#           - Xavier init for output for smoother initial pdf scaling
#           - Zero bias where appropriate
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 if m is self.fc_out:
#                     nn.init.xavier_uniform_(m.weight)
#                 else:
#                     nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                     bound = 1 / math.sqrt(fan_in)
#                     nn.init.uniform_(m.bias, -bound, bound)

#     def _build_inputs(self, x, t):
#         if self.input_feature == 5:
#             inputs = normalize_inputs(x, t, self.constants)
#         elif self.input_feature == 7:
#             inputs = normalize_inputs_6d_scaled(x, t, self.constants)
#         else:
#             raise RuntimeError(f"PINN not yet implemented for input feature: {self.input_feature}")

#         if self.use_fourier_t:
#             t_enc = self.time_embed(t if t.dim() == 2 else t.view(-1, 1))
#             inputs = torch.cat([inputs[:, :-1], t_enc], dim=1)
#         return inputs

#     def forward(self, x, t):
#         inputs = self._build_inputs(x, t)
#         h0 = self.fc_in(inputs)
#         h = h0
#         for i, block in enumerate(self.blocks):
#             h = block(h)
#             if i == self.input_skip_at and self.proj_after_skip is not None:
#                 h = torch.cat([h, inputs], dim=1)
#                 h = self.proj_after_skip(h)

#         out_hidden = h + h0
#         y = self.fc_out(out_hidden)
#         return self.softplus_out(y) * self.scale



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



##### These are for Equinoctial in scaled dimension #####

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
# class E1Net_XL(nn.Module):
#     def __init__(
#         self,
#         constants,
#         p_net=None,
#         scale=1.0,
#         normalize=1.0,
#         input_feature=7,
#         width=64,
#         depth=8,
#         act="silu",
#         use_layernorm=True,
#         fourier_t=False,
#         fourier_f=8,
#         input_skip_at=4,
#     ):
#         super().__init__()
#         self.constants = constants
#         self.scale = scale
#         self.p_net = None
#         self.normalize = normalize
#         self.input_feature = input_feature
#         self.width = int(width)
#         self.depth = int(depth)
#         self.use_layernorm = use_layernorm
#         self.input_skip_at = input_skip_at if (0 < input_skip_at < depth) else -1
#         if(p_net is not None):
#             self.p_net = p_net
#             self.p_net.eval()
#             for p in self.p_net.parameters():
#                 p.requires_grad_(False)
#         else:
#             # directly learning the error
#             self.scale = self.normalize

#         # Time encoding
#         self.use_fourier_t = bool(fourier_t)
#         if self.use_fourier_t:
#             self.time_embed = FourierFeatures(num_frequencies=fourier_f, include_input=True)
#             self.t_dim = 1 + 2 * fourier_f
#         else:
#             self.time_embed = nn.Identity()
#             self.t_dim = 1

#         in_dim = int(input_feature + (self.t_dim - 1))
#         self.fc_in = nn.Linear(in_dim, self.width)

#         # Residual trunk
#         self.blocks = nn.ModuleList([
#             ResidualBlock(self.width, act=act, use_layernorm=use_layernorm)
#             for _ in range(self.depth)
#         ])

#         if self.input_skip_at > 0:
#             self.proj_after_skip = nn.Linear(self.width + in_dim, self.width)
#         else:
#             self.proj_after_skip = None

#         self.fc_out = nn.Linear(self.width, 1)
#         self.act = get_activation(act)
#         self.softplus_out = nn.Softplus(beta=1.0)

#         # **Initialize weights after defining all layers**
#         self._init_weights()

#     def _init_weights(self):
#         """
#         Custom weight initialization:
#           - Kaiming init for hidden/residual layers (good for SiLU/GELU)
#           - Xavier init for output for smoother initial pdf scaling
#           - Zero bias where appropriate
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 if m is self.fc_out:
#                     nn.init.xavier_uniform_(m.weight)
#                 else:
#                     nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                     bound = 1 / math.sqrt(fan_in)
#                     nn.init.uniform_(m.bias, -bound, bound)

#     def _build_inputs(self, x, t):
#         if self.input_feature == 5:
#             inputs = normalize_inputs(x, t, self.constants)
#         elif self.input_feature == 7:
#             inputs = normalize_inputs_6d_scaled(x, t, self.constants)
#         else:
#             raise RuntimeError(f"PINN not yet implemented for input feature: {self.input_feature}")

#         if self.use_fourier_t:
#             t_enc = self.time_embed(t if t.dim() == 2 else t.view(-1, 1))
#             inputs = torch.cat([inputs[:, :-1], t_enc], dim=1)
#         return inputs

#     def forward(self, x, t):
#         inputs = self._build_inputs(x, t)
#         h0 = self.fc_in(inputs)
#         h = h0
#         for i, block in enumerate(self.blocks):
#             h = block(h)
#             if i == self.input_skip_at and self.proj_after_skip is not None:
#                 h = torch.cat([h, inputs], dim=1)
#                 h = self.proj_after_skip(h)

#         out_hidden = h + h0
#         y = self.fc_out(out_hidden)
#         if(self.p_net is not None):
#             p_net_out = self.p_net(x, t)
#             return y * self.scale - p_net_out
#         else:
#             return y * self.scale

#########################################################



# --- helper functions ---
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
        # print(loss_history.shape)
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


# --- PINN-GMM ---
class TimeToGMM6D(nn.Module):
    """
    Time-conditioned 6D Gaussian Mixture with K components.
    Covariance parameterization: Σ_k(t) = L_k(t) L_k(t)^T, L lower-triangular with positive diag.

    inputs
        x: (N, D)
        t: (N, 1)
    outputs
        pdf(x|t): (N, 1)

    Also provides:
        - log_prob(x,t): (N,1)
        - params(t): means (N,K,D), L (N,K,D,D), log_weights (N,K)
        - mean_and_covariance(t): mixture mean (N,D), covariance (N,D,D)
    """
    def __init__(self, constants, K=1, hidden=64, depth=2, min_diag=1e-4,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.D = 6
        self.Kc = int(K)
        self.K_tril = self.D * (self.D + 1) // 2
        self.min_diag = float(min_diag)
        self.depth = depth
        self.softplus_beta = 1.

        # ---------- normalization buffers ----------
        # x normalization from initial state stats
        x_mean0 = torch.as_tensor(np.copy(constants.N_MEAN_I), dtype=dtype)
        x_var0  = torch.as_tensor(np.copy(np.diag(constants.N_COV_I)), dtype=dtype)
        self.register_buffer("x_mean0",    x_mean0)          # (6,)
        self.register_buffer("x_var0",     x_var0)           # (6,)
        # time normalization
        T_end = float(constants.T_PRIME_SPAN[-1])
        self.register_buffer("T_end", torch.as_tensor(T_end, dtype=dtype))  # scalar

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=self.softplus_beta),
        )

        # parameter heads
        self.mean_head   = nn.Linear(hidden, self.Kc * self.D)          # (N, K*D)
        self.tri_head    = nn.Linear(hidden, self.Kc * self.K_tril)     # (N, K*K_tril)
        self.weight_head = nn.Linear(hidden, self.Kc)                   # (N, K) logits

        # indices/masks for LOWER triangle
        i, j = torch.tril_indices(self.D, self.D, 0)
        self.register_buffer("_i_tril", i, persistent=False)
        self.register_buffer("_j_tril", j, persistent=False)
        self.register_buffer("_diag_mask", (i == j), persistent=False)  # (K_tril,)
        self.register_buffer("LOG_2PI", torch.log(torch.full((), math.tau, dtype=dtype)))

        self._init_weights()
        # self._init_weights_XL()
        self.to(device=device, dtype=dtype)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _init_weights_XL(self):
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

    def _zparams_to_xparams(self, means_z, Ls_z):
        """
        means_z: (N,K,D), Ls_z: (N,K,D,D) [lower-tri in z]
        returns means_x: (N,K,D), Ls_x: (N,K,D,D) [lower-tri in x]
        """
        x_std0 = torch.sqrt(self.x_var0)                     # (D,)
        x_std0_b = x_std0.view(1, 1, self.D)                 # (1,1,D) for broadcast
        means_x = self.x_mean0.view(1, 1, self.D) + means_z * x_std0_b
        Ls_x    = Ls_z * x_std0_b.unsqueeze(-1)              # row-scale each L by std
        return means_x, Ls_x

    # ---------------- parameters ----------------
    def params(self, tau: torch.Tensor):
        """
        Inputs:
          normalized time: tau
        Returns:
          means: (N, K, D)
          Ls   : (N, K, D, D) lower-tri with positive diag (covariance Cholesky)
          logw : (N, K)       mixture log-weights (log-softmax)
        """
        p = next(self.parameters())

        # backbone
        h = self.backbone(tau)

        # means
        means = self.mean_head(h).view(-1, self.Kc, self.D)     # (N,K,D)

        # lower-tri raw -> L (covariance factor)
        raw = self.tri_head(h).view(-1, self.Kc, self.K_tril)   # (N,K,K_tril)
        raw = raw.reshape(-1, self.K_tril)                      # (N*K, K_tril)
        raw_adj = raw.clone()
        diag_raw = raw[:, self._diag_mask]                      # (N*K, D)
        diag_pos = F.softplus(diag_raw, beta=self.softplus_beta) + self.min_diag
        raw_adj[:, self._diag_mask] = diag_pos

        NK = raw.shape[0]
        L = torch.zeros(NK, self.D, self.D, dtype=p.dtype, device=p.device)
        L[:, self._i_tril, self._j_tril] = raw_adj              # (N*K,D,D)
        Ls = L.view(-1, self.Kc, self.D, self.D)                # (N,K,D,D)

        # log-weights
        logw = F.log_softmax(self.weight_head(h), dim=-1)       # (N,K)

        return means, Ls, logw

    # ---------------- densities ----------------
    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate mixture log-pdf directly in x-space:
        1) normalize time only (tau = t/T_end)
        2) get z-space params
        3) map to x-space
        4) compute log N(x | mu_x, Σ_x) and log-sum-exp over K
        Returns (N,1).
        """
        if x.ndim == 1: x = x.unsqueeze(0)
        tau = t / self.T_end                                  # (N,1)

        means_z, Ls_z, logw = self.params(tau)                # (N,K,D), (N,K,D,D), (N,K)
        means_x, Ls_x = self._zparams_to_xparams(means_z, Ls_z)

        x = x.to(dtype=means_x.dtype, device=means_x.device)   # (N,D)

        # center and solve with the x-space Cholesky
        xm = x.unsqueeze(1) - means_x                          # (N,K,D)
        N, K, D = xm.shape
        y = torch.linalg.solve_triangular(
            Ls_x.reshape(N*K, D, D), xm.reshape(N*K, D, 1),
            upper=False
        ).reshape(N, K, D, 1)

        quad = (y.squeeze(-1) ** 2).sum(-1)                    # (N,K)
        sum_log_diag = torch.log(torch.diagonal(
            Ls_x, dim1=-2, dim2=-1
        )).sum(-1)                                             # (N,K)

        log_comp = -0.5 * quad - sum_log_diag - 0.5 * D * self.LOG_2PI  # (N,K)
        logp = torch.logsumexp(logw + log_comp, dim=-1, keepdim=True)   # (N,1)
        return logp

    def pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x, t))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.pdf(x, t)

    # ---------------- mixture stats ----------------
    @torch.no_grad()
    def weights_means_covs_at(self, t: float):
        """
        Return x-space mixture parameters at time t (no batch dim):
        weights: (K,)
        means  : (K, D)
        covs   : (K, D, D)
        """
        p = next(self.parameters())
        # (1,1) tensor, normalize time
        t_tensor   = torch.as_tensor([[t]], dtype=p.dtype, device=p.device)
        tau_tensor = t_tensor / self.T_end

        # z-space params (batched)
        means_b, Ls_b, logw_b = self.params(tau_tensor)     # (1,K,D), (1,K,D,D), (1,K)

        # reuse the existing z->x transform (batched)
        means_x_b, Ls_x_b = self._zparams_to_xparams(means_b, Ls_b)  # (1,K,D), (1,K,D,D)

        # drop batch and build covariances from Cholesky
        means_x = means_x_b[0]                               # (K,D)
        Ls_x    = Ls_x_b[0]                                  # (K,D,D)
        covs_x  = Ls_x @ Ls_x.transpose(-1, -2)              # (K,D,D)

        # weights
        weights = logw_b[0].exp()                            # (K,)

        return weights, means_x, covs_x

class TimeToGMM6D_V0(nn.Module):
    """
    Time-conditioned 6D Gaussian Mixture with K components.
    Covariance parameterization: Σ_k(t) = L_k(t) L_k(t)^T, L lower-triangular with positive diag.

    inputs
        x: (N, D)
        t: (N, 1)
    outputs
        pdf(x|t): (N, 1)

    Also provides:
        - log_prob(x,t): (N,1)
        - params(t): means (N,K,D), L (N,K,D,D), log_weights (N,K)
        - mean_and_covariance(t): mixture mean (N,D), covariance (N,D,D)
    """
    def __init__(self, constants, K=1, D=6, hidden=64, depth=2, min_diag=1e-4,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.D = D
        self.Kc = int(K)
        self.K_tril = self.D * (self.D + 1) // 2
        self.min_diag = float(min_diag)
        self.depth = depth
        self.softplus_beta = 1.

        # ---------- normalization buffers ----------
        # x normalization from initial state stats
        x_mean0 = torch.as_tensor(np.copy(constants.N_MEAN_I), dtype=dtype)
        x_var0  = torch.as_tensor(np.copy(np.diag(constants.N_COV_I)), dtype=dtype)
        self.register_buffer("x_mean0",    x_mean0)          # (6,)
        self.register_buffer("x_var0",     x_var0)           # (6,)
        # time normalization
        T_end = float(constants.T_PRIME_SPAN[-1])
        self.register_buffer("T_end", torch.as_tensor(T_end, dtype=dtype))  # scalar

        # backbone
        # self.backbone = nn.Sequential(
        #     nn.Linear(1, hidden),
        #     nn.Softplus(beta=self.softplus_beta),
        #     nn.Linear(hidden, hidden),
        #     nn.Softplus(beta=self.softplus_beta),
        #     nn.Linear(hidden, hidden),
        #     nn.Softplus(beta=self.softplus_beta),
        # )
        # parameter heads
        # self.mean_head   = nn.Linear(hidden, self.Kc * self.D)          # (N, K*D)
        self.mean_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden,self.Kc * self.D),
        )

        # self.tri_head    = nn.Linear(hidden, self.Kc * self.K_tril)     # (N, K*K_tril)
        self.tri_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, self.Kc * self.K_tril),
        )

        # self.weight_head = nn.Linear(hidden, self.Kc)                   # (N, K) logits
        self.weight_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden, hidden),
            nn.Softplus(beta=self.softplus_beta),
            nn.Linear(hidden,self.Kc),
        )

        # indices/masks for LOWER triangle
        i, j = torch.tril_indices(self.D, self.D, 0)
        self.register_buffer("_i_tril", i, persistent=False)
        self.register_buffer("_j_tril", j, persistent=False)
        self.register_buffer("_diag_mask", (i == j), persistent=False)  # (K_tril,)
        self.register_buffer("LOG_2PI", torch.log(torch.full((), math.tau, dtype=dtype)))

        self._init_weights()
        self.to(device=device, dtype=dtype)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _zparams_to_xparams(self, tau, means_z, Ls_z):
        """
        means_z: (N,K,D), Ls_z: (N,K,D,D) [lower-tri in z]
        returns means_x: (N,K,D), Ls_x: (N,K,D,D) [lower-tri in x]
        """
        x_std0 = torch.sqrt(self.x_var0)                     # (D,)
        x_std0_b = x_std0.view(1, 1, self.D)                 # (1,1,D) for broadcast
        # tau_m    = tau.to(means_z.dtype).to(means_z.device).view(-1, 1, 1)    # (N,1,1)
        means_x = self.x_mean0.view(1, 1, self.D) + means_z * x_std0_b
        # means_x = means_z * x_std0_b
        # L0    = torch.diag_embed(x_std0).to(Ls_z.dtype).to(Ls_z.device)      # (D,D)
        # L0    = L0.view(1,1,self.D,self.D).expand_as(Ls_z)                    # (N,K,D,D)
        Ls_x  = Ls_z * x_std0_b.unsqueeze(-1)              # row-scale each L by std
        return means_x, Ls_x
    
    # def _zparams_to_xparams(self, means_z, Ls_z, tau):
        # """
        # means_z: (N,K,D), Ls_z: (N,K,D,D)  [lower-tri with positive diag in z]
        # tau    : (N,1) normalized time in [0,1]
        # returns:
        # means_x: (N,K,D)
        # Ls_x   : (N,K,D,D)
        # """
        # # --- broadcasts/helpers ---
        # x_std0   = torch.sqrt(self.x_var0)                  # (D,)
        # x_std0_b = x_std0.view(1, 1, self.D)                # (1,1,D)
        # tau_m    = tau.to(means_z.dtype).to(means_z.device).view(-1, 1, 1)    # (N,1,1)
        # tau_L    = tau_m.view(-1, 1, 1, 1)                                          # (N,1,1,1)
        # # --- means: linear interpolation between μ0 and dynamic μ ---
        # mu0      = self.x_mean0.view(1, 1, self.D)                                  # (1,1,D)
        # mu_dyn   = means_z * x_std0_b                                               # (N,K,D)
        # means_x  = mu0 + tau_m * mu_dyn                                             # (N,K,D)
        # L0    = torch.diag_embed(x_std0).to(Ls_z.dtype).to(Ls_z.device)      # (D,D)
        # L0    = L0.view(1,1,self.D,self.D).expand_as(Ls_z)                    # (N,K,D,D)
        # # dynamic L scaled to x-space
        # L_dyn = Ls_z * x_std0_b.unsqueeze(-1)                                 # (N,K,D,D)
        # # convex mix in L-space -> positive diagonal preserved
        # Ls_x  = (1.0 - tau_L)*L0 + tau_L * L_dyn
        # return means_x, Ls_x

    # ---------------- parameters ----------------
    def params(self, tau: torch.Tensor):
        """
        Inputs:
          normalized time: tau
        Returns:
          means: (N, K, D)
          Ls   : (N, K, D, D) lower-tri with positive diag (covariance Cholesky)
          logw : (N, K)       mixture log-weights (log-softmax)
        """
        p = next(self.parameters())

        # backbone
        # h = self.backbone(tau)
        h = tau

        # means
        means = self.mean_head(h).view(-1, self.Kc, self.D)     # (N,K,D)

        # lower-tri raw -> L (covariance factor)
        raw = self.tri_head(h).view(-1, self.Kc, self.K_tril)   # (N,K,K_tril)
        raw = raw.reshape(-1, self.K_tril)                      # (N*K, K_tril)
        raw_adj = raw.clone()
        diag_raw = raw[:, self._diag_mask]                      # (N*K, D)
        diag_pos = F.softplus(diag_raw, beta=self.softplus_beta) + self.min_diag
        raw_adj[:, self._diag_mask] = diag_pos

        NK = raw.shape[0]
        L = torch.zeros(NK, self.D, self.D, dtype=p.dtype, device=p.device)
        L[:, self._i_tril, self._j_tril] = raw_adj              # (N*K,D,D)
        Ls = L.view(-1, self.Kc, self.D, self.D)                # (N,K,D,D)

        # log-weights
        # (default)
        # logw = F.log_softmax(self.weight_head(tau), dim=-1)       # (N,K)
        
        # (try numerical stable parameterization of GMM log(weights)
        # turns out it is the v that turns NaN...
        if torch.isnan(h).any():
            print("h Nan")
        v = self.weight_head(h)                                   # (N, K)
        if torch.isnan(v).any():
            print("v Nan")
        temp = getattr(self, "softmax_temp_w", 1.0)
        logw = F.log_softmax(v / max(temp, 1e-6), dim=-1)         # (N,K)
        w = logw.exp()                                            # (N,K)
        alpha_floor = 0.10     # try 0.05–0.20; tune
        Kc = w.size(-1)
        w = (1.0 - alpha_floor) * w + (alpha_floor / Kc)          # min prob = α/K
        eps = torch.finfo(w.dtype).tiny
        logw = torch.log(w.clamp_min(eps))  

        return means, Ls, logw

    # ---------------- densities ----------------
    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate mixture log-pdf directly in x-space:
        1) normalize time only (tau = t/T_end)
        2) get z-space params
        3) map to x-space
        4) compute log N(x | mu_x, Σ_x) and log-sum-exp over K
        Returns (N,1).
        """
        if x.ndim == 1: x = x.unsqueeze(0)
        tau = t / self.T_end                                  # (N,1)

        means_z, Ls_z, logw = self.params(tau)                # (N,K,D), (N,K,D,D), (N,K)
        means_x, Ls_x = self._zparams_to_xparams(tau, means_z, Ls_z)

        x = x.to(dtype=means_x.dtype, device=means_x.device)   # (N,D)

        # center and solve with the x-space Cholesky
        xm = x.unsqueeze(1) - means_x                          # (N,K,D)
        N, K, D = xm.shape
        y = torch.linalg.solve_triangular(
            Ls_x.reshape(N*K, D, D), xm.reshape(N*K, D, 1),
            upper=False
        ).reshape(N, K, D, 1)

        quad = (y.squeeze(-1) ** 2).sum(-1)                    # (N,K)
        sum_log_diag = torch.log(torch.diagonal(
            Ls_x, dim1=-2, dim2=-1
        )).sum(-1)                                             # (N,K)

        log_comp = -0.5 * quad - sum_log_diag - 0.5 * D * self.LOG_2PI  # (N,K)
        logp = torch.logsumexp(logw + log_comp, dim=-1, keepdim=True)   # (N,1)
        return logp

    def pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x, t))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.pdf(x, t)

    # ---------------- mixture stats ----------------
    @torch.no_grad()
    def weights_means_covs_at(self, t: float):
        """
        Return x-space mixture parameters at time t (no batch dim):
        weights: (K,)
        means  : (K, D)
        covs   : (K, D, D)
        """
        p = next(self.parameters())
        # (1,1) tensor, normalize time
        t_tensor   = torch.as_tensor([[t]], dtype=p.dtype, device=p.device)
        tau_tensor = t_tensor / self.T_end

        # z-space params (batched)
        means_b, Ls_b, logw_b = self.params(tau_tensor)     # (1,K,D), (1,K,D,D), (1,K)

        # reuse the existing z->x transform (batched)
        means_x_b, Ls_x_b = self._zparams_to_xparams(tau_tensor, means_b, Ls_b)  # (1,K,D), (1,K,D,D)

        # drop batch and build covariances from Cholesky
        means_x = means_x_b[0]                               # (K,D)
        Ls_x    = Ls_x_b[0]                                  # (K,D,D)
        covs_x  = Ls_x @ Ls_x.transpose(-1, -2)              # (K,D,D)

        # weights
        weights = logw_b[0].exp()                            # (K,)

        return weights, means_x, covs_x
