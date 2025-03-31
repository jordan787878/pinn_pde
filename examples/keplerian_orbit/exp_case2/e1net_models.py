import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import Case2_4D_Constants
constants = Case2_4D_Constants()


# helper function
def normalize_inputs(x, t, constants):
    _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
    _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
    _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
    _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
    _t  = t/(constants.TF/constants.T) #[0,1]
    inputs = torch.cat([_x1, _x2, _x3, _x4, _t],axis=1)
    return inputs


# # [Skip connection] # #
class E1Net(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        super(E1Net, self).__init__()
        neurons = 50
        self.scale = scale
        self.hidden_layer1 = (nn.Linear(5,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.hidden_layer8 = (nn.Linear(neurons,neurons))
        self.hidden_layer9 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))

    def forward(self, x, t):
        inputs = normalize_inputs(x, t, constants)
        layer1_out = ((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = ((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = ((self.hidden_layer5(layer4_out)))
        layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
        layer7_out = ((self.hidden_layer7(layer6_out)))
        layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
        layer9_out = ((self.hidden_layer9(layer8_out)))
        output = (self.output_layer(layer9_out + layer1_out)) * 0.010610391
        return output


# # [Skip connection] # #
# class E1Net(nn.Module):
#     global constants
#     def __init__(self, scale=1.0): 
#         neurons = 64
#         self.scale = scale
#         super(E1Net, self).__init__()
#         self.hidden_layer1 = (nn.Linear(5,neurons))
#         self.hidden_layer2 = (nn.Linear(neurons,neurons))
#         self.hidden_layer3 = (nn.Linear(neurons,neurons))
#         self.hidden_layer4 = (nn.Linear(neurons,neurons))
#         self.hidden_layer5 = (nn.Linear(neurons,neurons))
#         self.hidden_layer6 = (nn.Linear(neurons,neurons))
#         self.hidden_layer7 = (nn.Linear(neurons,neurons))
#         self.hidden_layer8 = (nn.Linear(neurons,neurons))
#         self.hidden_layer9 = (nn.Linear(neurons,neurons))
#         self.output_layer =  (nn.Linear(neurons,1))
#     def forward(self, x, t):
#         inputs = normalize_inputs(x, t, constants)
#         layer1_out = F.softplus((self.hidden_layer1(inputs)))
#         layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
#         layer3_out = F.softplus((self.hidden_layer3(layer2_out))) + layer1_out
#         layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
#         layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
#         layer6_out = F.softplus((self.hidden_layer6(layer5_out))) + layer3_out
#         layer7_out = F.softplus((self.hidden_layer7(layer6_out))) 
#         layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
#         layer9_out = F.softplus((self.hidden_layer9(layer8_out))) + layer7_out
#         output = (self.output_layer(layer9_out))
#         return output


# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, out_features, activation, use_skip):
#         """
#         Args:
#             in_features: number of input neurons.
#             out_features: number of output neurons.
#             activation: activation function (callable).
#             use_skip: bool indicating whether to add a skip connection.
#         """
#         super(ResidualBlock, self).__init__()
#         self.use_skip = use_skip
#         self.activation = activation
#         self.linear = nn.Linear(in_features, out_features)
#         # If using a skip connection but dimensions differ, add a projection.
#         if use_skip and in_features != out_features:
#             self.shortcut = nn.Linear(in_features, out_features)
#         else:
#             self.shortcut = None
#     def forward(self, x):
#         out = self.activation(self.linear(x))
#         if self.use_skip:
#             shortcut = x if self.shortcut is None else self.shortcut(x)
#             # Residual addition then activation.
#             return self.activation(shortcut + out)
#         else:
#             return out
# class E1Net(nn.Module):
#     def __init__(self, scale=1.0, num_layers=3, num_neurons=64, activation=F.sigmoid, skip_layers=None):
#         global constants
#         """
#         Args:
#             scale: scaling factor for the final output.
#             num_layers: number of hidden layers (blocks) to stack.
#             num_neurons: number of neurons per hidden layer.
#             activation: activation function (callable) to use.
#             skip_layers: list of integers specifying the (1-indexed) layers at which to use skip connections.
#                          For example, skip_layers=[2, 3] means that after the 2nd and 3rd hidden layers, a residual addition is performed.
#                          If None, no residual skip connections are used.
#         """
#         super(E1Net, self).__init__()
#         self.scale = scale
#         self.activation = activation
#         # If not provided, set skip_layers to empty list.
#         if skip_layers is None:
#             skip_layers = []
#         self.skip_layers = skip_layers

#         # The input layer maps the 5-dimensional normalized input to num_neurons.
#         self.input_layer = nn.Linear(5, num_neurons)

#         # Create a list of hidden blocks.
#         self.blocks = nn.ModuleList()
#         for i in range(1, num_layers):
#             # Decide if a skip connection should be used at this block.
#             use_skip = (i in self.skip_layers)
#             self.blocks.append(ResidualBlock(num_neurons, num_neurons, activation, use_skip))
#         # The last hidden layer can also be a block.
#         # For simplicity, we use a block without skip (or you can add one as well).
#         self.last_block = ResidualBlock(num_neurons, num_neurons, activation, use_skip=False)
#         # Output layer mapping to a scalar.
#         self.output_layer = nn.Linear(num_neurons, 1)
#     def forward(self, x, t):
#         # Normalize and concatenate inputs.
#         inputs = normalize_inputs(x, t, constants)
#         out = self.activation(self.input_layer(inputs))
#         # Pass through the hidden blocks.
#         for i, block in enumerate(self.blocks, start=1):
#             out = block(out)
#         out = self.last_block(out)
#         output = self.output_layer(out)
#         return self.scale * output
    

# class E1Net(nn.Module):
#     global constants
#     def __init__(self, input_dim=5, num_frequencies=50, hidden_dim=64, num_hidden_layers=2, scale=1.0):
#         """
#         Args:
#             input_dim: Dimensionality of the input. Here 4 for x and 1 for t, so total 5.
#             num_frequencies: Number of Fourier features (i.e. number of basis functions).
#             hidden_dim: Hidden dimension for the combining MLP.
#             num_hidden_layers: Number of hidden layers in the MLP.
#         """
#         super(E1Net, self).__init__()
#         self.num_frequencies = num_frequencies
#         self.scale = scale
#         # Learnable frequencies and phases for the Fourier features.
#         # Alternatively, you could fix these using a deterministic scheme.
#         self.freq = nn.Parameter(torch.randn(num_frequencies, input_dim))
#         self.phase = nn.Parameter(torch.randn(num_frequencies))       
#         # Build a small MLP that maps from the Fourier feature space to the output.
#         layers = []
#         layers.append(nn.Linear(num_frequencies, hidden_dim))
#         layers.append(nn.Tanh())
#         for _ in range(num_hidden_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_dim, 1))
#         self.mlp = nn.Sequential(*layers)
#     def forward(self, x, t):
#         """
#         Args:
#             x: Tensor of shape (batch_size, 4) representing the spatial variables [x1,x2,x3,x4].
#             t: Tensor of shape (batch_size, 1) representing time.            
#         Returns:
#             A tensor of shape (batch_size, 1) approximating p(x,t).
#         """
#         # Concatenate the spatial and temporal inputs into one tensor of shape (batch_size, 5).
#         inp = normalize_inputs(x, t, constants)
#         # Compute the Fourier features.
#         # We compute: cos(2π (inp dot freq^T) + phase)
#         # Resulting shape: (batch_size, num_frequencies)
#         projection = 2 * torch.pi * torch.matmul(inp, self.freq.t()) + self.phase
#         fourier_features = torch.cos(projection)
#         # Optionally, one might include sin terms as well:
#         # sine_features = torch.sin(projection)
#         # fourier_features = torch.cat([fourier_features, sine_features], dim=1)
#         # Pass the Fourier features through the MLP to get the final output.
#         output = self.mlp(fourier_features) * self.scale
#         return output
    

# # [Pure Fourier] # #
# class E1Net(nn.Module):
#     def __init__(self, input_dim=5, num_terms=20, scale = 1.0):
#         global constants
#         """
#         Args:
#             input_dim: Dimensionality of the input (here 4 spatial + 1 temporal = 5).
#             num_terms: Number of Fourier basis terms (N in the sum).
#         """
#         super(E1Net, self).__init__()
#         self.num_terms = num_terms
#         self.scale = scale
#         # Learnable frequency matrix: each row is w_j in ℝ^(input_dim)
#         self.freq = nn.Parameter(torch.randn(num_terms, input_dim))
#         # Learnable phases for each Fourier term: b_j
#         self.phase = nn.Parameter(torch.randn(num_terms))
#         # Learnable combination weights: a_j
#         self.weight = nn.Parameter(torch.randn(num_terms))
#     def forward(self, x, t):
#         """
#         Args:
#             x: Tensor of shape (batch_size, 4) for the spatial inputs.
#             t: Tensor of shape (batch_size, 1) for the time input.
#         Returns:
#             Tensor of shape (batch_size, 1) approximating p(x,t).
#         """
#         # Concatenate spatial and temporal inputs into a (batch_size, 5) tensor.
#         z = normalize_inputs(x, t, constants)
#         # Compute the projection for each term: shape (batch_size, num_terms)
#         projection = 2 * torch.pi * torch.matmul(z, self.freq.t()) + self.phase
#         # Evaluate the cosine (Fourier basis functions)
#         fourier_features = torch.cos(projection)
#         # Compute the weighted sum: for each sample, sum_{j=1}^{num_terms} a_j * cos(...)
#         output = torch.matmul(fourier_features, self.weight.unsqueeze(1)) * self.scale
#         return output
    

# class E1Net(nn.Module):
#     global constants
#     def __init__(self, input_dim=5, num_terms=32, hidden_dim=16, scale = 1.0):
#         """
#         Args:
#             input_dim: Dimensionality of the full input z (4 spatial + 1 temporal = 5).
#             num_terms: Number of Fourier basis functions (N).
#             hidden_dim: Hidden dimension for the MLP that computes weights from t.
#         """
#         super(E1Net, self).__init__()
#         self.scale = scale
#         self.num_terms = num_terms
#         # Learnable frequency matrix: each row is a frequency vector w_j in R^5.
#         self.freq = nn.Parameter(torch.randn(num_terms, input_dim))
#         # Learnable phases for each Fourier term: b_j.
#         self.phase = nn.Parameter(torch.randn(num_terms))
        
#         # MLP to compute the time-dependent coefficients a(t).
#         # It takes a scalar t and outputs a vector of length num_terms.
#         self.mlp_t = nn.Sequential(
#             nn.Linear(1, hidden_dim),
#             nn.Softplus(),
#             nn.Linear(hidden_dim, num_terms)
#         )
#     def forward(self, x, t):
#         """
#         Args:
#             x: Tensor of shape (batch_size, 4) for the spatial variables [x1,x2,x3,x4].
#             t: Tensor of shape (batch_size, 1) for time.
            
#         Returns:
#             Tensor of shape (batch_size, 1) approximating p(x,t).
#         """
#         # Concatenate x and t to form the full input vector z (shape: batch_size x 5).
#         z = normalize_inputs(x, t, constants)
#         # Compute the Fourier features for each term:
#         # projection = 2π (z dot w_j) + b_j for j=1...num_terms.
#         projection = 2 * torch.pi * torch.matmul(z, self.freq.t()) + self.phase
#         fourier_features = torch.cos(projection)  # shape: (batch_size, num_terms)
#         # Compute the time-dependent weights a(t) via the MLP.
#         a_t = self.mlp_t(t)  # shape: (batch_size, num_terms)
#         # Weighted sum of the Fourier features.
#         output = torch.sum(a_t * fourier_features, dim=1, keepdim=True) * self.scale
#         return output