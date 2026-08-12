import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0, use_layer_norm=False, new_init=False,
                 activation=nn.ReLU, activation_last_layer=False):
        """
            the implementation of multi layer perceptrons (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = []
        self.use_layer_norm = use_layer_norm
        if num_layers == 1:
            print("num_layers=1")
            if activation_last_layer:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(activation())
            else:
                self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            print("num_layers=more")
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(activation())
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=True))
                # self.layers.append(DyT(hidden_dim))

            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(activation())
                if use_layer_norm:
                    # self.layers.append(DyT(hidden_dim))
                    self.layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=True))
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
            # self.layers.append(nn.Linear(hidden_dim, output_dim))
            if activation_last_layer:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(activation())
            else:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*self.layers)
        # self.init_weights_orthogonal(output_gain=3e-3)
        # self.apply(self._initialize_weights)

        # if new_init:
        #     # for layer in self.net[::2]:
        #     #     torch.nn.init.constant_(layer.bias, 0.1)
        #     torch.nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        #     torch.nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

    def init_weights_orthogonal(self, hidden_gain=np.sqrt(2), output_gain=0.01):
        """
        Applies orthogonal initialization to the linear layers.

        :param hidden_gain: Gain factor for orthogonal initialization of hidden layers.
                           sqrt(2) is recommended for ReLU activations.
        :param output_gain: Gain factor for orthogonal initialization of the output layer.
                           A small value like 0.01 is recommended for actor networks
                           in RL to keep initial action probabilities close to uniform.
        """
        print("Applying orthogonal initialization...")
        first_linear = True  # Flag for the very first layer (if needed, currently not used differently)
        last_linear_layer_idx = -1

        # Find the index of the last linear layer
        for i, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, nn.Linear):
                last_linear_layer_idx = i
                break

        if last_linear_layer_idx == -1 and self.num_layers > 0:
            # This case handles num_layers=1 where the linear layer is directly in self.layers[0]
            # and the loop above might not find it if self.layers was not built correctly
            # Let's search in self.net instead for robustness
            temp_last_linear_idx = -1
            for i, module in enumerate(self.net):
                if isinstance(module, nn.Linear):
                    temp_last_linear_idx = i
            if temp_last_linear_idx != -1:
                last_linear_layer_idx = temp_last_linear_idx  # Found it within the sequential net
            else:
                print("Warning: No linear layers found for orthogonal initialization.")
                return

        # Iterate through the modules in the sequential net
        # This is often more reliable than iterating through the self.layers list
        # if the structure was modified or complex.
        linear_layer_count = 0
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                linear_layer_count += 1
                # Check if this linear layer is the last one in the sequence
                # This check is slightly indirect but works for Sequential:
                # We assume the last nn.Linear encountered IS the output layer.
                # A more robust way is to check against the layer found earlier by index,
                # but comparing module objects directly can be tricky if Sequential wraps them.
                # Let's stick to the simpler approach: assume the last Linear IS the output.
                is_output_layer = True
                for subsequent_module in list(self.net.children())[i + 1:]:
                    if isinstance(subsequent_module, nn.Linear):
                        is_output_layer = False
                        break

                if is_output_layer:
                    print(f"Initializing LAST linear layer (output) {module} with gain {output_gain}")
                    nn.init.orthogonal_(module.weight, gain=output_gain)
                else:
                    # Determine the appropriate gain for hidden layers
                    # Defaulting to ReLU gain, but could be made more sophisticated
                    # if activation type varies significantly.
                    current_hidden_gain = hidden_gain
                    # Example: If using Tanh, maybe use gain=1.0 or 5/3
                    # if isinstance(self.activation, nn.Tanh):
                    #    current_hidden_gain = 1.0 # Or 5/3

                    print(f"Initializing HIDDEN linear layer {module} with gain {current_hidden_gain}")
                    nn.init.orthogonal_(module.weight, gain=current_hidden_gain)

                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if linear_layer_count == 0 and self.num_layers >= 1:
            print("Warning: Orthogonal init requested, but no nn.Linear layers found in self.net.")


    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0, use_layer_norm=False, new_init=False,
                 activation=nn.ReLU):
        """
            the implementation of multi layer perceptrons (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """

        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = []
        self.use_layer_norm = use_layer_norm
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(activation())

            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(activation())

                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*self.layers)
        # self.init_weights_orthogonal(output_gain=3e-3)

    def init_weights_orthogonal(self, hidden_gain=np.sqrt(2), output_gain=0.01):
        """
        Applies orthogonal initialization to the linear layers.

        :param hidden_gain: Gain factor for orthogonal initialization of hidden layers.
                           sqrt(2) is recommended for ReLU activations.
        :param output_gain: Gain factor for orthogonal initialization of the output layer.
                           A small value like 0.01 is recommended for actor networks
                           in RL to keep initial action probabilities close to uniform.
        """
        print("Applying orthogonal initialization...")
        first_linear = True  # Flag for the very first layer (if needed, currently not used differently)
        last_linear_layer_idx = -1

        # Find the index of the last linear layer
        for i, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, nn.Linear):
                last_linear_layer_idx = i
                break

        if last_linear_layer_idx == -1 and self.num_layers > 0:
            # This case handles num_layers=1 where the linear layer is directly in self.layers[0]
            # and the loop above might not find it if self.layers was not built correctly
            # Let's search in self.net instead for robustness
            temp_last_linear_idx = -1
            for i, module in enumerate(self.net):
                if isinstance(module, nn.Linear):
                    temp_last_linear_idx = i
            if temp_last_linear_idx != -1:
                last_linear_layer_idx = temp_last_linear_idx  # Found it within the sequential net
            else:
                print("Warning: No linear layers found for orthogonal initialization.")
                return

        # Iterate through the modules in the sequential net
        # This is often more reliable than iterating through the self.layers list
        # if the structure was modified or complex.
        linear_layer_count = 0
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                linear_layer_count += 1
                # Check if this linear layer is the last one in the sequence
                # This check is slightly indirect but works for Sequential:
                # We assume the last nn.Linear encountered IS the output layer.
                # A more robust way is to check against the layer found earlier by index,
                # but comparing module objects directly can be tricky if Sequential wraps them.
                # Let's stick to the simpler approach: assume the last Linear IS the output.
                is_output_layer = True
                for subsequent_module in list(self.net.children())[i + 1:]:
                    if isinstance(subsequent_module, nn.Linear):
                        is_output_layer = False
                        break

                if is_output_layer:
                    print(f"Initializing LAST linear layer (output) {module} with gain {output_gain}")
                    nn.init.orthogonal_(module.weight, gain=output_gain)
                else:
                    # Determine the appropriate gain for hidden layers
                    # Defaulting to ReLU gain, but could be made more sophisticated
                    # if activation type varies significantly.
                    current_hidden_gain = hidden_gain
                    # Example: If using Tanh, maybe use gain=1.0 or 5/3
                    # if isinstance(self.activation, nn.Tanh):
                    #    current_hidden_gain = 1.0 # Or 5/3

                    print(f"Initializing HIDDEN linear layer {module} with gain {current_hidden_gain}")
                    nn.init.orthogonal_(module.weight, gain=current_hidden_gain)

                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if linear_layer_count == 0 and self.num_layers >= 1:
            print("Warning: Orthogonal init requested, but no nn.Linear layers found in self.net.")

    def forward(self, x):
        return self.net(x)


# class Actor(nn.Module):
#     def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0):
#         """
#             the implementation of Actor network (refer to L2D)
#         :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
#                             If num_layers=1, this reduces to linear model.
#         :param input_dim: dimensionality of input features
#         :param hidden_dim: dimensionality of hidden units at ALL layers
#         :param output_dim:  number of classes for prediction
#         """
#         super(Actor, self).__init__()
#
#         self.linear_or_not = True  # default is linear model
#         self.num_layers = num_layers
#
#         # self.activative = torch.tanh
#         self.activative = torch.relu
#         self.dropout = nn.Dropout(dropout)
#
#         if num_layers < 1:
#             raise ValueError("number of layers should be positive!")
#         elif num_layers == 1:
#             # Linear model
#             self.linear = nn.Linear(input_dim, output_dim)
#         else:
#             # Multi-layer model
#             self.linear_or_not = False
#             self.linears = torch.nn.ModuleList()
#
#             self.linears.append(nn.Linear(input_dim, hidden_dim))
#             for layer in range(num_layers - 2):
#                 self.linears.append(nn.Linear(hidden_dim, hidden_dim))
#             self.linears.append(nn.Linear(hidden_dim, output_dim))
#
#     def forward(self, x):
#         if self.linear_or_not:
#             # If linear model
#             return self.linear(x)
#         else:
#             # If MLP
#             h = x
#             for layer in range(self.num_layers - 1):
#                 h = self.dropout(self.activative((self.linears[layer](h))))
#             return self.linears[self.num_layers - 1](h)




class QRDQN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(QRDQN, self).__init__()
        self.q1 = MLP(num_layers, input_dim, hidden_dim, output_dim, dropout=dropout)
        self.q2 = MLP(num_layers, input_dim, hidden_dim, output_dim, dropout=dropout)

    def forward(self, x, x_global, mask, sz_b):
        return self.q1(x), self.q2(x)


class QRDQN_single(nn.Module):
    def __init__(self, num_layers, input_dim_q, hidden_dim, output_dim, dropout=0, use_layer_norm=True):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(QRDQN_single, self).__init__()
        self.q = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout, use_layer_norm=use_layer_norm,
                     new_init=True)
        self.output_dim = output_dim
        self.num_quantiles = output_dim


    def forward(self, x, x_global, mask, sz_b):
        q = self.q(x)
        q[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')

        return q


class IQN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, num_cos=64, dropout=0):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(IQN, self).__init__()
        self.num_cos = num_cos
        self.emb_dim = hidden_dim
        self.cos_net1 = CosineEmbeddingNetwork(num_cos, input_dim)
        self.cos_net2 = CosineEmbeddingNetwork(num_cos, input_dim)
        self.q1 = MLP(num_layers, input_dim, hidden_dim, output_dim, dropout=dropout)
        self.q2 = MLP(num_layers, input_dim, hidden_dim, output_dim, dropout=dropout)


    def forward(self, x, x_global, mask, sz_b, tau1, tau2):
        tau_emb1 = self.cos_net1(tau1)
        print(tau_emb1.shape)
        exit()
        return self.q1(x), self.q2(x)




# class QRDQN_advantage(nn.Module):
#     def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0):
#         """
#             the implementation of Critic network (refer to L2D)
#             :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
#                                     If num_layers=1, this reduces to linear model.
#             :param input_dim: dimensionality of input features
#             :param hidden_dim: dimensionality of hidden units at ALL layers
#             :param output_dim:  number of classes for prediction
#         """
#         super(QRDQN_advantage, self).__init__()
#         # self.q1 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
#         self.v1 = MLP(num_layers, input_v, hidden_dim, output_dim * 50, dropout=dropout)
#         # self.q2 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
#         self.v2 = MLP(num_layers, input_v, hidden_dim, output_dim * 50, dropout=dropout)
#         self.output_dim = output_dim
#
#     def forward(self, x, x_global, mask, sz_b):
#         # print(x.shape, x_global.shape)
#         # print(self.q1)
#         # print(self.v1)
#         # exit()
#         # q1 = self.q1(x)
#         v1 = self.v1(x_global)
#         # q2 = self.q2(x)
#         v2 = self.v2(x_global)
#         v1 = v1.view(-1, 50, self.output_dim)
#         v2 = v2.view(-1, 50, self.output_dim)
#
#
#         #
#         # q1 = q1 - v1
#         # q2 = q2 - v2
#
#
#         return v1, v2

class QRDQN_advantage(nn.Module):
    def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(QRDQN_advantage, self).__init__()
        self.q1 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
        self.v1 = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout)
        self.q2 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
        self.v2 = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout)
        self.output_dim = output_dim
        self.num_quantiles = output_dim

    def forward(self, x, x_global, mask, sz_b):
        q1 = self.q1(x)
        v1 = self.v1(x_global).unsqueeze(1)
        q2 = self.q2(x)
        v2 = self.v2(x_global).unsqueeze(1)
        q1[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')
        q2[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')

        allow_actions = (~mask).sum(dim=-1).sum(dim=-1).view(-1, 1, 1)

        q1_masked = q1.clone()
        q2_masked = q2.clone()
        q1_masked[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.output_dim)] = 0
        q2_masked[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.output_dim)] = 0

        q1_mean = q1_masked.sum(dim=1, keepdim=True) / allow_actions
        q2_mean = q2_masked.sum(dim=1, keepdim=True) / allow_actions
        q1 = v1 + (q1 - q1_mean)
        q2 = v2 + (q2 - q2_mean)

        return q1, q2


# class QRDQN_advantage_single(nn.Module):
#     def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0, use_layer_norm=True):
#         """
#             the implementation of Critic network (refer to L2D)
#             :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
#                                     If num_layers=1, this reduces to linear model.
#             :param input_dim: dimensionality of input features
#             :param hidden_dim: dimensionality of hidden units at ALL layers
#             :param output_dim:  number of classes for prediction
#         """
#         super(QRDQN_advantage_single, self).__init__()
#         # self.q = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout, use_layer_norm=use_layer_norm,
#         #              new_init=True)
#         # self.v = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout, use_layer_norm=use_layer_norm,
#         #              new_init=True)
#         self.hidden_dim = hidden_dim
#         self.shared = MLP(num_layers - 1, input_dim_q, hidden_dim, output_dim, dropout=dropout,
#                           use_layer_norm=use_layer_norm, new_init=True, activation_last_layer=True)
#         print("shared", self.shared)
#         self.value_head = nn.Linear(hidden_dim, output_dim)
#         self.adv_head = nn.Linear(hidden_dim, output_dim)
#
#         self.output_dim = output_dim
#         self.num_quantiles = output_dim
#
#     def forward(self, x, x_global, mask, sz_b):
#         out = self.shared(x)
#         # out[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.hidden_dim)] = 0
#         out_masked = out.masked_fill(mask.reshape(sz_b, -1, 1).expand(-1, -1, self.hidden_dim), 0)
#         out_mean = nonzero_averaging(out_masked)
#
#         allow_actions = (~mask).sum(dim=-1).sum(dim=-1).view(-1, 1, 1)
#         # out_mean = out.sum(dim=1, keepdim=True) / allow_actions
#         v_out = self.value_head(out_mean).unsqueeze(1)
#         adv_out = self.adv_head(out)
#         adv_out[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.hidden_dim)] = 0
#         adv_mean = adv_out.sum(dim=1, keepdim=True) / allow_actions
#         # adv_mean = nonzero_averaging(adv_out).unsqueeze(1)
#
#         q = v_out + (adv_out - adv_mean)
#
#         q[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')
#
#
#         return q


class QRDQN_advantage_single(nn.Module):
    def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0, use_layer_norm=True):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(QRDQN_advantage_single, self).__init__()
        self.q = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout, use_layer_norm=use_layer_norm,
                     new_init=True)
        self.v = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout, use_layer_norm=use_layer_norm,
                     new_init=True)

        self.output_dim = output_dim
        self.num_quantiles = output_dim

    def forward(self, x, x_global, mask, sz_b):
        q = self.q(x)
        v = self.v(x_global).unsqueeze(1)

        q[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')

        allow_actions = (~mask).sum(dim=-1).sum(dim=-1).view(-1, 1, 1)

        q_masked = q.clone()
        q_masked[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.output_dim)] = 0

        q_mean = q_masked.sum(dim=1, keepdim=True) / allow_actions
        q = v + (q - q_mean)

        return q


class DQN_advantage(nn.Module):
    def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0):
        """
            the implementation of Critic network (refer to L2D)
            :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                                    If num_layers=1, this reduces to linear model.
            :param input_dim: dimensionality of input features
            :param hidden_dim: dimensionality of hidden units at ALL layers
            :param output_dim:  number of classes for prediction
        """
        super(DQN_advantage, self).__init__()
        self.q1 = MLP(num_layers, input_dim_q, hidden_dim, 1, dropout=dropout)
        self.v1 = MLP(num_layers, input_v, hidden_dim, 1, dropout=dropout)
        self.q2 = MLP(num_layers, input_dim_q, hidden_dim, 1, dropout=dropout)
        self.v2 = MLP(num_layers, input_v, hidden_dim, 1, dropout=dropout)
        self.output_dim = output_dim
        self.num_quantiles = output_dim

    def forward(self, x, x_global, mask, sz_b):

        q1 = self.q1(x)
        v1 = self.v1(x_global).unsqueeze(1)
        q2 = self.q2(x)
        v2 = self.v2(x_global).unsqueeze(1)

        q1[mask.reshape(sz_b, -1)] = float('-inf')
        q2[mask.reshape(sz_b, -1)] = float('-inf')

        allow_actions = (~mask).sum(dim=-1).sum(dim=-1).view(-1, 1, 1)
        # exit()
        # print(q1.shape, v1.shape, q2.shape, v2.shape)
        # print((q1 - v1).shape, (q2 - v2).shape)
        # exit()
        q1_masked = q1.clone()
        q2_masked = q2.clone()
        q1_masked[mask.reshape(sz_b, -1)] = 0
        q2_masked[mask.reshape(sz_b, -1)] = 0
        # mask_sum = (~mask.reshape(sz_b, -1)).sum(dim=1, keepdim=True)
        #

        q1_mean = q1_masked.sum(dim=1, keepdim=True) / allow_actions


        q2_mean = q2_masked.sum(dim=1, keepdim=True) / allow_actions
        q1 = v1 + (q1 - q1_mean)
        q2 = v2 + (q2 - q2_mean)


        return q1, q2



# class QRDQN_advantage(nn.Module):
#     def __init__(self, num_layers, input_dim_q, input_v, hidden_dim, output_dim, dropout=0):
#         """
#             the implementation of Critic network (refer to L2D)
#             :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
#                                     If num_layers=1, this reduces to linear model.
#             :param input_dim: dimensionality of input features
#             :param hidden_dim: dimensionality of hidden units at ALL layers
#             :param output_dim:  number of classes for prediction
#         """
#         super(QRDQN_advantage, self).__init__()
#         self.q1 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
#         self.v1 = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout)
#         self.q2 = MLP(num_layers, input_dim_q, hidden_dim, output_dim, dropout=dropout)
#         self.v2 = MLP(num_layers, input_v, hidden_dim, output_dim, dropout=dropout)
#         self.output_dim = output_dim
#
#     def forward(self, x, x_global, mask, sz_b):
#         # print(x.shape, x_global.shape)
#         # print(self.q1)
#         # print(self.v1)
#         # exit()
#         q1 = self.q1(x)
#         v1 = self.v1(x_global).unsqueeze(1)
#         q2 = self.q2(x)
#         v2 = self.v2(x_global).unsqueeze(1)
#         # print(q1.shape, v1.shape, q2.shape, v2.shape)
#         # print((q1 - v1).shape, (q2 - v2).shape)
#         # exit()
#         # q1_masked = q1.clone()
#         # q2_masked = q2.clone()
#         # q1_masked[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.output_dim)] = 0
#         # q2_masked[mask.reshape(sz_b, -1, 1).expand(-1, -1, self.output_dim)] = 0
#         # mask_sum = (~mask.reshape(sz_b, -1)).sum(dim=1, keepdim=True)
#         #
#         #
#         # q1_mean = q1_masked.sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
#         # q2_mean = q2_masked.sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
#
#
#         q1 = q1 - v1
#         q2 = q2 - v2
#
#
#         # print(q1.shape, v1.shape, q2.shape, v2.shape)
#         # exit()
#         return q1, q2


class Critic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_layer_norm=False):
        """
            the implementation of Critic network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Critic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers

        self.activative = torch.relu

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()


            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms = torch.nn.ModuleList()
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
                for layer in range(num_layers - 2):
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                if self.use_layer_norm:
                    h = self.activative(self.layer_norms[layer](self.linears[layer](h)))
                else:
                    h = self.activative((self.linears[layer](h)))


            return self.linears[self.num_layers - 1](h)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cos: int = 64, emb_dim: int = 64):
        super(CosineEmbeddingNetwork, self).__init__()
        self.num_cos = num_cos
        self.emb_dim = emb_dim
        self.net = nn.Linear(num_cos, emb_dim)

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]
        i_pi = torch.pi * torch.arange(
            start=1, end=self.num_cos + 1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cos)
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cos)
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.emb_dim)
        return tau_embeddings
