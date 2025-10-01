import torch
import torch.nn as nn
import numpy as np
import sys

class InstanceWiseParamNet(nn.Module):
    def __init__(self, input_feature_dim: int, theta_dim: int):
        super().__init__()
        self.theta_dim = theta_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.theta_dim)  
        )
        self.initialize_final_layer()

    def initialize_final_layer(self):
        final_layer = self.mlp[-1]  

        if final_layer.bias is not None:
            with torch.no_grad():
                final_layer.bias[0: self.theta_dim] = 1.0
                final_layer.bias[self.theta_dim:] = 0.0

    def forward(self, s_features: torch.Tensor):

        params = self.mlp(s_features)  
        gamma_vec = params[:, 0: self.theta_dim]  
        beta_vec = params[:, self.theta_dim:]  

        return gamma_vec, beta_vec


class InstanceWiseRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, input_feature_dim: int, base: float = 10000.0, init="exp", min_period=0.01,
                 max_period=1000, scale_method="log"):
        '''
        scale_method: "log" or "min_max"
        '''
        super(InstanceWiseRotaryEmbedding, self).__init__()
        if init == 'exp':
            theta = get_exp_period(min_period, max_period, dim)
        elif init == 'rope':
            theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        else:
            print("invalid theta init")
            sys.exit(0)
        if scale_method == "log":
            log_theta = torch.log(theta)
            self.register_buffer('log_freqs', log_theta)

        self.dim = dim
        self.scale_method = scale_method
        self.instance_wise_param_net = InstanceWiseParamNet(input_feature_dim, theta_dim=dim // 2)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, s_features: torch.Tensor):
        '''
        xq: [batch_size, num_heads, seq_len, hidden_dim]
        xk: [batch_size, num_heads, seq_len, hidden_dim]
        s_features: [batch_size, input_feature_dim]
        '''
        if self.scale_method == "log":
            gamma_vec, beta_vec = self.instance_wise_param_net(s_features)
            scaled_log_freqs = gamma_vec * self.log_freqs + beta_vec
            scaled_freqs = torch.exp(scaled_log_freqs)

        def get_position_embedding(x):
            # x: [batch_size, num_heads, seq_len, hidden_dim]
            bz, nh, seq_len, d = x.shape
            x = x.reshape(-1, seq_len, d)

            L = x.shape[-2]
            t = torch.arange(L, device=x.device)

            freqs = torch.einsum('l,bd->bld', t, scaled_freqs).float()  # batch_size, seq_len, dim//2
            freqs = freqs.repeat_interleave(nh, dim=0)  # batch_size * num_heads, seq_len, dim//2
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

            x_ = x.float().reshape(*x.shape[:-1], -1, 2)
            x_ = torch.view_as_complex(x_).to(x.device)

            # rotate and then map to real number field
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(2).to(x.device)

            x_out = x_out.reshape(bz, nh, seq_len, d)
            return x_out.type_as(x)

        xq_out = get_position_embedding(xq)
        xk_out = get_position_embedding(xk)
        return xq_out, xk_out

def get_exp_period(min_period, max_period, dim):
    i = torch.arange(0, dim, 2)[: (dim // 2)]
    max_theta = 2 * np.pi / min_period
    min_theta = 2 * np.pi / max_period
    alpha = np.log(max_theta / min_theta) * (1 / (dim - 2))
    thetas = max_theta * np.exp(-alpha * i)
    return thetas  # [dim//2]

