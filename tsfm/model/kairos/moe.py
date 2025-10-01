from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelArgs:
    def __init__(self):
        self.dim = 4
        self.n_real_experts = 3
        self.n_null_experts = 1
        self.n_routed_experts = self.n_real_experts + self.n_null_experts
        self.n_activated_experts = 1
        self.moe_inter_dim = 1408
        self.update_bias_rate = 0.001
        self.target_dist = None
        self.route_scale = 1.0


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.register_buffer('bias', torch.zeros(args.n_routed_experts, dtype=torch.float32))
        self.update_bias_rate = args.update_bias_rate
        self.target_dist = args.target_dist
        self.route_scale = args.route_scale
        if self.target_dist is not None:
            if isinstance(self.target_dist, float):
                self.target_dist = [self.target_dist]
            assert abs(sum(self.target_dist) - 1.0) < 1e-10
            self.target_dist = torch.tensor(args.target_dist)
        if self.training:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        import math
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        scores = F.linear(torch.nan_to_num(x, nan=0.0), self.weight)
        scores = scores + self.bias
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        original_scores: torch.Tensor = scores

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        flatten_indices = indices.view(-1)
        flatten_weights = weights.view(-1)
        expert_weights_sum = torch.bincount(flatten_indices, weights=flatten_weights, minlength=self.bias.size(0))
        total_weights = expert_weights_sum.sum()
        target_dist = self.target_dist.to(device=x.device, dtype=expert_weights_sum.dtype)
        expected_weights_sum = (target_dist * total_weights).to(x.device)
        load_error = expected_weights_sum - expert_weights_sum
        with torch.no_grad():
            self.bias += self.update_bias_rate * (load_error / total_weights)

        return weights.type_as(x), indices


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_activated_experts = args.n_activated_experts
        self.n_real_experts = args.n_real_experts
        self.n_null_experts = args.n_null_experts
        self.gate = Gate(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        return weights, indices