import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from .modeling_t5_instance_rope import ACT2FN as ACT2FN_rope, T5LayerNorm as T5LayerNorm_rope

from .utils import size_to_mask


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_dim: int,
            h_dim: int,
            out_dim: int,
            act_fn_name: str,
            dropout_p: float = 0.0,
            use_layer_norm: bool = False,
            position_embedding_type: str = "instance_wise_rope",
    ) -> None:
        super().__init__()
        self.position_embedding_type = position_embedding_type

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN_rope[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm_rope(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class MultiInResidualBlock(nn.Module):
    def __init__(
            self,
            in_dim_ls: Tuple[int, ...],
            h_dim: int,
            out_dim: int,
            act_fn_name: str,
            dropout_p: float = 0.0,
            use_layer_norm: bool = False,
            position_embedding_type: str = "instance_wise_rope",
    ) -> None:
        super().__init__()
        self.position_embedding_type = position_embedding_type

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = MultiInSizeLinear(in_dim_ls, h_dim)
        self.act = ACT2FN_rope[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = MultiInSizeLinear(in_dim_ls, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm_rope(out_dim)

    def forward(self, x: torch.Tensor, in_feat_size: torch.Tensor, expert_weights: Optional[torch.Tensor] = None,
                expert_indices: Optional[torch.Tensor] = None, x_final: Optional[torch.Tensor] = None):
        hid = self.act(self.hidden_layer(x, in_feat_size, expert_weights, expert_indices, x_final))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x, in_feat_size, expert_weights, expert_indices, x_final)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class MultiInSizeLinear(nn.Module):
    def __init__(
            self,
            in_features_ls: Tuple[int, ...],
            out_features: int,
            bias: bool = True,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((len(in_features_ls), out_features, max(in_features_ls) * 2), dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty((len(in_features_ls), out_features), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "mask",
            rearrange(
                torch.cat(
                    (
                        size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                        size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                    ),
                    dim=-1,
                ),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )
        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[idx, :, :feat_size])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
            self,
            x: Float[torch.Tensor, "*batch max_feat"],
            in_feat_size: Int[torch.Tensor, "*batch"],
            expert_weights: Optional[torch.Tensor] = None,
            expert_indices: Optional[torch.Tensor] = None,  # [*batch, n_experts]
            x_final: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "*batch out_feat"]:
        if expert_indices is not None:

            batch_shape = x.shape[:-1]
            x = x.view(-1, x.size(-1))
            x_final = torch.nan_to_num(x_final, nan=0.0)

            out = torch.zeros((x.size(0), self.out_features), device=x.device, dtype=x.dtype)
            expert_weights = expert_weights.view(-1, expert_weights.size(-1))  # [total_patches, n_experts]
            expert_indices = expert_indices.view(-1, expert_indices.size(-1))  # [total_patches, n_experts]
            n_real_experts = len(self.in_features_ls)

            for k in range(expert_indices.size(1)):
                indices_k = expert_indices[:, k]
                weights_k = expert_weights[:, k]

                is_real_expert_mask = (indices_k < n_real_experts)

                if not is_real_expert_mask.any():
                    continue

                for feat_idx, feat_size in enumerate(self.in_features_ls):
                    mask = (indices_k == feat_idx) & is_real_expert_mask
                    if not mask.any():
                        continue

                    weight = self.weight[feat_idx] * self.mask[feat_idx]
                    bias = self.bias[feat_idx] if self.bias is not None else 0
                    x_masked = x_final[feat_idx][mask]

                    expert_out = einsum(weight, x_masked, "out inp, ... inp -> ... out") + bias

                    real_expert_weights_sum = (expert_weights * (expert_indices < n_real_experts).float()).sum(dim=-1,
                                                                                                               keepdim=True)
                    real_expert_weights_sum[real_expert_weights_sum == 0] = 1.0

                    weights_k_norm = weights_k / real_expert_weights_sum.view(-1)

                    weighted_out = expert_out * weights_k_norm[mask].unsqueeze(-1)

                    out[mask] += weighted_out

            return out.view(*batch_shape, self.out_features)
        out = torch.tensor(0)
        # x: [256, 163, 32 * 2]
        for idx, feat_size in enumerate(self.in_features_ls):
            # self.weight: [3, 2048, 32 * 2]
            # self.mask: [3, 1, 32 * 2]
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                    torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                    * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        # expert_weights: [256, 163]
        # out: [256, 163, 2048]
        if expert_weights is not None:
            out = expert_weights.unsqueeze(-1) * out
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.weight.dtype}"
        )