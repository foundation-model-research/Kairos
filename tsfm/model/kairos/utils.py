import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import numpy as np


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
            self,
            x: torch.Tensor,
            loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


def size_to_mask(
        max_size: int,
        sizes: torch.Tensor,
):
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))


def get_log_decay_weights(prediction_length: int, device: torch.device) -> torch.Tensor:
    """
    Generates a logarithmically decayed weight vector.
    The earlier the time step, the larger the weight.
    """

    i_array = np.linspace(1 + 1e-5, prediction_length - 1e-3, prediction_length)
    weights = (1 / prediction_length) * (np.log(prediction_length) - np.log(i_array))
    return torch.tensor(weights, dtype=torch.float32, device=device)
