import math

import torch
from torch import nn


class DynamicTanh(nn.Module):
    """Learned scalar slope and channel-wise affine transform; no statistics."""

    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if (isinstance(alpha_init, bool) or not isinstance(alpha_init, (int, float))
                or not math.isfinite(alpha_init) or alpha_init <= 0):
            raise ValueError("alpha_init must be finite and positive")
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.tanh(self.alpha.to(x.dtype) * x) * self.weight.to(x.dtype) + self.bias.to(x.dtype)


class RMSNorm(nn.Module):
    """FP32 statistics for half inputs; no implicit promotion of the output."""

    def __init__(self, dim: int, eps: float, *, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = (stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        return y if self.weight is None else y * self.weight.to(x.dtype)
