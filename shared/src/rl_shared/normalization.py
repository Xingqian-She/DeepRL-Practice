import torch
from torch import nn


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
