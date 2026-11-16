"""Elementwise activations over the last feature dimension."""

import math

import torch
from torch import nn


class Snake(nn.Module):
    """Channel-wise positive frequency, learned in log space.

    epsilon protects the denominator as alpha approaches zero. FP16/BF16
    inputs use FP32 for the exponential and periodic term; FP64 is preserved.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim must be a positive integer")
        for name, value in (("alpha_init", alpha_init), ("eps", eps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.log_alpha = nn.Parameter(torch.full((dim,), math.log(alpha_init)))
        self.eps = eps

    def forward(self, x):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        alpha = self.log_alpha.to(work.dtype).exp()
        return (work + torch.sin(alpha * work).square() / (alpha + self.eps)).to(x.dtype)
