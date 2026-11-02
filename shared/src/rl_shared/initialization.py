import math

from torch import nn


def init_head(linear: nn.Linear, *, gain: float = 1.0, bias: float = 0.0) -> nn.Linear:
    """Initialize the explicitly selected output Linear; never guess an MLP's output."""
    if not isinstance(linear, nn.Linear):
        raise TypeError("init_head expects an nn.Linear; select your custom head's output layer")
    if not math.isfinite(gain) or gain < 0 or not math.isfinite(bias):
        raise ValueError("gain must be finite and nonnegative; bias must be finite")
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, bias)
    return linear
