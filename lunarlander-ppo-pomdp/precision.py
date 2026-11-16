"""Project compute policy. Shared model families remain dtype-agnostic."""

from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def fp32_math(device):
    """Prevent ambient autocast from changing projections or KV storage dtype."""
    with torch.autocast(device_type=device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        yield
