"""Small component factories; all normalization sites share this policy."""

from torch import nn

from ...layers.normalization import DynamicTanh, RMSNorm


def make_norm(config, dim, *, qk=False):
    if qk and not config.qk_norm:
        return nn.Identity()
    if config.norm_type == "dyt":
        # Q/K use head_dim as their feature axis, shared across heads.
        return DynamicTanh(dim, config.dyt_alpha_init)
    return RMSNorm(dim, config.rms_eps, affine=not qk)
