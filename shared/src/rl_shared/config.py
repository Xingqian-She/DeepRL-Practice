"""Model structure, independent of batch size and observation modality."""

from dataclasses import dataclass, fields, replace
import math


@dataclass(frozen=True)
class TransformerConfig:
    hidden_dim: int
    num_layers: int = 2
    num_heads: int = 4
    context_length: int = 256
    ffn_dim: int | None = None
    alpha_init: float = 1e-4
    qk_norm: bool = True
    final_norm: bool = False
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "num_layers", "num_heads", "context_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads or self.head_dim % 2:
            raise ValueError("hidden_dim must be divisible by num_heads, with an even head_dim")
        if self.ffn_dim is None:
            object.__setattr__(self, "ffn_dim", math.ceil(8 * self.hidden_dim / (3 * 64)) * 64)
        elif type(self.ffn_dim) is not int or self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be a positive integer or None")
        for name in ("alpha_init", "rms_eps", "rope_theta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.alpha_init < 0 or self.rms_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("alpha_init must be nonnegative; rms_eps and rope_theta must be positive")
        if type(self.qk_norm) is not bool or type(self.final_norm) is not bool:
            raise TypeError("qk_norm and final_norm must be bool")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass(frozen=True)
class EncoderDecoderConfig(TransformerConfig):
    """Aligned causal streams with a common width, clock and per-attention window.

    ``num_layers`` is the encoder depth; decoder cross-attention reads the final
    encoder output at every layer. Projections and task heads belong to callers.
    """
    num_layers: int = 4
    decoder_layers: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.decoder_layers) is not int or self.decoder_layers < 1:
            raise ValueError("decoder_layers must be a positive integer")

    def encoder_config(self) -> TransformerConfig:
        return TransformerConfig(**{field.name: getattr(self, field.name)
                                    for field in fields(TransformerConfig)})

    def cache_config(self) -> TransformerConfig:
        return replace(self.encoder_config(), num_layers=self.num_layers + 2 * self.decoder_layers)

    @property
    def value_warmup_length(self) -> int:
        return 1 + self.num_layers * (self.context_length - 1)

    @property
    def warmup_length(self) -> int:
        return 1 + (self.num_layers + self.decoder_layers) * (self.context_length - 1)
