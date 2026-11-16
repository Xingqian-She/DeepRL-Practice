"""Gated feedforward variants with a common projection layout."""

from torch import nn
from torch.nn import functional as F
from .config import TransformerConfig
from ...layers.activations import Snake


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_dim, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        for weight in self.gate_up.weight.chunk(2, dim=0):
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SwiSnakeGLU(SwiGLU):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.value_activation = Snake(config.ffn_dim, config.snake_alpha_init, config.snake_eps)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * self.value_activation(up))


def make_ffn(config):
    return SwiGLU(config) if config.ffn_type == "swiglu" else SwiSnakeGLU(config)


