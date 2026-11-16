"""Compatibility imports; implementations live in the model family or shared layers."""

from .models.transformer.attention import Attention, cached_attention
from .models.transformer.feedforward import SwiGLU, SwiSnakeGLU
from .models.transformer.blocks import TransformerBlock
from .models.transformer.backbone import TransformerBackbone
