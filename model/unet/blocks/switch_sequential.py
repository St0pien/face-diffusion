import torch
import torch.nn as nn
from .residual_block import UNetResidualBlock
from .attention_block import UNetAttentionBlock


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)

        return x
