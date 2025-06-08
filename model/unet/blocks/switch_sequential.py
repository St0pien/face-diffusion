import torch
import torch.nn as nn
from .residual_block import UNetResidualBlock


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x
