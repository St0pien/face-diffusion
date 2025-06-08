import torch.nn as nn
import torch.nn.functional as F
from .blocks.residual_block import VAEResidualBlock
from .blocks.attention_block import VAEAttentionBlock
import torch


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                         VAEResidualBlock(32, 32),
                         VAEResidualBlock(32, 32),
                         nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
                         VAEResidualBlock(32, 64),
                         VAEResidualBlock(64, 64),
                         nn.Conv2d(64, 64, kernel_size=3,
                                   stride=2, padding=0),
                         VAEResidualBlock(64, 128),
                         VAEResidualBlock(128, 128),
                         nn.Conv2d(128, 128, kernel_size=3,
                                   stride=2, padding=0),
                         VAEResidualBlock(128, 128),
                         VAEResidualBlock(128, 128),
                         VAEResidualBlock(128, 128),
                         VAEAttentionBlock(128),
                         VAEResidualBlock(128, 128),
                         nn.GroupNorm(16, 128),
                         nn.SiLU(),
                         nn.Conv2d(128, 8, kernel_size=3, padding=1),
                         nn.Conv2d(8, 8, kernel_size=1, padding=0)
                         )

    def forward(self, x: torch.Tensor, seed):
        # x: (Batch_Size, Channel, Height, Width)
        # seed: (Batch_Size, 4, Height, Width)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        x = mean + stdev*seed

        return (x, mean, log_variance)
