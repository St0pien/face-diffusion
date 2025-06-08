import torch.nn as nn
import torch.nn.functional as F
from .blocks.residual_block import VAEResidualBlock
from .blocks.attention_block import VAEAttentionBlock
import torch


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            VAEAttentionBlock(128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 64),
            VAEResidualBlock(64, 64),
            VAEResidualBlock(64, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            VAEResidualBlock(64, 32),
            VAEResidualBlock(32, 32),
            VAEResidualBlock(32, 32),
            nn.GroupNorm(16, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return super().forward(x)
