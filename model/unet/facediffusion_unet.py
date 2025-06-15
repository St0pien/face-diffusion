import torch.nn as nn
from .blocks.time_embedding import TimeEmbedding
from .unet import UNet
import torch


class FaceDiffusionUNet(nn.Module):
    def __init__(self, n_groupnorm=8):
        super().__init__()
        self.time_embedding = TimeEmbedding(320, 1280)
        self.unet = UNet()
        self.final = nn.Sequential(
            nn.GroupNorm(n_groupnorm, 80),
            nn.SiLU(),
            nn.Conv2d(80, 4, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        time = self.time_embedding(time)
        output = self.unet(x, context, time)
        return self.final(output)
