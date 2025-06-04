import torch.nn as nn
import torch
import torch.nn.functional as F


class UpSample(nn.Module):
    def __init__(self, channels, factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.factor = factor

    def forward(self, x: torch.Tensor):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=self.factor, mode='nearest')
        return self.conv(x)
