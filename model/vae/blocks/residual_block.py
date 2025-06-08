import torch.nn as nn
import torch.nn.functional as F
import torch


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groupnorm=16):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(n_groupnorm, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(n_groupnorm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        # x: (Batch_Size, In_Channels, Height, Width)

        residue = x
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv2(x)

        return x + self.residual_layer(residue)
