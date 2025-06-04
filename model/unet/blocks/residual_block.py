import torch.nn as nn
import torch
import torch.nn.functional as F


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280, n_groupnorm=32):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merge = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, N_Time)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_features(feature)
        time = F.silu(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merge(merged)
        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
