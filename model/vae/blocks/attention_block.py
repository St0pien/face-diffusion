import torch.nn as nn
import torch.nn.functional as F
from model.attention.self_attention import SelfAttention
import torch


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels, n_groupnorm=16):
        super().__init__()
        self.groupnorm = nn.GroupNorm(n_groupnorm, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):
        # x: (Batch_Size, Features, Height, Width)

        residue = x
        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(n, c, h, w)

        x += residue

        return x
