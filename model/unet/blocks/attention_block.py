import torch
import torch.nn as nn
from model.attention.self_attention import SelfAttention
from model.attention.cross_attention import CrossAttention
import torch.nn.functional as F


class UNetAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, n_groupnorm=10, d_context=512):
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(n_groupnorm, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear_geglu1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (Batch_Size, Features, Height, Width)

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residue_short = x
        x = self.layernorm1(x)
        x = self.attention1(x)
        x += residue_short

        # Normalization + Cross attention with skip connection
        residue_short = x
        x = self.layernorm2(x)
        x = self.attention2(x, context)
        x += residue_short

        # Normalization + FF with GeGLU and skip connection
        residue_short = x
        x = self.layernorm3(x)
        x, gate = self.linear_geglu1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu2(x)
        x += residue_short

        # (Batch_Size, Width * Height, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long
