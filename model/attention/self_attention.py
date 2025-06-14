from torch import nn
import torch
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_Size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length,
                              self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 * (Batch_Size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_Len, H, Dim // H) -> (Batch_Size, H, Seq_Len, Dim // H)
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        # Numerical stability
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim // H) -> (Batch_Size, H, Seq_Len, Dim // H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim // H) -> (Batch_Size, Seq_Len, H, Dim // H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output
