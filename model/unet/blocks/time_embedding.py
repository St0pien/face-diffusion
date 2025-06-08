import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)

    def forward(self, x: torch.Tensor):
        # x: (1, N_Embed)

        x = self.linear1(x)
        x = F.silu(x)

        # (1, N_Embed * dim_factor)
        return self.linear2(x)


def encode_timesteps(timesteps, n_freqs=1000, n_dim=320):
    half_dim = n_dim // 2
    freqs = torch.pow(n_freqs, -torch.arange(0,
                      half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)

    x = timesteps[:, None] * freqs[None, :]

    # (Batch_Size, N_Dim)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
