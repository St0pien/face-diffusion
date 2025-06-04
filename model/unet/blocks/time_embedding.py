import torch.nn as nn
import torch
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int, dim_factor=4):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, dim_factor
                                 * n_embd)
        self.linear2 = nn.Linear(dim_factor * n_embd, dim_factor * n_embd)

    def forward(self, x: torch.Tensor):
        # x: (1, 320)

        x = self.linear1(x)
        x = F.silu(x)

        # (1, 1280)
        return self.linear2(x)
