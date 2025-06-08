from .encoder import VAEEncoder
from .decoder import VAEDecoder
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self, x, seed):
        out, m, s = self.encoder(x, seed)
        dec = self.decoder(out)
        return dec, m, s
