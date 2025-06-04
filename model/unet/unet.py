import torch
import torch.nn as nn
from blocks.switch_sequential import SwitchSequential
from blocks.residual_block import UNetResidualBlock
from blocks.attention_block import UNetAttentionBlock
from blocks.upsample import UpSample


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 320),
                             UNetAttentionBlock(8, 40))
        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UpSample(1280))
        ])

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        skip_connections = []
        for layer in self.encoders:
            x = layer(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)

        for layer in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, time)

        return x
