import torch
import torch.nn as nn
from .blocks.switch_sequential import SwitchSequential
from .blocks.residual_block import UNetResidualBlock
from .blocks.attention_block import UNetAttentionBlock
from .blocks.upsample import UpSample


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 80, kernel_size=3,
                             padding=1), nn.Dropout2d(0.1)),

            SwitchSequential(UNetResidualBlock(80, 80),
                             UNetAttentionBlock(4, 20)),

            SwitchSequential(UNetResidualBlock(80, 80),
                             UNetAttentionBlock(4, 20)),

            SwitchSequential(
                nn.Conv2d(80, 80, kernel_size=3, stride=2, padding=1), nn.Dropout2d(0.1)),

            SwitchSequential(UNetResidualBlock(80, 160),
                             UNetAttentionBlock(4, 40)),

            SwitchSequential(UNetResidualBlock(160, 160),
                             UNetAttentionBlock(4, 40)),

            SwitchSequential(
                nn.Conv2d(160, 160, kernel_size=3, stride=2, padding=1), nn.Dropout(0.1)),

            SwitchSequential(UNetResidualBlock(160, 320),
                             UNetAttentionBlock(4, 80)),

            SwitchSequential(UNetResidualBlock(320, 320),
                             UNetAttentionBlock(4, 80)),

            SwitchSequential(
                nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1), nn.Dropout(0.1)),

            SwitchSequential(UNetResidualBlock(320, 320)),

            SwitchSequential(UNetResidualBlock(320, 320)),
        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(320, 320),

            UNetAttentionBlock(4, 80),

            UNetResidualBlock(320, 320),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(640, 320)),

            SwitchSequential(UNetResidualBlock(640, 320)),

            SwitchSequential(UNetResidualBlock(640, 320),
                             UpSample(320), nn.Dropout(0.1)),

            SwitchSequential(UNetResidualBlock(640, 320),
                             UNetAttentionBlock(4, 80)),

            SwitchSequential(UNetResidualBlock(640, 320),
                             UNetAttentionBlock(4, 80)),

            SwitchSequential(UNetResidualBlock(480, 320),
                             UNetAttentionBlock(4, 80), UpSample(320)),

            SwitchSequential(UNetResidualBlock(480, 160),
                             UNetAttentionBlock(4, 40)),

            SwitchSequential(UNetResidualBlock(320, 160),
                             UNetAttentionBlock(4, 40)),

            SwitchSequential(UNetResidualBlock(240, 160),
                             UNetAttentionBlock(4, 40), UpSample(160)),

            SwitchSequential(UNetResidualBlock(240, 80),
                             UNetAttentionBlock(4, 20)),

            SwitchSequential(UNetResidualBlock(160, 80),
                             UNetAttentionBlock(4, 20)),

            SwitchSequential(UNetResidualBlock(160, 80),
                             UNetAttentionBlock(4, 20)),
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)

        return x
