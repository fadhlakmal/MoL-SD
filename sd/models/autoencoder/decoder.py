import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.models.autoencoder.encoder import ResnetBlock, AttnBlock

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1)

        self.mid_block = nn.ModuleList([
            ResnetBlock(512, 512),
            AttnBlock(512),
            ResnetBlock(512, 512)
        ])

        self.up_stage1 = nn.ModuleList([
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            Upsample(512)
        ])

        self.up_stage2 = nn.ModuleList([
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            ResnetBlock(512, 512),
            Upsample(512)
        ])

        self.up_stage3 = nn.ModuleList([
            ResnetBlock(512, 256),
            ResnetBlock(256, 256),
            ResnetBlock(256, 256),
            Upsample(256)
        ])

        self.up_stage4 = nn.ModuleList([
            ResnetBlock(256, 128),
            ResnetBlock(128, 128),
            ResnetBlock(128, 128)
        ])

        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        for layer in self.mid_block:
            h = layer(h)
        for layer in self.up_stage1:
            h = layer(h)
        for layer in self.up_stage2:
            h = layer(h)
        for layer in self.up_stage3:
            h = layer(h)
        for layer in self.up_stage4:
            h = layer(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h