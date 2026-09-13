import torch
import torch.nn as nn
import torch.nn.functional as F

from sd.models.autoencoder.encoder import MidBlock, ResnetBlock


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int, add_upsample: bool, groups: int):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups) for i in range(num_layers)])
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if add_upsample else None


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        rev = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, rev[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock(rev[0], norm_num_groups)
        self.up_blocks = nn.ModuleList(
            [
                DecoderStage(rev[max(i - 1, 0)], ch, layers_per_block + 1, i < len(rev) - 1, norm_num_groups)
                for i, ch in enumerate(rev)
            ]
        )
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, rev[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(rev[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid_block(self.conv_in(z))
        for stage in self.up_blocks:
            for resnet in stage.resnets:
                h = resnet(h)
            if stage.upsamplers is not None:
                h = stage.upsamplers[0](h)
        return self.conv_out(F.silu(self.conv_norm_out(h)))
