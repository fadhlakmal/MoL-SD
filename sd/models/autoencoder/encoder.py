import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        shortcut = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return h + shortcut


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (Linear projections, as in diffusers)."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hs = self.group_norm(x).reshape(b, c, h * w).transpose(1, 2)
        q, k, v = (proj(hs)[:, None] for proj in (self.to_q, self.to_k, self.to_v))
        out = F.scaled_dot_product_attention(q, k, v)[:, 0]
        out = self.to_out[0](out)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class MidBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(channels, channels, groups), ResnetBlock(channels, channels, groups)])
        self.attentions = nn.ModuleList([AttnBlock(channels, groups)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int, add_downsample: bool, groups: int):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups) for i in range(num_layers)])
        self.downsamplers = nn.ModuleList([Downsample(out_ch)]) if add_downsample else None


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        chs = block_out_channels
        self.conv_in = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                EncoderStage(chs[max(i - 1, 0)], ch, layers_per_block, i < len(chs) - 1, norm_num_groups)
                for i, ch in enumerate(chs)
            ]
        )
        self.mid_block = MidBlock(chs[-1], norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, chs[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(chs[-1], 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.down_blocks:
            for resnet in stage.resnets:
                x = resnet(x)
            if stage.downsamplers is not None:
                x = stage.downsamplers[0](x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
