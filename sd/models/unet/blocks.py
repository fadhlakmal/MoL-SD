import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
    
class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
    
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = self.time_emb_proj(F.silu(temb))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.conv_shortcut(x)
