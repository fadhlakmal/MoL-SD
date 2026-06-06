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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w).permute(0, 2, 1) # (B, H*W, C)
        k = k.reshape(b, c, h * w) # (B, C, H*W)
        v = v.reshape(b, c, h * w) # (B, C, H*W)

        w_ = torch.bmm(q, k) * (c ** -0.5) # (B, H*W, H*W)
        w_ = F.softmax(w_, dim=-1)

        h_ = torch.bmm(v, w_.permute(0, 2, 1)) # (B, C, H*W)
        h_ = h_.reshape(b, c, h, w)
        return x + self.proj_out(h_)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)

        self.down_stage1 = nn.ModuleList([
            ResnetBlock(128, 128),
            ResnetBlock(128, 128),
            Downsample(128)
        ])

        self.down_stage2 = nn.ModuleList([
            ResnetBlock(128, 256),
            ResnetBlock(256, 256),
            Downsample(256)
        ])

        self.down_stage3 = nn.ModuleList([
            ResnetBlock(256, 512),
            ResnetBlock(512, 512),
            Downsample(512)
        ])

        self.down_stage4 = nn.ModuleList([
            ResnetBlock(512, 512),
            ResnetBlock(512, 512)
        ])

        self.mid_block = nn.ModuleList([
            ResnetBlock(512, 512),
            AttnBlock(512),
            ResnetBlock(512, 512)
        ])

        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv_in(x)

        for layer in self.down_stage1:
            x = layer(x)
        for layer in self.down_stage2:
            x = layer(x)
        for layer in self.down_stage3:
            x = layer(x)
        for layer in self.down_stage4:
            x = layer(x)
        for layer in self.mid_block:
            x = layer(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x