from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from molsd.models.unet.attention import SpatialTransformer
from molsd.models.unet.blocks import Downsample2D, ResnetBlock2D, TimestepEmbedding, Upsample2D, get_timestep_embedding


@dataclass
class UNetConfig:
    """Defaults reproduce SD1.5. Parameter names match diffusers' UNet2DConditionModel state dict."""

    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    attention_levels: tuple[bool, ...] = (True, True, True, False)
    layers_per_block: int = 2
    num_attention_heads: int = 8
    context_dim: int = 768
    norm_num_groups: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, cfg: UNetConfig, has_attn: bool, add_downsample: bool):
        super().__init__()
        g = cfg.norm_num_groups
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch, groups=g) for i in range(cfg.layers_per_block)]
        )
        self.attentions = (
            nn.ModuleList(
                [SpatialTransformer(out_ch, cfg.num_attention_heads, cfg.context_dim, groups=g) for _ in range(cfg.layers_per_block)]
            )
            if has_attn
            else None
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_downsample else None


class UpBlock(nn.Module):
    def __init__(
        self,
        prev_ch: int,
        out_ch: int,
        skip_chs: list[int],
        temb_ch: int,
        cfg: UNetConfig,
        has_attn: bool,
        add_upsample: bool,
    ):
        super().__init__()
        g = cfg.norm_num_groups
        n = len(skip_chs)
        self.resnets = nn.ModuleList(
            [ResnetBlock2D((prev_ch if i == 0 else out_ch) + skip_chs[i], out_ch, temb_ch, groups=g) for i in range(n)]
        )
        self.attentions = (
            nn.ModuleList([SpatialTransformer(out_ch, cfg.num_attention_heads, cfg.context_dim, groups=g) for _ in range(n)])
            if has_attn
            else None
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_upsample else None


class MidBlock(nn.Module):
    def __init__(self, ch: int, temb_ch: int, cfg: UNetConfig):
        super().__init__()
        g = cfg.norm_num_groups
        self.resnets = nn.ModuleList([ResnetBlock2D(ch, ch, temb_ch, groups=g), ResnetBlock2D(ch, ch, temb_ch, groups=g)])
        self.attentions = nn.ModuleList([SpatialTransformer(ch, cfg.num_attention_heads, cfg.context_dim, groups=g)])


class UNet2DConditionModel(nn.Module):
    def __init__(self, config: UNetConfig | None = None, **overrides):
        super().__init__()
        cfg = config or UNetConfig()
        if overrides:
            cfg = UNetConfig(**{**cfg.to_dict(), **overrides})
        self.config = cfg
        self.gradient_checkpointing = False

        chs = cfg.block_out_channels
        base = chs[0]
        temb_ch = base * 4
        self.time_embed_dim = base

        self.conv_in = nn.Conv2d(cfg.in_channels, base, kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(base, temb_ch)

        # Track channels of every skip connection pushed during the down pass.
        skip_chs = [base]
        self.down_blocks = nn.ModuleList()
        in_ch = base
        for i, out_ch in enumerate(chs):
            is_last = i == len(chs) - 1
            self.down_blocks.append(DownBlock(in_ch, out_ch, temb_ch, cfg, cfg.attention_levels[i], not is_last))
            skip_chs += [out_ch] * cfg.layers_per_block
            if not is_last:
                skip_chs.append(out_ch)
            in_ch = out_ch

        self.mid_block = MidBlock(chs[-1], temb_ch, cfg)

        self.up_blocks = nn.ModuleList()
        prev_ch = chs[-1]
        for i, out_ch in enumerate(reversed(chs)):
            is_last = i == len(chs) - 1
            block_skips = [skip_chs.pop() for _ in range(cfg.layers_per_block + 1)]
            has_attn = list(reversed(cfg.attention_levels))[i]
            self.up_blocks.append(UpBlock(prev_ch, out_ch, block_skips, temb_ch, cfg, has_attn, not is_last))
            prev_ch = out_ch
        assert not skip_chs, "skip connection bookkeeping mismatch"

        self.conv_norm_out = nn.GroupNorm(cfg.norm_num_groups, base, eps=1e-5)
        self.conv_out = nn.Conv2d(base, cfg.out_channels, kernel_size=3, padding=1)

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled

    def _run(self, module: nn.Module, *args) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """timestep: scalar or (B,) in the SD1.5 [0, 1000) scale; floats are allowed."""
        timestep = torch.as_tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.expand(sample.shape[0])

        t_emb = get_timestep_embedding(timestep, self.time_embed_dim).to(sample.dtype)
        temb = self.time_embedding(t_emb)
        ctx = encoder_hidden_states

        h = self.conv_in(sample)
        skips = [h]
        for block in self.down_blocks:
            for j, resnet in enumerate(block.resnets):
                h = self._run(resnet, h, temb)
                if block.attentions is not None:
                    h = self._run(block.attentions[j], h, ctx)
                skips.append(h)
            if block.downsamplers is not None:
                h = block.downsamplers[0](h)
                skips.append(h)

        h = self._run(self.mid_block.resnets[0], h, temb)
        h = self._run(self.mid_block.attentions[0], h, ctx)
        h = self._run(self.mid_block.resnets[1], h, temb)

        for block in self.up_blocks:
            for j, resnet in enumerate(block.resnets):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self._run(resnet, h, temb)
                if block.attentions is not None:
                    h = self._run(block.attentions[j], h, ctx)
            if block.upsamplers is not None:
                h = block.upsamplers[0](h)

        h = F.silu(self.conv_norm_out(h))
        return self.conv_out(h)
