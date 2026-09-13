import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int | None = None, heads: int = 8, dim_head: int = 40):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.to_k(context).view(b, context.shape[1], self.heads, -1).transpose(1, 2)
        v = self.to_v(context).view(b, context.shape[1], self.heads, -1).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out[1](self.to_out[0](out))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """Transformer2DModel equivalent for SD1.5 (conv projections, one block)."""

    def __init__(self, channels: int, n_heads: int, context_dim: int, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(dim=channels, n_heads=n_heads, d_head=channels // n_heads, context_dim=context_dim)]
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.proj_out(x) + residual
