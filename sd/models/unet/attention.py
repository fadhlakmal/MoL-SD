import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        heads = self.heads
        context = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        
        q = q.view(b, seq_len_q, heads, -1).transpose(1, 2)
        k = k.view(b, seq_len_k, heads, -1).transpose(1, 2)
        v = v.view(b, seq_len_k, heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, seq_len_q, -1)
        return self.to_out(out)
        
class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, context_dim: int = None):
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head)
        self.norm1 = nn.LayerNorm(dim)

        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 8),
            GEGLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(x)
        return x
    
class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, d_head: int, context_dim: int = None):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(dim=channels, n_heads=n_heads, d_head=d_head, context_dim=context_dim)
        ])
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = x.shape
        in_x = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
            
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.proj_out(x)
        return x + in_x