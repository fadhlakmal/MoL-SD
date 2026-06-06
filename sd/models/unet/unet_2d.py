import torch
import torch.nn as nn
from sd.models.unet.blocks import TimestepEmbedding, ResnetBlock2D, Downsample2D, Upsample2D
from sd.models.unet.attention import SpatialTransformer

class UNet2DConditionModel(nn.Module):
    def __init__(self, sample_size: int = 64, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.sample_size = sample_size
        
        self.time_embedding = TimestepEmbedding(in_channels=320, out_channels=1280)
        
        self.conv_in = nn.Conv2d(in_channels, 320, kernel_size=3, stride=1, padding=1)

        # downsample
        self.down_blocks_0_resnet1 = ResnetBlock2D(320, 320, temb_channels=1280)
        self.down_blocks_0_attn1   = SpatialTransformer(channels=320, n_heads=8, d_head=40, context_dim=768)
        self.down_blocks_0_resnet2 = ResnetBlock2D(320, 320, temb_channels=1280)
        self.down_blocks_0_attn2   = SpatialTransformer(channels=320, n_heads=8, d_head=40, context_dim=768)
        self.down_blocks_0_down    = Downsample2D(320) 
        
        self.down_blocks_1_resnet1 = ResnetBlock2D(320, 640, temb_channels=1280)
        self.down_blocks_1_attn1   = SpatialTransformer(channels=640, n_heads=8, d_head=80, context_dim=768)
        self.down_blocks_1_resnet2 = ResnetBlock2D(640, 640, temb_channels=1280)
        self.down_blocks_1_attn2   = SpatialTransformer(channels=640, n_heads=8, d_head=80, context_dim=768)
        self.down_blocks_1_down    = Downsample2D(640)
        
        self.down_blocks_2_resnet1 = ResnetBlock2D(640, 1280, temb_channels=1280)
        self.down_blocks_2_attn1   = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.down_blocks_2_resnet2 = ResnetBlock2D(1280, 1280, temb_channels=1280)
        self.down_blocks_2_attn2   = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.down_blocks_2_down    = Downsample2D(1280)
        
        self.down_blocks_3_resnet1 = ResnetBlock2D(1280, 1280, temb_channels=1280)
        self.down_blocks_3_resnet2 = ResnetBlock2D(1280, 1280, temb_channels=1280)

        # bottleneck
        self.mid_block_resnet1 = ResnetBlock2D(1280, 1280, temb_channels=1280)
        self.mid_block_attn = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.mid_block_resnet2 = ResnetBlock2D(1280, 1280, temb_channels=1280)

        # upsample
        self.up_blocks_0_resnet1 = ResnetBlock2D(1280 + 1280, 1280, temb_channels=1280)
        self.up_blocks_0_resnet2 = ResnetBlock2D(1280 + 1280, 1280, temb_channels=1280)
        self.up_blocks_0_resnet3 = ResnetBlock2D(1280 + 1280, 1280, temb_channels=1280)
        self.up_blocks_0_up      = Upsample2D(1280)
        
        self.up_blocks_1_resnet1 = ResnetBlock2D(1280 + 1280, 1280, temb_channels=1280)
        self.up_blocks_1_attn1   = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.up_blocks_1_resnet2 = ResnetBlock2D(1280 + 1280, 1280, temb_channels=1280)
        self.up_blocks_1_attn2   = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.up_blocks_1_resnet3 = ResnetBlock2D(1280 + 640, 1280, temb_channels=1280) # Receives 640 skip from level 2
        self.up_blocks_1_attn3   = SpatialTransformer(channels=1280, n_heads=8, d_head=160, context_dim=768)
        self.up_blocks_1_up      = Upsample2D(1280)
        
        self.up_blocks_2_resnet1 = ResnetBlock2D(1280 + 640, 640, temb_channels=1280)
        self.up_blocks_2_attn1   = SpatialTransformer(channels=640, n_heads=8, d_head=80, context_dim=768)
        self.up_blocks_2_resnet2 = ResnetBlock2D(640 + 640, 640, temb_channels=1280)
        self.up_blocks_2_attn2   = SpatialTransformer(channels=640, n_heads=8, d_head=80, context_dim=768)
        self.up_blocks_2_resnet3 = ResnetBlock2D(640 + 320, 640, temb_channels=1280) # Receives 320 skip from level 1
        self.up_blocks_2_attn3   = SpatialTransformer(channels=640, n_heads=8, d_head=80, context_dim=768)
        self.up_blocks_2_up      = Upsample2D(640)
        
        self.up_blocks_3_resnet1 = ResnetBlock2D(640 + 320, 320, temb_channels=1280)
        self.up_blocks_3_attn1   = SpatialTransformer(channels=320, n_heads=8, d_head=40, context_dim=768)
        self.up_blocks_3_resnet2 = ResnetBlock2D(320 + 320, 320, temb_channels=1280)
        self.up_blocks_3_attn2   = SpatialTransformer(channels=320, n_heads=8, d_head=40, context_dim=768)
        self.up_blocks_3_resnet3 = ResnetBlock2D(320 + 320, 320, temb_channels=1280)
        self.up_blocks_3_attn3   = SpatialTransformer(channels=320, n_heads=8, d_head=40, context_dim=768)

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_out = nn.Conv2d(320, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps = timestep.expand(sample.shape[0])
        t_emb = self.get_timestep_embedding(timesteps, max_period=10000)
        temb = self.time_embedding(t_emb)

        h = self.conv_in(sample)
        skip_connections = [h]

        # downsample
        h = self.down_blocks_0_resnet1(h, temb); h = self.down_blocks_0_attn1(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_0_resnet2(h, temb); h = self.down_blocks_0_attn2(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_0_down(h); skip_connections.append(h)

        h = self.down_blocks_1_resnet1(h, temb); h = self.down_blocks_1_attn1(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_1_resnet2(h, temb); h = self.down_blocks_1_attn2(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_1_down(h); skip_connections.append(h)

        h = self.down_blocks_2_resnet1(h, temb); h = self.down_blocks_2_attn1(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_2_resnet2(h, temb); h = self.down_blocks_2_attn2(h, encoder_hidden_states); skip_connections.append(h)
        h = self.down_blocks_2_down(h); skip_connections.append(h)

        h = self.down_blocks_3_resnet1(h, temb); skip_connections.append(h)
        h = self.down_blocks_3_resnet2(h, temb); skip_connections.append(h)

        # bottleneck
        h = self.mid_block_resnet1(h, temb)
        h = self.mid_block_attn(h, encoder_hidden_states)
        h = self.mid_block_resnet2(h, temb)

        # upsample
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_0_resnet1(h, temb)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_0_resnet2(h, temb)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_0_resnet3(h, temb)
        h = self.up_blocks_0_up(h)

        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_1_resnet1(h, temb); h = self.up_blocks_1_attn1(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_1_resnet2(h, temb); h = self.up_blocks_1_attn2(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_1_resnet3(h, temb); h = self.up_blocks_1_attn3(h, encoder_hidden_states)
        h = self.up_blocks_1_up(h)

        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_2_resnet1(h, temb); h = self.up_blocks_2_attn1(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_2_resnet2(h, temb); h = self.up_blocks_2_attn2(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_2_resnet3(h, temb); h = self.up_blocks_2_attn3(h, encoder_hidden_states)
        h = self.up_blocks_2_up(h)

        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_3_resnet1(h, temb); h = self.up_blocks_3_attn1(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_3_resnet2(h, temb); h = self.up_blocks_3_attn2(h, encoder_hidden_states)
        h = torch.cat([h, skip_connections.pop()], dim=1); h = self.up_blocks_3_resnet3(h, temb); h = self.up_blocks_3_attn3(h, encoder_hidden_states)

        h = self.conv_norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.conv_out(h)
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int = 320, max_period: int = 10000) -> torch.Tensor:
        half_dim = embedding_dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent * -(torch.log(torch.tensor(max_period, dtype=torch.float32)) / (half_dim - 1))
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb