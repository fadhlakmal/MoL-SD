import torch
import torch.nn as nn

from molsd.models.autoencoder.decoder import Decoder
from molsd.models.autoencoder.encoder import Encoder

# SD1.5 latent scaling factor. Everything outside this module works with scaled latents.
LATENT_SCALE = 0.18215


class VAE(nn.Module):
    """AutoencoderKL. Parameter names match diffusers' AutoencoderKL state dict."""

    def __init__(
        self,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        latent_channels: int = 4,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        scaling_factor: float = LATENT_SCALE,
    ):
        super().__init__()
        kw = dict(
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )
        self.encoder = Encoder(**kw)
        self.decoder = Decoder(**kw)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.scaling_factor = scaling_factor
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** (len(block_out_channels) - 1)

    def encode(self, x: torch.Tensor, sample: bool = True, generator: torch.Generator | None = None) -> torch.Tensor:
        """Image in [-1, 1] -> scaled latent. sample=False returns the posterior mean (use for conditions)."""
        mean, logvar = self.quant_conv(self.encoder(x)).chunk(2, dim=1)
        z = mean
        if sample:
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            z = mean + std * torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return z * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent -> image in roughly [-1, 1]."""
        return self.decoder(self.post_quant_conv(z / self.scaling_factor))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)
