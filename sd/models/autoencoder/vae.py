import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.post_quant_conv(x))

class VAE(nn.Module):
    def __init__(self, encoder, decoder, quant_conv, post_quant_conv):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, log_var = torch.chunk(moments, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(mean)
        return mean + std * noise

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))