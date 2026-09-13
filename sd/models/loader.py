"""Single place that builds models and loads weights. All loads are strict."""

import os

import torch
from safetensors.torch import load_file

from sd.models.autoencoder.vae import VAE
from sd.models.text_encoder.clip import CLIPEncoder
from sd.models.unet.unet_2d import UNet2DConditionModel, UNetConfig


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}. Run `python -m scripts.convert_sd15` first.")
    return path


def expand_conv_in(state_dict: dict, in_channels: int) -> dict:
    """Zero-init extra conv_in input channels so a pretrained 4-ch UNet starts out ignoring the condition."""
    w = state_dict["conv_in.weight"]
    have = w.shape[1]
    if in_channels == have:
        return state_dict
    if in_channels < have:
        raise ValueError(f"UNet in_channels={in_channels} is smaller than pretrained conv_in ({have})")
    expanded = torch.zeros(w.shape[0], in_channels, *w.shape[2:], dtype=w.dtype)
    expanded[:, :have] = w
    return {**state_dict, "conv_in.weight": expanded}


def build_unet(weights: str | None, in_channels: int = 4, config: UNetConfig | None = None) -> UNet2DConditionModel:
    """Build the UNet and load weights.

    `weights` may be a base SD1.5 UNet (4-ch, expanded as needed) or a fine-tuned checkpoint
    with matching shapes. `None` leaves random init (tests).
    """
    config = config or UNetConfig()
    unet = UNet2DConditionModel(config, in_channels=in_channels)
    if weights is not None:
        sd = load_file(_require(weights))
        unet.load_state_dict(expand_conv_in(sd, in_channels), strict=True)
    return unet


def build_vae(weights: str | None) -> VAE:
    vae = VAE()
    if weights is not None:
        vae.load_state_dict(load_file(_require(weights)), strict=True)
    vae.requires_grad_(False)
    return vae.eval()


def build_clip(model_name: str = "openai/clip-vit-large-patch14") -> CLIPEncoder:
    return CLIPEncoder(model_name)
