import os

import pytest
import torch
import torch.nn as nn
from PIL import Image

from sd.models.autoencoder.vae import VAE
from sd.models.unet.unet_2d import UNetConfig

TINY_UNET = UNetConfig(
    block_out_channels=(32, 64),
    attention_levels=(True, False),
    layers_per_block=1,
    num_attention_heads=2,
    context_dim=16,
    norm_num_groups=8,
)


def tiny_vae() -> VAE:
    return VAE(block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8)


class StubTextEncoder(nn.Module):
    """Deterministic prompt -> (B, 4, 16) embeddings without downloading CLIP."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.table = nn.Embedding(256, dim)
        self.table.requires_grad_(False)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        ids = torch.tensor([[ord(c) % 256 for c in (p + "    ")[:4]] for p in prompts], device=self.table.weight.device)
        return self.table(ids)


@pytest.fixture
def tiny_data(tmp_path):
    """Two tasks with 3 image/condition pairs each (32px) and a prompts file."""
    g = torch.Generator().manual_seed(0)
    tasks = {}
    for task in ("a", "b"):
        for sub in ("targets", "conditions"):
            os.makedirs(tmp_path / task / sub)
            for i in range(3):
                arr = torch.randint(0, 255, (32, 32, 3), generator=g, dtype=torch.uint8).numpy()
                Image.fromarray(arr).save(tmp_path / task / sub / f"{i}.png")
        tasks[task] = {"target_dir": str(tmp_path / task / "targets"), "cond_dir": str(tmp_path / task / "conditions")}
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"{i}: prompt {i}" for i in range(3)))
    return tmp_path, tasks, str(prompts)
