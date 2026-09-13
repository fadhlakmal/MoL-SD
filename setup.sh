#!/bin/bash
set -e

echo "MoL-SD setup"

if ! command -v uv &> /dev/null; then
    echo "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies (torch cu128, etc.)"
uv sync

if [ ! -f weights/sd15_unet.safetensors ]; then
    echo "Downloading and converting Stable Diffusion 1.5 weights"
    uv run python -m scripts.convert_sd15
fi

echo "
Setup complete. Next:
  uv run pytest                 # fast CPU tests
  uv run pytest -m weights      # parity vs diffusers with real weights
  wandb login                   # or set logging.wandb=false
  bash train.sh
"
