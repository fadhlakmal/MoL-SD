#!/bin/bash
set -e 

echo "MoL-SD setup"

echo "Ubuntu system packages"
sudo apt update
sudo apt install -y curl wget git python3-venv python3-pip

if ! command -v uv &> /dev/null; then
    echo "Installing 'uv'"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "'uv' is already installed."
fi

echo "Creating Python virtual environment"
uv venv .venv
source .venv/bin/activate

echo "Installing PyTorch and Neural Network dependencies"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers safetensors wandb tqdm pillow

WEIGHT_FILE="v1-5-pruned-emaonly.safetensors"
if [ ! -f "$WEIGHT_FILE" ]; then
    echo "Downloading Stable Diffusion v1.5 weights (~4GB)"
    wget "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
else
    echo "Weights already downloaded."
fi

echo "
Setup Complete!
To get started, activate the environment:
  source .venv/bin/activate

Then try generating an image:
  uv run python -m scripts.inference
"