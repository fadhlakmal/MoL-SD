# MoL-SD (Multi-Objective Learning Stable Diffusion)

Hi Mom

## Prerequisites
* Ubuntu / WSL2
* NVIDIA GPU with at least 8GB VRAM (CUDA toolkit installed)
* Weights & Biases (Wandb) account for training telemetry

## Installation

Run the automated Ubuntu setup script. This will install the `uv` package manager, configure the virtual environment, install PyTorch, and download the base weights.

```bash
chmod +x setup.sh
./setup.sh
```

## Project Structure

```
mol-sd
├── README.md
├── setup.sh
├── 📁 scripts/
│   ├── inference.py             # Text-to-Image execution script
│   └── train.py                 # U-Net fine-tuning loop
└── 📁 sd/
    ├── 📁 data/
    │   └── dataset.py           # ImageTextDataset and transforms
    ├── 📁 models/
    │   ├── 📁 autoencoder/
    │   │   ├── encoder.py       # Image to Latent compression
    │   │   ├── decoder.py       # Latent to Image generation
    │   │   └── vae.py           # VAEDecoder (inference) & VAE (training) wrappers
    │   ├── 📁 text_encoder/
    │   │   └── clip.py          # Frozen CLIP text embeddings
    │   └── 📁 unet/
    │       ├── unet_2d.py       # The core denoising architecture
    │       └── attention.py     # Self/Cross attention blocks
    ├── 📁 ops/
    ├── 📁 pipeline/
    │   └── sd_pipeline.py       # End-to-end generation loop
    ├── 📁 schedulers/
    |   ├── ddim.py              # DDIM Scheduler
    |   └── euler.py             # Euler Scheduler (for inference)
    └── 📁 utils/
        ├── checkpoint.py        # Saving/Resuming training states
        ├── logger.py            # Weights & Biases (Wandb) telemetry
        ├── utils.py             # Weights & Biases (Wandb) telemetry
        └── weight_mapping.py    # Safetensor weight translation dictionaries
```

## Usage: Inference

To test the `VAEDecoder` and generate an image from pure noise, run the inference script. This strictly loads the components needed for generation to save VRAM.

    uv run python -m scripts.inference

Outputs to: `test.png`

## Usage: Training

1. Place a target image inside any folder.
2. In it, add `metadata.json` with the corresponding text prompt.
3. Authenticate with Weights & Biases: `wandb login`
4. Execute the training loop: `uv run python -m scripts.train`

You can monitor the step-by-step image reconstruction and Mean Squared Error (MSE) loss dropping live on your Wandb dashboard. Weights will be saved automatically to `/checkpoints/`.