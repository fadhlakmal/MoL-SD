# MoL-SD 

Hi Mom

## Installation & Setup 
1. Setup to install `uv`, configure the venv, install PyTorch (bfloat16), and fetch base weights.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
wget "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
```

2. Authenticate WandB:
```bash
wandb login
```

## Project Structure
```
mol-sd/
├── README.md
├── setup.sh
├── train.sh                      # Wrapper script for training execution
├── inference.sh                  # Wrapper script for batch inference
├── configs/                      # YAML configurations for all runs
│   ├── default.yaml              # Standard single-task baseline config
│   └── mol.yaml                  # MoL config
├── results/                      # Generated images, CSV reports, and graphs
├── scripts/                      # Entry points
│   ├── train.py                  # Main training entry point
│   ├── train_ot.py               # OT training entry point
│   ├── batch_inference.py        # Inference runner
│   └── eval.py                   # Metric evaluation (FID/SSIM)
└── sd/                           # Core Library
    ├── data/                     # Dataset classes
    ├── engine/                   # Training logic (Trainer, OTTrainer)
    ├── models/                   # Autoencoder, CLIP, U-Net
    ├── pipelines/                # Generation pipelines (Standard, OT)
    ├── schedulers/               # Schedulers (DDIM, Euler, FlowMatching)
    └── utils/                    # Checkpoints, WandB logger, weight mapping
```

## Training
Training is now entirely driven by YAML configuration files located in the `configs/` directory.

To run a training job, change the YAML config, then run:
```bash
bash train.sh
```

For stable diffusion:
```bash
uv run python -m scripts.train --config configs/default.yaml
```

For Optimal Transport:
```bash
uv run python -m scripts.train_ot --config configs/default.yaml
```

## Inference
To generate the images, run:
```bash   
bash inference.sh
```