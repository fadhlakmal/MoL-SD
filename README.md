# MoL-SD (Multi-Objective Learning Stable Diffusion)

Official repository for evaluating Early-Fusion 8-Channel U-Net Architecture on Stable Diffusion for Multi-Objective Conditioning (Canny Edge & Depth Map).

## System Requirements
Due to the 8-channel U-Net modifications and early-fusion concatenation, this pipeline requires strict memory management:
* OS: Ubuntu 22.04/24.04 or Windows WSL2
* GPU: NVIDIA GPU with at least **16GB VRAM** (Developed & tested on RTX 4080 Super)
* RAM: 32GB - 64GB System RAM
* Storage: NVMe SSD recommended (Base weights + 5 Checkpoints require ~20GB)
* Telemetry: Weights & Biases (WandB) account for training logging

## Installation & Setup

1. Run the automated setup script to install the `uv` package manager, configure the virtual environment, install PyTorch (bfloat16 enabled), and download the v1.5 base weights.
   chmod +x setup.sh
   ./setup.sh

2. Authenticate your wandb account for training telemetry:
   wandb login

## 📂 Project Structure

mol-sd/
├── README.md
├── setup.sh
├── 📁 data_test/                # 50 isolated MS COCO images for evaluation
├── 📁 data/                     # 500 isolated MS COCO images for training
├── 📁 results/                  # Generated images, CSV reports, and graphs
├── 📁 sd/                       # Core Stable Diffusion Architecture
│   ├── 📁 data/                 # MS COCO streaming ImageTextDataset
│   ├── 📁 models/               # Autoencoder, CLIP, and U-Net (2D & Attention)
│   ├── 📁 pipeline/             # SD generation pipeline
│   ├── 📁 schedulers/           # DDIM and Euler Discrete schedulers
│   └── 📁 utils/                # Weight mapping, bfloat16 casting, and WandB config
├── get_data.py                   # Get & Preprocess 500 MS COCO Data for Training  
└── get_test_data.py              # Get & Preprocess 50 MS COCO Data for Testing  

## Usage 1: Training the Models
The training loop utilizes Hugging Face's dataset streaming to save disk space. It applies memory-aggressive optimizations including `torch.bfloat16` and `AdamW8bit`.

To execute the training loop:
   uv run python -m scripts.train

Note: You can monitor step-by-step image reconstruction, Canny/Depth loss, and peak VRAM allocation live on your WandB dashboard.

## Usage 2: Batch Inference
To reproduce the architecture study and generate the 500 test images using Classifier-Free Guidance (CFG = 1.0) and 20 inference steps:

   uv run python -m scripts.batch_inference

This will create nested folders inside `results/` containing outputs from both Euler and DDIM schedulers. It also outputs `inference_speed_report.csv`.

## Usage 3: Evaluation & Visualization
Once batch inference is complete, you can calculate the Fréchet Inception Distance (FID) and Structural Similarity Index (SSIM):

   uv run python -m scripts.eval