"""Train a (multi-task) conditional diffusion model.

    uv run python -m scripts.train --config configs/default.yaml [dot.list=overrides ...]
"""

import argparse
import os

import torch
from omegaconf import OmegaConf

from sd.config import load_config
from sd.data.dataset import ConditionalImageDataset
from sd.data.multitask import MultiTaskLoader
from sd.engine.trainer import Trainer, collate_val_batches
from sd.models.loader import build_clip, build_unet, build_vae
from sd.objectives import build_objective
from sd.utils.logger import Logger
from sd.utils.seed import set_seed


def build_datasets(cfg) -> dict[str, ConditionalImageDataset]:
    if not cfg.data.tasks:
        raise ValueError("config must define at least one task under data.tasks")
    datasets = {}
    for name, task in cfg.data.tasks.items():
        needs_cond = cfg.model.in_channels > 4
        if needs_cond and task.cond_dir is None:
            raise ValueError(f"task '{name}': model.in_channels={cfg.model.in_channels} requires cond_dir")
        datasets[name] = ConditionalImageDataset(
            target_dir=task.target_dir,
            condition_dir=task.cond_dir if needs_cond else None,
            prompt_file=task.prompts_file or cfg.data.prompts_file,
            size=cfg.data.size,
            task=name,
            prompt_dropout=cfg.data.prompt_dropout,
        )
        print(f"task {name}: {len(datasets[name])} samples, weight {task.weight}")
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. training.max_steps=100")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(OmegaConf.to_yaml(cfg))

    datasets = build_datasets(cfg)
    val_batches = collate_val_batches(datasets, cfg.training.num_val_samples)
    data = MultiTaskLoader(datasets, cfg.data.batch_size, cfg.data.num_workers, seed=cfg.seed)

    trainer = Trainer(
        cfg=cfg,
        unet=build_unet(cfg.model.unet_weights, cfg.model.in_channels),
        vae=build_vae(cfg.model.vae_weights),
        text_encoder=build_clip(cfg.model.text_encoder),
        objective=build_objective(cfg.objective),
        data=data,
        val_batches=val_batches,
        device=device,
        logger=Logger(
            cfg.logging.wandb,
            cfg.logging.project,
            cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.path.join(cfg.output_dir, cfg.experiment_name),
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
