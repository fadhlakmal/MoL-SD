"""Checkpoint layout:

    <run_dir>/checkpoints/step_0001000/
        unet.safetensors      # full UNet weights (loadable by sd.models.loader.build_unet)
        ema.safetensors       # optional
        state.pt              # optimizer, lr scheduler, step, RNG
        config.yaml           # resolved run config (sample.py reads this)
    <run_dir>/checkpoints/latest   # text file naming the newest step dir
"""

import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file


def _ckpt_root(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoints")


def _state_dict_for_save(module: torch.nn.Module) -> dict:
    return {k: v.detach().contiguous().cpu() for k, v in module.state_dict().items()}


def save_checkpoint(run_dir, step, cfg, unet, optimizer=None, lr_scheduler=None, ema=None) -> str:
    root = _ckpt_root(run_dir)
    path = os.path.join(root, f"step_{step:07d}")
    os.makedirs(path, exist_ok=True)

    save_file(_state_dict_for_save(unet), os.path.join(path, "unet.safetensors"))
    if ema is not None:
        save_file({k: v.contiguous().cpu() for k, v in ema.state_dict().items()}, os.path.join(path, "ema.safetensors"))

    state = {
        "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, os.path.join(path, "state.pt"))
    OmegaConf.save(cfg, os.path.join(path, "config.yaml"))

    with open(os.path.join(root, "latest"), "w") as f:
        f.write(os.path.basename(path))
    return path


def find_latest(run_dir: str) -> str | None:
    marker = os.path.join(_ckpt_root(run_dir), "latest")
    if not os.path.exists(marker):
        return None
    with open(marker) as f:
        path = os.path.join(_ckpt_root(run_dir), f.read().strip())
    return path if os.path.isdir(path) else None


def resolve_checkpoint(path_or_run: str) -> str:
    """Accepts a step dir or a run dir (-> its latest checkpoint)."""
    if os.path.exists(os.path.join(path_or_run, "unet.safetensors")):
        return path_or_run
    latest = find_latest(path_or_run)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found at {path_or_run}")
    return latest


def load_checkpoint(path, unet, optimizer=None, lr_scheduler=None, ema=None, restore_rng=True) -> int:
    unet.load_state_dict(load_file(os.path.join(path, "unet.safetensors")), strict=True)
    ema_path = os.path.join(path, "ema.safetensors")
    if ema is not None and os.path.exists(ema_path):
        ema.load_state_dict(load_file(ema_path))

    state = torch.load(os.path.join(path, "state.pt"), map_location="cpu", weights_only=False)
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and state["lr_scheduler"] is not None:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    if restore_rng:
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return state["step"]
