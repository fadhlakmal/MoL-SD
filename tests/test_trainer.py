import math
import os

import pytest
import torch
from omegaconf import OmegaConf

from sd.config import load_config
from sd.data.dataset import ConditionalImageDataset
from sd.data.multitask import MultiTaskLoader
from sd.engine.trainer import Trainer, collate_val_batches
from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.objectives import build_objective
from sd.utils.checkpoint import find_latest
from tests.conftest import TINY_UNET, StubTextEncoder, tiny_vae


def make_trainer(tmp_path, tasks, prompts, objective="flow_matching", **training):
    cfg = load_config(overrides=[f"experiment_name=smoke", f"output_dir={tmp_path / 'runs'}"])
    cfg = OmegaConf.merge(
        cfg,
        {
            "model": {"in_channels": 8, "grad_checkpointing": True},
            "objective": {"type": objective},
            "data": {"size": 32, "batch_size": 2, "num_workers": 0, "prompts_file": prompts, "tasks": tasks},
            "training": {
                "max_steps": 4,
                "warmup_steps": 2,
                "grad_accum": 2,
                "save_every": 2,
                "val_every": 2,
                "num_val_samples": 2,
                "optimizer": "adamw",
                "mixed_precision": "fp32",
                "ema_decay": 0.9,
                **training,
            },
            "sampling": {"steps": 3, "cfg_scale": 2.0},
            "logging": {"wandb": False},
        },
    )
    datasets = {
        name: ConditionalImageDataset(t["target_dir"], t["cond_dir"], prompts, size=32, task=name, prompt_dropout=0.5)
        for name, t in tasks.items()
    }
    torch.manual_seed(0)
    return Trainer(
        cfg=cfg,
        unet=UNet2DConditionModel(TINY_UNET, in_channels=8),
        vae=tiny_vae(),
        text_encoder=StubTextEncoder(),
        objective=build_objective(cfg.objective),
        data=MultiTaskLoader(datasets, batch_size=2, num_workers=0),
        val_batches=collate_val_batches(datasets, 2),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("objective", ["flow_matching", "epsilon"])
def test_train_save_resume(tiny_data, objective):
    tmp_path, tasks, prompts = tiny_data
    trainer = make_trainer(tmp_path, tasks, prompts, objective)

    modes = []
    trainer.unet.register_forward_pre_hook(lambda m, _: modes.append(m.training) if torch.is_grad_enabled() else None)
    metrics = trainer.train_step()
    trainer.step += 1
    assert all(modes) and modes, "UNet must be in train mode during training steps"
    assert math.isfinite(metrics["loss/total"])
    assert {"loss/a", "loss/b", "grad_norm", "lr"} <= metrics.keys()
    assert any(k.startswith("loss_t/") for k in metrics)

    trainer.train()
    ckpt = find_latest(trainer.run_dir)
    assert ckpt is not None and ckpt.endswith("step_0000004")
    assert os.path.exists(os.path.join(trainer.run_dir, "checkpoints", "step_0000002", "unet.safetensors"))
    for f in ("unet.safetensors", "ema.safetensors", "state.pt", "config.yaml"):
        assert os.path.exists(os.path.join(ckpt, f))

    resumed = make_trainer(tmp_path, tasks, prompts, objective, resume="auto")
    assert resumed.step == 4
    for (name, a), b in zip(trainer.unet.state_dict().items(), resumed.unet.state_dict().values()):
        torch.testing.assert_close(a, b, msg=name)
    assert resumed.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()
    for k, v in trainer.ema.state_dict().items():
        torch.testing.assert_close(resumed.ema.state_dict()[k], v)


def test_validation_restores_train_mode(tiny_data):
    tmp_path, tasks, prompts = tiny_data
    trainer = make_trainer(tmp_path, tasks, prompts)
    trainer.unet.train()
    images = trainer.validate()
    assert trainer.unet.training
    assert set(images) == {"a", "b"} and images["a"][0].size == (32 * 3, 32)  # condition | generated | target


def test_config_rejects_unknown_keys():
    with pytest.raises(Exception):
        load_config(overrides=["experiment_name=x", "training.learning_rat=1e-4"])
