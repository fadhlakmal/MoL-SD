import math
import os
import time

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import default_collate
from tqdm import tqdm

from molsd.data.multitask import MultiTaskLoader
from molsd.engine.ema import EMA
from molsd.objectives.base import Objective
from molsd.pipelines.conditional import ConditionalPipeline, to_pil
from molsd.utils.checkpoint import find_latest, load_checkpoint, save_checkpoint
from molsd.utils.logger import Logger

NUM_T_BINS = 5


class Trainer:
    """Objective-agnostic, multi-task trainer.

    Extension points for MTL work:
      - trainable_parameters(): which parameters train (freezing, LoRA, adapters)
      - compute_task_losses():   per-task losses (e.g. task-specific conditioning or heads)
      - combine_losses():        how task losses become one scalar (weighted sum by default)
    """

    def __init__(
        self,
        cfg: DictConfig,
        unet: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        objective: Objective,
        data: MultiTaskLoader,
        val_batches: dict[str, dict],
        device: torch.device,
        logger: Logger | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.unet = unet.to(device)
        self.dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float32
        self.vae = vae.to(device, dtype=self.dtype)
        self.text_encoder = text_encoder.to(device, dtype=self.dtype)
        self.objective = objective
        self.data = data
        self.val_batches = val_batches
        self.logger = logger or Logger(False, "", "")
        self.run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        self.task_weights = {name: t.weight for name, t in cfg.data.tasks.items()}

        if cfg.model.grad_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.optimizer = self.build_optimizer()
        warmup = max(cfg.training.warmup_steps, 1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: min(1.0, (s + 1) / warmup))
        self.ema = EMA(self.unet, cfg.training.ema_decay) if cfg.training.ema_decay > 0 else None
        self.pipeline = ConditionalPipeline(self.unet, self.vae, self.text_encoder, objective.make_sampler())

        self.step = 0
        self._resume()

    # ----- extension points -------------------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def compute_task_losses(self, batches: dict[str, dict]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Returns ({task: scalar loss}, {task: (per_sample_loss, t_unit)} for diagnostics)."""
        losses, details = {}, {}
        for task, batch in batches.items():
            per_sample, t_unit = self.task_loss(batch)
            losses[task] = per_sample.mean()
            details[task] = (per_sample.detach(), t_unit.detach())
        return losses, details

    def combine_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.task_weights[task] * loss for task, loss in losses.items())

    # ----- core -------------------------------------------------------------------------------

    def build_optimizer(self) -> torch.optim.Optimizer:
        t = self.cfg.training
        params = self.trainable_parameters()
        if t.optimizer in ("adamw8bit", "paged_adamw8bit") and self.device.type == "cuda":
            import bitsandbytes as bnb

            cls = bnb.optim.AdamW8bit if t.optimizer == "adamw8bit" else bnb.optim.PagedAdamW8bit
            return cls(params, lr=t.learning_rate, weight_decay=t.weight_decay)
        return torch.optim.AdamW(params, lr=t.learning_rate, weight_decay=t.weight_decay)

    @torch.no_grad()
    def encode_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        target = batch["target"].to(self.device, self.dtype, non_blocking=True)
        x0 = self.vae.encode(target, sample=True).float()
        cond = None
        if "condition" in batch:
            condition = batch["condition"].to(self.device, self.dtype, non_blocking=True)
            cond = self.vae.encode(condition, sample=False).float()
        context = self.text_encoder(list(batch["text"])).float()
        return x0, cond, context

    def task_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x0, cond, context = self.encode_batch(batch)
        noise = torch.randn_like(x0)
        t = self.objective.sample_t(x0.shape[0], self.device)
        pair = self.objective.prepare(x0, noise, t)

        model_in = pair.x_t if cond is None else torch.cat([pair.x_t, cond], dim=1)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            pred = self.unet(model_in, pair.model_t, context)
        return self.objective.loss(pred, pair.target), pair.t_unit

    def train_step(self) -> dict:
        self.unet.train()
        t = self.cfg.training
        metrics: dict[str, float] = {}
        per_sample_all, t_all = [], []

        for _ in range(t.grad_accum):
            losses, details = self.compute_task_losses(next(self.data))
            total = self.combine_losses(losses)
            (total / t.grad_accum).backward()
            for task, loss in losses.items():
                metrics[f"loss/{task}"] = metrics.get(f"loss/{task}", 0.0) + loss.item() / t.grad_accum
            metrics["loss/total"] = metrics.get("loss/total", 0.0) + total.item() / t.grad_accum
            for ps, tu in details.values():
                per_sample_all.append(ps)
                t_all.append(tu)

        max_norm = t.max_grad_norm if t.max_grad_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_norm)

        if math.isfinite(metrics["loss/total"]) and torch.isfinite(grad_norm):
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.ema is not None:
                self.ema.update(self.unet)
        else:
            metrics["skipped_nonfinite"] = 1.0
            tqdm.write(f"step {self.step}: non-finite loss/grad, skipping update")
        self.optimizer.zero_grad(set_to_none=True)

        metrics["grad_norm"] = grad_norm.item()
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        self._t_bin_metrics(metrics, torch.cat(per_sample_all), torch.cat(t_all))
        return metrics

    def _t_bin_metrics(self, metrics: dict, per_sample: torch.Tensor, t_unit: torch.Tensor):
        """Loss by noise level (bin 0 = near data, last = near noise). Spots time-direction bugs quickly."""
        bins = (t_unit.float().clamp(0, 1 - 1e-6) * NUM_T_BINS).long()
        for b in range(NUM_T_BINS):
            mask = bins == b
            if mask.any():
                metrics[f"loss_t/bin{b}"] = per_sample[mask].mean().item()

    def train(self):
        t = self.cfg.training
        os.makedirs(self.run_dir, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(self.run_dir, "config.yaml"))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        bar = tqdm(total=t.max_steps, initial=self.step, desc=self.cfg.experiment_name)
        last = time.time()
        while self.step < t.max_steps:
            metrics = self.train_step()
            self.step += 1
            bar.update(1)

            if self.step % t.log_every == 0 or self.step == 1:
                now = time.time()
                metrics["sec_per_step"] = (now - last) / (t.log_every if self.step > 1 else 1)
                last = now
                if self.device.type == "cuda":
                    metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                self.logger.log_metrics(metrics, step=self.step)
                bar.set_postfix(loss=f"{metrics['loss/total']:.4f}", lr=f"{metrics['lr']:.1e}")

            if t.val_every > 0 and self.step % t.val_every == 0:
                self.validate()
            if t.save_every > 0 and self.step % t.save_every == 0:
                self.save()

        bar.close()
        if find_latest(self.run_dir) != os.path.join(self.run_dir, "checkpoints", f"step_{self.step:07d}"):
            self.save()
        self.logger.finish()

    @torch.no_grad()
    def validate(self) -> dict[str, list[Image.Image]]:
        s = self.cfg.sampling
        out = {}
        for task, batch in self.val_batches.items():
            generator = torch.Generator(self.device).manual_seed(self.cfg.seed)
            images = self.pipeline(
                prompts=list(batch["text"]),
                condition=batch.get("condition"),
                negative_prompt=s.negative_prompt,
                height=self.cfg.data.size,
                width=self.cfg.data.size,
                num_steps=s.steps,
                cfg_scale=s.cfg_scale,
                generator=generator,
                dtype=self.dtype,
            )
            panels = [images.cpu(), batch["target"]]
            if "condition" in batch:
                panels.insert(0, batch["condition"])
            rows = to_pil(torch.cat(panels, dim=3))  # [condition | generated | target]
            self.logger.log_images(f"val/{task}", rows, list(batch["text"]), step=self.step)
            out[task] = rows
        return out

    def save(self) -> str:
        path = save_checkpoint(self.run_dir, self.step, self.cfg, self.unet, self.optimizer, self.lr_scheduler, self.ema)
        tqdm.write(f"saved {path}")
        return path

    def _resume(self):
        resume = self.cfg.training.resume
        if not resume:
            return
        path = find_latest(self.run_dir) if resume == "auto" else resume
        if path is None:
            return
        self.step = load_checkpoint(path, self.unet, self.optimizer, self.lr_scheduler, self.ema)
        print(f"resumed from {path} at step {self.step}")


def collate_val_batches(datasets: dict, num_samples: int) -> dict[str, dict]:
    """Fixed validation items per task: the first `num_samples` items, prompt dropout disabled."""
    batches = {}
    for task, ds in datasets.items():
        dropout, ds.prompt_dropout = ds.prompt_dropout, 0.0
        batches[task] = default_collate([ds[i] for i in range(min(num_samples, len(ds)))])
        ds.prompt_dropout = dropout
    return batches
