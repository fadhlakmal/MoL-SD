"""Generate images from a trained checkpoint (or the base SD1.5 weights).

The checkpoint's saved config decides objective/sampler, in_channels and image size, so sampling always
matches training.

Single image:
    uv run python -m scripts.sample --ckpt runs/tes-default --prompt "a car" --condition data/canny/conditions/0.png
Batch over a dataset (outputs are named like the targets, e.g. 12.png, for scripts.eval):
    uv run python -m scripts.sample --ckpt runs/tes-default --task canny --output_dir results/tes-default/canny
Base SD1.5 sanity check (no checkpoint, text only, epsilon + DDIM):
    uv run python -m scripts.sample --base objective.type=epsilon model.in_channels=4 --prompt "a photo of a cat"
"""

import argparse
import csv
import os
import time

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from sd.config import Config
from sd.data.dataset import ConditionalImageDataset, image_transform
from sd.engine.ema import EMA
from sd.models.loader import build_clip, build_unet, build_vae
from sd.objectives import build_objective
from sd.pipelines.conditional import ConditionalPipeline, to_pil
from sd.utils.checkpoint import resolve_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="step dir or run dir (uses its latest checkpoint)")
    src.add_argument("--base", action="store_true", help="use base weights from config defaults/overrides")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("overrides", nargs="*", help="config dotlist overrides, e.g. sampling.cfg_scale=3")

    p.add_argument("--prompt")
    p.add_argument("--condition", help="condition image path (single mode)")
    p.add_argument("--output", default="results/sample.png")

    p.add_argument("--task", help="task name from the checkpoint config (batch mode)")
    p.add_argument("--target_dir")
    p.add_argument("--cond_dir")
    p.add_argument("--prompts_file")
    p.add_argument("--output_dir")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--timing_csv", default="results/inference_speed_report.csv")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def load_run_config(args):
    if args.ckpt:
        ckpt = resolve_checkpoint(args.ckpt)
        cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(os.path.join(ckpt, "config.yaml")))
        unet_weights = os.path.join(ckpt, "unet.safetensors")
    else:
        ckpt = None
        cfg = OmegaConf.merge(OmegaConf.structured(Config), {"experiment_name": "base"})
        unet_weights = None
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg, ckpt, unet_weights or cfg.model.unet_weights


def main():
    args = parse_args()
    cfg, ckpt, unet_weights = load_run_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    unet = build_unet(unet_weights, cfg.model.in_channels)
    if args.use_ema:
        ema_path = os.path.join(ckpt or "", "ema.safetensors")
        if not os.path.exists(ema_path):
            raise FileNotFoundError(f"no EMA weights at {ema_path}")
        from safetensors.torch import load_file

        ema = EMA(unet, decay=0.0)
        ema.load_state_dict(load_file(ema_path))
        ema.copy_to(unet)
    unet.to(device)
    vae = build_vae(cfg.model.vae_weights).to(device, dtype)
    clip = build_clip(cfg.model.text_encoder).to(device, dtype)
    pipeline = ConditionalPipeline(unet, vae, clip, build_objective(cfg.objective).make_sampler())
    needs_cond = cfg.model.in_channels > 4
    s = cfg.sampling
    gen_kwargs = dict(
        negative_prompt=s.negative_prompt,
        height=cfg.data.size,
        width=cfg.data.size,
        num_steps=s.steps,
        cfg_scale=s.cfg_scale,
        dtype=dtype,
    )
    print(f"objective={cfg.objective.type} steps={s.steps} cfg={s.cfg_scale} in_channels={cfg.model.in_channels}")

    if args.prompt is not None:
        condition = None
        if needs_cond:
            if not args.condition:
                raise ValueError("this model needs --condition")
            img = Image.open(args.condition).convert("RGB")
            condition = image_transform(cfg.data.size)(img)[None]
        generator = torch.Generator(device).manual_seed(args.seed)
        image = to_pil(pipeline([args.prompt], condition=condition, generator=generator, **gen_kwargs))[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        image.save(args.output)
        print(f"saved {args.output}")
        return

    # batch mode
    if args.task:
        task = cfg.data.tasks[args.task]
        target_dir, cond_dir = task.target_dir, task.cond_dir
        prompts_file = task.prompts_file or cfg.data.prompts_file
    else:
        target_dir, cond_dir, prompts_file = args.target_dir, args.cond_dir, args.prompts_file
    for name, value in (("--target_dir or --task", target_dir), ("--output_dir", args.output_dir)):
        if value is None:
            raise ValueError(f"batch mode needs {name} (or pass --prompt for single mode)")

    dataset = ConditionalImageDataset(target_dir, cond_dir if needs_cond else None, prompts_file, size=cfg.data.size)
    subset = torch.utils.data.Subset(dataset, range(min(args.num_images, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    os.makedirs(args.output_dir, exist_ok=True)

    generator = torch.Generator(device).manual_seed(args.seed)
    start = time.time()
    for batch in loader:
        images = pipeline(list(batch["text"]), condition=batch.get("condition"), generator=generator, **gen_kwargs)
        for img, idx in zip(to_pil(images), batch["index"].tolist()):
            img.save(os.path.join(args.output_dir, f"{idx}.png"))
    total = time.time() - start
    n = len(subset)
    print(f"generated {n} images in {total:.1f}s ({total / n:.2f}s/image) -> {args.output_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.timing_csv)), exist_ok=True)
    new_file = not os.path.isfile(args.timing_csv)
    with open(args.timing_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Checkpoint", "Objective", "Steps", "Total_Time_Sec", "Time_Per_Image_Sec"])
        writer.writerow([ckpt or "base", cfg.objective.type, s.steps, f"{total:.2f}", f"{total / n:.2f}"])


if __name__ == "__main__":
    main()
