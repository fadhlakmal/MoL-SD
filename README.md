# MoL-SD

An SD1.5 backbone (UNet, VAE and CLIP-L) trained with the **SD3 rectified-flow recipe**, with image conditioning concatenated into the UNet input. Multi-task training is built in. The ε-prediction (DDPM/DDIM) objective is kept as a baseline.

## Setup
```bash
bash setup.sh     # uv sync + download/convert SD1.5 weights into weights/
# or manually:
uv sync                                                     # also installs molsd (editable)
uv run python -m scripts.convert_sd15                       # from the HF hub
uv run python -m scripts.convert_sd15 --single_file v1-5-pruned-emaonly.safetensors
```
Our module names match the diffusers state dicts, so conversion just re-saves the weights, and every load is strict.

## Tests
```bash
uv run pytest               # CPU, tiny models, no downloads
uv run pytest -m weights    # real SD1.5 weights vs diffusers (needs weights/)
```

## Project structure
```
configs/
  default.yaml      experiments: compose groups via `defaults:`, then override
  mol.yaml
  objective/        flow_matching.yaml, epsilon.yaml
  tasks/            one file per task (canny.yaml, depth.yaml)
notebooks/          analysis and figures (imports molsd)
scripts/
  convert_sd15.py   one-time weight conversion
  train.py          training entry point (any objective, any number of tasks)
  sample.py         single-image or batch generation from a checkpoint
  eval.py           FID / SSIM
  get_data.py       dummy data for smoke runs
molsd/
  config.py         structured config
  models/           UNet, VAE, CLIP wrapper; loader.py builds them and loads weights
  objectives/       flow_matching.py (SD3 RF), epsilon.py (DDPM baseline)
  samplers/         flow_euler.py, ddim.py (shared interface)
  pipelines/        conditional.py: text + image-condition generation with CFG
  data/             ConditionalImageDataset, MultiTaskLoader
  engine/           Trainer (MTL hooks), EMA
  utils/            checkpoints, wandb logger, seeding
tests/
```

## Configs
The schema and defaults live in `molsd/config.py`, and unknown keys are rejected.

An experiment file lists reusable pieces under `defaults:`. They are merged in order, and the experiment file's own keys win:
```yaml
defaults:
  - default           # another experiment (its own defaults are expanded first)
  - tasks/depth       # adds a task
data:
  tasks:
    depth:
      weight: 0.8
```
Override anything on the command line with a dotlist, e.g. `objective.shift=3.0 training.max_steps=500`.

To add a task, add `configs/tasks/<name>.yaml`. To add an objective variant, add `configs/objective/<name>.yaml`.

## Objectives
**flow_matching** (default) follows the SD3 convention:
- `x_t = (1 − t)·x0 + t·ε`, and the target is `v = ε − x0`.
- `t = 1` is pure noise. The UNet receives `t·1000`, which matches the timestep SD1.5 associates with noise.
- `t` is drawn from a logit-normal distribution (`objective.logit_mean/std`) and then passed through the resolution shift `objective.shift`.
- Sampling uses Euler steps from t=1 to t=0.

**epsilon** is SD1.5's own ε-prediction objective, sampled with DDIM.

SD1.5's pretrained weights predict ε. Training with `flow_matching` is therefore an adaptation fine-tune: expect it to take a few thousand steps before samples look good. Watch the `loss_t/bin*` curves in wandb.

## Training
```bash
uv run python -m scripts.train --config configs/default.yaml
uv run python -m scripts.train --config configs/mol.yaml training.max_steps=2000 logging.wandb=false
```
- Runs are written to `runs/<experiment_name>/`. Checkpoints go to `checkpoints/step_N/`, holding `unet.safetensors`, `state.pt`, `config.yaml` and an optional `ema.safetensors`.
- A checkpoint is always saved at the end of training. Set `training.resume=auto` to continue from the latest one.
- Each task in `data.tasks` draws from its own data stream, so no task is truncated. Task losses are combined in `Trainer.combine_losses` (a weighted sum by default).
- For multi-task methods, override `Trainer.combine_losses`, `Trainer.compute_task_losses` and `Trainer.trainable_parameters`.

**8 GB GPUs:** a full fp32 UNet fine-tune with 8-bit AdamW needs roughly 8.5 GB before activations. Options:
- `training.optimizer=paged_adamw8bit`
- keep `model.grad_checkpointing=true`
- train a subset of parameters via `trainable_parameters()`

## Sampling & evaluation
```bash
# single image (objective, sampler, size and in_channels come from the checkpoint's config)
uv run python -m scripts.sample --ckpt runs/tes-default --prompt "a car" --condition data/canny/conditions/0.png
# batch over a task's data, then evaluate
uv run python -m scripts.sample --ckpt runs/tes-default --task canny --output_dir results/tes-default/canny
uv run python -m scripts.eval --generated_dir results/tes-default/canny --target_dir data/canny/targets
# base SD1.5 sanity check (no training)
uv run python -m scripts.sample --base objective.type=epsilon model.in_channels=4 sampling.cfg_scale=7.5 --prompt "a photo of a cat"
```
