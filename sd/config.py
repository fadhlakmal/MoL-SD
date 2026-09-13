"""Structured config. Unknown keys in YAML raise, and types are checked on merge."""

from dataclasses import dataclass, field

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ModelConfig:
    unet_weights: str = "weights/sd15_unet.safetensors"
    vae_weights: str = "weights/sd15_vae.safetensors"
    text_encoder: str = "openai/clip-vit-large-patch14"
    in_channels: int = 8  # 4 noisy latent + 4 condition latent; 4 for text-only
    grad_checkpointing: bool = True


@dataclass
class ObjectiveConfig:
    type: str = "flow_matching"  # flow_matching | epsilon
    timestep_sampling: str = "logit_normal"  # logit_normal | uniform (flow_matching only)
    logit_mean: float = 0.0
    logit_std: float = 1.0
    shift: float = 1.0


@dataclass
class TaskConfig:
    target_dir: str = MISSING
    cond_dir: str | None = None
    prompts_file: str | None = None  # falls back to data.prompts_file
    weight: float = 1.0


@dataclass
class DataConfig:
    size: int = 512
    batch_size: int = 1
    num_workers: int = 2
    prompts_file: str | None = None
    prompt_dropout: float = 0.1
    tasks: dict[str, TaskConfig] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    max_steps: int = 10000
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    optimizer: str = "adamw8bit"  # adamw8bit | paged_adamw8bit | adamw
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # bf16 | fp32
    ema_decay: float = 0.0  # 0 disables EMA
    save_every: int = 1000
    val_every: int = 500
    log_every: int = 10
    num_val_samples: int = 2  # per task
    resume: str | None = None  # null | "auto" | path to a step_N dir


@dataclass
class SamplingConfig:
    steps: int = 28
    cfg_scale: float = 5.0
    negative_prompt: str = ""


@dataclass
class LoggingConfig:
    wandb: bool = True
    project: str = "mol-sd"


@dataclass
class Config:
    experiment_name: str = MISSING
    output_dir: str = "runs"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.structured(Config)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
