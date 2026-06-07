import wandb
from PIL import Image

class WandbLogger:
    def __init__(self, project_name: str = "mol-sd-training", run_name: str = "baseline-unet", config: dict = None):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )

    def log_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def log_image(self, image: Image.Image, prompt: str, step: int):
        wandb.log({
            "generated_samples": wandb.Image(image, caption=f"Step {step}: {prompt}")
        }, step=step)

    def finish(self):
        wandb.finish()