from PIL import Image


class Logger:
    """wandb wrapper; a no-op when disabled so tests and offline runs need no account."""

    def __init__(self, enabled: bool, project: str, run_name: str, config: dict | None = None, dir: str | None = None):
        self.run = None
        if enabled:
            import wandb

            self.run = wandb.init(project=project, name=run_name, config=config, dir=dir)

    def log_metrics(self, metrics: dict, step: int):
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_images(self, key: str, images: list[Image.Image], captions: list[str], step: int):
        if self.run is not None:
            import wandb

            self.run.log({key: [wandb.Image(img, caption=c) for img, c in zip(images, captions)]}, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
