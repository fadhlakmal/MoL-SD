import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def image_transform(size: int) -> transforms.Compose:
    """The one image preprocessing used for both training and sampling: RGB, resize, [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_prompts(prompt_file: str) -> dict[int, str]:
    """Parses lines of the form `<index>: <prompt>`."""
    prompts = {}
    with open(prompt_file, encoding="utf-8") as f:
        for line in f:
            if ": " in line:
                idx, text = line.strip().split(": ", 1)
                prompts[int(idx)] = text
    return prompts


class ConditionalImageDataset(Dataset):
    """Pairs `<target_dir>/<i>.png` with `<cond_dir>/<i>.png` and prompt i.

    cond_dir may be None for text-only training. Missing files raise instead of silently
    substituting blank images.
    """

    def __init__(
        self,
        target_dir: str,
        condition_dir: str | None,
        prompt_file: str | None,
        size: int = 512,
        task: str = "",
        prompt_dropout: float = 0.0,
    ):
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.task = task
        self.prompt_dropout = prompt_dropout
        self.prompts = load_prompts(prompt_file) if prompt_file else {}
        self.indices = sorted(int(f.split(".")[0]) for f in os.listdir(target_dir) if f.endswith(".png"))
        if not self.indices:
            raise ValueError(f"no .png files in {target_dir}")
        self.transform = image_transform(size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        prompt = self.prompts.get(file_idx, "")
        if self.prompt_dropout > 0 and random.random() < self.prompt_dropout:
            prompt = ""

        item = {
            "target": self.transform(Image.open(os.path.join(self.target_dir, f"{file_idx}.png")).convert("RGB")),
            "text": prompt,
            "index": file_idx,
            "task": self.task,
        }
        if self.condition_dir is not None:
            cond = Image.open(os.path.join(self.condition_dir, f"{file_idx}.png")).convert("RGB")
            item["condition"] = self.transform(cond)
        return item
