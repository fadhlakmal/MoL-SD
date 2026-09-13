import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPEncoder(nn.Module):
    """Frozen CLIP-L text encoder; returns last_hidden_state (B, 77, 768) as SD1.5 expects."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.transformer = CLIPTextModel.from_pretrained(model_name)
        self.transformer.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Always frozen: never switch to train mode.
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.transformer(input_ids=tokens).last_hidden_state
