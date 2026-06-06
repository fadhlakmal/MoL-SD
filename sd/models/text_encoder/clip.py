import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class CLIPEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.transformer = CLIPTextModel.from_pretrained(model_name)

        for param in self.transformer.parameters():
            param.requires_grad = False
        
    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        tokens = batch_encoding["input_ids"].to(device)
        outputs = self.transformer(input_ids=tokens)
        embeddings = outputs.last_hidden_state

        return embeddings