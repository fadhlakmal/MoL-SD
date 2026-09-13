import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of a module's trainable parameters (kept on the same device)."""

    def __init__(self, module: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, module: nn.Module):
        names = [n for n, p in module.named_parameters() if n in self.shadow]
        params = dict(module.named_parameters())
        torch._foreach_lerp_(
            [self.shadow[n] for n in names], [params[n].detach().float() for n in names], 1.0 - self.decay
        )

    @torch.no_grad()
    def copy_to(self, module: nn.Module):
        params = dict(module.named_parameters())
        for n, v in self.shadow.items():
            params[n].copy_(v)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict):
        for n, v in state.items():
            self.shadow[n].copy_(v)
