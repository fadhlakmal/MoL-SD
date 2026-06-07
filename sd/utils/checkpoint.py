import os
import torch

class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def save(self, unet: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, loss: float):
        file_path = os.path.join(self.save_dir, f"unet_step_{step}.pt")
        checkpoint = {
            "step": step,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, file_path)
    
    def load(self, file_path: str, unet: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")
        
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        unet.load_state_dict(checkpoint["unet_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["step"]