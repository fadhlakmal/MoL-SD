import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageTextDataset(Dataset):
    def __init__(self, data_dir: str, metadata_file: str = "metadata.json", size: int = 512):
        self.data_dir = data_dir
        self.size = size

        metadata_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.data_dir, item["file_name"])
        prompt = item["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (self.size, self.size), (255, 255, 255))

        image = self.transform(image)
        return {
            "image": image,
            "text": prompt
        }

class ConditionalImageDataset(Dataset):
    def __init__(self, target_dir: str, condition_dir: str, prompt_file: str, size: int = 512):
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.size = size

        self.prompts = {}
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ": " in line:
                    idx, text = line.strip().split(": ", 1)
                    self.prompts[int(idx)] = text

        self.indices = [
            int(f.split('.')[0]) for f in os.listdir(target_dir) if f.endswith('.png')
        ]
        self.indices.sort()

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        target_path = os.path.join(self.target_dir, f"{file_idx}.png")
        condition_path = os.path.join(self.condition_dir, f"{file_idx}.png")
        prompt = self.prompts.get(file_idx, "")

        try:
            target_image = Image.open(target_path).convert("RGB")
            condition_image = Image.open(condition_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {file_idx}: {e}")
            target_image = Image.new("RGB", (self.size, self.size), (255, 255, 255))
            condition_image = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        return {
            "target": self.transform(target_image),
            "condition": self.transform(condition_image),
            "text": prompt
        }