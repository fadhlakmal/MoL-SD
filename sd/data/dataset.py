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
