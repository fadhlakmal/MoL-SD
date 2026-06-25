import os
import torch
from PIL import Image

def make_dummy_data():
    base_dir = "data/canny"
    target_dir = os.path.join(base_dir, "targets")
    cond_dir = os.path.join(base_dir, "conditions")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cond_dir, exist_ok=True)
    
    prompts = []
    
    print("Generating 5 dummy images...")
    for i in range(5):
        target = Image.fromarray(torch.randint(0, 255, (512, 512, 3), dtype=torch.uint8).numpy())
        condition = Image.fromarray(torch.randint(0, 255, (512, 512, 3), dtype=torch.uint8).numpy())
        
        target.save(os.path.join(target_dir, f"{i}.png"))
        condition.save(os.path.join(cond_dir, f"{i}.png"))
        
        prompts.append(f"{i}: A beautiful dummy landscape {i}")
        
    with open("data/prompts.txt", "w") as f:
        f.write("\n".join(prompts))
        
    print("Dummy data created at data/")

if __name__ == "__main__":
    make_dummy_data()