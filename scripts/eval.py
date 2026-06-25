import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FID and SSIM for generated images.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images folder")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to ground truth images folder")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to evaluate")
    parser.add_argument("--output_csv", type=str, default="results/evaluation_metrics.csv", help="Where to save the report")
    parser.add_argument("--model_name", type=str, default="unknown_model", help="Label for the CSV report")
    parser.add_argument("--scheduler_name", type=str, default="unknown_scheduler", help="Label for the CSV report")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating Model: {args.model_name} | Scheduler: {args.scheduler_name}")
    print("Initializing FID Metric...")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    real_images_tensor = []
    real_images_numpy = []
    for i in range(args.num_images):
        img_path = os.path.join(args.target_dir, f"{i}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing target image: {img_path}")
            
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        np_img = np.array(img)
        real_images_numpy.append(np_img)
        real_images_tensor.append(torch.from_numpy(np_img).permute(2, 0, 1).to(torch.uint8))

    real_batch = torch.stack(real_images_tensor).to(device)
    fid_metric.update(real_batch, real=True)

    fake_images_tensor = []
    ssim_scores = []
    
    if not os.path.exists(args.generated_dir):
        raise FileNotFoundError(f"Missing generated directory: {args.generated_dir}")

    for i in tqdm(range(args.num_images), desc="Calculating SSIM & Packing FID"):
        fake_path = os.path.join(args.generated_dir, f"image_{i:04d}.png") 
        
        fake_img = Image.open(fake_path).convert("RGB").resize((512, 512))
        np_fake = np.array(fake_img)
        
        score = ssim(real_images_numpy[i], np_fake, data_range=255, channel_axis=-1)
        ssim_scores.append(score)
        
        fake_images_tensor.append(torch.from_numpy(np_fake).permute(2, 0, 1).to(torch.uint8))

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    
    fake_batch = torch.stack(fake_images_tensor).to(device)
    fid_metric.update(fake_batch, real=False)
    
    print("Crunching InceptionV3 latent distances for FID...")
    fid_score = fid_metric.compute().item()
    print(f"Results -> FID: {fid_score:.2f} | SSIM: {avg_ssim:.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    file_exists = os.path.isfile(args.output_csv)
    
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Scheduler", "FID_Score", "Average_SSIM"])
        writer.writerow([args.model_name, args.scheduler_name, f"{fid_score:.2f}", f"{avg_ssim:.4f}"])

if __name__ == "__main__":
    main()