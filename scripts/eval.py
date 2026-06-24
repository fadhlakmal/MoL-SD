import os
import csv
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_IMAGES = 50
    TARGET_DIR = "data_test/canny/targets"
    
    MODELS = [
        "standard_baseline",
        "canny_specialist",
        "generalist_50_50",
        "generalist_80_20",
        "generalist_20_80"
    ]
    SCHEDULERS = ["Euler", "DDIM"]

    print("Initializing FID Metric...")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    print("Loading Ground-Truth Target Images...")
    real_images_tensor = []
    real_images_numpy = []

    for i in range(NUM_IMAGES):
        img_path = os.path.join(TARGET_DIR, f"{i}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing target image: {img_path}")
            
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        
        np_img = np.array(img)
        real_images_numpy.append(np_img)
        
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).to(torch.uint8)
        real_images_tensor.append(tensor_img)

    real_batch = torch.stack(real_images_tensor).to(device)

    metrics_log = [("Model", "Scheduler", "FID_Score", "Average_SSIM")]

    print("\n" + "="*50)
    print("STARTING METRIC EVALUATION")
    print("="*50)

    for model in MODELS:
        for sched in SCHEDULERS:
            folder_path = os.path.join("results", model, sched)
            
            if not os.path.exists(folder_path):
                print(f"Skipping {model}/{sched} - Folder not found.")
                continue

            print(f"\nEvaluating: {model.upper()} via {sched}...")
            
            fid_metric.reset()
            fid_metric.update(real_batch, real=True)

            fake_images_tensor = []
            ssim_scores = []

            for i in tqdm(range(NUM_IMAGES), desc="Calculating SSIM & Packing FID"):
                fake_path = os.path.join(folder_path, f"image_{i:04d}.png")
                
                fake_img = Image.open(fake_path).convert("RGB").resize((512, 512))
                np_fake = np.array(fake_img)
                
                score = ssim(real_images_numpy[i], np_fake, data_range=255, channel_axis=-1)
                ssim_scores.append(score)
                
                tensor_fake = torch.from_numpy(np_fake).permute(2, 0, 1).to(torch.uint8)
                fake_images_tensor.append(tensor_fake)

            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            
            fake_batch = torch.stack(fake_images_tensor).to(device)
            fid_metric.update(fake_batch, real=False)
            
            print("Crunching InceptionV3 latent distances for FID...")
            fid_score = fid_metric.compute().item()

            print(f"Results -> FID: {fid_score:.2f} | SSIM: {avg_ssim:.4f}")
            
            metrics_log.append((model, sched, f"{fid_score:.2f}", f"{avg_ssim:.4f}"))

            del fake_batch
            torch.cuda.empty_cache()

    report_path = "results/evaluation_metrics_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metrics_log)

    print("\n" + "="*50)
    print(f"Final Evaluation Metrics saved to: {report_path}")
    print("="*50)

if __name__ == "__main__":
    main()