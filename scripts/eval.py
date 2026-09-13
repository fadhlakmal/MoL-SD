"""FID and SSIM between generated images and targets. Files are matched by name (sample.py writes <i>.png).

    uv run python -m scripts.eval --generated_dir results/tes-default/canny --target_dir data/canny/targets
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated_dir", required=True)
    p.add_argument("--target_dir", required=True)
    p.add_argument("--num_images", type=int, default=None, help="default: all generated images")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_csv", default="results/evaluation_metrics.csv")
    p.add_argument("--model_name", default="unknown_model")
    return p.parse_args()


def load_rgb(path: str, size: int) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC))


def to_uint8_batch(arrays: list[np.ndarray], device) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = sorted((f for f in os.listdir(args.generated_dir) if f.endswith(".png")), key=lambda f: int(f.split(".")[0]))
    if args.num_images is not None:
        names = names[: args.num_images]
    if not names:
        raise FileNotFoundError(f"no generated .png files in {args.generated_dir}")
    missing = [n for n in names if not os.path.exists(os.path.join(args.target_dir, n))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} targets missing, e.g. {missing[:3]}")

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    ssim_scores = []
    for i in tqdm(range(0, len(names), args.batch_size), desc="eval"):
        chunk = names[i : i + args.batch_size]
        real = [load_rgb(os.path.join(args.target_dir, n), args.size) for n in chunk]
        fake = [load_rgb(os.path.join(args.generated_dir, n), args.size) for n in chunk]
        ssim_scores += [ssim(r, f, data_range=255, channel_axis=-1) for r, f in zip(real, fake)]
        fid.update(to_uint8_batch(real, device), real=True)
        fid.update(to_uint8_batch(fake, device), real=False)

    fid_score = fid.compute().item()
    avg_ssim = float(np.mean(ssim_scores))
    print(f"{args.model_name}: FID {fid_score:.2f} | SSIM {avg_ssim:.4f} over {len(names)} images")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    new_file = not os.path.isfile(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Model", "Generated_Dir", "Num_Images", "FID_Score", "Average_SSIM"])
        writer.writerow([args.model_name, args.generated_dir, len(names), f"{fid_score:.2f}", f"{avg_ssim:.4f}"])


if __name__ == "__main__":
    main()
