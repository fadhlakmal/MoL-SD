"""One-time conversion of SD1.5 weights into this repo's format.

Our module names match diffusers' state dicts, so conversion is: load with diffusers, save the state dict.

    uv run python -m scripts.convert_sd15                                   # from the HF hub
    uv run python -m scripts.convert_sd15 --single_file v1-5-pruned-emaonly.safetensors
"""

import argparse
import os

from diffusers import AutoencoderKL, UNet2DConditionModel
from safetensors.torch import save_file

from molsd.models.autoencoder.vae import VAE
from molsd.models.unet.unet_2d import UNet2DConditionModel as OurUNet

HUB_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_file", default=None, help="Original LDM .safetensors/.ckpt; default: download from hub")
    parser.add_argument("--repo", default=HUB_REPO)
    parser.add_argument("--out_dir", default="weights")
    args = parser.parse_args()

    if args.single_file:
        unet = UNet2DConditionModel.from_single_file(args.single_file, config=args.repo, subfolder="unet")
        vae = AutoencoderKL.from_single_file(args.single_file, config=args.repo, subfolder="vae")
    else:
        unet = UNet2DConditionModel.from_pretrained(args.repo, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(args.repo, subfolder="vae")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ref, ours in (("unet", unet, OurUNet()), ("vae", vae, VAE())):
        sd = {k: v.contiguous() for k, v in ref.state_dict().items()}
        ours.load_state_dict(sd, strict=True)  # fail here, not at train time, if names drift
        path = os.path.join(args.out_dir, f"sd15_{name}.safetensors")
        save_file(sd, path)
        print(f"wrote {path} ({sum(v.numel() for v in sd.values()) / 1e6:.0f}M params)")


if __name__ == "__main__":
    main()
