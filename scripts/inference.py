import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as T

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from safetensors.torch import load_file
from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.euler import EulerDiscreteScheduler
from sd.schedulers.ddim import DDIMScheduler
from sd.pipeline.sd_pipeline import StableDiffusionPipeline
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.utils.utils import load_expanded_unet

def load_condition_image(image_path, device):
    """Membaca dan memproses gambar kondisi (Canny/Depth) menjadi Tensor"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar kondisi tidak ditemukan di: {image_path}")
    
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]) 
    ])
    return transform(img).unsqueeze(0).to(device)

def main():
    parser = argparse.ArgumentParser(description="Singular Inference for MoL-SD (Standard & Early-Fusion)")
    parser.add_argument("--prompt", type=str, required=True, help="Teks prompt untuk generasi gambar")
    parser.add_argument("--ckpt", type=str, default=None, help="Path ke custom unet.pt (Abaikan untuk base SD)")
    parser.add_argument("--channels", type=int, default=4, choices=[4, 8], help="Jumlah channel input U-Net (4 untuk Standard, 8 untuk MoL)")
    parser.add_argument("--condition", type=str, default=None, help="Path ke gambar Canny/Depth (Wajib jika channels=8)")
    parser.add_argument("--steps", type=int, default=20, help="Jumlah langkah denoising")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-Free Guidance Scale (Set ke 1.0 untuk MoL)")
    parser.add_argument("--scheduler", type=str, default="DDIM", choices=["Euler", "DDIM"], help="Algoritma scheduler")
    parser.add_argument("--output", type=str, default="output.png", help="Nama file hasil output")
    args = parser.parse_args()

    if args.channels == 8 and args.condition is None:
        raise ValueError("Jika menggunakan 8-Channels, Anda WAJIB menyertakan --condition <path_gambar>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = "v1-5-pruned-emaonly.safetensors"
    
    print(f"\nMemuat base weights dari {weight_path}...")
    state_dict = load_file(weight_path)

    print("Inisialisasi Full VAE (Encoder + Decoder)...")
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(map_encoder_keys(state_dict), strict=True)
    decoder.load_state_dict(map_decoder_keys(state_dict), strict=True)

    quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1)
    quant_conv.weight.data = state_dict["first_stage_model.quant_conv.weight"].clone()
    quant_conv.bias.data = state_dict["first_stage_model.quant_conv.bias"].clone()
    post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1)
    post_quant_conv.weight.data = state_dict["first_stage_model.post_quant_conv.weight"].clone()
    post_quant_conv.bias.data = state_dict["first_stage_model.post_quant_conv.bias"].clone()

    vae = VAE(encoder, decoder, quant_conv, post_quant_conv).to(device)
    vae.eval()

    print("Inisialisasi CLIP Text Encoder...")
    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14").to(device)
    clip.eval()

    print(f"Inisialisasi U-Net ({args.channels}-Channels)...")
    unet = UNet2DConditionModel(in_channels=args.channels)
    unet_dict = map_unet_keys(state_dict)

    if args.channels == 8:
        mapped_unet_dict = {k: unet_dict[k] if k in unet_dict else v for k, v in unet.state_dict().items()}
        unet = load_expanded_unet(unet, mapped_unet_dict, device)
    else:
        unet.load_state_dict(unet_dict, strict=True)
        unet.to(device)

    if args.ckpt is not None:
        print(f"    -> Menyuntikkan custom weights dari: {args.ckpt}")
        custom_weights = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "unet_state_dict" in custom_weights:
            unet.load_state_dict(custom_weights["unet_state_dict"])
        elif "model_state_dict" in custom_weights:
            unet.load_state_dict(custom_weights["model_state_dict"])
        else:
            unet.load_state_dict(custom_weights)
    
    unet.eval()

    print(f"Mengaktifkan Scheduler: {args.scheduler}")
    if args.scheduler == "DDIM":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    else:
        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)

    print("Merakit Pipeline...")
    pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)

    gen_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": "",
        "height": 512,
        "width": 512,
        "num_inference_steps": args.steps,
        "cfg_scale": args.cfg,
        "device": device
    }

    if args.channels == 8:
        print(f"    -> Memproses gambar kondisi: {args.condition}")
        condition_tensor = load_condition_image(args.condition, device)
        gen_kwargs["condition_image"] = condition_tensor

    print(f"\nMemulai Generasi Gambar...")
    print(f"   Prompt: '{args.prompt}'")
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = pipeline.generate(**gen_kwargs)

    image.save(args.output)
    print(f"Selesai! Gambar berhasil disimpan sebagai: {args.output}\n")

if __name__ == "__main__":
    main()
