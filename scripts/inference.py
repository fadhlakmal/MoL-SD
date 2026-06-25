import os
import argparse
import torch
from PIL import Image
from safetensors.torch import load_file

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.ddim import DDIMScheduler
from sd.pipelines.stable_diffusion import StableDiffusionPipeline
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.utils.core import load_expanded_unet

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference with trained MoL-SD checkpoints.")
    
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--condition_image", type=str, required=True, help="Path to the condition image (e.g., Canny/Depth)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained UNet checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image")
    
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, low quality", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-Free Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--size", type=int, default=512, help="Image generation height/width")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print(f"Loading base Stable Diffusion weights...")
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")

    print(f"Loading custom UNet checkpoint from {args.checkpoint}...")
    unet = UNet2DConditionModel(in_channels=8).to(device)
    mapped_unet_dict = map_unet_keys(state_dict)
    unet = load_expanded_unet(unet, mapped_unet_dict, device)
    
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    unet.eval()

    encoder = Encoder()
    encoder.load_state_dict(map_encoder_keys(state_dict), strict=True)
    decoder = Decoder()
    decoder.load_state_dict(map_decoder_keys(state_dict), strict=True)
    
    quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1)
    quant_conv.weight.data = state_dict["first_stage_model.quant_conv.weight"].clone()
    quant_conv.bias.data = state_dict["first_stage_model.quant_conv.bias"].clone()
    post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1)
    post_quant_conv.weight.data = state_dict["first_stage_model.post_quant_conv.weight"].clone()
    post_quant_conv.bias.data = state_dict["first_stage_model.post_quant_conv.bias"].clone()

    vae = VAE(encoder, decoder, quant_conv, post_quant_conv).to(device)
    vae.eval()

    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14").to(device)
    clip.eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)

    print(f"Processing condition image: {args.condition_image}")
    if not os.path.exists(args.condition_image):
        raise FileNotFoundError(f"Condition image not found: {args.condition_image}")
        
    raw_image = Image.open(args.condition_image).convert("RGB").resize((args.size, args.size))
    import numpy as np
    cond_tensor = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    cond_tensor = cond_tensor.to(device)

    print(f"Generating image for prompt: '{args.prompt}'...")
    with torch.no_grad():
        output_image = pipeline.generate(
            prompt=args.prompt,
            condition_image=cond_tensor,
            negative_prompt=args.negative_prompt,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            device=device
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_image.save(args.output)
    print(f"Success! Image saved to {args.output}")

if __name__ == "__main__":
    main()