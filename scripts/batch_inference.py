import os
import gc
import csv
import time
import argparse
import torch
from safetensors.torch import load_file

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.euler import EulerDiscreteScheduler
from sd.schedulers.ddim import DDIMScheduler
from sd.pipelines.stable_diffusion import StableDiffusionPipeline
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.data.dataset import ConditionalImageDataset
from sd.utils.core import load_expanded_unet

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference for MoL-SD Models")
    
    parser.add_argument("--model_name", type=str, required=True, help="Label for this run (e.g., generalist_50_50)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the custom UNet checkpoint (.pt)")
    parser.add_argument("--in_channels", type=int, default=8, help="UNet input channels (8 for conditional, 4 for standard)")
    
    parser.add_argument("--target_dir", type=str, required=True, help="Path to ground truth targets")
    parser.add_argument("--condition_dir", type=str, default=None, help="Path to condition images (required if in_channels=8)")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to the text prompts file")
    
    parser.add_argument("--scheduler", type=str, choices=["Euler", "DDIM"], default="DDIM", help="Scheduler type to use")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to generate from the dataset")
    parser.add_argument("--size", type=int, default=512, help="Image size (height and width)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--output_dir", type=str, default="results", help="Base directory to save generated images")
    parser.add_argument("--output_csv", type=str, default="results/inference_speed_report.csv", help="Path to append speed metrics")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.in_channels == 8 and not args.condition_dir:
        raise ValueError("--condition_dir must be provided when --in_channels is 8.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"\n" + "="*50)
    print(f" BATCH INFERENCE: {args.model_name.upper()}")
    print(f" Checkpoint: {args.checkpoint} | Scheduler: {args.scheduler}")
    print("="*50)

    print("Loading official base weights (v1-5-pruned-emaonly.safetensors)...")
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")

    encoder_dict = map_encoder_keys(state_dict)
    decoder_dict = map_decoder_keys(state_dict)
    
    encoder = Encoder()
    encoder.load_state_dict(encoder_dict)
    decoder = Decoder()
    decoder.load_state_dict(decoder_dict)

    quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1)
    quant_conv.weight.data = state_dict["first_stage_model.quant_conv.weight"].clone()
    quant_conv.bias.data = state_dict["first_stage_model.quant_conv.bias"].clone()
    post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1)
    post_quant_conv.weight.data = state_dict["first_stage_model.post_quant_conv.weight"].clone()
    post_quant_conv.bias.data = state_dict["first_stage_model.post_quant_conv.bias"].clone()

    vae = VAE(encoder, decoder, quant_conv, post_quant_conv).to(device)
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False

    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14").to(device)
    clip.eval()
    for param in clip.parameters(): param.requires_grad = False

    print(f"Loading Test Dataset from {args.target_dir}...")
    dataset = ConditionalImageDataset(args.target_dir, args.condition_dir, args.prompts_file, size=args.size)
    num_to_generate = min(args.num_images, len(dataset))
    
    unet_dict = map_unet_keys(state_dict)
    unet = UNet2DConditionModel(in_channels=args.in_channels).to(device)
    
    if args.in_channels == 8:
        mapped_unet_dict = {}
        for k, v in unet.state_dict().items():
            mapped_unet_dict[k] = unet_dict.get(k, v)
        unet = load_expanded_unet(unet, mapped_unet_dict, device)
    else:
        unet.load_state_dict(unet_dict, strict=True)

    print(f"Injecting fine-tuned weights from {args.checkpoint}...")
    custom_weights = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if "unet_state_dict" in custom_weights:
        unet.load_state_dict(custom_weights["unet_state_dict"])
    elif "model_state_dict" in custom_weights:
        unet.load_state_dict(custom_weights["model_state_dict"])
    else:
        unet.load_state_dict(custom_weights)
        
    unet.eval()

    if args.scheduler == "Euler":
        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    else:
        scheduler = DDIMScheduler(num_train_timesteps=1000)

    run_output_dir = os.path.join(args.output_dir, args.model_name, args.scheduler)
    os.makedirs(run_output_dir, exist_ok=True)
    
    pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)
    
    print(f"Starting generation of {num_to_generate} images...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(num_to_generate):
            sample = dataset[i]
            prompt = sample["text"]
            
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": "",
                "height": args.size, 
                "width": args.size,
                "num_inference_steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "device": device
            }
            
            if args.in_channels == 8:
                gen_kwargs["condition_image"] = sample["condition"].unsqueeze(0).to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image = pipeline.generate(**gen_kwargs)
            
            filename = f"image_{i:04d}.png"
            image.save(os.path.join(run_output_dir, filename))

            if (i + 1) % 10 == 0 or (i + 1) == num_to_generate:
                print(f"  -> Generated {i + 1}/{num_to_generate} images...")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_to_generate
    
    print(f"Finished in {total_time:.2f}s ({avg_time:.2f}s per image)")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    file_exists = os.path.isfile(args.output_csv)
    
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Scheduler", "Total_Time_Sec", "Time_Per_Image_Sec"])
        writer.writerow([args.model_name, args.scheduler, f"{total_time:.2f}", f"{avg_time:.2f}"])

    del unet, pipeline, vae, clip
    gc.collect()
    torch.cuda.empty_cache()
    print("Run complete and VRAM cleared.")

if __name__ == "__main__":
    main()