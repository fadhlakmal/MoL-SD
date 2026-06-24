import os
import gc
import time
import torch
import torch.nn.functional as F
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
from sd.pipeline.sd_pipeline import StableDiffusionPipeline
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.data.dataset import ConditionalImageDataset
from sd.utils.utils import load_expanded_unet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_IMAGES = 50
    
    MODELS = [
        {"name": "standard_baseline", "path": "checkpoints_standard_baseline/unet_step_9000.pt", "channels": 4},
        {"name": "canny_specialist", "path": "checkpoints_baseline/unet_step_9000.pt", "channels": 8},
        {"name": "generalist_50_50", "path": "checkpoints_50_50/unet_step_9000.pt", "channels": 8},
        {"name": "generalist_80_20", "path": "checkpoints_80_20/unet_step_9000.pt", "channels": 8},
        {"name": "generalist_20_80", "path": "checkpoints_20_80/unet_step_9000.pt", "channels": 8},
    ]

    SCHEDULERS = {
        "Euler": (EulerDiscreteScheduler(num_train_timesteps=1000), 20),
        "DDIM": (DDIMScheduler(num_train_timesteps=1000), 20)
    }

    print("Loading official base weights (v1-5-pruned-emaonly.safetensors)...")
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")

    print("Initializing Frozen Components (VAE & CLIP)...")
    encoder_dict = map_encoder_keys(state_dict)
    decoder_dict = map_decoder_keys(state_dict)
    
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

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

    print("Loading Test Dataset...")
    dataset = ConditionalImageDataset("data_test/canny/targets", "data_test/canny/conditions", "data_test/prompts.txt", size=512)
    test_indices = range(len(dataset))
    
    unet_dict = map_unet_keys(state_dict)

    speed_metrics_log = [("Model", "Scheduler", "Total_Time_Sec", "Time_Per_Image_Sec")]

    for model_cfg in MODELS:
        run_name = model_cfg["name"]
        ckpt_path = model_cfg["path"]
        in_channels = model_cfg["channels"]
        
        print(f"\n" + "="*50)
        print(f" LOADING MODEL: {run_name.upper()}")
        print(f" Channels: {in_channels} | Path: {ckpt_path}")
        print("="*50)

        unet = UNet2DConditionModel(in_channels=in_channels)
        
        if in_channels == 8:
            mapped_unet_dict = {}
            for k, v in unet.state_dict().items():
                if k in unet_dict:
                    mapped_unet_dict[k] = unet_dict[k]
                else:
                    mapped_unet_dict[k] = v
            unet = load_expanded_unet(unet, mapped_unet_dict, device)
        else:
            unet.load_state_dict(unet_dict, strict=True)
            unet.to(device)

        print(f"Injecting fine-tuned weights from step 9000...")
        custom_weights = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        if "unet_state_dict" in custom_weights:
            unet.load_state_dict(custom_weights["unet_state_dict"])
        elif "model_state_dict" in custom_weights:
            unet.load_state_dict(custom_weights["model_state_dict"])
        else:
            unet.load_state_dict(custom_weights)
            
        unet.eval()

        for sched_name, (scheduler, steps) in SCHEDULERS.items():
            print(f"\n---> Running Architecture Study: {sched_name} Scheduler ({steps} steps)")
            
            output_dir = os.path.join("results", run_name, sched_name)
            os.makedirs(output_dir, exist_ok=True)
            
            pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)
            start_time = time.time()

            with torch.no_grad():
                for i, dataset_idx in enumerate(test_indices):
                    sample = dataset[dataset_idx]
                    prompt = sample["text"]
                    
                    gen_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": "",
                        "height": 512, 
                        "width": 512,
                        "num_inference_steps": steps,
                        "cfg_scale": 1.0,
                        "device": device
                    }
                    
                    if in_channels == 8:
                        condition_tensor = sample["condition"].unsqueeze(0).to(device)
                        gen_kwargs["condition_image"] = condition_tensor
                    
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image = pipeline.generate(**gen_kwargs)
                    
                    filename = f"image_{i:04d}.png"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath)

                    if (i + 1) % 10 == 0:
                        print(f"  -> Generated {i + 1}/{NUM_IMAGES} images...")

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / NUM_IMAGES
            
            speed_metrics_log.append((run_name, sched_name, f"{total_time:.2f}", f"{avg_time:.2f}"))
            print(f"⏱️ Finished in {total_time:.2f}s ({avg_time:.2f}s per image)")
            print(f"Saved {NUM_IMAGES} images to '{output_dir}'.")

        print(f"Cleaning VRAM cache for {run_name}...")
        del unet
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    import csv
    report_path = "results/inference_speed_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(speed_metrics_log)
        
    print("\n" + "="*50)
    print(f"📊 Speed metrics saved to: {report_path}")
    print("✅ ALL BATCH INFERENCE RUNS COMPLETED SUCCESSFULLY.")
    print("="*50)

if __name__ == "__main__":
    main()