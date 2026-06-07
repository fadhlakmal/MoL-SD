import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from safetensors.torch import load_file
from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAEDecoder
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.ddim import DDIMScheduler
from sd.pipeline.sd_pipeline import StableDiffusionPipeline
from sd.utils.weight_mapping import map_unet_keys, map_decoder_keys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = "v1-5-pruned-emaonly.safetensors"
    
    print(f"Loading official weights from {weight_path}...")
    state_dict = load_file(weight_path)

    print("Initializing U-Net...")
    unet = UNet2DConditionModel()
    unet.load_state_dict(map_unet_keys(state_dict), strict=True)
    unet.to(device)

    print("Initializing VAE Decoder...")
    decoder = Decoder()
    decoder.load_state_dict(map_decoder_keys(state_dict), strict=True)
    decoder.to(device)

    print("Initializing VAE Post-Quant Conv...")
    post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1)
    post_quant_conv.weight.data = state_dict["first_stage_model.post_quant_conv.weight"].clone()
    post_quant_conv.bias.data = state_dict["first_stage_model.post_quant_conv.bias"].clone()
    post_quant_conv.to(device)

    vae = VAEDecoder(post_quant_conv, decoder)

    print("Initializing CLIP Text Encoder...")
    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14")
    clip.to(device)

    print("Initializing DDIM Scheduler...")
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    print("Assembling Pipeline...")
    pipeline = StableDiffusionPipeline(
        vae=vae, 
        text_encoder=clip, 
        unet=unet, 
        scheduler=scheduler
    )

    prompt = "A highly detailed, cinematic photograph of a futuristic city at sunset, neon lights, 8k resolution, unreal engine 5"
    negative_prompt = "blurry, low quality, distorted, watermark"
    
    print(f"\nPrompt: '{prompt}'")
    
    image = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        cfg_scale=7.5,
        seed=42,
        device=device
    )

    output_filename = "test.png"
    image.save(output_filename)
    print(f"\nSuccess: Image saved to {output_filename}")

if __name__ == "__main__":
    main()
