import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from safetensors.torch import load_file
from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.decoder import Decoder
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.ddim import DDIMScheduler
from sd.pipeline.sd_pipeline import StableDiffusionPipeline

from scripts.convert_weights import map_unet_keys

def map_decoder_keys(state_dict):
    """Translates the CompVis VAE keys to match our custom Decoder architecture."""
    new_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("first_stage_model.decoder."): 
            continue
            
        k = k.replace("first_stage_model.decoder.", "")
        
        # Map the bottleneck
        k = k.replace("mid.block_1", "mid_block.0")
        k = k.replace("mid.attn_1", "mid_block.1")
        k = k.replace("mid.block_2", "mid_block.2")
        
        # ==========================================================
        # Map the upsampling stages (CompVis numbers them backwards)
        # ==========================================================
        
        # Level 1 (Lowest Res: 64x64, 512 channels) -> CompVis up.3
        k = k.replace("up.3.block.0", "up_stage1.0")
        k = k.replace("up.3.block.1", "up_stage1.1")
        k = k.replace("up.3.block.2", "up_stage1.2")
        k = k.replace("up.3.upsample", "up_stage1.3")
        
        # Level 2 (128x128, 512 channels) -> CompVis up.2
        k = k.replace("up.2.block.0", "up_stage2.0")
        k = k.replace("up.2.block.1", "up_stage2.1")
        k = k.replace("up.2.block.2", "up_stage2.2")
        k = k.replace("up.2.upsample", "up_stage2.3")
        
        # Level 3 (256x256, 256 channels) -> CompVis up.1
        k = k.replace("up.1.block.0", "up_stage3.0")
        k = k.replace("up.1.block.1", "up_stage3.1")
        k = k.replace("up.1.block.2", "up_stage3.2")
        k = k.replace("up.1.upsample", "up_stage3.3")
        
        # Level 4 (Highest Res: 512x512, 128 channels) -> CompVis up.0
        k = k.replace("up.0.block.0", "up_stage4.0")
        k = k.replace("up.0.block.1", "up_stage4.1")
        k = k.replace("up.0.block.2", "up_stage4.2")
        
        # Map internal ResNet and convolution variables
        k = k.replace("in_layers.0", "norm1")
        k = k.replace("in_layers.2", "conv1")
        k = k.replace("out_layers.0", "norm2")
        k = k.replace("out_layers.3", "conv2")
        k = k.replace("upsample.conv", "conv")
        
        new_dict[k] = v
    return new_dict

class VAEWrapper(torch.nn.Module):
    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.post_quant_conv(x))

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

    vae = VAEWrapper(post_quant_conv, decoder)

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