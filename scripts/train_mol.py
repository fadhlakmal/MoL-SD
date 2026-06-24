import torch
import bitsandbytes as bnb
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.ddim import DDIMScheduler
from sd.utils.logger import WandbLogger
from sd.utils.checkpoint import CheckpointManager
from sd.data.dataset import ConditionalImageDataset
from sd.pipeline.sd_pipeline import StableDiffusionPipeline

from sd.utils.utils import load_expanded_unet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = WandbLogger(project_name="mol-sd-finetune", run_name="mol-early-fusion-20-80")
    checkpointer = CheckpointManager(save_dir="checkpoints_20_80")
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")

    unet = UNet2DConditionModel(in_channels=8).to(device)
    mapped_unet_dict = map_unet_keys(state_dict)
    unet = load_expanded_unet(unet, mapped_unet_dict, device)
    unet.train()

    # VAE & CLIP
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

    vae = VAE(encoder, decoder, quant_conv, post_quant_conv).to(device); vae.eval()
    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14").to(device); clip.eval()

    for param in vae.parameters(): param.requires_grad = False
    for param in clip.parameters(): param.requires_grad = False

    pipeline = StableDiffusionPipeline(
        vae=vae, 
        text_encoder=clip, 
        unet=unet, 
        scheduler=scheduler
    )

    # optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=1e-2)
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(), 
        lr=1e-5, 
        weight_decay=1e-2
    )

    ds_canny = ConditionalImageDataset("data/canny/targets", "data/canny/conditions", "data/prompts.txt", size=512)
    ds_depth = ConditionalImageDataset("data/depth/targets", "data/depth/conditions", "data/prompts.txt", size=512)

    dl_canny = DataLoader(ds_canny, batch_size=1, shuffle=True)
    dl_depth = DataLoader(ds_depth, batch_size=1, shuffle=True)

    val_sample_canny = ds_canny[425]
    val_cond_canny = val_sample_canny["condition"].unsqueeze(0).to(device)
    val_prompt_canny = val_sample_canny["text"]

    val_sample_depth = ds_depth[425]
    val_cond_depth = val_sample_depth["condition"].unsqueeze(0).to(device)
    val_prompt_depth = val_sample_depth["text"]

    num_epochs = 20
    global_step = 0
    save_every_n_steps = 1000

    resume_path = "checkpoints/unet_step_xxxx.pt"
    start_epoch = 0
    global_step = 0
    
    if os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}...")
        global_step = checkpointer.load(resume_path, unet, optimizer) 
        start_epoch = global_step // len(dl_canny)

    val_batch = next(iter(dl_canny))
    val_condition = val_batch["condition"].to(device)
    val_prompt = val_batch["text"][0]

    def compute_objective_loss(batch):
        targets = batch["target"].to(device)
        conditions = batch["condition"].to(device)
        prompts = batch["text"]

        with torch.no_grad():
            target_latents = vae.encode(targets) * 0.18215
            condition_latents = vae.encode(conditions) * 0.18215
            encoder_hidden_states = clip(prompts, device)
        
        noise = torch.randn_like(target_latents)
        bsz = target_latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()

        noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
        
        unet_input = torch.cat([noisy_latents, condition_latents], dim=1)
        
        # noise_pred = unet(unet_input, timesteps, encoder_hidden_states)
        # return F.mse_loss(noise_pred, noise)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise_pred = unet(unet_input, timesteps, encoder_hidden_states)
            loss = F.mse_loss(noise_pred, noise)
        return loss

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(zip(dl_canny, dl_depth), total=min(len(dl_canny), len(dl_depth)), desc=f"Epoch {epoch+1}")

        for batch_canny, batch_depth in progress_bar:
            optimizer.zero_grad()

            loss_canny = compute_objective_loss(batch_canny)
            (0.2 * loss_canny).backward() 
            
            loss_depth = compute_objective_loss(batch_depth)
            (0.8 * loss_depth).backward()

            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss = (0.2 * loss_canny.item()) + (0.8 * loss_depth.item())

            logger.log_metrics({
                "loss_canny": loss_canny.item(), 
                "loss_depth": loss_depth.item(), 
                "total_loss": total_loss
            }, step=global_step)
            
            progress_bar.set_postfix(
                total=total_loss, 
                canny=loss_canny.item(), 
                depth=loss_depth.item()
            )

            if global_step > 0 and global_step % save_every_n_steps == 0:
                checkpointer.save(unet, optimizer, global_step, total_loss)
                unet.eval()
                with torch.no_grad():
                    image_c = pipeline.generate(
                        prompt=val_prompt_canny,
                        condition_image=val_cond_canny,
                        negative_prompt="blurry, distorted, low quality",
                        height=512, width=512,
                        num_inference_steps=20, cfg_scale=7.5, device=device
                    )                    
                    image_d = pipeline.generate(
                        prompt=val_prompt_depth,
                        condition_image=val_cond_depth,
                        negative_prompt="blurry, distorted, low quality",
                        height=512, width=512,
                        num_inference_steps=20, cfg_scale=7.5, device=device
                    )                    
                    from PIL import Image
                    combined_image = Image.new('RGB', (1024, 512))
                    combined_image.paste(image_c, (0, 0))
                    combined_image.paste(image_d, (512, 0))
                    
                    caption = f"Step {global_step} | Left: Canny ({val_prompt_canny}) | Right: Depth ({val_prompt_depth})"
                    logger.log_image(combined_image, prompt=caption, step=global_step)               
                unet.train()

            global_step += 1

    logger.finish()

if __name__ == "__main__":
    train()