import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.models.text_encoder.clip import CLIPEncoder
from sd.schedulers.ddim import DDIMScheduler
from sd.utils.logger import WandbLogger
from sd.utils.checkpoint import CheckpointManager
from sd.pipeline.sd_pipeline import StableDiffusionPipeline
from sd.data.dataset import ConditionalImageDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = WandbLogger(project_name="mol-sd-finetune", run_name="sd-standard-baseline-run")
    checkpointer = CheckpointManager(save_dir="checkpoints_standard_baseline")
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    state_dict = load_file("v1-5-pruned-emaonly.safetensors")
    
    unet = UNet2DConditionModel(in_channels=4)

    encoder_dict = map_encoder_keys(state_dict)
    decoder_dict = map_decoder_keys(state_dict)
    unet_dict = map_unet_keys(state_dict)

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

    clip = CLIPEncoder().to(device)
    clip.eval()

    for param in vae.parameters(): param.requires_grad = False
    for param in clip.parameters(): param.requires_grad = False

    mapped_unet_dict = {}
    for k, v in unet.state_dict().items():
        if k in unet_dict:
            mapped_unet_dict[k] = unet_dict[k]
        else:
            mapped_unet_dict[k] = v

    unet.load_state_dict(mapped_unet_dict)
    unet = unet.to(device)
    unet.train()

    pipeline = StableDiffusionPipeline(vae, clip, unet, scheduler)

    train_dataset = ConditionalImageDataset("data/canny/targets", "data/canny/conditions", "data/prompts.txt", size=512)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_sample = train_dataset[425]
    val_prompt = val_sample["text"]

    learning_rate = 1e-5
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate, weight_decay=1e-2)

    num_epochs = 20
    save_every_n_steps = 1000
    global_step = 0

    print("Starting Standard 4-Channel Baseline Training Loop...")
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()

            targets = batch["target"].to(device)
            encoder_hidden_states = batch["text"]

            with torch.no_grad():
                latents = vae.encode(targets) * 0.18215
                encoder_hidden_states = clip(encoder_hidden_states, device)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            unet_input = noisy_latents
            unet_input.requires_grad_(True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                noise_pred = checkpoint.checkpoint(
                    unet,
                    unet_input,
                    timesteps,
                    encoder_hidden_states,
                    use_reentrant=False
                )
                loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            logger.log_metrics({"train_loss": loss.item(), "lr": learning_rate}, step=global_step)
            progress_bar.set_postfix(loss=loss.item())

            if global_step > 0 and global_step % save_every_n_steps == 0:
                checkpointer.save(unet, optimizer, global_step, loss.item())
                
                unet.eval()
                with torch.no_grad():
                    image = pipeline.generate(
                        prompt=val_prompt,
                        negative_prompt="blurry, distorted, low quality, bad composition",
                        height=512, 
                        width=512,
                        num_inference_steps=20, 
                        cfg_scale=7.5,
                        device=device
                    )
                    logger.log_image(image, prompt=f"Standard Baseline Step {global_step} | {val_prompt}", step=global_step)
                unet.train()

            global_step += 1
            
    logger.finish()

if __name__ == "__main__":
    train()