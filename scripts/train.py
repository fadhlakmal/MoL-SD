import torch

from sd.utils.utils import load_expanded_unet

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

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
from sd.pipeline.sd_pipeline import StableDiffusionPipeline
from sd.data.dataset import ImageTextDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init
    logger = WandbLogger(project_name="mol-sd-finetune", run_name="unet-run-01")
    checkpointer = CheckpointManager(save_dir="checkpoints")
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # model
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")
    unet = UNet2DConditionModel(in_channels=4).to(device)
    mapped_unet_dict = map_unet_keys(state_dict) 
    unet = load_expanded_unet(unet, mapped_unet_dict, device)
    unet.train()

    
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

    for param in vae.parameters():
        param.requires_grad = False
    for param in clip.parameters():
        param.requires_grad = False

    pipeline = StableDiffusionPipeline(
        vae=vae, 
        text_encoder=clip, 
        unet=unet, 
        scheduler=scheduler
    )

    # optim & hyperparams
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    num_epochs = 100
    save_every_n_steps = 50
    global_step = 0

    # dataset
    dataset = ImageTextDataset(
        data_dir="dummy_data", 
        metadata_file="metadata.json", 
        size=512
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            images = batch["image"].to(device) # (B, 3, 512, 512)
            prompts = batch["text"]
            optimizer.zero_grad()

            with torch.no_grad():
                latents = vae.encode(images)
                latents = latents * 0.18215
                encoder_hidden_states = clip(prompts, device)
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)

            loss = F.mse_loss(noise_pred, noise)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            # log & ckpt
            logger.log_metrics({"train_loss": loss.item(), "lr": learning_rate}, step=global_step)
            progress_bar.set_postfix(loss=loss.item())

            if global_step > 0 and global_step % save_every_n_steps == 0:
                checkpointer.save(unet, optimizer, global_step, loss.item())
                validation_prompts = [
                    "A highly detailed, cinematic photograph of a futuristic city at sunset, neon lights, 8k resolution, unreal engine 5"
                ]
                unet.eval()
                with torch.no_grad():
                    for val_prompt in validation_prompts:
                        image = pipeline.generate(
                            prompt=val_prompt,
                            negative_prompt="blurry, distorted, low quality, bad composition",
                            height=512, 
                            width=512,
                            num_inference_steps=20, 
                            cfg_scale=7.5,
                            device=device
                        )
                        logger.log_image(image, prompt=val_prompt, step=global_step)                
                unet.train()
            global_step += 1
            
    logger.finish()

if __name__ == "__main__":
    train()
    #pass