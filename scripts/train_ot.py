import argparse
import torch
import bitsandbytes as bnb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from safetensors.torch import load_file

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.models.autoencoder.encoder import Encoder
from sd.models.autoencoder.decoder import Decoder
from sd.models.autoencoder.vae import VAE
from sd.models.text_encoder.clip import CLIPEncoder
from sd.utils.weight_mapping import map_encoder_keys, map_decoder_keys, map_unet_keys
from sd.utils.core import load_expanded_unet
from sd.data.dataset import ConditionalImageDataset

from sd.schedulers.flow_matching import FlowMatchingScheduler
from sd.engine.ot_trainer import OTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scheduler = FlowMatchingScheduler(num_train_timesteps=1000)
    state_dict = load_file("v1-5-pruned-emaonly.safetensors")

    print(f"Initializing {config.experiment_name}...")

    unet = UNet2DConditionModel(in_channels=8).to(device)
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

    vae = VAE(encoder, decoder, quant_conv, post_quant_conv).to(device)
    vae.eval()
    clip = CLIPEncoder(model_name="openai/clip-vit-large-patch14").to(device)
    clip.eval()

    for param in vae.parameters(): param.requires_grad = False
    for param in clip.parameters(): param.requires_grad = False

    dataloaders = {}
    for task_name, task_cfg in config.data.tasks.items():
        print(f"Loading dataset for task: {task_name}")
        ds = ConditionalImageDataset(
            target_dir=task_cfg.target_dir,
            condition_dir=task_cfg.cond_dir,
            prompt_file=config.data.prompts_file,
            size=config.data.size
        )
        dataloaders[task_name] = DataLoader(ds, batch_size=config.data.batch_size, shuffle=True)

    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )

    trainer = OTTrainer(
        config=config,
        unet=unet,
        vae=vae,
        clip=clip,
        scheduler=scheduler,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device
    )

    trainer.train()

if __name__ == "__main__":
    main()