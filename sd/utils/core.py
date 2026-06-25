import torch
import torch.nn as nn

def load_expanded_unet(unet: nn.Module, state_dict: dict, device: torch.device):
    pretrained_conv_in = state_dict.get("conv_in.weight")
    
    if pretrained_conv_in is None:
        raise ValueError("Could not find 'conv_in.weight'. Did you forget to run map_unet_keys() first?")

    target_channels = unet.conv_in.in_channels
    pretrained_channels = pretrained_conv_in.shape[1]
    
    if target_channels > pretrained_channels:
        print(f"Expanding UNet conv_in from {pretrained_channels} to {target_channels} channels...")
        
        expanded_weight = torch.zeros_like(unet.conv_in.weight)
        expanded_weight[:, :pretrained_channels, :, :] = pretrained_conv_in
        
        # Replace the weight in the state dictionary
        state_dict["conv_in.weight"] = expanded_weight
        
    elif target_channels == pretrained_channels:
        print(f"Loading standard {target_channels}-channel UNet weights...")
        
    else:
        print(f"Warning: Target UNet expects {target_channels} channels but checkpoint has {pretrained_channels}.")

    unet.load_state_dict(state_dict, strict=False)
    return unet.to(device)