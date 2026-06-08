UNET_REPLACE_MAP = {
    "model.diffusion_model.": "",
    "input_blocks.0.0": "conv_in",
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "input_blocks.1.0": "down_blocks_0_resnet1",
    "input_blocks.1.1": "down_blocks_0_attn1",
    "input_blocks.2.0": "down_blocks_0_resnet2",
    "input_blocks.2.1": "down_blocks_0_attn2",
    "input_blocks.3.0": "down_blocks_0_down",
    "input_blocks.4.0": "down_blocks_1_resnet1",
    "input_blocks.4.1": "down_blocks_1_attn1",
    "input_blocks.5.0": "down_blocks_1_resnet2",
    "input_blocks.5.1": "down_blocks_1_attn2",
    "input_blocks.6.0": "down_blocks_1_down",
    "input_blocks.7.0": "down_blocks_2_resnet1",
    "input_blocks.7.1": "down_blocks_2_attn1",
    "input_blocks.8.0": "down_blocks_2_resnet2",
    "input_blocks.8.1": "down_blocks_2_attn2",
    "input_blocks.9.0": "down_blocks_2_down",
    "input_blocks.10.0": "down_blocks_3_resnet1",
    "input_blocks.11.0": "down_blocks_3_resnet2",
    "middle_block.0": "mid_block_resnet1",
    "middle_block.1": "mid_block_attn",
    "middle_block.2": "mid_block_resnet2",
    "output_blocks.0.0": "up_blocks_0_resnet1",
    "output_blocks.1.0": "up_blocks_0_resnet2",
    "output_blocks.2.0": "up_blocks_0_resnet3",
    "output_blocks.2.1": "up_blocks_0_up",
    "output_blocks.3.0": "up_blocks_1_resnet1",
    "output_blocks.3.1": "up_blocks_1_attn1",
    "output_blocks.4.0": "up_blocks_1_resnet2",
    "output_blocks.4.1": "up_blocks_1_attn2",
    "output_blocks.5.0": "up_blocks_1_resnet3",
    "output_blocks.5.1": "up_blocks_1_attn3",
    "output_blocks.5.2": "up_blocks_1_up",
    "output_blocks.6.0": "up_blocks_2_resnet1",
    "output_blocks.6.1": "up_blocks_2_attn1",
    "output_blocks.7.0": "up_blocks_2_resnet2",
    "output_blocks.7.1": "up_blocks_2_attn2",
    "output_blocks.8.0": "up_blocks_2_resnet3",
    "output_blocks.8.1": "up_blocks_2_attn3",
    "output_blocks.8.2": "up_blocks_2_up",
    "output_blocks.9.0": "up_blocks_3_resnet1",
    "output_blocks.9.1": "up_blocks_3_attn1",
    "output_blocks.10.0": "up_blocks_3_resnet2",
    "output_blocks.10.1": "up_blocks_3_attn2",
    "output_blocks.11.0": "up_blocks_3_resnet3",
    "output_blocks.11.1": "up_blocks_3_attn3",
    "out.0": "conv_norm_out",
    "out.2": "conv_out",
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "emb_layers.1": "time_emb_proj",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "skip_connection": "conv_shortcut",
    "norm3": "ff.0",
    "ff.net.0.proj": "ff.1",
    "ff.net.2": "ff.3",
    "to_conv_norm_out": "to_out.0",
    ".op.": ".conv."
}

DECODER_REPLACE_MAP = {
    "first_stage_model.decoder.": "",
    "mid.block_1": "mid_block.0",
    "mid.attn_1": "mid_block.1",
    "mid.block_2": "mid_block.2",
    "up.3.block.0": "up_stage1.0",
    "up.3.block.1": "up_stage1.1",
    "up.3.block.2": "up_stage1.2",
    "up.3.upsample": "up_stage1.3",
    "up.2.block.0": "up_stage2.0",
    "up.2.block.1": "up_stage2.1",
    "up.2.block.2": "up_stage2.2",
    "up.2.upsample": "up_stage2.3",
    "up.1.block.0": "up_stage3.0",
    "up.1.block.1": "up_stage3.1",
    "up.1.block.2": "up_stage3.2",
    "up.1.upsample": "up_stage3.3",
    "up.0.block.0": "up_stage4.0",
    "up.0.block.1": "up_stage4.1",
    "up.0.block.2": "up_stage4.2",
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "upsample.conv": "conv"
}

ENCODER_REPLACE_MAP = {
    "first_stage_model.encoder.": "",
    "mid.block_1": "mid_block.0",
    "mid.attn_1": "mid_block.1",
    "mid.block_2": "mid_block.2",
    "down.0.block.0": "down_stage1.0",
    "down.0.block.1": "down_stage1.1",
    "down.0.downsample": "down_stage1.2",
    "down.1.block.0": "down_stage2.0",
    "down.1.block.1": "down_stage2.1",
    "down.1.downsample": "down_stage2.2",
    "down.2.block.0": "down_stage3.0",
    "down.2.block.1": "down_stage3.1",
    "down.2.downsample": "down_stage3.2",
    "down.3.block.0": "down_stage4.0",
    "down.3.block.1": "down_stage4.1",
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "downsample.conv": "conv"
}

def _apply_mapping(state_dict, prefix, replace_map):
    new_dict = {}
    for key, weight in state_dict.items():
        if not key.startswith(prefix):
            continue
            
        new_key = key
        for old_str, new_str in replace_map.items():
            if old_str in new_key:
                new_key = new_key.replace(old_str, new_str)
                
        new_dict[new_key] = weight
    return new_dict

def map_unet_keys(state_dict):
    return _apply_mapping(state_dict, "model.diffusion_model.", UNET_REPLACE_MAP)

def map_decoder_keys(state_dict):
    return _apply_mapping(state_dict, "first_stage_model.decoder.", DECODER_REPLACE_MAP)

def map_encoder_keys(state_dict):
    return _apply_mapping(state_dict, "first_stage_model.encoder.", ENCODER_REPLACE_MAP)