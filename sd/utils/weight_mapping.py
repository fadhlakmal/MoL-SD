def map_unet_keys(state_dict):
    new_dict = {}
    
    for key, weight in state_dict.items():
        if not key.startswith("model.diffusion_model."):
            continue
            
        k = key.replace("model.diffusion_model.", "")
        
        k = k.replace("input_blocks.0.0", "conv_in")
        k = k.replace("time_embed.0", "time_embedding.linear_1")
        k = k.replace("time_embed.2", "time_embedding.linear_2")
        
        k = k.replace("input_blocks.1.0", "down_blocks_0_resnet1")
        k = k.replace("input_blocks.1.1", "down_blocks_0_attn1")
        k = k.replace("input_blocks.2.0", "down_blocks_0_resnet2")
        k = k.replace("input_blocks.2.1", "down_blocks_0_attn2")
        k = k.replace("input_blocks.3.0", "down_blocks_0_down")
        
        k = k.replace("input_blocks.4.0", "down_blocks_1_resnet1")
        k = k.replace("input_blocks.4.1", "down_blocks_1_attn1")
        k = k.replace("input_blocks.5.0", "down_blocks_1_resnet2")
        k = k.replace("input_blocks.5.1", "down_blocks_1_attn2")
        k = k.replace("input_blocks.6.0", "down_blocks_1_down")
        
        k = k.replace("input_blocks.7.0", "down_blocks_2_resnet1")
        k = k.replace("input_blocks.7.1", "down_blocks_2_attn1")
        k = k.replace("input_blocks.8.0", "down_blocks_2_resnet2")
        k = k.replace("input_blocks.8.1", "down_blocks_2_attn2")
        k = k.replace("input_blocks.9.0", "down_blocks_2_down")
        
        k = k.replace("input_blocks.10.0", "down_blocks_3_resnet1")
        k = k.replace("input_blocks.11.0", "down_blocks_3_resnet2")
        
        k = k.replace("middle_block.0", "mid_block_resnet1")
        k = k.replace("middle_block.1", "mid_block_attn")
        k = k.replace("middle_block.2", "mid_block_resnet2")
        
        k = k.replace("output_blocks.0.0", "up_blocks_0_resnet1")
        k = k.replace("output_blocks.1.0", "up_blocks_0_resnet2")
        k = k.replace("output_blocks.2.0", "up_blocks_0_resnet3")
        k = k.replace("output_blocks.2.1", "up_blocks_0_up")
        
        k = k.replace("output_blocks.3.0", "up_blocks_1_resnet1")
        k = k.replace("output_blocks.3.1", "up_blocks_1_attn1")
        k = k.replace("output_blocks.4.0", "up_blocks_1_resnet2")
        k = k.replace("output_blocks.4.1", "up_blocks_1_attn2")
        k = k.replace("output_blocks.5.0", "up_blocks_1_resnet3")
        k = k.replace("output_blocks.5.1", "up_blocks_1_attn3")
        k = k.replace("output_blocks.5.2", "up_blocks_1_up")
        
        k = k.replace("output_blocks.6.0", "up_blocks_2_resnet1")
        k = k.replace("output_blocks.6.1", "up_blocks_2_attn1")
        k = k.replace("output_blocks.7.0", "up_blocks_2_resnet2")
        k = k.replace("output_blocks.7.1", "up_blocks_2_attn2")
        k = k.replace("output_blocks.8.0", "up_blocks_2_resnet3")
        k = k.replace("output_blocks.8.1", "up_blocks_2_attn3")
        k = k.replace("output_blocks.8.2", "up_blocks_2_up")

        k = k.replace("output_blocks.9.0", "up_blocks_3_resnet1")
        k = k.replace("output_blocks.9.1", "up_blocks_3_attn1")
        k = k.replace("output_blocks.10.0", "up_blocks_3_resnet2")
        k = k.replace("output_blocks.10.1", "up_blocks_3_attn2")
        k = k.replace("output_blocks.11.0", "up_blocks_3_resnet3")
        k = k.replace("output_blocks.11.1", "up_blocks_3_attn3")
        
        k = k.replace("out.0", "conv_norm_out")
        k = k.replace("out.2", "conv_out")
        
        k = k.replace("in_layers.0", "norm1")
        k = k.replace("in_layers.2", "conv1")
        k = k.replace("emb_layers.1", "time_emb_proj")
        k = k.replace("out_layers.0", "norm2")
        k = k.replace("out_layers.3", "conv2")
        k = k.replace("skip_connection", "conv_shortcut")
        
        k = k.replace("norm3", "ff.0")
        k = k.replace("ff.net.0.proj", "ff.1")
        k = k.replace("ff.net.2", "ff.3")
        k = k.replace("to_conv_norm_out", "to_out.0")
        
        k = k.replace(".op.", ".conv.")
        
        new_dict[k] = weight
        
    return new_dict

def map_decoder_keys(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("first_stage_model.decoder."): 
            continue
            
        k = k.replace("first_stage_model.decoder.", "")
        
        k = k.replace("mid.block_1", "mid_block.0")
        k = k.replace("mid.attn_1", "mid_block.1")
        k = k.replace("mid.block_2", "mid_block.2")

        k = k.replace("up.3.block.0", "up_stage1.0")
        k = k.replace("up.3.block.1", "up_stage1.1")
        k = k.replace("up.3.block.2", "up_stage1.2")
        k = k.replace("up.3.upsample", "up_stage1.3")
        
        k = k.replace("up.2.block.0", "up_stage2.0")
        k = k.replace("up.2.block.1", "up_stage2.1")
        k = k.replace("up.2.block.2", "up_stage2.2")
        k = k.replace("up.2.upsample", "up_stage2.3")
        
        k = k.replace("up.1.block.0", "up_stage3.0")
        k = k.replace("up.1.block.1", "up_stage3.1")
        k = k.replace("up.1.block.2", "up_stage3.2")
        k = k.replace("up.1.upsample", "up_stage3.3")
        
        k = k.replace("up.0.block.0", "up_stage4.0")
        k = k.replace("up.0.block.1", "up_stage4.1")
        k = k.replace("up.0.block.2", "up_stage4.2")
        
        k = k.replace("in_layers.0", "norm1")
        k = k.replace("in_layers.2", "conv1")
        k = k.replace("out_layers.0", "norm2")
        k = k.replace("out_layers.3", "conv2")
        k = k.replace("upsample.conv", "conv")
        
        new_dict[k] = v
    return new_dict

def map_encoder_keys(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("first_stage_model.encoder."): 
            continue
            
        k = k.replace("first_stage_model.encoder.", "")
        
        k = k.replace("mid.block_1", "mid_block.0")
        k = k.replace("mid.attn_1", "mid_block.1")
        k = k.replace("mid.block_2", "mid_block.2")
        
        k = k.replace("down.0.block.0", "down_stage1.0")
        k = k.replace("down.0.block.1", "down_stage1.1")
        k = k.replace("down.0.downsample", "down_stage1.2")
        
        k = k.replace("down.1.block.0", "down_stage2.0")
        k = k.replace("down.1.block.1", "down_stage2.1")
        k = k.replace("down.1.downsample", "down_stage2.2")
        
        k = k.replace("down.2.block.0", "down_stage3.0")
        k = k.replace("down.2.block.1", "down_stage3.1")
        k = k.replace("down.2.downsample", "down_stage3.2")
        
        k = k.replace("down.3.block.0", "down_stage4.0")
        k = k.replace("down.3.block.1", "down_stage4.1")
        
        k = k.replace("in_layers.0", "norm1")
        k = k.replace("in_layers.2", "conv1")
        k = k.replace("out_layers.0", "norm2")
        k = k.replace("out_layers.3", "conv2")
        k = k.replace("downsample.conv", "conv")
        
        new_dict[k] = v
    return new_dict
