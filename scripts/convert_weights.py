import torch
from safetensors.torch import load_file
from sd.models.unet.unet_2d import UNet2DConditionModel

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

def test(safetensors_path: str):
    custom_unet = UNet2DConditionModel()
    official_state_dict = load_file(safetensors_path)
    translated_dict = map_unet_keys(official_state_dict)
    
    try:
        custom_unet.load_state_dict(translated_dict, strict=True)
        print("success: Architecture matches official weights")
    except Exception as e:
        print("fail: Architecture mismatch.")
        print(e)

if __name__ == "__main__":
    test("v1-5-pruned-emaonly.safetensors")