import torch
from safetensors.torch import load_file
from sd.models.unet.unet_2d import UNet2DConditionModel
from sd.utils.checkpoint import map_unet_keys

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
