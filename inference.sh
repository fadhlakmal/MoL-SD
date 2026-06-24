uv run python -m scripts.inference \
  --prompt "A person on a motor bike on a street." \
  --ckpt "checkpoints_50_50/unet_step_9000.pt" \
  --channels 8 \
  --condition "data/depth/conditions/441.png" \
  --scheduler DDIM \
  --output "out_depth.png"