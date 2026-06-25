python scripts/inference.py \
    --prompt "a futuristic sports car on a neon-lit street" \
    --condition_image data_test/canny/conditions/0.png \
    --checkpoint checkpoints/unet_step_10000.pt \
    --output results/tes.png \
    --seed 123