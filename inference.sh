uv run python -m scripts.sample \
    --ckpt runs/tes-default \
    --prompt "a futuristic sports car on a neon-lit street" \
    --condition data_test/canny/conditions/0.png \
    --output results/tes.png \
    --seed 123
