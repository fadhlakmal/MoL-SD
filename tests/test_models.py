import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel as DiffusersUNet
from diffusers.models.embeddings import get_timestep_embedding as diffusers_embedding

from molsd.models.loader import expand_conv_in
from molsd.models.unet.blocks import get_timestep_embedding
from molsd.models.unet.unet_2d import UNet2DConditionModel
from tests.conftest import TINY_UNET, tiny_vae


def reference_tiny_unet(in_channels: int = 4) -> DiffusersUNet:
    return DiffusersUNet(
        sample_size=16,
        in_channels=in_channels,
        out_channels=4,
        block_out_channels=TINY_UNET.block_out_channels,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=TINY_UNET.layers_per_block,
        attention_head_dim=TINY_UNET.num_attention_heads,
        cross_attention_dim=TINY_UNET.context_dim,
        norm_num_groups=TINY_UNET.norm_num_groups,
    ).eval()


def test_timestep_embedding_matches_sd15():
    t = torch.tensor([0.0, 1.0, 37.5, 500.0, 999.0])
    ours = get_timestep_embedding(t, 320)
    ref = diffusers_embedding(t, 320, flip_sin_to_cos=True, downscale_freq_shift=0)
    torch.testing.assert_close(ours, ref)


def test_unet_parity_tiny():
    torch.manual_seed(0)
    ref = reference_tiny_unet()
    ours = UNet2DConditionModel(TINY_UNET).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)

    x = torch.randn(2, 4, 16, 16)
    ctx = torch.randn(2, 5, TINY_UNET.context_dim)
    t = torch.tensor([3.0, 871.25])
    with torch.no_grad():
        torch.testing.assert_close(ours(x, t, ctx), ref(x, t, ctx).sample, atol=1e-5, rtol=0)


def test_vae_parity_tiny():
    torch.manual_seed(0)
    ref = AutoencoderKL(
        block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        layers_per_block=1,
        norm_num_groups=8,
    ).eval()
    ours = tiny_vae().eval()
    ours.load_state_dict(ref.state_dict(), strict=True)

    img = torch.randn(1, 3, 32, 32)
    z = torch.randn(1, 4, 16, 16)
    with torch.no_grad():
        mean_ref = ref.encode(img).latent_dist.mean
        torch.testing.assert_close(ours.encode(img, sample=False), mean_ref * ours.scaling_factor, atol=1e-5, rtol=0)
        torch.testing.assert_close(ours.decode(z * ours.scaling_factor), ref.decode(z).sample, atol=1e-5, rtol=0)


def test_conv_in_expansion_ignores_condition():
    torch.manual_seed(0)
    base = UNet2DConditionModel(TINY_UNET).eval()
    expanded = UNet2DConditionModel(TINY_UNET, in_channels=8).eval()
    expanded.load_state_dict(expand_conv_in(base.state_dict(), 8), strict=True)

    x = torch.randn(1, 4, 16, 16)
    cond = torch.randn(1, 4, 16, 16)
    ctx = torch.randn(1, 5, TINY_UNET.context_dim)
    with torch.no_grad():
        torch.testing.assert_close(expanded(torch.cat([x, cond], 1), 500, ctx), base(x, 500, ctx))


def test_gradient_checkpointing_same_grads():
    torch.manual_seed(0)
    a = UNet2DConditionModel(TINY_UNET).train()
    b = UNet2DConditionModel(TINY_UNET).train()
    b.load_state_dict(a.state_dict())
    b.enable_gradient_checkpointing()

    x, ctx, t = torch.randn(2, 4, 16, 16), torch.randn(2, 5, TINY_UNET.context_dim), torch.tensor([10.0, 900.0])
    a(x, t, ctx).square().mean().backward()
    b(x, t, ctx).square().mean().backward()
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, msg=name)
