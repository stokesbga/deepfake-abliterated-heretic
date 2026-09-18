import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from deepfakeheretic._vendor.wan.model import WanModel
from deepfakeheretic._vendor.wan.solver import FlowUniPCMultistepScheduler
from deepfakeheretic.attention import flash_attention
from deepfakeheretic.engine import inference_precision


def test_attention_matches_reference_and_ignores_padding():
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 7, 2, 8) for _ in range(3)]
    result = flash_attention(q, k, v, q_lens=[5, 7], k_lens=[3, 6], softmax_scale=0.7, q_scale=2)
    for index, (nq, nk) in enumerate([(5, 3), (7, 6)]):
        qi = q[index, :nq].transpose(0, 1)
        ki = k[index, :nk].transpose(0, 1)
        vi = v[index, :nk].transpose(0, 1)
        expected = ((qi * 2 @ ki.transpose(-2, -1)) * 0.7).softmax(-1) @ vi
        torch.testing.assert_close(result[index, :nq], expected.transpose(0, 1))
    assert torch.count_nonzero(result[0, 5:]) == 0
    k[:, 6:] = 1e6
    v[:, 6:] = 1e6
    changed = flash_attention(q, k, v, q_lens=[5, 7], k_lens=[3, 6], softmax_scale=0.7, q_scale=2)
    torch.testing.assert_close(result, changed)


def tiny_model():
    return WanModel(
        model_type="i2v",
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        in_dim=12,
        out_dim=4,
        in_dim_ref_conv=4,
        num_heads=2,
        num_layers=1,
        text_len=8,
        cross_attn_norm=True,
    )


@pytest.mark.parametrize("latent_frames", [1, 2, 5])
def test_real_backbone_forward_supports_temporal_shapes(latent_frames):
    model = tiny_model().eval()
    with torch.inference_mode():
        output = model(
            [torch.randn(4, latent_frames, 4, 4)],
            t=torch.tensor([500.0]),
            context=[torch.randn(6, 16)],
            seq_len=latent_frames * 4,
            y=[torch.randn(8, latent_frames, 4, 4)],
            img_ref=[torch.randn(4, 1, 4, 4)],
        )[0]
    assert output.shape == (4, latent_frames, 4, 4)
    assert torch.isfinite(output).all()


def test_precision_keeps_fp32_islands():
    model = tiny_model()
    original = model.time_embedding[0].weight.detach().clone()
    inference_precision(model)
    assert model.patch_embedding.weight.dtype == torch.bfloat16
    assert model.time_embedding[0].weight.dtype == torch.float32
    assert model.head.head.weight.dtype == torch.float32
    assert model.blocks[0].norm3.weight.dtype == torch.float32
    torch.testing.assert_close(model.time_embedding[0].weight, original, rtol=0, atol=0)


def test_real_scheduler_finishes_without_nan():
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
    scheduler.set_timesteps(4, device="cpu", shift=5)
    latent = torch.randn(1, 4, 2, 4, 4)
    for timestep in scheduler.timesteps:
        latent = scheduler.step(torch.zeros_like(latent), timestep, latent, return_dict=False)[0]
    assert torch.isfinite(latent).all()
