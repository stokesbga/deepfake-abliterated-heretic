"""Optional CPU smoke test with actual weights; does not verify CUDA or visual quality."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from deepfakeheretic._vendor.wan.model import WanModel
from deepfakeheretic._vendor.wan.vae import WanVAE
from deepfakeheretic.assets import check_assets
from deepfakeheretic.faces import FaceMasker

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", type=Path, default=Path("models"))
args = parser.parse_args()
problems = check_assets(args.models, hashes=True)
if problems:
    raise SystemExit("\n".join(problems))
torch.set_num_threads(4)
state = torch.load(
    args.models / "dreamidv_faster.pth", map_location="cpu", weights_only=True, mmap=True
)
model = WanModel(
    model_type="i2v",
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=48,
    num_heads=12,
    num_layers=30,
    cross_attn_norm=True,
    eps=1e-6,
)
model.load_state_dict(state, strict=True)
model.eval().requires_grad_(False)
context = torch.load(args.models / "context.pth", map_location="cpu", weights_only=True)
with torch.inference_mode():
    result = model(
        [torch.zeros(16, 2, 8, 8)],
        t=torch.tensor([500.0]),
        context=[tensor.float() for tensor in context],
        seq_len=32,
        y=[torch.zeros(32, 2, 8, 8)],
        img_ref=[torch.zeros(16, 1, 8, 8)],
    )[0]
assert result.shape == (16, 2, 8, 8) and torch.isfinite(result).all()
report = {"checkpoint_keys": len(state), "backbone_forward_finite": True}
del model, state, result, context
gc.collect()
vae = WanVAE(vae_pth=str(args.models / "Wan2.1_VAE.pth"), dtype=torch.float32, device="cpu")
with torch.inference_mode():
    latent = vae.encode([torch.zeros(3, 5, 64, 64)], "cpu")[0]
    result = vae.decode([latent])[0]
assert latent.shape == (16, 2, 8, 8)
assert result.shape == (3, 5, 64, 64) and torch.isfinite(result).all()
report["vae_roundtrip_finite"] = True
del vae, result, latent
gc.collect()
masker = FaceMasker(args.models)
mask = masker(np.full((64, 64, 3), 127, np.uint8))
assert mask.shape == (64, 64) and mask.dtype == np.uint8
report["dwpose_execution"] = True
report["device"] = "CPU, FP32"
report["cuda_tested"] = False
report["quality_tested"] = False
print(json.dumps(report, indent=2))
