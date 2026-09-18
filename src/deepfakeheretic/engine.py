"""Single-GPU DreamID-V Faster inference. No training or randomly initialized fallback."""

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from .assets import check_assets
from .config import Settings, padded_length

LOG = logging.getLogger(__name__)
GIB = 1024**3


def inference_precision(model):
    """Keep upstream's explicitly FP32 time/head/norm operations compatible with BF16 weights."""
    fp32_parameters = set()
    for module in (model.time_embedding, model.time_projection, model.head):
        fp32_parameters.update(id(parameter) for parameter in module.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            fp32_parameters.update(id(parameter) for parameter in module.parameters())
    # Preserve original FP32 values; casting the whole model down and back would round them.
    for parameter in model.parameters():
        dtype = torch.float32 if id(parameter) in fp32_parameters else torch.bfloat16
        parameter.data = parameter.data.to(dtype)
    return model


class DreamIDEngine:
    def __init__(self, models: Path, settings: Settings):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run inference on your NVIDIA PC (Linux/WSL2).")
        if settings.device >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {settings.device} does not exist.")
        self.settings = settings
        self.device = torch.device(f"cuda:{settings.device}")
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This runtime requires BF16 support (RTX 30/40/50 series or newer).")
        problems = check_assets(models, hashes=True)
        if problems:
            raise RuntimeError("\n".join(problems) + "\nRun: heretic download --models <directory>")
        free, total = torch.cuda.mem_get_info(self.device)
        budget = min(settings.vram_gib, total / GIB - 1.0, free / GIB - 0.75)
        if budget < 6:
            raise RuntimeError(f"Only {free / GIB:.1f} GiB free; close GPU applications first.")
        torch.cuda.set_per_process_memory_fraction(budget * GIB / total, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.hardware = {
            "gpu": torch.cuda.get_device_name(self.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "total_vram_gib": total / GIB,
            "initial_free_vram_gib": free / GIB,
            "allocator_budget_gib": budget,
        }
        LOG.info("GPU: %s; PyTorch allocator budget %.2f GiB", self.hardware["gpu"], budget)
        from ._vendor.wan.model import WanModel
        from ._vendor.wan.vae import WanVAE

        # mmap avoids duplicating the entire checkpoint in ordinary host memory.
        state = torch.load(
            models / "dreamidv_faster.pth", map_location="cpu", weights_only=True, mmap=True
        )
        self.model = WanModel(
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
        self.model.load_state_dict(state, strict=True)
        del state
        inference_precision(self.model).eval().requires_grad_(False)
        # Load VAE on CPU; only one large component is on CUDA at a time.
        self.vae = WanVAE(
            vae_pth=str(models / "Wan2.1_VAE.pth"), dtype=torch.bfloat16, device="cpu"
        )
        self.vae.model.to(dtype=torch.bfloat16)
        self.context = torch.load(models / "context.pth", map_location="cpu", weights_only=True)
        gc.collect()

    def _vae_to(self, device):
        self.vae.model.to(device)
        self.vae.device = device
        self.vae.mean = self.vae.mean.to(device)
        self.vae.std = self.vae.std.to(device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

    @torch.inference_mode()
    def generate(
        self, frames: np.ndarray, masks: np.ndarray, reference: Image.Image, *, seed: int
    ) -> np.ndarray:
        from ._vendor.wan.solver import FlowUniPCMultistepScheduler

        settings = self.settings
        actual, height, width, channels = frames.shape
        if channels != 3 or masks.shape != (actual, height, width):
            raise ValueError("Expected RGB frames [T,H,W,3] and masks [T,H,W].")
        if height % 16 or width % 16:
            raise ValueError("Model dimensions must be divisible by 16.")
        count = padded_length(actual)
        frames = np.concatenate([frames, np.repeat(frames[-1:], count - actual, axis=0)])
        masks = np.concatenate([masks, np.repeat(masks[-1:], count - actual, axis=0)])
        ref = ImageOps.pad(
            reference.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            color="white",
        )

        def tensor(array, *, normalize=True):
            value = torch.from_numpy(np.array(array, copy=True)).permute(3, 0, 1, 2).float() / 255
            return value * 2 - 1 if normalize else value

        self.model.cpu()
        self._vae_to(self.device)
        # Match the pretrained pipeline: RGB in [-1,1], conditioning mask in [0,1].
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target = self.vae.encode([tensor(frames)], self.device)[0].to(torch.bfloat16)
            mask_rgb = np.repeat(masks[..., None], 3, axis=-1)
            mask_latent = self.vae.encode([tensor(mask_rgb, normalize=False)], self.device)[0]
            source = self.vae.encode([tensor(np.array(ref)[None])], self.device)[0]
        self._vae_to("cpu")
        torch.cuda.empty_cache()
        self.model.to(self.device)
        context = [value.to(self.device, torch.bfloat16) for value in self.context]
        condition = torch.cat([target, mask_latent.to(torch.bfloat16)])
        source = source.to(torch.bfloat16)
        args = {
            "context": context,
            "seq_len": target.shape[1] * target.shape[2] * target.shape[3] // 4,
            "y": [condition],
        }
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(
            target.shape, dtype=torch.float32, device=self.device, generator=generator
        )
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(settings.steps, device=self.device, shift=settings.shift)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for timestep in tqdm(scheduler.timesteps, desc="Diffusion", leave=False):
                t = timestep.reshape(1)
                positive = self.model([latent], t=t, img_ref=[source], **args)[0]
                negative = self.model([latent], t=t, img_ref=[torch.zeros_like(source)], **args)[0]
                # Preserve DreamID-V's guidance convention: positive + s*(positive-negative).
                prediction = positive + settings.guidance * (positive - negative)
                latent = scheduler.step(
                    prediction[None], timestep, latent[None], return_dict=False, generator=generator
                )[0][0]
        self.model.cpu()
        # The rotary cache is not a registered buffer in the upstream model.
        self.model.freqs = self.model.freqs.cpu()
        del args, context, condition, source, target, mask_latent, positive, negative, prediction
        del scheduler
        torch.cuda.empty_cache()
        self._vae_to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = self.vae.decode([latent])[0][:, :actual]
        output = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte()
        output = output.permute(1, 2, 3, 0).cpu().numpy()
        self._vae_to("cpu")
        del latent, decoded
        torch.cuda.empty_cache()
        return output

    def metrics(self) -> dict:
        torch.cuda.synchronize(self.device)
        return {
            **self.hardware,
            "peak_torch_allocated_gib": torch.cuda.max_memory_allocated(self.device) / GIB,
            "peak_torch_reserved_gib": torch.cuda.max_memory_reserved(self.device) / GIB,
            "note": "PyTorch peaks exclude CUDA context, driver and other processes.",
        }
