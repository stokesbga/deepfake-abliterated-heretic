# DeepfakeHeretic

A video-first face-swap pipeline built around **DreamID-V Faster (Wan 1.3B)**, with
an inference runtime designed for a **24 GB NVIDIA RTX 4090**. It accepts a cropped
source face plus a target video or still image. Model architecture code is included;
pretrained weights are downloaded from pinned official sources.

**Status:** this is a pretrained-model integration, not a newly trained model or a
claim of new state-of-the-art results. The default preset targets 24 GB using BF16,
efficient attention, stage offloading, and bounded video windows. Actual CUDA peak
memory, speed, and visual quality must be measured on the target GPU. Development
and CPU tests can run on macOS; generation requires NVIDIA CUDA.

To move this build to another machine, extract
`dist/deepfakeheretic-0.1.0.tar.gz` there. Copy the local `models/` directory into
the extracted project to reuse the verified weights, or run `heretic download`
on that machine. The archive excludes weights and machine-specific environments.

## Install on the NVIDIA machine

Use Linux or WSL2, Python 3.10–3.12 (3.11 recommended), an NVIDIA driver compatible
with CUDA 12.4, and FFmpeg on PATH. Allow about 15 GB disk space for the weights and
download cache, plus the Python environment. **32 GB system RAM recommended**;
CPU offloading needs host memory in addition to VRAM. No CUDA toolkit or separately
compiled FlashAttention package is required.

```bash
# From this project directory on your NVIDIA machine:
bash scripts/setup-linux.sh
source .venv/bin/activate
heretic download
heretic doctor --verify

# Real model execution with synthetic input, measuring peak PyTorch CUDA memory:
heretic benchmark --preset 4090 --output outputs/4090-benchmark.json
```

The installer does not install drivers or system packages. If needed, install
FFmpeg with your operating system's package manager first. You can also install
manually into a Python environment:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[gpu]'
```

## Swap a video

Use a clear, tightly cropped source portrait (roughly 512 × 512) and a target with
one dominant visible face. The source provides identity; the target supplies
motion, pose, expression, and scene.

```bash
heretic swap \
  --source inputs/source-face.jpg \
  --target inputs/target.mp4 \
  --output outputs/swapped.mp4 \
  --preset 4090
```

The full video is streamed through overlapping windows. Tail frames are padded
internally and trimmed back to the correct length. Output retains the original
dimensions and nominal duration; variable-frame-rate sources are normalized to
their average frame rate. The first audio track is preserved through AAC encoding.
Odd dimensions are rounded down by at most one pixel for H.264. Output is H.264 MP4.

By default, a feathered face mask composites the generated region into the
original frame. This preserves background detail at the input resolution; it does
not make the model generate native 4K face detail. `--full-frame` exports the whole
generated frame, resized to the original dimensions.

## Swap a still image

```bash
heretic swap \
  --source inputs/source-face.jpg \
  --target inputs/target.jpg \
  --output outputs/swapped.png
```

Image mode uses five identical target frames and exports the first result. This
reuses the video model; image quality is experimental and has not been benchmarked
against dedicated image-swap models. Output is lossless PNG.

## Memory and quality controls

| Preset | Maximum landscape dimensions | Window / overlap | Steps | Intended use |
| --- | --- | --- | --- | --- |
| `4090` | 832 × 480 | 33 / 9 | 16 | Starting point for 24 GB |
| `low-memory` | 640 × 368 | 17 / 5 | 16 | Smaller working set |
| `720p` | 1280 × 720 | 17 / 5 | 20 | Experimental higher resolution; benchmark first |

These are **unmeasured engineering targets**, not VRAM guarantees. Portrait input
uses a portrait bounding box. Dimensions retain approximately the input aspect
ratio and round down to multiples of 16; a 1920 × 1080 clip uses 832 × 464 with the
default preset. Smaller inputs are not enlarged before inference.

The runtime caps PyTorch's allocator at 22 GiB by default, or less if free memory
requires it. CUDA context/driver memory and other programs are outside that cap.
Only the VAE or diffusion transformer is on the GPU at a time. DWPose runs on CPU.
The checkpoint is loaded on CPU, and large transformer weights are stored in BF16
while numerically sensitive layers stay in FP32. Guidance branches run serially.
PyTorch SDPA uses efficient CUDA kernels; quadratic math fallback is disabled on
CUDA. This is not a real-time pipeline.

```bash
# Diagnose an out-of-memory error by reducing resolution and context:
heretic swap --source inputs/source-face.jpg --target inputs/target.mp4 \
  --output outputs/swapped.mp4 --preset low-memory

# Tune context without changing resolution (window must be 4n+1):
heretic swap --source inputs/source-face.jpg --target inputs/target.mp4 \
  --output outputs/swapped.mp4 --window 17 --overlap 5 --steps 20 --seed 123

# Inspect the plan on any machine, without loading model weights:
heretic swap --source inputs/source-face.jpg --target inputs/target.mp4 \
  --output outputs/swapped.mp4 --dry-run
```

Existing outputs are protected unless `--overwrite` is specified. Use
`--models /path/to/models` consistently if you keep weights elsewhere. Set
`HERETIC_DEBUG=1` for a full exception traceback. Generation never substitutes an
untrained model, silently reduces quality, or drops failed windows.

## Reports and current limitations

Each output has a `.json` sidecar with settings, input/output SHA-256 hashes, weight
identifiers, frame counts, elapsed time, GPU details, and peak allocated/reserved
PyTorch CUDA memory. The benchmark uses real model weights and synthetic inputs
to test execution and memory; it does **not** evaluate identity or visual quality.
Media metadata and reports identify the output as synthetic; this is ordinary
metadata, not a durable watermark or C2PA signature.

- One dominant face is supported. The largest visible face is selected per frame;
  there is no persistent identity tracker for multi-person footage.
- Windows are generated independently with temporal context inside each window;
  overlap blending reduces seams but can still show flicker or ghosting. This is
  not long-video temporal fine-tuning. Split footage at hard scene cuts first.
- Occlusion, profiles, small faces, motion blur, and extreme expressions can fail.
  Frames with no detected face pass through unchanged; a wholly undetected target
  fails without publishing an output.
- Masks are landmark hulls, not occlusion-aware segmentation. Background pixels
  outside the feathered region are preserved before lossy video encoding.
- Source portraits are not automatically cropped. Use a prepared face crop.
- CUDA inference, image quality, and 4090 memory fit still require hardware
  validation. Nothing here claims training-from-scratch feasibility on a 4090.

Use media you own or have permission to modify and follow the upstream model terms.

## Development and model sources

```bash
python -m pip install -e '.[gpu,dev]'
pytest -q
ruff check src tests
```

Tests exercise streaming reconstruction, tails, media muxing, provenance, error
handling, actual attention math, the real backbone at tiny dimensions, and the
upstream scheduler. Media integration tests use an explicit test double for the
generator; they do not establish model quality.

See [architecture and validation](docs/architecture.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).

- [DreamID-V official implementation](https://github.com/bytedance/DreamID-V)
- [DreamID-V paper](https://arxiv.org/abs/2601.01425)
- [Official Faster checkpoint](https://huggingface.co/XuGuo699/DreamID-V/blob/main/dreamidv_faster.pth)
- [Wan 2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
