# DeepfakeHeretic

Open-source video face swapping for a single NVIDIA GPU.

DeepfakeHeretic combines the pretrained **DreamID-V Faster (Wan 1.3B)** model with
a memory-conscious inference pipeline targeting **24 GB GPUs such as the RTX
4090**. Provide a source portrait and a target video to transfer the source
identity while retaining the target's motion and scene. Still images are also
supported through the same pipeline.

> **Experimental:** CUDA execution, RTX 4090 memory usage, and visual quality are
> awaiting hardware validation. The included presets are starting points for
> benchmarking. See the [validation record](docs/validation.md) for completed checks.

## Features

- **Video and image input** with H.264 MP4 and lossless PNG output.
- **Bounded video processing** with overlapping windows and complete tail-frame handling.
- **Memory controls** through BF16 weights, efficient attention, CPU offloading,
  and a configurable PyTorch allocator budget.
- **Background preservation** through feathered face compositing at the input resolution.
- **Audio retention** with the first audio track re-encoded to AAC.
- **Reproducible runs** with pinned model assets, checksum verification, seed
  controls, and JSON reports containing settings and memory measurements.

## Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Linux or Windows with WSL2 |
| GPU | NVIDIA GPU with CUDA and BF16 support; presets target an RTX 4090 with 24 GB VRAM |
| Driver | Compatible with the PyTorch CUDA 12.4 build |
| Python | 3.10–3.12; 3.11 recommended |
| System RAM | 32 GB recommended for CPU offloading |
| Disk space | About 15 GB for weights and download cache, plus the Python environment and outputs |
| Media tools | Working `ffmpeg` and `ffprobe` executables on `PATH` |

The runtime uses PyTorch's built-in attention kernels. A separate CUDA toolkit or
FlashAttention build is not required. CPU development and tests can run on macOS;
generation requires CUDA.

## Installation

Clone or download this repository, then run the following from its root directory:

```bash
bash scripts/setup-linux.sh
source .venv/bin/activate
heretic download
heretic doctor --verify
```

Install the NVIDIA driver, Python's venv support, and FFmpeg before running the
setup script. The script creates a local Python environment and installs the
project dependencies. `heretic download` fetches approximately 6.6 GB of model
assets from pinned upstream revisions and verifies their SHA-256 checksums.

For an existing Python environment, install manually:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[gpu]'
heretic download
heretic doctor --verify
```

Use `--models /path/to/models` on each command to select a custom model directory.

## Quick start

Prepare a clear, tightly cropped source portrait, approximately 512 × 512 pixels,
and a target containing one dominant visible face. Source portraits must be
cropped before use.

### Video

```bash
heretic swap \
  --source inputs/source-face.jpg \
  --target inputs/target.mp4 \
  --output outputs/swapped.mp4 \
  --preset 4090
```

Videos are processed in overlapping windows. Output retains the input dimensions
and nominal duration, with variable-frame-rate input normalized to its average
frame rate. Odd dimensions are rounded down by at most one pixel for H.264.

The generated face is blended into the original frame by default. Use
`--full-frame` to export the entire generated frame, resized to the input dimensions.
Face detail is limited by the inference resolution in either mode.

### Still image

```bash
heretic swap \
  --source inputs/source-face.jpg \
  --target inputs/target.jpg \
  --output outputs/swapped.png
```

Image mode repeats the target across five frames and exports the first result.
It is experimental and has not been evaluated against dedicated image-swap models.

## Presets and memory

| Preset | Maximum landscape dimensions | Window / overlap | Steps |
| --- | --- | --- | --- |
| `4090` | 832 × 480 | 33 / 9 | 16 |
| `low-memory` | 640 × 368 | 17 / 5 | 16 |
| `720p` | 1280 × 720 | 17 / 5 | 20 |

Presets have not yet been benchmarked on the target GPU. Run the benchmark before
processing a full video, especially when using `720p`:

```bash
heretic benchmark --preset 4090 --output outputs/4090-benchmark.json
```

The benchmark runs the actual model on synthetic inputs and records execution
time and peak PyTorch CUDA memory. It measures execution and memory, not likeness
or visual quality.

The default PyTorch allocator budget is 22 GiB, reduced when available memory is
lower. CUDA context memory, driver allocations, and other applications are outside
this limit. Only the VAE or diffusion transformer is resident on the GPU at a time;
face detection runs on CPU. This pipeline is intended for offline processing.

Portrait inputs use a portrait bounding box. Inference dimensions approximately
preserve the input aspect ratio and round down to multiples of 16. For example,
a 1920 × 1080 video uses 832 × 464 with the default preset. Smaller inputs are not
upscaled before inference.

### Useful options

| Option | Purpose |
| --- | --- |
| `--preset low-memory` | Reduce inference resolution and context length |
| `--window 17 --overlap 5` | Reduce context length without changing resolution |
| `--steps 20` | Set the number of diffusion sampling steps |
| `--seed 123` | Set the random seed |
| `--vram-gib 20` | Set the PyTorch allocator budget |
| `--device 1` | Select a CUDA device |
| `--dry-run` | Inspect the processing plan without loading weights |
| `--overwrite` | Replace an existing output and its report |

Run `heretic --help` or `heretic swap --help` for command details. If a run exhausts
GPU memory, close other GPU applications and retry with `--preset low-memory`.
Set `HERETIC_DEBUG=1` to include a full traceback when diagnosing errors.

## Outputs and limitations

Each output includes a `.json` sidecar containing settings, input and output
hashes, model identifiers, frame counts, elapsed time, GPU details, and peak
allocated and reserved PyTorch CUDA memory. Output metadata identifies the media
as synthetic; it is ordinary metadata, not a durable watermark or C2PA signature.

- **Single dominant face:** the largest visible face is selected per frame.
  Persistent identity tracking across multiple people is not implemented.
- **Temporal consistency:** overlapping windows can still produce seams, flicker,
  or ghosting. Split footage at hard scene cuts before processing.
- **Difficult footage:** profiles, occlusion, small faces, motion blur, and extreme
  expressions can reduce quality. Masks use facial landmarks rather than
  occlusion-aware segmentation.
- **Missed detections:** frames without a detected face pass through unchanged.
  A target with no usable face fails without publishing an output.
- **Pretrained inference:** the project integrates upstream model weights;
  training and fine-tuning workflows are not included.

Use media you own or have permission to modify, and follow the upstream model terms.

## Development and contributing

Install development dependencies in your environment and run the checks:

```bash
python -m pip install -e '.[gpu,dev]'
pytest -q
ruff check src tests scripts/verify-checkpoints.py
```

Tests cover streaming reconstruction, audio muxing, error handling, attention
math, a reduced-size backbone, and the upstream scheduler. Media integration tests
use a test generator. To check compatibility with the actual downloaded weights:

```bash
python scripts/verify-checkpoints.py --models models
```

The checkpoint verifier runs on CPU and requires enough host memory to load the
FP32 model. GPU benchmarks and visual evaluation remain separate checks.

Bug reports, reproducible benchmarks, and pull requests are welcome. Include the
command used, operating system, GPU, dependency versions, and relevant error output
when reporting a problem. Share only sample media you have permission to distribute.

- [Architecture and hardware validation procedure](docs/architecture.md)
- [Validation record](docs/validation.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Acknowledgments and license

DeepfakeHeretic builds on the work of the DreamID-V, Wan, DWPose, and Hugging Face
Diffusers contributors.

- [DreamID-V implementation](https://github.com/bytedance/DreamID-V) and
  [paper](https://arxiv.org/abs/2601.01425)
- [DreamID-V Faster checkpoint](https://huggingface.co/XuGuo699/DreamID-V/blob/main/dreamidv_faster.pth)
- [Wan 2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

Source code is licensed under [Apache 2.0](LICENSE). Original copyright notices
are retained for vendored components. Model assets remain subject to their
upstream terms; see [third-party notices](THIRD_PARTY_NOTICES.md) for attribution
and sources.
