# Validation record

Local validation completed on 2026-09-18 using macOS ARM64, Python 3.11.3,
PyTorch 2.6.0, Diffusers 0.35.2, and NumPy 1.26.4. No CUDA device was available.

| Check | Result |
| --- | --- |
| Automated tests | 60 passed |
| Ruff | Passed for first-party source, tests, and checkpoint verifier |
| Wheel and source archive | Built successfully; package includes model code, manifest, licenses |
| All five downloaded assets | Byte sizes and SHA-256 hashes verified |
| Full checkpoint compatibility | All 827 tensor names and shapes match; strict load passed |
| Real pretrained transformer | CPU FP32 forward on a small latent input returned finite output |
| Real pretrained VAE | Five 64 × 64 frames encoded and decoded with finite output and correct shapes |
| Real DWPose networks | ONNX CPU execution completed on a synthetic blank image |
| Video media integration | 58 frames at 30000/1001 fps, three windows, final tail and audio retained |
| Still-image integration | PNG and provenance report produced using an explicit test generator |
| Actual CUDA inference | **Not run** |
| RTX 4090 peak VRAM and speed | **Not measured** |
| Human likeness / visual quality | **Not evaluated** |

The local Homebrew FFmpeg binary had a missing x265 shared library. Media tests
used isolated npm-distributed ARM64 FFmpeg/FFprobe binaries in `.tools/bin`, with
their archive SHA-512 integrity checked. No system binaries were modified. These
local test binaries are not included in the transfer archive; the target machine
must provide its own working FFmpeg installation.

`checkpoint-smoke.json` contains the actual checkpoint smoke-test result.
Reproduce it with:

```bash
python scripts/verify-checkpoints.py --models models
```

This script requires the full weights and enough host RAM to load the FP32 model.
Its synthetic-input checks establish checkpoint compatibility and numerical
execution only, not face-swap quality. The CUDA benchmark and visual evaluation
procedure are in `architecture.md`.

