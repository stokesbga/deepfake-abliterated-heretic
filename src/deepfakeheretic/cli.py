import argparse
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from .assets import check_assets, download, manifest
from .config import PRESETS, fitted_size


def add_settings(parser):
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--preset", choices=PRESETS, default="4090")
    parser.add_argument("--window", type=int, help="Context length, 4n+1 frames")
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vram-gib", type=float, help="PyTorch allocator cap (default: 22 GiB)")
    parser.add_argument("--device", type=int, help="CUDA device index")
    parser.add_argument("--full-frame", action="store_true", help="Export full model frame")


def settings_from(args):
    fields = ("window", "overlap", "steps", "seed", "vram_gib", "device")
    values = {name: getattr(args, name) for name in fields if getattr(args, name) is not None}
    if args.full_frame:
        values["composite"] = False
    return replace(PRESETS[args.preset], **values)


def doctor(models: Path, verify: bool) -> tuple[dict, bool]:
    from .media import executable_check

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "asset_problems": check_assets(models, hashes=verify),
        "packages": {},
    }
    report["media_tools"] = {
        name: executable_check(name, shutil.which(name)) for name in ("ffmpeg", "ffprobe")
    }
    for name in ("torch", "diffusers", "einops", "scipy", "onnxruntime", "numpy", "Pillow"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_version"] = torch.version.cuda
        report["gpus"] = []
        if report["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(index)
                report["gpus"].append(
                    {"index": index, "name": gpu.name, "vram_gib": gpu.total_memory / 1024**3}
                )
    except ImportError:
        report["cuda_available"] = False
    ready = bool(
        report["cuda_available"]
        and all(item["working"] for item in report["media_tools"].values())
        and not report["asset_problems"]
        and all(report["packages"].values())
    )
    report["ready_for_inference"] = ready
    return report, ready


def benchmark(args):
    import numpy as np
    from PIL import Image

    from .engine import DreamIDEngine

    settings = settings_from(args)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Benchmark report exists: {output}")
    start = time.monotonic()
    engine = DreamIDEngine(args.models.expanduser().resolve(), settings)
    # Exercise a full preset window and real model weights without needing personal media.
    frames = np.full((settings.window, settings.height, settings.width, 3), 127, np.uint8)
    masks = np.zeros(frames.shape[:3], np.uint8)
    masks[
        :,
        settings.height // 4 : 3 * settings.height // 4,
        settings.width // 3 : 2 * settings.width // 3,
    ] = 255
    loaded = time.monotonic()
    result = engine.generate(
        frames, masks, Image.new("RGB", (512, 512), "gray"), seed=settings.seed
    )
    report = {
        "kind": "synthetic-input GPU execution and memory benchmark; not a quality evaluation",
        "settings": settings.to_dict(),
        "frames_generated": len(result),
        "load_seconds": loaded - start,
        "inference_seconds": time.monotonic() - loaded,
        "hardware": engine.metrics(),
        "quality_tested": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.overwrite else "x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="heretic", description="Video-first face swapping with pretrained DreamID-V Faster."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Download and verify pinned model assets (~6.6 GB)")
    fetch.add_argument("--models", type=Path, default=Path("models"))
    fetch.add_argument("--list", action="store_true", help="Print manifest without downloading")
    check = sub.add_parser("doctor", help="Check CUDA, FFmpeg, dependencies and weights")
    check.add_argument("--models", type=Path, default=Path("models"))
    check.add_argument("--verify", action="store_true", help="Verify full weight checksums")
    swap = sub.add_parser("swap", help="Swap a cropped reference identity into a video or image")
    add_settings(swap)
    swap.add_argument("--source", type=Path, required=True, help="Cropped source face image")
    swap.add_argument("--target", type=Path, required=True, help="Target video or still image")
    swap.add_argument("--output", type=Path, required=True, help=".mp4 for video; .png for images")
    swap.add_argument("--overwrite", action="store_true")
    swap.add_argument("--dry-run", action="store_true", help="Show plan without loading weights")
    bench = sub.add_parser("benchmark", help="Measure actual model execution and peak CUDA memory")
    add_settings(bench)
    bench.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    bench.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Set before importing torch or initializing its CUDA allocator.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        if args.command == "download":
            if args.list:
                print(json.dumps(manifest(), indent=2))
            else:
                download(args.models.expanduser().resolve())
        elif args.command == "doctor":
            report, ready = doctor(args.models.expanduser().resolve(), args.verify)
            print(json.dumps(report, indent=2))
            return 0 if ready else 1
        elif args.command == "benchmark":
            benchmark(args)
        else:
            settings = settings_from(args)
            if args.dry_run:
                from .media import IMAGE_EXTENSIONS, probe, read_image

                target = args.target.expanduser().resolve()
                reference = read_image(args.source.expanduser().resolve())
                if target.suffix.lower() in IMAGE_EXTENSIONS:
                    width, height = read_image(target).size
                else:
                    info = probe(target)
                    width, height = info.width, info.height
                print(
                    json.dumps(
                        {
                            "settings": settings.to_dict(),
                            "source_size": reference.size,
                            "inference_size": fitted_size(
                                width, height, settings.width, settings.height
                            ),
                            "output": str(args.output.expanduser().resolve()),
                            "vram_fit": "Requires measurement with heretic benchmark",
                        },
                        indent=2,
                    )
                )
            else:
                from .pipeline import run

                report = run(
                    args.source,
                    args.target,
                    args.output,
                    args.models,
                    settings,
                    overwrite=args.overwrite,
                )
                print(
                    json.dumps(
                        {
                            "output": str(args.output),
                            "frames": report["frames"],
                            "hardware": report["hardware"],
                        },
                        indent=2,
                    )
                )
        return 0
    except KeyboardInterrupt:
        print("Cancelled; incomplete media is not published.", file=sys.stderr)
        return 130
    except Exception as error:
        # Imports are lazy so doctor and media planning also work on machines without torch.
        if type(error).__name__ == "OutOfMemoryError":
            print(
                "CUDA ran out of memory. Retry with --preset low-memory, or --window 17 "
                "--overlap 5. Close other GPU applications. No frames were silently dropped.",
                file=sys.stderr,
            )
        else:
            print(f"Error: {error}", file=sys.stderr)
        if os.environ.get("HERETIC_DEBUG"):
            raise
        return 1
