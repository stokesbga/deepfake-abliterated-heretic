import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .assets import manifest, sha256
from .config import Settings, fitted_size
from .faces import composite
from .media import (
    IMAGE_EXTENSIONS,
    VideoWriter,
    mux_audio,
    probe,
    read_image,
    video_frames,
)
from .temporal import OverlapBlender, windows

LOG = logging.getLogger(__name__)


def publish(source: Path, destination: Path, *, overwrite: bool):
    if overwrite:
        source.replace(destination)
    else:
        # Atomic no-clobber: concurrent jobs cannot overwrite one another's result.
        os.link(source, destination)
        source.unlink()


def run(
    source: Path,
    target: Path,
    output: Path,
    models: Path,
    settings: Settings,
    *,
    overwrite: bool = False,
    engine=None,
    masker=None,
) -> dict:
    source, target, output, models = [
        p.expanduser().resolve() for p in (source, target, output, models)
    ]
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError("Source image and target must both exist.")
    if output in (source, target):
        raise ValueError("Output must be different from both inputs.")
    sidecar = output.with_suffix(output.suffix + ".json")
    if not overwrite and (output.exists() or sidecar.exists()):
        raise FileExistsError(f"Output or report already exists: {output}. Use --overwrite.")
    is_image = target.suffix.lower() in IMAGE_EXTENSIONS
    if is_image and output.suffix.lower() != ".png":
        raise ValueError("Still-image output must be .png.")
    if not is_image and output.suffix.lower() != ".mp4":
        raise ValueError("Video output must be .mp4.")
    reference = read_image(source)
    if is_image:
        still = np.array(read_image(target))
        input_frames = iter([still])
        width, height = still.shape[1], still.shape[0]
        info = None
    else:
        info = probe(target)
        width, height = info.width, info.height
        input_frames = video_frames(target, info)
    size = fitted_size(width, height, settings.width, settings.height)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    if engine is None:
        from .engine import DreamIDEngine

        engine = DreamIDEngine(models, settings)
    if masker is None:
        from .faces import FaceMasker

        masker = FaceMasker(models)
    blender = OverlapBlender(settings.overlap)
    detected_count = 0
    count = 0
    chunk_count = 0
    # Cache overlapping detections so a frame's conditioning stays identical.
    mask_cache = {}
    with tempfile.TemporaryDirectory(prefix=".heretic-", dir=output.parent) as temp:
        directory = Path(temp)
        artifact = directory / output.name
        writer = VideoWriter(directory / "silent.mp4", info) if info else None
        completed = False
        try:
            for window in windows(input_frames, settings.window, settings.overlap):
                LOG.info(
                    "Window %d: input frames %d–%d",
                    chunk_count + 1,
                    window.start,
                    window.start + len(window.frames) - 1,
                )
                small = np.stack(
                    [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in window.frames]
                )
                masks = []
                for offset, frame in enumerate(small):
                    index = window.start + offset
                    if index not in mask_cache:
                        mask_cache[index] = masker(frame)
                        if mask_cache[index].any():
                            detected_count += 1
                    masks.append(mask_cache[index])
                masks = np.stack(masks)
                if masks.any():
                    generated = engine.generate(
                        small, masks, reference, seed=(settings.seed + window.start) % (2**63)
                    )
                    if generated.shape != small.shape:
                        raise RuntimeError("Model returned unexpected frame count or dimensions.")
                    results = []
                    for original, frame, mask in zip(window.frames, generated, masks):
                        if not mask.any():
                            results.append(original)
                        elif settings.composite:
                            results.append(composite(original, frame, mask))
                        else:
                            results.append(cv2.resize(frame, (width, height)))
                    generated = np.stack(results)
                else:
                    generated = np.stack(window.frames)
                blended = blender.push(window.start, generated, final=window.final)
                count += len(blended)
                chunk_count += 1
                if writer:
                    writer.write(blended)
                else:
                    metadata = PngInfo()
                    metadata.add_text("Description", "AI-generated face swap / DreamID-V")
                    Image.fromarray(blended[0]).save(artifact, pnginfo=metadata)
                mask_cache = {
                    key: value
                    for key, value in mask_cache.items()
                    if key >= window.start + len(window.frames) - settings.overlap
                }
            if count == 0:
                raise ValueError("No decodable frames in the target.")
            if detected_count == 0:
                raise ValueError("No usable target face detected; no output was published.")
            completed = True
        finally:
            if hasattr(input_frames, "close"):
                input_frames.close()
            if writer:
                writer.close(abort=not completed)
        if info:
            mux_audio(directory / "silent.mp4", target, artifact, count, info)
        report = {
            "synthetic_media": True,
            "model": "DreamID-V Faster 1.3B",
            "settings": settings.to_dict(),
            "inference_size": list(size),
            "output_size": [width, height],
            "frames": count,
            "windows": chunk_count,
            "frames_with_face": detected_count,
            "fps": str(info.fps) if info else None,
            "audio_preserved": bool(info and info.audio),
            "elapsed_seconds": time.monotonic() - start_time,
            "source_sha256": sha256(source),
            "target_sha256": sha256(target),
            "output_sha256": sha256(artifact),
            "weights": [{"name": item["name"], "sha256": item["sha256"]} for item in manifest()],
            "hardware": engine.metrics(),
            "image_mode": "five repeated frames; first frame exported" if is_image else None,
        }
        staged_report = directory / "report.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        publish(artifact, output, overwrite=overwrite)
        publish(staged_report, sidecar, overwrite=overwrite)
    LOG.info("Saved %s (%d frames); report: %s", output, count, sidecar)
    return report
