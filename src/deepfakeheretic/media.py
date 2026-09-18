"""FFmpeg streaming I/O. Subprocesses never use a shell."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@lru_cache(maxsize=8)
def executable_check(executable: str, path: str | None) -> dict:
    if not path:
        return {"path": None, "working": False, "error": f"{executable} is not on PATH"}
    try:
        result = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=10, check=False
        )
        return {
            "path": path,
            "working": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode else None,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"path": path, "working": False, "error": str(error)}


def require_ffmpeg():
    for executable in ("ffmpeg", "ffprobe"):
        check = executable_check(executable, shutil.which(executable))
        if not check["working"]:
            raise RuntimeError(f"{executable} is unavailable or broken: {check['error']}")


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    duration: float
    audio: bool


def probe(path: Path) -> VideoInfo:
    require_ffmpeg()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot read video: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    stream = next((s for s in payload["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Input has no video stream.")
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            candidate = Fraction(stream.get(key, "0/1"))
            if 0 < candidate <= 240:
                fps = candidate
                break
        except (ValueError, ZeroDivisionError):
            pass
    if fps is None:
        raise ValueError("Input has no usable frame rate.")
    width, height = int(stream["width"]), int(stream["height"])
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    if round(rotation) % 180:
        width, height = height, width
    # H.264 yuv420p requires even output dimensions; lose at most one edge pixel.
    width, height = width // 2 * 2, height // 2 * 2
    duration = float(stream.get("duration", payload.get("format", {}).get("duration", 0)))
    return VideoInfo(
        width, height, fps, duration, any(s["codec_type"] == "audio" for s in payload["streams"])
    )


def video_frames(path: Path, info: VideoInfo):
    """Normalize variable-rate input to its average FPS while retaining its duration."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-vf",
                f"setpts=PTS-STARTPTS,fps={info.fps},scale={info.width}:{info.height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            frame_bytes = info.width * info.height * 3
            while True:
                data = process.stdout.read(frame_bytes)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise RuntimeError("FFmpeg returned an incomplete frame.")
                yield np.frombuffer(data, np.uint8).reshape(info.height, info.width, 3).copy()
            if process.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


class VideoWriter:
    def __init__(self, path: Path, info: VideoInfo):
        self.info = info
        self.count = 0
        self.errors = tempfile.TemporaryFile()  # noqa: SIM115 -- lifetime ends in close()
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{info.width}x{info.height}",
                "-r",
                str(info.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=self.errors,
        )

    def write(self, frames: np.ndarray):
        for frame in frames:
            if frame.shape != (self.info.height, self.info.width, 3) or frame.dtype != np.uint8:
                raise ValueError("Video writer requires uint8 RGB frames at the output size.")
            try:
                self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError as error:
                self.process.wait()
                self.errors.seek(0)
                raise RuntimeError(
                    f"FFmpeg encoder failed: {self.errors.read().decode(errors='replace')}"
                ) from error
            self.count += 1

    def close(self, *, abort=False):
        try:
            if abort and self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            if code and not abort:
                self.errors.seek(0)
                raise RuntimeError(self.errors.read().decode(errors="replace"))
        finally:
            self.errors.close()


def mux_audio(silent: Path, source: Path, output: Path, frames: int, info: VideoInfo):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(silent),
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if info.audio:
        command += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-af", "apad"]
    command += [
        "-c:v",
        "copy",
        "-t",
        f"{frames / float(info.fps):.9f}",
        "-metadata",
        "comment=AI-generated face swap; DeepfakeHeretic / DreamID-V",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Audio mux failed: {result.stderr.strip()}")
