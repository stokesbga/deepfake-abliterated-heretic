import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepfakeheretic.config import Settings
from deepfakeheretic.faces import composite
from deepfakeheretic.media import VideoInfo, VideoWriter, probe, video_frames
from deepfakeheretic.pipeline import run


class TestEngine:
    __test__ = False

    def __init__(self):
        self.calls = []

    def generate(self, frames, masks, reference, *, seed):
        self.calls.append(len(frames))
        return frames.copy()

    def metrics(self):
        return {"test_double": True}


def mask(frame):
    return np.full(frame.shape[:2], 255, np.uint8)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "gray").save(path)
    return path


def test_image_pipeline_provenance_and_no_clobber(source, tmp_path):
    target = tmp_path / "target.png"
    output = tmp_path / "out.png"
    Image.new("RGB", (64, 64), "red").save(target)
    report = run(source, target, output, tmp_path, Settings(), engine=TestEngine(), masker=mask)
    assert report["frames"] == 1
    assert report["synthetic_media"]
    assert json.loads(Path(str(output) + ".json").read_text())["frames_with_face"] == 1
    np.testing.assert_array_equal(np.array(Image.open(target)), np.array(Image.open(output)))
    with pytest.raises(FileExistsError):
        run(source, target, output, tmp_path, Settings(), engine=TestEngine(), masker=mask)


def test_no_face_does_not_publish_unchanged_output(source, tmp_path):
    output = tmp_path / "out.png"
    with pytest.raises(ValueError, match="No usable target face"):
        run(
            source,
            source,
            output,
            tmp_path,
            Settings(),
            engine=TestEngine(),
            masker=lambda frame: np.zeros(frame.shape[:2], np.uint8),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".heretic-*"))


def test_cannot_overwrite_input(source, tmp_path):
    with pytest.raises(ValueError, match="different"):
        run(source, source, source, tmp_path, Settings(), overwrite=True)


def test_compositing_preserves_pixels_outside_face():
    original = np.zeros((128, 128, 3), np.uint8)
    generated = np.full_like(original, 255)
    face_mask = np.zeros((128, 128), np.uint8)
    face_mask[48:80, 48:80] = 255
    result = composite(original, generated, face_mask)
    assert result[64, 64, 0] == 255
    assert not result[:16].any()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_video_roundtrip_tail_fractional_fps_audio_and_bounded_batches(source, tmp_path):
    silent = tmp_path / "silent.mp4"
    target = tmp_path / "target.mp4"
    output = tmp_path / "out.mp4"
    fps = Fraction(30000, 1001)
    info = VideoInfo(64, 64, fps, 58 / float(fps), False)
    writer = VideoWriter(silent, info)
    frames = np.stack([np.full((64, 64, 3), index * 3, np.uint8) for index in range(58)])
    writer.write(frames)
    writer.close()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
    )
    engine = TestEngine()
    report = run(source, target, output, tmp_path, Settings(), engine=engine, masker=mask)
    output_info = probe(output)
    decoded = list(video_frames(output, output_info))
    assert len(decoded) == 58
    assert report["frames"] == report["frames_with_face"] == 58
    assert output_info.fps == fps
    assert output_info.audio and report["audio_preserved"]
    assert engine.calls == [33, 33, 10]
    assert abs(output_info.duration - 58 / float(fps)) < 1 / float(fps)
    assert abs(float(decoded[-1].mean()) - float(frames[-1].mean())) < 5


def test_failed_generation_cleans_staging(source, tmp_path):
    class BrokenEngine(TestEngine):
        def generate(self, *args, **kwargs):
            raise RuntimeError("injected failure")

    output = tmp_path / "out.png"
    with pytest.raises(RuntimeError, match="injected failure"):
        run(source, source, output, tmp_path, Settings(), engine=BrokenEngine(), masker=mask)
    assert not output.exists()
    assert not list(tmp_path.glob(".heretic-*"))
