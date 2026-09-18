import numpy as np
import pytest

from deepfakeheretic.config import Settings, fitted_size, padded_length
from deepfakeheretic.temporal import OverlapBlender, windows


@pytest.mark.parametrize("length", [1, 4, 5, 9, 16, 17, 18, 24, 32, 33, 34, 48, 57, 58, 101])
@pytest.mark.parametrize("size,overlap", [(33, 9), (17, 5), (5, 0)])
def test_stream_reconstruction_keeps_every_frame(length, size, overlap):
    frames = [np.full((2, 2, 3), index, np.uint8) for index in range(length)]
    blender = OverlapBlender(overlap)
    output = []
    for window in windows(iter(frames), size, overlap):
        assert len(window.frames) <= size
        output.extend(blender.push(window.start, np.stack(window.frames), final=window.final))
    np.testing.assert_array_equal(np.stack(output), np.stack(frames))
    assert blender.pending is None


def test_window_iterator_is_bounded():
    seen = []

    def frames():
        for index in range(1000):
            seen.append(index)
            yield np.zeros((1, 1, 3), np.uint8)

    iterator = windows(frames(), 33, 9)
    next(iterator)
    assert len(seen) == 34
    next(iterator)
    assert len(seen) == 58


def test_invalid_configuration_and_shape_rounding():
    for values in ({"window": 32}, {"overlap": 17}, {"steps": 1}, {"vram_gib": float("nan")}):
        with pytest.raises(ValueError):
            Settings(**values)
    assert fitted_size(1920, 1080, 832, 480) == (832, 464)
    assert fitted_size(1080, 1920, 832, 480) == (464, 832)
    assert fitted_size(100, 100, 832, 480) == (96, 96)
    assert [padded_length(i) for i in (1, 4, 5, 6, 33, 34)] == [5, 5, 5, 9, 33, 37]


def test_misordered_windows_fail():
    with pytest.raises(ValueError, match="expected"):
        OverlapBlender(1).push(3, np.zeros((4, 1, 1, 3)), final=True)
