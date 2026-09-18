import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settings:
    # Presets are engineering targets, not GPU benchmark claims.
    width: int = 832
    height: int = 480
    window: int = 33
    overlap: int = 9
    steps: int = 16
    guidance: float = 4.0
    shift: float = 5.0
    seed: int = 42
    vram_gib: float = 22.0
    device: int = 0
    composite: bool = True

    def __post_init__(self):
        if min(self.width, self.height) < 64 or self.width % 16 or self.height % 16:
            raise ValueError("Width and height must be multiples of 16 and at least 64.")
        if self.window < 5 or (self.window - 1) % 4:
            raise ValueError("Window must be 4n+1 frames (e.g. 17, 33, 49).")
        if not 0 <= self.overlap < self.window / 2:
            raise ValueError("Overlap must be nonnegative and less than half the window.")
        if self.steps < 2:
            raise ValueError("At least two sampling steps are required.")
        if not all(math.isfinite(x) for x in (self.guidance, self.shift, self.vram_gib)):
            raise ValueError("Guidance, shift and memory budget must be finite.")
        if self.guidance < 0 or self.shift <= 0 or self.vram_gib < 4:
            raise ValueError("Invalid guidance, shift or memory budget.")
        if not 0 <= self.seed < 2**63 or self.device < 0:
            raise ValueError("Seed must be in [0, 2^63); device must be nonnegative.")

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "4090": Settings(),
    "low-memory": Settings(width=640, height=368, window=17, overlap=5),
    "720p": Settings(width=1280, height=720, window=17, overlap=5, steps=20),
}


def fitted_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Preserve orientation; round down for the VAE's spatial stride of 16."""
    if min(width, height) < 16:
        raise ValueError("Input dimensions must be at least 16 pixels.")
    if height > width and max_width > max_height:
        max_width, max_height = max_height, max_width
    scale = min(1.0, max_width / width, max_height / height)
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


def padded_length(length: int) -> int:
    if length < 1:
        raise ValueError("Cannot pad an empty sequence.")
    return max(5, ((length - 1 + 3) // 4) * 4 + 1)
