"""Bounded windowing and overlap reconstruction without losing tail frames."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np


@dataclass
class Window:
    start: int
    frames: list[np.ndarray]
    final: bool


def windows(frames: Iterable[np.ndarray], size: int, overlap: int) -> Iterator[Window]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size.")
    source = iter(frames)
    buffer = list(islice(source, size))
    start = 0
    while buffer:
        # One-frame lookahead distinguishes a full final window from a continuing one.
        following = next(source, None)
        yield Window(start, buffer, following is None)
        if following is None:
            return
        start += len(buffer) - overlap
        buffer = (buffer[-overlap:] if overlap else []) + [following]
        buffer += list(islice(source, size - len(buffer)))


class OverlapBlender:
    def __init__(self, overlap: int):
        self.overlap = overlap
        self.pending: np.ndarray | None = None
        self.written = 0

    def push(self, start: int, frames: np.ndarray, *, final: bool) -> np.ndarray:
        if start != self.written:
            raise ValueError(f"Window starts at {start}, expected {self.written}.")
        result = frames.astype(np.float32)
        if self.pending is not None:
            count = len(self.pending)
            if len(result) < count:
                raise ValueError("Window is shorter than the pending overlap.")
            alpha = (np.arange(1, count + 1) / (count + 1)).reshape(-1, 1, 1, 1)
            result[:count] = self.pending * (1 - alpha) + result[:count] * alpha
        keep = 0 if final else self.overlap
        split = len(result) - keep
        self.pending = result[split:].copy() if keep else None
        self.written += split
        return np.clip(np.rint(result[:split]), 0, 255).astype(np.uint8)
