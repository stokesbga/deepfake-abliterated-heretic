"""Local replacement: PyTorch SDPA with padding support and bounded CUDA memory."""
from ...attention import flash_attention

__all__ = ["flash_attention"]

