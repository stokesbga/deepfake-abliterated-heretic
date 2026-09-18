"""Memory-efficient SDPA adapter for Wan's [batch, sequence, heads, channels] layout."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    if window_size != (-1, -1):
        raise ValueError("This runtime supports global attention only.")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Incompatible attention shapes.")
    original_dtype = q.dtype
    compute_dtype = dtype if q.is_cuda else q.dtype
    # Do not silently fall back to an O(sequence^2) CUDA math implementation.
    backend = (
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        if q.is_cuda
        else nullcontext()
    )
    outputs = []
    with backend:
        for index in range(q.shape[0]):
            nq = int(q_lens[index]) if q_lens is not None else q.shape[1]
            nk = int(k_lens[index]) if k_lens is not None else k.shape[1]
            if not 0 < nq <= q.shape[1] or not 0 < nk <= k.shape[1]:
                raise ValueError("Attention lengths must be within the sequence bounds.")
            qi = q[index : index + 1, :nq].transpose(1, 2).to(compute_dtype)
            ki = k[index : index + 1, :nk].transpose(1, 2).to(compute_dtype)
            vi = v[index : index + 1, :nk].transpose(1, 2).to(compute_dtype)
            if q_scale is not None:
                qi = qi * q_scale
            result = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            ).transpose(1, 2)
            if nq < q.shape[1]:
                result = F.pad(result, (0, 0, 0, 0, 0, q.shape[1] - nq))
            outputs.append(result)
    return torch.cat(outputs, dim=0).to(original_dtype)
