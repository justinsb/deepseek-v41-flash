# Copyright (c) 2026 Google LLC. MIT License, same as this repository.
"""Kernel backend selection.

The published model calls six tilelang kernels (kernel.py), which need a
CUDA GPU and tilelang. This module exposes the same six names, taking
them from kernel.py when tilelang imports and CUDA is available, and
from kernel_aten.py otherwise (CPU, the meta device for torch.export,
or DEEPSEEK_KERNELS=aten to force it). The arithmetic is the same; see
kernel_aten.py for what each one computes.
"""

import os

import torch

_backend = os.environ.get("DEEPSEEK_KERNELS", "").lower()
if _backend not in ("", "aten", "tilelang"):
    raise ValueError(f"DEEPSEEK_KERNELS must be 'aten' or 'tilelang', got {_backend!r}")

if _backend != "aten" and torch.cuda.is_available():
    try:
        from kernel import (  # noqa: F401
            act_quant,
            fp4_act_quant,
            fp4_gemm,
            fp8_gemm,
            hc_split_sinkhorn,
            sparse_attn,
        )
        BACKEND = "tilelang"
    except ImportError:
        if _backend == "tilelang":
            raise
        _backend = "aten"
if _backend == "aten" or not torch.cuda.is_available():
    from kernel_aten import (  # noqa: F401
        act_quant,
        fp4_act_quant,
        fp4_gemm,
        fp8_gemm,
        hc_split_sinkhorn,
        sparse_attn,
    )
    BACKEND = "aten"
