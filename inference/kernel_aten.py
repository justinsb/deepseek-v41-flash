# Copyright (c) 2026 Google LLC. MIT License, same as this repository.
"""Plain-aten versions of the tilelang kernels in kernel.py.

These exist so that the model can be captured by torch.export on a
machine without a GPU and run on any device: the executor may later
substitute the tilelang kernels (or others) where it has them. Each
function reproduces the arithmetic of its kernel, including the
quantize-dequantize rounding of activations, which is part of the model
maker's computation and not an implementation detail. Matmuls accumulate
in float32; the kernels accumulate on tensor cores in float32 too, so
the results agree to summation order.

FP8 here is float8_e4m3fn (max 448); FP4 is e2m1 (values 0, 0.5, 1, 1.5,
2, 3, 4, 6 with a sign), stored two per byte, low nibble first, as
torch.float4_e2m1fn_x2 or as the checkpoint's int8.
"""

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 code -> value, for the 16 nibble patterns (sign in bit 3).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# The positive grid and the midpoints between its steps, for round-to-nearest-even.
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive x, as the kernels' fast_round_scale: the
    smallest power of two that is >= x (exact powers of two map to themselves)."""
    e = torch.frexp(x.float()).exponent  # x = m * 2**e with m in [0.5, 1)
    # m * 2**e <= 2**e, with equality only when m == 0.5 (x a power of two)
    is_pow2 = torch.ldexp(torch.full_like(x.float(), 0.5), e) == x.float()
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), torch.where(is_pow2, e - 1, e))


def _to_fp8_rounded(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values to the fp8 e4m3 grid and back (what a cast does)."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def _to_fp4_rounded(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values in [-6, 6] to the e2m1 grid (nearest, ties to even
    in the grid index, which is what hardware casts do)."""
    grid = _FP4_GRID.to(x.device)
    a = x.abs()
    # index of the largest grid value <= a, then decide between it and the next
    idx = torch.bucketize(a, grid, right=True) - 1
    idx = idx.clamp(0, len(grid) - 1)
    lo = grid[idx]
    hi = grid[(idx + 1).clamp(max=len(grid) - 1)]
    mid = (lo + hi) / 2
    up = (a > mid) | ((a == mid) & (idx % 2 == 1))
    q = torch.where(up, hi, lo)
    return q.copysign(x)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[..., K//2] packed e2m1 pairs (float4_e2m1fn_x2 or int8/uint8) -> [..., K] float32."""
    b = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    table = FP4_TABLE.to(b.device)
    low = table[(b & 0x0F).long()]
    high = table[((b >> 4) & 0x0F).long()]
    return torch.stack([low, high], dim=-1).flatten(-2)


# ---- quantization

def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise fp8 quantization along the last dim. With scale_fmt set the
    scale is rounded up to a power of two (MXFP). inplace=True writes the
    dequantized values back into x (and returns x); otherwise returns (fp8, scale)."""
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX)
    if inplace:
        y = (_to_fp8_rounded(q) * scale).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    y = q.to(torch.float8_e4m3fn).flatten(-2)
    s = scale.squeeze(-1).to(scale_dtype)
    return y, s


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise fp4 (e2m1) quantization along the last dim, with e8m0
    (power-of-two) or e4m3 scales. inplace=True writes the dequantized values
    back into x; otherwise returns (packed fp4, scale)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        scale = _to_fp8_rounded(amax / FP4_MAX)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        scale = _pow2_ceil(amax / FP4_MAX)
    q = _to_fp4_rounded((xf / scale).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * scale).flatten(-2).to(x.dtype))
        return x
    y = pack_fp4(q.flatten(-2))
    s = scale.squeeze(-1).to(scale_dtype)
    return y, s


def fp4_quant_dequant(x, block_size=32, scale_dtype=torch.float8_e8m0fnu):
    """fp4_act_quant(inplace=True) without the in-place write: returns the quantize-dequantized
    tensor. (The in-place form on a tensor whose size is a derived symbolic expression makes
    torch.export's shape reasoner ask whether that size is 1; this form does not.)"""
    y = x.clone()
    fp4_act_quant(y, block_size, True, scale_dtype)
    return y


def pack_fp4(values: torch.Tensor) -> torch.Tensor:
    """float32 values already on the e2m1 grid -> packed uint8 pairs (low nibble first)."""
    grid = _FP4_GRID.to(values.device)
    code = torch.bucketize(values.abs(), grid, right=True) - 1
    code = code.clamp(0, 7) | (values.signbit().to(torch.int64) << 3)
    pairs = code.unflatten(-1, (-1, 2))
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


# ---- matmuls

def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M, N] = A[M, K] @ B[N, K]^T with per-block fp8 scaling: A scaled per
    (row, K-block), B per (N-block, K-block)."""
    K = a.size(-1)
    N = b.size(0)
    assert K % block_size == 0
    kb = K // block_size
    af = a.float().unflatten(-1, (kb, block_size)) * a_s.float().unsqueeze(-1)
    nb = (N + block_size - 1) // block_size
    bf = b.float()
    if N % block_size:
        bf = F.pad(bf, (0, 0, 0, nb * block_size - N))
    bf = bf.view(nb, block_size, kb, block_size) * b_s.float().view(nb, 1, kb, 1)
    bf = bf.view(nb * block_size, K)[:N]
    return (af.flatten(-2) @ bf.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T. A has a per-(row, act block) scale;
    B is packed e2m1 [N, K//2] with a per-(row, 32) scale."""
    K = a.size(-1)
    assert K % act_block_size == 0 and K % 32 == 0
    af = a.float().unflatten(-1, (K // act_block_size, act_block_size)) * a_s.float().unsqueeze(-1)
    bf = unpack_fp4(b).unflatten(-1, (K // 32, 32)) * b_s.float().unsqueeze(-1)
    return (af.flatten(-2) @ bf.flatten(-2).T).to(torch.get_default_dtype())


# ---- attention and hyper-connections

def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q: [b, m, h, d]; kv: [b, n, d] (K and V are the same rows); attn_sink: [h];
    topk_idxs: [b, m, topk] int32 with -1 for an absent slot. For each query,
    softmax over the gathered rows plus a sink term that only adds to the
    denominator; a row with no valid slot gives zeros."""
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    rows = torch.gather(kv, 1, topk_idxs.clamp_min(0).long().reshape(b, -1, 1).expand(-1, -1, d))
    rows = rows.view(b, m, -1, d).float()  # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), rows) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    smax = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(scores - smax)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - smax)
    o = torch.einsum("bmht,bmtd->bmhd", p, rows) / denom
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """mixes: [b, s, (2 + hc) * hc] -> pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc],
    comb made doubly stochastic by sinkhorn iterations."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
