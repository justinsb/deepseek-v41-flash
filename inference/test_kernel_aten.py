# Copyright (c) 2026 Google LLC. MIT License, same as this repository.
"""Checks of kernel_aten by construction: round trips, dense equivalents, and
the kernels' documented arithmetic. Run: python -m pytest test_kernel_aten.py
or python test_kernel_aten.py."""

import torch

import kernel_aten as K


def test_pow2_ceil():
    x = torch.tensor([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 448.0, 1e-4])
    want = torch.tensor([0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 512.0, 2.0 ** -13])
    assert torch.equal(K._pow2_ceil(x), want), K._pow2_ceil(x)


def test_fp4_round_trip_and_rounding():
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0])
    assert torch.equal(K._to_fp4_rounded(vals), vals)
    # nearest, ties to even grid index: 0.25 -> 0 (idx 0 even), 0.75 -> 1.0 (idx 2), 2.5 -> 2.0 (idx 4), 5 -> 4 (idx 6)
    ties = torch.tensor([0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 5.9, 0.26])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 6.0, 0.5])
    assert torch.equal(K._to_fp4_rounded(ties), want), K._to_fp4_rounded(ties)
    packed = K.pack_fp4(vals[:8])
    assert packed.dtype == torch.uint8 and packed.shape == (4,)
    assert torch.equal(K.unpack_fp4(packed), vals[:8])
    neg = K.pack_fp4(torch.tensor([-0.5, -6.0]))
    assert torch.equal(K.unpack_fp4(neg), torch.tensor([-0.5, -6.0]))


def test_act_quant_matches_definition():
    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype=torch.bfloat16) * 20
    y, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert y.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu and s.shape == (3, 2)
    # dequantized values are within half an fp8 ulp of the input, scaled
    deq = y.float().unflatten(-1, (2, 32)) * s.float().unsqueeze(-1)
    rel = ((deq.flatten(-2) - x.float()).abs() / x.float().abs().clamp_min(1e-3)).max()
    assert rel < 0.07, rel  # e4m3 has 3 mantissa bits: 1/16 relative
    # scale is a power of two and >= amax / 448
    amax = x.float().abs().unflatten(-1, (2, 32)).amax(-1)
    assert torch.all(s.float() >= amax / 448) and torch.all(torch.frexp(s.float()).mantissa == 0.5)
    # inplace == dequantized out-of-place
    z = x.clone()
    K.act_quant(z, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    assert torch.equal(z.float(), deq.flatten(-2).to(torch.bfloat16).float())


def test_fp4_act_quant_two_scale_formats():
    torch.manual_seed(1)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    for sd in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        y, s = K.fp4_act_quant(x, 32, scale_dtype=sd)
        assert y.shape == (2, 32) and s.dtype == sd and s.shape == (2, 2)
        deq = K.unpack_fp4(y).unflatten(-1, (2, 32)) * s.float().unsqueeze(-1)
        err = (deq.flatten(-2) - x.float()).abs().max()
        assert err < x.float().abs().max() / 4, err  # e2m1: coarse, but bounded by a grid step
        z = x.clone()
        K.fp4_act_quant(z, 32, inplace=True, scale_dtype=sd)
        assert torch.equal(z.float(), deq.flatten(-2).to(torch.bfloat16).float())


def test_fp8_gemm_equals_dequantized_matmul():
    torch.manual_seed(2)
    torch.set_default_dtype(torch.bfloat16)
    M, K_, N = 5, 64, 40  # N not a multiple of the block: exercises the padding
    a = torch.randn(M, K_, dtype=torch.bfloat16) * 3
    aq, a_s = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    # a 64-row weight quantized per 32x32 block with e8m0 scales, of which the model uses 40 rows
    wb = torch.randn(64, K_).view(2, 32, 2, 32)
    ws = K._pow2_ceil(wb.abs().amax(dim=(1, 3)) / 448)  # [2, 2]
    wq_full = (wb / ws.view(2, 1, 2, 1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(64, K_)[:N]
    c = K.fp8_gemm(aq, a_s, wq_full, ws.to(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
    a_deq = aq.float().view(M, 2, 32) * a_s.float().unsqueeze(-1)
    w_deq = wq_full.float().view(N, 2, 32) * ws.repeat_interleave(32, 0)[:N].unsqueeze(-1)
    want = (a_deq.flatten(1) @ w_deq.flatten(1).T).to(torch.bfloat16)
    assert c.dtype == torch.bfloat16 and torch.allclose(c.float(), want.float(), rtol=1e-2, atol=1e-2)
    torch.set_default_dtype(torch.float32)


def test_fp4_gemm_equals_dequantized_matmul():
    torch.manual_seed(3)
    torch.set_default_dtype(torch.bfloat16)
    M, K_, N = 4, 64, 6
    a = torch.randn(M, K_, dtype=torch.bfloat16)
    aq, a_s = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    w = torch.randn(N, K_)
    ws = K._pow2_ceil(w.abs().view(N, 2, 32).amax(-1) / 6)  # [N, K/32]
    wq = K._to_fp4_rounded((w.view(N, 2, 32) / ws.unsqueeze(-1)).clamp(-6, 6)).flatten(1)
    packed = K.pack_fp4(wq)
    c = K.fp4_gemm(aq, a_s, packed, ws.to(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
    a_deq = aq.float().view(M, 2, 32) * a_s.float().unsqueeze(-1)
    w_deq = wq.view(N, 2, 32) * ws.unsqueeze(-1)
    want = (a_deq.flatten(1) @ w_deq.flatten(1).T).to(torch.bfloat16)
    assert torch.allclose(c.float(), want.float(), rtol=1e-2, atol=1e-2)
    torch.set_default_dtype(torch.float32)


def test_sparse_attn_against_dense_softmax_with_sink():
    torch.manual_seed(4)
    b, m, h, d, n = 2, 3, 4, 16, 10
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.tensor([[[0, 1, 2, -1], [3, 4, -1, -1], [-1, -1, -1, -1]]] * b, dtype=torch.int32)
    o = K.sparse_attn(q, kv, sink, idx, 0.25)
    # dense: softmax over the selected rows, with exp(sink) in the denominator
    for bi in range(b):
        for mi in range(m):
            sel = [i for i in idx[bi, mi].tolist() if i >= 0]
            if not sel:
                assert torch.all(o[bi, mi] == 0)
                continue
            rows = kv[bi, sel].float()
            s = q[bi, mi].float() @ rows.T * 0.25  # [h, t]
            p = torch.exp(s - s.amax(-1, keepdim=True))
            den = p.sum(-1, keepdim=True) + torch.exp(sink.view(h, 1) - s.amax(-1, keepdim=True))
            want = (p @ rows) / den
            assert torch.allclose(o[bi, mi].float(), want, atol=2e-2), (bi, mi)


def test_hc_split_sinkhorn_is_doubly_stochastic():
    torch.manual_seed(5)
    hc = 4
    mixes = torch.randn(2, 3, (2 + hc) * hc)
    pre, post, comb = K.hc_split_sinkhorn(mixes, torch.tensor([1.0, 0.5, 2.0]), torch.randn((2 + hc) * hc), hc, 20, 1e-6)
    assert pre.shape == (2, 3, hc) and post.shape == (2, 3, hc) and comb.shape == (2, 3, hc, hc)
    assert torch.all(pre > 0) and torch.all(post >= 0) and torch.all(post <= 2)
    # the kernel's last step normalizes columns exactly (to eps); rows converge
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, hc), atol=1e-4), comb.sum(-2)
    assert torch.allclose(comb.sum(-1), torch.ones(2, 3, hc), atol=5e-2), comb.sum(-1)
    # and the layout of the 24 coefficients: pre from the first hc, post from the next hc



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
