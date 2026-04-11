# Copyright (C) 2026 Nicholas Perez — AGPL-3.0-or-later

"""Fanned-out AR decode: per-operation Triton kernels with full GPU utilization.

Instead of one persistent kernel per sample (3 of 30 SMs busy), each VMM fans out
across all SMs via grid=(batch, n_tiles). SSM/attention also parallel per head.
Python loop over decode steps; ~16 kernel launches per step per layer.
"""

import torch
import triton
import triton.language as tl

from ar_kernel import pack_weights, sync_weights, precompute_kv_cache

# ═══════════════════════════════════════════════════════════════════════════════
# Fanned-out Triton kernels
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _embed_k(wb_ptr, embed_off, x_ptr, x_s0, tok_ptr):
    """Embedding lookup. Grid=(batch,)."""
    bid = tl.program_id(0)
    cur_tok = tl.load(tok_ptr + bid)
    offs = tl.arange(0, 512)
    mask = offs < 384
    POS_SCALE: tl.constexpr = 19.595917942265423
    emb = tl.load(wb_ptr + embed_off + cur_tok * 384 + offs, mask=mask, other=0.0).to(tl.float32) * POS_SCALE
    tl.store(x_ptr + bid * x_s0 + offs, emb.to(tl.float16), mask=mask)


@triton.jit
def _ln_k(x_ptr, x_s0, w_off, b_off, wb_ptr, out_ptr, out_s0):
    """LayerNorm(384). Grid=(batch,)."""
    bid = tl.program_id(0)
    offs = tl.arange(0, 512)
    mask = offs < 384
    x = tl.load(x_ptr + bid * x_s0 + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / 384.0
    xc = x - mean
    xc = tl.where(mask, xc, 0.0)
    var = tl.sum(xc * xc) / 384.0
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    w = tl.load(wb_ptr + w_off + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(wb_ptr + b_off + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + bid * out_s0 + offs, (xc * rstd * w + b).to(tl.float16), mask=mask)


@triton.jit
def _rmsn_k(x_ptr, x_s0, x_off, w_off, wb_ptr, N: tl.constexpr, NP2: tl.constexpr):
    """In-place RMSNorm at offset within buffer. Grid=(batch,)."""
    bid = tl.program_id(0)
    base = x_ptr + bid * x_s0 + x_off
    offs = tl.arange(0, NP2)
    mask = offs < N
    x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / N + 1e-8)
    w = tl.load(wb_ptr + w_off + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(base + offs, (x / rms * w).to(tl.float16), mask=mask)


@triton.jit
def _ln_vmm_k(x_ptr, x_s0, lnw_off, lnb_off, wb_ptr,
              w_off, b_off, y_ptr, y_s0,
              K: tl.constexpr, N: tl.constexpr, TILE_N: tl.constexpr):
    """Fused LayerNorm(384) + GEMV. Each tile computes norm stats, applies on-the-fly. Grid=(batch, tiles)."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    # Compute LayerNorm stats (mean, rstd) — all tiles do this redundantly, it's cheap
    d_offs = tl.arange(0, 512)
    d_mask = d_offs < 384
    x_full = tl.load(x_ptr + bid * x_s0 + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    mean = tl.sum(x_full) / 384.0
    xc = x_full - mean
    xc = tl.where(d_mask, xc, 0.0)
    var = tl.sum(xc * xc) / 384.0
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    # VMM tile with on-the-fly norm
    n_start = tid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    w_base = wb_ptr + w_off
    for k_start in tl.range(0, K, 64):
        k_offs = k_start + tl.arange(0, 64)
        k_mask = k_offs < K
        xv = tl.load(x_ptr + bid * x_s0 + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        lnw_k = tl.load(wb_ptr + lnw_off + k_offs, mask=k_mask, other=1.0).to(tl.float32)
        lnb_k = tl.load(wb_ptr + lnb_off + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        xv_n = (xv - mean) * rstd * lnw_k + lnb_k
        wv = tl.load(w_base + n_offs[:, None] * K + k_offs[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(xv_n[None, :] * wv, axis=1)
    bv = tl.load(wb_ptr + b_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + bid * y_s0 + n_offs, (acc + bv).to(tl.float16), mask=n_mask)


@triton.jit
def _vmm_res_k(x_ptr, x_s0, res_ptr, res_s0, w_off, b_off, wb_ptr,
               K: tl.constexpr, N: tl.constexpr, TILE_N: tl.constexpr):
    """Fanned-out GEMV + residual add: X[tile] = res[tile] + x @ W[tile].T + b[tile]. Grid=(batch, tiles)."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    n_start = tid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    x_base = x_ptr + bid * x_s0
    w_base = wb_ptr + w_off
    for k_start in tl.range(0, K, 64):
        k_offs = k_start + tl.arange(0, 64)
        k_mask = k_offs < K
        xv = tl.load(x_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_base + n_offs[:, None] * K + k_offs[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(xv[None, :] * wv, axis=1)
    bv = tl.load(wb_ptr + b_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    rv = tl.load(res_ptr + bid * res_s0 + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(res_ptr + bid * res_s0 + n_offs, (rv + acc + bv).to(tl.float16), mask=n_mask)


@triton.jit
def _quad_rmsn_k(x_ptr, x_s0, wb_ptr,
                 off0, w0, n0: tl.constexpr, np0: tl.constexpr,
                 off1, w1, n1: tl.constexpr, np1: tl.constexpr,
                 off2, w2, n2: tl.constexpr, np2: tl.constexpr,
                 off3, w3, n3: tl.constexpr, np3: tl.constexpr):
    """4 in-place RMSNorms at different offsets in one launch. Grid=(batch,)."""
    bid = tl.program_id(0)
    base = x_ptr + bid * x_s0
    # Norm 0
    offs = tl.arange(0, np0)
    mask = offs < n0
    x = tl.load(base + off0 + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / n0 + 1e-8)
    w = tl.load(wb_ptr + w0 + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(base + off0 + offs, (x / rms * w).to(tl.float16), mask=mask)
    # Norm 1
    offs = tl.arange(0, np1)
    mask = offs < n1
    x = tl.load(base + off1 + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / n1 + 1e-8)
    w = tl.load(wb_ptr + w1 + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(base + off1 + offs, (x / rms * w).to(tl.float16), mask=mask)
    # Norm 2
    offs = tl.arange(0, np2)
    mask = offs < n2
    x = tl.load(base + off2 + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / n2 + 1e-8)
    w = tl.load(wb_ptr + w2 + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(base + off2 + offs, (x / rms * w).to(tl.float16), mask=mask)
    # Norm 3
    offs = tl.arange(0, np3)
    mask = offs < n3
    x = tl.load(base + off3 + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / n3 + 1e-8)
    w = tl.load(wb_ptr + w3 + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(base + off3 + offs, (x / rms * w).to(tl.float16), mask=mask)


@triton.jit
def _vmm_gelu_k(x_ptr, x_s0, w_off, b_off, wb_ptr, y_ptr, y_s0,
                K: tl.constexpr, N: tl.constexpr, TILE_N: tl.constexpr):
    """Fanned-out GEMV + GELU fused into store. Grid=(batch, tiles)."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    n_start = tid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    x_base = x_ptr + bid * x_s0
    w_base = wb_ptr + w_off
    for k_start in tl.range(0, K, 64):
        k_offs = k_start + tl.arange(0, 64)
        k_mask = k_offs < K
        xv = tl.load(x_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_base + n_offs[:, None] * K + k_offs[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(xv[None, :] * wv, axis=1)
    bv = tl.load(wb_ptr + b_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    y = acc + bv
    y = y * tl.sigmoid(1.702 * y)  # GELU
    tl.store(y_ptr + bid * y_s0 + n_offs, y.to(tl.float16), mask=n_mask)


@triton.jit
def _vmm_k(x_ptr, x_s0, w_off, b_off, wb_ptr, y_ptr, y_s0,
           K: tl.constexpr, N: tl.constexpr, TILE_N: tl.constexpr):
    """Fanned-out GEMV. Grid=(batch, ceil(N/TILE_N))."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    n_start = tid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    x_base = x_ptr + bid * x_s0
    w_base = wb_ptr + w_off
    for k_start in tl.range(0, K, 64):
        k_offs = k_start + tl.arange(0, 64)
        k_mask = k_offs < K
        xv = tl.load(x_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_base + n_offs[:, None] * K + k_offs[None, :],
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(xv[None, :] * wv, axis=1)
    bv = tl.load(wb_ptr + b_off + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + bid * y_s0 + n_offs, (acc + bv).to(tl.float16), mask=n_mask)


@triton.jit
def _ssm_step_k(
    ip_ptr, ip_s0,
    wb_ptr, dtb, Doff, Bboff, Cboff,
    ssm_ptr, ssm_s0, ssm_s1, ssm_s2, ssm_s3, ssm_s4,
    kst_ptr, kst_s0, kst_s1, kst_s2, kst_s3,
    vst_ptr, vst_s0, vst_s1, vst_s2, vst_s3,
    ast_ptr, ast_s0, ast_s1, ast_s2,
    sc_ptr, sc_s0,  # scratch (batch, 12, 128) for per-head B/C rotary
    y_ptr, y_s0,
    layer_idx,
):
    """SSM step for one head. Grid=(batch, 12)."""
    bid = tl.program_id(0)
    h = tl.program_id(1)
    li = layer_idx

    s64 = tl.arange(0, 64)
    ip = ip_ptr + bid * ip_s0

    raw_A = tl.load(ip + 1676 + h).to(tl.float32)
    raw_dt = tl.load(ip + 1664 + h).to(tl.float32)
    raw_trap = tl.load(ip + 1688 + h).to(tl.float32)

    A_NEG_FLOOR: tl.constexpr = -0.0001
    neg_sp_A = -tl.log(1.0 + tl.exp(raw_A))
    A_clamped = tl.minimum(neg_sp_A, A_NEG_FLOOR)
    dt_bias_h = tl.load(wb_ptr + dtb + h).to(tl.float32)
    DT_val = tl.log(1.0 + tl.exp(raw_dt + dt_bias_h))
    decay = tl.exp(A_clamped * DT_val)
    trap_val = tl.sigmoid(raw_trap)
    D_val = tl.load(wb_ptr + Doff + h).to(tl.float32)

    B_base = tl.load(ip + 1536 + s64).to(tl.float32)
    C_base = tl.load(ip + 1600 + s64).to(tl.float32)
    B_bias_h = tl.load(wb_ptr + Bboff + h * 64 + s64).to(tl.float32)
    C_bias_h = tl.load(wb_ptr + Cboff + h * 64 + s64).to(tl.float32)
    # Store B/C biased to per-head scratch for pointer-based rotary (no race between heads)
    sc_base = sc_ptr + bid * sc_s0 + h * 128  # 128 = 64(B) + 64(C) per head
    B_off = sc_base
    C_off = sc_base + 64
    tl.store(B_off + s64, (B_base + B_bias_h).to(tl.float16))
    tl.store(C_off + s64, (C_base + C_bias_h).to(tl.float16))

    # Rotary
    s16 = tl.arange(0, 16)
    ast_base_h = ast_ptr + bid * ast_s0 + li * ast_s1 + h * ast_s2
    angles_in = tl.load(ip + 1700 + s16).to(tl.float32)
    angle_state = tl.load(ast_base_h + s16).to(tl.float32)
    PI: tl.constexpr = 3.141592653589793
    TWO_PI: tl.constexpr = 6.283185307179586
    new_angle = angle_state + tl.extra.cuda.libdevice.tanh(angles_in) * DT_val * PI
    new_angle = new_angle - TWO_PI * tl.extra.cuda.libdevice.floor(new_angle / TWO_PI)
    tl.store(ast_base_h + s16, new_angle)

    for ri in range(16):
        ang = tl.load(ast_base_h + ri).to(tl.float32)
        cos_ri = tl.cos(ang)
        sin_ri = tl.sin(ang)
        b_e = tl.load(B_off + ri * 2).to(tl.float32)
        b_o = tl.load(B_off + ri * 2 + 1).to(tl.float32)
        tl.store(B_off + ri * 2, (b_e * cos_ri - b_o * sin_ri).to(tl.float16))
        tl.store(B_off + ri * 2 + 1, (b_e * sin_ri + b_o * cos_ri).to(tl.float16))
        c_e = tl.load(C_off + ri * 2).to(tl.float32)
        c_o = tl.load(C_off + ri * 2 + 1).to(tl.float32)
        tl.store(C_off + ri * 2, (c_e * cos_ri - c_o * sin_ri).to(tl.float16))
        tl.store(C_off + ri * 2 + 1, (c_e * sin_ri + c_o * cos_ri).to(tl.float16))

    B_new = tl.load(B_off + s64).to(tl.float32)
    C_new = tl.load(C_off + s64).to(tl.float32)
    x_h = tl.load(ip + 768 + h * 64 + s64).to(tl.float32)

    kst_base = kst_ptr + bid * kst_s0 + li * kst_s1
    vst_base = vst_ptr + bid * vst_s0 + li * vst_s1
    k_old = tl.load(kst_base + h * kst_s2 + s64 * kst_s3).to(tl.float32)

    gamma = trap_val * DT_val
    beta = decay * DT_val * (1.0 - trap_val)
    ssm_base = ssm_ptr + bid * ssm_s0 + li * ssm_s1

    for di in range(64):
        state_row = tl.load(ssm_base + h * ssm_s2 + di * ssm_s3 + s64 * ssm_s4).to(tl.float32)
        x_di = tl.load(ip + 768 + h * 64 + di).to(tl.float32)
        v_di = tl.load(vst_base + h * vst_s2 + di * vst_s3).to(tl.float32)
        state_row = decay * state_row + gamma * x_di * B_new + beta * v_di * k_old
        tl.store(ssm_base + h * ssm_s2 + di * ssm_s3 + s64 * ssm_s4, state_row)
        y_di = tl.sum(state_row * C_new) + D_val * x_di
        z_di = tl.load(ip + h * 64 + di).to(tl.float32)
        y_di = y_di * z_di * tl.sigmoid(z_di)
        tl.store(y_ptr + bid * y_s0 + h * 64 + di, y_di.to(tl.float16))

    tl.store(kst_base + h * kst_s2 + s64 * kst_s3, B_new)
    tl.store(vst_base + h * vst_s2 + s64 * vst_s3, x_h)


@triton.jit
def _ca_k(
    q_ptr, q_s0, qnw_off, wb_ptr,
    kc_ptr, kc_s0, kc_s1, kc_s2, kc_s3,
    vc_ptr, vc_s0, vc_s1, vc_s2, vc_s3,
    mask_ptr, mask_s0,
    out_ptr, out_s0,
    layer_idx, src_len,
):
    """Cross-attention for one head. Grid=(batch, 6)."""
    bid = tl.program_id(0)
    h = tl.program_id(1)
    li = layer_idx

    s64 = tl.arange(0, 64)
    q_h = tl.load(q_ptr + bid * q_s0 + h * 64 + s64).to(tl.float32)
    rms = tl.sqrt(tl.sum(q_h * q_h) / 64.0 + 1e-8)
    w = tl.load(wb_ptr + qnw_off + s64).to(tl.float32)
    q_h = (q_h / rms * w).to(tl.float16)
    # Store q_h to output buffer temporarily for pointer-based scalar access in loop
    q_tmp = out_ptr + bid * out_s0 + h * 64
    tl.store(q_tmp + s64, q_h)

    CA_SCALE: tl.constexpr = 0.125
    kc_base = kc_ptr + bid * kc_s0 + li * kc_s1 + h * kc_s2
    vc_base = vc_ptr + bid * vc_s0 + li * vc_s1 + h * vc_s2

    s_offs = tl.arange(0, 1024)
    s_valid = s_offs < src_len
    pad_flags = tl.load(mask_ptr + bid * mask_s0 + s_offs, mask=s_valid, other=True)
    active = s_valid & ~pad_flags

    scores = tl.zeros((1024,), dtype=tl.float32)
    for d in range(64):
        q_d = tl.load(q_tmp + d).to(tl.float32)
        k_col = tl.load(kc_base + s_offs * kc_s3 + d, mask=s_valid, other=0.0).to(tl.float32)
        scores += q_d * k_col
    scores = scores * CA_SCALE
    scores = tl.where(active, scores, float('-inf'))
    max_s = tl.max(scores)
    exp_s = tl.exp(scores - max_s)
    exp_s = tl.where(active, exp_s, 0.0)
    sum_s = tl.sum(exp_s) + 1e-9
    attn = exp_s / sum_s

    for d in range(64):
        v_col = tl.load(vc_base + s_offs * vc_s3 + d, mask=s_valid, other=0.0).to(tl.float32)
        val = tl.sum(attn * v_col)
        tl.store(out_ptr + bid * out_s0 + h * 64 + d, val.to(tl.float16))


@triton.jit
def _gelu_k(x_ptr, x_s0, N: tl.constexpr):
    """GELU in-place. Grid=(batch,)."""
    bid = tl.program_id(0)
    base = x_ptr + bid * x_s0
    for start in tl.range(0, N, 64):
        offs = start + tl.arange(0, 64)
        mask = offs < N
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(base + offs, (x * tl.sigmoid(1.702 * x)).to(tl.float16), mask=mask)


@triton.jit
def _residual_k(x_ptr, x_s0, t_ptr, t_s0):
    """x[:, :384] += t[:, :384]. Grid=(batch,)."""
    bid = tl.program_id(0)
    offs = tl.arange(0, 512)
    mask = offs < 384
    x = tl.load(x_ptr + bid * x_s0 + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + bid * t_s0 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + bid * x_s0 + offs, (x + t).to(tl.float16), mask=mask)


@triton.jit
def _out_argmax_k(x_ptr, x_s0, wb_ptr, embed_off,
                  sc_ptr, sc_s0, ix_ptr, ix_s0,
                  TILE_V: tl.constexpr):
    """Output proj for one vocab tile + local argmax. Grid=(batch, n_tiles)."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    v_start = tid * TILE_V
    v_offs = v_start + tl.arange(0, TILE_V)
    v_mask = v_offs < 8000

    scores = tl.zeros((TILE_V,), dtype=tl.float32)
    for k_start in tl.range(0, 384, 64):
        k_offs = k_start + tl.arange(0, 64)
        k_mask = k_offs < 384
        xk = tl.load(x_ptr + bid * x_s0 + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(wb_ptr + embed_off + v_offs[:, None] * 384 + k_offs[None, :],
                     mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        scores += tl.sum(xk[None, :] * wv, axis=1)
    scores = tl.where(v_mask, scores, float('-inf'))
    local_max = tl.max(scores)
    local_argmax = tl.argmax(scores, axis=0)
    tl.store(sc_ptr + bid * sc_s0 + tid, local_max)
    tl.store(ix_ptr + bid * ix_s0 + tid, v_start + local_argmax)


# ═══════════════════════════════════════════════════════════════════════════════
# Weight offset extraction
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_OT_START = 3   # offset_table[3] = first layer's first weight
LAYER_OT_STRIDE = 27  # 27 weight entries per decoder layer


def _layer_offsets(ot_cpu, li):
    """Extract weight offsets for layer li from offset table (already on CPU)."""
    b = LAYER_OT_START + li * LAYER_OT_STRIDE
    return [ot_cpu[b + i] for i in range(LAYER_OT_STRIDE)]


# ═══════════════════════════════════════════════════════════════════════════════
# Main decode function
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ar_decode_parallel(model, src_ids, sp, max_len=1536, src_key_padding_mask=None,
                       initial_ar_state=None, memory=None, K_all=None, V_all=None):
    """AR decode with fanned-out per-operation kernels.

    For chunked training, pass initial_ar_state (from previous chunk's return),
    and pre-computed memory/K_all/V_all (encoder output computed once).
    Returns (result_ids, final_ar_state) when initial_ar_state is provided,
    otherwise returns result_ids only (backward compat).
    """
    device = src_ids.device
    batch = src_ids.shape[0]
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
    n_layers = len(model.decoder_layers)
    return_state = initial_ar_state is not None

    # Encode + KV cache (skip if pre-computed)
    if memory is None:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            memory = model.encode(src_ids)
    if K_all is None or V_all is None:
        K_all, V_all = precompute_kv_cache(model, memory, src_key_padding_mask)

    if not hasattr(model, '_ar_weight_blob'):
        pack_weights(model)
    sync_weights(model)
    wb = model._ar_weight_blob
    ot = model._ar_offset_table

    src_len = src_ids.shape[1]
    ot_cpu = ot.cpu().tolist()
    embed_off = ot_cpu[0]
    dn_w = ot_cpu[1]
    dn_b = ot_cpu[2]

    # Pre-extract per-layer offsets
    loffs = [_layer_offsets(ot_cpu, li) for li in range(n_layers)]

    # Buffers
    X = torch.zeros(batch, 512, dtype=torch.float16, device=device)
    TMP = torch.zeros(batch, 512, dtype=torch.float16, device=device)
    IP = torch.zeros(batch, 1716, dtype=torch.float16, device=device)
    Y = torch.zeros(batch, 768, dtype=torch.float16, device=device)
    QBUF = torch.zeros(batch, 384, dtype=torch.float16, device=device)
    AOUT = torch.zeros(batch, 384, dtype=torch.float16, device=device)
    FFMID = torch.zeros(batch, 1536, dtype=torch.float16, device=device)
    TMP384 = torch.zeros(batch, 384, dtype=torch.float16, device=device)
    out_ids = torch.full((batch, max_len), pad_id, dtype=torch.int32, device=device)

    # SSM states — use initial_ar_state if provided
    if initial_ar_state is not None:
        ssm = initial_ar_state["ssm"].clone()
        kst = initial_ar_state["kst"].clone()
        vst = initial_ar_state["vst"].clone()
        ast = initial_ar_state["ast"].clone()
        cur_tok = initial_ar_state["cur_tok"].clone()
    else:
        ssm = torch.zeros(batch, n_layers, 12, 64, 64, dtype=torch.float16, device=device)
        kst = torch.zeros(batch, n_layers, 12, 64, dtype=torch.float16, device=device)
        vst = torch.zeros(batch, n_layers, 12, 64, dtype=torch.float16, device=device)
        ast = torch.zeros(batch, n_layers, 12, 16, dtype=torch.float16, device=device)
        cur_tok = torch.full((batch,), bos_id, dtype=torch.int32, device=device)

    # Per-head scratch for B/C rotary (12 heads × 128 = 64 B + 64 C)
    ssm_sc = torch.zeros(batch, 12 * 128, dtype=torch.float16, device=device)

    if src_key_padding_mask is None:
        src_mask = torch.zeros(batch, src_len, dtype=torch.bool, device=device)
    else:
        src_mask = src_key_padding_mask

    # Output argmax buffers
    NVT = (8000 + 63) // 64  # 125 tiles
    t_scores = torch.empty(batch, NVT, dtype=torch.float32, device=device)
    t_idxs = torch.empty(batch, NVT, dtype=torch.int32, device=device)

    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    # Grid sizes for fanned-out VMMs
    G_IP = (batch, (1716 + 63) // 64)   # 27 tiles
    G_OP = (batch, (384 + 63) // 64)    # 6 tiles
    G_FF1 = (batch, (1536 + 63) // 64)  # 24 tiles
    G_B = (batch,)
    G_SSM = (batch, 12)
    G_CA = (batch, 6)
    G_OA = (batch, NVT)

    for step in range(max_len):
        if step % 8 == 0 and step > 0 and finished.all():
            break

        # Embed
        _embed_k[G_B](wb, embed_off, X, X.stride(0), cur_tok)

        for li in range(n_layers):
            o = loffs[li]  # [snw, snb, ipw, ipb, znw, xnw, dtb, D, Bb, Cb, Bnw, Cnw, opw, opb, cnw, cnb, cqw, cqb, cow, cob, cqnw, fnw, fnb, f1w, f1b, f2w, f2b]

            # ── 1. Self-attn: fused(norm+in_proj) → quad_rmsnorm → SSM → fused(out_proj+residual→X) ──
            _ln_vmm_k[G_IP](X, X.stride(0), o[0], o[1], wb, o[2], o[3], IP, IP.stride(0), K=384, N=1716, TILE_N=64)
            _quad_rmsn_k[G_B](IP, IP.stride(0), wb,
                              0, o[4], 768, 1024,        # z
                              768, o[5], 768, 1024,      # x
                              1536, o[10], 64, 64,       # B
                              1600, o[11], 64, 64)       # C
            _ssm_step_k[G_SSM](
                IP, IP.stride(0), wb, o[6], o[7], o[8], o[9],
                ssm, ssm.stride(0), ssm.stride(1), ssm.stride(2), ssm.stride(3), ssm.stride(4),
                kst, kst.stride(0), kst.stride(1), kst.stride(2), kst.stride(3),
                vst, vst.stride(0), vst.stride(1), vst.stride(2), vst.stride(3),
                ast, ast.stride(0), ast.stride(1), ast.stride(2),
                ssm_sc, ssm_sc.stride(0),
                Y, Y.stride(0), li,
            )
            _vmm_res_k[G_OP](Y, Y.stride(0), X, X.stride(0), o[12], o[13], wb, K=768, N=384, TILE_N=64)

            # ── 2. Cross-attn: fused(norm+Q_proj) → attn → fused(out_proj+residual→X) ──
            _ln_vmm_k[G_OP](X, X.stride(0), o[14], o[15], wb, o[16], o[17], QBUF, QBUF.stride(0), K=384, N=384, TILE_N=64)
            _ca_k[G_CA](
                QBUF, QBUF.stride(0), o[20], wb,
                K_all, K_all.stride(0), K_all.stride(1), K_all.stride(2), K_all.stride(3),
                V_all, V_all.stride(0), V_all.stride(1), V_all.stride(2), V_all.stride(3),
                src_mask, src_mask.stride(0),
                AOUT, AOUT.stride(0), li, src_len,
            )
            _vmm_res_k[G_OP](AOUT, AOUT.stride(0), X, X.stride(0), o[18], o[19], wb, K=384, N=384, TILE_N=64)

            # ── 3. FFN: fused(norm+linear1) → GELU → fused(linear2+residual→X) ──
            _ln_vmm_k[G_FF1](X, X.stride(0), o[21], o[22], wb, o[23], o[24], FFMID, FFMID.stride(0), K=384, N=1536, TILE_N=64)
            _gelu_k[G_B](FFMID, FFMID.stride(0), N=1536)
            _vmm_res_k[G_OP](FFMID, FFMID.stride(0), X, X.stride(0), o[25], o[26], wb, K=1536, N=384, TILE_N=64)

        # Final norm + output argmax
        _ln_k[G_B](X, X.stride(0), dn_w, dn_b, wb, TMP, TMP.stride(0))
        _out_argmax_k[G_OA](TMP, TMP.stride(0), wb, embed_off, t_scores, t_scores.stride(0), t_idxs, t_idxs.stride(0), TILE_V=64)

        # Reduce argmax (GPU)
        best_tiles = t_scores.argmax(dim=1)  # (batch,)
        cur_tok = t_idxs.gather(1, best_tiles.unsqueeze(1).long()).squeeze(1).int()
        out_ids[:, step] = cur_tok
        finished |= (cur_tok == eos_id)

    # Collect results
    result = []
    ids_cpu = out_ids.cpu().tolist()
    for i in range(batch):
        ids = ids_cpu[i]
        if eos_id in ids:
            ids = ids[:ids.index(eos_id) + 1]
        else:
            while ids and ids[-1] == pad_id:
                ids.pop()
        result.append(ids)

    if return_state:
        final_ar_state = {
            "ssm": ssm, "kst": kst, "vst": vst, "ast": ast,
            "cur_tok": cur_tok,
        }
        return result, final_ar_state
    return result
