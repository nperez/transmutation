# Copyright (C) 2026 Nicholas Perez — AGPL-3.0-or-later

"""Fused persistent Triton kernel for Mamba3 autoregressive decode.

One kernel launch, one program per batch element, all steps + all layers.
Zero Python dispatch overhead.

Model: d_model=384, d_inner=768, d_state=64, nheads=12, headdim=64
       6 decoder layers, each: SSM self-attn → cross-attn → FFN
       cross-attn: 6 heads × 64 dim, FFN: 384→1536→384
       in_proj: 384→1716 (z768+x768+B64+C64+dt12+A12+trap12+angles16)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# Weight packing: all model weights → one flat fp16 tensor + offset table
# ═══════════════════════════════════════════════════════════════════════════════

# Weight names per layer, in order. Each maps to a 1D slice of the weight blob.
_LAYER_WEIGHT_NAMES = [
    "sn_w", "sn_b",           # self_norm weight, bias (384 each)
    "ip_w", "ip_b",           # in_proj weight (1716,384), bias (1716)
    "z_norm_w", "x_norm_w",   # NormedInProj RMSNorm weights (768 each)
    "dt_bias",                # (12,)
    "D",                      # (12,)
    "B_bias",                 # (12, 64) → flattened 768
    "C_bias",                 # (12, 64) → flattened 768
    "B_norm_w", "C_norm_w",   # (64,) each
    "op_w", "op_b",           # out_proj weight (384,768), bias (384)
    "cn_w", "cn_b",           # cross_norm weight, bias (384 each)
    "ca_qw", "ca_qb",         # q_proj weight (384,384), bias (384)
    "ca_ow", "ca_ob",         # out_proj weight (384,384), bias (384)
    "ca_qnorm_w",             # q_norm RMSNorm weight (64,)
    "fn_w", "fn_b",           # ff_norm weight, bias (384 each)
    "ff1_w", "ff1_b",         # FFN linear1 weight (1536,384), bias (1536)
    "ff2_w", "ff2_b",         # FFN linear2 weight (384,1536), bias (384)
]


def pack_weights(model):
    """Make model parameters live inside one contiguous flat buffer.

    First call: allocates buffer, copies weights in, replaces every param's .data
    with a view into the buffer. Optimizer updates modify the buffer directly.
    Subsequent calls: returns the same buffer. Zero copy. The kernel always
    reads live weights because the params ARE the buffer.

    Returns: (weight_blob, offset_table)
    """
    if hasattr(model, '_ar_weight_blob'):
        return model._ar_weight_blob, model._ar_offset_table

    dtype = torch.float16
    device = next(model.parameters()).device

    # ── Phase 1: collect all tensors in order, compute sizes and offsets ──
    entries = []  # list of (numel, param_or_None, shape_or_None)
    #   param_or_None: the nn.Parameter whose .data we'll replace (None for static zeros)
    #   shape_or_None: the shape to view back into after flattening

    def _add_param(param):
        """Trainable parameter."""
        entries.append((param.numel(), param, param.shape))

    def _add_static(t):
        """Fixed tensor (zeros/ones for missing biases) — copied once, never updated."""
        entries.append((t.numel(), None, None))

    def _add_param_or_static(param, fallback_size):
        if param is not None:
            _add_param(param)
        else:
            _add_static(torch.zeros(fallback_size, device=device, dtype=dtype))

    # Embedding (shared with output_proj via weight tying)
    _add_param(model.embedding.weight)  # idx 0: (vocab, d_model)

    # Decoder norm
    _add_param(model.decoder_norm.weight)
    _add_param(model.decoder_norm.bias)

    for layer in model.decoder_layers:
        m = layer.self_mamba
        ip = m.in_proj

        _add_param(layer.self_norm.weight)
        _add_param(layer.self_norm.bias)

        # in_proj (may be NormedInProj wrapping a Linear)
        ip_linear = ip.linear if hasattr(ip, 'linear') else ip
        _add_param(ip_linear.weight)  # (1716, 384)
        _add_param_or_static(ip_linear.bias, 1716)

        # NormedInProj RMSNorm weights
        if hasattr(ip, 'z_norm'):
            _add_param(ip.z_norm.weight)
            _add_param(ip.x_norm.weight)
        else:
            _add_static(torch.ones(768, device=device, dtype=dtype))
            _add_static(torch.ones(768, device=device, dtype=dtype))

        _add_param(m.dt_bias)
        _add_param(m.D)
        _add_param(m.B_bias)  # (12, 1, 64) — view preserves shape
        _add_param(m.C_bias)  # (12, 1, 64)
        _add_param(m.B_norm.weight)
        _add_param(m.C_norm.weight)
        _add_param(m.out_proj.weight)  # (384, 768)
        _add_param_or_static(m.out_proj.bias, 384)

        _add_param(layer.cross_norm.weight)
        _add_param(layer.cross_norm.bias)

        ca = layer.cross_attn
        _add_param(ca.q_proj.weight)  # (384, 384)
        _add_param_or_static(ca.q_proj.bias, 384)
        _add_param(ca.out_proj.weight)  # (384, 384)
        _add_param_or_static(ca.out_proj.bias, 384)
        _add_param(ca.q_norm.weight)

        _add_param(layer.ff_norm.weight)
        _add_param(layer.ff_norm.bias)
        _add_param(layer.ff[0].weight)  # (1536, 384)
        _add_param_or_static(layer.ff[0].bias, 1536)
        _add_param(layer.ff[3].weight)  # (384, 1536)
        _add_param_or_static(layer.ff[3].bias, 384)

    # ── Phase 2: allocate flat buffer and build offset table ──
    offsets = []
    total = 0
    for numel, _, _ in entries:
        offsets.append(total)
        total += numel

    blob = torch.empty(total, dtype=dtype, device=device)
    offset_table = torch.tensor(offsets, dtype=torch.int64, device=device)

    # ── Phase 3: copy data into blob (do NOT replace param.data — training needs fp32 params) ──
    param_refs = []  # (offset, numel, param) for sync_weights
    for i, (numel, param, shape) in enumerate(entries):
        off = offsets[i]
        region = blob[off:off + numel]

        if param is not None:
            region.copy_(param.data.to(dtype).flatten())
            param_refs.append((off, numel, param))
        else:
            region.zero_()

    model._ar_weight_blob = blob
    model._ar_offset_table = offset_table
    model._ar_param_refs = param_refs
    return blob, offset_table


def sync_weights(model):
    """Copy current model params into the AR weight blob (fp32→fp16)."""
    blob = model._ar_weight_blob
    for off, numel, param in model._ar_param_refs:
        blob[off:off + numel].copy_(param.data.to(torch.float16).flatten())


def precompute_kv_cache(model, memory, src_key_padding_mask=None):
    """Pre-compute cross-attention K (with norm) and V for all layers.

    Returns (K_all, V_all) each (batch, n_layers, n_ca_heads, src_len, ca_head_dim), contiguous.
    """
    batch, src_len, _ = memory.shape
    dtype = memory.dtype
    device = memory.device
    n_layers = len(model.decoder_layers)

    K_all = torch.empty(batch, n_layers, 6, src_len, 64, dtype=dtype, device=device)
    V_all = torch.empty(batch, n_layers, 6, src_len, 64, dtype=dtype, device=device)

    for i, layer in enumerate(model.decoder_layers):
        ca = layer.cross_attn
        k = ca.k_proj(memory).view(batch, src_len, 6, 64).transpose(1, 2)
        v = ca.v_proj(memory).view(batch, src_len, 6, 64).transpose(1, 2)
        k = ca.k_norm(k.float()).to(dtype)
        K_all[:, i] = k
        V_all[:, i] = v

    return K_all.contiguous(), V_all.contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent decode kernel
# ═══════════════════════════════════════════════════════════════════════════════
#
# Scratch layout per sample (total ~6KB, fits in L1):
#   [0 .. 384)       x: current hidden state
#   [384 .. 2100)     ip_out: in_proj output (1716)
#   [2100 .. 2484)    tmp384: temp buffer (384) for norms, projections
#   [2484 .. 4020)    tmp1536: temp buffer (1536) for FFN mid
#   [4020 .. 4404)    q_buf: cross-attn query (384)
#   [4404 .. 4788)    attn_out: cross-attn output (384)

SCRATCH_SIZE = 5200  # elements per sample, fp16

# Offset table layout:
#   0: embedding
#   1: decoder_norm weight
#   2: decoder_norm bias
#   3 + layer*27 + 0..26: per-layer weights


@triton.jit
def _vmm_inline(x_ptr, w_ptr, b_ptr, y_ptr,
                K: tl.constexpr, N: tl.constexpr):
    """y = x @ W.T + b, where W is (N, K) row-major.

    x is (K,), y is (N,). Tiles: 64-wide in N, 64-wide in K reduction.
    """
    for n_start in tl.range(0, N, 64):
        n_offs = n_start + tl.arange(0, 64)
        n_mask = n_offs < N
        acc = tl.zeros((64,), dtype=tl.float32)
        for k_start in tl.range(0, K, 64):
            k_offs = k_start + tl.arange(0, 64)
            k_mask = k_offs < K
            xv = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            # W is (N, K) row-major: W[n, k] at w_ptr + n*K + k
            wv = tl.load(w_ptr + n_offs[:, None] * K + k_offs[None, :],
                         mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(xv[None, :] * wv, axis=1)
        bv = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + n_offs, (acc + bv).to(tl.float16), mask=n_mask)


@triton.jit
def _layernorm_inline(x_ptr, w_ptr, b_ptr, out_ptr,
                      N: tl.constexpr, NP2: tl.constexpr):
    """LayerNorm: out = (x - mean) / sqrt(var + eps) * w + b. NP2 = next power of 2 >= N."""
    offs = tl.arange(0, NP2)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / N
    xc = x - mean
    xc = tl.where(mask, xc, 0.0)
    var = tl.sum(xc * xc) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (xc * rstd * w + b).to(tl.float16), mask=mask)


@triton.jit
def _rmsnorm_inline(x_ptr, w_ptr, out_ptr, N: tl.constexpr, NP2: tl.constexpr):
    """RMSNorm: out = x / rms(x) * w. NP2 = next power of 2 >= N."""
    offs = tl.arange(0, NP2)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x) / N + 1e-8)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offs, (x / rms * w).to(tl.float16), mask=mask)


@triton.jit
def _persistent_decode_kernel(
    # Output token IDs (batch, max_steps) int32
    out_ptr, out_batch_stride,
    # Weight blob (flat fp16)
    wb_ptr,
    # Offset table (int64)
    ot_ptr,
    # Scratch (batch, SCRATCH_SIZE) fp16
    sc_ptr, sc_batch_stride,
    # SSM state (batch, n_layers, nheads, headdim, d_state) fp16
    ssm_ptr, ssm_b_stride, ssm_l_stride, ssm_h_stride, ssm_d_stride, ssm_s_stride,
    # SSM k_state (batch, n_layers, nheads, d_state) fp16
    kst_ptr, kst_b_stride, kst_l_stride, kst_h_stride, kst_s_stride,
    # SSM v_state (batch, n_layers, nheads, headdim) fp16
    vst_ptr, vst_b_stride, vst_l_stride, vst_h_stride, vst_d_stride,
    # SSM angle_state (batch, n_layers, nheads, n_angles) fp16
    ast_ptr, ast_b_stride, ast_l_stride, ast_h_stride,
    # Cross-attn K cache (batch, n_layers, n_ca_heads, src_len, ca_hd) fp16
    kc_ptr, kc_b_stride, kc_l_stride, kc_nh_stride, kc_s_stride,
    # Cross-attn V cache (same layout)
    vc_ptr, vc_b_stride, vc_l_stride, vc_nh_stride, vc_s_stride,
    # Src padding mask (batch, src_len) bool
    mask_ptr, mask_b_stride,
    # Scalars
    src_len, bos_id, eos_id, pad_id, max_steps,
    # Constexpr
    N_LAYERS: tl.constexpr,
    D_MODEL: tl.constexpr,
    VOCAB: tl.constexpr,
):
    bid = tl.program_id(0)
    POS_SCALE: tl.constexpr = 19.595917942265423  # sqrt(384)

    # Scratch pointer for this sample
    sc = sc_ptr + bid * sc_batch_stride
    X = sc + 0           # hidden state: [0..384)
    IP = sc + 384         # in_proj out: [384..2100)
    TMP = sc + 2100       # temp 384: [2100..2484)
    FFMID = sc + 2484     # FFN mid 1536: [2484..4020)
    QBUF = sc + 4020      # query 384: [4020..4404)
    AOUT = sc + 4404      # attn out 384: [4404..4788)
    B_BASE = sc + 4788    # base B after BCNorm (64): [4788..4852)
    C_BASE = sc + 4852    # base C after BCNorm (64): [4852..4916)

    # Weight offsets
    embed_off = tl.load(ot_ptr + 0)
    dn_w_off = tl.load(ot_ptr + 1)
    dn_b_off = tl.load(ot_ptr + 2)

    cur_tok = bos_id
    finished: tl.constexpr = False  # can't use bool in persistent loop, use int
    fin = 0

    for step in tl.range(0, max_steps):
        if fin == 1:
            tl.store(out_ptr + bid * out_batch_stride + step, pad_id)
        else:
            # ── Embed ──
            offs = tl.arange(0, 512)  # next power of 2 above 384
            d_mask = offs < D_MODEL
            # Embedding is (VOCAB, D_MODEL) row-major: W[tok, d] at embed_off + tok*D_MODEL + d
            emb = tl.load(wb_ptr + embed_off + cur_tok * D_MODEL + offs, mask=d_mask, other=0.0).to(tl.float32) * POS_SCALE
            tl.store(X + offs, emb.to(tl.float16), mask=d_mask)

            # ── Decoder layers ──
            d_offs = tl.arange(0, 512)
            d_mask = d_offs < 384

            for li in range(N_LAYERS):
                base = 3 + li * 27
                # Load all weight offsets for this layer
                sn_w  = tl.load(ot_ptr + base + 0)
                sn_b  = tl.load(ot_ptr + base + 1)
                ip_w  = tl.load(ot_ptr + base + 2)
                ip_b  = tl.load(ot_ptr + base + 3)
                znw   = tl.load(ot_ptr + base + 4)   # z_norm weight
                xnw   = tl.load(ot_ptr + base + 5)   # x_norm weight
                dtb   = tl.load(ot_ptr + base + 6)   # dt_bias
                Doff  = tl.load(ot_ptr + base + 7)   # D
                Bboff = tl.load(ot_ptr + base + 8)   # B_bias (768 flat = 12*64)
                Cboff = tl.load(ot_ptr + base + 9)   # C_bias (768 flat)
                Bnw   = tl.load(ot_ptr + base + 10)  # B_norm weight (64)
                Cnw   = tl.load(ot_ptr + base + 11)  # C_norm weight (64)
                op_w  = tl.load(ot_ptr + base + 12)
                op_b  = tl.load(ot_ptr + base + 13)
                cn_w  = tl.load(ot_ptr + base + 14)
                cn_b  = tl.load(ot_ptr + base + 15)
                cqw   = tl.load(ot_ptr + base + 16)  # ca q_proj weight
                cqb   = tl.load(ot_ptr + base + 17)
                cow   = tl.load(ot_ptr + base + 18)  # ca out_proj weight
                cob   = tl.load(ot_ptr + base + 19)
                cqnw  = tl.load(ot_ptr + base + 20)  # ca q_norm weight (64)
                fn_w  = tl.load(ot_ptr + base + 21)
                fn_b  = tl.load(ot_ptr + base + 22)
                f1w   = tl.load(ot_ptr + base + 23)  # ff1 weight (1536,384)
                f1b   = tl.load(ot_ptr + base + 24)
                f2w   = tl.load(ot_ptr + base + 25)  # ff2 weight (384,1536)
                f2b   = tl.load(ot_ptr + base + 26)

                # ════════════════════════════════════════════
                # 1. Self-norm → in_proj → SSM step → out_proj → residual
                # ════════════════════════════════════════════

                # Save residual
                res = tl.load(X + d_offs, mask=d_mask, other=0.0)

                # LayerNorm
                _layernorm_inline(X, wb_ptr + sn_w, wb_ptr + sn_b, TMP, 384, 512)

                # in_proj: TMP(384) @ ip_w(1716,384).T + ip_b → IP(1716)
                _vmm_inline(TMP, wb_ptr + ip_w, wb_ptr + ip_b, IP, 384, 1716)

                # Split in_proj output: z(768) x(768) B(64) C(64) dt(12) A(12) trap(12) angles(16)
                Z_off = IP + 0          # 768
                XV_off = IP + 768       # 768
                B_off = IP + 1536       # 64
                C_off = IP + 1600       # 64
                DT_off = IP + 1664      # 12
                A_off = IP + 1676       # 12
                TRAP_off = IP + 1688    # 12
                ANG_off = IP + 1700     # 16

                # RMSNorm on z and x (NormedInProj)
                _rmsnorm_inline(Z_off, wb_ptr + znw, Z_off, 768, 1024)
                _rmsnorm_inline(XV_off, wb_ptr + xnw, XV_off, 768, 1024)

                s64 = tl.arange(0, 64)
                s16 = tl.arange(0, 16)

                # RMSNorm on B and C (BCNorm), then save base values
                _rmsnorm_inline(B_off, wb_ptr + Bnw, B_off, 64, 64)
                _rmsnorm_inline(C_off, wb_ptr + Cnw, C_off, 64, 64)
                # Save BCNorm'd base (shared across all heads before per-head bias+rotary)
                tl.store(B_BASE + s64, tl.load(B_off + s64))
                tl.store(C_BASE + s64, tl.load(C_off + s64))

                # SSM step per head: 12 heads, headdim=64, d_state=64
                ssm_base = ssm_ptr + bid * ssm_b_stride + li * ssm_l_stride
                kst_base = kst_ptr + bid * kst_b_stride + li * kst_l_stride
                vst_base = vst_ptr + bid * vst_b_stride + li * vst_l_stride

                for h in range(12):
                    # Load scalars
                    raw_A = tl.load(A_off + h).to(tl.float32)
                    raw_dt = tl.load(DT_off + h).to(tl.float32)
                    raw_trap = tl.load(TRAP_off + h).to(tl.float32)

                    # Discretize matching Mamba3._preprocess:
                    #   A = clamp(-softplus(A_raw), max=-A_floor)  → negative
                    #   DT = softplus(dt_raw + dt_bias)
                    #   decay = exp(A * DT)  → in (0, 1)
                    A_NEG_FLOOR: tl.constexpr = -0.0001
                    neg_sp_A = -tl.log(1.0 + tl.exp(raw_A))  # -softplus
                    A_clamped = tl.minimum(neg_sp_A, A_NEG_FLOOR)  # clamp to max=-floor
                    dt_bias_h = tl.load(wb_ptr + dtb + h).to(tl.float32)
                    DT_val = tl.log(1.0 + tl.exp(raw_dt + dt_bias_h))  # softplus
                    decay = tl.exp(A_clamped * DT_val)  # exp(negative * positive) → (0, 1)
                    trap_val = tl.sigmoid(raw_trap)
                    D_val = tl.load(wb_ptr + Doff + h).to(tl.float32)

                    # Reload base B/C (before per-head bias) and add this head's bias
                    B_bias_h = tl.load(wb_ptr + Bboff + h * 64 + s64).to(tl.float32)
                    C_bias_h = tl.load(wb_ptr + Cboff + h * 64 + s64).to(tl.float32)
                    B_biased = tl.load(B_BASE + s64).to(tl.float32) + B_bias_h
                    C_biased = tl.load(C_BASE + s64).to(tl.float32) + C_bias_h
                    tl.store(B_off + s64, B_biased.to(tl.float16))
                    tl.store(C_off + s64, C_biased.to(tl.float16))

                    # ── Rotary embeddings on B (=K) and C (=Q) ──
                    ast_base_h = ast_ptr + bid * ast_b_stride + li * ast_l_stride + h * ast_h_stride
                    angles_in = tl.load(ANG_off + s16).to(tl.float32)
                    angle_state = tl.load(ast_base_h + s16).to(tl.float32)
                    PI: tl.constexpr = 3.141592653589793
                    TWO_PI: tl.constexpr = 6.283185307179586
                    new_angle = angle_state + tl.extra.cuda.libdevice.tanh(angles_in) * DT_val * PI
                    new_angle = new_angle - TWO_PI * tl.extra.cuda.libdevice.floor(new_angle / TWO_PI)
                    tl.store(ast_base_h + s16, new_angle)

                    # Apply pairwise rotary to first 32 elements of B and C (in scratch)
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

                    # Reload B and C with rotary applied
                    B_new = tl.load(B_off + s64).to(tl.float32)
                    C_new = tl.load(C_off + s64).to(tl.float32)

                    # Load x_val for this head (64 elements from XV)
                    x_h = tl.load(XV_off + h * 64 + s64).to(tl.float32)
                    z_h = tl.load(Z_off + h * 64 + s64).to(tl.float32)

                    # Load old k_state, v_state
                    k_old = tl.load(kst_base + h * kst_h_stride + s64 * kst_s_stride).to(tl.float32)
                    v_old = tl.load(vst_base + h * vst_h_stride + s64 * vst_d_stride).to(tl.float32)

                    # Reference: gamma = trap * dt (for NEW input)
                    #            beta  = decay * dt * (1-trap) (for OLD/prev input)
                    gamma = trap_val * DT_val        # coefficient for current v ⊗ k
                    beta  = decay * DT_val * (1.0 - trap_val)  # coefficient for prev v ⊗ k

                    # SSM recurrence per headdim row
                    for di in range(64):
                        # Load state row: (d_state=64,)
                        state_row = tl.load(
                            ssm_base + h * ssm_h_stride + di * ssm_d_stride + s64 * ssm_s_stride
                        ).to(tl.float32)

                        x_di = tl.load(XV_off + h * 64 + di).to(tl.float32)
                        v_di = tl.load(vst_base + h * vst_h_stride + di * vst_d_stride).to(tl.float32)

                        # Update: state = decay*state + gamma*x_new*B_new + beta*v_old*k_old
                        state_row = decay * state_row + gamma * x_di * B_new + beta * v_di * k_old

                        # Store updated state
                        tl.store(
                            ssm_base + h * ssm_h_stride + di * ssm_d_stride + s64 * ssm_s_stride,
                            state_row
                        )

                        # Output: dot(state, C) + D*x
                        y_di = tl.sum(state_row * C_new) + D_val * x_di
                        # Gating: y * silu(z)
                        z_di = tl.load(Z_off + h * 64 + di).to(tl.float32)
                        y_di = y_di * z_di * tl.sigmoid(z_di)

                        # Store to scratch (overwrites XV_off which we're done reading for this head)
                        tl.store(XV_off + h * 64 + di, y_di.to(tl.float16))

                    # Update k_state ← B_new, v_state ← x_h (original, saved before SSM loop)
                    tl.store(kst_base + h * kst_h_stride + s64 * kst_s_stride, B_new)
                    tl.store(vst_base + h * vst_h_stride + s64 * vst_d_stride, x_h)

                # out_proj: XV(768) @ op_w(384,768).T + op_b → TMP(384)
                _vmm_inline(XV_off, wb_ptr + op_w, wb_ptr + op_b, TMP, 768, 384)

                # Residual add
                tmp_vals = tl.load(TMP + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                tl.store(X + d_offs, (res.to(tl.float32) + tmp_vals).to(tl.float16), mask=d_mask)

                # ════════════════════════════════════════════
                # 2. Cross-norm → Cross-attention (cached K/V) → residual
                # ════════════════════════════════════════════

                res = tl.load(X + d_offs, mask=d_mask, other=0.0)
                _layernorm_inline(X, wb_ptr + cn_w, wb_ptr + cn_b, TMP, 384, 512)

                # q = q_proj(TMP) → QBUF(384)
                _vmm_inline(TMP, wb_ptr + cqw, wb_ptr + cqb, QBUF, 384, 384)

                # QK-norm on query: RMSNorm per head (6 heads × 64 dim)
                for h in range(6):
                    _rmsnorm_inline(QBUF + h * 64, wb_ptr + cqnw, QBUF + h * 64, 64, 64)

                # Cross-attention: Q(6,1,64) @ K(6,S,64).T → softmax → @ V(6,S,64)
                # K cache: (batch, layers, 6, src_len, 64)
                # V cache: same layout
                kc_base = kc_ptr + bid * kc_b_stride + li * kc_l_stride
                vc_base = vc_ptr + bid * vc_b_stride + li * vc_l_stride
                CA_SCALE: tl.constexpr = 0.125  # 1/sqrt(64)

                # Vectorized cross-attention: process all src positions in parallel
                # K/V cache layout: (batch, layers, ca_heads, src_len, 64)
                # kc_s_stride = stride for src_len dimension = 64
                # Load padding mask once as vector
                s_offs = tl.arange(0, 1024)  # MAX_SRC bucket (power of 2)
                s_mask_valid = s_offs < src_len
                pad_flags = tl.load(mask_ptr + bid * mask_b_stride + s_offs,
                                    mask=s_mask_valid, other=True)  # True = pad = ignore
                active = s_mask_valid & ~pad_flags

                for h in range(6):
                    # Compute scores: q(64,) dot K(src_len, 64) for all positions
                    scores = tl.zeros((1024,), dtype=tl.float32)
                    for d in range(64):
                        q_d = tl.load(QBUF + h * 64 + d).to(tl.float32)
                        k_col = tl.load(kc_base + h * kc_nh_stride + s_offs * kc_s_stride + d,
                                        mask=s_mask_valid, other=0.0).to(tl.float32)
                        scores += q_d * k_col
                    scores = scores * CA_SCALE

                    # Masked softmax
                    scores = tl.where(active, scores, float('-inf'))
                    max_s = tl.max(scores)
                    exp_s = tl.exp(scores - max_s)
                    exp_s = tl.where(active, exp_s, 0.0)
                    sum_s = tl.sum(exp_s) + 1e-9
                    attn = exp_s / sum_s

                    # Weighted V sum: attn(src_len,) @ V(src_len, 64) → (64,)
                    for d in range(64):
                        v_col = tl.load(vc_base + h * vc_nh_stride + s_offs * vc_s_stride + d,
                                        mask=s_mask_valid, other=0.0).to(tl.float32)
                        val = tl.sum(attn * v_col)
                        tl.store(AOUT + h * 64 + d, val.to(tl.float16))

                # out_proj: AOUT(384) @ cow(384,384).T + cob → TMP(384)
                _vmm_inline(AOUT, wb_ptr + cow, wb_ptr + cob, TMP, 384, 384)

                # Residual add
                tmp_vals = tl.load(TMP + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                tl.store(X + d_offs, (res.to(tl.float32) + tmp_vals).to(tl.float16), mask=d_mask)

                # ════════════════════════════════════════════
                # 3. FF-norm → FFN (384→1536→384) → residual
                # ════════════════════════════════════════════

                res = tl.load(X + d_offs, mask=d_mask, other=0.0)
                _layernorm_inline(X, wb_ptr + fn_w, wb_ptr + fn_b, TMP, 384, 512)

                # FFN linear1: TMP(384) @ f1w(1536,384).T + f1b → FFMID(1536)
                _vmm_inline(TMP, wb_ptr + f1w, wb_ptr + f1b, FFMID, 384, 1536)

                # GELU activation on FFMID
                for g_start in tl.range(0, 1536, 64):
                    g_offs = g_start + tl.arange(0, 64)
                    g_mask = g_offs < 1536
                    gx = tl.load(FFMID + g_offs, mask=g_mask, other=0.0).to(tl.float32)
                    tl.store(FFMID + g_offs, (gx * tl.sigmoid(1.702 * gx)).to(tl.float16), mask=g_mask)

                # FFN linear2: FFMID(1536) @ f2w(384,1536).T + f2b → TMP(384)
                _vmm_inline(FFMID, wb_ptr + f2w, wb_ptr + f2b, TMP, 1536, 384)

                # Residual add
                tmp_vals = tl.load(TMP + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                tl.store(X + d_offs, (res.to(tl.float32) + tmp_vals).to(tl.float16), mask=d_mask)

            # ── Final norm + output proj + argmax ──
            _layernorm_inline(X, wb_ptr + dn_w_off, wb_ptr + dn_b_off, TMP, D_MODEL, 512)

            # Output proj: score[v] = dot(x, embed[v]) for all v, then argmax
            # Embedding is (VOCAB, D_MODEL) row-major.
            d_offs = tl.arange(0, 512)
            d_mask = d_offs < D_MODEL
            x_vec = tl.load(TMP + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            best_score = float('-inf')
            best_v = 0
            for v in range(VOCAB):
                e_vec = tl.load(wb_ptr + embed_off + v * D_MODEL + d_offs,
                                mask=d_mask, other=0.0).to(tl.float32)
                score = tl.sum(x_vec * e_vec)
                if score > best_score:
                    best_score = score
                    best_v = v

            cur_tok = best_v
            tl.store(out_ptr + bid * out_batch_stride + step, cur_tok)

            if cur_tok == eos_id:
                fin = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Python wrapper
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ar_prepare(model, src_ids, sp, max_len=1536, src_key_padding_mask=None):
    """Prepare AR decode: encode, KV cache, sync weights, allocate buffers.

    Returns a context dict that ar_launch() uses to fire the kernel.
    Runs on the current CUDA stream — uses model encoder + cross-attn layers.
    """
    device = src_ids.device
    batch = src_ids.shape[0]
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    with torch.amp.autocast('cuda', dtype=torch.float16):
        memory = model.encode(src_ids)
    K_all, V_all = precompute_kv_cache(model, memory, src_key_padding_mask)

    if not hasattr(model, '_ar_weight_blob'):
        pack_weights(model)
    sync_weights(model)

    src_len = src_ids.shape[1]
    n_layers = len(model.decoder_layers)

    # Allocate scratch + SSM states + output
    scratch = torch.zeros(batch, SCRATCH_SIZE, dtype=torch.float16, device=device)
    out_ids = torch.full((batch, max_len), pad_id, dtype=torch.int32, device=device)
    ssm_state = torch.zeros(batch, n_layers, 12, 64, 64, dtype=torch.float16, device=device)
    kst_state = torch.zeros(batch, n_layers, 12, 64, dtype=torch.float16, device=device)
    vst_state = torch.zeros(batch, n_layers, 12, 64, dtype=torch.float16, device=device)
    ast_state = torch.zeros(batch, n_layers, 12, 16, dtype=torch.float16, device=device)

    if src_key_padding_mask is None:
        src_mask = torch.zeros(batch, src_len, dtype=torch.bool, device=device)
    else:
        src_mask = src_key_padding_mask

    return {
        "batch": batch, "src_len": src_len, "max_len": max_len, "n_layers": n_layers,
        "bos_id": bos_id, "eos_id": eos_id, "pad_id": pad_id,
        "wb": model._ar_weight_blob, "ot": model._ar_offset_table,
        "scratch": scratch, "out_ids": out_ids,
        "ssm_state": ssm_state, "kst_state": kst_state,
        "vst_state": vst_state, "ast_state": ast_state,
        "K_all": K_all, "V_all": V_all, "src_mask": src_mask,
    }


def ar_launch(ctx):
    """Launch the persistent decode kernel. Async — returns immediately.

    Can be launched on any CUDA stream (use torch.cuda.stream context).
    """
    grid = (ctx["batch"],)
    out_ids = ctx["out_ids"]
    _persistent_decode_kernel[grid](
        out_ids, out_ids.stride(0),
        ctx["wb"], ctx["ot"],
        ctx["scratch"], ctx["scratch"].stride(0),
        ctx["ssm_state"], ctx["ssm_state"].stride(0), ctx["ssm_state"].stride(1),
        ctx["ssm_state"].stride(2), ctx["ssm_state"].stride(3), ctx["ssm_state"].stride(4),
        ctx["kst_state"], ctx["kst_state"].stride(0), ctx["kst_state"].stride(1),
        ctx["kst_state"].stride(2), ctx["kst_state"].stride(3),
        ctx["vst_state"], ctx["vst_state"].stride(0), ctx["vst_state"].stride(1),
        ctx["vst_state"].stride(2), ctx["vst_state"].stride(3),
        ctx["ast_state"], ctx["ast_state"].stride(0), ctx["ast_state"].stride(1),
        ctx["ast_state"].stride(2),
        ctx["K_all"], ctx["K_all"].stride(0), ctx["K_all"].stride(1),
        ctx["K_all"].stride(2), ctx["K_all"].stride(3),
        ctx["V_all"], ctx["V_all"].stride(0), ctx["V_all"].stride(1),
        ctx["V_all"].stride(2), ctx["V_all"].stride(3),
        ctx["src_mask"], ctx["src_mask"].stride(0),
        ctx["src_len"], ctx["bos_id"], ctx["eos_id"], ctx["pad_id"], ctx["max_len"],
        N_LAYERS=ctx["n_layers"],
        D_MODEL=384,
        VOCAB=8000,
    )


def ar_collect(ctx):
    """Read decode results from GPU. Call after kernel stream is synchronized."""
    out_ids = ctx["out_ids"]
    eos_id = ctx["eos_id"]
    pad_id = ctx["pad_id"]
    result = []
    ids_cpu = out_ids.cpu().tolist()
    for i in range(ctx["batch"]):
        ids = ids_cpu[i]
        if eos_id in ids:
            ids = ids[:ids.index(eos_id) + 1]
        else:
            while ids and ids[-1] == pad_id:
                ids.pop()
        result.append(ids)
    return result


@torch.no_grad()
def ar_decode_fused(model, src_ids, sp, max_len=1536, src_key_padding_mask=None):
    """AR decode with fused persistent kernel (synchronous convenience wrapper)."""
    ctx = ar_prepare(model, src_ids, sp, max_len, src_key_padding_mask)
    ar_launch(ctx)
    torch.cuda.current_stream().synchronize()
    return ar_collect(ctx)
