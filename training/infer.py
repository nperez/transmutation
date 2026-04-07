# Copyright (C) 2026 Nicholas Perez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Inference with optional GPU-accelerated Mamba3 decode via Triton kernels."""

import json
import math
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _mamba3_step_kernel(
        # Pre-split inputs (after in_proj + RMSNorm + bias).
        # Q=C, K=B already have bias added. (B, H, D_STATE) contiguous.
        Q_ptr, K_ptr, V_ptr, Z_ptr,
        # Scalars per (B, H), contiguous.
        ADT_ptr, DT_ptr, Trap_ptr,
        # Angles input (B, H, ANGLES) contiguous.
        Angles_ptr,
        # Per-head: D (H,).
        D_ptr,
        # States — all contiguous, updated in-place.
        Angle_State_ptr,    # (B, H, ANGLES) f32
        SSM_State_ptr,      # (B, H, HEADDIM, D_STATE) f32
        K_State_ptr,        # (B, H, D_STATE) f32
        V_State_ptr,        # (B, H, HEADDIM) f16/f32
        # Output (B, H, HEADDIM).
        Out_ptr,
        # Dims.
        nheads,
        HEADDIM: tl.constexpr,
        D_STATE: tl.constexpr,
        ANGLES: tl.constexpr,
    ):
        """Fused Mamba3 single-token decode step.

        Grid: (nheads, batch). One program per (head, batch).
        All tensors assumed contiguous with layout [B, H, ...].
        """
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        PI: tl.constexpr = 3.141592653589793

        offs_d = tl.arange(0, D_STATE)     # d_state indices
        offs_v = tl.arange(0, HEADDIM)     # headdim indices
        offs_r = tl.arange(0, ANGLES)      # rope angle indices

        # ── Flat offsets for contiguous (B, H, dim) layout ──
        bh = pid_b * nheads + pid_h
        base_qk = bh * D_STATE          # for Q, K, K_State (dim=D_STATE)
        base_v = bh * HEADDIM            # for V, Z, V_State, Out (dim=HEADDIM)
        base_a = bh * ANGLES             # for Angles, Angle_State
        base_scalar = pid_b * nheads + pid_h  # for ADT, DT, Trap (scalar per head)
        base_ssm = bh * (HEADDIM * D_STATE)   # for SSM_State (HEADDIM × D_STATE)

        # ── Load scalars ──
        adt = tl.load(ADT_ptr + base_scalar).to(tl.float32)
        dt = tl.load(DT_ptr + base_scalar).to(tl.float32)
        trap_raw = tl.load(Trap_ptr + base_scalar).to(tl.float32)
        trap = tl.sigmoid(trap_raw)

        # ── RoPE: accumulate angles, compute cos/sin ──
        angles_in = tl.load(Angles_ptr + base_a + offs_r).to(tl.float32)
        angle_state = tl.load(Angle_State_ptr + base_a + offs_r).to(tl.float32)
        TWO_PI: tl.constexpr = 6.283185307179586
        new_angle = angle_state + tl.extra.cuda.libdevice.tanh(angles_in) * dt * PI
        new_angle = new_angle - TWO_PI * tl.extra.cuda.libdevice.floor(new_angle / TWO_PI)  # mod 2π
        tl.store(Angle_State_ptr + base_a + offs_r, new_angle)
        cos_a = tl.cos(new_angle)
        sin_a = tl.sin(new_angle)

        # ── Load Q (=C+bias), K (=B+bias), apply pairwise rotary ──
        q = tl.load(Q_ptr + base_qk + offs_d).to(tl.float32)
        k = tl.load(K_ptr + base_qk + offs_d).to(tl.float32)

        # Rotary on pairs: dims [0,1], [2,3], ..., [2A-2, 2A-1].
        q_even = tl.load(Q_ptr + base_qk + offs_r * 2).to(tl.float32)
        q_odd = tl.load(Q_ptr + base_qk + offs_r * 2 + 1).to(tl.float32)
        k_even = tl.load(K_ptr + base_qk + offs_r * 2).to(tl.float32)
        k_odd = tl.load(K_ptr + base_qk + offs_r * 2 + 1).to(tl.float32)

        # Rotate.
        q_rot_e = q_even * cos_a - q_odd * sin_a
        q_rot_o = q_even * sin_a + q_odd * cos_a
        k_rot_e = k_even * cos_a - k_odd * sin_a
        k_rot_o = k_even * sin_a + k_odd * cos_a

        # Scatter rotary dims back. Non-rotary dims keep original values.
        # Use tl.where with index masks.
        dim_idx = tl.arange(0, D_STATE)
        is_rot_even = (dim_idx < ANGLES * 2) & (dim_idx % 2 == 0)
        is_rot_odd = (dim_idx < ANGLES * 2) & (dim_idx % 2 == 1)
        # Build rotary-applied Q: interleave even/odd results.
        # For rot dims: even positions get q_rot_e, odd get q_rot_o.
        # Map dim_idx -> rot pair index: dim_idx // 2.
        rot_idx = dim_idx // 2  # which angle pair
        # Gather from rotary results (only valid for rot_idx < ANGLES).
        q_re = tl.load(Q_ptr + base_qk + rot_idx * 2).to(tl.float32)  # reload even
        q_ro = tl.load(Q_ptr + base_qk + rot_idx * 2 + 1).to(tl.float32)
        q_cos = tl.load(Angle_State_ptr + base_a + rot_idx)  # just use new_angle
        # Actually this is getting circular. Let me use the reference pattern.

        # Reference approach: reshape to [D_STATE//2, 2], split, rotate, rejoin.
        # But Triton doesn't have great reshape support for this.
        # Simplest correct approach: load Q/K as flat, apply rotary via explicit indexing.

        # Build full rotated Q and K in registers using where().
        # Even rot positions: q[2i]*cos[i] - q[2i+1]*sin[i]
        # Odd rot positions:  q[2i]*sin[i] + q[2i+1]*cos[i]
        # Non-rot positions:  q[j] unchanged.
        rot_pair = dim_idx // 2
        cos_val = tl.where(rot_pair < ANGLES,
                          tl.load(Angle_State_ptr + base_a + tl.minimum(rot_pair, ANGLES - 1)),
                          0.0)
        # Recompute cos/sin from new_angle gathered at rot_pair index.
        # new_angle is in registers for offs_r. Need to "gather" for all dim_idx.
        # Since new_angle is only ANGLES wide, gather with clamp.
        ang_gathered = tl.load(Angle_State_ptr + base_a + tl.minimum(rot_pair, ANGLES - 1)).to(tl.float32)
        cos_g = tl.cos(ang_gathered)
        sin_g = tl.sin(ang_gathered)

        q_partner_even = tl.load(Q_ptr + base_qk + (dim_idx | 1)).to(tl.float32)   # odd partner
        q_partner_odd = tl.load(Q_ptr + base_qk + (dim_idx & ~1)).to(tl.float32)  # even partner
        q_rot = tl.where(is_rot_even, q * cos_g - q_partner_even * sin_g,
                 tl.where(is_rot_odd, q_partner_odd * sin_g + q * cos_g,
                          q))

        k_partner_even = tl.load(K_ptr + base_qk + (dim_idx | 1)).to(tl.float32)
        k_partner_odd = tl.load(K_ptr + base_qk + (dim_idx & ~1)).to(tl.float32)
        k_rot = tl.where(is_rot_even, k * cos_g - k_partner_even * sin_g,
                 tl.where(is_rot_odd, k_partner_odd * sin_g + k * cos_g,
                          k))

        # ── SSM recurrence (2D: HEADDIM × D_STATE) ──
        alpha = tl.math.exp2(adt * 1.44269504089)  # exp(x) = exp2(x/ln2), exp2 is single PTX insn
        beta = alpha * dt * (1.0 - trap)
        gamma = trap * dt

        v = tl.load(V_ptr + base_v + offs_v).to(tl.float32)
        v_prev = tl.load(V_State_ptr + base_v + offs_v).to(tl.float32)
        k_prev = tl.load(K_State_ptr + base_qk + offs_d).to(tl.float32)

        # State diff: outer products.  (HEADDIM,1) × (1,D_STATE) = (HEADDIM, D_STATE)
        diff = (gamma * v)[:, None] * k_rot[None, :] + (beta * v_prev)[:, None] * k_prev[None, :]

        # Load full state matrix, update, compute output via matmul.
        ssm = tl.load(SSM_State_ptr + base_ssm + offs_v[:, None] * D_STATE + offs_d[None, :]).to(tl.float32)
        ssm = alpha * ssm + diff
        tl.store(SSM_State_ptr + base_ssm + offs_v[:, None] * D_STATE + offs_d[None, :], ssm.to(tl.float16))

        # Output: state @ q_rot = (HEADDIM, D_STATE) × (D_STATE,) = (HEADDIM,)
        out = tl.sum(ssm * q_rot[None, :], axis=1)

        # D-skip + SiLU gating.
        D_val = tl.load(D_ptr + pid_h).to(tl.float32)
        out = out + D_val * v
        z = tl.load(Z_ptr + base_v + offs_v).to(tl.float32)
        out = out * z * tl.sigmoid(z)

        # Store output and updated K/V states.
        tl.store(Out_ptr + base_v + offs_v, out)
        tl.store(K_State_ptr + base_qk + offs_d, k_rot.to(tl.float16))
        tl.store(V_State_ptr + base_v + offs_v, v.to(tl.float16))


def mamba_forward_cpu(mamba, x):
    """Pure-Python Mamba forward pass replacing CUDA-only mamba_inner_fn."""
    batch, seq_len, d_model = x.shape
    d_inner = mamba.d_inner
    dt_rank = mamba.dt_rank
    d_state = mamba.d_state
    d_conv = mamba.d_conv

    # Project input to 2*d_inner (xz)
    xz = mamba.in_proj(x)  # (B, L, 2*d_inner)
    x_part, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

    # Causal conv1d (manual implementation)
    x_conv = x_part.transpose(1, 2)  # (B, d_inner, L)
    x_conv = F.pad(x_conv, (d_conv - 1, 0))  # left-pad
    x_conv = F.conv1d(x_conv, mamba.conv1d.weight, mamba.conv1d.bias, groups=d_inner)
    x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

    # SSM parameters
    x_dbl = mamba.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
    dt, B_param, C_param = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
    dt = mamba.dt_proj(dt)  # (B, L, d_inner)
    dt = F.softplus(dt)

    # Selective scan (sequential)
    A = -torch.exp(mamba.A_log.float())  # (d_inner, d_state)
    D = mamba.D.float()

    y = torch.zeros_like(x_conv)
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)

    for t in range(seq_len):
        dt_t = dt[:, t, :]  # (B, d_inner)
        B_t = B_param[:, t, :]  # (B, d_state)
        C_t = C_param[:, t, :]  # (B, d_state)
        x_t = x_conv[:, t, :].float()  # (B, d_inner)

        # Discretize: dA = exp(A * dt), dB = dt * B
        dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, d_inner, d_state)
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)

        # Update state: h = dA * h + dB * x
        h = dA * h + dB * x_t.unsqueeze(-1)

        # Output: y = (h @ C) + D * x
        y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        y_t = y_t + D.unsqueeze(0) * x_t
        y[:, t, :] = y_t

    # Gate and output
    y = y * F.silu(z)
    return mamba.out_proj(y)


def mamba_step_cpu(mamba, x, state):
    """Process a single token through Mamba, updating cached state in-place.

    Args:
        x: (B, 1, d_model) — single token embedding
        state: dict with 'h' (B, d_inner, d_state) and 'conv_buf' (B, d_inner, d_conv-1)

    Returns:
        output (B, 1, d_model), updated state
    """
    d_inner = mamba.d_inner
    d_conv = mamba.d_conv

    # Project input
    xz = mamba.in_proj(x)  # (B, 1, 2*d_inner)
    x_part, z = xz.chunk(2, dim=-1)  # each (B, 1, d_inner)
    x_part = x_part.squeeze(1)  # (B, d_inner)

    # Causal conv1d with cached buffer
    conv_buf = state["conv_buf"]  # (B, d_inner, d_conv-1)
    conv_input = torch.cat([conv_buf, x_part.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
    state["conv_buf"] = conv_input[:, :, 1:]  # shift buffer

    # Apply conv weights manually: sum over conv dimension per channel
    # conv1d weight shape: (d_inner, 1, d_conv) for groups=d_inner
    w = mamba.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
    x_conv = (conv_input * w.unsqueeze(0)).sum(dim=-1)  # (B, d_inner)
    if mamba.conv1d.bias is not None:
        x_conv = x_conv + mamba.conv1d.bias
    x_conv = F.silu(x_conv)  # (B, d_inner)

    # SSM parameters
    x_dbl = mamba.x_proj(x_conv.unsqueeze(1)).squeeze(1)  # (B, dt_rank + 2*d_state)
    dt_rank = mamba.dt_rank
    d_state = mamba.d_state
    dt, B_param, C_param = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
    dt = mamba.dt_proj(dt.unsqueeze(1)).squeeze(1)  # (B, d_inner)
    dt = F.softplus(dt)

    # Selective scan (single step)
    A = -torch.exp(mamba.A_log.float())  # (d_inner, d_state)
    D = mamba.D.float()
    x_t = x_conv.float()

    dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (B, d_inner, d_state)
    dB = dt.unsqueeze(-1) * B_param.unsqueeze(1)  # (B, d_inner, d_state)

    h = state["h"]
    h = dA * h + dB * x_t.unsqueeze(-1)
    state["h"] = h

    y_t = (h * C_param.unsqueeze(1)).sum(dim=-1) + D.unsqueeze(0) * x_t  # (B, d_inner)

    # Gate and output
    y_t = y_t * F.silu(z.squeeze(1))  # (B, d_inner)
    return mamba.out_proj(y_t.unsqueeze(1))  # (B, 1, d_model)


def init_mamba_state(mamba, batch, device):
    """Create initial empty state for incremental Mamba decoding."""
    return {
        "h": torch.zeros(batch, mamba.d_inner, mamba.d_state, device=device, dtype=torch.float32),
        "conv_buf": torch.zeros(batch, mamba.d_inner, mamba.d_conv - 1, device=device),
    }


def _rms_norm_cpu(x, weight, eps=1e-5):
    """Pure-PyTorch RMSNorm replacing Triton-based RMSNormGated."""
    x_f32 = x.float()
    rms = (x_f32 * x_f32).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * weight.float()).to(x.dtype)


def _apply_rotary_pairwise_cpu(q, cos, sin, rotary_dim):
    """Apply pairwise rotary embedding to first rotary_dim dimensions."""
    q_f32 = q.float()
    q0 = q_f32[..., 0:rotary_dim:2]
    q1 = q_f32[..., 1:rotary_dim:2]
    out = q_f32.clone()
    out[..., 0:rotary_dim:2] = q0 * cos - q1 * sin
    out[..., 1:rotary_dim:2] = q0 * sin + q1 * cos
    return out.to(q.dtype)


def init_mamba3_state(mamba3, batch, device):
    """Create initial empty state for incremental Mamba3 decoding.

    States in fp32 to prevent quantization drift over hundreds of decode steps.
    Training's parallel scan computes internally in fp32; step decode must match.
    """
    return {
        "type": "mamba3",
        "angle_state": torch.zeros(batch, mamba3.nheads, mamba3.num_rope_angles,
                                   device=device, dtype=torch.float32),
        "ssm_state": torch.zeros(batch, mamba3.nheads, mamba3.headdim, mamba3.d_state,
                                 device=device, dtype=torch.float32),
        "k_state": torch.zeros(batch, mamba3.nheads, mamba3.d_state,
                               device=device, dtype=torch.float32),
        "v_state": torch.zeros(batch, mamba3.nheads, mamba3.headdim,
                               device=device, dtype=torch.float32),
    }


def mamba3_step_triton(mamba3, x, state):
    """Single-token Mamba3 step using fused Triton kernel for SSM recurrence."""
    u = x.squeeze(1)  # (batch, d_model)

    # in_proj (may be NormedInProj with RMSNorm on z/x_val).
    zxBCdt = mamba3.in_proj(u)
    z, x_val, B_raw, C_raw, dd_dt, dd_A, trap_raw, angles_raw = torch.split(
        zxBCdt,
        [mamba3.d_inner, mamba3.d_inner,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.nheads, mamba3.nheads, mamba3.nheads, mamba3.num_rope_angles],
        dim=-1,
    )

    # Discretization (small per-head scalars, not worth fusing).
    A = torch.clamp(-F.softplus(dd_A.float()), max=-mamba3.A_floor)
    DT = F.softplus(dd_dt + mamba3.dt_bias)
    ADT = A * DT  # combined for kernel

    # RMSNorm on B, C then add bias — kernel expects bias-added Q/K.
    B_norm = _rms_norm_cpu(B_raw.view(-1, 1, mamba3.num_bc_heads, mamba3.d_state), mamba3.B_norm.weight)
    C_norm = _rms_norm_cpu(C_raw.view(-1, 1, mamba3.num_bc_heads, mamba3.d_state), mamba3.C_norm.weight)
    K = (B_norm.expand(-1, -1, mamba3.nheads, -1).squeeze(1).float()
         + mamba3.B_bias.squeeze(1).float()).contiguous()       # (B, H, d_state)
    Q = (C_norm.expand(-1, -1, mamba3.nheads, -1).squeeze(1).float()
         + mamba3.C_bias.squeeze(1).float()).contiguous()       # (B, H, d_state)

    V = x_val.view(-1, mamba3.nheads, mamba3.headdim).contiguous()
    Z = z.view(-1, mamba3.nheads, mamba3.headdim).contiguous()
    Angles = angles_raw.unsqueeze(1).expand(-1, mamba3.nheads, -1).contiguous()

    batch_size = u.shape[0]
    nheads = mamba3.nheads
    Out = torch.empty(batch_size, nheads, mamba3.headdim, device=u.device, dtype=torch.float32)

    _mamba3_step_kernel[(nheads, batch_size)](
        Q, K, V, Z,
        ADT.contiguous(), DT.contiguous(), trap_raw.contiguous(),
        Angles, mamba3.D,
        state["angle_state"], state["ssm_state"],
        state["k_state"], state["v_state"],
        Out,
        nheads,
        HEADDIM=mamba3.headdim,
        D_STATE=mamba3.d_state,
        ANGLES=mamba3.num_rope_angles,
    )

    out_flat = Out.reshape(batch_size, mamba3.d_inner).to(x.dtype)
    return mamba3.out_proj(out_flat).unsqueeze(1)


def _mamba3_step(mamba3, x, state):
    """Dispatch: Triton kernel on CUDA, pure PyTorch fallback on CPU."""
    if HAS_TRITON and x.is_cuda:
        return mamba3_step_triton(mamba3, x, state)
    return mamba3_step_cpu(mamba3, x, state)


def mamba3_step_cpu(mamba3, x, state):
    """Single-token Mamba3 step in pure PyTorch (fallback when Triton unavailable)."""
    u = x.squeeze(1)  # (batch, d_model)

    zxBCdt = mamba3.in_proj(u)
    z, x_val, B, C, dd_dt, dd_A, trap_raw, angles_raw = torch.split(
        zxBCdt,
        [mamba3.d_inner, mamba3.d_inner,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.nheads, mamba3.nheads, mamba3.nheads, mamba3.num_rope_angles],
        dim=-1,
    )

    A = torch.clamp(-F.softplus(dd_A.float()), max=-mamba3.A_floor)
    DT = F.softplus(dd_dt + mamba3.dt_bias)
    trap = torch.sigmoid(trap_raw)

    rank = 1
    B = _rms_norm_cpu(B.view(-1, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.B_norm.weight)
    C = _rms_norm_cpu(C.view(-1, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.C_norm.weight)
    B = B.expand(-1, -1, mamba3.nheads, -1)
    C = C.expand(-1, -1, mamba3.nheads, -1)

    x_val = x_val.view(-1, mamba3.nheads, mamba3.headdim)
    z = z.view(-1, mamba3.nheads, mamba3.headdim)

    angles = angles_raw.unsqueeze(-2).expand(-1, mamba3.nheads, -1)

    # Rotary: accumulate angle_state
    tanh_proj = torch.tanh(angles.float())
    dt_exp = DT.float().unsqueeze(-1)
    new_angle_state = state["angle_state"].float() + tanh_proj * dt_exp * math.pi
    cos_a = torch.cos(new_angle_state).unsqueeze(1)
    sin_a = torch.sin(new_angle_state).unsqueeze(1)

    B_bias = mamba3.B_bias.squeeze(1)
    C_bias = mamba3.C_bias.squeeze(1)
    B_biased = B.float() + B_bias.float().unsqueeze(0).unsqueeze(0)
    C_biased = C.float() + C_bias.float().unsqueeze(0).unsqueeze(0)

    rotary_dim = mamba3.num_rope_angles * 2
    B_rot = _apply_rotary_pairwise_cpu(B_biased, cos_a, sin_a, rotary_dim)
    C_rot = _apply_rotary_pairwise_cpu(C_biased, cos_a, sin_a, rotary_dim)

    B_cur = B_rot.squeeze(1).float()
    C_cur = C_rot.squeeze(1).float()

    alpha = torch.exp(A.float() * DT.float())
    gamma = trap.float() * DT.float()
    beta = (1.0 - trap.float()) * DT.float() * alpha

    x_f32 = x_val.float()
    x_gamma = x_f32 * gamma.unsqueeze(-1)
    x_beta = state["v_state"].float() * beta.unsqueeze(-1)
    k_prev = state["k_state"].float()

    alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1)
    new_state = (alpha_4d * state["ssm_state"]
                 + x_gamma.unsqueeze(-1) * B_cur.unsqueeze(-2)
                 + x_beta.unsqueeze(-1) * k_prev.unsqueeze(-2))

    out = (new_state * C_cur.unsqueeze(-2)).sum(dim=-1)
    out = out + mamba3.D.float().unsqueeze(0).unsqueeze(-1) * x_f32
    out = out * F.silu(z.float())

    state["angle_state"].copy_(new_angle_state)
    state["ssm_state"].copy_(new_state)
    state["k_state"].copy_(B_cur.to(state["k_state"].dtype))
    state["v_state"].copy_(x_val.to(state["v_state"].dtype))

    batch_size = u.shape[0]
    out_flat = out.view(batch_size, mamba3.d_inner).to(x.dtype)
    return mamba3.out_proj(out_flat).unsqueeze(1)


def mamba3_forward_cpu(mamba3, x):
    """Pure-PyTorch Mamba3 full-sequence forward pass."""
    batch, seq_len, d_model = x.shape

    zxBCdt = mamba3.in_proj(x)
    z, x_val, B, C, dd_dt, dd_A, trap_raw, angles_raw = torch.split(
        zxBCdt,
        [mamba3.d_inner, mamba3.d_inner,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.d_state * mamba3.num_bc_heads * mamba3.mimo_rank,
         mamba3.nheads, mamba3.nheads, mamba3.nheads, mamba3.num_rope_angles],
        dim=-1,
    )

    z = z.view(batch, seq_len, mamba3.nheads, mamba3.headdim)
    x_val = x_val.view(batch, seq_len, mamba3.nheads, mamba3.headdim)

    rank = 1
    B = _rms_norm_cpu(B.view(batch, seq_len, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.B_norm.weight).squeeze(2)
    C = _rms_norm_cpu(C.view(batch, seq_len, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.C_norm.weight).squeeze(2)

    A = torch.clamp(-F.softplus(dd_A.float()), max=-mamba3.A_floor)
    DT = F.softplus(dd_dt + mamba3.dt_bias)
    trap = torch.sigmoid(trap_raw)

    angles_expanded = angles_raw.unsqueeze(-2).expand(-1, -1, mamba3.nheads, -1)
    tanh_angles = torch.tanh(angles_expanded.float())
    angle_increments = tanh_angles * DT.float().unsqueeze(-1) * math.pi
    angles_cumsum = torch.cumsum(angle_increments, dim=1)

    cos_a = torch.cos(angles_cumsum)
    sin_a = torch.sin(angles_cumsum)

    B_bias = mamba3.B_bias.squeeze(1)
    C_bias = mamba3.C_bias.squeeze(1)
    B_expanded = B.expand(-1, -1, mamba3.nheads, -1).float() + B_bias.float().unsqueeze(0).unsqueeze(0)
    C_expanded = C.expand(-1, -1, mamba3.nheads, -1).float() + C_bias.float().unsqueeze(0).unsqueeze(0)

    rotary_dim = mamba3.num_rope_angles * 2
    B_rot = _apply_rotary_pairwise_cpu(B_expanded, cos_a, sin_a, rotary_dim)
    C_rot = _apply_rotary_pairwise_cpu(C_expanded, cos_a, sin_a, rotary_dim)

    D = mamba3.D.float()
    ssm_state = torch.zeros(batch, mamba3.nheads, mamba3.headdim, mamba3.d_state,
                            device=x.device, dtype=torch.float32)
    k_prev = torch.zeros(batch, mamba3.nheads, mamba3.d_state, device=x.device, dtype=torch.float32)
    v_prev = torch.zeros(batch, mamba3.nheads, mamba3.headdim, device=x.device, dtype=torch.float32)

    y = torch.zeros(batch, seq_len, mamba3.nheads, mamba3.headdim, device=x.device, dtype=torch.float32)

    for t in range(seq_len):
        x_t = x_val[:, t, :, :].float()
        B_t = B_rot[:, t, :, :].float()
        C_t = C_rot[:, t, :, :].float()
        A_t = A[:, t, :].float()
        DT_t = DT[:, t, :].float()
        trap_t = trap[:, t, :].float()
        z_t = z[:, t, :, :].float()

        alpha_t = torch.exp(A_t * DT_t)
        gamma_t = trap_t * DT_t
        beta_t = (1.0 - trap_t) * DT_t * alpha_t

        x_gamma = x_t * gamma_t.unsqueeze(-1)
        x_beta = v_prev * beta_t.unsqueeze(-1)

        alpha_4d = alpha_t.unsqueeze(-1).unsqueeze(-1)
        ssm_state = (alpha_4d * ssm_state
                     + x_gamma.unsqueeze(-1) * B_t.unsqueeze(-2)
                     + x_beta.unsqueeze(-1) * k_prev.unsqueeze(-2))

        y_t = (ssm_state * C_t.unsqueeze(-2)).sum(dim=-1)
        y_t = y_t + D.unsqueeze(0).unsqueeze(-1) * x_t
        y_t = y_t * F.silu(z_t)
        y[:, t, :, :] = y_t

        k_prev = B_t
        v_prev = x_t

    y_flat = y.view(batch, seq_len, mamba3.d_inner).to(x.dtype)
    return mamba3.out_proj(y_flat)


def patch_mamba_for_cpu(model):
    """Monkey-patch all Mamba/Mamba3 modules to use CPU forward pass."""
    for module in model.modules():
        name = type(module).__name__
        if name == "Mamba":
            module._original_forward = module.forward
            module.forward = lambda x, m=module: mamba_forward_cpu(m, x)
        elif name == "Mamba3":
            module._original_forward = module.forward
            module.forward = lambda x, m=module: mamba3_forward_cpu(m, x)


def greedy_decode(model, src_ids, sp, max_len=1536, device="cpu"):
    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory = model.encode(src)

        # Initialize cached state for each decoder layer's Mamba/Mamba3 block.
        layer_states = []
        for layer in model.decoder_layers:
            mamba = layer.self_mamba
            if type(mamba).__name__ == "Mamba3":
                layer_states.append(init_mamba3_state(mamba, 1, device))
            else:
                layer_states.append(init_mamba_state(mamba, 1, device))

        tgt_ids = [sp.bos_id()]
        for _ in range(max_len):
            tok = torch.tensor([[tgt_ids[-1]]], dtype=torch.long, device=device)
            x = model.embedding(tok) * model.pos_scale  # (1, 1, d_model)

            for layer, state in zip(model.decoder_layers, layer_states):
                # Mamba/Mamba3 step (incremental, cached).
                residual = x
                x = layer.self_norm(x)
                if state.get("type") == "mamba3":
                    x = _mamba3_step(layer.self_mamba, x, state)
                else:
                    x = mamba_step_cpu(layer.self_mamba, x, state)
                x = residual + x

                # Cross-attention.
                residual = x
                x = layer.cross_norm(x)
                x, _ = layer.cross_attn(x, memory)
                x = residual + x

                # Feedforward.
                residual = x
                x = layer.ff_norm(x)
                x = layer.ff(x)
                x = residual + x

            x = model.decoder_norm(x)
            logits = model.output_proj(x)  # (1, 1, vocab)
            next_id = logits[0, 0].argmax().item()

            if next_id == sp.eos_id():
                break
            tgt_ids.append(next_id)
    return tgt_ids[1:]


def batched_greedy_decode(model, src_ids_list, sp, max_len=1536, device="cuda"):
    """Batched greedy AR decode. All samples decoded in parallel on GPU.

    Zero GPU↔CPU syncs in the decode loop. All token accumulation on GPU.
    One bulk transfer at the end.

    src_ids_list: list of token ID lists (variable length).
    Returns: list of decoded token ID lists (without BOS).
    """
    B = len(src_ids_list)
    if B == 0:
        return []

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    # Batch-encode all sources. Right-padding is safe: Mamba3's left-to-right SSM
    # scan means padding after real content doesn't affect real positions' hidden states.
    src_lens = [len(s) for s in src_ids_list]
    max_src = max(src_lens)

    with torch.no_grad():
        # Encode in chunks to avoid OOM on large batches.
        pad_id = sp.pad_id()
        ENCODE_CHUNK = 32
        d_model = model.embedding.embedding_dim
        enc_dtype = next(model.parameters()).dtype
        memory = torch.zeros(B, max_src, d_model, dtype=enc_dtype, device=device)
        for c_start in range(0, B, ENCODE_CHUNK):
            c_end = min(c_start + ENCODE_CHUNK, B)
            c_lens = src_lens[c_start:c_end]
            c_max = max(c_lens)
            c_batch = torch.full((c_end - c_start, c_max), pad_id, dtype=torch.long, device=device)
            for i, s in enumerate(src_ids_list[c_start:c_end]):
                c_batch[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            c_mem = model.encode(c_batch)  # (chunk, c_max, d_model)
            memory[c_start:c_end, :c_max] = c_mem

        # Init Mamba states for batch. Cache type flags to avoid dict lookup per step.
        layer_states = []
        layer_is_mamba3 = []
        for layer in model.decoder_layers:
            mamba = layer.self_mamba
            is_m3 = type(mamba).__name__ == "Mamba3"
            layer_is_mamba3.append(is_m3)
            if is_m3:
                layer_states.append(init_mamba3_state(mamba, B, device))
            else:
                layer_states.append(init_mamba_state(mamba, B, device))

        # Source padding mask for cross-attention (True = ignore).
        src_pad_mask = torch.ones(B, max_src, dtype=torch.bool, device=device)
        for i, sl in enumerate(src_lens):
            src_pad_mask[i, :sl] = False

        # Per-sample max decode length on GPU.
        max_decode_t = torch.tensor(
            [min(max_len, sl * 2) for sl in src_lens],
            dtype=torch.long, device=device)
        global_max = max_decode_t.max().item()  # one sync

        # GPU output buffer + finished mask. Zero syncs in loop.
        output_buf = torch.full((B, global_max), pad_id, dtype=torch.long, device=device)
        output_lens = torch.zeros(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cur_tok = torch.full((B,), bos_id, dtype=torch.long, device=device)
        eos_t = torch.tensor(eos_id, dtype=torch.long, device=device)

        for step in range(global_max):
            # Early exit + progress: one sync per N steps (amortize).
            if step % 50 == 49:
                n_done = finished.sum().item()
                if n_done == B:
                    break
                print(f"  decode step {step+1}/{global_max} "
                      f"{n_done}/{B} finished ({100*n_done/B:.0f}%)", flush=True, file=sys.stderr)

            tok = cur_tok.unsqueeze(1)
            x = model.embedding(tok) * model.pos_scale

            for layer, state, is_m3 in zip(model.decoder_layers, layer_states, layer_is_mamba3):
                residual = x
                x = layer.self_norm(x)
                if is_m3:
                    x = _mamba3_step(layer.self_mamba, x, state)
                else:
                    x = mamba_step_cpu(layer.self_mamba, x, state)
                x = residual + x

                residual = x
                x = layer.cross_norm(x)
                x, _ = layer.cross_attn(x, memory, key_padding_mask=src_pad_mask)
                x = residual + x

                residual = x
                x = layer.ff_norm(x)
                x = layer.ff(x)
                x = residual + x

            x = model.decoder_norm(x)
            logits = model.output_proj(x)[:, 0, :]
            next_ids = logits.argmax(dim=-1)

            # All updates on GPU. Zero sync.
            just_finished = (~finished) & ((next_ids == eos_t) | (step >= max_decode_t))
            active = ~finished & ~just_finished
            output_buf[active, output_lens[active]] = next_ids[active]
            output_lens += active.long()
            finished |= just_finished
            cur_tok = next_ids

    # One bulk sync at the end.
    output_buf_cpu = output_buf.cpu().tolist()
    output_lens_cpu = output_lens.cpu().tolist()
    return [output_buf_cpu[i][:output_lens_cpu[i]] for i in range(B)]


def _decoder_step(model, x, layer_states, memory, src_ids=None):
    """Run one decoder step through all layers. x: (B, 1, d_model).

    Returns log-probs (B, vocab) if copy gate present and src_ids provided,
    otherwise returns logits (B, vocab).
    """
    B = x.size(0)
    mem = memory.expand(B, -1, -1) if memory.size(0) != B else memory
    has_copy = hasattr(model, 'copy_gate') and src_ids is not None
    n_layers = len(model.decoder_layers)

    attn_weights = None
    for li, (layer, state) in enumerate(zip(model.decoder_layers, layer_states)):
        residual = x
        x = layer.self_norm(x)
        if state.get("type") == "mamba3":
            x = mamba3_step_cpu(layer.self_mamba, x, state)
        else:
            x = mamba_step_cpu(layer.self_mamba, x, state)
        x = residual + x

        residual = x
        x = layer.cross_norm(x)
        is_last = (li == n_layers - 1)
        x, attn_w = layer.cross_attn(
            x, mem,
            need_weights=is_last and has_copy,
        )
        if is_last:
            attn_weights = attn_w
        x = residual + x

        residual = x
        x = layer.ff_norm(x)
        x = layer.ff(x)
        x = residual + x

    x = model.decoder_norm(x)
    gen_logits = model.output_proj(x)[:, 0, :]  # (B, vocab)

    if has_copy and attn_weights is not None:
        p_copy = torch.sigmoid(model.copy_gate(x[:, 0, :]))  # (B, 1)
        gen_probs = F.softmax(gen_logits, dim=-1)
        copy_probs = torch.zeros_like(gen_probs)
        src_expanded = src_ids.expand(B, -1)  # (B, src_len)
        copy_probs.scatter_add_(1, src_expanded, attn_weights[:, 0, :])  # (B, vocab)
        blended = (1 - p_copy) * gen_probs + p_copy * copy_probs
        return torch.log(blended + 1e-10)

    return gen_logits


def beam_decode(model, src_ids, sp, max_len=1536, beam_width=3,
                length_penalty=0.6, device="cpu"):
    """Beam search decoding. Falls back to greedy when beam_width <= 1."""
    if beam_width <= 1:
        return greedy_decode(model, src_ids, sp, max_len=max_len, device=device)

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory = model.encode(src)  # (1, src_len, d_model)
        has_copy = hasattr(model, 'copy_gate')
        src_tensor = src if has_copy else None

        # Start with a single beam (batch=1).
        layer_states = []
        for layer in model.decoder_layers:
            mamba = layer.self_mamba
            if type(mamba).__name__ == "Mamba3":
                layer_states.append(init_mamba3_state(mamba, 1, device))
            else:
                layer_states.append(init_mamba_state(mamba, 1, device))

        scores = torch.zeros(1, device=device)  # (K,)
        seqs = [[]]  # token lists, BOS excluded
        completed = []  # (score, token_list)

        current_token = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            K = current_token.size(0)
            x = model.embedding(current_token) * model.pos_scale  # (K, 1, d_model)
            step_out = _decoder_step(model, x, layer_states, memory, src_ids=src_tensor)  # (K, vocab)
            # _decoder_step returns log-probs when copy is active, logits otherwise.
            log_probs = step_out if has_copy else F.log_softmax(step_out, dim=-1)
            vocab_size = log_probs.size(-1)

            # Combined scores: (K, vocab)
            combined = scores.unsqueeze(-1) + log_probs

            # Top candidates across all beams.
            flat = combined.view(-1)
            n_candidates = min(beam_width * 2, flat.size(0))
            topk_scores, topk_flat = flat.topk(n_candidates)

            new_scores = []
            new_seqs = []
            parent_indices = []

            for s, f in zip(topk_scores.tolist(), topk_flat.tolist()):
                beam_idx = f // vocab_size
                token_id = f % vocab_size

                if token_id == eos_id:
                    completed.append((s, list(seqs[beam_idx])))
                elif len(new_scores) < beam_width:
                    new_scores.append(s)
                    new_seqs.append(seqs[beam_idx] + [token_id])
                    parent_indices.append(beam_idx)

            if not new_scores:
                break

            scores = torch.tensor(new_scores, device=device)
            seqs = new_seqs

            # Reindex Mamba states to match surviving beams.
            idx = torch.tensor(parent_indices, dtype=torch.long, device=device)
            for state in layer_states:
                state["h"] = state["h"][idx].clone()
                state["conv_buf"] = state["conv_buf"][idx].clone()

            current_token = torch.tensor(
                [[s[-1]] for s in seqs], dtype=torch.long, device=device,
            )

            # Early stop: no active beam can beat the best completed sequence
            # (log-probs are non-positive, so active scores only decrease).
            if completed:
                best_completed = max(c[0] for c in completed)
                best_active = scores.max().item()
                if best_completed >= best_active:
                    break

        # Add remaining active beams as completed.
        for s, seq in zip(scores.tolist(), seqs):
            completed.append((s, seq))

        if not completed:
            return []

        # Return best by length-normalized score.
        def normed(score, length):
            if length == 0 or length_penalty == 0:
                return score
            return score / length ** length_penalty

        best = max(completed, key=lambda c: normed(c[0], len(c[1])))
        return best[1]


def load_model(checkpoint, device):
    sp = spm.SentencePieceProcessor()
    tok_path = str(Path(checkpoint).parent / "tokenizer.model")
    sp.load(tok_path)

    sys.path.insert(0, str(Path(__file__).parent))
    from model import TransmutationModel

    model = TransmutationModel(
        vocab_size=sp.get_piece_size(), d_model=384,
        n_encoder_layers=6, n_decoder_layers=6,
        d_state=64, headdim=64, n_heads=6, pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        if all("copy_gate" in k for k in missing):
            nn.init.zeros_(model.copy_gate.weight)
            nn.init.constant_(model.copy_gate.bias, -5.0)
            print(f"  Initialized copy_gate from scratch (old checkpoint)")
        else:
            raise RuntimeError(f"Unexpected missing keys: {missing}")
    model.eval()
    patch_mamba_for_cpu(model)
    print(f"Loaded: epoch={ckpt['epoch']}, global_step={ckpt['global_step']}")
    print()
    return model, sp


def read_records(sp, max_src_len, input_file=None):
    """Read JSONL records from file or stdin, filtering by source token length."""
    source = open(input_file, encoding="utf-8") if input_file else sys.stdin
    try:
        for line in source:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            r = json.loads(line)
            if len(sp.encode(r["input"])) <= max_src_len:
                yield r
    finally:
        if input_file:
            source.close()


def xml_semantically_equal(a, b):
    """Compare two XML strings by flattening to canonical (element, text) tokens.

    Treats CDATA vs plain text as equivalent and ignores insignificant whitespace.
    """
    def flatten(s):
        try:
            root = ET.fromstring(s)
        except ET.ParseError:
            return None
        tokens = []
        def walk(elem):
            tokens.append(("start", elem.tag))
            text = re.sub(r"\s+", " ", (elem.text or "").strip())
            if text:
                tokens.append(("text", text))
            for child in elem:
                walk(child)
                tail = re.sub(r"\s+", " ", (child.tail or "").strip())
                if tail:
                    tokens.append(("text", tail))
            tokens.append(("end", elem.tag))
        walk(root)
        return tokens

    af, bf = flatten(a), flatten(b)
    if af is None or bf is None:
        return False
    return af == bf


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CPU inference for transmutation model")
    parser.add_argument("checkpoint", nargs="?", default="models/epoch_1.pt")
    parser.add_argument("-n", type=int, default=10, help="number of samples")
    parser.add_argument("--max-src-len", type=int, default=1152)
    parser.add_argument("--beam-width", type=int, default=1,
                        help="beam width (1 = greedy)")
    parser.add_argument("--length-penalty", type=float, default=0.6,
                        help="length normalization exponent for beam search")
    parser.add_argument("--json", action="store_true",
                        help="output JSONL per sample (for eval pipeline)")
    parser.add_argument("--input", type=str, default=None,
                        help="read JSONL from file instead of stdin")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model, sp = load_model(args.checkpoint, device)

    if args.beam_width > 1:
        print(f"Beam search: width={args.beam_width}, length_penalty={args.length_penalty}")
    print()

    # Collect records.
    records = []
    for rec in read_records(sp, args.max_src_len, input_file=args.input):
        if len(records) >= args.n:
            break
        records.append(rec)
    total = len(records)

    # Batched GPU decode or sequential CPU decode.
    if args.gpu and device.type == "cuda" and args.beam_width <= 1:
        src_ids_list = [sp.encode(r["input"]) for r in records]
        print(f"Batched GPU decode: {total} samples...", flush=True, file=sys.stderr)
        t0 = time.monotonic()
        all_pred_ids = batched_greedy_decode(model, src_ids_list, sp, device=device)
        batch_elapsed = time.monotonic() - t0
        print(f"Decode done in {batch_elapsed:.1f}s ({batch_elapsed/total:.2f}s/sample)", flush=True, file=sys.stderr)
        all_elapsed = [batch_elapsed / total] * total  # approximate per-sample
    else:
        all_pred_ids = []
        all_elapsed = []
        for rec in records:
            src_ids = sp.encode(rec["input"])
            t0 = time.monotonic()
            if args.beam_width > 1:
                pred_ids = beam_decode(model, src_ids, sp, device=device,
                                       beam_width=args.beam_width,
                                       length_penalty=args.length_penalty)
            else:
                pred_ids = greedy_decode(model, src_ids, sp, device=device)
            all_pred_ids.append(pred_ids)
            all_elapsed.append(time.monotonic() - t0)

    # Score and output.
    xml_ok_count = 0
    semantic_count = 0
    exact_count = 0

    for i, (rec, pred_ids, elapsed) in enumerate(zip(records, all_pred_ids, all_elapsed)):
        src_ids = sp.encode(rec["input"])
        pred = sp.decode(pred_ids)
        target = rec["target"]

        norm_pred = unicodedata.normalize("NFKD", re.sub(r"\s+", " ", pred.strip()))
        norm_tgt = unicodedata.normalize("NFKD", re.sub(r"\s+", " ", target.strip()))
        exact = norm_pred == norm_tgt

        try:
            ET.fromstring(pred.strip())
            xml_ok = True
        except ET.ParseError:
            xml_ok = False

        semantic = False
        if not exact and xml_ok:
            semantic = xml_semantically_equal(pred.strip(), target.strip())

        if exact:
            exact_count += 1
        if semantic:
            semantic_count += 1
        if xml_ok:
            xml_ok_count += 1

        tag = "EXACT" if exact else ("SEMANTIC" if semantic else ("XML_OK" if xml_ok else "FAIL"))

        if args.json:
            print(json.dumps({
                "sample": i + 1,
                "tag": tag,
                "src_tokens": len(src_ids),
                "pred_tokens": len(pred_ids),
                "tgt_tokens": len(sp.encode(target)),
                "elapsed": round(elapsed, 2),
                "exact": exact,
                "semantic": semantic,
                "xml_ok": xml_ok,
            }), flush=True)
        else:
            print(f"=== Sample {i+1} [{tag}] {elapsed:.2f}s, {len(pred_ids)} tokens ===")
            print(f"INPUT:\n{rec['input']}\n")
            if exact or semantic:
                print(f"OUTPUT (matches target):\n{pred.strip()}\n")
            else:
                print(f"TARGET:\n{target.strip()}\n")
                print(f"OUTPUT:\n{pred.strip()}\n")
            print()

    if not args.json:
        print(f"===== {total} samples: exact={exact_count} semantic={semantic_count} xml_ok={xml_ok_count - exact_count - semantic_count} fail={total - xml_ok_count} =====")


if __name__ == "__main__":
    main()
