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

"""Export trained model to ONNX format.

Encoder: full-sequence forward (called once per input).
Decoder: single-step with explicit SSM state as inputs/outputs.
         No loops in the decoder ONNX — the autoregressive loop runs in the caller.
         Supports both Mamba1 (h + conv_buf) and Mamba3 (angle + ssm + k + v) state.
"""

import argparse
import math
import os

import numpy as np
import onnx
import onnxruntime as ort
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TransmutationModel


# ---------------------------------------------------------------------------
# Encoder: full-sequence with scripted selective scan
# ---------------------------------------------------------------------------

@torch.jit.script
def selective_scan_loop(
    dt: torch.Tensor, B_param: torch.Tensor, C_param: torch.Tensor,
    x_conv: torch.Tensor, A: torch.Tensor, D: torch.Tensor,
) -> torch.Tensor:
    batch = x_conv.size(0)
    seq_len = x_conv.size(1)
    d_inner = x_conv.size(2)

    y = torch.zeros_like(x_conv)
    h = torch.zeros(batch, d_inner, A.size(1), device=x_conv.device, dtype=torch.float32)

    for t in range(seq_len):
        dt_t = dt[:, t, :]
        B_t = B_param[:, t, :]
        C_t = C_param[:, t, :]
        x_t = x_conv[:, t, :].to(torch.float32)
        dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
        h = dA * h + dB * x_t.unsqueeze(-1)
        y_t = (h * C_t.unsqueeze(1)).sum(-1) + D.unsqueeze(0) * x_t
        y[:, t, :] = y_t

    return y


def mamba_forward_onnx(mamba, x):
    batch, seq_len, _ = x.shape
    d_inner = mamba.d_inner
    d_conv = mamba.d_conv

    xz = mamba.in_proj(x)
    x_part, z = xz.chunk(2, dim=-1)

    x_conv = x_part.transpose(1, 2)
    x_conv = F.pad(x_conv, (d_conv - 1, 0))
    x_conv = F.conv1d(x_conv, mamba.conv1d.weight, mamba.conv1d.bias, groups=d_inner)
    x_conv = F.silu(x_conv).transpose(1, 2)

    x_dbl = mamba.x_proj(x_conv)
    dt, B_param, C_param = x_dbl.split([mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    dt = F.softplus(mamba.dt_proj(dt))

    A = -torch.exp(mamba.A_log.float())
    D = mamba.D.float()
    y = selective_scan_loop(dt, B_param, C_param, x_conv, A, D)
    y = y * F.silu(z)
    return mamba.out_proj(y)


def mamba3_forward_onnx(mamba3, x):
    """Pure-PyTorch Mamba3 full-sequence forward for ONNX encoder export."""
    from infer import mamba3_forward_cpu
    return mamba3_forward_cpu(mamba3, x)


def patch_mamba_for_onnx(model):
    for module in model.modules():
        name = type(module).__name__
        if name == "Mamba":
            module.forward = lambda x, m=module: mamba_forward_onnx(m, x)
        elif name == "Mamba3":
            module.forward = lambda x, m=module: mamba3_forward_onnx(m, x)


class EncoderWrapper(nn.Module):
    """Encoder that pre-computes cross-attention K/V for all decoder layers.

    This moves the expensive memory projection (src_len × d_model²) to the
    encoder (called once) instead of repeating it every decoder step.

    Outputs:
        all_k: (n_layers, n_heads, src_len, head_dim) - cached K projections
        all_v: (n_layers, n_heads, src_len, head_dim) - cached V projections
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids):
        memory = self.model.encode(src_ids)  # (1, src_len, d_model)
        batch = memory.size(0)

        k_list = []
        v_list = []
        for layer in self.model.decoder_layers:
            mha = layer.cross_attn
            w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
            k = F.linear(memory, w_k, b_k)  # (1, src_len, d_model)
            v = F.linear(memory, w_v, b_v)
            k = k.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)  # (1, n_heads, src_len, head_dim)
            v = v.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)
            k_list.append(k)
            v_list.append(v)

        all_k = torch.cat(k_list, dim=0)  # (n_layers, n_heads, src_len, head_dim)
        all_v = torch.cat(v_list, dim=0)  # (n_layers, n_heads, src_len, head_dim)
        return all_k, all_v


# ---------------------------------------------------------------------------
# Decoder: single-step with explicit Mamba state (no loops)
# ---------------------------------------------------------------------------

class SingleStepDecoderWrapper(nn.Module):
    """One decoder step with pre-computed KV cache and copy mechanism.

    Inputs:
        tgt_token: (1, 1) int64
        all_k: (n_layers, n_heads, src_len, head_dim) float32  — cached from encoder
        all_v: (n_layers, n_heads, src_len, head_dim) float32  — cached from encoder
        all_h: (n_layers, d_inner, d_state) float32
        all_conv: (n_layers, d_inner, d_conv-1) float32
        src_ids: (1, src_len) int64  — source token IDs for copy mechanism
    Outputs:
        log_probs: (1, vocab_size) float32  — blended log-probabilities
        all_h_out: (n_layers, d_inner, d_state) float32
        all_conv_out: (n_layers, d_inner, d_conv-1) float32
    """

    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_scale = model.pos_scale
        self.decoder_layers = model.decoder_layers
        self.decoder_norm = model.decoder_norm
        self.output_proj = model.output_proj
        self.copy_gate = model.copy_gate
        self.n_layers = len(model.decoder_layers)

    def forward(self, tgt_token, all_k, all_v, all_h, all_conv, src_ids):
        x = self.embedding(tgt_token) * self.pos_scale  # (1, 1, d_model)

        h_outs = []
        conv_outs = []
        attn_weights = None

        for i in range(self.n_layers):
            layer = self.decoder_layers[i]
            h_in = all_h[i:i+1]       # (1, d_inner, d_state)
            conv_in = all_conv[i:i+1]  # (1, d_inner, d_conv-1)
            k_i = all_k[i:i+1]        # (1, n_heads, src_len, head_dim)
            v_i = all_v[i:i+1]        # (1, n_heads, src_len, head_dim)

            # Mamba self-attention (single token, no loop).
            residual = x
            x_norm = layer.self_norm(x)
            x_step, h_new, conv_new = _mamba_step(layer.self_mamba, x_norm, h_in, conv_in)
            x = residual + x_step
            h_outs.append(h_new)
            conv_outs.append(conv_new)

            # Cross-attention with cached K/V (only Q projection needed per step).
            residual = x
            x_norm = layer.cross_norm(x)
            is_last = (i == self.n_layers - 1)
            if is_last:
                x_attn, attn_weights = _cross_attn_cached(layer.cross_attn, x_norm, k_i, v_i, return_attn=True)
            else:
                x_attn = _cross_attn_cached(layer.cross_attn, x_norm, k_i, v_i)
            x = residual + x_attn

            # Feedforward.
            residual = x
            x = layer.ff_norm(x)
            x = layer.ff(x)
            x = residual + x

        x = self.decoder_norm(x)
        gen_logits = self.output_proj(x).squeeze(1)  # (1, vocab_size)

        # Copy mechanism: blend generate and copy distributions (float32).
        x_f = x.float()
        p_copy = torch.sigmoid(self.copy_gate(x_f).squeeze(1))  # (1, 1)
        gen_probs = F.softmax(gen_logits.float(), dim=-1)
        copy_probs = torch.zeros_like(gen_probs)
        # attn_weights: (1, 1, src_len) → squeeze to (1, src_len)
        copy_probs.scatter_add_(1, src_ids, attn_weights.squeeze(1).float())
        blended = (1 - p_copy) * gen_probs + p_copy * copy_probs
        log_probs = torch.log(blended.clamp(min=1e-8))

        all_h_out = torch.cat(h_outs, dim=0)      # (n_layers, d_inner, d_state)
        all_conv_out = torch.cat(conv_outs, dim=0) # (n_layers, d_inner, d_conv-1)
        return log_probs, all_h_out, all_conv_out


class Mamba3SingleStepDecoderWrapper(nn.Module):
    """One decoder step for Mamba3 with pre-computed KV cache and copy mechanism.

    Inputs:
        tgt_token: (1, 1) int64
        all_k: (n_layers, n_heads, src_len, head_dim) float32
        all_v: (n_layers, n_heads, src_len, head_dim) float32
        all_angle: (n_layers, nheads, num_rope_angles) float32
        all_ssm: (n_layers, nheads, headdim, d_state) float32
        all_k_state: (n_layers, nheads, d_state) float32
        all_v_state: (n_layers, nheads, headdim) float32
        src_ids: (1, src_len) int64
    Outputs:
        log_probs: (1, vocab_size) float32
        all_angle_out, all_ssm_out, all_k_state_out, all_v_state_out
    """

    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_scale = model.pos_scale
        self.decoder_layers = model.decoder_layers
        self.decoder_norm = model.decoder_norm
        self.output_proj = model.output_proj
        self.copy_gate = model.copy_gate
        self.n_layers = len(model.decoder_layers)

    def forward(self, tgt_token, all_k, all_v, all_angle, all_ssm, all_k_state, all_v_state, src_ids):
        x = self.embedding(tgt_token) * self.pos_scale

        angle_outs = []
        ssm_outs = []
        k_state_outs = []
        v_state_outs = []
        attn_weights = None

        for i in range(self.n_layers):
            layer = self.decoder_layers[i]
            angle_i = all_angle[i:i+1]
            ssm_i = all_ssm[i:i+1]
            ks_i = all_k_state[i:i+1]
            vs_i = all_v_state[i:i+1]
            k_i = all_k[i:i+1]
            v_i = all_v[i:i+1]

            # Mamba3 self-attention step.
            residual = x
            x_norm = layer.self_norm(x)
            x_step, angle_new, ssm_new, ks_new, vs_new = _mamba3_step(
                layer.self_mamba, x_norm, angle_i, ssm_i, ks_i, vs_i)
            x = residual + x_step
            angle_outs.append(angle_new)
            ssm_outs.append(ssm_new)
            k_state_outs.append(ks_new)
            v_state_outs.append(vs_new)

            # Cross-attention with cached K/V.
            residual = x
            x_norm = layer.cross_norm(x)
            is_last = (i == self.n_layers - 1)
            if is_last:
                x_attn, attn_weights = _cross_attn_cached(layer.cross_attn, x_norm, k_i, v_i, return_attn=True)
            else:
                x_attn = _cross_attn_cached(layer.cross_attn, x_norm, k_i, v_i)
            x = residual + x_attn

            # Feedforward.
            residual = x
            x = layer.ff_norm(x)
            x = layer.ff(x)
            x = residual + x

        x = self.decoder_norm(x)
        gen_logits = self.output_proj(x).squeeze(1)

        # Copy mechanism.
        x_f = x.float()
        p_copy = torch.sigmoid(self.copy_gate(x_f).squeeze(1))
        gen_probs = F.softmax(gen_logits.float(), dim=-1)
        copy_probs = torch.zeros_like(gen_probs)
        copy_probs.scatter_add_(1, src_ids, attn_weights.squeeze(1).float())
        blended = (1 - p_copy) * gen_probs + p_copy * copy_probs
        log_probs = torch.log(blended.clamp(min=1e-8))

        return (log_probs,
                torch.cat(angle_outs, dim=0),
                torch.cat(ssm_outs, dim=0),
                torch.cat(k_state_outs, dim=0),
                torch.cat(v_state_outs, dim=0))


def _mamba_step(mamba, x, h, conv_buf):
    """Single-token Mamba. Pure tensor ops, no loops."""
    xz = mamba.in_proj(x)            # (1, 1, 2*d_inner)
    x_part, z = xz.chunk(2, dim=-1)  # (1, 1, d_inner) each
    x_part = x_part.squeeze(1)       # (1, d_inner)

    # Conv with buffer
    conv_input = torch.cat([conv_buf, x_part.unsqueeze(-1)], dim=-1)  # (1, d_inner, d_conv)
    conv_new = conv_input[:, :, 1:]  # (1, d_inner, d_conv-1)

    w = mamba.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
    x_conv = (conv_input * w.unsqueeze(0)).sum(dim=-1)
    if mamba.conv1d.bias is not None:
        x_conv = x_conv + mamba.conv1d.bias
    x_conv = F.silu(x_conv)

    # SSM params
    x_dbl = mamba.x_proj(x_conv.unsqueeze(1)).squeeze(1)
    dt, B_param, C_param = x_dbl.split([mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
    dt = F.softplus(mamba.dt_proj(dt.unsqueeze(1)).squeeze(1))

    # Single scan step
    A = -torch.exp(mamba.A_log.float())
    D = mamba.D.float()
    x_t = x_conv.float()

    dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))
    dB = dt.unsqueeze(-1) * B_param.unsqueeze(1)
    h_new = dA * h + dB * x_t.unsqueeze(-1)
    y_t = (h_new * C_param.unsqueeze(1)).sum(dim=-1) + D.unsqueeze(0) * x_t

    y_t = y_t * F.silu(z.squeeze(1))
    output = mamba.out_proj(y_t.unsqueeze(1))
    return output, h_new, conv_new


def _rms_norm_onnx(x, weight, eps=1e-5):
    """Pure-PyTorch RMSNorm for ONNX export."""
    x_f32 = x.float()
    rms = (x_f32 * x_f32).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * weight.float()).to(x.dtype)


def _apply_rotary_pairwise_onnx(q, cos, sin, rotary_dim):
    """Pairwise rotary embedding for ONNX export."""
    q_f32 = q.float()
    q0 = q_f32[..., 0:rotary_dim:2]
    q1 = q_f32[..., 1:rotary_dim:2]
    out = q_f32.clone()
    out[..., 0:rotary_dim:2] = q0 * cos - q1 * sin
    out[..., 1:rotary_dim:2] = q0 * sin + q1 * cos
    return out.to(q.dtype)


def _mamba3_step(mamba3, x, angle_in, ssm_in, k_in, v_in):
    """Single-token Mamba3 step. Pure tensor ops for ONNX tracing.

    Args:
        x: (1, 1, d_model)
        angle_in: (1, nheads, num_rope_angles) float32
        ssm_in: (1, nheads, headdim, d_state) float32
        k_in: (1, nheads, d_state) float32 — previous B
        v_in: (1, nheads, headdim) float32 — previous x

    Returns:
        output: (1, 1, d_model)
        angle_out, ssm_out, k_out, v_out — updated states
    """
    u = x.squeeze(1)  # (1, d_model)

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
    B = _rms_norm_onnx(B.view(-1, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.B_norm.weight)
    C = _rms_norm_onnx(C.view(-1, rank, mamba3.num_bc_heads, mamba3.d_state), mamba3.C_norm.weight)
    B = B.expand(-1, -1, mamba3.nheads, -1)
    C = C.expand(-1, -1, mamba3.nheads, -1)

    x_val = x_val.view(-1, mamba3.nheads, mamba3.headdim)
    z = z.view(-1, mamba3.nheads, mamba3.headdim)

    angles = angles_raw.unsqueeze(-2).expand(-1, mamba3.nheads, -1)

    # Rotary: accumulate angle_state
    tanh_proj = torch.tanh(angles.float())
    dt_exp = DT.float().unsqueeze(-1)
    angle_out = angle_in.float() + tanh_proj * dt_exp * math.pi
    cos_a = torch.cos(angle_out).unsqueeze(1)
    sin_a = torch.sin(angle_out).unsqueeze(1)

    B_bias = mamba3.B_bias.squeeze(1)
    C_bias = mamba3.C_bias.squeeze(1)
    B_biased = B.float() + B_bias.float().unsqueeze(0).unsqueeze(0)
    C_biased = C.float() + C_bias.float().unsqueeze(0).unsqueeze(0)

    rotary_dim = mamba3.num_rope_angles * 2
    B_rot = _apply_rotary_pairwise_onnx(B_biased, cos_a, sin_a, rotary_dim).squeeze(1)
    C_rot = _apply_rotary_pairwise_onnx(C_biased, cos_a, sin_a, rotary_dim).squeeze(1)

    # Three-term recurrence
    alpha = torch.exp(A.float() * DT.float())
    gamma = trap.float() * DT.float()
    beta = (1.0 - trap.float()) * DT.float() * alpha

    x_f32 = x_val.float()
    x_gamma = x_f32 * gamma.unsqueeze(-1)
    x_beta = v_in.float() * beta.unsqueeze(-1)

    alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1)
    ssm_out = (alpha_4d * ssm_in
               + x_gamma.unsqueeze(-1) * B_rot.float().unsqueeze(-2)
               + x_beta.unsqueeze(-1) * k_in.float().unsqueeze(-2))

    out = (ssm_out * C_rot.float().unsqueeze(-2)).sum(dim=-1)
    out = out + mamba3.D.float().unsqueeze(0).unsqueeze(-1) * x_f32
    out = out * F.silu(z.float())

    k_out = B_rot.float()
    v_out = x_val.float()

    out_flat = out.view(1, mamba3.d_inner).to(x.dtype)
    return mamba3.out_proj(out_flat).unsqueeze(1), angle_out, ssm_out, k_out, v_out


def _cross_attn(mha, query, memory):
    """Cross-attention with full K/V projection (used by encoder validation only)."""
    d_model = query.size(-1)
    batch = query.size(0)

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query, w_q, b_q)
    k = F.linear(memory, w_k, b_k)
    v = F.linear(memory, w_v, b_v)

    q = q.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)
    k = k.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)
    v = v.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(mha.head_dim)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, -1, d_model)
    return mha.out_proj(out)


def _cross_attn_cached(mha, query, k, v, return_attn=False):
    """Cross-attention with pre-computed K/V. Only Q projection per step."""
    d_model = query.size(-1)
    batch = query.size(0)

    w_q, _, _ = mha.in_proj_weight.chunk(3, dim=0)
    b_q, _, _ = mha.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query, w_q, b_q)
    q = q.view(batch, -1, mha.num_heads, mha.head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(mha.head_dim)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, -1, d_model)
    if return_attn:
        # Average across heads: (batch, n_heads, 1, src_len) → (batch, 1, src_len)
        attn_avg = attn.mean(dim=1)
        return mha.out_proj(out), attn_avg
    return mha.out_proj(out)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_encoder(model, output_path, vocab_size):
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randint(0, vocab_size, (1, 64), device=device)

    torch.onnx.export(
        EncoderWrapper(model), (dummy,), output_path,
        input_names=["src_ids"], output_names=["all_k", "all_v"],
        dynamic_axes={
            "src_ids": {1: "src_len"},
            "all_k": {2: "src_len"},
            "all_v": {2: "src_len"},
        },
        opset_version=17,
    )
    print(f"Encoder exported to {output_path}")


def _is_mamba3(model):
    """Check if model uses Mamba3 blocks."""
    return type(model.decoder_layers[0].self_mamba).__name__ == "Mamba3"


def export_decoder(model, output_path, vocab_size):
    model.eval()
    device = next(model.parameters()).device

    n_layers = len(model.decoder_layers)
    mha = model.decoder_layers[0].cross_attn
    n_heads, head_dim = mha.num_heads, mha.head_dim
    m = model.decoder_layers[0].self_mamba

    dummy_token = torch.randint(0, vocab_size, (1, 1), device=device)
    dummy_k = torch.randn(n_layers, n_heads, 32, head_dim, device=device)
    dummy_v = torch.randn(n_layers, n_heads, 32, head_dim, device=device)
    dummy_src_ids = torch.randint(0, vocab_size, (1, 32), dtype=torch.long, device=device)

    if _is_mamba3(model):
        nheads = m.nheads
        headdim_m = m.headdim
        d_state = m.d_state
        num_rope_angles = m.num_rope_angles

        dummy_angle = torch.zeros(n_layers, nheads, num_rope_angles, device=device)
        dummy_ssm = torch.zeros(n_layers, nheads, headdim_m, d_state, device=device)
        dummy_ks = torch.zeros(n_layers, nheads, d_state, device=device)
        dummy_vs = torch.zeros(n_layers, nheads, headdim_m, device=device)

        wrapper = Mamba3SingleStepDecoderWrapper(model).to(device)

        torch.onnx.export(
            wrapper,
            (dummy_token, dummy_k, dummy_v, dummy_angle, dummy_ssm, dummy_ks, dummy_vs, dummy_src_ids),
            output_path,
            input_names=["tgt_token", "all_k", "all_v",
                         "all_angle", "all_ssm", "all_k_state", "all_v_state", "src_ids"],
            output_names=["log_probs", "all_angle_out", "all_ssm_out",
                          "all_k_state_out", "all_v_state_out"],
            dynamic_axes={
                "all_k": {2: "src_len"}, "all_v": {2: "src_len"},
                "src_ids": {1: "src_len"},
            },
            opset_version=17,
        )
        print(f"Decoder (Mamba3, single-step, KV cached, copy gate) exported to {output_path}")
    else:
        d_inner, d_state, d_conv = m.d_inner, m.d_state, m.d_conv
        dummy_h = torch.zeros(n_layers, d_inner, d_state, device=device)
        dummy_conv = torch.zeros(n_layers, d_inner, d_conv - 1, device=device)

        wrapper = SingleStepDecoderWrapper(model).to(device)

        torch.onnx.export(
            wrapper,
            (dummy_token, dummy_k, dummy_v, dummy_h, dummy_conv, dummy_src_ids),
            output_path,
            input_names=["tgt_token", "all_k", "all_v", "all_h", "all_conv", "src_ids"],
            output_names=["log_probs", "all_h_out", "all_conv_out"],
            dynamic_axes={
                "all_k": {2: "src_len"}, "all_v": {2: "src_len"},
                "src_ids": {1: "src_len"},
            },
            opset_version=17,
        )
        print(f"Decoder (single-step, KV cached, copy gate) exported to {output_path}")


def validate(model, encoder_path, decoder_path, sp):
    """Compare ONNX single-step decode vs PyTorch incremental decode, step by step."""
    from infer import (patch_mamba_for_cpu, mamba_step_cpu, init_mamba_state,
                       mamba3_step_cpu, init_mamba3_state)

    is_m3 = _is_mamba3(model)

    # PyTorch reference (known-good incremental path).
    # Build with same args as the loaded model.
    m0 = model.decoder_layers[0].self_mamba
    build_kwargs = dict(
        vocab_size=sp.get_piece_size(), d_model=model.d_model,
        n_encoder_layers=len(model.encoder_layers),
        n_decoder_layers=len(model.decoder_layers),
        n_heads=model.decoder_layers[0].cross_attn.num_heads,
        pad_id=sp.pad_id(),
    )
    if is_m3:
        build_kwargs.update(d_state=m0.d_state, headdim=m0.headdim)
    else:
        build_kwargs.update(d_state=m0.d_state)
    ref = TransmutationModel(**build_kwargs).cpu()
    ref.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    ref.eval()
    patch_mamba_for_cpu(ref)

    test_input = '{"name": "Alice", "age": 30}'
    src_ids = sp.encode(test_input)
    max_steps = 60

    # --- PyTorch step-by-step (reference) ---
    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long)
        pt_memory = ref.encode(src)
        pt_layer_states = []
        for layer in ref.decoder_layers:
            mamba = layer.self_mamba
            if is_m3:
                pt_layer_states.append(init_mamba3_state(mamba, 1, "cpu"))
            else:
                pt_layer_states.append(init_mamba_state(mamba, 1, "cpu"))

    # --- ONNX step-by-step ---
    enc_sess = ort.InferenceSession(encoder_path)
    dec_sess = ort.InferenceSession(decoder_path)
    all_k, all_v = enc_sess.run(None, {"src_ids": np.array([src_ids], dtype=np.int64)})

    n_layers = len(ref.decoder_layers)
    m = ref.decoder_layers[0].self_mamba

    if is_m3:
        ort_angle = np.zeros((n_layers, m.nheads, m.num_rope_angles), dtype=np.float32)
        ort_ssm = np.zeros((n_layers, m.nheads, m.headdim, m.d_state), dtype=np.float32)
        ort_ks = np.zeros((n_layers, m.nheads, m.d_state), dtype=np.float32)
        ort_vs = np.zeros((n_layers, m.nheads, m.headdim), dtype=np.float32)
    else:
        ort_h = np.zeros((n_layers, m.d_inner, m.d_state), dtype=np.float32)
        ort_conv = np.zeros((n_layers, m.d_inner, m.d_conv - 1), dtype=np.float32)

    bos, eos = sp.bos_id(), sp.eos_id()
    src_ids_np = np.array([src_ids], dtype=np.int64)
    src_ids_pt = torch.tensor([src_ids], dtype=torch.long)
    pt_ids, ort_ids = [], []
    pt_token, ort_token = bos, bos
    first_diverge = None

    print(f"\n  Step-by-step validation: {test_input}")
    print(f"  {'step':>4s}  {'pt_id':>6s} {'ort_id':>6s}  {'out_maxdiff':>12s}  {'state_maxdiff':>14s}  match")
    print(f"  {'─'*60}")

    with torch.no_grad():
        for step in range(max_steps):
            # --- PyTorch step ---
            tok_pt = torch.tensor([[pt_token]], dtype=torch.long)
            x_pt = ref.embedding(tok_pt) * ref.pos_scale

            pt_attn = None
            for li, (layer, state) in enumerate(zip(ref.decoder_layers, pt_layer_states)):
                residual = x_pt
                x_pt = layer.self_norm(x_pt)
                if is_m3:
                    x_pt = mamba3_step_cpu(layer.self_mamba, x_pt, state)
                else:
                    x_pt = mamba_step_cpu(layer.self_mamba, x_pt, state)
                x_pt = residual + x_pt
                residual = x_pt
                x_pt = layer.cross_norm(x_pt)
                is_last = (li == n_layers - 1)
                x_pt, attn_w = layer.cross_attn(
                    x_pt, pt_memory, pt_memory,
                    need_weights=is_last, average_attn_weights=True,
                )
                if is_last:
                    pt_attn = attn_w
                x_pt = residual + x_pt
                residual = x_pt
                x_pt = layer.ff_norm(x_pt)
                x_pt = layer.ff(x_pt)
                x_pt = residual + x_pt

            x_pt = ref.decoder_norm(x_pt)
            gen_logits_pt = ref.output_proj(x_pt)[0, 0]
            p_copy = torch.sigmoid(ref.copy_gate(x_pt))[0, 0]
            gen_probs = F.softmax(gen_logits_pt, dim=-1)
            copy_probs = torch.zeros_like(gen_probs)
            copy_probs.scatter_add_(0, src_ids_pt[0], pt_attn[0, 0])
            pt_blended = (1 - p_copy) * gen_probs + p_copy * copy_probs
            pt_out = torch.log(pt_blended + 1e-10)
            pt_next = pt_out.argmax().item()

            # --- ONNX step ---
            if is_m3:
                ort_results = dec_sess.run(None, {
                    "tgt_token": np.array([[ort_token]], dtype=np.int64),
                    "all_k": all_k, "all_v": all_v,
                    "all_angle": ort_angle, "all_ssm": ort_ssm,
                    "all_k_state": ort_ks, "all_v_state": ort_vs,
                    "src_ids": src_ids_np,
                })
                ort_out = ort_results[0]
                ort_angle, ort_ssm, ort_ks, ort_vs = ort_results[1:]
            else:
                ort_out, ort_h, ort_conv = dec_sess.run(None, {
                    "tgt_token": np.array([[ort_token]], dtype=np.int64),
                    "all_k": all_k, "all_v": all_v,
                    "all_h": ort_h, "all_conv": ort_conv,
                    "src_ids": src_ids_np,
                })
            ort_next = int(ort_out[0].argmax())

            # --- Compare ---
            logit_diff = np.max(np.abs(pt_out.numpy() - ort_out[0]))

            if is_m3:
                pt_ssm_all = torch.cat([s["ssm_state"] for s in pt_layer_states], dim=0).numpy()
                state_diff = np.max(np.abs(pt_ssm_all - ort_ssm))
            else:
                pt_h_all = torch.cat([s["h"] for s in pt_layer_states], dim=0).numpy()
                state_diff = np.max(np.abs(pt_h_all - ort_h))

            match = pt_next == ort_next
            flag = " " if match else " <<<DIVERGE"
            print(f"  {step:4d}  {pt_next:6d} {ort_next:6d}  {logit_diff:13.6f}  {state_diff:14.6f}  {match}{flag}")

            if not match and first_diverge is None:
                first_diverge = step

            if pt_next == eos and ort_next == eos:
                break
            if pt_next != eos:
                pt_ids.append(pt_next)
                pt_token = pt_next
            if ort_next != eos:
                ort_ids.append(ort_next)
                ort_token = ort_next

    print(f"\n  PT  IDs (first 15): {pt_ids[:15]}")
    print(f"  ONNX IDs (first 15): {ort_ids[:15]}")
    print(f"  Match: {pt_ids == ort_ids}")
    if first_diverge is not None:
        print(f"  First divergence at step {first_diverge}")
    print(f"  PT output:   {sp.decode(pt_ids).strip()[:150]}")
    print(f"  ONNX output: {sp.decode(ort_ids).strip()[:150]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.get_piece_size()

    model = TransmutationModel(
        vocab_size=vocab_size, d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_state=args.d_state, headdim=args.headdim,
        n_heads=args.n_heads, pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        if all("copy_gate" in k for k in missing):
            nn.init.zeros_(model.copy_gate.weight)
            nn.init.constant_(model.copy_gate.bias, -5.0)
            print(f"  Initialized copy_gate from scratch (old checkpoint, p_copy≈0.007)")
        else:
            raise RuntimeError(f"Unexpected missing keys: {missing}")
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, device={device})")

    # Patch encoder Mamba for ONNX export.
    patch_mamba_for_onnx(model)

    os.makedirs(args.output_dir, exist_ok=True)
    enc_path = os.path.join(args.output_dir, "encoder.onnx")
    dec_path = os.path.join(args.output_dir, "decoder.onnx")

    with torch.no_grad():
        export_encoder(model, enc_path, vocab_size)
        export_decoder(model, dec_path, vocab_size)

    onnx.checker.check_model(onnx.load(enc_path))
    onnx.checker.check_model(onnx.load(dec_path))
    print("ONNX validation passed")

    validate(model, enc_path, dec_path, sp)

    enc_mb = os.path.getsize(enc_path) / 1024 / 1024
    dec_mb = os.path.getsize(dec_path) / 1024 / 1024
    print(f"\nEncoder: {enc_mb:.1f} MB, Decoder: {dec_mb:.1f} MB, Total: {enc_mb+dec_mb:.1f} MB")

    # Dynamic int8 quantization for faster CPU inference.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    enc_q_path = os.path.join(args.output_dir, "encoder_int8.onnx")
    dec_q_path = os.path.join(args.output_dir, "decoder_int8.onnx")
    quantize_dynamic(enc_path, enc_q_path, weight_type=QuantType.QInt8)
    quantize_dynamic(dec_path, dec_q_path, weight_type=QuantType.QInt8)
    enc_q_mb = os.path.getsize(enc_q_path) / 1024 / 1024
    dec_q_mb = os.path.getsize(dec_q_path) / 1024 / 1024
    print(f"\nQuantized (int8): Encoder: {enc_q_mb:.1f} MB, Decoder: {dec_q_mb:.1f} MB, Total: {enc_q_mb+dec_q_mb:.1f} MB")


if __name__ == "__main__":
    main()
