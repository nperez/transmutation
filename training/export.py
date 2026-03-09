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

Encoder: full-sequence with scripted selective scan loop (called once per input).
Decoder: single-step with explicit Mamba state (h, conv_buf) as inputs/outputs.
         No loops in the decoder ONNX — the autoregressive loop runs in the caller.
         This is the standard pattern for autoregressive ONNX models (GPT-2, etc).
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


def patch_mamba_for_onnx(model):
    for module in model.modules():
        if type(module).__name__ == "Mamba":
            module.forward = lambda x, m=module: mamba_forward_onnx(m, x)


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
    """One decoder step with pre-computed KV cache.

    Inputs:
        tgt_token: (1, 1) int64
        all_k: (n_layers, n_heads, src_len, head_dim) float32  — cached from encoder
        all_v: (n_layers, n_heads, src_len, head_dim) float32  — cached from encoder
        all_h: (n_layers, d_inner, d_state) float32
        all_conv: (n_layers, d_inner, d_conv-1) float32
    Outputs:
        logits: (1, vocab_size) float32
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
        self.n_layers = len(model.decoder_layers)

    def forward(self, tgt_token, all_k, all_v, all_h, all_conv):
        x = self.embedding(tgt_token) * self.pos_scale  # (1, 1, d_model)

        h_outs = []
        conv_outs = []

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
            x_attn = _cross_attn_cached(layer.cross_attn, x_norm, k_i, v_i)
            x = residual + x_attn

            # Feedforward.
            residual = x
            x = layer.ff_norm(x)
            x = layer.ff(x)
            x = residual + x

        x = self.decoder_norm(x)
        logits = self.output_proj(x).squeeze(1)  # (1, vocab_size)

        all_h_out = torch.cat(h_outs, dim=0)      # (n_layers, d_inner, d_state)
        all_conv_out = torch.cat(conv_outs, dim=0) # (n_layers, d_inner, d_conv-1)
        return logits, all_h_out, all_conv_out


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


def _cross_attn_cached(mha, query, k, v):
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


def export_decoder(model, output_path, vocab_size):
    model.eval()
    device = next(model.parameters()).device

    n_layers = len(model.decoder_layers)
    mha = model.decoder_layers[0].cross_attn
    n_heads, head_dim = mha.num_heads, mha.head_dim
    m = model.decoder_layers[0].self_mamba
    d_inner, d_state, d_conv = m.d_inner, m.d_state, m.d_conv

    dummy_token = torch.randint(0, vocab_size, (1, 1), device=device)
    dummy_k = torch.randn(n_layers, n_heads, 32, head_dim, device=device)
    dummy_v = torch.randn(n_layers, n_heads, 32, head_dim, device=device)
    dummy_h = torch.zeros(n_layers, d_inner, d_state, device=device)
    dummy_conv = torch.zeros(n_layers, d_inner, d_conv - 1, device=device)

    wrapper = SingleStepDecoderWrapper(model).to(device)

    torch.onnx.export(
        wrapper,
        (dummy_token, dummy_k, dummy_v, dummy_h, dummy_conv),
        output_path,
        input_names=["tgt_token", "all_k", "all_v", "all_h", "all_conv"],
        output_names=["logits", "all_h_out", "all_conv_out"],
        dynamic_axes={"all_k": {2: "src_len"}, "all_v": {2: "src_len"}},
        opset_version=17,
    )
    print(f"Decoder (single-step, KV cached) exported to {output_path}")


def validate(model, encoder_path, decoder_path, sp):
    """Compare ONNX single-step decode vs PyTorch incremental decode."""
    from infer_cpu import patch_mamba_for_cpu, greedy_decode

    # PyTorch reference (known-good incremental path).
    ref = TransmutationModel(
        vocab_size=sp.get_piece_size(), d_model=model.d_model,
        n_encoder_layers=6, n_decoder_layers=6, d_state=16, n_heads=6,
        pad_id=sp.pad_id(),
    ).cpu()
    ref.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    ref.eval()
    patch_mamba_for_cpu(ref)

    test_input = '{"name": "Alice", "age": 30}'
    src_ids = sp.encode(test_input)
    pt_ids = greedy_decode(ref, src_ids, sp, max_len=60, device="cpu")

    # ONNX single-step decode with KV cache.
    enc_sess = ort.InferenceSession(encoder_path)
    dec_sess = ort.InferenceSession(decoder_path)

    all_k, all_v = enc_sess.run(None, {"src_ids": np.array([src_ids], dtype=np.int64)})

    n_layers = len(model.decoder_layers)
    m = model.decoder_layers[0].self_mamba
    all_h = np.zeros((n_layers, m.d_inner, m.d_state), dtype=np.float32)
    all_conv = np.zeros((n_layers, m.d_inner, m.d_conv - 1), dtype=np.float32)

    bos, eos = sp.bos_id(), sp.eos_id()
    ort_ids = []
    token = np.array([[bos]], dtype=np.int64)

    for _ in range(60):
        logits, all_h, all_conv = dec_sess.run(None, {
            "tgt_token": token, "all_k": all_k, "all_v": all_v,
            "all_h": all_h, "all_conv": all_conv,
        })
        next_id = int(logits[0].argmax())
        if next_id == eos:
            break
        ort_ids.append(next_id)
        token = np.array([[next_id]], dtype=np.int64)

    match = pt_ids == ort_ids
    print(f"\n  Validation input: {test_input}")
    print(f"  PT  IDs (first 15): {pt_ids[:15]}")
    print(f"  ONNX IDs (first 15): {ort_ids[:15]}")
    print(f"  Match: {match}")
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
    parser.add_argument("--d-state", type=int, default=16)
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
        d_state=args.d_state, n_heads=args.n_heads,
        pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
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
