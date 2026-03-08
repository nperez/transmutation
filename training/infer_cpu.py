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

"""CPU inference using pure-Python Mamba forward pass (no CUDA kernels needed)."""

import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm


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


def patch_mamba_for_cpu(model):
    """Monkey-patch all Mamba modules to use CPU forward pass."""
    for module in model.modules():
        if type(module).__name__ == "Mamba":
            module._original_forward = module.forward
            module.forward = lambda x, m=module: mamba_forward_cpu(m, x)


def greedy_decode(model, src_ids, sp, max_len=512, device="cpu"):
    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory = model.encode(src)

        # Initialize cached state for each decoder layer's Mamba block.
        layer_states = []
        for layer in model.decoder_layers:
            layer_states.append(init_mamba_state(layer.self_mamba, 1, device))

        tgt_ids = [sp.bos_id()]
        for _ in range(max_len):
            # Only embed and process the last token.
            tok = torch.tensor([[tgt_ids[-1]]], dtype=torch.long, device=device)
            x = model.embedding(tok) * model.pos_scale  # (1, 1, d_model)

            for layer, state in zip(model.decoder_layers, layer_states):
                # Mamba step (incremental, cached).
                residual = x
                x = layer.self_norm(x)
                x = mamba_step_cpu(layer.self_mamba, x, state)
                x = residual + x

                # Cross-attention (single query token against full memory).
                residual = x
                x = layer.cross_norm(x)
                x, _ = layer.cross_attn(x, memory, memory)
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


def load_model(checkpoint, device):
    sp = spm.SentencePieceProcessor()
    sp.load("models/tokenizer.model")

    sys.path.insert(0, str(Path(__file__).parent))
    from model import TransmutationModel

    model = TransmutationModel(
        vocab_size=sp.get_piece_size(), d_model=384,
        n_encoder_layers=6, n_decoder_layers=6,
        d_state=16, n_heads=6, pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    patch_mamba_for_cpu(model)
    print(f"Loaded: epoch={ckpt['epoch']}, global_step={ckpt['global_step']}")
    print()
    return model, sp


def read_records(sp, max_src_len):
    """Read JSONL records from stdin, filtering by source token length."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if len(sp.encode(r["input"])) <= max_src_len:
            yield r


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "models/epoch_1.pt"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_src_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    device = torch.device("cpu")
    model, sp = load_model(checkpoint, device)

    xml_ok_count = 0
    exact_count = 0
    total = 0

    for i, rec in enumerate(read_records(sp, max_src_len)):
        if i >= n_samples:
            break
        total += 1

        src_ids = sp.encode(rec["input"])[:max_src_len]
        t0 = time.monotonic()
        pred_ids = greedy_decode(model, src_ids, sp, device=device)
        elapsed = time.monotonic() - t0
        pred = sp.decode(pred_ids)
        target = rec["target"]

        norm_pred = re.sub(r"\s+", " ", pred.strip())
        norm_tgt = re.sub(r"\s+", " ", target.strip())
        exact = norm_pred == norm_tgt

        try:
            ET.fromstring(pred.strip())
            xml_ok = True
        except ET.ParseError:
            xml_ok = False

        if exact:
            exact_count += 1
        if xml_ok:
            xml_ok_count += 1

        tag = "EXACT" if exact else ("XML_OK" if xml_ok else "FAIL")
        print(f"=== Sample {i+1} [{tag}] {elapsed:.2f}s, {len(pred_ids)} tokens ===")
        print(f"INPUT:\n{rec['input']}\n")
        if exact:
            print(f"OUTPUT (matches target):\n{pred.strip()}\n")
        else:
            print(f"TARGET:\n{target.strip()}\n")
            print(f"OUTPUT:\n{pred.strip()}\n")
        print()

    print(f"===== {total} samples: exact={exact_count} xml_ok={xml_ok_count} =====")


if __name__ == "__main__":
    main()
