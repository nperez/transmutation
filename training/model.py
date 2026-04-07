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

"""Mamba3-based encoder-decoder for broken JSON to XML translation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd


# ── Triton kernel: Mamba3 SSM step (replaces CuTe DSL mamba3_step_fn) ──────
#
# Works on sm_75+ (RTX 2060). The CuTe kernel only works on H100.
# One program per (batch, head). Processes headdim rows of d_state-wide vectors.
# Non-MIMO only (R=1): xproj/zproj/outproj are all ones, skipped.
#
# Math per (batch, head):
#   state[i,:] = A * state[i,:] + dt * ((1-trap)*x[i]*B[:] + trap*v_old[i]*k_old[:])
#   y[i] = dot(state[i,:], C[:]) + D * x[i]
#   y[i] = y[i] * silu(z[i])

@triton.jit
def _mamba3_step_kernel(
    # SSM state (B, H, HEADDIM, D_STATE) — in-place update
    ssm_ptr, ssm_s0, ssm_s1, ssm_s2, ssm_s3,
    # v_state (B, H, HEADDIM) — updated to x after kernel
    v_ptr, v_s0, v_s1, v_s2,
    # k_state (B, H, D_STATE) — updated to B after kernel (R=1 squeezed)
    k_ptr, k_s0, k_s1, k_s2,
    # x input (B, H, HEADDIM)
    x_ptr, x_s0, x_s1, x_s2,
    # B input (B, H, D_STATE) — R=1 squeezed
    B_ptr, B_s0, B_s1, B_s2,
    # C input (B, H, D_STATE) — R=1 squeezed
    C_ptr, C_s0, C_s1, C_s2,
    # z gate (B, H, HEADDIM)
    z_ptr, z_s0, z_s1, z_s2,
    # A decay (B, H)
    A_ptr, A_s0, A_s1,
    # DT timestep (B, H)
    DT_ptr, DT_s0, DT_s1,
    # trap coeff (B, H)
    trap_ptr, trap_s0, trap_s1,
    # D skip (H,)
    D_ptr,
    # y output (B, H, HEADDIM)
    y_ptr, y_s0, y_s1, y_s2,
    HEADDIM: tl.constexpr,
    D_STATE: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # Scalars for this (batch, head)
    decay = tl.load(A_ptr + bid * A_s0 + hid * A_s1).to(tl.float32)
    dt = tl.load(DT_ptr + bid * DT_s0 + hid * DT_s1).to(tl.float32)
    trap_val = tl.load(trap_ptr + bid * trap_s0 + hid * trap_s1).to(tl.float32)
    D_val = tl.load(D_ptr + hid).to(tl.float32)

    new_coeff = dt * (1.0 - trap_val)
    old_coeff = dt * trap_val

    # Load d_state-length vectors (reused across headdim loop)
    ds = tl.arange(0, D_STATE)
    B_new = tl.load(B_ptr + bid * B_s0 + hid * B_s1 + ds * B_s2).to(tl.float32)
    C_vec = tl.load(C_ptr + bid * C_s0 + hid * C_s1 + ds * C_s2).to(tl.float32)
    k_old = tl.load(k_ptr + bid * k_s0 + hid * k_s1 + ds * k_s2).to(tl.float32)

    # Process each row of headdim
    for i in range(HEADDIM):
        x_i = tl.load(x_ptr + bid * x_s0 + hid * x_s1 + i * x_s2).to(tl.float32)
        v_i = tl.load(v_ptr + bid * v_s0 + hid * v_s1 + i * v_s2).to(tl.float32)
        z_i = tl.load(z_ptr + bid * z_s0 + hid * z_s1 + i * z_s2).to(tl.float32)

        # Load state row: (D_STATE,)
        state = tl.load(ssm_ptr + bid * ssm_s0 + hid * ssm_s1 + i * ssm_s2 + ds * ssm_s3).to(tl.float32)

        # State update: decay * state + dt*((1-trap)*x_new*B_new + trap*v_old*k_old)
        state = decay * state + new_coeff * x_i * B_new + old_coeff * v_i * k_old

        # Store updated state (cast back to input dtype)
        tl.store(ssm_ptr + bid * ssm_s0 + hid * ssm_s1 + i * ssm_s2 + ds * ssm_s3, state)

        # Output: dot(state, C) + D * x
        y_i = tl.sum(state * C_vec) + D_val * x_i

        # Gating: y * silu(z) = y * z * sigmoid(z)
        y_i = y_i * z_i * tl.sigmoid(z_i)

        tl.store(y_ptr + bid * y_s0 + hid * y_s1 + i * y_s2, y_i)

    # Update k_state ← B_new, v_state ← x
    tl.store(k_ptr + bid * k_s0 + hid * k_s1 + ds * k_s2, B_new)
    for i in range(HEADDIM):
        x_i = tl.load(x_ptr + bid * x_s0 + hid * x_s1 + i * x_s2)
        tl.store(v_ptr + bid * v_s0 + hid * v_s1 + i * v_s2, x_i)


def mamba3_step_triton(ssm_state, k_state, v_state, A, B, C, D, x, DT, trap, z):
    """Pure-Triton replacement for mamba3_step_fn (non-MIMO, R=1 only).

    Updates ssm_state, k_state, v_state in-place. Returns y (batch, nheads, headdim).
    All samples in the batch run in parallel on the GPU.
    """
    batch, nheads, headdim, d_state = ssm_state.shape
    y = torch.empty(batch, nheads, headdim, device=x.device, dtype=x.dtype)

    # Squeeze R=1 dimension for B, C, k_state — views share memory
    B_sq = B.squeeze(1).contiguous()
    C_sq = C.squeeze(1).contiguous()
    k_sq = k_state.squeeze(1).contiguous()

    grid = (batch, nheads)
    _mamba3_step_kernel[grid](
        ssm_state, *ssm_state.stride(),
        v_state, *v_state.stride(),
        k_sq, *k_sq.stride(),
        x, *x.stride(),
        B_sq, *B_sq.stride(),
        C_sq, *C_sq.stride(),
        z, *z.stride(),
        A, *A.stride(),
        DT, *DT.stride(),
        trap, *trap.stride(),
        D,
        y, *y.stride(),
        HEADDIM=headdim,
        D_STATE=d_state,
    )

    # Copy updated k back to original (with R dimension)
    k_state[:, 0].copy_(k_sq)

    return y


class TransmutationModel(nn.Module):
    """
    Encoder-decoder model:
    - Encoder: Mamba3 blocks processing the corrupted input
    - Decoder: Mamba3 blocks with cross-attention to encoder states
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        n_heads: int = 6,
        chunk_size: int = 32,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Shared embedding (input and output share the same vocabulary).
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_scale = math.sqrt(d_model)

        # Encoder: stack of Mamba3 blocks.
        self.encoder_layers = nn.ModuleList([
            Mamba3EncoderLayer(d_model, d_state, expand, headdim, chunk_size, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder: Mamba3 blocks interleaved with cross-attention.
        self.decoder_layers = nn.ModuleList([
            Mamba3DecoderLayer(d_model, d_state, expand, headdim, chunk_size, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection.
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding weights with output projection.
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name or "output_proj" in name:
                continue
            # Skip Mamba3 internal parameters — they have their own init.
            if "mamba." in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Encode source sequence. src_ids: (batch, src_len)"""
        x = self.embedding(src_ids) * self.pos_scale
        for layer in self.encoder_layers:
            x = layer(x)
        return self.encoder_norm(x)

    def decode(
        self, tgt_ids: torch.Tensor, memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target sequence with cross-attention to encoder output.

        Returns raw logits (batch, tgt_len, vocab_size).
        """
        x = self.embedding(tgt_ids) * self.pos_scale

        for layer in self.decoder_layers:
            x = layer(x, memory, memory_key_padding_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full forward pass.
        src_ids: (batch, src_len)
        tgt_ids: (batch, tgt_len)
        Returns: logits (batch, tgt_len, vocab_size).
        """
        memory = self.encode(src_ids)
        return self.decode(tgt_ids, memory, src_key_padding_mask)

    def _ar_ensure_cache(self, batch, max_src_len, device, dtype, bos_id):
        """Build or reuse cached CUDA graph + static buffers for AR decode.

        Cached by (batch, max_src_len). First call compiles Triton + captures graph.
        Subsequent calls with same shape: ~0 overhead.
        """
        from infer import init_mamba3_state

        key = (batch, max_src_len)
        if not hasattr(self, '_ar_caches'):
            self._ar_caches = {}
        if key in self._ar_caches:
            return self._ar_caches[key]

        # Static buffers — same memory addresses reused across calls.
        cache = {"key": key}
        cache["static_memory"] = torch.zeros(batch, max_src_len, self.d_model, device=device, dtype=dtype)
        cache["static_src_mask"] = torch.ones(batch, max_src_len, dtype=torch.bool, device=device)
        cache["static_tok"] = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        cache["static_next"] = torch.empty(batch, dtype=torch.long, device=device)
        cache["layer_states"] = [init_mamba3_state(layer.self_mamba, batch, device)
                                 for layer in self.decoder_layers]

        mem = cache["static_memory"]
        mask = cache["static_src_mask"]
        tok = cache["static_tok"]
        nxt = cache["static_next"]
        states = cache["layer_states"]

        def _step():
            x = self.embedding(tok) * self.pos_scale
            for layer, state in zip(self.decoder_layers, states):
                x, _ = layer.step(x, mem, state, memory_key_padding_mask=mask)
            x = self.decoder_norm(x)
            nxt.copy_(self.output_proj(x[:, 0, :]).argmax(dim=-1))

        cache["step_fn"] = _step

        # Warmup (populates Triton compilation cache)
        _step()
        _step()
        for s in states:
            for v in s.values():
                if isinstance(v, torch.Tensor):
                    v.zero_()

        # Capture CUDA graph
        cache["graph"] = None
        try:
            s_stream = torch.cuda.Stream()
            s_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_stream):
                _step()
            torch.cuda.current_stream().wait_stream(s_stream)
            for s in states:
                for v in s.values():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s_stream):
                _step()
            cache["graph"] = g
            import sys
            print(f"  AR CUDA graph captured for shape ({batch}, {max_src_len})", file=sys.stderr, flush=True)
        except Exception as e:
            import sys
            print(f"  AR graph capture failed: {e}", file=sys.stderr, flush=True)

        self._ar_caches[key] = cache
        return cache

    @torch.no_grad()
    def ar_decode(
        self, src_ids: torch.Tensor, bos_id: int, eos_id: int,
        max_len: int = 1536,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Autoregressive decode using cached CUDA graph.

        First call for a given (batch, src_len) shape captures the graph (~30s).
        Subsequent calls with same shape: ~2ms/tok via graph replay.
        """
        device = src_ids.device
        dtype = next(self.parameters()).dtype
        batch = src_ids.shape[0]
        actual_src_len = src_ids.shape[1]

        # Round up to bucket for cache stability: 128, 256, 384, 512, 768, 1152.
        # Few distinct sizes → few graph captures, minimal wasted padding.
        _buckets = [128, 256, 384, 512, 768, 1152]
        padded_src_len = next((b for b in _buckets if b >= actual_src_len), _buckets[-1])

        cache = self._ar_ensure_cache(batch, padded_src_len, device, dtype, bos_id)

        # Encode at actual length, copy into padded static buffer.
        memory = self.encode(src_ids)
        cache["static_memory"].zero_()
        cache["static_memory"][:, :actual_src_len].copy_(memory)
        cache["static_src_mask"].fill_(True)
        if src_key_padding_mask is not None:
            cache["static_src_mask"][:, :actual_src_len].copy_(src_key_padding_mask[:, :actual_src_len])
        else:
            cache["static_src_mask"][:, :actual_src_len] = False

        # Reset states
        for s in cache["layer_states"]:
            for v in s.values():
                if isinstance(v, torch.Tensor):
                    v.zero_()

        # Decode loop
        tok = cache["static_tok"]
        nxt = cache["static_next"]
        graph = cache["graph"]
        step_fn = cache["step_fn"]

        import time as _time
        _ar_t0 = _time.monotonic()

        tok.fill_(bos_id)
        all_ids = [tok[:, 0].clone()]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        eos_t = torch.tensor(eos_id, dtype=torch.long, device=device)

        for step in range(max_len - 1):
            if graph is not None:
                graph.replay()
            else:
                step_fn()

            next_ids = nxt.clone()
            next_ids.masked_fill_(finished, self.pad_id)
            finished = finished | (next_ids == eos_t)
            all_ids.append(next_ids)
            tok[:, 0] = next_ids

            if step % 50 == 49 and finished.all():
                break

        _ar_elapsed = _time.monotonic() - _ar_t0
        _n_steps = len(all_ids) - 1
        if _n_steps > 0:
            import sys
            _mode = "graph" if graph is not None else "plain"
            print(f"  AR decode: {_n_steps} steps in {_ar_elapsed:.2f}s ({_ar_elapsed/_n_steps*1000:.1f}ms/step, {_mode})",
                  file=sys.stderr, flush=True)

        all_ids = torch.stack(all_ids, dim=1)
        result = []
        for i in range(batch):
            ids = all_ids[i].tolist()
            if eos_id in ids:
                ids = ids[:ids.index(eos_id) + 1]
            result.append(ids)
        return result


class QKNormCrossAttention(nn.Module):
    """Cross-attention with RMSNorm on Q and K to prevent fp16 overflow.

    Normalizing Q and K bounds the dot product scores to O(sqrt(d_head))
    regardless of weight magnitude, eliminating NaN from attention overflow.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, key_padding_mask=None, need_weights=False):
        batch, tgt_len, d_model = query.shape
        src_len = memory.shape[1]

        q = self.q_proj(query).view(batch, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch, src_len, self.n_heads, self.head_dim).transpose(1, 2)

        # QK normalization: bounds scores regardless of weight magnitude.
        q = self.q_norm(q.float()).to(q.dtype)
        k = self.k_norm(k.float()).to(k.dtype)

        # Scaled dot-product attention.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, tgt_len, d_model)
        out = self.out_proj(out)

        if need_weights:
            # Average across heads for copy mechanism.
            return out, attn.mean(dim=1)
        return out, None


class NormedInProj(nn.Module):
    """Wraps Mamba3 in_proj with RMSNorm on x_val and z to prevent fp16 overflow.

    Mamba3 in_proj output layout: [z, x_val, B, C, dt, A, trap, angles].
    B and C have internal BCNorm, but x_val and z are unbounded — as weights
    grow during training, they push the SSM scan past fp16 range on long sequences.
    Same principle as QKNormCrossAttention: normalize before the dot products.
    """

    def __init__(self, linear, d_inner):
        super().__init__()
        self.linear = linear
        self.d_inner = d_inner
        self.z_norm = nn.RMSNorm(d_inner)
        self.x_norm = nn.RMSNorm(d_inner)

    def forward(self, x):
        out = self.linear(x).clone()
        out[..., :self.d_inner] = self.z_norm(out[..., :self.d_inner].float()).to(out.dtype)
        out[..., self.d_inner:2 * self.d_inner] = self.x_norm(
            out[..., self.d_inner:2 * self.d_inner].float()).to(out.dtype)
        return out


class Mamba3EncoderLayer(nn.Module):
    def __init__(self, d_model, d_state, expand, headdim, chunk_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.mamba.in_proj = NormedInProj(self.mamba.in_proj, self.mamba.d_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)


class Mamba3DecoderLayer(nn.Module):
    def __init__(self, d_model, d_state, expand, headdim, chunk_size, n_heads, dropout):
        super().__init__()
        # Self-"attention" via Mamba3.
        self.self_norm = nn.LayerNorm(d_model)
        self.self_mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.self_mamba.in_proj = NormedInProj(self.self_mamba.in_proj, self.self_mamba.d_inner)

        # Cross-attention to encoder output with QK normalization.
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = QKNormCrossAttention(d_model, n_heads, dropout=dropout)

        # Feedforward.
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None, return_attn_weights=False):
        # Self-attention (Mamba3).
        residual = x
        x = self.self_norm(x)
        x = self.self_mamba(x)
        x = residual + self.dropout(x)

        # Cross-attention with QK normalization (prevents fp16 overflow).
        residual = x
        x = self.cross_norm(x)
        x, attn_weights = self.cross_attn(
            x, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attn_weights,
        )
        x = residual + self.dropout(x)

        # Feedforward.
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x

        if return_attn_weights:
            return x, attn_weights  # attn_weights: (batch, tgt_len, src_len)
        return x

    def init_step_state(self, batch_size, device, dtype):
        """Allocate SSM cache for step-by-step decode."""
        from infer import init_mamba3_state
        return init_mamba3_state(self.self_mamba, batch_size, device)

    def step(self, x_t, memory, state, memory_key_padding_mask=None):
        """Single-token decode step using Triton SSM kernel from infer.py.

        x_t: (batch, 1, d_model) — single token embedding
        memory: (batch, src_len, d_model) — encoder output
        state: dict from init_step_state
        Returns: (output, updated_state)
        """
        from infer import _mamba3_step

        # ── Self-attention via Mamba3 step ──
        residual = x_t
        x = self.self_norm(x_t)
        x = _mamba3_step(self.self_mamba, x, state)
        x = residual + x

        # ── Cross-attention ──
        residual = x
        x = self.cross_norm(x)
        x, _ = self.cross_attn(x, memory, key_padding_mask=memory_key_padding_mask)
        x = residual + x

        # ── FFN ──
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x

        return x, state


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(vocab_size: int, **kwargs) -> TransmutationModel:
    """Build model and print parameter count."""
    model = TransmutationModel(vocab_size, **kwargs)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    return model
