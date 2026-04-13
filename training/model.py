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

"""DiT-based diffusion model for JSON to XML structured transduction.

Token-manifold noise with fixed t-bucket denoising schedule.
Single bidirectional transformer with prefix conditioning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Factored Embedding ─────────────────────────────────────────────────────────

class FactoredEmbedding(nn.Module):
    """Low-rank factored embedding: (vocab, rank) @ (rank, d_model).

    Reduces parameters from vocab*d_model to vocab*rank + rank*d_model.
    project_to_vocab() produces logits over the vocabulary for the output layer.
    """

    def __init__(self, vocab_size, d_model, rank, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank = rank
        self.down = nn.Embedding(vocab_size, rank, padding_idx=pad_id)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, token_ids):
        """(batch, seq) -> (batch, seq, d_model)"""
        return self.up(self.down(token_ids))

    def project_to_vocab(self, embeddings):
        """(batch, seq, d_model) -> (batch, seq, vocab_size) logits for discretization."""
        # d_model -> rank: embeddings @ up.weight.T.T = embeddings @ up.weight
        # But up.weight is (d_model, rank), and F.linear does x @ W.T,
        # so we need the transpose: (rank, d_model) as the weight.
        low_rank = embeddings @ self.up.weight  # (batch, seq, d_model) @ (d_model, rank) -> (batch, seq, rank)
        return F.linear(low_rank, self.down.weight)  # (batch, seq, rank) @ (rank, vocab).T -> (batch, seq, vocab)


# ── RoPE ────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE). Precomputes freqs up to max_seq_len."""

    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len, offset=0):
        """Returns (cos, sin) each of shape (seq_len, dim//2)."""
        end = offset + seq_len
        if end > self._cached_len:
            self._build_cache(end * 2)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def apply_rope(x, cos, sin):
    """Apply RoPE to x of shape (batch, heads, seq, head_dim).

    cos, sin: (seq, head_dim//2).
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d2)
    sin = sin[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Timestep Embedding ──────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> conditioning vector."""

    def __init__(self, d_model, max_period=10000):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        half = d_model // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def _sinusoidal(self, x):
        """x: (batch,) -> (batch, d_model) sinusoidal features."""
        args = x.unsqueeze(-1).float() * self.freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t, r=None):
        """t: (batch,) noise level, r: (batch,) step size. Both in [0, 1]."""
        emb = self._sinusoidal(t)
        if r is not None:
            emb = emb + self._sinusoidal(r)
        return self.mlp(emb)


# ── DiT Block ───────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero timestep conditioning.

    Gate params initialized to zero so each layer starts as identity.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: 6 modulation params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the final linear so gates start at 0 (layer = identity at init)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, c, rope_cos, rope_sin):
        """
        x: (batch, seq, d_model) — padding positions zeroed
        c: (batch, d_model) — timestep conditioning
        rope_cos, rope_sin: (seq, head_dim//2)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        B, S, D = h.shape
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE to Q, K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Flash attention via SDPA (no mask — padding zeroed at model level)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        h = self.attn_dropout(h)
        x = x + gate1.unsqueeze(1) * h

        # FFN with adaLN
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.ff(h)
        x = x + gate2.unsqueeze(1) * h

        return x


# ── Length Predictor ─────────────────────────────────────────────────────────────

LENGTH_BUCKETS = [64, 128, 256, 384, 512, 768, 1024, 1536]


class LengthPredictor(nn.Module):
    """Predicts output length bucket from JSON input embeddings."""

    def __init__(self, d_model, n_buckets=len(LENGTH_BUCKETS)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_buckets),
        )

    def forward(self, x, mask):
        """
        x: (batch, src_len, d_model) — embedded JSON tokens
        mask: (batch, src_len) — True for padding
        Returns: (batch, n_buckets) logits
        """
        # Mean pool over non-pad positions
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = (x * (~mask).unsqueeze(-1).float()).sum(dim=1) / lengths
        return self.head(pooled)


# ── Diffusion Transmutation Model ────────────────────────────────────────────────

class DiffusionTransmutationModel(nn.Module):
    """Discrete diffusion for JSON→XML transduction.

    Single bidirectional DiT transformer with prefix conditioning.
    Input: [JSON token IDs | corrupted XML token IDs]. Output: logits over vocab.
    Timestep conditions all layers via adaLN-Zero.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
        d_ff: int = 1536,
        emb_rank: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        self.embedding = FactoredEmbedding(vocab_size, d_model, emb_rank, pad_id)
        self.segment_emb = nn.Embedding(2, d_model)
        self.timestep_emb = TimestepEmbedding(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # Final output: adaLN + projection to embedding space
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_proj = nn.Linear(d_model, d_model)

        self.length_predictor = LengthPredictor(d_model)

    def forward(self, src_ids, tgt_ids, t, src_mask=None, tgt_mask=None, r=None):
        """
        src_ids: (batch, src_len) — JSON token IDs
        tgt_ids: (batch, tgt_len) — XML token IDs (some corrupted with random tokens)
        t: (batch,) — corruption fraction in [0, 1]
        src_mask: (batch, src_len) — True for padding in JSON
        tgt_mask: (batch, tgt_len) — True for padding in XML
        r: (batch,) — step size, optional
        Returns: (batch, tgt_len, vocab_size) — logits over vocabulary
        """
        B, src_len = src_ids.shape
        tgt_len = tgt_ids.shape[1]
        total_len = src_len + tgt_len

        # Embed both as token IDs — same embedding for JSON and XML.
        json_emb = self.embedding(src_ids) + self.segment_emb(
            torch.zeros(B, src_len, dtype=torch.long, device=src_ids.device))
        xml_emb = self.embedding(tgt_ids) + self.segment_emb(
            torch.ones(B, tgt_len, dtype=torch.long, device=src_ids.device))

        # Concatenate prefix: [JSON | XML]
        x = torch.cat([json_emb, xml_emb], dim=1)

        # Zero out padding positions for flash attention.
        if src_mask is not None or tgt_mask is not None:
            if src_mask is None:
                src_mask = torch.zeros(B, src_len, dtype=torch.bool, device=src_ids.device)
            if tgt_mask is None:
                tgt_mask = torch.zeros(B, tgt_len, dtype=torch.bool, device=src_ids.device)
            pad_mask = torch.cat([src_mask, tgt_mask], dim=1).unsqueeze(-1)
            x = x.masked_fill(pad_mask, 0.0)

        rope_cos, rope_sin = self.rope(total_len)
        c = self.timestep_emb(t, r)

        for layer in self.layers:
            x = layer(x, c, rope_cos, rope_sin)

        # Extract XML portion → logits over vocab.
        x_xml = x[:, src_len:]
        scale, shift = self.final_adaLN(c).chunk(2, dim=-1)
        x_xml = self.final_norm(x_xml) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x_xml = self.final_proj(x_xml)  # (B, tgt_len, d_model)
        return self.embedding.project_to_vocab(x_xml)  # (B, tgt_len, vocab)

    def predict_tokens(self, src_ids, tgt_ids, t, src_mask=None, tgt_mask=None, r=None, chunk_size=256):
        """Memory-efficient inference: returns argmax token IDs without materializing full logits.

        Same computation as forward(), but chunks project_to_vocab along the sequence
        dimension so peak VRAM is (B, chunk_size, vocab) instead of (B, tgt_len, vocab).
        """
        B, src_len = src_ids.shape
        tgt_len = tgt_ids.shape[1]
        total_len = src_len + tgt_len

        json_emb = self.embedding(src_ids) + self.segment_emb(
            torch.zeros(B, src_len, dtype=torch.long, device=src_ids.device))
        xml_emb = self.embedding(tgt_ids) + self.segment_emb(
            torch.ones(B, tgt_len, dtype=torch.long, device=src_ids.device))

        x = torch.cat([json_emb, xml_emb], dim=1)

        if src_mask is not None or tgt_mask is not None:
            if src_mask is None:
                src_mask = torch.zeros(B, src_len, dtype=torch.bool, device=src_ids.device)
            if tgt_mask is None:
                tgt_mask = torch.zeros(B, tgt_len, dtype=torch.bool, device=src_ids.device)
            pad_mask = torch.cat([src_mask, tgt_mask], dim=1).unsqueeze(-1)
            x = x.masked_fill(pad_mask, 0.0)

        rope_cos, rope_sin = self.rope(total_len)
        c = self.timestep_emb(t, r)

        for layer in self.layers:
            x = layer(x, c, rope_cos, rope_sin)

        x_xml = x[:, src_len:]
        scale, shift = self.final_adaLN(c).chunk(2, dim=-1)
        x_xml = self.final_norm(x_xml) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x_xml = self.final_proj(x_xml)  # (B, tgt_len, d_model) — compact

        # Chunk vocab projection: never hold (B, tgt_len, vocab) at once.
        token_ids = torch.empty(B, tgt_len, dtype=torch.long, device=src_ids.device)
        for start in range(0, tgt_len, chunk_size):
            end = min(start + chunk_size, tgt_len)
            token_ids[:, start:end] = self.embedding.project_to_vocab(x_xml[:, start:end]).argmax(dim=-1)
        return token_ids

    def predict_length(self, src_ids, src_mask=None):
        """Predict output length bucket from JSON input."""
        json_emb = self.embedding(src_ids)
        if src_mask is None:
            src_mask = (src_ids == self.pad_id)
        return self.length_predictor(json_emb, src_mask)



def build_model(
    vocab_size: int = 16384,
    d_model: int = 512,
    n_layers: int = 10,
    n_heads: int = 8,
    d_ff: int = 1536,
    emb_rank: int = 128,
    dropout: float = 0.1,
    pad_id: int = 0,
) -> DiffusionTransmutationModel:
    """Build the diffusion transmutation model."""
    model = DiffusionTransmutationModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        emb_rank=emb_rank,
        dropout=dropout,
        pad_id=pad_id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model
