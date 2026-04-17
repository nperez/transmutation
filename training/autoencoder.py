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

"""Hybrid Autoencoder for latent translation (Run 8).

Compresses token sequences into continuous latent vectors via
convolutional downsampling + transformer bottleneck. 8x spatial compression.
Encoder: FactoredEmbedding -> Conv1d stack -> Transformer -> latent (d_tf)
Decoder: latent (d_tf) -> Transformer -> ConvTranspose1d stack -> LayerNorm -> tied vocab projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FactoredEmbedding, RotaryEmbedding, apply_rope


class BottleneckTransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE. No timestep conditioning."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin):
        """x: (B, S, D), rope_cos/sin: (S, head_dim//2)."""
        # Pre-norm self-attention with RoPE
        h = self.norm1(x)
        B, S, D = h.shape
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        h = self.attn_dropout(h)
        x = x + h

        # Pre-norm FFN
        h = self.ff(self.norm2(x))
        x = x + h

        return x


class SlimHybridAutoencoder(nn.Module):
    """Hybrid CNN + Transformer autoencoder with 8x spatial compression.

    ~11M params with default config. All bucket sizes [64..1536] are divisible
    by 8, so conv stride math produces exact lengths (no trim needed).
    """

    def __init__(
        self,
        vocab_size=16384,
        d_emb=384,
        emb_rank=128,
        conv_channels=(384, 384, 384),
        n_enc_layers=4,
        n_dec_layers=2,
        n_heads=6,
        d_ff=1152,
        dropout=0.0,
        pad_id=0,
        max_seq_len=1536,
        conv_strides=None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_emb = d_emb

        if conv_strides is None:
            conv_strides = (2,) * len(conv_channels)

        # Shared embedding (encoder input + decoder output via project_to_vocab)
        self.embedding = FactoredEmbedding(vocab_size, d_emb, emb_rank, pad_id)

        # ── Encoder ──────────────────────────────────────────────────────
        # Strided Conv1d stack: (B, d_emb, L) -> (B, d_tf, L/total_stride)
        enc_layers = []
        in_ch = d_emb
        for out_ch, stride in zip(conv_channels, conv_strides):
            enc_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
                nn.GroupNorm(32, out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        self.enc_convs = nn.Sequential(*enc_layers)

        total_stride = 1
        for s in conv_strides:
            total_stride *= s

        d_tf = conv_channels[-1]  # transformer dim = last conv channel
        head_dim = d_tf // n_heads
        self.enc_rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len // total_stride + 1)
        self.enc_transformer = nn.ModuleList([
            BottleneckTransformerBlock(d_tf, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])

        # ── Decoder ──────────────────────────────────────────────────────
        self.dec_rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len // total_stride + 1)
        self.dec_transformer = nn.ModuleList([
            BottleneckTransformerBlock(d_tf, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])

        # Transposed Conv1d stack (reverse of encoder): (B, d_tf, L/S) -> (B, d_emb, L)
        dec_channels = list(reversed(conv_channels)) + [d_emb]
        dec_strides = list(reversed(conv_strides))
        dec_layers = []
        for in_c, out_c, stride in zip(dec_channels[:-1], dec_channels[1:], dec_strides):
            dec_layers.extend([
                nn.ConvTranspose1d(in_c, out_c, kernel_size=7, stride=stride,
                                   padding=3, output_padding=stride - 1),
                nn.GroupNorm(32, out_c),
                nn.GELU(),
            ])
        self.dec_convs = nn.Sequential(*dec_layers)

        self.dec_norm = nn.LayerNorm(d_emb)

    def encode(self, token_ids):
        """(B, L) -> (B, L/8, d_tf)"""
        x = self.embedding(token_ids)       # (B, L, d_emb)
        x = x.permute(0, 2, 1)             # (B, d_emb, L)
        x = self.enc_convs(x)              # (B, d_tf, L/8)
        x = x.permute(0, 2, 1)             # (B, L/8, d_tf)

        cos, sin = self.enc_rope(x.shape[1])
        for layer in self.enc_transformer:
            x = layer(x, cos, sin)

        return x                            # (B, L/8, d_tf)

    def _decode_to_emb(self, latent, seq_len):
        """Shared decode path up to embedding space (before vocab projection)."""
        x = latent

        cos, sin = self.dec_rope(x.shape[1])
        for layer in self.dec_transformer:
            x = layer(x, cos, sin)

        x = x.permute(0, 2, 1)             # (B, d_tf, L/8)
        x = self.dec_convs(x)              # (B, d_emb, L)
        x = x.permute(0, 2, 1)             # (B, L, d_emb)

        if x.shape[1] != seq_len:
            x = x[:, :seq_len]

        return self.dec_norm(x)             # (B, L, d_emb)

    def decode(self, latent, seq_len):
        """(B, L/8, d_tf) -> (B, seq_len, vocab_size) logits."""
        x = self._decode_to_emb(latent, seq_len)
        return self.embedding.project_to_vocab(x)

    def forward(self, token_ids):
        """(B, L) -> (B, L, vocab_size) logits."""
        latent = self.encode(token_ids)
        return self.decode(latent, token_ids.shape[1])

    def predict_tokens(self, token_ids, chunk_size=256):
        """Memory-efficient inference: argmax token IDs via chunked vocab projection."""
        seq_len = token_ids.shape[1]
        latent = self.encode(token_ids)
        x = self._decode_to_emb(latent, seq_len)

        B = x.shape[0]
        out = torch.empty(B, seq_len, dtype=torch.long, device=token_ids.device)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            out[:, start:end] = self.embedding.project_to_vocab(x[:, start:end]).argmax(dim=-1)
        return out


def build_autoencoder(
    vocab_size=16384,
    d_emb=384,
    emb_rank=128,
    conv_channels=(384, 384, 384),
    conv_strides=None,
    n_enc_layers=4,
    n_dec_layers=2,
    n_heads=6,
    d_ff=1152,
    dropout=0.0,
    pad_id=0,
    max_seq_len=1536,
):
    """Build autoencoder and print param count."""
    model = SlimHybridAutoencoder(
        vocab_size=vocab_size,
        d_emb=d_emb,
        emb_rank=emb_rank,
        conv_channels=conv_channels,
        conv_strides=conv_strides,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Autoencoder parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model
