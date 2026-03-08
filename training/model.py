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

"""Mamba-based encoder-decoder for broken JSON to XML translation."""

import math

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba


class TransmutationModel(nn.Module):
    """
    Encoder-decoder model:
    - Encoder: Mamba blocks processing the corrupted input
    - Decoder: Mamba blocks with cross-attention to encoder states
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 6,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Shared embedding (input and output share the same vocabulary).
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_scale = math.sqrt(d_model)

        # Encoder: stack of Mamba blocks.
        self.encoder_layers = nn.ModuleList([
            MambaEncoderLayer(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder: Mamba blocks interleaved with cross-attention.
        self.decoder_layers = nn.ModuleList([
            MambaDecoderLayer(d_model, d_state, d_conv, expand, n_heads, dropout)
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
        """Decode target sequence with cross-attention to encoder output."""
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
        Returns: logits (batch, tgt_len, vocab_size)
        """
        memory = self.encode(src_ids)
        logits = self.decode(tgt_ids, memory, src_key_padding_mask)
        return logits


class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)


class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, n_heads, dropout):
        super().__init__()
        # Self-"attention" via Mamba.
        self.self_norm = nn.LayerNorm(d_model)
        self.self_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Cross-attention to encoder output.
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

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

    def forward(self, x, memory, memory_key_padding_mask=None):
        # Self-attention (Mamba).
        residual = x
        x = self.self_norm(x)
        x = self.self_mamba(x)
        x = residual + self.dropout(x)

        # Cross-attention.
        residual = x
        x = self.cross_norm(x)
        x, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = residual + self.dropout(x)

        # Feedforward.
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x

        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(vocab_size: int, **kwargs) -> TransmutationModel:
    """Build model and print parameter count."""
    model = TransmutationModel(vocab_size, **kwargs)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    return model
