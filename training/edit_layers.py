#!/usr/bin/env python3
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

"""Edit autoencoder transformer layer counts via Net2DeeperNet.

Expand: inserts identity-initialized layers (zero output projections so
residual passes through). Strategy 'append' adds at end, 'interleave'
alternates new layers between existing ones.

Contract: removes layers, keeping evenly spaced ones (including first).

Drops optimizer state (shapes changed). Resets best_token_acc.
"""

import argparse
import math
import torch


def _detect_layers(state, prefix):
    """Count transformer layers with given prefix in state dict."""
    n = 0
    while f"{prefix}.{n}.qkv.weight" in state:
        n += 1
    return n


def _extract_layer(state, prefix, idx):
    """Extract all weights for a single transformer layer."""
    layer = {}
    for key, val in state.items():
        layer_prefix = f"{prefix}.{idx}."
        if key.startswith(layer_prefix):
            suffix = key[len(layer_prefix):]
            layer[suffix] = val.clone()
    return layer


def _identity_layer(ref):
    """Create an identity-initialized transformer layer.

    Copies structure from a reference layer dict. Zero-inits out_proj and ff
    output so the residual connection passes through: x + 0 = x.
    QKV and ff input get small Kaiming noise for symmetry breaking.
    """
    new = {}
    for suffix, val in ref.items():
        if suffix in ("out_proj.weight", "out_proj.bias"):
            # Zero attention output -> residual passthrough.
            new[suffix] = torch.zeros_like(val)
        elif suffix == "ff.3.weight":
            # Zero FFN output weight -> residual passthrough.
            new[suffix] = torch.zeros_like(val)
        elif suffix == "ff.3.bias":
            # Zero FFN output bias.
            new[suffix] = torch.zeros_like(val)
        elif suffix in ("norm1.weight", "norm2.weight"):
            # LayerNorm weight = 1 (standard init).
            new[suffix] = torch.ones_like(val)
        elif suffix in ("norm1.bias", "norm2.bias"):
            # LayerNorm bias = 0 (standard init).
            new[suffix] = torch.zeros_like(val)
        elif suffix in ("qkv.weight", "ff.0.weight"):
            # Small Kaiming noise for symmetry breaking.
            fan_in = val.shape[1]
            noise_scale = 0.01 / math.sqrt(fan_in)
            new[suffix] = torch.randn_like(val) * noise_scale
        elif suffix in ("qkv.bias", "ff.0.bias"):
            new[suffix] = torch.zeros_like(val)
        else:
            # Anything else: zero.
            new[suffix] = torch.zeros_like(val)
    return new


def _write_layer(state, prefix, idx, layer_weights):
    """Write layer weights into state dict at given index."""
    for suffix, val in layer_weights.items():
        state[f"{prefix}.{idx}.{suffix}"] = val


def _remove_layers(state, prefix, n_old):
    """Remove all layers with given prefix from state dict."""
    for idx in range(n_old):
        keys = [k for k in state if k.startswith(f"{prefix}.{idx}.")]
        for k in keys:
            del state[k]


def expand_layers(state, prefix, n_old, n_new, strategy):
    """Add layers to a transformer stack."""
    n_add = n_new - n_old
    old_layers = [_extract_layer(state, prefix, i) for i in range(n_old)]

    ref_layer = old_layers[-1]  # reference for identity init shape

    if strategy == "append":
        # Keep old layers in place, append new identity layers at end.
        _remove_layers(state, prefix, n_old)
        for i in range(n_old):
            _write_layer(state, prefix, i, old_layers[i])
        for i in range(n_old, n_new):
            _write_layer(state, prefix, i, _identity_layer(ref_layer))
    elif strategy == "interleave":
        # Place old layers at evenly spaced positions, fill gaps with identity.
        # E.g., 2 old + 2 new = 4 total: positions 0,2 get old; 1,3 get new.
        _remove_layers(state, prefix, n_old)
        old_positions = set()
        for i in range(n_old):
            old_positions.add(round(i * n_new / n_old))

        old_iter = iter(range(n_old))
        for i in range(n_new):
            if i in old_positions:
                _write_layer(state, prefix, i, old_layers[next(old_iter)])
            else:
                _write_layer(state, prefix, i, _identity_layer(ref_layer))

    return n_new


def contract_layers(state, prefix, n_old, n_new):
    """Remove layers from a transformer stack, keeping evenly spaced ones."""
    old_layers = [_extract_layer(state, prefix, i) for i in range(n_old)]
    _remove_layers(state, prefix, n_old)

    # Keep evenly spaced layers (always including first).
    keep_indices = []
    for i in range(n_new):
        idx = round(i * (n_old - 1) / max(n_new - 1, 1))
        keep_indices.append(idx)

    for new_idx, old_idx in enumerate(keep_indices):
        _write_layer(state, prefix, new_idx, old_layers[old_idx])

    return n_new


def edit(checkpoint_path, output_path, enc_layers, dec_layers, strategy):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    cur_enc = _detect_layers(state, "enc_transformer")
    cur_dec = _detect_layers(state, "dec_transformer")

    print(f"Current layers: encoder={cur_enc}, decoder={cur_dec}")
    print(f"Target layers:  encoder={enc_layers}, decoder={dec_layers}")
    print(f"Strategy: {strategy}")

    if enc_layers > cur_enc:
        print(f"\n  Encoder: {cur_enc} → {enc_layers} (+{enc_layers - cur_enc} identity layers, {strategy})")
        expand_layers(state, "enc_transformer", cur_enc, enc_layers, strategy)
    elif enc_layers < cur_enc:
        print(f"\n  Encoder: {cur_enc} → {enc_layers} (-{cur_enc - enc_layers} layers removed)")
        contract_layers(state, "enc_transformer", cur_enc, enc_layers)
    else:
        print(f"\n  Encoder: unchanged ({cur_enc} layers)")

    if dec_layers > cur_dec:
        print(f"  Decoder: {cur_dec} → {dec_layers} (+{dec_layers - cur_dec} identity layers, {strategy})")
        expand_layers(state, "dec_transformer", cur_dec, dec_layers, strategy)
    elif dec_layers < cur_dec:
        print(f"  Decoder: {cur_dec} → {dec_layers} (-{cur_dec - dec_layers} layers removed)")
        contract_layers(state, "dec_transformer", cur_dec, dec_layers)
    else:
        print(f"  Decoder: unchanged ({cur_dec} layers)")

    # Verify layer counts.
    final_enc = _detect_layers(state, "enc_transformer")
    final_dec = _detect_layers(state, "dec_transformer")
    assert final_enc == enc_layers, f"Encoder: expected {enc_layers}, got {final_enc}"
    assert final_dec == dec_layers, f"Decoder: expected {dec_layers}, got {final_dec}"

    # Save without optimizer/scaler state (shapes changed).
    new_ckpt = {
        "epoch": ckpt["epoch"],
        "epoch_complete": True,
        "epoch_step": 0,
        "epoch_seed": None,
        "global_step": ckpt.get("global_step", 0),
        "best_token_acc": 0.0,  # reset — architecture changed
        "lr_patience_counter": 0,
        "format": ckpt.get("format", "unknown"),
        "wallclock": ckpt.get("wallclock", 0.0),
        "model_state_dict": state,
    }

    torch.save(new_ckpt, output_path)
    print(f"\nSaved to {output_path}")
    print(f"  Final: encoder={final_enc} layers, decoder={final_dec} layers")
    print(f"  Optimizer state dropped (fresh Adam on resume)")
    print(f"  best_token_acc reset to 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edit AE transformer layer counts via Net2DeeperNet")
    parser.add_argument("checkpoint", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--enc-layers", type=int, required=True,
                        help="Target encoder transformer layers")
    parser.add_argument("--dec-layers", type=int, required=True,
                        help="Target decoder transformer layers")
    parser.add_argument("--strategy", choices=["append", "interleave"],
                        default="append",
                        help="Where to insert new layers (default: append)")
    args = parser.parse_args()
    edit(args.checkpoint, args.output, args.enc_layers, args.dec_layers,
         args.strategy)
