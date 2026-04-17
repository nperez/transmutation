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

"""Edit autoencoder conv stack: drop middle conv layer, adjust strides.

Converts a 3-conv (stride 2,2,2 = 8x) checkpoint to a 2-conv (stride 2,4 = 8x)
checkpoint. Keeps first and last conv weights. The last conv's weights transfer
directly (same kernel shape), just run at stride 4 instead of 2.

Drops optimizer state and resets best_token_acc.
"""

import argparse
import torch


def _detect_conv_layers(state, prefix):
    """Count conv layers in a Sequential. Conv layers are at indices 0, 3, 6, ..."""
    n = 0
    while f"{prefix}.{n * 3}.weight" in state:
        n += 1
    return n


def _remap_convs(state, prefix, keep_indices):
    """Remap conv blocks in a Sequential, dropping unneeded ones.

    Each block is 3 modules: Conv/ConvTranspose (idx*3), GroupNorm (idx*3+1), GELU (no params).
    keep_indices: which old block indices to keep, in order.
    """
    # Extract blocks to keep.
    blocks = []
    for old_idx in keep_indices:
        block = {}
        for suffix in ["weight", "bias"]:
            # Conv/ConvTranspose params
            key = f"{prefix}.{old_idx * 3}.{suffix}"
            if key in state:
                block[f"conv.{suffix}"] = state[key]
            # GroupNorm params
            key = f"{prefix}.{old_idx * 3 + 1}.{suffix}"
            if key in state:
                block[f"norm.{suffix}"] = state[key]
        blocks.append(block)

    # Remove all old keys.
    old_keys = [k for k in state if k.startswith(prefix + ".")]
    for k in old_keys:
        del state[k]

    # Write back at new indices.
    for new_idx, block in enumerate(blocks):
        for key, val in block.items():
            if key.startswith("conv."):
                suffix = key[5:]  # strip "conv."
                state[f"{prefix}.{new_idx * 3}.{suffix}"] = val
            elif key.startswith("norm."):
                suffix = key[5:]  # strip "norm."
                state[f"{prefix}.{new_idx * 3 + 1}.{suffix}"] = val


def edit(checkpoint_path, output_path, target_strides):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    n_enc = _detect_conv_layers(state, "enc_convs")
    n_dec = _detect_conv_layers(state, "dec_convs")
    n_target = len(target_strides)

    # Detect current channels.
    enc_channels = []
    for i in range(n_enc):
        enc_channels.append(state[f"enc_convs.{i * 3}.weight"].shape[0])

    print(f"Current: {n_enc} conv layers, channels={tuple(enc_channels)}")
    print(f"Target:  {n_target} conv layers, strides={target_strides}")

    if n_enc == 3 and n_target == 2:
        # Drop middle conv. Keep first (index 0) and last (index 2).
        # Encoder: block 0 stays, block 1 dropped, block 2 → block 1
        print(f"\n  Encoder: keeping blocks [0, 2], dropping block 1")
        print(f"    Block 0: {state['enc_convs.0.weight'].shape} (stride {target_strides[0]})")
        print(f"    Block 2→1: {state['enc_convs.6.weight'].shape} (stride {target_strides[1]})")
        _remap_convs(state, "enc_convs", [0, 2])

        # Decoder: reverse order. Block 0 is the mirror of enc block 2,
        # block 2 is the mirror of enc block 0.
        # Keep first (index 0, mirrors enc's last) and last (index 2, mirrors enc's first).
        print(f"\n  Decoder: keeping blocks [0, 2], dropping block 1")
        print(f"    Block 0: {state['dec_convs.0.weight'].shape} (stride {target_strides[1]})")
        print(f"    Block 2→1: {state['dec_convs.6.weight'].shape} (stride {target_strides[0]})")
        _remap_convs(state, "dec_convs", [0, 2])

        new_channels = (enc_channels[0], enc_channels[2])
    else:
        raise ValueError(f"Unsupported conversion: {n_enc} convs → {n_target} convs")

    # Save.
    new_ckpt = {
        "epoch": ckpt["epoch"],
        "epoch_complete": True,
        "epoch_step": 0,
        "epoch_seed": None,
        "global_step": ckpt.get("global_step", 0),
        "best_token_acc": 0.0,
        "lr_patience_counter": 0,
        "format": ckpt.get("format", "unknown"),
        "wallclock": ckpt.get("wallclock", 0.0),
        "conv_strides": list(target_strides),
        "model_state_dict": state,
    }

    torch.save(new_ckpt, output_path)
    print(f"\nSaved to {output_path}")
    print(f"  Channels: {new_channels}, strides: {target_strides}")
    print(f"  Total compression: {target_strides[0] * target_strides[1]}x")
    print(f"  Optimizer state dropped, best_token_acc reset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edit AE conv stack: drop middle layer, adjust strides")
    parser.add_argument("checkpoint", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--strides", required=True,
                        help="Comma-separated target strides (e.g. 2,4)")
    args = parser.parse_args()
    target_strides = tuple(int(x) for x in args.strides.split(","))
    edit(args.checkpoint, args.output, target_strides)
