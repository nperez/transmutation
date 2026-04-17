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

"""Expand autoencoder latent dimension via Net2Net widening.

Copies existing latent weights, initializes new dimensions with small noise
(to_latent) and zeros (from_latent) so the model is functionally identical
at init. Drops optimizer state (shapes no longer match).
"""

import argparse
import math
import torch


def expand(checkpoint_path, output_path, new_latent_dim):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    old_to_w = state["to_latent.weight"]      # (old_dim, d_tf)
    old_to_b = state["to_latent.bias"]         # (old_dim,)
    old_from_w = state["from_latent.weight"]   # (d_tf, old_dim)
    old_from_b = state["from_latent.bias"]     # (d_tf,)

    old_dim = old_to_w.shape[0]
    d_tf = old_to_w.shape[1]
    extra = new_latent_dim - old_dim

    if extra <= 0:
        print(f"Already at {old_dim} dims, nothing to expand.")
        return

    print(f"Expanding latent: {old_dim} -> {new_latent_dim} (+{extra} dims)")
    print(f"  to_latent:   ({old_dim}, {d_tf}) -> ({new_latent_dim}, {d_tf})")
    print(f"  from_latent: ({d_tf}, {old_dim}) -> ({d_tf}, {new_latent_dim})")

    # to_latent: new dims get small Kaiming-scale noise to break symmetry.
    fan_in = d_tf
    noise_scale = 0.01 / math.sqrt(fan_in)
    new_to_w = torch.zeros(new_latent_dim, d_tf)
    new_to_w[:old_dim] = old_to_w
    new_to_w[old_dim:] = torch.randn(extra, d_tf) * noise_scale

    new_to_b = torch.zeros(new_latent_dim)
    new_to_b[:old_dim] = old_to_b

    # from_latent: new columns are zero (model output unchanged at init).
    new_from_w = torch.zeros(d_tf, new_latent_dim)
    new_from_w[:, :old_dim] = old_from_w

    state["to_latent.weight"] = new_to_w
    state["to_latent.bias"] = new_to_b
    state["from_latent.weight"] = new_from_w
    state["from_latent.bias"] = old_from_b  # unchanged (d_tf,)

    # Save without optimizer/scaler state (shapes no longer match).
    new_ckpt = {
        "epoch": ckpt["epoch"],
        "epoch_complete": True,
        "epoch_step": 0,
        "epoch_seed": None,
        "global_step": ckpt.get("global_step", 0),
        "best_token_acc": 0.0,  # reset — new architecture
        "lr_patience_counter": 0,
        "format": ckpt.get("format", "unknown"),
        "wallclock": ckpt.get("wallclock", 0.0),
        "model_state_dict": state,
    }

    torch.save(new_ckpt, output_path)
    print(f"Saved expanded checkpoint to {output_path}")
    print(f"  Optimizer state dropped (fresh Adam on resume)")
    print(f"  best_token_acc reset to 0 (new architecture)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand AE latent dimension via Net2Net widening")
    parser.add_argument("checkpoint", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--new-dim", type=int, default=128,
                        help="New latent dimension (default: 128)")
    args = parser.parse_args()
    expand(args.checkpoint, args.output, args.new_dim)
