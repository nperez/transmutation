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

"""Export diffusion model to ONNX.

Exports three artifacts:
  1. diffusion.onnx — main denoising model (src_ids, noised_emb, timestep → pred_emb)
  2. length_predictor.onnx — output length bucket classifier (src_ids → logits)
  3. embedding tables as numpy files for Go discretization
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import sentencepiece as spm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


class DenoiserWrapper(nn.Module):
    """Wrapper for ONNX export of the main denoising forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids, noised_emb, timestep, step_size):
        src_mask = (src_ids == self.model.pad_id)
        return self.model(src_ids, noised_emb, timestep, src_mask, None, r=step_size)


class LengthPredictorWrapper(nn.Module):
    """Wrapper for ONNX export of the length prediction head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids):
        src_mask = (src_ids == self.model.pad_id)
        return self.model.predict_length(src_ids, src_mask)


def export_denoiser(model, output_path, device):
    """Export the main denoising model to ONNX."""
    wrapper = DenoiserWrapper(model).to(device)
    wrapper.eval()

    # Dummy inputs
    src_ids = torch.randint(1, 100, (1, 64), dtype=torch.long, device=device)
    noised_emb = torch.randn(1, 32, model.d_model, device=device)
    timestep = torch.tensor([0.5], device=device)
    step_size = torch.tensor([0.25], device=device)

    torch.onnx.export(
        wrapper,
        (src_ids, noised_emb, timestep, step_size),
        output_path,
        input_names=["src_ids", "noised_emb", "timestep", "step_size"],
        output_names=["pred_emb"],
        dynamic_axes={
            "src_ids": {0: "batch", 1: "src_len"},
            "noised_emb": {0: "batch", 1: "tgt_len"},
            "timestep": {0: "batch"},
            "step_size": {0: "batch"},
            "pred_emb": {0: "batch", 1: "tgt_len"},
        },
        opset_version=17,
    )
    print(f"  Denoiser exported: {output_path}")


def export_length_predictor(model, output_path, device):
    """Export the length prediction head to ONNX."""
    wrapper = LengthPredictorWrapper(model).to(device)
    wrapper.eval()

    src_ids = torch.randint(1, 100, (1, 64), dtype=torch.long, device=device)

    torch.onnx.export(
        wrapper,
        (src_ids,),
        output_path,
        input_names=["src_ids"],
        output_names=["length_logits"],
        dynamic_axes={
            "src_ids": {0: "batch", 1: "src_len"},
            "length_logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  Length predictor exported: {output_path}")


def export_embedding_tables(model, output_dir):
    """Export embedding tables as numpy for Go discretization."""
    # down: (vocab, rank) — the lookup table
    emb_down = model.embedding.down.weight.detach().cpu().float().numpy()
    # up: (d_model, rank) — the projection matrix
    emb_up = model.embedding.up.weight.detach().cpu().float().numpy()

    down_path = os.path.join(output_dir, "emb_down.npy")
    up_path = os.path.join(output_dir, "emb_up.npy")
    np.save(down_path, emb_down)
    np.save(up_path, emb_up)
    print(f"  Embedding tables: down {emb_down.shape} -> {down_path}")
    print(f"  Embedding tables: up {emb_up.shape} -> {up_path}")


def validate_export(model, denoiser_path, length_path, sp, device):
    """Validate ONNX output matches PyTorch output."""
    import onnxruntime as ort

    src_ids = torch.randint(4, sp.get_piece_size(), (1, 50), dtype=torch.long, device=device)
    noised_emb = torch.randn(1, 30, model.d_model, device=device)
    timestep = torch.tensor([0.3], device=device)
    src_mask = (src_ids == sp.pad_id())

    # PyTorch reference
    with torch.no_grad():
        pt_pred = model(src_ids, noised_emb, timestep, src_mask, None)
        pt_length = model.predict_length(src_ids, src_mask)

    # ONNX inference
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    ort_denoiser = ort.InferenceSession(denoiser_path, providers=providers)
    ort_length = ort.InferenceSession(length_path, providers=providers)

    ort_pred = ort_denoiser.run(None, {
        "src_ids": src_ids.cpu().numpy(),
        "noised_emb": noised_emb.cpu().numpy(),
        "timestep": timestep.cpu().numpy(),
    })[0]

    ort_len = ort_length.run(None, {
        "src_ids": src_ids.cpu().numpy(),
    })[0]

    # Compare
    pred_diff = abs(pt_pred.cpu().numpy() - ort_pred).max()
    len_diff = abs(pt_length.cpu().numpy() - ort_len).max()
    print(f"  Validation: denoiser max diff = {pred_diff:.6f}, length max diff = {len_diff:.6f}")
    if pred_diff > 0.01:
        print(f"  WARNING: denoiser diff {pred_diff:.6f} exceeds threshold")
    if len_diff > 0.01:
        print(f"  WARNING: length predictor diff {len_diff:.6f} exceeds threshold")


def main():
    parser = argparse.ArgumentParser(description="Export diffusion model to ONNX")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--output-dir", default="models/onnx")
    args = parser.parse_args()

    device = torch.device("cpu")  # export on CPU

    # Load model from checkpoint (auto-detect config, loads tokenizer from checkpoint dir)
    from infer import load_model
    model, sp = load_model(args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)
    denoiser_path = os.path.join(args.output_dir, "diffusion.onnx")
    length_path = os.path.join(args.output_dir, "length_predictor.onnx")

    with torch.no_grad():
        export_denoiser(model, denoiser_path, device)
        export_length_predictor(model, length_path, device)
        export_embedding_tables(model, args.output_dir)

    onnx.checker.check_model(onnx.load(denoiser_path))
    onnx.checker.check_model(onnx.load(length_path))
    print("ONNX validation passed")

    validate_export(model, denoiser_path, length_path, sp, device)

    den_mb = os.path.getsize(denoiser_path) / 1024 / 1024
    len_mb = os.path.getsize(length_path) / 1024 / 1024
    print(f"\nDenoiser: {den_mb:.1f} MB, Length predictor: {len_mb:.1f} MB")

    # Dynamic int8 quantization.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    den_q_path = os.path.join(args.output_dir, "diffusion_int8.onnx")
    len_q_path = os.path.join(args.output_dir, "length_predictor_int8.onnx")
    quantize_dynamic(denoiser_path, den_q_path, weight_type=QuantType.QInt8)
    quantize_dynamic(length_path, len_q_path, weight_type=QuantType.QInt8)
    den_q_mb = os.path.getsize(den_q_path) / 1024 / 1024
    len_q_mb = os.path.getsize(len_q_path) / 1024 / 1024
    print(f"Quantized (int8): Denoiser: {den_q_mb:.1f} MB, Length: {len_q_mb:.1f} MB")


if __name__ == "__main__":
    main()
