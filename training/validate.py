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

"""Validate trained model with actual inference (greedy decoding, not teacher-forced)."""

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from model import TransmutationModel


def greedy_decode(model, src_ids, sp, max_len=2048, device="cuda"):
    """Autoregressive greedy decoding."""
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory = model.encode(src)

        tgt_ids = [sp.bos_id()]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            logits = model.decode(tgt, memory)
            next_id = logits[0, -1].argmax().item()
            if next_id == sp.eos_id():
                break
            tgt_ids.append(next_id)

    return tgt_ids[1:]  # strip BOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--data-dir", default="data/val")
    parser.add_argument("--n-samples", type=int, default=20)
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
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        pad_id=sp.pad_id(),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}")

    # Load val samples.
    records = []
    data_path = Path(args.data_dir)
    for shard in sorted(data_path.glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= args.n_samples:
                    break
        if len(records) >= args.n_samples:
            break

    exact = 0
    xml_valid = 0
    for i, rec in enumerate(records):
        src_ids = sp.encode(rec["input"])
        if len(src_ids) > 1024:
            src_ids = src_ids[:1024]

        pred_ids = greedy_decode(model, src_ids, sp, device=device)
        pred_text = sp.decode(pred_ids)
        target_text = rec["target"]

        is_exact = pred_text.strip() == target_text.strip()
        if is_exact:
            exact += 1

        # Check if output is parseable XML-ish (has matching tags).
        has_root = pred_text.strip().startswith("<object>") or pred_text.strip().startswith("<array>")
        if has_root:
            xml_valid += 1

        print(f"\n{'='*60}")
        print(f"Sample {i+1}")
        print(f"INPUT (first 200 chars):")
        print(f"  {rec['input'][:200]}")
        print(f"EXPECTED (first 200 chars):")
        print(f"  {target_text[:200]}")
        print(f"PREDICTED (first 200 chars):")
        print(f"  {pred_text[:200]}")
        print(f"EXACT MATCH: {is_exact}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {exact}/{len(records)} exact match ({exact/len(records):.1%})")
    print(f"XML valid root: {xml_valid}/{len(records)} ({xml_valid/len(records):.1%})")


if __name__ == "__main__":
    main()
