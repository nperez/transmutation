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

"""Validate with whitespace-normalized comparison + XML parse check."""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import sentencepiece as spm
import torch

from model import TransmutationModel


def greedy_decode(model, src_ids, sp, max_len=2048, device="cuda"):
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
    return tgt_ids[1:]


def normalize(text):
    """Collapse all whitespace for comparison."""
    return re.sub(r'\s+', ' ', text.strip())


def try_parse_xml(text):
    """Try to parse as XML. Return (success, error)."""
    try:
        ET.fromstring(text.strip())
        return True, None
    except ET.ParseError as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--data-dir", default="data/val")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    model = TransmutationModel(
        vocab_size=sp.get_piece_size(),
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}\n")

    records = []
    for shard in sorted(Path(args.data_dir).glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= args.n_samples:
                    break
        if len(records) >= args.n_samples:
            break

    exact_raw = 0
    exact_norm = 0
    xml_parseable = 0
    target_parseable = 0
    correct_structure = 0

    for i, rec in enumerate(records):
        src_ids = sp.encode(rec["input"])[:1024]
        pred_ids = greedy_decode(model, src_ids, sp, device=device)
        pred = sp.decode(pred_ids)
        target = rec["target"]

        raw_match = pred.strip() == target.strip()
        norm_match = normalize(pred) == normalize(target)
        pred_xml_ok, pred_xml_err = try_parse_xml(pred)
        tgt_xml_ok, _ = try_parse_xml(target)

        if raw_match:
            exact_raw += 1
        if norm_match:
            exact_norm += 1
        if pred_xml_ok:
            xml_parseable += 1
        if tgt_xml_ok:
            target_parseable += 1

        # Check if structural tags match (ignoring text content).
        pred_tags = re.findall(r'</?(?:object|entry|key|value|array)>', pred)
        tgt_tags = re.findall(r'</?(?:object|entry|key|value|array)>', target)
        if pred_tags == tgt_tags:
            correct_structure += 1

        status = "EXACT" if norm_match else ("XML_OK" if pred_xml_ok else "FAIL")
        if not norm_match and i < 5:
            print(f"--- Sample {i+1} [{status}] ---")
            print(f"  IN:  {rec['input'][:120]}...")
            print(f"  TGT: {normalize(target)[:120]}...")
            print(f"  OUT: {normalize(pred)[:120]}...")
            if pred_xml_err:
                print(f"  XML ERR: {pred_xml_err}")
            print()

    print(f"========== RESULTS ({len(records)} samples) ==========")
    print(f"Exact match (raw):        {exact_raw:3d}/{len(records)} ({exact_raw/len(records):.1%})")
    print(f"Exact match (normalized): {exact_norm:3d}/{len(records)} ({exact_norm/len(records):.1%})")
    print(f"XML parseable (pred):     {xml_parseable:3d}/{len(records)} ({xml_parseable/len(records):.1%})")
    print(f"XML parseable (target):   {target_parseable:3d}/{len(records)} ({target_parseable/len(records):.1%})")
    print(f"Correct tag structure:    {correct_structure:3d}/{len(records)} ({correct_structure/len(records):.1%})")


if __name__ == "__main__":
    main()
