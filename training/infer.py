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

"""Diffusion inference for the transmutation model.

Iterative denoising: init noise -> denoise N steps -> discretize -> XML tokens.
Supports CPU and GPU. No sequential state management (unlike AR decode).
"""

import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

from model import LENGTH_BUCKETS


def load_model(checkpoint, device):
    """Load diffusion model from checkpoint."""
    sp = spm.SentencePieceProcessor()
    tok_path = str(Path(checkpoint).parent / "tokenizer.model")
    sp.load(tok_path)

    sys.path.insert(0, str(Path(__file__).parent))
    from model import build_model

    # Detect model config from checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Infer d_model from embedding up projection
    d_model = state["embedding.up.weight"].shape[0]
    emb_rank = state["embedding.up.weight"].shape[1]
    vocab_size = state["embedding.down.weight"].shape[0]

    # Count layers
    n_layers = 0
    while f"layers.{n_layers}.qkv.weight" in state:
        n_layers += 1

    # adaLN modulation output is 6*d_model, n_heads inferred from QKV weight
    # QKV weight shape is (3*d_model, d_model), so n_heads = d_model / head_dim
    # Try common head dims (64 is standard for DiT)
    n_heads = d_model // 64  # default: head_dim=64
    for hd in [64, 128, 32]:
        if d_model % hd == 0:
            n_heads = d_model // hd
            break

    d_ff = state["layers.0.ff.0.weight"].shape[0]

    model = build_model(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, emb_rank=emb_rank, pad_id=sp.pad_id(),
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        print(f"  Ignoring unexpected keys: {unexpected}", file=sys.stderr)


    model.eval()
    print(f"Loaded: {n_layers}L d={d_model} ff={d_ff} rank={emb_rank} vocab={vocab_size} "
          f"epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}",
          file=sys.stderr)
    return model, sp


@torch.no_grad()
def denoise(model, src_ids, src_mask, tgt_len, n_steps=4, device="cuda"):
    """Iterative denoising: x_T (noise) -> x_0 (clean) in n_steps.

    Args:
        model: DiffusionTransmutationModel
        src_ids: (batch, src_len) JSON token IDs
        src_mask: (batch, src_len) True for padding
        tgt_len: output sequence length (from bucket prediction)
        n_steps: number of denoising steps
        device: torch device
    Returns:
        token_ids: (batch, tgt_len) discretized output
    """
    B = src_ids.shape[0]
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    vocab_size = m.vocab_size

    # Start from random tokens, iterate.
    current_ids = torch.randint(0, vocab_size, (B, tgt_len), device=device)

    for i in range(n_steps):
        if i == 0:
            t_val = torch.ones(B, device=device)
            r_val = torch.ones(B, device=device)
        else:
            t_val = torch.full((B,), 0.125, device=device)
            r_val = torch.full((B,), 0.125, device=device)

        current_ids = m.predict_tokens(src_ids, current_ids, t_val, src_mask, None, r=r_val)

    return current_ids


@torch.no_grad()
def batched_denoise(model, records, sp, n_steps=4, device="cuda"):
    """Denoise a batch of records. Returns list of predicted token ID lists."""
    src_ids_list = [sp.encode(r["input"]) for r in records]
    B = len(src_ids_list)

    # Pad source
    max_src = max(len(ids) for ids in src_ids_list)
    src_t = torch.zeros(B, max_src, dtype=torch.long, device=device)
    src_mask = torch.ones(B, max_src, dtype=torch.bool, device=device)
    for i, ids in enumerate(src_ids_list):
        src_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        src_mask[i, :len(ids)] = False

    # Predict length bucket
    length_logits = model.predict_length(src_t, src_mask)
    bucket_idx = length_logits.argmax(dim=-1)
    # One bucket up for safety
    bucket_idx = (bucket_idx + 1).clamp(max=len(LENGTH_BUCKETS) - 1)
    tgt_len = LENGTH_BUCKETS[bucket_idx.max().item()]

    # Denoise
    print(f"  denoise: {B} samples, tgt_len={tgt_len}, {n_steps} steps...",
          file=sys.stderr, flush=True)
    token_ids = denoise(model, src_t, src_mask, tgt_len, n_steps, device)

    # Trim at EOS/PAD per sample
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    results = []
    for i in range(B):
        ids = token_ids[i].tolist()
        trimmed = []
        for tid in ids:
            if tid == eos_id or tid == pad_id:
                break
            trimmed.append(tid)
        results.append(trimmed)
    return results


def read_records(sp, max_src_len, input_file=None):
    """Read JSONL records from file or stdin, filtering by source token length."""
    source = open(input_file, encoding="utf-8") if input_file else sys.stdin
    try:
        for line in source:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            r = json.loads(line)
            if len(sp.encode(r["input"])) <= max_src_len:
                yield r
    finally:
        if input_file:
            source.close()


def xml_semantically_equal(a, b):
    """Compare two XML strings by flattening to canonical (element, text) tokens."""
    def flatten(s):
        try:
            root = ET.fromstring(s)
        except ET.ParseError:
            return None
        tokens = []
        def walk(elem):
            tokens.append(("start", elem.tag))
            text = re.sub(r"\s+", " ", (elem.text or "").strip())
            if text:
                tokens.append(("text", text))
            for child in elem:
                walk(child)
                tail = re.sub(r"\s+", " ", (child.tail or "").strip())
                if tail:
                    tokens.append(("text", tail))
            tokens.append(("end", elem.tag))
        walk(root)
        return tokens

    af, bf = flatten(a), flatten(b)
    if af is None or bf is None:
        return False
    return af == bf


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diffusion inference for transmutation model")
    parser.add_argument("checkpoint", nargs="?", default="models/epoch_1.pt")
    parser.add_argument("-n", type=int, default=10, help="number of samples")
    parser.add_argument("--max-src-len", type=int, default=1152)
    parser.add_argument("--denoise-steps", type=int, default=4,
                        help="number of denoising steps")
    parser.add_argument("--json", action="store_true",
                        help="output JSONL per sample (for eval pipeline)")
    parser.add_argument("--input", type=str, default=None,
                        help="read JSONL from file instead of stdin")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model, sp = load_model(args.checkpoint, device)
    print(file=sys.stderr)

    # Collect records.
    records = []
    for rec in read_records(sp, args.max_src_len, input_file=args.input):
        if len(records) >= args.n:
            break
        records.append(rec)
    total = len(records)

    # Batched denoise.
    t0 = time.monotonic()
    all_pred_ids = batched_denoise(model, records, sp,
                                   n_steps=args.denoise_steps, device=device)
    elapsed = time.monotonic() - t0
    per_sample = elapsed / max(total, 1)
    print(f"Denoise done in {elapsed:.1f}s ({per_sample:.2f}s/sample)",
          file=sys.stderr, flush=True)

    # Score and output.
    xml_ok_count = 0
    semantic_count = 0
    exact_count = 0

    for i, (rec, pred_ids) in enumerate(zip(records, all_pred_ids)):
        pred = sp.decode(pred_ids)
        target = rec["target"]

        norm_pred = unicodedata.normalize("NFKD", re.sub(r"\s+", " ", pred.strip()))
        norm_tgt = unicodedata.normalize("NFKD", re.sub(r"\s+", " ", target.strip()))
        exact = norm_pred == norm_tgt

        try:
            ET.fromstring(pred.strip())
            xml_ok = True
        except ET.ParseError:
            xml_ok = False

        semantic = False
        if not exact and xml_ok:
            semantic = xml_semantically_equal(pred.strip(), target.strip())

        if exact:
            exact_count += 1
        if semantic:
            semantic_count += 1
        if xml_ok:
            xml_ok_count += 1

        tag = "EXACT" if exact else ("SEMANTIC" if semantic else ("XML_OK" if xml_ok else "FAIL"))

        if args.json:
            print(json.dumps({
                "sample": i + 1,
                "tag": tag,
                "pred_tokens": len(pred_ids),
                "elapsed": round(per_sample, 2),
                "exact": exact,
                "semantic": semantic,
                "xml_ok": xml_ok,
            }), flush=True)
        else:
            print(f"=== Sample {i+1} [{tag}] {per_sample:.2f}s, {len(pred_ids)} tokens ===")
            print(f"INPUT:\n{rec['input']}\n")
            if exact or semantic:
                print(f"OUTPUT (matches target):\n{pred.strip()}\n")
            else:
                print(f"TARGET:\n{target.strip()}\n")
                print(f"OUTPUT:\n{pred.strip()}\n")
            print()

    if not args.json:
        print(f"===== {total} samples: exact={exact_count} semantic={semantic_count} xml_ok={xml_ok_count - exact_count - semantic_count} fail={total - xml_ok_count} =====")


if __name__ == "__main__":
    main()
