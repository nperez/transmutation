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

"""Diagnose failures: VRAM usage, inference speed, failure modes."""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import sentencepiece as spm
import torch

from model import TransmutationModel


def greedy_decode(model, src_ids, sp, max_len=4096, device="cuda"):
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


def try_parse_xml(text):
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
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    model = TransmutationModel(
        vocab_size=sp.get_piece_size(),
        d_model=384, n_encoder_layers=6, n_decoder_layers=6,
        d_state=16, n_heads=6, pad_id=sp.pad_id(),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # VRAM after model load
    print(f"VRAM allocated after model load: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"VRAM reserved after model load:  {torch.cuda.memory_reserved()/1024**2:.0f} MB")
    print(f"VRAM total:                      {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print()

    records = []
    for shard in sorted(Path(args.data_dir).glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                records.append(json.loads(line))
                if len(records) >= args.n_samples:
                    break
        if len(records) >= args.n_samples:
            break

    # Bucketize by input token length
    buckets = {"short(<256)": [], "med(256-512)": [], "long(512-1024)": [], "xl(1024+)": []}
    fail_reasons = {"unclosed_tag": 0, "unclosed_cdata": 0, "truncated": 0, "other": 0}

    times = []
    xml_ok = 0
    peak_vram = 0

    for i, rec in enumerate(records):
        src_ids = sp.encode(rec["input"])
        src_len = len(src_ids)
        # DON'T truncate - let's see what happens with full length
        if src_len > 2048:
            src_ids = src_ids[:2048]
            src_len = 2048

        t0 = time.time()
        pred_ids = greedy_decode(model, src_ids, sp, device=device)
        elapsed = time.time() - t0
        times.append(elapsed)

        pred = sp.decode(pred_ids)
        tgt_len = len(pred_ids)

        cur_vram = torch.cuda.max_memory_allocated() / 1024**2
        if cur_vram > peak_vram:
            peak_vram = cur_vram

        ok, err = try_parse_xml(pred)
        if ok:
            xml_ok += 1

        # Bucket
        if src_len < 256:
            bk = "short(<256)"
        elif src_len < 512:
            bk = "med(256-512)"
        elif src_len < 1024:
            bk = "long(512-1024)"
        else:
            bk = "xl(1024+)"
        buckets[bk].append((ok, src_len, tgt_len, elapsed))

        # Failure analysis
        if not ok and err:
            if "unclosed" in err.lower() and "cdata" in err.lower():
                fail_reasons["unclosed_cdata"] += 1
            elif "unclosed" in err.lower() or "mismatched" in err.lower():
                fail_reasons["unclosed_tag"] += 1
            elif len(pred_ids) >= 4090:
                fail_reasons["truncated"] += 1
            else:
                fail_reasons["other"] += 1
                if fail_reasons["other"] <= 3:
                    print(f"  OTHER FAIL sample {i}: err={err}")
                    print(f"    src_len={src_len} tgt_len={tgt_len}")
                    print(f"    last 100 chars: ...{pred[-100:]}")
                    print()

        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{len(records)}] xml_ok={xml_ok} peak_vram={peak_vram:.0f}MB")

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(records)} samples)")
    print(f"{'='*60}")
    print(f"XML parseable: {xml_ok}/{len(records)} ({xml_ok/len(records):.1%})")
    print(f"Peak VRAM:     {peak_vram:.0f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print()

    print("BY INPUT LENGTH:")
    for bk, items in buckets.items():
        if not items:
            continue
        ok_count = sum(1 for x in items if x[0])
        avg_time = sum(x[3] for x in items) / len(items)
        avg_src = sum(x[1] for x in items) / len(items)
        avg_tgt = sum(x[2] for x in items) / len(items)
        print(f"  {bk:20s}: {ok_count:3d}/{len(items):3d} ({ok_count/len(items):.0%}) "
              f"avg_src={avg_src:.0f} avg_tgt={avg_tgt:.0f} avg_time={avg_time:.2f}s")

    print(f"\nFAILURE REASONS:")
    for reason, count in fail_reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")

    print(f"\nINFERENCE SPEED:")
    times.sort()
    print(f"  p50: {times[len(times)//2]:.2f}s")
    print(f"  p95: {times[int(len(times)*0.95)]:.2f}s")
    print(f"  p99: {times[int(len(times)*0.99)]:.2f}s")
    print(f"  max: {times[-1]:.2f}s")
    print(f"  min: {times[0]:.2f}s")


if __name__ == "__main__":
    main()
