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

"""Pre-tokenize and package training data into tensor files.

Processes records in chunks to keep memory bounded: raw JSON strings
are only held for one chunk at a time, then discarded.
"""

import argparse
import array
import json
import os
from multiprocessing import Pool

import sentencepiece as spm
import torch

CHUNK_SIZE = 50000

_sp = None


def _init_worker(tok_path):
    global _sp
    _sp = spm.SentencePieceProcessor()
    _sp.load(tok_path)


def _tokenize(rec):
    """Tokenize and reduce a record to compact form.

    Returns (src_ids, tgt_ids, complexity, is_corrupt) or None if filtered.
    Raw strings are NOT returned — caller discards them.
    """
    src_ids = _sp.encode(rec["input"])
    if len(src_ids) > rec["_max_src"]:
        return None
    tgt_ids = _sp.encode(rec["target"])
    if len(tgt_ids) > rec["_max_tgt"]:
        return None

    cplx = rec.get("complexity")
    if cplx is None:
        cplx = complexity_score(rec["input"])
    corrupt = rec.get("aug_type", "clean") == "corrupted"

    return (array.array('h', src_ids), array.array('h', tgt_ids), cplx, corrupt)


def complexity_score(input_json):
    """Assign a 1-8 score based on structural complexity.

    Memory:  0 items=0, 1-3=+1, 4+=+2
    Tool:    null=0, present=+2, 3+ args=+1, nested arg values=+1
    Answer:  null=0, plain text=+1, markdown=+1, code blocks=+1, tables=+1
    Length:  >2500 chars=+1
    """
    score = 1
    try:
        obj = json.loads(input_json)
    except (json.JSONDecodeError, TypeError):
        return 1

    mem = obj.get("memory")
    if isinstance(mem, list) and len(mem) > 0:
        score += 1
        if len(mem) >= 4:
            score += 1

    tool = obj.get("tool")
    if tool is not None:
        score += 2
        if isinstance(tool, dict):
            args = tool.get("arguments")
            if isinstance(args, dict):
                if len(args) >= 3:
                    score += 1
                for v in args.values():
                    if isinstance(v, (dict, list)):
                        score += 1
                        break

    answer = obj.get("answer")
    if answer is not None and isinstance(answer, str) and len(answer) > 0:
        score += 1
        if "## " in answer or "- " in answer:
            score += 1
        if "```" in answer:
            score += 1
        if "| ---" in answer or "|---" in answer:
            score += 1

    if len(input_json) > 2500:
        score += 1

    return min(score, 8)


def iter_jsonl(data_dir, max_src, max_tgt):
    """Yield parsed JSON records from all .jsonl files, with length limits injected."""
    max_tgt_m1 = max_tgt - 1
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.endswith(".jsonl"):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["_max_src"] = max_src
                        rec["_max_tgt"] = max_tgt_m1
                        yield rec


def build_dataset(data_dir, tok_path, max_src, max_tgt):
    """Tokenize in chunks via multiprocessing. Only compact data survives between chunks."""
    all_src = []
    all_tgt = []
    all_complexity = []
    all_corrupt = []
    total = 0
    dropped = 0

    n_workers = min(os.cpu_count() or 4, 8)
    chunk = []

    def flush_chunk(pool, chunk):
        nonlocal total, dropped
        total += len(chunk)
        for result in pool.imap(_tokenize, chunk, chunksize=512):
            if result is None:
                dropped += 1
            else:
                src_ids, tgt_ids, cplx, corrupt = result
                all_src.append(src_ids)
                all_tgt.append(tgt_ids)
                all_complexity.append(cplx)
                all_corrupt.append(corrupt)
        print(f"    processed {total}, kept {len(all_src)}, dropped {dropped}", flush=True)

    with Pool(n_workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        for rec in iter_jsonl(data_dir, max_src, max_tgt):
            chunk.append(rec)
            if len(chunk) >= CHUNK_SIZE:
                flush_chunk(pool, chunk)
                chunk = []
        if chunk:
            flush_chunk(pool, chunk)

    n = len(all_src)
    if n == 0:
        return {
            "src": torch.zeros(0, 1, dtype=torch.int16),
            "tgt": torch.zeros(0, 1, dtype=torch.int16),
            "src_lens": torch.zeros(0, dtype=torch.int32),
            "tgt_lens": torch.zeros(0, dtype=torch.int32),
            "complexity": torch.zeros(0, dtype=torch.int8),
            "corrupt": torch.zeros(0, dtype=torch.bool),
        }

    src_lens = [len(s) for s in all_src]
    tgt_lens = [len(t) for t in all_tgt]
    ms = max(src_lens)
    mt = max(tgt_lens)

    print(f"    building tensors: {n} samples, max_src={ms}, max_tgt={mt}", flush=True)
    src_pad = torch.zeros(n, ms, dtype=torch.int16)
    tgt_pad = torch.zeros(n, mt, dtype=torch.int16)
    for i in range(n):
        s, t = all_src[i], all_tgt[i]
        src_pad[i, :len(s)] = torch.frombuffer(s, dtype=torch.int16)
        tgt_pad[i, :len(t)] = torch.frombuffer(t, dtype=torch.int16)

    # Free the lists before saving.
    del all_src, all_tgt

    return {
        "src": src_pad,
        "tgt": tgt_pad,
        "src_lens": torch.tensor(src_lens, dtype=torch.int32),
        "tgt_lens": torch.tensor(tgt_lens, dtype=torch.int32),
        "complexity": torch.tensor(all_complexity, dtype=torch.int8),
        "corrupt": torch.tensor(all_corrupt, dtype=torch.bool),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize training data into tensor files.")
    parser.add_argument("--tokenizer", required=True, help="Path to sentencepiece .model file")
    parser.add_argument("--data-dir", required=True, help="Root data dir (e.g. data/run6)")
    parser.add_argument("--max-src-len", type=int, default=1152)
    parser.add_argument("--max-tgt-len", type=int, default=1536)
    args = parser.parse_args()

    for split in ["train", "val"]:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  Skipping {split}: {split_dir} not found")
            continue

        print(f"  {split}: tokenizing...", flush=True)
        data = build_dataset(split_dir, args.tokenizer, args.max_src_len, args.max_tgt_len)
        out = os.path.join(split_dir, "dataset.pt")
        torch.save(data, out)
        n = len(data["src_lens"])
        print(f"  {split}: {n} samples saved to {out}")

        c = data["complexity"]
        for v in range(1, 9):
            count = (c == v).sum().item()
            if count > 0:
                print(f"    complexity {v}: {count}")
        print(f"    corrupt: {data['corrupt'].sum().item()}")

        # Free before next split.
        del data
