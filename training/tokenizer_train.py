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

"""Train a SentencePiece BPE tokenizer on the generated corpus."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm


def extract_corpus(data_dir: str, output_path: str, max_samples: int = 200000):
    """Extract text from JSONL shards into a plain text file for tokenizer training."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for split in ["train", "val"]:
            split_dir = Path(data_dir) / split
            if not split_dir.exists():
                continue
            for shard in sorted(split_dir.glob("*.jsonl")):
                with open(shard, encoding="utf-8") as f:
                    for line in f:
                        if count >= max_samples:
                            return count
                        record = json.loads(line)
                        # Write both input and target as separate lines.
                        out.write(record["input"] + "\n")
                        out.write(record["target"] + "\n")
                        count += 1
    return count


def train_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int = 8000,
):
    """Train a SentencePiece BPE tokenizer."""
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,  # full byte coverage
        byte_fallback=True,  # handle arbitrary bytes
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        # Ensure XML structural tokens get their own pieces.
        user_defined_symbols=[
            "<object>", "</object>",
            "<entry>", "</entry>",
            "<key>", "</key>",
            "<value>", "</value>",
            "<array>", "</array>",
            "<![CDATA[", "]]>",
        ],
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--data-dir", default="data", help="Data directory with train/val splits")
    parser.add_argument("--output-dir", default="models", help="Output directory for tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--max-samples", type=int, default=200000, help="Max samples for tokenizer training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "tokenizer")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        corpus_path = tmp.name

    try:
        print(f"Extracting corpus from {args.data_dir}...")
        n = extract_corpus(args.data_dir, corpus_path, args.max_samples)
        print(f"Extracted {n} samples ({n * 2} text segments)")

        print(f"Training tokenizer with vocab_size={args.vocab_size}...")
        train_tokenizer(corpus_path, model_prefix, args.vocab_size)
        print(f"Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")

        # Quick sanity check.
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        test_input = '<object>\n  <entry>\n    <key>test</key>\n    <value>hello</value>\n  </entry>\n</object>'
        tokens = sp.encode(test_input, out_type=str)
        print(f"\nSanity check:")
        print(f"  Input:  {test_input[:60]}...")
        print(f"  Tokens: {tokens[:20]}...")
        print(f"  IDs:    {sp.encode(test_input)[:20]}...")
        print(f"  Vocab:  {sp.get_piece_size()} pieces")
    finally:
        os.unlink(corpus_path)


if __name__ == "__main__":
    main()
