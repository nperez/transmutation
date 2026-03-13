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

"""Dataset and data loading for transmutation training."""

import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class TransmutationDataset(Dataset):
    """Loads JSONL training pairs and tokenizes them on the fly."""

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_src_len: int = 2048,
        max_tgt_len: int = 4096,
    ):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        # Load all records into memory (they're small relative to available RAM).
        self.records = []
        data_path = Path(data_dir)
        for shard in sorted(data_path.glob("*.jsonl")):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        src_ids = self.sp.encode(record["input"])
        tgt_ids = self.sp.encode(record["target"])

        # Truncate if necessary.
        src_ids = src_ids[: self.max_src_len]
        tgt_ids = tgt_ids[: self.max_tgt_len - 1]  # leave room for EOS

        # Add BOS/EOS to target.
        tgt_input = [self.bos_id] + tgt_ids
        tgt_labels = tgt_ids + [self.eos_id]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_labels": torch.tensor(tgt_labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id=0):
    """Pad sequences to the same length within a batch."""
    src_ids = [item["src_ids"] for item in batch]
    tgt_input = [item["tgt_input"] for item in batch]
    tgt_labels = [item["tgt_labels"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input, batch_first=True, padding_value=pad_id)
    tgt_labels_padded = torch.nn.utils.rnn.pad_sequence(tgt_labels, batch_first=True, padding_value=-100)

    src_key_padding_mask = src_padded == pad_id

    return {
        "src_ids": src_padded,
        "tgt_input": tgt_input_padded,
        "tgt_labels": tgt_labels_padded,
        "src_key_padding_mask": src_key_padding_mask,
    }


class ResumableRandomSampler(Sampler):
    """Random sampler with a known seed that can resume from a given offset."""

    def __init__(self, data_source, seed: int, start_index: int = 0):
        self.data_source = data_source
        self.seed = seed
        self.start_index = start_index

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(perm[self.start_index:])

    def __len__(self):
        return len(self.data_source) - self.start_index


def create_dataloader(
    data_dir: str,
    tokenizer_path: str,
    batch_size: int = 16,
    max_src_len: int = 2048,
    max_tgt_len: int = 4096,
    shuffle: bool = True,
    num_workers: int = 2,
    pad_id: int = 0,
    epoch_seed: int | None = None,
    start_index: int = 0,
) -> DataLoader:
    dataset = TransmutationDataset(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    sampler = None
    if shuffle:
        if epoch_seed is None:
            epoch_seed = torch.randint(0, 2**31, (1,)).item()
        sampler = ResumableRandomSampler(dataset, seed=epoch_seed, start_index=start_index)
    elif start_index > 0:
        # Sequential skip: yield indices from start_index onward.
        sampler = list(range(start_index, len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True,
    ), epoch_seed
