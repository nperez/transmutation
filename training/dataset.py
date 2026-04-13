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
import os
from multiprocessing import Pool
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# Module-level globals for multiprocessing workers (avoids pickling SP).
_worker_sp = None
_worker_max_src = 0
_worker_max_tgt_m1 = 0


def _init_filter_worker(tokenizer_path, max_src_len, max_tgt_len_m1):
    global _worker_sp, _worker_max_src, _worker_max_tgt_m1
    _worker_sp = spm.SentencePieceProcessor()
    _worker_sp.load(tokenizer_path)
    _worker_max_src = max_src_len
    _worker_max_tgt_m1 = max_tgt_len_m1


def _filter_record(rec):
    src_ids = _worker_sp.encode(rec["input"])
    if len(src_ids) > _worker_max_src:
        return None
    tgt_ids = _worker_sp.encode(rec["target"])
    if len(tgt_ids) > _worker_max_tgt_m1:
        return None
    rec["_src_ids"] = src_ids
    rec["_tgt_ids"] = tgt_ids
    return rec


def _tokenize_record(rec):
    """Tokenize a record in a worker process. Returns rec with cached IDs."""
    rec["_src_ids"] = _worker_sp.encode(rec["input"])
    rec["_tgt_ids"] = _worker_sp.encode(rec["target"])
    return rec


class TransmutationDataset(Dataset):
    """Loads JSONL training pairs, tokenizes once, caches token IDs to disk."""

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_src_len: int = 1152,
        max_tgt_len: int = 1536,
    ):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        data_path = Path(data_dir)
        cache_path = data_path / "tokens.pt"

        # ── Check for token cache ──
        # Cache is valid if it exists and is newer than all JSONL shards.
        shards = sorted(data_path.glob("*.jsonl"))
        cache_valid = False
        if cache_path.exists() and shards:
            cache_mtime = cache_path.stat().st_mtime
            latest_shard = max(s.stat().st_mtime for s in shards)
            cache_valid = cache_mtime > latest_shard

        if cache_valid:
            cached = torch.load(cache_path, weights_only=True)
            if "src" in cached:
                # New format: padded 2D tensors + lengths
                self._src_pad = cached["src"]        # (N, max_src) int16
                self._tgt_pad = cached["tgt"]        # (N, max_tgt) int16
                self._src_lens = cached["src_lens"]  # (N,) int32
                self._tgt_lens = cached["tgt_lens"]  # (N,) int32
                self.records = [None] * len(self._src_lens)
            else:
                # Old format: list of tensors — convert on next save
                self._src_pad = None
                old_src = cached["src_ids"]
                old_tgt = cached["tgt_ids"]
                self._src_lens = torch.tensor([len(s) for s in old_src], dtype=torch.int32)
                self._tgt_lens = torch.tensor([len(t) for t in old_tgt], dtype=torch.int32)
                max_src = self._src_lens.max().item()
                max_tgt = self._tgt_lens.max().item()
                n = len(old_src)
                self._src_pad = torch.zeros(n, max_src, dtype=torch.int16)
                self._tgt_pad = torch.zeros(n, max_tgt, dtype=torch.int16)
                for i in range(n):
                    sl, tl_ = len(old_src[i]), len(old_tgt[i])
                    self._src_pad[i, :sl] = old_src[i]
                    self._tgt_pad[i, :tl_] = old_tgt[i]
                self.records = [None] * n
                # Re-save in new format
                torch.save({"src": self._src_pad, "tgt": self._tgt_pad,
                            "src_lens": self._src_lens, "tgt_lens": self._tgt_lens}, cache_path)
            self._from_cache = True
            print(f"  Dataset: {len(self.records)} loaded from token cache", flush=True)
            return
        self._from_cache = False

        # ── No cache: load JSONL, tokenize, save cache ──
        all_records = []
        for shard in shards:
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))

        # Two-tier: char pre-filter skips tokenization for short samples.
        safe_src_chars = int(max_src_len * 2.5)
        safe_tgt_chars = int((max_tgt_len - 1) * 2.5)

        safe = []
        borderline = []
        for rec in all_records:
            if len(rec["input"]) <= safe_src_chars and len(rec["target"]) <= safe_tgt_chars:
                safe.append(rec)
            else:
                borderline.append(rec)

        # Pre-tokenize ALL samples across CPU cores.
        n_workers = min(os.cpu_count() or 4, max(len(borderline), len(safe), 1))
        kept_borderline = []
        with Pool(n_workers, initializer=_init_filter_worker,
                  initargs=(tokenizer_path, max_src_len, max_tgt_len - 1)) as pool:
            if borderline:
                for rec in pool.imap_unordered(_filter_record, borderline, chunksize=256):
                    if rec is not None:
                        kept_borderline.append(rec)
            if safe:
                safe = list(pool.imap(_tokenize_record, safe, chunksize=256))

        skipped = len(borderline) - len(kept_borderline)
        self.records = safe + kept_borderline
        for rec in self.records:
            rec.pop("input", None)
            rec.pop("target", None)
        n_total = len(all_records)

        # ── Save token cache as padded 2D tensors (fast to load) ──
        src_lens = [len(r["_src_ids"]) for r in self.records]
        tgt_lens = [len(r["_tgt_ids"]) for r in self.records]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        n = len(self.records)
        src_padded = torch.zeros(n, max_src, dtype=torch.int16)
        tgt_padded = torch.zeros(n, max_tgt, dtype=torch.int16)
        for i, r in enumerate(self.records):
            s = r["_src_ids"]
            t = r["_tgt_ids"]
            src_padded[i, :len(s)] = torch.tensor(s, dtype=torch.int16)
            tgt_padded[i, :len(t)] = torch.tensor(t, dtype=torch.int16)
        torch.save({
            "src": src_padded, "tgt": tgt_padded,
            "src_lens": torch.tensor(src_lens, dtype=torch.int32),
            "tgt_lens": torch.tensor(tgt_lens, dtype=torch.int32),
        }, cache_path)

        print(f"  Dataset: {len(self.records)} loaded, {skipped} skipped, "
              f"pre-tokenized ({n_workers} procs), cache saved",
              flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self._from_cache:
            sl = self._src_lens[idx].item()
            tl_ = self._tgt_lens[idx].item()
            src_ids = self._src_pad[idx, :sl].long()
            tgt_ids = self._tgt_pad[idx, :tl_].long()
        else:
            record = self.records[idx]
            src_ids = torch.tensor(record["_src_ids"], dtype=torch.long)
            tgt_ids = torch.tensor(record["_tgt_ids"], dtype=torch.long)

        # Add BOS/EOS to target.
        bos = torch.tensor([self.bos_id], dtype=torch.long)
        eos = torch.tensor([self.eos_id], dtype=torch.long)

        return {
            "src_ids": src_ids,
            "tgt_input": torch.cat([bos, tgt_ids]),
            "tgt_labels": torch.cat([tgt_ids, eos]),
        }


class PrebuiltDataset(Dataset):
    """Loads pre-tokenized dataset.pt and filters by curriculum stage."""

    def __init__(self, data_path, tokenizer_path):
        data = torch.load(data_path, weights_only=True)
        self._src_pad = data["src"]           # (N, max_src) int16
        self._tgt_pad = data["tgt"]           # (N, max_tgt) int16
        self._src_lens = data["src_lens"]     # (N,) int32
        self._tgt_lens = data["tgt_lens"]     # (N,) int32
        self._complexity = data["complexity"] # (N,) int8
        self._corrupt = data["corrupt"]       # (N,) bool
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
        self.pad_id = sp.pad_id()
        self._active_indices = torch.arange(len(self._src_lens))
        print(f"  PrebuiltDataset: {len(self._src_lens)} total samples loaded", flush=True)

    def apply_stage_filter(self, max_src_tokens, max_complexity, allow_corrupt):
        """Filter samples by stage criteria. Returns count of active samples."""
        mask = (self._src_lens <= max_src_tokens)
        mask &= (self._complexity <= max_complexity)
        if not allow_corrupt:
            mask &= ~self._corrupt
        self._active_indices = mask.nonzero(as_tuple=True)[0]
        return len(self._active_indices)

    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, idx):
        real_idx = self._active_indices[idx].item()
        sl = self._src_lens[real_idx].item()
        tl = self._tgt_lens[real_idx].item()
        src_ids = self._src_pad[real_idx, :sl].long()
        tgt_ids = self._tgt_pad[real_idx, :tl].long()
        bos = torch.tensor([self.bos_id], dtype=torch.long)
        eos = torch.tensor([self.eos_id], dtype=torch.long)
        return {
            "src_ids": src_ids,
            "tgt_input": torch.cat([bos, tgt_ids]),
            "tgt_labels": torch.cat([tgt_ids, eos]),
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


DIFFUSION_LENGTH_BUCKETS = [64, 128, 256, 384, 512, 768, 1024, 1536]


def _assign_bucket(length):
    """Assign a token length to the smallest bucket that fits it."""
    for i, b in enumerate(DIFFUSION_LENGTH_BUCKETS):
        if length <= b:
            return i
    return len(DIFFUSION_LENGTH_BUCKETS) - 1  # last bucket


class PrebuiltDiffusionDataset(Dataset):
    """Pre-tokenized dataset for diffusion training.

    Returns raw token IDs without BOS/EOS (diffusion operates in continuous
    embedding space, not autoregressive token space).
    """

    def __init__(self, data_path, tokenizer_path):
        data = torch.load(data_path, weights_only=True)
        self._src_pad = data["src"]           # (N, max_src) int16
        self._tgt_pad = data["tgt"]           # (N, max_tgt) int16
        self._src_lens = data["src_lens"]     # (N,) int32
        self._tgt_lens = data["tgt_lens"]     # (N,) int32
        self._complexity = data["complexity"] # (N,) int8
        self._corrupt = data["corrupt"]       # (N,) bool
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        self.pad_id = sp.pad_id()
        self._active_indices = torch.arange(len(self._src_lens))
        print(f"  PrebuiltDiffusionDataset: {len(self._src_lens)} total samples loaded", flush=True)

    def apply_stage_filter(self, max_src_tokens, max_complexity, allow_corrupt):
        """Filter samples by stage criteria. Returns count of active samples."""
        mask = (self._src_lens <= max_src_tokens)
        mask &= (self._complexity <= max_complexity)
        if not allow_corrupt:
            mask &= ~self._corrupt
        self._active_indices = mask.nonzero(as_tuple=True)[0]
        return len(self._active_indices)

    def __len__(self):
        return len(self._active_indices)

    def __getitem__(self, idx):
        real_idx = self._active_indices[idx].item()
        sl = self._src_lens[real_idx].item()
        tl = self._tgt_lens[real_idx].item()
        src_ids = self._src_pad[real_idx, :sl].long()
        tgt_ids = self._tgt_pad[real_idx, :tl].long()
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "tgt_len": tl,
        }


def diffusion_collate_fn(batch, pad_id=0):
    """Collate for diffusion: pad src to max in batch, pad tgt to bucket ceiling."""
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]
    tgt_lens = [item["tgt_len"] for item in batch]

    # Pad source to max in batch
    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    src_mask = (src_padded == pad_id)

    # Pad target to bucket ceiling (all targets in batch get same length)
    max_tgt = max(tgt_lens)
    bucket_idx = _assign_bucket(max_tgt)
    bucket_len = DIFFUSION_LENGTH_BUCKETS[bucket_idx]
    tgt_padded = torch.full((len(batch), bucket_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(tgt_ids):
        tgt_padded[i, :len(ids)] = ids
    tgt_mask = (tgt_padded == pad_id)

    # Bucket labels for length prediction loss
    bucket_labels = torch.tensor([_assign_bucket(l) for l in tgt_lens], dtype=torch.long)

    return {
        "src_ids": src_padded,
        "tgt_ids": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "bucket_labels": bucket_labels,
    }


class ResumableRandomSampler(Sampler):
    """Random sampler with a known seed that can resume from a given offset.

    If max_samples is set, each epoch yields at most that many samples from the
    shuffled order. Different seeds give different subsets — fair coverage over time.
    """

    def __init__(self, data_source, seed: int, start_index: int = 0, max_samples: int = 0):
        self.data_source = data_source
        self.seed = seed
        self.start_index = start_index
        self.max_samples = max_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(self.data_source), generator=g).tolist()
        # Cap total epoch FIRST, then skip already-trained.
        if self.max_samples > 0:
            perm = perm[:self.max_samples]
        return iter(perm[self.start_index:])

    def __len__(self):
        n = min(len(self.data_source), self.max_samples) if self.max_samples > 0 else len(self.data_source)
        return max(0, n - self.start_index)


class BucketedBatchSampler(Sampler):
    """Stratified length-bucketed batch sampler.

    Equal samples per bucket (stratified), larger batch sizes for shorter
    sequences (efficient padding), interleaved bucket order (diversity).
    """

    BINS = [0, 64, 128, 256, 384, 512, 768, 1024, 1152]

    def __init__(self, dataset, seed, base_batch_size, max_src_len,
                 start_index=0, max_samples=0):
        self.seed = seed
        src_lens = dataset._src_lens[dataset._active_indices]
        tgt_lens = dataset._tgt_lens[dataset._active_indices]
        # Bucket by max(src, tgt) — controls both padding waste and AR decode length.
        seq_lens = torch.maximum(src_lens, tgt_lens)
        n = len(seq_lens)
        g = torch.Generator()
        g.manual_seed(seed)

        # Assign samples to length buckets.
        n_bins = len(self.BINS)
        bucket_ids = torch.zeros(n, dtype=torch.long)
        for i in range(n_bins - 1):
            mask = (seq_lens >= self.BINS[i]) & (seq_lens < self.BINS[i + 1])
            bucket_ids[mask] = i
        bucket_ids[seq_lens >= self.BINS[-1]] = n_bins - 1

        # Collect per-bucket indices, compute batch sizes.
        bucket_indices = []
        bucket_batch_sizes = []
        for b in range(n_bins):
            indices = (bucket_ids == b).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            bucket_max = self.BINS[min(b + 1, n_bins - 1)]
            if bucket_max == 0:
                bucket_max = self.BINS[1]
            # Scale batch size inversely with sequence length.
            # AR decode roughly doubles memory vs forward-only, so scale conservatively.
            bs = max(base_batch_size, int(base_batch_size * max_src_len / bucket_max / 2))
            # Shuffle within bucket.
            perm = torch.randperm(len(indices), generator=g)
            bucket_indices.append(indices[perm])
            bucket_batch_sizes.append(bs)

        # Stratify: cap each bucket to equal sample count.
        n_active = len(bucket_indices)
        if max_samples > 0 and n_active > 0:
            per_bucket = max_samples // n_active
        else:
            per_bucket = max(len(b) for b in bucket_indices) if bucket_indices else 0

        # Build batches per bucket with per-bucket cap.
        self._batches = []
        for indices, bs in zip(bucket_indices, bucket_batch_sizes):
            capped = indices[:per_bucket]
            for i in range(0, len(capped), bs):
                self._batches.append(capped[i:i + bs].tolist())

        # Shuffle batch order for diversity across buckets.
        batch_perm = torch.randperm(len(self._batches), generator=g)
        self._batches = [self._batches[i] for i in batch_perm]

        # Skip batches for resume.
        if start_index > 0:
            self._batches = self._batches[start_index:]

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


def create_dataloader(
    data_dir: str,
    tokenizer_path: str,
    batch_size: int = 16,
    max_src_len: int = 1152,
    max_tgt_len: int = 1536,
    shuffle: bool = True,
    num_workers: int = 2,
    pad_id: int = 0,
    epoch_seed: int | None = None,
    start_index: int = 0,
    max_samples: int = 0,
    dataset: "TransmutationDataset | PrebuiltDataset | None" = None,
    bucketed: bool = False,
) -> "tuple[DataLoader, int, TransmutationDataset | PrebuiltDataset]":
    if dataset is None:
        dataset = TransmutationDataset(
            data_dir=data_dir,
            tokenizer_path=tokenizer_path,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )
    if epoch_seed is None:
        epoch_seed = torch.randint(0, 2**31, (1,)).item()

    if bucketed and isinstance(dataset, (PrebuiltDataset, PrebuiltDiffusionDataset)):
        batch_sampler = BucketedBatchSampler(
            dataset, seed=epoch_seed, base_batch_size=batch_size,
            max_src_len=max_src_len, start_index=start_index,
            max_samples=max_samples,
        )
        cfn = diffusion_collate_fn if isinstance(dataset, PrebuiltDiffusionDataset) else collate_fn
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=lambda batch, _cfn=cfn, _pid=pad_id: _cfn(batch, _pid),
            pin_memory=True,
        ), epoch_seed, dataset

    sampler = None
    if shuffle:
        sampler = ResumableRandomSampler(dataset, seed=epoch_seed,
                                         start_index=start_index, max_samples=max_samples)
    elif start_index > 0:
        sampler = list(range(start_index, len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True,
    ), epoch_seed, dataset
