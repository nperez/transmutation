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

"""Shared metrics: GPU token decoder, Triton Levenshtein, CER/WER utilities.

Extracted from train.py for reuse across training pipelines.
"""

import re
import xml.etree.ElementTree as ET

import torch
import triton
import triton.language as tl


# ── GPU Token Decoder ──────────────────────────────────────────────────────────

class GPUTokenDecoder:
    """Decode token IDs → strings using GPU tensor ops instead of sp.decode().

    Pre-builds a flat byte buffer of all vocab pieces on GPU. At decode time:
    1. Gather piece lengths/offsets for token IDs (parallel lookup)
    2. Prefix-sum for output positions (parallel scan)
    3. Scatter piece bytes into output buffer (parallel gather)
    4. Transfer to CPU and convert to Python string

    No Python loops in the hot path — all tensor ops.
    """

    def __init__(self, sp, device='cuda'):
        vocab_size = sp.get_piece_size()
        all_bytes = bytearray()
        offsets = []
        lengths = []

        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            decoded = piece.replace('\u2581', ' ').encode('utf-8')
            offsets.append(len(all_bytes))
            lengths.append(len(decoded))
            all_bytes.extend(decoded)

        self.piece_bytes = torch.frombuffer(bytearray(all_bytes), dtype=torch.uint8).to(device)
        self.piece_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.piece_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.max_piece_len = max(lengths) if lengths else 0
        self.max_piece_block = triton.next_power_of_2(max(self.max_piece_len, 1))
        self.device = device

    def decode(self, token_ids):
        """Decode 1D token ID tensor/list → Python string. GPU scatter + CPU convert."""
        result = self._scatter_bytes(token_ids)
        if len(result) == 0:
            return ""
        return result.cpu().numpy().tobytes().decode('utf-8', errors='replace')

    def _scatter_bytes(self, token_ids):
        """Shared scatter logic: token IDs → GPU uint8 byte tensor."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        else:
            token_ids = token_ids.to(self.device)

        if len(token_ids) == 0:
            return torch.tensor([], dtype=torch.uint8, device=self.device)

        lengths = self.piece_lengths[token_ids]
        offsets = self.piece_offsets[token_ids]

        mask = lengths > 0
        if not mask.any():
            return torch.tensor([], dtype=torch.uint8, device=self.device)
        lengths = lengths[mask]
        offsets = offsets[mask]

        total_len = lengths.sum().item()
        if total_len == 0:
            return torch.tensor([], dtype=torch.uint8, device=self.device)

        cum = torch.cumsum(lengths, dim=0)
        seg = torch.zeros(total_len, dtype=torch.long, device=self.device)
        if len(cum) > 1:
            seg[cum[:-1]] = 1
        seg = seg.cumsum(0)

        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), cum[:-1]])
        local = torch.arange(total_len, device=self.device) - starts[seg]
        src = offsets[seg] + local
        result = self.piece_bytes[src]

        # Strip leading space
        if len(result) > 0 and result[0].item() == 32:
            result = result[1:]
        return result

    def decode_to_bytes(self, token_ids):
        """Decode token IDs → GPU uint8 byte tensor (no CPU transfer)."""
        return self._scatter_bytes(token_ids)

    def byte_length(self, token_ids):
        """Fast byte length of decoded text. Gather + sum, no scatter."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        else:
            token_ids = token_ids.to(self.device)
        if len(token_ids) == 0:
            return 0
        total = self.piece_lengths[token_ids].sum().item()
        if total > 0 and self.piece_lengths[token_ids[0]].item() > 0:
            first_byte = self.piece_bytes[self.piece_offsets[token_ids[0]]].item()
            if first_byte == 32:
                total -= 1
        return total


def gpu_normalize_ws(t):
    """Collapse whitespace runs to single space, strip edges. GPU uint8 tensor → GPU uint8 tensor."""
    if len(t) == 0:
        return t
    ws = (t == 32) | (t == 9) | (t == 10) | (t == 13)
    prev_ws = torch.cat([torch.tensor([False], device=t.device), ws[:-1]])
    keep = ~ws | (ws & ~prev_ws)
    out = t[keep].clone()
    out_ws = (out == 9) | (out == 10) | (out == 13)
    out[out_ws] = 32
    if len(out) > 0 and out[0] == 32:
        out = out[1:]
    if len(out) > 0 and out[-1] == 32:
        out = out[:-1]
    return out


# ── Triton Levenshtein ─────────────────────────────────────────────────────────

@triton.jit
def _lev_kernel(
    a_ptr, b_ptr, na_ptr, nb_ptr, out_ptr, work_ptr,
    stride_seq, stride_work,
    MAX_LEN: tl.constexpr,
):
    """Batched Levenshtein kernel. One program per pair."""
    pid = tl.program_id(0)
    na = tl.load(na_ptr + pid)
    nb = tl.load(nb_ptr + pid)

    ROW: tl.constexpr = MAX_LEN + 1
    a_base = a_ptr + pid * stride_seq
    b_base = b_ptr + pid * stride_seq
    w_base = work_ptr + pid * stride_work
    row0 = w_base
    row1 = w_base + ROW

    for j in tl.range(0, ROW):
        tl.store(row0 + j, j)

    use_row0_as_prev = True
    for i in tl.range(1, ROW):
        if i <= na:
            ai = tl.load(a_base + i - 1)
            if use_row0_as_prev:
                prev = row0
                curr = row1
            else:
                prev = row1
                curr = row0
            left = i
            tl.store(curr, i)

            for j in tl.range(1, ROW):
                if j <= nb:
                    bj = tl.load(b_base + j - 1)
                    cost = tl.where(ai == bj, 0, 1)

                    d = tl.load(prev + j - 1) + cost
                    u = tl.load(prev + j) + 1
                    l = left + 1

                    val = tl.minimum(d, tl.minimum(u, l))
                    tl.store(curr + j, val)
                    left = val

            use_row0_as_prev = not use_row0_as_prev

    if use_row0_as_prev:
        tl.store(out_ptr + pid, tl.load(row0 + nb))
    else:
        tl.store(out_ptr + pid, tl.load(row1 + nb))


def _next_pow2(x):
    p = 16
    while p < x:
        p *= 2
    return min(p, 8192)


def batched_triton_levenshtein(a_list, b_list, device):
    """Batch Levenshtein for multiple pairs. a_list/b_list: lists of GPU uint8 tensors.
    Returns list of int distances. Groups by sequence length for optimal kernel dispatch."""
    B = len(a_list)
    if B == 0:
        return []

    results = [0] * B

    groups = {}
    for idx in range(B):
        na, nb = len(a_list[idx]), len(b_list[idx])
        if na == 0 or nb == 0:
            results[idx] = max(na, nb)
            continue
        if max(na, nb) > 8192:
            results[idx] = max(na, nb)
            continue
        block = _next_pow2(max(na, nb))
        groups.setdefault(block, []).append(idx)

    for block, indices in groups.items():
        k = len(indices)
        a_pad = torch.zeros(k, block, dtype=torch.uint8, device=device)
        b_pad = torch.zeros(k, block, dtype=torch.uint8, device=device)
        na_t = torch.zeros(k, dtype=torch.int32, device=device)
        nb_t = torch.zeros(k, dtype=torch.int32, device=device)

        for li, gi in enumerate(indices):
            a, b = a_list[gi], b_list[gi]
            a_pad[li, :len(a)] = a
            b_pad[li, :len(b)] = b
            na_t[li] = len(a)
            nb_t[li] = len(b)

        out_t = torch.zeros(k, dtype=torch.int32, device=device)
        row = block + 1
        work_t = torch.zeros(k, 2 * row, dtype=torch.int32, device=device)

        _lev_kernel[(k,)](a_pad, b_pad, na_t, nb_t, out_t, work_t,
                          block, 2 * row, MAX_LEN=block)

        out_list = out_t.tolist()
        for li, gi in enumerate(indices):
            results[gi] = out_list[li]

    return results


# ── CPU fallback ───────────────────────────────────────────────────────────────

def levenshtein(a, b):
    """Compute Levenshtein edit distance between two sequences (CPU)."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


# ── XML / WER utilities ───────────────────────────────────────────────────────

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


def char_weighted_wer(pred_words, ref_words):
    """Word-level Levenshtein where each edit is weighted by character length."""
    n, m = len(ref_words), len(pred_words)
    if n == 0:
        return sum(len(w) for w in pred_words)
    if m == 0:
        return sum(len(w) for w in ref_words)
    prev = [0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + len(pred_words[j - 1])
    for i in range(1, n + 1):
        curr = [prev[0] + len(ref_words[i - 1])]
        for j in range(1, m + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                curr.append(prev[j - 1])
            else:
                sub = prev[j - 1] + max(len(ref_words[i - 1]), len(pred_words[j - 1]))
                delete = prev[j] + len(ref_words[i - 1])
                insert = curr[j - 1] + len(pred_words[j - 1])
                curr.append(min(sub, delete, insert))
        prev = curr
    return prev[m]
