#!/usr/bin/env python3
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

"""Autoencoder inference for ad-hoc reconstruction testing.

Loads a trained AE checkpoint, runs clean encode->decode->argmax on val samples,
and reports per-sample and summary token accuracy.
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from dataset import AutoencoderDataset, ae_collate_fn, AE_LENGTH_BUCKETS
from metrics import (GPUTokenDecoder, gpu_normalize_ws,
                     batched_triton_levenshtein, levenshtein)


def load_ae_model(checkpoint, device):
    """Load autoencoder from checkpoint. Auto-detects architecture from state_dict."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Detect architecture from weight shapes.
    d_emb = state["embedding.up.weight"].shape[0]
    emb_rank = state["embedding.up.weight"].shape[1]
    vocab_size = state["embedding.down.weight"].shape[0]

    # Conv channels from encoder conv weights (variable number of conv layers).
    conv_channels = []
    i = 0
    while f"enc_convs.{i * 3}.weight" in state:
        conv_channels.append(state[f"enc_convs.{i * 3}.weight"].shape[0])
        i += 1
    conv_channels = tuple(conv_channels)
    conv_strides = tuple(ckpt.get("conv_strides", [2] * len(conv_channels)))

    # Transformer layers (encoder and decoder may differ).
    n_enc_layers = 0
    while f"enc_transformer.{n_enc_layers}.qkv.weight" in state:
        n_enc_layers += 1
    n_dec_layers = 0
    while f"dec_transformer.{n_dec_layers}.qkv.weight" in state:
        n_dec_layers += 1

    # Heads from QKV weight shape: (3*d_tf, d_tf), head_dim from RoPE.
    d_tf = conv_channels[-1]
    n_heads = d_tf // 64  # head_dim=64 is standard

    d_ff = state["enc_transformer.0.ff.0.weight"].shape[0]

    sys.path.insert(0, str(Path(__file__).parent))
    from autoencoder import build_autoencoder

    pad_id = 0  # Will be overridden by tokenizer below.
    model = build_autoencoder(
        vocab_size=vocab_size, d_emb=d_emb, emb_rank=emb_rank,
        conv_channels=conv_channels, conv_strides=conv_strides,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        n_heads=n_heads,
        d_ff=d_ff, pad_id=pad_id,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        print(f"  Ignoring unexpected keys: {unexpected}")

    model.eval()
    format_name = ckpt.get("format", "unknown")
    best_acc = ckpt.get("best_token_acc", 0.0)
    epoch = ckpt.get("epoch", 0)
    print(f"Loaded {format_name} AE: epoch={epoch}, best_acc={best_acc:.4f}")
    return model, format_name


def _assign_bucket(length):
    """Assign a sequence length to its bucket index."""
    for i, b in enumerate(AE_LENGTH_BUCKETS):
        if length <= b:
            return i
    return len(AE_LENGTH_BUCKETS) - 1


@torch.no_grad()
def run_inference(model, dataset, sp, device, n_per_bucket=10, show_errors=True,
                  diagnose=False):
    """Run clean reconstruction on val samples, stratified by length bucket.

    Samples n_per_bucket from each length bucket for uniform coverage.
    Reports per-bucket and overall stats. With diagnose=True, reports
    per-error correct token rank and probability to attribute encoder vs decoder.
    """
    pad_id = sp.pad_id()
    CHUNK = 8
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    use_gpu_cer = device.type == "cuda"
    gpu_decoder = GPUTokenDecoder(sp, device=device) if use_gpu_cer else None

    # Diagnostic accumulators.
    diag_encoder_fault = 0  # correct token rank > 5
    diag_decoder_fault = 0  # correct token rank 2-5
    diag_total_errors = 0

    # Per-position accuracy accumulators.
    POS_BINS = [(0, 100), (100, 300), (300, 500), (500, 800), (800, 1200), (1200, 1536)]
    POS_NAMES = ["0-99", "100-299", "300-499", "500-799", "800-1199", "1200-1535"]
    pos_correct = [0] * len(POS_BINS)
    pos_total = [0] * len(POS_BINS)

    # Inter-error distance accumulators (samples with 2+ errors).
    all_error_gaps = []

    # Bucket all dataset indices by sequence length.
    bucket_indices = {i: [] for i in range(len(AE_LENGTH_BUCKETS))}
    for idx in range(len(dataset)):
        item = dataset[idx]
        b = _assign_bucket(item["length"])
        bucket_indices[b].append(idx)

    # Sample n_per_bucket from each non-empty bucket.
    g = torch.Generator()
    g.manual_seed(42)
    selected = {}
    for b, indices in bucket_indices.items():
        if not indices:
            continue
        perm = torch.randperm(len(indices), generator=g)
        k = min(n_per_bucket, len(indices))
        selected[b] = [indices[perm[i].item()] for i in range(k)]

    # Run inference per bucket.
    overall_correct = 0
    overall_tokens = 0
    overall_perfect = 0
    overall_total = 0
    overall_cer_edits = 0
    overall_cer_chars = 0

    for b in range(len(AE_LENGTH_BUCKETS)):
        if b not in selected:
            continue
        bucket_name = f"0-{AE_LENGTH_BUCKETS[0]}" if b == 0 else \
            f"{AE_LENGTH_BUCKETS[b-1]+1}-{AE_LENGTH_BUCKETS[b]}"
        indices = selected[b]
        n = len(indices)

        bucket_correct = 0
        bucket_tokens = 0
        bucket_perfect = 0
        bucket_cer_edits = 0
        bucket_cer_chars = 0
        bucket_errors = []

        for chunk_start in range(0, n, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n)
            batch_items = [dataset[indices[i]] for i in range(chunk_start, chunk_end)]
            batch = ae_collate_fn(batch_items, pad_id)

            token_ids = batch["token_ids"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            lengths = batch["lengths"]
            seq_len = token_ids.shape[1]

            if diagnose:
                # Keep embeddings for per-error logit analysis.
                latent = raw.encode(token_ids)
                emb = raw._decode_to_emb(latent, seq_len)
                # Chunked argmax (same as predict_tokens but we keep emb).
                pred_ids = torch.empty_like(token_ids)
                for s in range(0, seq_len, 256):
                    e = min(s + 256, seq_len)
                    pred_ids[:, s:e] = raw.embedding.project_to_vocab(
                        emb[:, s:e]).argmax(dim=-1)
            else:
                emb = None
                pred_ids = model.predict_tokens(token_ids)

            ref_bytes_list = []
            pred_bytes_list = []
            for i in range(len(batch_items)):
                length = lengths[i].item()
                if use_gpu_cer:
                    ref_bytes_list.append(gpu_normalize_ws(
                        gpu_decoder.decode_to_bytes(token_ids[i, :length])))
                    pred_bytes_list.append(gpu_normalize_ws(
                        gpu_decoder.decode_to_bytes(pred_ids[i, :length])))

            if use_gpu_cer:
                distances = batched_triton_levenshtein(
                    pred_bytes_list, ref_bytes_list, device)

            for i in range(len(batch_items)):
                length = lengths[i].item()
                inp = token_ids[i, :length]
                pred = pred_ids[i, :length]
                match = (inp == pred)
                correct = match.sum().item()
                sample_acc = correct / length
                bucket_correct += correct
                bucket_tokens += length

                # Accumulate per-position accuracy.
                for bi, (lo, hi) in enumerate(POS_BINS):
                    if lo >= length:
                        break
                    end = min(hi, length)
                    pos_total[bi] += end - lo
                    pos_correct[bi] += match[lo:end].sum().item()

                if use_gpu_cer:
                    sample_edits = distances[i]
                    ref_len = len(ref_bytes_list[i])
                else:
                    inp_text = sp.decode(inp.tolist())
                    pred_text = sp.decode(pred.tolist())
                    sample_edits = levenshtein(inp_text, pred_text)
                    ref_len = len(inp_text)
                sample_cer = sample_edits / max(ref_len, 1)
                bucket_cer_edits += sample_edits
                bucket_cer_chars += ref_len

                if correct == length:
                    bucket_perfect += 1
                else:
                    mismatches = (inp != pred).nonzero(as_tuple=True)[0].tolist()

                    # Collect inter-error gaps for multi-error samples.
                    if len(mismatches) >= 2:
                        for j in range(1, len(mismatches)):
                            all_error_gaps.append(mismatches[j] - mismatches[j - 1])

                    # Per-error diagnosis: rank of correct token in decoder logits.
                    error_details = []
                    if diagnose and emb is not None:
                        for pos in mismatches[:10]:
                            logits = raw.embedding.project_to_vocab(
                                emb[i:i+1, pos:pos+1, :]).squeeze()  # (vocab,)
                            probs = torch.softmax(logits.float(), dim=0)
                            correct_id = inp[pos].item()
                            predicted_id = pred[pos].item()
                            correct_prob = probs[correct_id].item()
                            predicted_prob = probs[predicted_id].item()
                            # Rank: how many tokens have higher prob than correct.
                            rank = (probs > correct_prob).sum().item() + 1
                            error_details.append({
                                "pos": pos,
                                "predicted": sp.id_to_piece(predicted_id),
                                "correct": sp.id_to_piece(correct_id),
                                "rank": rank,
                                "correct_prob": correct_prob,
                                "predicted_prob": predicted_prob,
                            })
                            diag_total_errors += 1
                            if rank <= 5:
                                diag_decoder_fault += 1
                            else:
                                diag_encoder_fault += 1

                    if show_errors and len(bucket_errors) < 3:
                        inp_text = sp.decode(inp.tolist())
                        pred_text = sp.decode(pred.tolist())
                        bucket_errors.append({
                            "length": length,
                            "acc": sample_acc,
                            "cer": sample_cer,
                            "n_wrong": length - correct,
                            "mismatch_pos": mismatches[:10],
                            "input_snippet": inp_text[:200],
                            "pred_snippet": pred_text[:200],
                            "error_details": error_details,
                        })

        bucket_acc = bucket_correct / max(bucket_tokens, 1)
        bucket_cer = bucket_cer_edits / max(bucket_cer_chars, 1)

        print(f"\n=== Bucket {bucket_name} ({n} samples, {len(bucket_indices[b])} in dataset) ===")
        print(f"  Acc: {bucket_acc:.4f} ({bucket_acc*100:.2f}%)  "
              f"CER: {bucket_cer:.4f} ({bucket_cer*100:.2f}%)  "
              f"Perfect: {bucket_perfect}/{n} ({100*bucket_perfect/max(n,1):.0f}%)")

        for e in bucket_errors:
            print(f"\n  [{e['length']} tok, acc={e['acc']:.4f}, CER={e['cer']:.2%}, "
                  f"{e['n_wrong']} wrong @ {e['mismatch_pos']}]")
            print(f"    In:  {e['input_snippet']}")
            print(f"    Out: {e['pred_snippet']}")
            for ed in e.get("error_details", []):
                blame = "DECODER" if ed["rank"] <= 5 else "ENCODER"
                print(f"    pos={ed['pos']}: \"{ed['correct']}\" → \"{ed['predicted']}\" "
                      f"(rank={ed['rank']}, p_correct={ed['correct_prob']:.4f}, "
                      f"p_predicted={ed['predicted_prob']:.4f}) [{blame}]")

        overall_correct += bucket_correct
        overall_tokens += bucket_tokens
        overall_perfect += bucket_perfect
        overall_total += n
        overall_cer_edits += bucket_cer_edits
        overall_cer_chars += bucket_cer_chars

    avg_acc = overall_correct / max(overall_tokens, 1)
    avg_cer = overall_cer_edits / max(overall_cer_chars, 1)
    print(f"\n=== Overall ({overall_total} samples) ===")
    print(f"  Avg token accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"  CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"  Perfect: {overall_perfect}/{overall_total} "
          f"({100*overall_perfect/max(overall_total,1):.1f}%)")

    # Per-position accuracy breakdown.
    pos_parts = []
    for bi, name in enumerate(POS_NAMES):
        if pos_total[bi] > 0:
            pos_parts.append(f"{name}={100*pos_correct[bi]/pos_total[bi]:.3f}%")
    if pos_parts:
        print(f"  Token acc by position: {' | '.join(pos_parts)}")

    # Inter-error distance statistics.
    if all_error_gaps:
        gaps = sorted(all_error_gaps)
        n = len(gaps)
        mean_gap = sum(gaps) / n
        median_gap = gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2
        print(f"  Error gaps ({n} gaps from multi-error samples): "
              f"min={gaps[0]} max={gaps[-1]} mean={mean_gap:.0f} median={median_gap:.0f}")

    if diagnose and diag_total_errors > 0:
        print(f"\n=== Error Attribution ({diag_total_errors} errors) ===")
        print(f"  DECODER fault (correct in top-5): {diag_decoder_fault} "
              f"({100*diag_decoder_fault/diag_total_errors:.0f}%)")
        print(f"  ENCODER fault (correct rank >5):  {diag_encoder_fault} "
              f"({100*diag_encoder_fault/diag_total_errors:.0f}%)")
        if diag_decoder_fault > diag_encoder_fault:
            print(f"  → Decoder is the bottleneck. Encoder preserves info but decoder picks wrong token.")
        else:
            print(f"  → Encoder is the bottleneck. Information lost in latent compression.")

    return avg_acc, avg_cer, overall_perfect, overall_total


def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder reconstruction testing")
    parser.add_argument("checkpoint", help="Path to AE checkpoint")
    parser.add_argument("-n", type=int, default=10,
                        help="Samples per length bucket")
    parser.add_argument("--format", default=None,
                        help="Override format (json/xml). Auto-detected from checkpoint.")
    parser.add_argument("--data-dir", default="data/run7",
                        help="Data directory containing val/dataset.pt")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path (default: same dir as checkpoint)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--clean", action="store_true",
                        help="Filter out corrupt/augmented samples")
    parser.add_argument("--diagnose", action="store_true",
                        help="Attribute errors to encoder vs decoder")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model.
    model, detected_format = load_ae_model(args.checkpoint, device)
    format_name = args.format or detected_format

    # Tokenizer.
    tok_path = args.tokenizer
    if tok_path is None:
        tok_path = str(Path(args.checkpoint).parent.parent / "tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)

    # Load val dataset.
    val_dataset = AutoencoderDataset(
        Path(args.data_dir) / "val" / "dataset.pt",
        tok_path, format=format_name)
    n_before = len(val_dataset)
    val_dataset.apply_stage_filter(args.max_seq_len, allow_corrupt=not args.clean)
    print(f"Val samples: {len(val_dataset)}/{n_before} ({format_name}"
          f"{', clean only' if args.clean else ''})")

    run_inference(model, val_dataset, sp, device, n_per_bucket=args.n,
                  diagnose=args.diagnose)


if __name__ == "__main__":
    main()
