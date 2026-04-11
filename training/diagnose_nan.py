# Copyright (C) 2026 Nicholas Perez
#
# Diagnostic: trace where extreme values / NaN originate in AR training.
# Run inside Docker: run_gpu training/diagnose_nan.py --tokenizer ... --data-dir ...

import argparse
import os
import torch
import torch.nn as nn
import sentencepiece as spm
from torch.amp import autocast
from dataset import PrebuiltDataset, create_dataloader
from model import build_model


def check_tensor(name, t):
    """Print stats for a tensor, flag non-finite values."""
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    t_float = t.float()
    mx = t_float.max().item()
    mn = t_float.min().item()
    mean = t_float.mean().item()
    std = t_float.std().item()
    n_nan = t_float.isnan().sum().item()
    n_inf = t_float.isinf().sum().item()
    flag = ""
    if n_nan > 0:
        flag += f" *** {n_nan} NaN ***"
    if n_inf > 0:
        flag += f" *** {n_inf} inf ***"
    if abs(mx) > 1000 or abs(mn) > 1000:
        flag += " *** EXTREME ***"
    print(f"  {name:40s} min={mn:12.4f} max={mx:12.4f} mean={mean:10.4f} std={std:10.4f}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()

    model = build_model(
        vocab_size=vocab_size, d_model=384,
        n_encoder_layers=6, n_decoder_layers=6,
        d_state=64, expand=2, headdim=64, n_heads=6,
        dropout=0.1, pad_id=pad_id,
    ).to(device)
    model.eval()

    # Load stage 1 data.
    dataset = PrebuiltDataset(
        os.path.join(args.data_dir, "train", "dataset.pt"), args.tokenizer)
    dataset.apply_stage_filter(max_src_tokens=128, max_complexity=3, allow_corrupt=False)

    loader, _, _ = create_dataloader(
        data_dir="unused", shuffle=True, epoch_seed=42,
        dataset=dataset, tokenizer_path=args.tokenizer,
        batch_size=args.batch_size, max_src_len=1152, max_tgt_len=1536,
        num_workers=0, pad_id=pad_id,
    )

    print(f"=== Diagnostic: {args.n_batches} batches, batch_size={args.batch_size} ===")
    print(f"Device: {device}, dtype: fp16")
    print(f"Logit soft cap: {model.logit_soft_cap}")
    print()

    n_nan = 0
    n_inf = 0
    all_tf_losses = []
    all_ar_losses = []

    for step, batch in enumerate(loader):
        if step >= args.n_batches:
            break

        src = batch["src_ids"].to(device)
        tgt_in = batch["tgt_input"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)

        print(f"--- Batch {step} (src shape {src.shape}, tgt shape {tgt_in.shape}) ---")

        # 1. Teacher-forced forward pass.
        print("  [Teacher-forced]")
        with torch.no_grad():
            with autocast("cuda", enabled=True):
                tf_logits = model(src, tgt_in, src_mask)
        check_tensor("TF logits", tf_logits)
        n_nan += tf_logits.isnan().sum().item()
        n_inf += tf_logits.isinf().sum().item()
        tf_ce = nn.functional.cross_entropy(
            tf_logits.reshape(-1, vocab_size), tgt_labels.reshape(-1), ignore_index=-100)
        tf_loss_val = tf_ce.item()
        all_tf_losses.append(tf_loss_val)
        if tf_loss_val != tf_loss_val:
            n_nan += 1
        print(f"  TF loss: {tf_loss_val:.4f}")

        # 2. AR decode (same path as training loop).
        print("  [AR decode (ar_decode_parallel)]")
        from ar_parallel import ar_decode_parallel
        ar_tgt_in = tgt_in.clone()
        with torch.no_grad():
            ar_ids = ar_decode_parallel(
                model, src, sp,
                max_len=tgt_in.shape[1],
                src_key_padding_mask=src_mask,
            )
        for i, ids in enumerate(ar_ids):
            ar_len = min(len(ids), tgt_in.shape[1] - 1)
            ar_tgt_in[i, 1:1 + ar_len] = torch.tensor(ids[:ar_len], device=device)

        check_tensor("AR decoded IDs", ar_tgt_in)

        # How different is AR input from ground truth?
        valid = tgt_labels != -100
        S = min(ar_tgt_in.shape[1], tgt_in.shape[1]) - 1
        ar_content = ar_tgt_in[:, 1:1+S]
        gt_content = tgt_in[:, 1:1+S]
        valid = valid[:, :S]
        match = (ar_content == gt_content) & valid
        n_valid = valid.sum().item()
        n_match = match.sum().item()
        print(f"  AR vs GT token match: {n_match}/{n_valid} ({100*n_match/max(n_valid,1):.1f}%)")

        # 3. Forward pass with AR input.
        print("  [AR forward pass]")
        with torch.no_grad():
            with autocast("cuda", enabled=True):
                ar_logits = model(src, ar_tgt_in, src_mask)
        check_tensor("AR logits", ar_logits)
        n_nan += ar_logits.isnan().sum().item()
        n_inf += ar_logits.isinf().sum().item()
        ar_ce = nn.functional.cross_entropy(
            ar_logits.reshape(-1, vocab_size), tgt_labels.reshape(-1), ignore_index=-100)
        ar_loss_val = ar_ce.item()
        all_ar_losses.append(ar_loss_val)
        if ar_loss_val != ar_loss_val:
            n_nan += 1
        print(f"  AR loss: {ar_loss_val:.4f}")

        # 4. Check encoder output.
        print("  [Encoder output]")
        with torch.no_grad():
            with autocast("cuda", enabled=True):
                enc_out = model.encode(src)
        check_tensor("Encoder output", enc_out)

        print()


    # Summary.
    print("=" * 60)
    print(f"SUMMARY: {len(all_tf_losses)} batches, {len(all_tf_losses) * args.batch_size} samples")
    print(f"  NaN count:    {n_nan}")
    print(f"  Inf count:    {n_inf}")
    print(f"  TF loss:      min={min(all_tf_losses):.4f} max={max(all_tf_losses):.4f} mean={sum(all_tf_losses)/len(all_tf_losses):.4f}")
    print(f"  AR loss:      min={min(all_ar_losses):.4f} max={max(all_ar_losses):.4f} mean={sum(all_ar_losses)/len(all_ar_losses):.4f}")
    if n_nan == 0 and n_inf == 0:
        print("  VERDICT: CLEAN — no NaN or inf detected")
    else:
        print("  VERDICT: FAILED — non-finite values detected")


if __name__ == "__main__":
    main()
