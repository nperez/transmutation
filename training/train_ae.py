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

"""Autoencoder training for latent translation (Run 8).

Trains a denoising autoencoder on individual JSON or XML token sequences.
Corrupt 20% of tokens -> reconstruct clean tokens via CE loss.
Self-contained training loop (same infra patterns as train.py, no imports).
"""

import atexit
import argparse
from datetime import datetime
import json
import os
import signal
import time

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from autoencoder import build_autoencoder
from dataset import (AutoencoderDataset, ae_collate_fn, AEBucketedBatchSampler,
                     AE_LENGTH_BUCKETS)
from metrics import GPUTokenDecoder, gpu_normalize_ws, batched_triton_levenshtein


def _unwrap(model):
    """Get the underlying module from a torch.compiled model."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _ce_loss(logits, tgt_ids, pad_id, grad_accum, weight=None):
    """CE loss wrapped for gradient checkpointing (frees logits after forward)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tgt_ids.reshape(-1),
        ignore_index=pad_id,
        weight=weight,
    ) / grad_accum


def interrupt_filename(format_name):
    """Format-prefixed timestamped interrupt checkpoint filename."""
    return f"{format_name}_interrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"


def save_checkpoint(model, optimizer, scaler, epoch, global_step, best_token_acc,
                    output_dir, filename, format_name, epoch_complete=True,
                    epoch_step=0, epoch_seed=None, wallclock=0.0,
                    lr_patience_counter=0, conv_strides=None, val_state=None,
                    training_done=False, train_loss=0.0,
                    best_cer=float("inf"),
                    current_noise_frac=None, noise_dropped=False):
    path = os.path.join(output_dir, filename)
    ckpt = {
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "training_done": training_done,
        "epoch_step": epoch_step,
        "epoch_seed": epoch_seed,
        "global_step": global_step,
        "best_token_acc": best_token_acc,
        "best_cer": best_cer,
        "train_loss": train_loss,
        "format": format_name,
        "wallclock": wallclock,
        "lr_patience_counter": lr_patience_counter,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if conv_strides is not None:
        ckpt["conv_strides"] = list(conv_strides)
    if val_state is not None:
        ckpt["val_state"] = val_state
    if current_noise_frac is not None:
        ckpt["current_noise_frac"] = current_noise_frac
        ckpt["noise_dropped"] = noise_dropped
    torch.save(ckpt, path)


# ── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataset, pad_id, gpu_decoder, device, n_samples=1000,
             epoch=0, fp16=True, atexit_state=None, sig_state=None):
    """Clean input -> encode -> chunked decode -> token accuracy + GPU CER.

    Evaluates a fixed-size deterministic subset (seeded by epoch).
    Uses chunked vocab projection to avoid materializing full (B, L, 16k) logits.
    CER via GPUTokenDecoder + batched Triton Levenshtein (same pipeline as train.py).
    Resumable via atexit_state['val_state'].
    """
    model.eval()
    raw = _unwrap(model)
    BATCH = 8
    VOCAB_CHUNK = 256

    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    total_perfect = 0
    total_cer_edits = 0
    total_cer_chars = 0

    # Per-bucket accumulators.
    n_buckets = len(AE_LENGTH_BUCKETS)
    bk_correct = [0] * n_buckets
    bk_tokens = [0] * n_buckets
    bk_perfect = [0] * n_buckets
    bk_cer_edits = [0] * n_buckets
    bk_cer_chars = [0] * n_buckets
    bk_count = [0] * n_buckets

    def _bucket_idx(length):
        for i, b in enumerate(AE_LENGTH_BUCKETS):
            if length <= b:
                return i
        return n_buckets - 1

    n_avail = len(dataset)
    n = min(n_samples, n_avail)
    g = torch.Generator()
    g.manual_seed(epoch * 7777 + 42)
    indices = torch.randperm(n_avail, generator=g)[:n].tolist()

    # Resume from saved val state if present and matches this epoch/n.
    start_idx = 0
    if atexit_state and atexit_state.get("val_state"):
        vs = atexit_state["val_state"]
        if vs.get("epoch") == epoch and vs.get("n_samples") == n:
            start_idx = vs.get("completed", 0)
            total_correct = vs.get("correct", 0)
            total_tokens = vs.get("tokens", 0)
            total_loss = vs.get("loss", 0.0)
            total_perfect = vs.get("perfect", 0)
            total_cer_edits = vs.get("cer_edits", 0)
            total_cer_chars = vs.get("cer_chars", 0)
            bk_correct = list(vs.get("bk_correct", bk_correct))
            bk_tokens = list(vs.get("bk_tokens", bk_tokens))
            bk_perfect = list(vs.get("bk_perfect", bk_perfect))
            bk_cer_edits = list(vs.get("bk_cer_edits", bk_cer_edits))
            bk_cer_chars = list(vs.get("bk_cer_chars", bk_cer_chars))
            bk_count = list(vs.get("bk_count", bk_count))
            if start_idx > 0:
                print(f"  Resuming validation from sample {start_idx}/{n}", flush=True)

    print(f"  Validation: {n}/{n_avail} samples (epoch seed={epoch})", flush=True)
    for chunk_start in range(start_idx, n, BATCH):
        chunk_end = min(chunk_start + BATCH, n)
        batch_items = [dataset[i] for i in indices[chunk_start:chunk_end]]
        batch = ae_collate_fn(batch_items, pad_id)

        token_ids = batch["token_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        lengths = batch["lengths"]
        non_pad = ~pad_mask
        n_non_pad = non_pad.sum()
        seq_len = token_ids.shape[1]
        B = token_ids.shape[0]

        with autocast("cuda", enabled=fp16):
            latent = raw.encode(token_ids)
            emb = raw._decode_to_emb(latent, seq_len)

            # Chunked vocab projection: accuracy + loss without full logits.
            pred_ids = torch.empty_like(token_ids)
            chunk_correct = torch.zeros(1, dtype=torch.long, device=device)
            chunk_loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
            for s in range(0, seq_len, VOCAB_CHUNK):
                e = min(s + VOCAB_CHUNK, seq_len)
                chunk_logits = raw.embedding.project_to_vocab(emb[:, s:e])
                chunk_tgt = token_ids[:, s:e]
                chunk_mask = non_pad[:, s:e]
                chunk_pred = chunk_logits.argmax(dim=-1)
                pred_ids[:, s:e] = chunk_pred
                n_chunk = chunk_mask.sum()
                if n_chunk > 0:
                    chunk_loss_sum += F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                        chunk_tgt.reshape(-1),
                        ignore_index=pad_id,
                        reduction="sum",
                    ).double()
                chunk_correct += ((chunk_pred == chunk_tgt) & chunk_mask).sum()

        total_correct += chunk_correct.item()
        total_tokens += n_non_pad.item()
        total_loss += chunk_loss_sum.item()

        # Per-sample exact match: all non-pad tokens correct.
        sample_correct = ((pred_ids == token_ids) | pad_mask).all(dim=1)
        total_perfect += sample_correct.sum().item()

        # Batched GPU CER: decode to bytes on GPU, Triton Levenshtein.
        ref_bytes = []
        pred_bytes = []
        for i in range(B):
            length = lengths[i].item()
            ref_bytes.append(gpu_normalize_ws(
                gpu_decoder.decode_to_bytes(token_ids[i, :length])))
            pred_bytes.append(gpu_normalize_ws(
                gpu_decoder.decode_to_bytes(pred_ids[i, :length])))
            total_cer_chars += len(ref_bytes[-1])
        distances = batched_triton_levenshtein(pred_bytes, ref_bytes, device)
        total_cer_edits += sum(distances)

        # Per-bucket stats.
        sample_correct_cpu = sample_correct.cpu()
        for i in range(B):
            length = lengths[i].item()
            bi = _bucket_idx(length)
            bk_count[bi] += 1
            non_pad_i = (~pad_mask[i]).sum().item()
            correct_i = ((pred_ids[i] == token_ids[i]) & ~pad_mask[i]).sum().item()
            bk_correct[bi] += correct_i
            bk_tokens[bi] += non_pad_i
            bk_cer_edits[bi] += distances[i]
            bk_cer_chars[bi] += len(ref_bytes[i])
            if sample_correct_cpu[i]:
                bk_perfect[bi] += 1

        # Persist progress for resumability.
        if atexit_state is not None:
            atexit_state["val_state"] = {
                "epoch": epoch, "n_samples": n, "completed": chunk_end,
                "correct": total_correct, "tokens": total_tokens,
                "loss": total_loss, "perfect": total_perfect,
                "cer_edits": total_cer_edits, "cer_chars": total_cer_chars,
                "bk_correct": list(bk_correct), "bk_tokens": list(bk_tokens),
                "bk_perfect": list(bk_perfect),
                "bk_cer_edits": list(bk_cer_edits),
                "bk_cer_chars": list(bk_cer_chars),
                "bk_count": list(bk_count),
            }

        if sig_state and sig_state.get("stop"):
            print(f"  Validation interrupted at {chunk_end}/{n}", flush=True)
            model.train()
            return None  # signal to caller that val was interrupted

    token_acc = total_correct / max(total_tokens, 1)
    avg_loss = total_loss / max(total_tokens, 1)
    cer = total_cer_edits / max(total_cer_chars, 1)
    perfect = total_perfect

    # Print per-bucket breakdown.
    print(f"  Per-bucket:", flush=True)
    for bi in range(n_buckets):
        if bk_count[bi] == 0:
            continue
        bname = f"{AE_LENGTH_BUCKETS[bi]:>4}"
        bacc = bk_correct[bi] / max(bk_tokens[bi], 1)
        bcer = bk_cer_edits[bi] / max(bk_cer_chars[bi], 1)
        bpf = bk_perfect[bi] / max(bk_count[bi], 1)
        print(f"    ≤{bname}: acc={bacc:.4f} CER={bcer:.2%} "
              f"pf={bk_perfect[bi]}/{bk_count[bi]}({bpf:.0%}) n={bk_count[bi]}",
              flush=True)

    model.train()
    return token_acc, avg_loss, cer, perfect, n


# ── Training ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()
    print(f"Vocab size: {vocab_size}, format: {args.format}")

    gpu_decoder = GPUTokenDecoder(sp, device=device)
    print(f"GPU token decoder: {len(gpu_decoder.piece_bytes)} bytes, {vocab_size} pieces")

    conv_channels = tuple(int(x) for x in args.conv_channels.split(","))
    conv_strides = tuple(int(x) for x in args.conv_strides.split(","))

    # Auto-detect architecture from checkpoint if resuming.
    n_enc_layers = args.n_enc_layers
    n_dec_layers = args.n_dec_layers
    if args.resume:
        ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ckpt_peek["model_state_dict"]
        # Detect transformer layers.
        n_enc = 0
        while f"enc_transformer.{n_enc}.qkv.weight" in sd:
            n_enc += 1
        n_dec = 0
        while f"dec_transformer.{n_dec}.qkv.weight" in sd:
            n_dec += 1
        if n_enc != n_enc_layers or n_dec != n_dec_layers:
            print(f"  Checkpoint layers: enc={n_enc}, dec={n_dec} "
                  f"(overriding CLI: enc={n_enc_layers}, dec={n_dec_layers})")
            n_enc_layers = n_enc
            n_dec_layers = n_dec
        # Detect conv config.
        ckpt_channels = []
        i = 0
        while f"enc_convs.{i * 3}.weight" in sd:
            ckpt_channels.append(sd[f"enc_convs.{i * 3}.weight"].shape[0])
            i += 1
        ckpt_channels = tuple(ckpt_channels)
        ckpt_strides = tuple(ckpt_peek.get("conv_strides", [2] * len(ckpt_channels)))
        if ckpt_channels != conv_channels or ckpt_strides != conv_strides:
            print(f"  Checkpoint convs: channels={ckpt_channels}, strides={ckpt_strides} "
                  f"(overriding CLI)")
            conv_channels = ckpt_channels
            conv_strides = ckpt_strides
        del ckpt_peek, sd

    model = build_autoencoder(
        vocab_size=vocab_size,
        d_emb=args.d_emb,
        emb_rank=args.emb_rank,
        conv_channels=conv_channels,
        conv_strides=conv_strides,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        pad_id=pad_id,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # Load datasets.
    train_dataset = AutoencoderDataset(
        os.path.join(args.data_dir, "train", "dataset.pt"),
        args.tokenizer, format=args.format)
    val_dataset = AutoencoderDataset(
        os.path.join(args.data_dir, "val", "dataset.pt"),
        args.tokenizer, format=args.format)

    n_train_filtered = train_dataset.apply_stage_filter(args.max_seq_len, allow_corrupt=not args.clean)
    n_val_filtered = val_dataset.apply_stage_filter(args.max_seq_len, allow_corrupt=not args.clean)
    print(f"  Train: {n_train_filtered} samples (max_seq_len={args.max_seq_len})")
    print(f"  Val: {n_val_filtered} samples")

    # Inverse-sqrt-frequency loss weighting to counteract rare token collapse.
    ce_weight = None
    if args.freq_weight:
        print("  Computing token frequency weights...", flush=True)
        src = train_dataset._pad[train_dataset._active_indices]
        freq = torch.bincount(src.long().flatten(), minlength=vocab_size).double()
        freq[pad_id] = 0  # pad handled by ignore_index
        freq = freq.clamp(min=1)  # avoid div by zero for unseen tokens
        ce_weight = (1.0 / freq.sqrt()).float()
        # Normalize so mean weight ≈ 1 (preserves loss scale).
        ce_weight = ce_weight * (vocab_size / ce_weight.sum())
        ce_weight[pad_id] = 0
        ce_weight = ce_weight.to(device)
        top5 = freq.argsort(descending=True)[:5]
        bot5 = freq[freq > 1].argsort()[:5]  # rarest non-zero
        print(f"  Freq weight range: {ce_weight.min().item():.4f} - {ce_weight.max().item():.4f}")
        print(f"  Most common: {[(sp.id_to_piece(t.item()), f'w={ce_weight[t].item():.3f}') for t in top5]}")
        print(f"  Rarest: {[(sp.id_to_piece(t.item()), f'w={ce_weight[t].item():.3f}') for t in bot5]}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    warmup_steps = args.warmup_steps
    scaler = GradScaler("cuda", enabled=args.fp16)

    # Training state.
    os.makedirs(args.output_dir, exist_ok=True)
    best_token_acc = 0.0
    best_cer = float("inf")
    lr_patience_counter = 0
    current_noise_frac = args.noise_frac
    noise_dropped = False  # single-shot guard for --noise-schedule
    log_entries = []
    global_step = 0
    resume_global_step = 0
    start_epoch = 1
    resume_epoch_seed = None
    resume_epoch_step = 0
    wallclock = 0.0
    resume_val_state = None
    resume_training_done = False
    resume_train_loss = 0.0

    # Resume from checkpoint.
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if args.override_lr is not None:
            print("  LR override — dropping optimizer state (fresh Adam + warmup)")
        elif "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("  No optimizer state in checkpoint — starting fresh Adam")
        if "scaler_state_dict" in ckpt and args.override_lr is None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        completed = ckpt.get("epoch_complete", False)
        start_epoch = ckpt["epoch"] + 1 if completed else ckpt["epoch"]
        global_step = ckpt.get("global_step", 0)
        resume_global_step = global_step
        best_token_acc = ckpt.get("best_token_acc", 0.0)
        best_cer = ckpt.get("best_cer", float("inf"))
        lr_patience_counter = ckpt.get("lr_patience_counter", 0)
        current_noise_frac = ckpt.get("current_noise_frac", args.noise_frac)
        noise_dropped = ckpt.get("noise_dropped", False)
        wallclock = ckpt.get("wallclock", 0.0)
        resume_training_done = ckpt.get("training_done", False)
        resume_train_loss = ckpt.get("train_loss", 0.0)
        if "val_state" in ckpt:
            resume_val_state = ckpt["val_state"]
        if not completed:
            resume_epoch_seed = ckpt.get("epoch_seed")
            resume_epoch_step = ckpt.get("epoch_step", 0)
        log_path = os.path.join(args.output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = json.load(f)
        resumed_lr = optimizer.param_groups[0]["lr"]
        if args.override_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = args.override_lr
            print(f"  epoch={start_epoch}, step={resume_epoch_step}, "
                  f"global_step={global_step}, best_acc={best_token_acc:.4f}, "
                  f"lr={resumed_lr:.2e}→{args.override_lr:.2e} (override), "
                  f"lr_patience={lr_patience_counter}")
        else:
            print(f"  epoch={start_epoch}, step={resume_epoch_step}, "
                  f"global_step={global_step}, best_acc={best_token_acc:.4f}, "
                  f"lr={resumed_lr:.2e}, lr_patience={lr_patience_counter}")

    # Freeze encoder or decoder for targeted training.
    if args.freeze_decoder:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("dec_transformer", "dec_convs", "dec_norm")):
                param.requires_grad = False
                frozen += 1
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Decoder frozen: {frozen} params frozen, {trainable} trainable")
    elif args.freeze_encoder:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("enc_convs", "enc_transformer", "enc_rope")):
                param.requires_grad = False
                frozen += 1
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Encoder frozen: {frozen} params frozen, {trainable} trainable")

    # torch.compile after checkpoint load.
    model = torch.compile(model, dynamic=True)

    # Safety: refuse to train if checkpoints exist but no --resume.
    existing = [f for f in os.listdir(args.output_dir)
                if f.endswith(".pt") and f != "tokenizer.model"]
    if existing and start_epoch == 1 and not args.resume:
        print("ERROR: found existing checkpoints but no --resume flag:")
        for f in sorted(existing):
            print(f"  {f}")
        print("Use --resume to continue, or delete checkpoints to start fresh.")
        return

    # Signal handling.
    sig_state = {"save": False, "stop": False}

    def handle_save(signum, _frame):
        sig_state["save"] = True
        print(f"\n>>> {signal.Signals(signum).name} — saving checkpoint, continuing <<<")

    def handle_stop(signum, _frame):
        sig_state["save"] = True
        sig_state["stop"] = True
        print(f"\n>>> {signal.Signals(signum).name} — saving checkpoint and exiting <<<")

    signal.signal(signal.SIGUSR1, handle_save)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR2):
        signal.signal(sig, handle_stop)
    print("Signals: USR1=checkpoint, TERM/INT/HUP/USR2=checkpoint+exit")

    # atexit safety net.
    atexit_state = {"active": False, "epoch": 0, "step": 0, "epoch_seed": None,
                    "val_state": resume_val_state,
                    "training_done": resume_training_done,
                    "train_loss": resume_train_loss}

    def atexit_save():
        if atexit_state["active"]:
            try:
                print("\n>>> atexit: saving emergency checkpoint <<<")
                fname = interrupt_filename(args.format)
                save_checkpoint(model, optimizer, scaler,
                                atexit_state["epoch"], global_step, best_token_acc,
                                args.output_dir, fname, args.format,
                                epoch_complete=False,
                                training_done=atexit_state.get("training_done", False),
                                epoch_step=atexit_state["step"],
                                epoch_seed=atexit_state["epoch_seed"],
                                wallclock=current_wallclock(),
                                lr_patience_counter=lr_patience_counter,
                                conv_strides=conv_strides,
                                best_cer=best_cer,
                                current_noise_frac=current_noise_frac,
                                noise_dropped=noise_dropped,
                                val_state=atexit_state.get("val_state"),
                                train_loss=atexit_state.get("train_loss", 0.0))
                print(f">>> atexit: saved {fname} <<<")
            except Exception as e:
                print(f">>> atexit: FAILED: {e} <<<")

    atexit.register(atexit_save)

    # Wallclock tracking.
    _wallclock_epoch_start = [time.time()]

    def current_wallclock():
        return wallclock + (time.time() - _wallclock_epoch_start[0])

    print(f"\nTraining {args.format} AE for {args.epochs} epochs")
    print(f"Noise fraction: {args.noise_frac}, grad_accum: {args.grad_accum}, "
          f"base batch: {args.batch_size}")

    for epoch in range(start_epoch, args.epochs + 1):
        _wallclock_epoch_start[0] = time.time()
        start_index = 0
        epoch_seed = epoch * 1000 + 42

        # Resume-into-validation path: training for this epoch is already done.
        # val_state may be None if we interrupted between end-of-training and
        # first val chunk — validate() handles that by starting fresh at index 0.
        skip_training = (epoch == start_epoch and resume_training_done)

        if epoch == start_epoch and resume_epoch_step > 0 and not skip_training:
            if resume_epoch_seed is not None:
                epoch_seed = resume_epoch_seed
            start_index = resume_epoch_step
            print(f"Resuming epoch {epoch} from batch {start_index}")

        if skip_training:
            if resume_epoch_seed is not None:
                epoch_seed = resume_epoch_seed
            print(f"Resuming epoch {epoch} directly into validation "
                  f"(train_loss={resume_train_loss:.4f})")
            atexit_state.update(epoch=epoch, epoch_seed=epoch_seed,
                                step=0, active=True, training_done=True,
                                train_loss=resume_train_loss)
            train_loader = []
            step = -1
            # Stub: train-loss accumulators unused since we skip the loop.
            gpu_train_loss = torch.zeros(1, dtype=torch.float64, device=device)
            gpu_train_tokens = torch.ones(1, dtype=torch.long, device=device)
            epoch_start_time = time.time()
            last_log_time = epoch_start_time
            n_batches = 0
        else:
            batch_sampler = AEBucketedBatchSampler(
                train_dataset, seed=epoch_seed, base_batch_size=args.batch_size,
                max_seq_len=args.max_seq_len, start_index=start_index,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=2,
                collate_fn=lambda batch, _pid=pad_id: ae_collate_fn(batch, _pid),
                pin_memory=True,
                persistent_workers=True,
            )
            n_batches = len(train_loader)
            print(f"  Epoch {epoch}: {len(train_dataset)} samples, {n_batches} batches",
                  flush=True)
            atexit_state.update(epoch=epoch, epoch_seed=epoch_seed,
                                step=start_index, active=True,
                                training_done=False, train_loss=0.0)

            model.train()
            gpu_train_loss = torch.zeros(1, dtype=torch.float64, device=device)
            gpu_train_tokens = torch.zeros(1, dtype=torch.long, device=device)
            optimizer.zero_grad()

            epoch_start_time = time.time()
            last_log_time = epoch_start_time
            step = -1
        for step, batch in enumerate(train_loader):
            atexit_state["step"] = start_index + step
            token_ids = batch["token_ids"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            with autocast("cuda", enabled=args.fp16):
                # Corrupt current_noise_frac of non-pad tokens with random IDs.
                # current_noise_frac may be dropped by --noise-schedule on plateau.
                noise_ids = torch.randint(0, vocab_size, token_ids.shape, device=device)
                corrupt_mask = torch.rand(token_ids.shape, device=device) < current_noise_frac
                corrupt_mask = corrupt_mask & ~pad_mask
                corrupted = torch.where(corrupt_mask, noise_ids, token_ids)

                logits = model(corrupted)

                loss = torch.utils.checkpoint.checkpoint(
                    _ce_loss, logits, token_ids, pad_id, args.grad_accum,
                    ce_weight,
                    use_reentrant=False,
                )

            if not torch.isfinite(loss):
                print(f"  Non-finite loss at step {step}, skipping", flush=True)
                continue

            scaler.scale(loss).backward()

            n_tokens = (~pad_mask).sum()
            gpu_train_loss += loss.detach().double() * args.grad_accum * n_tokens
            gpu_train_tokens += n_tokens

            if (start_index + step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Warmup only on initial run or when --override-lr drops optimizer state.
                # Keyed off steps_since_resume (not absolute global_step) so that
                # override_lr can warm up fresh; guarded so it does NOT re-fire on
                # ordinary resumes that restore the optimizer.
                if args.override_lr is not None:
                    steps_since_resume = global_step - resume_global_step
                    if steps_since_resume <= warmup_steps:
                        warmup_lr = args.override_lr * steps_since_resume / warmup_steps
                        for pg in optimizer.param_groups:
                            pg["lr"] = warmup_lr
                elif global_step <= warmup_steps:
                    warmup_lr = args.lr * global_step / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

            if (start_index + step + 1) % args.grad_accum == 0:
                now = time.time()
                if now - last_log_time >= 10:
                    loss_val = loss.item() * args.grad_accum
                    pct = 100 * (step + 1) / n_batches
                    elapsed = now - epoch_start_time
                    rate = (step + 1) / max(elapsed, 1)
                    eta_m = (n_batches - step - 1) / max(rate, 0.01) / 60
                    lr = optimizer.param_groups[0]["lr"]
                    bsz = token_ids.shape[0]
                    seq = token_ids.shape[1]
                    print(f"Epoch {epoch} train: {pct:5.1f}% | {step+1}/{n_batches} | "
                          f"loss={loss_val:.4f} | lr={lr:.2e} | {rate:.1f} it/s | "
                          f"ETA {eta_m:.0f}m | B={bsz} seq={seq}", flush=True)
                    last_log_time = now

            # Signal-triggered checkpoint.
            if sig_state["save"]:
                sig_state["save"] = False
                actual_step = start_index + step + 1
                fname = interrupt_filename(args.format)
                mid_train_loss = (gpu_train_loss.item() /
                                  max(gpu_train_tokens.item(), 1))
                save_checkpoint(model, optimizer, scaler,
                                epoch, global_step, best_token_acc,
                                args.output_dir, fname, args.format,
                                epoch_complete=False, epoch_step=actual_step,
                                epoch_seed=epoch_seed, wallclock=current_wallclock(),
                                lr_patience_counter=lr_patience_counter,
                                conv_strides=conv_strides,
                                best_cer=best_cer,
                                current_noise_frac=current_noise_frac,
                                noise_dropped=noise_dropped,
                                train_loss=mid_train_loss)
                print(f"\n>>> Checkpoint saved ({fname}): epoch {epoch} batch {actual_step} <<<")
                if sig_state["stop"]:
                    print("Exiting cleanly.")
                    atexit_state["active"] = False
                    return

        # Flush remaining accumulated gradients.
        if (start_index + step + 1) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        if skip_training:
            avg_train_loss = resume_train_loss
        else:
            avg_train_loss = (gpu_train_loss.item() /
                              max(gpu_train_tokens.item(), 1))
            atexit_state["train_loss"] = avg_train_loss

        # Validation sample count scales with accuracy to maintain CI precision.
        # n = ceil(15 / error_rate): CER has higher variance than binomial token
        # accuracy (correlated errors from structural tokens), so ~4x the binomial
        # requirement. No cap — measurement precision matters most at high accuracy.
        error_rate = max(1 - best_token_acc, 1e-4)
        n_val = max(500, int(15.0 / error_rate + 0.5))
        # Mark epoch training complete; atexit save will now persist val_state
        # along with training_done=True so resume goes straight to validation.
        atexit_state["step"] = 0
        atexit_state["training_done"] = True
        val_result = validate(
            model, val_dataset, pad_id, gpu_decoder, device,
            n_samples=n_val, epoch=epoch, fp16=args.fp16,
            atexit_state=atexit_state, sig_state=sig_state)

        if val_result is None:
            # Validation was interrupted mid-way. Save checkpoint with val_state;
            # epoch_complete=False + training_done=True signals resume-into-val.
            fname = interrupt_filename(args.format)
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_token_acc,
                            args.output_dir, fname, args.format,
                            epoch_complete=False, training_done=True,
                            epoch_seed=epoch_seed,
                            wallclock=current_wallclock(),
                            lr_patience_counter=lr_patience_counter,
                            conv_strides=conv_strides,
                            best_cer=best_cer,
                            current_noise_frac=current_noise_frac,
                            noise_dropped=noise_dropped,
                            val_state=atexit_state.get("val_state"),
                            train_loss=avg_train_loss)
            print(f"Exiting cleanly (signal during validation). Saved {fname}.")
            atexit_state["active"] = False
            return
        token_acc, val_loss, cer, perfect, n_evaluated = val_result

        # Validation completed successfully — clear val_state and resume-into-val guard.
        atexit_state["val_state"] = None
        atexit_state["training_done"] = False
        resume_training_done = False
        resume_val_state = None

        lr = optimizer.param_groups[0]["lr"]
        perfect_rate = perfect / max(n_evaluated, 1)
        log_entry = {
            "epoch": epoch,
            "format": args.format,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "token_acc": round(token_acc, 6),
            "cer": round(cer, 6),
            "perfect": perfect,
            "n_val": n_evaluated,
            "perfect_rate": round(perfect_rate, 6),
            "lr": lr,
            "global_step": global_step,
            "wallclock": round(current_wallclock(), 1),
        }
        log_entries.append(log_entry)

        print(f"Epoch {epoch}: train={avg_train_loss:.4f} val={val_loss:.4f} "
              f"acc={token_acc:.4f} CER={cer:.2%} perfect={perfect}/{n_evaluated} "
              f"({perfect_rate:.1%}) lr={lr:.2e} ({current_wallclock()/3600:.1f}h)")

        # Save best checkpoint + LR plateau tracking.
        # Primary criterion: token_acc. Tiebreaker: lower CER.
        improved = (token_acc > best_token_acc or
                    (token_acc == best_token_acc and cer < best_cer))
        if improved:
            best_token_acc = token_acc
            best_cer = cer
            lr_patience_counter = 0
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_token_acc,
                            args.output_dir, f"{args.format}_best.pt", args.format,
                            wallclock=current_wallclock(),
                            lr_patience_counter=lr_patience_counter,
                            conv_strides=conv_strides,
                            best_cer=best_cer,
                            current_noise_frac=current_noise_frac,
                            noise_dropped=noise_dropped,
                            train_loss=avg_train_loss)
            print(f"  -> New best (acc={token_acc:.4f} CER={cer:.2%})")
        else:
            lr_patience_counter += 1
            cur_lr = optimizer.param_groups[0]["lr"]
            if lr_patience_counter >= args.lr_patience:
                # Single-shot noise drop takes precedence on first plateau.
                if (args.noise_schedule and not noise_dropped
                        and current_noise_frac > 0):
                    old_noise = current_noise_frac
                    current_noise_frac = 0.0
                    noise_dropped = True
                    lr_patience_counter = 0
                    print(f"  -> Plateau: noise_frac {old_noise:.2f} -> 0.00 "
                          f"(schedule)")
                elif cur_lr > args.min_lr:
                    new_lr = max(cur_lr * args.lr_factor, args.min_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    lr_patience_counter = 0
                    print(f"  -> Plateau: LR {cur_lr:.2e} -> {new_lr:.2e}")

        # Periodic checkpoint.
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_token_acc,
                            args.output_dir, f"{args.format}_epoch_{epoch}.pt",
                            args.format, wallclock=current_wallclock(),
                            lr_patience_counter=lr_patience_counter,
                            conv_strides=conv_strides,
                            best_cer=best_cer,
                            current_noise_frac=current_noise_frac,
                            noise_dropped=noise_dropped,
                            train_loss=avg_train_loss)

        # Commit wallclock. Clear atexit BEFORE log save so a crash between log
        # save and next epoch start does NOT trigger a duplicate atexit save
        # that would re-run this epoch and duplicate the log entry on resume.
        wallclock = current_wallclock()
        atexit_state["active"] = False

        # Save training log.
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(log_entries, f, indent=2)

        # Clean same-format interrupt checkpoints after successful epoch.
        prefix = f"{args.format}_interrupt_"
        for f in os.listdir(args.output_dir):
            if f.startswith(prefix) and f.endswith(".pt"):
                os.remove(os.path.join(args.output_dir, f))

        # Signal received after full epoch complete (train + val + log).
        # Save with epoch_complete=True so resume starts the next epoch.
        if sig_state["stop"]:
            fname = interrupt_filename(args.format)
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_token_acc,
                            args.output_dir, fname, args.format,
                            epoch_complete=True, wallclock=current_wallclock(),
                            lr_patience_counter=lr_patience_counter,
                            conv_strides=conv_strides,
                            best_cer=best_cer,
                            current_noise_frac=current_noise_frac,
                            noise_dropped=noise_dropped)
            print("Exiting cleanly (signal after epoch complete).")
            atexit_state["active"] = False
            return

    print(f"\nTraining complete. Best token accuracy: {best_token_acc:.4f}")
    atexit_state["active"] = False


def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder training for latent translation")
    parser.add_argument("--format", required=True, choices=["json", "xml"],
                        help="Which sequences to train on")
    parser.add_argument("--data-dir", default="data/run7",
                        help="Data directory containing train/val dataset.pt")
    parser.add_argument("--tokenizer", default="models/run8/tokenizer.model",
                        help="Tokenizer model path")
    parser.add_argument("--output-dir", default="models/run8/ae",
                        help="Output directory for checkpoints")

    # Model architecture.
    parser.add_argument("--d-emb", type=int, default=384)
    parser.add_argument("--emb-rank", type=int, default=128)
    parser.add_argument("--conv-channels", default="384,384,384",
                        help="Comma-separated conv channel dims")
    parser.add_argument("--conv-strides", default="2,2,2",
                        help="Comma-separated conv strides (auto-detected from checkpoint)")
    parser.add_argument("--n-enc-layers", type=int, default=4,
                        help="Encoder transformer layers (auto-detected from checkpoint)")
    parser.add_argument("--n-dec-layers", type=int, default=2,
                        help="Decoder transformer layers (auto-detected from checkpoint)")
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1152)

    # Training.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--noise-frac", type=float, default=0.20)
    parser.add_argument("--noise-schedule", action="store_true",
                        help="On first plateau, drop noise_frac to 0 (single shot)")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--lr-patience", type=int, default=3,
                        help="Epochs without improvement before reducing LR")
    parser.add_argument("--lr-factor", type=float, default=0.5,
                        help="Factor to multiply LR on plateau")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum learning rate floor")
    parser.add_argument("--fp16", action="store_true")

    # Resume.
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--override-lr", type=float, default=None,
                        help="Force this LR on resume")
    parser.add_argument("--freeze-decoder", action="store_true",
                        help="Freeze decoder params (encoder-only training)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder params (decoder-only training)")
    parser.add_argument("--freq-weight", action="store_true",
                        help="Inverse-sqrt-frequency CE loss weighting")
    parser.add_argument("--clean", action="store_true",
                        help="Filter out structurally corrupt (aug_type=corrupted) samples")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
