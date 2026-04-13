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

"""Diffusion training loop for the transmutation model.

Token-manifold noise with fixed t-bucket schedule and CE against clean tokens.
"""

import atexit
import argparse
from datetime import datetime
import json
import math
import os
import random
import re
import signal
import time
import unicodedata
import xml.etree.ElementTree as ET

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.amp import GradScaler, autocast

from dataset import create_dataloader, PrebuiltDiffusionDataset
from model import build_model, LENGTH_BUCKETS

# Module-level ref for signal handler to kill child subprocesses
_active_child = None


# ── Run7: 2-stage curriculum (same as run6) ─────────────────────────────────
#
# Stage 1: clean samples only (learn faithful mapping).
# Stage 2: + corrupted samples (learn robustness).

STAGE_FILTERS = {
    1: {"max_src_tokens": 1152, "max_complexity": 8, "allow_corrupt": False},
    2: {"max_src_tokens": 1152, "max_complexity": 8, "allow_corrupt": True},
}

STAGE_ADVANCE_THRESHOLD = 0.90

# Fixed t/r buckets for training. Overlapping at 0.125 increments, r=0.25 step size.
# Matches 4-step inference: t=[1.0, 0.75, 0.50, 0.25] with r=0.25 each.
# Overlap gives coverage at intermediate points.
T_BUCKETS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]


def _unwrap(model):
    """Get the underlying module from a torch.compiled model."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def _ce_loss(logits, tgt_ids, pad_id):
    """Cross-entropy loss. Wrapped in gradient checkpoint to free logits after forward."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tgt_ids.reshape(-1),
        ignore_index=pad_id,
    )




# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer for vocab size.
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()
    print(f"Vocab size: {vocab_size}")

    # GPU-accelerated token decoder for eval CER/WER.
    gpu_decoder = GPUTokenDecoder(sp, device=device)
    print(f"GPU token decoder: {len(gpu_decoder.piece_bytes)} bytes, {vocab_size} pieces")

    # Build model.
    model = build_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        emb_rank=args.emb_rank,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    # Load pre-built datasets once — filtered per stage at runtime.
    train_dataset = PrebuiltDiffusionDataset(
        os.path.join(args.data_dir, "train", "dataset.pt"), args.tokenizer)
    val_dataset = PrebuiltDiffusionDataset(
        os.path.join(args.data_dir, "val", "dataset.pt"), args.tokenizer)
    dl_kwargs = dict(
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        pad_id=pad_id,
    )

    # Optimizer (constant LR after warmup).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    warmup_steps = args.warmup_steps

    # Mixed precision.
    scaler = GradScaler("cuda", enabled=args.fp16)

    # Training state.
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_eval_exact = 0
    log_entries = []
    global_step = 0
    start_epoch = 1
    resume_epoch_seed = None
    resume_epoch_step = 0
    resume_training_done = False
    resume_train_loss = 0.0
    current_stage = args.stage
    stage_good_epochs = 0
    wallclock = 0.0  # cumulative wall-clock seconds across all epochs/resumes

    # Resume from checkpoint.
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            print(f"  Ignoring unexpected checkpoint keys: {unexpected}")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        completed_epoch = ckpt.get("epoch_complete", False)
        start_epoch = ckpt["epoch"] + 1 if completed_epoch else ckpt["epoch"]
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_eval_exact = ckpt.get("best_eval_exact", 0)
        current_stage = ckpt.get("stage", 1)
        stage_good_epochs = ckpt.get("stage_good_epochs", 0)
        wallclock = ckpt.get("wallclock", 0.0)
        if not completed_epoch:
            resume_epoch_seed = ckpt.get("epoch_seed")
            resume_epoch_step = ckpt.get("epoch_step", 0)
            resume_training_done = ckpt.get("training_done", False)
            resume_train_loss = ckpt.get("train_loss", 0.0)
        if "val_state" in ckpt:
            atexit_state["val_state"] = ckpt["val_state"]
        # Reload existing log.
        log_path = os.path.join(args.output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = json.load(f)
        resumed_lr = optimizer.param_groups[0]["lr"]
        if args.override_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = args.override_lr
            print(f"Resuming from epoch {start_epoch}, step {resume_epoch_step}, global_step={global_step}, best_val_loss={best_val_loss:.4f}, lr={resumed_lr}→{args.override_lr} (override), stage={current_stage}")
        else:
            print(f"Resuming from epoch {start_epoch}, step {resume_epoch_step}, global_step={global_step}, best_val_loss={best_val_loss:.4f}, lr={resumed_lr}, stage={current_stage}")

    # torch.compile AFTER checkpoint load so state_dict keys don't have _orig_mod prefix.
    model = torch.compile(model, dynamic=True)

    # Safety: refuse to train if checkpoint state is inconsistent.
    existing = [f for f in os.listdir(args.output_dir)
                if f.endswith(".pt") and f != "tokenizer.model"]
    if existing:
        if start_epoch == 1 and not args.resume:
            print("ERROR: found existing checkpoints but no --resume flag:")
            for f in sorted(existing):
                print(f"  {f}")
            print("Use --resume to continue, or delete checkpoints to start fresh.")
            return model

    # Signal handling: preserve state on any interruption.
    sig_state = {"save": False, "stop": False}
    def handle_save(signum, _frame):
        sig_state["save"] = True
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint, continuing <<<")
    def handle_stop(signum, _frame):
        sig_state["save"] = True
        sig_state["stop"] = True
        if _active_child and _active_child.poll() is None:
            _active_child.terminate()
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint and exiting <<<")
    signal.signal(signal.SIGUSR1, handle_save)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR2):
        signal.signal(sig, handle_stop)
    print("Signals: USR1=checkpoint, TERM/INT/HUP/USR2=checkpoint+exit")

    # atexit safety net: save state if process exits unexpectedly.
    atexit_state = {"active": False, "epoch": 0, "step": 0, "epoch_seed": None,
                    "training_done": False, "train_loss": 0.0}
    def atexit_save():
        if atexit_state["active"]:
            try:
                print("\n>>> atexit: saving emergency checkpoint <<<")
                actual_step = atexit_state["step"]
                fname = interrupt_filename()
                save_checkpoint(model, optimizer, scaler,
                                atexit_state["epoch"], global_step, best_val_loss,
                                args.output_dir, fname, epoch_complete=False,
                                epoch_step=actual_step, epoch_seed=atexit_state["epoch_seed"],
                                stage=current_stage, stage_good_epochs=stage_good_epochs,
                                training_done=atexit_state["training_done"],
                                val_state=atexit_state.get("val_state"),
                                train_loss=atexit_state.get("train_loss", 0.0),
                                best_eval_exact=best_eval_exact, wallclock=current_wallclock())
                print(f">>> atexit: saved at epoch {atexit_state['epoch']} batch {actual_step} (training_done={atexit_state['training_done']}) <<<")
            except Exception as e:
                print(f">>> atexit: FAILED to save checkpoint: {e} <<<")
    atexit.register(atexit_save)

    # Stage filtering.
    def apply_stage_filters():
        sf = STAGE_FILTERS[current_stage]
        n_train = train_dataset.apply_stage_filter(**sf)
        n_val = val_dataset.apply_stage_filter(**sf)
        print(f"  Stage {current_stage}: corrupt={'yes' if sf['allow_corrupt'] else 'no'} → {n_train} train, {n_val} val")
    apply_stage_filters()
    print(f"Training stage: {current_stage} (advance at {STAGE_ADVANCE_THRESHOLD:.0%} eval_exact, max={args.max_stage})")

    # Wallclock tracking.
    _wallclock_epoch_start = [time.time()]
    def current_wallclock():
        return wallclock + (time.time() - _wallclock_epoch_start[0])

    print(f"\nTraining for {args.epochs} epochs (constant LR after warmup)")
    print(f"Grad accumulation: {args.grad_accum}, base batch: {args.batch_size} (bucketed by length)")
    print(f"Denoising training: eval_steps={args.eval_denoise_steps}")

    for epoch in range(start_epoch, args.epochs + 1):
        _wallclock_epoch_start[0] = time.time()
        start_index = 0
        epoch_seed = epoch * 1000 + 42

        if resume_training_done and epoch == start_epoch:
            print(f"Resuming epoch {epoch} post-training (skipping to validation)")
            avg_train_loss = resume_train_loss
        else:
            if epoch == start_epoch and resume_epoch_step > 0:
                if resume_epoch_seed is not None:
                    epoch_seed = resume_epoch_seed
                start_index = resume_epoch_step
                print(f"Resuming epoch {epoch} from batch {start_index} (seed={epoch_seed})")

            train_loader, epoch_seed, _ = create_dataloader(
                data_dir="unused", shuffle=True,
                epoch_seed=epoch_seed, start_index=start_index,
                max_samples=args.max_epoch_samples, dataset=train_dataset,
                bucketed=True, **dl_kwargs,
            )
            n_dataset = len(train_loader.dataset)
            print(f"  Epoch {epoch}: {n_dataset} in dataset, {len(train_loader)} batches (bucketed)", flush=True)
            atexit_state.update(epoch=epoch, epoch_seed=epoch_seed, step=start_index, active=True, training_done=False)

            # Train.
            model.train()
            gpu_train_loss = torch.zeros(1, dtype=torch.float64, device=device)
            gpu_train_tokens = torch.zeros(1, dtype=torch.long, device=device)
            optimizer.zero_grad()

            n_batches = len(train_loader)
            epoch_start_time = time.time()
            last_log_time = epoch_start_time
            for step, batch in enumerate(train_loader):
                atexit_state["step"] = start_index + step
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                src_mask = batch["src_mask"].to(device)
                tgt_mask = batch["tgt_mask"].to(device)
                bucket_labels = batch["bucket_labels"].to(device)

                with autocast("cuda", enabled=args.fp16):
                    B_batch = src_ids.shape[0]

                    # Sample t from fixed buckets, r=0.25 (or r=t for last bucket).
                    bucket_idx = torch.randint(0, len(T_BUCKETS), (B_batch,), device=device)
                    t = torch.tensor(T_BUCKETS, device=device)[bucket_idx]
                    r = torch.where(t > 0.125, torch.full_like(t, 0.25), t)

                    # Discrete corruption: replace t-fraction of positions with random tokens.
                    noise_ids = torch.randint(0, vocab_size, tgt_ids.shape, device=device)
                    corrupt_mask = torch.rand(tgt_ids.shape, device=device) < t.unsqueeze(1)
                    if tgt_mask is not None:
                        corrupt_mask = corrupt_mask & ~tgt_mask  # never corrupt padding
                    corrupted_ids = torch.where(corrupt_mask, noise_ids, tgt_ids)

                    # Model takes token IDs, returns logits.
                    logits = model(src_ids, corrupted_ids, t, src_mask, tgt_mask, r=r)

                    # CE against clean tokens.
                    denoising_loss = torch.utils.checkpoint.checkpoint(
                        _ce_loss, logits, tgt_ids, pad_id,
                        use_reentrant=False,
                    )

                    # Length prediction loss
                    length_logits = model.predict_length(src_ids, src_mask)
                    length_loss = F.cross_entropy(length_logits, bucket_labels)

                    loss = denoising_loss + args.length_loss_weight * length_loss
                    loss = loss / args.grad_accum

                if not torch.isfinite(loss):
                    print(f"  Non-finite loss at step {step}, skipping batch", flush=True)
                    continue

                scaler.scale(loss).backward()

                n_tokens = (~tgt_mask).sum()
                gpu_train_loss += loss.detach().double() * args.grad_accum * n_tokens
                gpu_train_tokens += n_tokens

                if (start_index + step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step <= warmup_steps:
                        warmup_lr = args.lr * global_step / warmup_steps
                        for pg in optimizer.param_groups:
                            pg["lr"] = warmup_lr

                if (start_index + step + 1) % args.grad_accum == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    loss_val = loss.item() * args.grad_accum
                    now = time.time()
                    if now - last_log_time >= 10:
                        pct = 100 * (step + 1) / n_batches
                        elapsed = now - epoch_start_time
                        rate = (step + 1) / max(elapsed, 1)
                        eta_m = (n_batches - step - 1) / max(rate, 0.01) / 60
                        bsz = src_ids.shape[0]
                        seq = max(src_ids.shape[1], tgt_ids.shape[1])
                        print(f"Epoch {epoch} train: {pct:5.1f}% | {step+1}/{n_batches} | loss={loss_val:.4f} | lr={cur_lr:.2e} | {rate:.1f} it/s | ETA {eta_m:.0f}m | B={bsz} seq={seq}", flush=True)
                        last_log_time = now

                # Handle signal-triggered checkpoint.
                if sig_state["save"]:
                    sig_state["save"] = False
                    actual_step = start_index + step + 1
                    fname = interrupt_filename()
                    save_checkpoint(model, optimizer, scaler,
                                    epoch, global_step, best_val_loss,
                                    args.output_dir, fname, epoch_complete=False,
                                    epoch_step=actual_step, epoch_seed=epoch_seed,
                                    stage=current_stage, stage_good_epochs=stage_good_epochs,
                                    best_eval_exact=best_eval_exact, wallclock=current_wallclock())
                    print(f"\n>>> Checkpoint saved ({fname}): epoch {epoch} batch {actual_step} stage={current_stage} <<<")
                    if sig_state["stop"]:
                        print("Exiting cleanly.")
                        return model

            # Flush any remaining accumulated gradients.
            if (start_index + step + 1) % args.grad_accum != 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            train_loss = gpu_train_loss.item()
            train_tokens = gpu_train_tokens.item()
            avg_train_loss = train_loss / max(train_tokens, 1)
            atexit_state["training_done"] = True
            atexit_state["train_loss"] = avg_train_loss

        # Validation = diffusion eval.
        atexit_state["training_done"] = True
        eval_n = 500
        try:
            eval_exact, eval_semantic, eval_xml_ok, eval_total, eval_cer, eval_wer = diffusion_eval(
                model, sp, n_samples=eval_n,
                val_dataset=val_dataset,
                max_src_len=args.max_src_len,
                n_steps=args.eval_denoise_steps,
                output_dir=args.output_dir, epoch=epoch,
                device=device, gpu_decoder=gpu_decoder,
                atexit_state=atexit_state, sig_state=sig_state,
            )
        except Exception as e:
            eval_exact, eval_semantic, eval_xml_ok, eval_total, eval_cer, eval_wer = 0, 0, 0, eval_n, 1.0, 1.0
            print(f"Eval error: {e}", flush=True)

        eval_rate = eval_exact / max(eval_total, 1)

        # Constant LR after warmup (standard for diffusion training).
        new_lr = optimizer.param_groups[0]["lr"]

        # Stage advance: clean → clean+corrupt.
        if current_stage < args.max_stage:
            if eval_rate >= STAGE_ADVANCE_THRESHOLD:
                stage_good_epochs += 1
            else:
                stage_good_epochs = 0
            if stage_good_epochs >= args.stage_patience:
                current_stage += 1
                stage_good_epochs = 0
                apply_stage_filters()
                print(f"  >>> Stage advanced to {current_stage} (eval_exact={eval_rate:.0%} for {args.stage_patience} epochs)")

        log_entry = {
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": avg_train_loss,
            "eval_exact": eval_exact,
            "eval_semantic": eval_semantic,
            "eval_xml_ok": eval_xml_ok,
            "eval_total": eval_total,
            "eval_cer": round(eval_cer, 6),
            "eval_wer": round(eval_wer, 6),
            "lr": new_lr,
            "global_step": global_step,
            "wallclock": round(current_wallclock(), 1),
            "denoise_steps": args.eval_denoise_steps,
        }
        log_entries.append(log_entry)

        print(f"Epoch {epoch}: train={avg_train_loss:.4f} eval={eval_exact}/{eval_total}exact {eval_semantic}/{eval_total}sem {eval_xml_ok}/{eval_total}xml CER={eval_cer:.2%} WER={eval_wer:.2%} lr={new_lr:.2e}")

        # Save best checkpoint.
        if eval_exact > best_eval_exact or (eval_exact == best_eval_exact and eval_cer < best_val_loss):
            best_eval_exact = eval_exact
            best_val_loss = eval_cer
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_val_loss, args.output_dir, "best.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            best_eval_exact=best_eval_exact, wallclock=current_wallclock())
            print(f"  -> New best (eval={eval_exact}/{eval_total} CER={eval_cer:.2%})")

        # Save periodic checkpoint.
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_val_loss, args.output_dir, f"epoch_{epoch}.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            best_eval_exact=best_eval_exact, wallclock=current_wallclock())

        # Epoch complete — commit wallclock delta.
        wallclock = current_wallclock()
        atexit_state["active"] = False

        # Clean up interrupt checkpoints after successful epoch completion.
        for f in os.listdir(args.output_dir):
            if f.startswith("interrupt_") and f.endswith(".pt"):
                os.remove(os.path.join(args.output_dir, f))

        # Save training log.
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(log_entries, f, indent=2)

        # Stop signal received during eval — save as incomplete so eval re-runs on resume.
        if sig_state["stop"]:
            save_checkpoint(model, optimizer, scaler,
                            epoch, global_step, best_val_loss,
                            args.output_dir, interrupt_filename(), epoch_complete=False,
                            epoch_step=0, training_done=True,
                            val_state=atexit_state.get("val_state"),
                            train_loss=avg_train_loss,
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            best_eval_exact=best_eval_exact, wallclock=current_wallclock())
            print(f"Exiting cleanly (signal caught during eval).")
            atexit_state["active"] = False
            return model

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    atexit_state["active"] = False
    return model


# ── Diffusion Eval ───────────────────────────────────────────────────────────

@torch.no_grad()
def diffusion_eval(model, sp, n_samples=50,
                   val_dataset=None,
                   max_src_len=1152, n_steps=4,
                   output_dir=None, epoch=0,
                   device="cuda", gpu_decoder=None,
                   atexit_state=None, sig_state=None):
    """Evaluate by denoising: init noise, denoise N steps, discretize, score.
    Resumable via atexit_state['val_state']."""
    model.eval()
    CHUNK = 4  # eval samples per batch — keeps transformer intermediates in VRAM

    n_avail = len(val_dataset)
    n = min(n_samples, n_avail)
    g = torch.Generator()
    g.manual_seed(epoch * 7777 + 42)
    indices = torch.randperm(n_avail, generator=g)[:n]

    # Resume from saved eval state.
    start_idx = 0
    exact_count = 0
    semantic_count = 0
    xml_ok_count = 0
    total_cer_edits = 0
    ne_cer_chars = 0
    total_wer_edits = 0
    ne_wer_words = 0
    inferences = []

    if atexit_state and atexit_state.get("val_state"):
        vs = atexit_state["val_state"]
        if vs.get("epoch") == epoch and vs.get("n_samples") == n:
            start_idx = vs.get("completed", 0)
            r = vs.get("results", {})
            exact_count = r.get("exact", 0)
            semantic_count = r.get("semantic", 0)
            xml_ok_count = r.get("xml_ok", 0)
            total_cer_edits = r.get("cer_edits", 0)
            ne_cer_chars = r.get("cer_chars", 0)
            total_wer_edits = r.get("wer_edits", 0)
            ne_wer_words = r.get("wer_words", 0)
            if start_idx > 0:
                print(f"  Resuming eval from sample {start_idx}/{n}", flush=True)

    print(f"Diffusion eval: {n} samples, {n_steps} denoise steps on {device}...", flush=True)

    for chunk_start in range(start_idx, n, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n)
        chunk_indices = indices[chunk_start:chunk_end]
        chunk_data = [val_dataset[i] for i in chunk_indices]

        # Decode targets to strings.
        chunk_src_ids = [s["src_ids"].tolist() for s in chunk_data]
        chunk_targets = [sp.decode(s["tgt_ids"].tolist()) for s in chunk_data]
        chunk_inputs = [sp.decode(ids) for ids in chunk_src_ids]

        # Pad source
        _max_src = max(len(ids) for ids in chunk_src_ids)
        _src_t = torch.zeros(len(chunk_src_ids), _max_src, dtype=torch.long, device=device)
        _src_mask = torch.ones(len(chunk_src_ids), _max_src, dtype=torch.bool, device=device)
        for i, ids in enumerate(chunk_src_ids):
            _src_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            _src_mask[i, :len(ids)] = False

        # Predict output length
        length_logits = model.predict_length(_src_t, _src_mask)
        bucket_indices_pred = length_logits.argmax(dim=-1)
        # One bucket up for safety
        bucket_indices_pred = (bucket_indices_pred + 1).clamp(max=len(LENGTH_BUCKETS) - 1)
        tgt_len = LENGTH_BUCKETS[bucket_indices_pred.max().item()]

        # Discrete denoising: start from random tokens, iterate.
        B = len(chunk_src_ids)
        current_ids = torch.randint(0, sp.get_piece_size(), (B, tgt_len), device=device)
        prev_pred = None
        steps_used = n_steps
        is_first_chunk = (chunk_start == 0)

        for step_i in range(n_steps):
            if step_i == 0:
                t_val = torch.ones(B, device=device)
                r_val = torch.ones(B, device=device)
            else:
                # Refinement: tell model input is partially correct.
                t_val = torch.full((B,), 0.125, device=device)
                r_val = torch.full((B,), 0.125, device=device)

            pred_tokens = model.predict_tokens(_src_t, current_ids, t_val, _src_mask, None, r=r_val)

            # Log first sample's intermediate output.
            if is_first_chunk:
                changed = 0 if prev_pred is None else (pred_tokens[0] != prev_pred[0]).sum().item()
                total_toks = (pred_tokens[0] != sp.pad_id()).sum().item()
                step_trimmed = []
                for tid in pred_tokens[0].tolist():
                    if tid == sp.eos_id() or tid == sp.pad_id():
                        break
                    step_trimmed.append(tid)
                step_text = sp.decode(step_trimmed)[:200]
                chg_str = f" Δ{changed}/{total_toks}" if prev_pred is not None else ""
                print(f"    step {step_i+1}/{n_steps} (t={t_val[0]:.3f} r={r_val[0]:.3f}){chg_str}: {step_text}", flush=True)

            # Short-circuit if tokens stopped changing.
            if prev_pred is not None:
                if (pred_tokens == prev_pred).all():
                    steps_used = step_i + 1
                    break
            prev_pred = pred_tokens.clone()

            # Feed prediction directly as next input — no perturbation.
            current_ids = pred_tokens

        token_ids = pred_tokens

        # Decode predictions to strings
        pred_texts = []
        for i in range(B):
            ids = token_ids[i].tolist()
            # Trim at EOS or PAD
            eos_id = sp.eos_id()
            trimmed = []
            for tid in ids:
                if tid == eos_id or tid == sp.pad_id():
                    break
                trimmed.append(tid)
            pred_texts.append(sp.decode(trimmed))

        # Score each sample
        chunk_nonexact = []
        for i, (pred, target) in enumerate(zip(pred_texts, chunk_targets)):
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

            char_ref_len = len(norm_tgt)
            wer_chars = 0

            if exact:
                exact_count += 1
            elif semantic:
                semantic_count += 1
            else:
                chunk_nonexact.append(i)
                ne_cer_chars += char_ref_len
                ne_wer_words += char_ref_len
                wer_chars = char_weighted_wer(norm_pred.split(), norm_tgt.split())
                total_wer_edits += wer_chars

            if xml_ok:
                xml_ok_count += 1

            inferences.append({
                "input": chunk_inputs[i],
                "expected": target,
                "predicted": pred,
                "exact": exact,
                "semantic": semantic,
                "xml_ok": xml_ok,
                "cer": 0.0 if exact or semantic else -1.0,
                "wer": round(wer_chars / max(char_ref_len, 1), 6),
            })

        # Batched Triton CER for non-exact pairs
        if chunk_nonexact and gpu_decoder is not None:
            ne_pred_bytes = [gpu_normalize_ws(gpu_decoder.decode_to_bytes(
                token_ids[i].tolist())) for i in chunk_nonexact]
            ne_tgt_bytes = [gpu_normalize_ws(gpu_decoder.decode_to_bytes(
                [t for t in chunk_data[i]["tgt_ids"].tolist()])) for i in chunk_nonexact]
            distances = batched_triton_levenshtein(ne_pred_bytes, ne_tgt_bytes, gpu_decoder.device)
            for ci, dist in zip(chunk_nonexact, distances):
                global_i = len(inferences) - len(pred_texts) + ci
                ref = max(len(re.sub(r"\s+", " ", chunk_targets[ci].strip())), 1)
                inferences[global_i]["cer"] = round(dist / ref, 6)
                total_cer_edits += dist

        completed = chunk_end
        degen_tag = f" (degen@step{steps_used})" if steps_used < n_steps else ""
        print(f"  Eval {completed}/{n}: {exact_count} exact, {semantic_count} sem, {xml_ok_count} xml{degen_tag}", flush=True)

        # Save eval progress for resumability.
        if atexit_state is not None:
            atexit_state["val_state"] = {
                "epoch": epoch, "n_samples": n, "completed": completed,
                "results": {
                    "exact": exact_count, "semantic": semantic_count, "xml_ok": xml_ok_count,
                    "cer_edits": total_cer_edits, "cer_chars": ne_cer_chars,
                    "wer_edits": total_wer_edits, "wer_words": ne_wer_words,
                },
            }

        # Check stop signal.
        if sig_state and sig_state["stop"]:
            print(f"  Eval interrupted at {completed}/{n}", flush=True)
            break

    # Fill CER for exact/semantic matches.
    for inf in inferences:
        if inf["cer"] < 0:
            inf["cer"] = 0.0

    # Save inference results.
    if output_dir:
        inf_dir = os.path.join(output_dir, "eval_inferences")
        os.makedirs(inf_dir, exist_ok=True)
        inf_path = os.path.join(inf_dir, f"epoch_{epoch}.jsonl")
        with open(inf_path, "w") as f:
            for inf in inferences:
                json.dump(inf, f, ensure_ascii=False)
                f.write("\n")

    total = len(inferences)
    avg_cer = total_cer_edits / max(ne_cer_chars, 1)
    avg_wer = total_wer_edits / max(ne_wer_words, 1)

    return exact_count, semantic_count, xml_ok_count, total, avg_cer, avg_wer


# ── GPU utilities (preserved from run6) ──────────────────────────────────────

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


# ── Triton kernels (preserved from run6) ─────────────────────────────────────

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


# ── Metrics utilities (preserved from run6) ──────────────────────────────────

def levenshtein(a, b):
    """Compute Levenshtein edit distance between two sequences."""
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


# ── Checkpointing ───────────────────────────────────────────────────────────

def interrupt_filename():
    """Timestamped interrupt checkpoint filename (never overwrites previous)."""
    return f"interrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"


def save_checkpoint(model, optimizer, scaler,
                    epoch, global_step, best_val_loss, output_dir, filename,
                    epoch_complete=True, epoch_step=0, epoch_seed=None,
                    stage=1, stage_good_epochs=0, training_done=False,
                    val_state=None, train_loss=0.0, best_eval_exact=0, wallclock=0.0):
    path = os.path.join(output_dir, filename)
    ckpt = {
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "epoch_step": epoch_step,
        "epoch_seed": epoch_seed,
        "training_done": training_done,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "stage": stage,
        "stage_good_epochs": stage_good_epochs,
        "train_loss": train_loss,
        "best_eval_exact": best_eval_exact,
        "wallclock": wallclock,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if val_state is not None:
        ckpt["val_state"] = val_state
    torch.save(ckpt, path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diffusion training for transmutation model")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--tokenizer", default="models/tokenizer.model", help="Tokenizer model path")
    parser.add_argument("--output-dir", default="models", help="Output directory")

    # Model.
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--emb-rank", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--override-lr", type=float, default=None,
                        help="Force this LR on resume")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")

    # Diffusion.
    parser.add_argument("--eval-denoise-steps", type=int, default=1,
                        help="Denoising steps during evaluation")
    parser.add_argument("--length-loss-weight", type=float, default=0.1,
                        help="Weight for length prediction loss")

    # Curriculum.
    parser.add_argument("--stage", type=int, default=1,
                        help="Starting curriculum stage (1=clean, 2=+corrupt)")
    parser.add_argument("--max-stage", type=int, default=2,
                        help="Maximum curriculum stage")
    parser.add_argument("--stage-patience", type=int, default=2,
                        help="Consecutive epochs above threshold before advancing")

    parser.add_argument("--max-src-len", type=int, default=1152)
    parser.add_argument("--max-tgt-len", type=int, default=1536)
    parser.add_argument("--max-epoch-samples", type=int, default=0,
                        help="Cap samples per epoch (0=no cap)")
    parser.add_argument("--num-workers", type=int, default=2)

    # Checkpointing.
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")


    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
