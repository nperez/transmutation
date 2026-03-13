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

"""Training loop for the transmutation model (haiku-first pipeline)."""

import atexit
import argparse
import copy
from datetime import datetime
import json
import os
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import create_dataloader
from infer_cpu import greedy_decode, patch_mamba_for_cpu
from model import build_model


# ── Haiku-first curriculum stages ────────────────────────────────────────────
#
# Each stage increases difficulty by adjusting augmentation ratio, special
# character injection probability, and input corruption percentage.
#
# Stage 1: Natural haiku only — learn basic JSON→XML structure on real data.
# Stage 2: 1:5 augmentation — begin generalizing beyond memorized content.
# Stage 3: 1:10 augmentation + moderate special chars — CDATA practice.
# Stage 4: Full augmentation + special chars + light corruption.
# Stage 5: Full augmentation + high special chars + heavier corruption.

HAIKU_STAGES = {
    1: {"aug_ratio": 0,  "special_prob": 0.0,  "corrupt_pct": 0,  "sample_pct": 10},
    2: {"aug_ratio": 5,  "special_prob": 0.15, "corrupt_pct": 0,  "sample_pct": 5},
    3: {"aug_ratio": 10, "special_prob": 0.30, "corrupt_pct": 0,  "sample_pct": 5},
    4: {"aug_ratio": 10, "special_prob": 0.40, "corrupt_pct": 10, "sample_pct": 5},
    5: {"aug_ratio": 10, "special_prob": 0.40, "corrupt_pct": 20, "sample_pct": 5},
}


def generate_haiku_data(augment_bin, data_dir, split, stage, seed):
    """Call the Go augment binary to produce training or validation data.

    The augment binary reads haiku JSONL from data_dir/haiku, samples a
    percentage of the corpus, and outputs 1:N augmented variants to stdout.
    Stage parameters control augmentation ratio, special char injection,
    and corruption.
    """
    params = HAIKU_STAGES[stage]
    haiku_dir = os.path.join(data_dir, "haiku")

    cmd = [
        augment_bin,
        "-dir", haiku_dir,
        "-sample-pct", str(params["sample_pct"]),
        "-aug-ratio", str(params["aug_ratio"]),
        "-special-prob", str(params["special_prob"]),
        "-corrupt-pct", str(params["corrupt_pct"]),
        "-seed", str(seed),
    ]
    if split == "val":
        cmd.append("-val")

    out_dir = os.path.join(data_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "haiku_augmented.jsonl")

    label = f"stage {stage} aug={params['aug_ratio']} sp={params['special_prob']} cor={params['corrupt_pct']}%"
    print(f"Generating {split} data ({label}, seed {seed})...")

    t0 = time.time()
    with open(out_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=120)
    # Print augment binary stats (written to stderr).
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  {line}")
    result.check_returncode()
    print(f"  Generated in {time.time() - t0:.1f}s")


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

    # Build model.
    model = build_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    # Data directories and dataloader args.
    train_data_dir = os.path.join(args.data_dir, "train")
    val_data_dir = os.path.join(args.data_dir, "val")
    dl_kwargs = dict(
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        pad_id=pad_id,
    )

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    warmup_steps = args.warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
    )

    # Loss — optionally upweight content tokens (numbers, strings, code)
    # vs structural XML tokens (IDs 0-15: special + XML tags).
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    use_content_weight = args.content_weight != 1.0
    if use_content_weight:
        print(f"Content weight: {args.content_weight}x (structural tokens 0-{args.structural_max_id} at 1.0x)")
    if args.professor_forcing:
        print(f"Professor forcing: ON (noise={args.token_noise}, using model predictions)")

    # Mixed precision.
    scaler = GradScaler("cuda", enabled=args.fp16)

    # Training state.
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ar_exact = 0
    log_entries = []
    global_step = 0
    start_epoch = 1
    resume_epoch_seed = None
    resume_epoch_step = 0
    current_stage = args.stage
    stage_good_epochs = 0  # consecutive epochs above stage-advance threshold

    # Resume from checkpoint.
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except (KeyError, ValueError):
            print("  Scheduler state incompatible (type changed?), starting fresh")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        completed_epoch = ckpt.get("epoch_complete", False)
        start_epoch = ckpt["epoch"] + 1 if completed_epoch else ckpt["epoch"]
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        current_stage = ckpt.get("stage", args.stage)
        stage_good_epochs = ckpt.get("stage_good_epochs", 0)
        if not completed_epoch:
            resume_epoch_seed = ckpt.get("epoch_seed")
            resume_epoch_step = ckpt.get("epoch_step", 0)
        # Reload existing log.
        log_path = os.path.join(args.output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = json.load(f)
        resumed_lr = optimizer.param_groups[0]["lr"]
        if args.override_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = args.override_lr
            # Reset scheduler so it doesn't immediately reduce the new LR.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=args.lr_patience)
            print(f"Resuming from epoch {start_epoch}, step {resume_epoch_step}, global_step={global_step}, best_val_loss={best_val_loss:.4f}, lr={resumed_lr}→{args.override_lr} (override), stage={current_stage}")
        else:
            print(f"Resuming from epoch {start_epoch}, step {resume_epoch_step}, global_step={global_step}, best_val_loss={best_val_loss:.4f}, lr={resumed_lr}, stage={current_stage}")

    # Signal handling: preserve state on any interruption.
    # SIGUSR1 = save checkpoint, keep training
    # SIGTERM/SIGINT/SIGHUP/SIGUSR2 = save checkpoint, exit cleanly
    sig_state = {"save": False, "stop": False}
    def handle_save(signum, frame):
        sig_state["save"] = True
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint, continuing <<<")
    def handle_stop(signum, frame):
        sig_state["save"] = True
        sig_state["stop"] = True
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint and exiting <<<")
    signal.signal(signal.SIGUSR1, handle_save)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR2):
        signal.signal(sig, handle_stop)
    print("Signals: USR1=checkpoint, TERM/INT/HUP/USR2=checkpoint+exit")

    # atexit safety net: save state if process exits unexpectedly (e.g. unhandled exception).
    atexit_state = {"epoch": 0, "step": 0, "epoch_seed": None, "active": False}
    def atexit_save():
        if not atexit_state["active"]:
            return
        try:
            print("\n>>> atexit: saving emergency checkpoint <<<")
            actual_step = atexit_state["step"]
            fname = interrupt_filename()
            save_checkpoint(model, optimizer, scheduler, scaler,
                            atexit_state["epoch"], global_step, best_val_loss,
                            args.output_dir, fname, epoch_complete=False,
                            epoch_step=actual_step, epoch_seed=atexit_state["epoch_seed"],
                            stage=current_stage, stage_good_epochs=stage_good_epochs)
            print(f">>> atexit: saved at epoch {atexit_state['epoch']} batch {actual_step} <<<")
        except Exception as e:
            print(f">>> atexit: FAILED to save checkpoint: {e} <<<")
    atexit.register(atexit_save)

    # Generate held-out validation data ONCE at max stage with a fixed seed.
    VAL_SEED = 7777777
    generate_haiku_data(args.augment_bin, args.data_dir, "val", args.max_stage, VAL_SEED)
    print(f"Fixed validation set (stage {args.max_stage}, seed {VAL_SEED}), reused every epoch")
    print(f"Training stage: {current_stage} (auto-advance at AR>{args.stage_advance_ar:.0%} for {args.stage_patience} epochs, max={args.max_stage})")

    print(f"\nTraining for {args.epochs} epochs (ReduceLROnPlateau, patience={args.lr_patience})")
    print(f"Grad accumulation: {args.grad_accum}, effective batch: {args.batch_size * args.grad_accum}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = epoch * 1000 + 42
        start_index = 0

        # Generate fresh training data for this epoch (val is fixed).
        if epoch == start_epoch and resume_epoch_step > 0:
            if resume_epoch_seed is not None:
                epoch_seed = resume_epoch_seed
            start_index = resume_epoch_step
            print(f"Resuming epoch {epoch} from batch {start_index} (seed={epoch_seed})")
        generate_haiku_data(args.augment_bin, args.data_dir, "train", current_stage, epoch_seed)

        train_loader, epoch_seed = create_dataloader(
            data_dir=train_data_dir, shuffle=True,
            epoch_seed=epoch_seed, start_index=start_index, **dl_kwargs,
        )
        atexit_state.update(epoch=epoch, epoch_seed=epoch_seed, step=start_index, active=True)

        # Train.
        model.train()
        train_loss = 0.0
        train_tokens = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
        for step, batch in enumerate(pbar):
            atexit_state["step"] = start_index + step
            src = batch["src_ids"].to(device)
            tgt_in = batch["tgt_input"].to(device)
            tgt_labels = batch["tgt_labels"].to(device)
            src_mask = batch["src_key_padding_mask"].to(device)

            if args.token_noise > 0:
                if args.professor_forcing:
                    # Extra forward pass (no grad) to get model's own predictions.
                    with torch.no_grad():
                        with autocast("cuda", enabled=args.fp16):
                            pf_logits = model(src, tgt_in, src_mask)
                        # pf_logits[:, i] predicts target[i], which is tgt_in[:, i+1].
                        # Shift right so replacement_ids aligns with tgt_in.
                        pred_ids = pf_logits.argmax(dim=-1)
                        replacement_ids = torch.cat(
                            [tgt_in[:, :1], pred_ids[:, :-1]], dim=1,
                        )
                    tgt_in = corrupt_content_tokens(
                        tgt_in, args.token_noise, args.structural_max_id,
                        vocab_size, pad_id, replacement_ids=replacement_ids,
                    )
                else:
                    tgt_in = corrupt_content_tokens(
                        tgt_in, args.token_noise, args.structural_max_id, vocab_size, pad_id,
                    )

            with autocast("cuda", enabled=args.fp16):
                logits = model(src, tgt_in, src_mask)
                if use_content_weight:
                    loss = weighted_content_loss(
                        logits, tgt_labels, vocab_size,
                        args.content_weight, args.structural_max_id,
                    )
                else:
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            n_tokens = (tgt_labels != -100).sum().item()
            train_loss += loss.item() * args.grad_accum * n_tokens
            train_tokens += n_tokens

            if (start_index + step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Linear warmup: scale LR during initial steps.
                if global_step <= warmup_steps:
                    warmup_lr = args.lr * global_step / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", lr=f"{cur_lr:.2e}")

            # Handle signal-triggered checkpoint.
            if sig_state["save"]:
                sig_state["save"] = False
                actual_step = start_index + step + 1
                fname = interrupt_filename()
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                                args.output_dir, fname, epoch_complete=False,
                                epoch_step=actual_step, epoch_seed=epoch_seed,
                                stage=current_stage, stage_good_epochs=stage_good_epochs)
                print(f"\n>>> Checkpoint saved ({fname}): epoch {epoch} batch {actual_step} stage={current_stage} <<<")
                if sig_state["stop"]:
                    print("Exiting cleanly.")
                    return model

        avg_train_loss = train_loss / max(train_tokens, 1)

        # Validate (fixed held-out val set, generated once before training).
        val_loader, _ = create_dataloader(
            data_dir=val_data_dir, shuffle=False, **dl_kwargs,
        )
        val_loss, val_tokens, val_exact = validate(model, val_loader, criterion, vocab_size, device, args.fp16, sp)
        avg_val_loss = val_loss / max(val_tokens, 1)

        # Autoregressive eval on fresh augmented haiku samples.
        ar_exact, ar_xml_ok, ar_total = autoregressive_eval(
            model, sp, n_samples=args.ar_eval_samples, device=device,
            augment_bin=args.augment_bin, data_dir=args.data_dir,
            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
            output_dir=args.output_dir, epoch=epoch,
        )

        # Step the plateau scheduler based on AR error rate, not val loss.
        # Val loss is near-zero with teacher forcing even when AR inference is bad.
        ar_error = 1.0 - (ar_exact / max(ar_total, 1))
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(ar_error)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  LR reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Auto-advance curriculum stage.
        ar_rate = ar_exact / max(ar_total, 1)
        if current_stage < args.max_stage:
            if ar_rate >= args.stage_advance_ar:
                stage_good_epochs += 1
            else:
                stage_good_epochs = 0
            if stage_good_epochs >= args.stage_patience:
                current_stage += 1
                stage_good_epochs = 0
                new_params = HAIKU_STAGES[current_stage]
                print(f"  >>> Stage advanced to {current_stage} (AR={ar_rate:.0%} for {args.stage_patience} epochs)")
                print(f"      aug={new_params['aug_ratio']} sp={new_params['special_prob']} cor={new_params['corrupt_pct']}%")

        log_entry = {
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_exact_match": val_exact,
            "ar_exact": ar_exact,
            "ar_xml_ok": ar_xml_ok,
            "ar_total": ar_total,
            "lr": new_lr,
            "global_step": global_step,
        }
        log_entries.append(log_entry)

        print(f"Epoch {epoch}: stage={current_stage} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_exact={val_exact:.2%} ar={ar_exact}/{ar_total}exact {ar_xml_ok}/{ar_total}xml lr={new_lr:.2e}")

        # Save best (by AR exact match, the only reliable metric).
        if ar_exact > best_ar_exact or (ar_exact == best_ar_exact and avg_val_loss < best_val_loss):
            best_ar_exact = ar_exact
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, "best.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs)
            print(f"  -> New best model saved (ar={ar_exact}/{ar_total} val_loss={avg_val_loss:.4f})")

        # Save periodic checkpoint.
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, f"epoch_{epoch}.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs)

        # Epoch complete — atexit not needed until next epoch starts.
        atexit_state["active"] = False

        # Clean up interrupt checkpoints after successful epoch completion.
        for f in os.listdir(args.output_dir):
            if f.startswith("interrupt_") and f.endswith(".pt"):
                os.remove(os.path.join(args.output_dir, f))

        # Save training log.
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(log_entries, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    atexit_state["active"] = False
    return model


@torch.no_grad()
def validate(model, loader, criterion, vocab_size, device, fp16, sp):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        src = batch["src_ids"].to(device)
        tgt_in = batch["tgt_input"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)

        with autocast("cuda", enabled=fp16):
            logits = model(src, tgt_in, src_mask)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

        n_tokens = (tgt_labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Check exact match (greedy decoding from teacher-forced logits).
        preds = logits.argmax(dim=-1)
        mask = tgt_labels != -100
        for i in range(preds.size(0)):
            m = mask[i]
            if torch.equal(preds[i][m], tgt_labels[i][m]):
                exact_matches += 1
            total_samples += 1

    exact_rate = exact_matches / max(total_samples, 1)
    return total_loss, total_tokens, exact_rate


def corrupt_content_tokens(tgt_in, noise_prob, structural_max_id, vocab_size, pad_id,
                           replacement_ids=None):
    """Replace content tokens in decoder input with noise tokens.

    Only content tokens (ID > structural_max_id) are candidates.
    BOS (position 0) and padding are never touched.
    If replacement_ids is provided (professor forcing), use those instead of
    uniform random tokens.
    """
    content_mask = tgt_in > structural_max_id  # content tokens only
    content_mask[:, 0] = False  # never corrupt BOS
    content_mask &= tgt_in != pad_id  # never corrupt padding

    noise_mask = torch.rand_like(tgt_in, dtype=torch.float) < noise_prob
    replace_mask = content_mask & noise_mask

    if replacement_ids is None:
        replacement_ids = torch.randint(
            structural_max_id + 1, vocab_size, tgt_in.shape,
            device=tgt_in.device, dtype=tgt_in.dtype,
        )
    return torch.where(replace_mask, replacement_ids, tgt_in)


def weighted_content_loss(logits, tgt_labels, vocab_size, content_weight, structural_max_id):
    """Cross-entropy loss with higher weight on content tokens.

    Structural tokens (IDs 0..structural_max_id) get weight 1.0.
    All other tokens (actual content being copied) get content_weight.
    Padding positions (label == -100) contribute 0.
    """
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = tgt_labels.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)

    valid = flat_labels != -100
    structural = (flat_labels >= 0) & (flat_labels <= structural_max_id)
    weights = torch.where(structural, 1.0, content_weight)
    weights = torch.where(valid, weights, 0.0)

    return (per_token * weights).sum() / weights.sum().clamp(min=1)


@torch.no_grad()
def autoregressive_eval(model, sp, n_samples=10, device="cuda",
                        augment_bin="/app/augment", data_dir="data",
                        max_src_len=2048, max_tgt_len=4096,
                        output_dir=None, epoch=0):
    """Run greedy autoregressive decoding on fresh augmented haiku samples.

    Uses the augment binary to generate fresh samples with full augmentation
    and special char injection. Writes per-sample results to
    output_dir/ar_inferences/epoch_N.jsonl for failure analysis.
    """
    model.eval()

    # Generate fresh augmented samples — use max difficulty, time-based seed.
    seed = int(time.time()) % 2**32
    max_stage = max(HAIKU_STAGES.keys())
    params = HAIKU_STAGES[max_stage]
    # Request more samples than needed since augment produces natural+augmented.
    # With aug_ratio=10, each sampled haiku yields 11 outputs.
    # We want n_samples augmented ones, so request enough to cover.
    needed_haiku = max(1, (n_samples // max(params["aug_ratio"], 1)) + 2)
    sample_pct = max(0.01, needed_haiku / 750)  # ~75k corpus

    cmd = [
        augment_bin,
        "-dir", os.path.join(data_dir, "haiku"),
        "-sample-pct", f"{sample_pct:.4f}",
        "-aug-ratio", str(params["aug_ratio"]),
        "-special-prob", str(params["special_prob"]),
        "-seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    records = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))

    # Copy model to CPU for autoregressive decoding (Mamba CUDA kernels
    # don't support single-step mode, but the CPU forward pass does).
    cpu_model = copy.deepcopy(model).cpu()
    patch_mamba_for_cpu(cpu_model)
    cpu_model.eval()

    exact_count = 0
    xml_ok_count = 0
    total = 0
    inferences = []

    for record in records[:n_samples]:
        src_ids = sp.encode(record["input"])[:max_src_len]
        target = record["target"]

        pred_ids = greedy_decode(cpu_model, src_ids, sp, max_len=max_tgt_len, device="cpu")
        pred = sp.decode(pred_ids)

        norm_pred = re.sub(r"\s+", " ", pred.strip())
        norm_tgt = re.sub(r"\s+", " ", target.strip())
        exact = norm_pred == norm_tgt

        try:
            ET.fromstring(pred.strip())
            xml_ok = True
        except ET.ParseError:
            xml_ok = False

        if exact:
            exact_count += 1
        if xml_ok:
            xml_ok_count += 1
        total += 1

        inferences.append({
            "input": record["input"],
            "expected": target,
            "predicted": pred,
            "exact": exact,
            "xml_ok": xml_ok,
        })

    # Write inference log.
    if output_dir:
        ar_dir = os.path.join(output_dir, "ar_inferences")
        os.makedirs(ar_dir, exist_ok=True)
        ar_path = os.path.join(ar_dir, f"epoch_{epoch}.jsonl")
        with open(ar_path, "w") as f:
            for inf in inferences:
                f.write(json.dumps(inf) + "\n")

    del cpu_model
    return exact_count, xml_ok_count, total


def interrupt_filename():
    """Timestamped interrupt checkpoint filename (never overwrites previous)."""
    return f"interrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, output_dir, filename, epoch_complete=True, epoch_step=0, epoch_seed=None, stage=1, stage_good_epochs=0):
    path = os.path.join(output_dir, filename)
    torch.save({
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "epoch_step": epoch_step,
        "epoch_seed": epoch_seed,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "stage": stage,
        "stage_good_epochs": stage_good_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train transmutation model (haiku-first)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--tokenizer", default="models/tokenizer.model", help="Tokenizer model path")
    parser.add_argument("--output-dir", default="models", help="Output directory")

    # Model.
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--override-lr", type=float, default=None,
                        help="Force this LR on resume (resets scheduler)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-patience", type=int, default=2,
                        help="ReduceLROnPlateau patience (epochs without improvement before LR decay)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--ar-eval-samples", type=int, default=50,
                        help="Number of samples for autoregressive eval per epoch")

    # Loss weighting.
    parser.add_argument("--content-weight", type=float, default=10.0,
                        help="Weight multiplier for content tokens (numbers, strings). "
                             "Structural XML tokens (0..structural-max-id) stay at 1.0.")
    parser.add_argument("--structural-max-id", type=int, default=15,
                        help="Token IDs 0..N are considered structural (XML tags, special tokens)")
    parser.add_argument("--token-noise", type=float, default=0.15,
                        help="Probability of replacing a content token in decoder input with a random content token (0=off)")
    parser.add_argument("--professor-forcing", action="store_true", default=True,
                        help="Use model predictions instead of random tokens for noise (requires --token-noise > 0)")

    # Haiku augmentation pipeline.
    parser.add_argument("--augment-bin", type=str, default="/app/augment",
                        help="Path to Go augment binary")
    parser.add_argument("--stage", type=int, default=1,
                        help="Starting curriculum stage (1-5)")
    parser.add_argument("--max-stage", type=int, default=5,
                        help="Maximum curriculum stage (auto-advance stops here)")
    parser.add_argument("--stage-advance-ar", type=float, default=0.7,
                        help="AR exact rate threshold to advance stage (0-1)")
    parser.add_argument("--stage-patience", type=int, default=2,
                        help="Consecutive epochs above threshold before advancing")

    parser.add_argument("--max-src-len", type=int, default=1536)
    parser.add_argument("--max-tgt-len", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)

    # Checkpointing.
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
