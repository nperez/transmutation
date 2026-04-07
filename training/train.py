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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
import random
import re
import signal
import threading
import unicodedata
import subprocess
import time
import xml.etree.ElementTree as ET

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.amp import GradScaler, autocast

# Module-level ref for signal handler to kill child subprocesses
_active_child = None

from dataset import create_dataloader
from infer import greedy_decode, batched_greedy_decode
from model import build_model


# ── Curriculum stages ────────────────────────────────────────────────────────
#
# Complexity-stratified curriculum: overfit on clean structure, then generalize.
#
# Phase A (stages 1-3): Overfit on clean JSON→XML. Advance on val_exact > 90%.
#   - Complexity ramps from simple flat → full structural vocabulary
#   - No corruption, no truncation. Compact OK (not structural complexity).
#   - PF base is low (overfit mode), ramps on plateau via pf_boost.
#
# Phase B (stage 4): AR gate on clean data. Advance on AR exact > 90%.
#   - Full complexity, clean. Model must GENERATE correct XML end-to-end.
#   - PF boost kicks in hard here (teacher-forced mastery but AR gap).
#
# Phase C (stages 5-6): Generalize with augmentation. Advance on AR exact > 90%.
#   - Progressive corruption, truncation, special chars.
#   - PF driven by plateau detection.

HAIKU_STAGES = {
    # Stage 1: Clean, complexity<=5, fp16 warmup. Advance on val_loss < 2.0.
    1: {"type": "all", "max_complexity": 5, "aug_ratio": 0, "special_prob": 0.0, "corrupt_pct": 0, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 0, "truncate_pct": 0, "shorten_pct": 30},
    # Stage 2: Clean + special chars (CDATA training), full complexity. Advance on val_exact >= 90%.
    2: {"type": "all", "max_complexity": 10, "aug_ratio": 0, "special_prob": 0.10, "corrupt_pct": 0, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 0, "truncate_pct": 0, "shorten_pct": 30},
    # Stage 3: Light augmentation. Advance on val_exact >= 89%.
    3: {"type": "all", "max_complexity": 10, "aug_ratio": 1, "special_prob": 0.05, "corrupt_pct": 2, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 20, "truncate_pct": 2, "shorten_pct": 30},
    # Stages 4-6: Heavy augmentation. AR eval active. Advance on ar_exact >= 89%.
    4: {"type": "all", "max_complexity": 10, "aug_ratio": 2, "special_prob": 0.15, "corrupt_pct": 5, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 30, "truncate_pct": 5, "shorten_pct": 30},
    5: {"type": "all", "max_complexity": 10, "aug_ratio": 3, "special_prob": 0.25, "corrupt_pct": 10, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 40, "truncate_pct": 10, "shorten_pct": 30},
    6: {"type": "all", "max_complexity": 10, "aug_ratio": 3, "special_prob": 0.35, "corrupt_pct": 15, "compact_pct": 50, "sample_pct": 50, "dict_word_pct": 0, "drop_memory_pct": 50, "truncate_pct": 20, "shorten_pct": 30},
}

# Professor forcing base rate per stage. Low during overfit (1-3), moderate for AR (4-6).
# pf_boost adds on top when the stage metric plateaus (sawtooth pattern, resets on advance).
PF_NOISE_SCHEDULE = {1: 0.05, 2: 0.05, 3: 0.10, 4: 1.00, 5: 1.00, 6: 1.00}

# Stage advance thresholds. Stages 1-3 use val_exact, stages 4-6 use AR exact.
STAGE_ADVANCE_THRESHOLD = 0.90  # 90% mastery — stage 2 special chars ensure CDATA is well-exercised


def compute_cw_boost(prev_val, curr_val, rate_threshold=0.10, max_boost=1.0):
    """Compute content weight boost from val loss improvement rate.

    Returns boost in [0, max_boost]. High improvement → 0 boost. Stalled → full boost.
    Sawtooth pattern: accumulate boosts within a stage, reset on stage advance.
    """
    if prev_val <= 0:
        return 0.0
    improvement = (prev_val - curr_val) / prev_val
    return max(0.0, min(max_boost, (1.0 - improvement / rate_threshold) * max_boost))

# Validation budget by training stage — small early (fast epochs), full later (precise eval).
# Full val set is generated once at max stage; this caps how many batches we actually run.
# None = use full val set.
VAL_MAX_BATCHES = {1: 50, 2: 1667, 3: 1667, 4: None, 5: None, 6: None}


def generate_haiku_data(augment_bin, data_dir, split, stage, seed, dict_word_pct_override=None, tokenizer_path=None):
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
        "-compact-pct", str(params.get("compact_pct", 0)),
        "-dict-word-pct", str(dict_word_pct_override if dict_word_pct_override is not None else params.get("dict_word_pct", 50)),
        "-drop-memory-pct", str(params.get("drop_memory_pct", 20)),
        "-truncate-pct", str(params.get("truncate_pct", 0)),
        "-shorten-pct", str(params.get("shorten_pct", 0)),
        "-min-chars", str(params.get("min_chars", 0)),
        "-max-complexity", str(params.get("max_complexity", 10)),
        "-type", str(params.get("type", "all")),
        "-seed", str(seed),
    ]
    if params.get("stratify", False):
        cmd.append("-stratify")
    if tokenizer_path:
        cmd.extend(["-tokenizer", tokenizer_path])
    if split == "val":
        cmd.append("-val")

    out_dir = os.path.join(data_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "haiku_augmented.jsonl")
    marker_path = os.path.join(out_dir, ".gen_marker")
    marker_key = f"stage={stage} seed={seed}"

    # Idempotent: marker matches = complete, skip.
    if os.path.exists(marker_path) and os.path.exists(out_path):
        with open(marker_path) as f:
            if f.read().strip() == marker_key:
                print(f"Reusing existing {split} data ({marker_key})")
                return

    # Count existing lines — if file exists without marker, it's a partial/interrupted generation.
    # Binary is deterministic (same seed), so existing lines are valid prefix. Resume from there.
    existing_lines = 0
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing_lines = sum(1 for _ in f)

    mc = params.get("min_chars", 0)
    label = f"stage {stage} aug={params['aug_ratio']} sp={params['special_prob']} cor={params['corrupt_pct']}% cmp={params.get('compact_pct', 0)}%"
    if mc > 0:
        label += f" min={mc}ch"
    if existing_lines > 0:
        print(f"Resuming {split} data from line {existing_lines} ({label}, seed {seed})...")
    else:
        print(f"Generating {split} data ({label}, seed {seed})...")

    t0 = time.time()
    global _active_child
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _active_child = proc
    with open(out_path, "a" if existing_lines > 0 else "w") as f:
        for i, line in enumerate(proc.stdout):
            if i >= existing_lines:
                f.write(line)
    stderr = proc.stderr.read()
    proc.wait()
    _active_child = None
    if stderr:
        for line in stderr.strip().split("\n"):
            print(f"  {line}")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Marker written IMMEDIATELY after successful generation.
    with open(marker_path, "w") as f:
        f.write(marker_key)
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

    # GPU-accelerated token decoder for val CER/WER.
    gpu_decoder = GPUTokenDecoder(sp, device=device)
    print(f"GPU token decoder: {len(gpu_decoder.piece_bytes)} bytes, {vocab_size} pieces")

    # Build model.
    model = build_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_state=args.d_state,
        expand=2,
        headdim=args.headdim,
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
    # Model returns raw logits (no copy mechanism), use CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    print(f"Content weight: adaptive sawtooth 1.0→{args.content_weight}x (resets on stage advance, structural tokens 0-{args.structural_max_id} at {args.structural_weight}x)")
    if args.professor_forcing:
        iter_str = f", {args.pf_iterations} iterations" if args.pf_iterations > 1 else ""
        print(f"Professor forcing: ON (schedule: {' → '.join(f's{s}={n:.3f}' for s, n in sorted(PF_NOISE_SCHEDULE.items()))}{iter_str})")
    if args.mrt_weight > 0:
        cdata_str = f", CDATA weight={args.cdata_weight}x" if args.cdata_weight > 0 else ""
        print(f"Online MRT: weight={args.mrt_weight}x on near-miss samples, threshold={args.mrt_threshold:.0%}{cdata_str}")
    if args.ar_train_frac > 0:
        print(f"AR training: {args.ar_train_frac:.0%} of batches use full AR decode as input (stage 4+)")
        model._ar_fixed_src_len = args.max_src_len  # fixed size for CUDA graph cache

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
    resume_training_done = False
    resume_val_state = None
    resume_train_loss = 0.0
    current_stage = args.stage
    stage_good_epochs = 0  # consecutive epochs above stage-advance threshold
    cw_boost = 0.0  # accumulated content weight boost (sawtooth: resets on stage advance)
    pf_boost = 0.0  # professor forcing boost (sawtooth: resets on stage advance)
    sw_boost = 0.0  # structural weight boost (sawtooth: resets on stage advance)
    prev_val_loss = None  # for computing improvement rate

    # Resume from checkpoint.
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            if all("copy_gate" in k for k in missing):
                # Old checkpoint without copy gate — initialize to near-zero copy.
                nn.init.zeros_(model.copy_gate.weight)
                nn.init.constant_(model.copy_gate.bias, -5.0)
                print(f"  Initialized copy_gate from scratch (old checkpoint, p_copy≈0.007)")
            else:
                raise RuntimeError(f"Unexpected missing keys: {missing}")
        if unexpected:
            print(f"  Ignoring unexpected checkpoint keys: {unexpected}")
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
        best_ar_exact = ckpt.get("best_ar_exact", 0)
        current_stage = ckpt.get("stage", args.stage)
        if args.force_stage and args.force_stage != current_stage:
            print(f"  Force-advancing stage: {current_stage} → {args.force_stage}")
            current_stage = args.force_stage
            # Reset everything as if normal stage advance happened.
            stage_good_epochs = 0
            cw_boost = 0.0
            pf_boost = 0.0
            sw_boost = 0.0
            prev_val_loss = None
            restart_lr = args.lr / 2
            for pg in optimizer.param_groups:
                pg['lr'] = restart_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=args.lr_patience,
            )
            print(f"  LR reset to {restart_lr:.1e}, scheduler reset, boosts zeroed")
        else:
            stage_good_epochs = ckpt.get("stage_good_epochs", 0)
            cw_boost = ckpt.get("cw_boost", 0.0)
            pf_boost = ckpt.get("pf_boost", 0.0)
            sw_boost = ckpt.get("sw_boost", 0.0)
            prev_val_loss = ckpt.get("prev_val_loss")
        if not completed_epoch:
            resume_epoch_seed = ckpt.get("epoch_seed")
            resume_epoch_step = ckpt.get("epoch_step", 0)
            resume_training_done = ckpt.get("training_done", False)
            resume_val_state = ckpt.get("val_state")
            resume_train_loss = ckpt.get("train_loss", 0.0)
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

    # Safety: refuse to train if checkpoint state is inconsistent.
    existing = [f for f in os.listdir(args.output_dir)
                if f.endswith(".pt") and f != "tokenizer.model"]
    if existing:
        if start_epoch == 1 and not args.resume:
            # Case 1: no --resume but checkpoints exist (fresh start would overwrite).
            print("ERROR: found existing checkpoints but no --resume flag:")
            for f in sorted(existing):
                print(f"  {f}")
            print("Pass --resume <checkpoint> to continue, or move/delete old checkpoints.")
            raise SystemExit(1)
        if args.resume:
            # Case 2: --resume points to a stale checkpoint when newer ones exist.
            # Find the newest checkpoint by modification time.
            newest = max(
                (os.path.join(args.output_dir, f) for f in existing),
                key=os.path.getmtime,
            )
            resume_mtime = os.path.getmtime(args.resume)
            newest_mtime = os.path.getmtime(newest)
            if newest_mtime > resume_mtime + 60 and not args.force_resume:  # 60s tolerance
                newest_name = os.path.basename(newest)
                resume_name = os.path.basename(args.resume)
                print(f"ERROR: --resume {resume_name} is stale — newer checkpoint exists:")
                print(f"  {resume_name}: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resume_mtime))}")
                print(f"  {newest_name}: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_mtime))}")
                print(f"Use --resume {os.path.join(args.output_dir, newest_name)} instead, or --force-resume to override.")
                raise SystemExit(1)

    # Signal handling: preserve state on any interruption.
    # SIGUSR1 = save checkpoint, keep training
    # SIGTERM/SIGINT/SIGHUP/SIGUSR2 = save checkpoint, exit cleanly
    sig_state = {"save": False, "stop": False}
    def handle_save(signum, _frame):
        sig_state["save"] = True
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint, continuing <<<")
    def handle_stop(signum, _frame):
        sig_state["save"] = True
        sig_state["stop"] = True
        # Kill any running subprocess (augment binary) so we don't block on stop
        if _active_child and _active_child.poll() is None:
            _active_child.terminate()
        print(f"\n>>> {signal.Signals(signum).name} received — saving checkpoint and exiting <<<")
    signal.signal(signal.SIGUSR1, handle_save)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR2):
        signal.signal(sig, handle_stop)
    print("Signals: USR1=checkpoint, TERM/INT/HUP/USR2=checkpoint+exit")

    # atexit safety net: save state if process exits unexpectedly (e.g. unhandled exception).
    atexit_state = {"epoch": 0, "step": 0, "epoch_seed": None, "active": False, "training_done": False, "val_state": None, "train_loss": 0.0}
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
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            training_done=atexit_state["training_done"],
                            val_state=atexit_state.get("val_state"),
                            train_loss=atexit_state.get("train_loss", 0.0),
                            cw_boost=cw_boost, pf_boost=pf_boost, sw_boost=sw_boost, prev_val_loss=prev_val_loss,
                            best_ar_exact=best_ar_exact)
            print(f">>> atexit: saved at epoch {atexit_state['epoch']} batch {actual_step} (training_done={atexit_state['training_done']}) <<<")
        except Exception as e:
            print(f">>> atexit: FAILED to save checkpoint: {e} <<<")
    atexit.register(atexit_save)

    # Generate validation data matching the current stage's complexity.
    # Regenerated on stage advance so val_exact measures mastery of the current level.
    VAL_SEED = 7777777
    last_val_stage = current_stage if args.resume else None  # on resume, val data matches current stage
    prev_stage_metric = None  # for computing pf_boost improvement rate

    def regenerate_val_if_needed():
        nonlocal last_val_stage
        if last_val_stage != current_stage:
            generate_haiku_data(args.augment_bin, args.data_dir, "val", current_stage, VAL_SEED, args.dict_word_pct, args.tokenizer)
            print(f"  Val set regenerated for stage {current_stage}")
            last_val_stage = current_stage

    regenerate_val_if_needed()
    print(f"Training stage: {current_stage} (advance at {STAGE_ADVANCE_THRESHOLD:.0%} mastery, max={args.max_stage})")
    print(f"  Stages 1-3: advance on val_exact, Stages 4+: advance on AR exact")

    print(f"\nTraining for {args.epochs} epochs (ReduceLROnPlateau, patience={args.lr_patience})")
    print(f"Grad accumulation: {args.grad_accum}, effective batch: {args.batch_size * args.grad_accum}")
    print()

    _val_dataset = None  # cached val dataset — loaded once, reused across epochs
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = epoch * 1000 + 42
        start_index = 0

        # If resuming an epoch where training already finished, skip to validation.
        if epoch == start_epoch and resume_training_done:
            if resume_epoch_seed is not None:
                epoch_seed = resume_epoch_seed
            print(f"Resuming epoch {epoch} post-training (skipping to validation)")
            atexit_state.update(epoch=epoch, epoch_seed=epoch_seed, step=0, active=True, training_done=True)
            avg_train_loss = resume_train_loss
            # Jump straight to validation (below the training block).
        else:
            # Generate fresh training data for this epoch (val is fixed).
            if epoch == start_epoch and resume_epoch_step > 0:
                if resume_epoch_seed is not None:
                    epoch_seed = resume_epoch_seed
                start_index = resume_epoch_step
                print(f"Resuming epoch {epoch} from batch {start_index} (seed={epoch_seed})")
            generate_haiku_data(args.augment_bin, args.data_dir, "train", current_stage, epoch_seed, args.dict_word_pct, args.tokenizer)

            train_loader, epoch_seed, _ = create_dataloader(
                data_dir=train_data_dir, shuffle=True,
                epoch_seed=epoch_seed, start_index=start_index * args.batch_size,
                max_samples=args.max_epoch_samples, **dl_kwargs,
            )
            n_dataset = len(train_loader.dataset)
            n_sampled = len(train_loader) * args.batch_size
            print(f"  Epoch {epoch}: {n_dataset} in dataset, {n_sampled} sampled (max_epoch={args.max_epoch_samples}), {len(train_loader)} batches", flush=True)
            atexit_state.update(epoch=epoch, epoch_seed=epoch_seed, step=start_index, active=True, training_done=False)

            # ── Pre-decode AR batches for the epoch (chunked, amortized) ──
            ar_decoded_cache = {}
            do_ar_training = (args.ar_train_frac > 0 and current_stage >= 4)
            ar_inline = do_ar_training  # always inline — no pre-decode
            if do_ar_training and not ar_inline:
                from ar_kernel import ar_decode_fused
                model.eval()
                ar_rng = torch.Generator().manual_seed(epoch_seed + 999)
                ar_step_indices = []
                for s in range(start_index, len(train_loader)):
                    if torch.rand(1, generator=ar_rng).item() < args.ar_train_frac:
                        ar_step_indices.append(s)
                if ar_step_indices:
                    AR_BATCH = 30  # decode this many samples at once (10 training batches)
                    ar_batches = {}
                    for s, batch in enumerate(train_loader):
                        if start_index + s in ar_step_indices:
                            ar_batches[start_index + s] = (
                                batch["src_ids"].to(device),
                                batch["src_key_padding_mask"].to(device),
                                batch["tgt_input"].shape[1],
                            )
                    keys = sorted(ar_batches.keys())
                    batches_per_chunk = AR_BATCH // args.batch_size
                    t0_ar = time.time()
                    n_chunks = 0
                    for chunk_start in range(0, len(keys), batches_per_chunk):
                        chunk_keys = keys[chunk_start:chunk_start + batches_per_chunk]
                        src_list = [ar_batches[k][0] for k in chunk_keys]
                        mask_list = [ar_batches[k][1] for k in chunk_keys]
                        max_src = max(s.shape[1] for s in src_list)
                        max_tgt = max(ar_batches[k][2] for k in chunk_keys)
                        cat_src = torch.full((len(chunk_keys) * args.batch_size, max_src),
                                            sp.pad_id(), dtype=torch.long, device=device)
                        cat_mask = torch.ones(cat_src.shape[0], max_src, dtype=torch.bool, device=device)
                        for i, (s, m) in enumerate(zip(src_list, mask_list)):
                            bs = s.shape[0]
                            sl = s.shape[1]
                            cat_src[i*bs:(i+1)*bs, :sl] = s
                            cat_mask[i*bs:(i+1)*bs, :sl] = m
                        with torch.no_grad():
                            ar_ids = ar_decode_fused(
                                model, cat_src, sp,
                                max_len=max_tgt, src_key_padding_mask=cat_mask,
                            )
                        for i, k in enumerate(chunk_keys):
                            bs = ar_batches[k][0].shape[0]
                            ar_decoded_cache[k] = ar_ids[i*bs:(i+1)*bs]
                        n_chunks += 1
                    ar_elapsed = time.time() - t0_ar
                    print(f"  AR pre-decode: {len(keys)} batches ({len(keys)*args.batch_size} samples) "
                          f"in {ar_elapsed:.1f}s", flush=True)
                    del ar_batches
                model.train()

            # Train. Accumulators on GPU — zero sync in training loop.
            model.train()
            gpu_train_loss = torch.zeros(1, dtype=torch.float64, device=device)
            gpu_train_tokens = torch.zeros(1, dtype=torch.long, device=device)
            optimizer.zero_grad()

            n_batches = len(train_loader)
            epoch_start_time = time.time()
            last_log_time = epoch_start_time
            _ar_did_decode = False
            for step, batch in enumerate(train_loader):
                atexit_state["step"] = start_index + step
                src = batch["src_ids"].to(device)
                tgt_in = batch["tgt_input"].to(device)
                tgt_labels = batch["tgt_labels"].to(device)
                src_mask = batch["src_key_padding_mask"].to(device)

                # AR training: fanned-out parallel decode — full GPU utilization.
                if ar_inline and torch.rand(1).item() < args.ar_train_frac:
                    from ar_parallel import ar_decode_parallel
                    ar_ids = ar_decode_parallel(
                        model, src, sp,
                        max_len=tgt_in.shape[1],
                        src_key_padding_mask=src_mask,
                    )
                    tgt_len = tgt_in.shape[1]
                    for i, ids in enumerate(ar_ids):
                        ar_len = min(len(ids), tgt_len - 1)
                        tgt_in[i, 1:1 + ar_len] = torch.tensor(ids[:ar_len], device=device)
                    _ar_did_decode = True
                else:
                    _ar_did_decode = False
                    effective_noise = min(PF_NOISE_SCHEDULE.get(current_stage, args.token_noise) + pf_boost, 1.0)
                    if effective_noise > 0:
                        if args.professor_forcing:
                            # Iterative PF: N passes, each predicting from increasingly
                            # corrupted context. Approximates AR error cascading.
                            for _pf_iter in range(args.pf_iterations):
                                with torch.no_grad():
                                    with autocast("cuda", enabled=args.fp16):
                                        pf_logits = model(src, tgt_in, src_mask)
                                    pred_ids = pf_logits.argmax(dim=-1)
                                    replacement_ids = torch.cat(
                                        [tgt_in[:, :1], pred_ids[:, :-1]], dim=1,
                                    )
                                tgt_in = corrupt_content_tokens(
                                    tgt_in, effective_noise, args.structural_max_id,
                                    vocab_size, pad_id, replacement_ids=replacement_ids,
                                )
                        else:
                            tgt_in = corrupt_content_tokens(
                                tgt_in, effective_noise, args.structural_max_id, vocab_size, pad_id,
                            )

                with autocast("cuda", enabled=args.fp16):
                    logits = model(src, tgt_in, src_mask)
                    cw = min(args.content_weight, 1.0 + cw_boost)
                    sw = args.structural_weight + sw_boost
                    if cw > 1.0 or sw > 1.0:
                        loss = weighted_content_loss(
                            logits, tgt_labels, vocab_size,
                            cw, args.structural_max_id,
                            structural_weight=sw,
                        )
                    elif args.mrt_weight > 0:
                        loss = mrt_error_weighted_loss(
                            logits, tgt_labels, vocab_size,
                            args.mrt_weight, args.mrt_threshold,
                            cdata_weight=args.cdata_weight,
                        )
                    else:
                        loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()

                n_tokens = (tgt_labels != -100).sum()
                gpu_train_loss += loss.detach().double() * args.grad_accum * n_tokens
                gpu_train_tokens += n_tokens

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

                # Display + NaN check every grad_accum steps (one sync per optimizer step).
                if (start_index + step + 1) % args.grad_accum == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    loss_val = loss.item() * args.grad_accum
                    if loss_val != loss_val:  # NaN
                        print(f"  NaN loss detected at step {step}, skipping", flush=True)
                    now = time.time()
                    if now - last_log_time >= 10:
                        pct = 100 * (step + 1) / n_batches
                        elapsed = now - epoch_start_time
                        rate = (step + 1) / max(elapsed, 1)
                        eta_m = (n_batches - step - 1) / max(rate, 0.01) / 60
                        ar_tag = " [AR]" if _ar_did_decode else ""
                        print(f"Epoch {epoch} train: {pct:5.1f}% | {step+1}/{n_batches} | loss={loss_val:.4f} | lr={cur_lr:.2e} | {rate:.1f} it/s | ETA {eta_m:.0f}m{ar_tag}", flush=True)
                        last_log_time = now

                # Handle signal-triggered checkpoint.
                if sig_state["save"]:
                    sig_state["save"] = False
                    actual_step = start_index + step + 1
                    fname = interrupt_filename()
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                                    args.output_dir, fname, epoch_complete=False,
                                    epoch_step=actual_step, epoch_seed=epoch_seed,
                                    stage=current_stage, stage_good_epochs=stage_good_epochs,
                                    cw_boost=cw_boost, prev_val_loss=prev_val_loss,
                                    best_ar_exact=best_ar_exact)
                    print(f"\n>>> Checkpoint saved ({fname}): epoch {epoch} batch {actual_step} stage={current_stage} <<<")
                    if sig_state["stop"]:
                        print("Exiting cleanly.")
                        return model

            train_loss = gpu_train_loss.item()
            train_tokens = gpu_train_tokens.item()
            avg_train_loss = train_loss / max(train_tokens, 1)
            atexit_state["training_done"] = True
            atexit_state["train_loss"] = avg_train_loss

        # Validate (fixed held-out val set, generated once before training).
        atexit_state["training_done"] = True
        atexit_state["val_state"] = None
        val_state_for_resume = None
        if epoch == start_epoch and resume_val_state is not None:
            val_state_for_resume = resume_val_state
            resume_val_state = None  # only use once
        val_start_index = val_state_for_resume["batch"] * args.batch_size if val_state_for_resume else 0
        # Cache val dataset across epochs — reuse until stage changes
        val_loader, _, _val_dataset = create_dataloader(
            data_dir=val_data_dir, shuffle=True, start_index=val_start_index,
            max_samples=args.max_epoch_samples, epoch_seed=VAL_SEED,
            dataset=_val_dataset, **dl_kwargs,
        )
        val_max_batches = VAL_MAX_BATCHES.get(current_stage)
        val_loss, val_tokens, val_exact, val_cer, total_wer_chars, wer_args = validate(
            model, val_loader, criterion, vocab_size, device, args.fp16,
            sp=sp, gpu_decoder=gpu_decoder, sig_state=sig_state,
            atexit_state=atexit_state,
            resume_val_state=val_state_for_resume,
            max_batches=val_max_batches, output_dir=args.output_dir)
        avg_val_loss = val_loss / max(val_tokens, 1)

        # Launch WER in background thread (CPU) so AR eval can use GPU concurrently.
        wer_result = [0]
        def _compute_wer_bg():
            if wer_args:
                n_workers = min(os.cpu_count() or 4, len(wer_args))
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    wer_result[0] = sum(pool.map(_wer_worker, wer_args, chunksize=64))
                print(f"  WER done ({len(wer_args)} pairs, {os.cpu_count()} cores).", flush=True)
        wer_thread = threading.Thread(target=_compute_wer_bg, daemon=True)
        wer_thread.start()

        # Check for stop signal after validation (may have aborted early).
        if sig_state["stop"]:
            wer_thread.join()
            fname = interrupt_filename()
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                            args.output_dir, fname, epoch_complete=False,
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            training_done=True,
                            val_state=atexit_state.get("val_state"),
                            train_loss=avg_train_loss,
                            cw_boost=cw_boost, pf_boost=pf_boost, sw_boost=sw_boost, prev_val_loss=prev_val_loss,
                            best_ar_exact=best_ar_exact)
            print(f"\n>>> Checkpoint saved ({fname}): epoch {epoch} (training_done, val_state saved) <<<")
            print("Exiting cleanly.")
            return model

        atexit_state["val_state"] = None  # validation complete, clear partial state

        # Autoregressive eval: always run. Stage gates 1-3 use val_exact, 4+ use ar_rate.
        run_ar = True
        if run_ar:
            try:
                ar_exact, ar_semantic, ar_xml_ok, ar_total, ar_cer, ar_wer, ar_buckets = autoregressive_eval(
                    model, sp, n_samples=args.ar_eval_samples,
                    augment_bin=args.augment_bin, data_dir=args.data_dir,
                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len,
                    output_dir=args.output_dir, epoch=epoch, stage=current_stage,
                    device=device, gpu_decoder=gpu_decoder,
                )
            except NotImplementedError as e:
                ar_exact, ar_semantic, ar_xml_ok, ar_total, ar_cer, ar_wer, ar_buckets = 0, 0, 0, args.ar_eval_samples, 1.0, 1.0, {}
                print(f"AR eval skipped: {e}", flush=True)
        else:
            ar_exact, ar_xml_ok, ar_total, ar_cer, ar_wer = 0, 0, args.ar_eval_samples, 1.0, 1.0

        # Join WER thread (was running on CPU in parallel with AR eval on GPU).
        wer_thread.join()
        val_wer = wer_result[0] / max(total_wer_chars, 1)

        # Stage metrics.
        ar_rate = ar_exact / max(ar_total, 1)
        # Stage 1: fp16 stability (val_loss gate).
        # Stages 2-3: val_exact mastery (AR eval not active until stage 4).
        # Stages 4+: AR exact robustness under augmentation.
        if current_stage == 1:
            # Stage 1→2: fp16 stability gate. Just need loss low enough.
            stage_metric = 1.0 if avg_val_loss < 2.0 else 0.0
            stage_metric_name = "val_loss<2.0"
        elif current_stage <= 3:
            # Stages 2-3: val_exact mastery gate. AR eval not yet active.
            stage_metric = val_exact
            stage_metric_name = "val_exact"
        else:
            # Stages 4+: AR robustness under augmentation.
            stage_metric = ar_rate
            stage_metric_name = "ar_exact"

        # LR scheduler — always use val_loss (deterministic, smooth).
        # AR rate is too noisy at 250 samples to drive LR — it triggers false plateaus.
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < cur_lr:
            print(f"  LR reduced: {cur_lr:.2e} -> {new_lr:.2e}")

        # Auto-advance curriculum stage on mastery.
        if current_stage < args.max_stage:
            if stage_metric >= STAGE_ADVANCE_THRESHOLD:
                stage_good_epochs += 1
            else:
                stage_good_epochs = 0
            if stage_good_epochs >= args.stage_patience:
                current_stage += 1
                stage_good_epochs = 0
                new_params = HAIKU_STAGES[current_stage]
                print(f"  >>> Stage advanced to {current_stage} ({stage_metric_name}={stage_metric:.0%} for {args.stage_patience} epochs)")
                cw_boost = 0.0  # sawtooth reset
                pf_boost = 0.0  # sawtooth reset
                sw_boost = 0.0  # sawtooth reset
                prev_val_loss = None
                prev_stage_metric = None
                # Regenerate val set for new complexity level.
                regenerate_val_if_needed()
                _val_dataset = None  # invalidate cached val dataset
                # LR warm restart.
                restart_lr = args.lr / 2
                for pg in optimizer.param_groups:
                    pg['lr'] = restart_lr
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5,
                    patience=args.lr_patience, min_lr=1e-6,
                )
                print(f"      complexity<={new_params.get('max_complexity', 10)} aug={new_params['aug_ratio']} cor={new_params['corrupt_pct']}% pf/cw reset lr→{restart_lr:.1e}")

        # PF boost: ramp when stage metric isn't improving fast enough. Reset on advance.
        # Checked every epoch. Threshold 10% — anything slower gets a boost.
        effective_pf = min(PF_NOISE_SCHEDULE.get(current_stage, args.token_noise) + pf_boost, 1.0)
        if prev_stage_metric is not None:
            if prev_stage_metric > 0:
                improvement = (stage_metric - prev_stage_metric) / max(prev_stage_metric, 0.01)
            else:
                improvement = stage_metric
            boost = compute_cw_boost(1.0, 1.0 - improvement, rate_threshold=0.10, max_boost=0.10)
            old_pf = effective_pf
            pf_boost = min(pf_boost + boost, 0.50)
            effective_pf = min(PF_NOISE_SCHEDULE.get(current_stage, args.token_noise) + pf_boost, 1.0)
            if effective_pf > old_pf + 0.005:
                print(f"  PF boost: {old_pf:.3f} -> {effective_pf:.3f} ({stage_metric_name} improvement {improvement:.1%})")
        prev_stage_metric = stage_metric

        # Structural weight boost: DISABLED. Re-enable by setting SW_CAP > 0.
        # In run5 SW ramped 0→8 during stages 4-6 but unclear if it helped vs natural adaptation.
        SW_CAP = 0.0
        effective_sw = args.structural_weight + sw_boost
        sw_at_cap = True  # disabled — always "at cap" so CW section is also skipped
        if SW_CAP > 0 and not (effective_sw >= SW_CAP) and prev_stage_metric is not None:
            if prev_stage_metric > 0:
                sw_improvement = (stage_metric - prev_stage_metric) / max(prev_stage_metric, 0.01)
            else:
                sw_improvement = stage_metric
            sw_inc = compute_cw_boost(1.0, 1.0 - sw_improvement, rate_threshold=0.05, max_boost=2.0)
            old_sw = effective_sw
            sw_boost = min(sw_boost + sw_inc, SW_CAP - args.structural_weight)
            effective_sw = args.structural_weight + sw_boost
            if effective_sw > old_sw + 0.1:
                print(f"  SW boost: {old_sw:.1f} -> {effective_sw:.1f}")

        # Content weight boost: DISABLED. Re-enable by setting SW_CAP > 0 (activates after SW caps).
        # Was broken for most of run5 (content_weight=1.0 clamped boost to no-op).
        cw_eval_interval = 2
        if False and sw_at_cap and prev_stage_metric is not None and epoch % cw_eval_interval == 0:
            # CW boost driven by AR metric plateau (same signal as PF/SW).
            if prev_stage_metric > 0:
                cw_improvement = (stage_metric - prev_stage_metric) / max(prev_stage_metric, 0.01)
            else:
                cw_improvement = stage_metric
            boost = compute_cw_boost(1.0, 1.0 - cw_improvement, rate_threshold=0.05, max_boost=0.5)
            old_cw = min(args.content_weight, 1.0 + cw_boost)
            cw_boost += boost
            new_cw = min(args.content_weight, 1.0 + cw_boost)
            if new_cw > old_cw + 0.01:
                print(f"  CW boost (AR plateau): {old_cw:.2f} -> {new_cw:.2f}")
        elif prev_val_loss is not None and epoch % cw_eval_interval == 0:
            # Fallback: CW driven by val_loss stall (pre-cap behavior).
            boost = compute_cw_boost(prev_val_loss, avg_val_loss)
            old_cw = min(args.content_weight, 1.0 + cw_boost)
            cw_boost += boost
            new_cw = min(args.content_weight, 1.0 + cw_boost)
            if new_cw > old_cw + 0.01:
                print(f"  Content weight: {old_cw:.2f} -> {new_cw:.2f} (boost +{boost:.2f}, improvement {(prev_val_loss - avg_val_loss) / prev_val_loss:.1%} over {cw_eval_interval} epochs)")
            prev_val_loss = avg_val_loss
        elif prev_val_loss is None:
            prev_val_loss = avg_val_loss

        log_entry = {
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_exact_match": val_exact,
            "ar_exact": ar_exact,
            "ar_semantic": ar_semantic,
            "ar_xml_ok": ar_xml_ok,
            "ar_total": ar_total,
            "val_cer": round(val_cer, 6),
            "val_wer": round(val_wer, 6),
            "ar_cer": round(ar_cer, 6),
            "ar_wer": round(ar_wer, 6),
            "ar_buckets": ar_buckets,
            "lr": new_lr,
            "content_weight": min(args.content_weight, 1.0 + cw_boost),
            "pf_rate": round(effective_pf, 4),
            "global_step": global_step,
        }
        log_entries.append(log_entry)

        print(f"Epoch {epoch}: stage={current_stage} train={avg_train_loss:.4f} val={avg_val_loss:.4f} val_exact={val_exact:.2%} vCER={val_cer:.2%} vWER={val_wer:.2%} ar={ar_exact}/{ar_total}exact {ar_semantic}/{ar_total}sem {ar_xml_ok}/{ar_total}xml CER={ar_cer:.2%} WER={ar_wer:.2%} lr={new_lr:.2e} pf={effective_pf:.3f} sw={effective_sw:.1f}")

        # Save best checkpoints — two tracks:
        #   best.pt    — best val_loss (teacher-forced mastery, any stage)
        #   best_ar.pt — best AR exact (generation quality, stages 4+ only)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, "best.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            cw_boost=cw_boost, pf_boost=pf_boost, sw_boost=sw_boost, prev_val_loss=prev_val_loss,
                            best_ar_exact=best_ar_exact)
            print(f"  -> New best val (stage {current_stage}: val={avg_val_loss:.4f})")
        if run_ar and (ar_exact > best_ar_exact or (ar_exact == best_ar_exact and avg_val_loss < best_val_loss)):
            best_ar_exact = ar_exact
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, "best_ar.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            cw_boost=cw_boost, pf_boost=pf_boost, sw_boost=sw_boost, prev_val_loss=prev_val_loss,
                            best_ar_exact=best_ar_exact)
            print(f"  -> New best AR (stage {current_stage}: ar={ar_exact}/{ar_total} val={avg_val_loss:.4f})")

        # Save periodic checkpoint.
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, f"epoch_{epoch}.pt",
                            stage=current_stage, stage_good_epochs=stage_good_epochs,
                            cw_boost=cw_boost, pf_boost=pf_boost, sw_boost=sw_boost, prev_val_loss=prev_val_loss,
                            best_ar_exact=best_ar_exact)

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
def validate(model, loader, criterion, vocab_size, device, fp16,
             sp=None, gpu_decoder=None, sig_state=None, atexit_state=None,
             resume_val_state=None, max_batches=None, output_dir=None):
    model.eval()
    # GPU accumulators for loss — zero sync in the loop.
    gpu_total_loss = torch.zeros(1, dtype=torch.float64, device=device)
    gpu_total_tokens = torch.zeros(1, dtype=torch.long, device=device)

    # ── Phase 1: Forward pass + fill GPU buffers. Zero sync in loop. ─────
    # Pre-allocate full pred/tgt buffers on GPU (int16, ~830MB for 135K×1536).
    # No gradients during validation so plenty of VRAM.
    n_val = len(loader) * loader.batch_size  # respects sampler cap, not full dataset
    max_seq = 1536  # matches max_tgt_len in run.sh — covers all padded batch lengths
    torch.cuda.empty_cache()  # free reserved-but-unallocated memory before big allocation
    all_preds = torch.zeros(n_val, max_seq, dtype=torch.int16, device=device)
    all_tgts = torch.full((n_val, max_seq), -100, dtype=torch.int16, device=device)
    all_src_lens = torch.zeros(n_val, dtype=torch.int32, device=device)
    all_sample_losses = torch.zeros(n_val, dtype=torch.float32, device=device)
    sample_idx = 0

    val_start_time = time.time()
    val_last_log = val_start_time
    val_n_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if sig_state and sig_state["stop"]:
            print("\n>>> Stop signal during validation — aborting early <<<")
            break

        src = batch["src_ids"].to(device)
        tgt_in = batch["tgt_input"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)

        with autocast("cuda", enabled=fp16):
            logits = model(src, tgt_in, src_mask)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

        # Accumulate loss on GPU — zero sync.
        n_tokens = (tgt_labels != -100).sum()
        gpu_total_loss += loss.double() * n_tokens
        gpu_total_tokens += n_tokens

        # Per-sample loss for bucketed stats (same logits, no extra forward pass).
        with torch.no_grad():
            per_tok = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1),
                                      ignore_index=-100, reduction='none').reshape(tgt_labels.shape)
            tok_counts = (tgt_labels != -100).sum(dim=1).clamp(min=1).float()
            sample_loss = per_tok.sum(dim=1) / tok_counts

        # Argmax + copy to buffer. No sync, no .item(), no Python bookkeeping.
        preds = logits.argmax(dim=-1)  # (B, S)
        B, S = preds.shape
        all_preds[sample_idx:sample_idx + B, :S] = preds.to(torch.int16)
        all_tgts[sample_idx:sample_idx + B, :S] = tgt_labels.to(torch.int16)
        all_src_lens[sample_idx:sample_idx + B] = (~src_mask).sum(dim=1).to(torch.int32)
        all_sample_losses[sample_idx:sample_idx + B] = sample_loss.float()
        sample_idx += B

        now = time.time()
        if now - val_last_log >= 30:
            pct = 100 * (batch_idx + 1) / val_n_batches
            rate = (batch_idx + 1) / max(now - val_start_time, 1)
            eta_m = (val_n_batches - batch_idx - 1) / max(rate, 0.01) / 60
            print(f"  Validating: {pct:5.1f}% | {batch_idx+1}/{val_n_batches} | {rate:.1f} it/s | ETA {eta_m:.0f}m", flush=True)
            val_last_log = now

        # atexit: save progress so interrupted validation can resume.
        if atexit_state is not None:
            atexit_state["val_state"] = {
                "batch": batch_idx + 1,
                "sample_idx": sample_idx,
            }

    total_samples = sample_idx
    total_loss = gpu_total_loss.item()
    total_tokens = gpu_total_tokens.item()
    all_preds = all_preds[:total_samples]
    all_tgts = all_tgts[:total_samples]

    # ── Phase 2: One Triton kernel for all bookkeeping. ──────────────────
    # Exact match, ref_len, non-exact buffer — 135K programs, one launch.
    max_ne = max(total_samples // 5, 1000)
    gpu_exact_count = torch.zeros(1, dtype=torch.int64, device=device)
    gpu_ref_bytes = torch.zeros(1, dtype=torch.int64, device=device)
    ne_preds_buf = torch.zeros(max_ne, max_seq, dtype=torch.int16, device=device)
    ne_tgts_buf = torch.zeros(max_ne, max_seq, dtype=torch.int16, device=device)
    ne_lens_buf = torch.zeros(max_ne, dtype=torch.int32, device=device)
    ne_count = torch.zeros(1, dtype=torch.int32, device=device)

    piece_lens = gpu_decoder.piece_lengths if gpu_decoder is not None else torch.zeros(
        vocab_size, dtype=torch.long, device=device)

    if total_samples > 0:
        print(f"  Bookkeeping: {total_samples} samples, Triton kernel...", flush=True)
        BLOCK_S = triton.next_power_of_2(max_seq)
        _val_bookkeep_kernel[(total_samples,)](
            all_preds, all_tgts, piece_lens,
            gpu_exact_count, gpu_ref_bytes,
            ne_preds_buf, ne_tgts_buf, ne_lens_buf, ne_count,
            max_seq, max_seq, max_ne,
            BLOCK_S=BLOCK_S,
        )

    # ── Bucketed val stats: collect per-sample data, print after CER. ──
    all_src_lens = all_src_lens[:total_samples]
    all_sample_losses = all_sample_losses[:total_samples]
    # Aligned with augment pipeline tokenBins: [0, 64, 128, 256, 384, 512, 768, 1024]
    TOKEN_BINS = [(0, 64), (64, 128), (128, 256), (256, 384), (384, 512), (512, 768), (768, 1024), (1024, 9999)]
    BIN_NAMES = ["0-63", "64-127", "128-255", "256-383", "384-511", "512-767", "768-1023", "1024+"]
    bucket_data = []  # (name, count, exact_count, avg_loss)
    ne_src_lens = None  # source lengths for non-exact samples (for per-bucket CER)
    bucket_printed = False
    if total_samples > 0:
        mask = all_tgts != -100
        per_sample_exact = ((all_preds == all_tgts) | ~mask).all(dim=1)
        # Save source lengths of non-exact samples (in original order, matching ne_*_buf).
        ne_src_lens = all_src_lens[~per_sample_exact].clone()

        # Error token analysis: structural (IDs 0-15) vs content (IDs 16+).
        wrong = (all_preds != all_tgts) & mask  # (N, S) — wrong token positions
        wrong_tgt_ids = all_tgts.clone()
        wrong_tgt_ids[~wrong] = -1  # mask out correct positions
        n_wrong = wrong.sum().item()
        n_struct_wrong = ((wrong_tgt_ids >= 0) & (wrong_tgt_ids <= 15)).sum().item()
        n_content_wrong = n_wrong - n_struct_wrong
        print(f"  Error tokens: {n_wrong} total — {n_struct_wrong} structural ({100*n_struct_wrong/max(n_wrong,1):.0f}%), "
              f"{n_content_wrong} content ({100*n_content_wrong/max(n_wrong,1):.0f}%)", flush=True)
        del wrong_tgt_ids

        # Per-position token accuracy: does accuracy degrade at later positions?
        correct = (all_preds == all_tgts) & mask
        POS_BINS = [(0, 100), (100, 300), (300, 500), (500, 800), (800, 1200), (1200, 9999)]
        POS_NAMES = ["0-99", "100-299", "300-499", "500-799", "800-1199", "1200+"]
        print(f"  Token acc by position: ", end="", flush=True)
        pos_parts = []
        for (lo, hi), name in zip(POS_BINS, POS_NAMES):
            pos_mask = torch.zeros(max_seq, dtype=torch.bool, device=device)
            pos_mask[lo:min(hi, max_seq)] = True
            pos_valid = mask & pos_mask.unsqueeze(0)
            n_valid = pos_valid.sum().item()
            if n_valid > 0:
                n_correct = (correct & pos_valid).sum().item()
                pos_parts.append(f"{name}={100*n_correct/n_valid:.3f}%")
        print(" | ".join(pos_parts), flush=True)
        del correct

        for (lo, hi), name in zip(TOKEN_BINS, BIN_NAMES):
            in_bin = (all_src_lens >= lo) & (all_src_lens < hi)
            count = in_bin.sum().item()
            if count > 0:
                exact_in_bin = (per_sample_exact & in_bin).sum().item()
                avg_loss = all_sample_losses[in_bin].mean().item()
                bucket_data.append((name, count, exact_in_bin, avg_loss))
        del per_sample_exact

    # Free the big pred/tgt buffers.
    del all_preds, all_tgts, all_src_lens, all_sample_losses

    # ── Phase 3: Pull 3 scalars, then all-GPU decode + CER + batch WER. ──
    exact_matches = gpu_exact_count.item()
    total_ref_bytes = gpu_ref_bytes.item()
    n_nonexact = min(ne_count.item(), max_ne)
    print(f"  Exact: {exact_matches}/{total_samples} ({100*exact_matches/max(total_samples,1):.1f}%), {n_nonexact} non-exact for CER/WER", flush=True)

    total_cer_edits = 0
    ne_total_bytes = 0  # non-exact bytes only (set after decode)
    total_wer_edits = 0

    if gpu_decoder is not None and n_nonexact > 0:
        print(f"  Triton decode + CER for {n_nonexact} non-exact pairs...", flush=True)
        max_out = min(max_seq * 10, 16384)
        pred_bytes_buf = torch.zeros(n_nonexact, max_out, dtype=torch.uint8, device=device)
        tgt_bytes_buf = torch.zeros(n_nonexact, max_out, dtype=torch.uint8, device=device)
        pred_byte_lens = torch.zeros(n_nonexact, dtype=torch.int32, device=device)
        tgt_byte_lens = torch.zeros(n_nonexact, dtype=torch.int32, device=device)

        BLOCK_T = triton.next_power_of_2(max_seq)
        _batched_decode_norm_kernel[(n_nonexact,)](
            ne_preds_buf, ne_lens_buf,
            gpu_decoder.piece_bytes, gpu_decoder.piece_offsets, gpu_decoder.piece_lengths,
            pred_bytes_buf, pred_byte_lens,
            max_seq, max_out, MAX_TOK=BLOCK_T, MAX_PIECE=gpu_decoder.max_piece_block,
        )
        _batched_decode_norm_kernel[(n_nonexact,)](
            ne_tgts_buf, ne_lens_buf,
            gpu_decoder.piece_bytes, gpu_decoder.piece_offsets, gpu_decoder.piece_lengths,
            tgt_bytes_buf, tgt_byte_lens,
            max_seq, max_out, MAX_TOK=BLOCK_T, MAX_PIECE=gpu_decoder.max_piece_block,
        )

        del ne_preds_buf, ne_tgts_buf, ne_lens_buf

        # Batched Triton Levenshtein: one kernel launch over all non-exact pairs.
        max_byte = torch.stack([pred_byte_lens.max(), tgt_byte_lens.max()]).max().item()
        if max_byte > 0:
            block = _next_pow2(max_byte)
            cer_out = torch.zeros(n_nonexact, dtype=torch.int32, device=device)
            row = block + 1
            cer_work = torch.zeros(n_nonexact, 2 * row, dtype=torch.int32, device=device)

            _lev_kernel[(n_nonexact,)](
                pred_bytes_buf, tgt_bytes_buf,
                pred_byte_lens, tgt_byte_lens,
                cer_out, cer_work,
                max_out, 2 * row, MAX_LEN=block,
            )
            total_cer_edits = cer_out.sum().item()
            ne_total_bytes = tgt_byte_lens.sum().item()

            # Per-bucket CER using non-exact source lengths.
            if ne_src_lens is not None and len(ne_src_lens) == n_nonexact:
                ne_tgt_lens = tgt_byte_lens[:n_nonexact]
                print(f"  Val by src tokens:  {'Bucket':>10} {'Count':>6} {'Exact%':>7} {'AvgLoss':>8} {'CER':>7} {'NE':>4}", flush=True)
                for name, count, exact_count, avg_loss in bucket_data:
                    lo, hi = next((l, h) for (l, h), n in zip(TOKEN_BINS, BIN_NAMES) if n == name)
                    ne_in_bin = (ne_src_lens >= lo) & (ne_src_lens < hi)
                    ne_count_bin = ne_in_bin.sum().item()
                    if ne_count_bin > 0:
                        bin_edits = cer_out[ne_in_bin].sum().item()
                        bin_chars = ne_tgt_lens[ne_in_bin].sum().item()
                        bin_cer = bin_edits / max(bin_chars, 1)
                    else:
                        bin_cer = 0.0
                    print(f"                      {name:>10} {count:>6} {100*exact_count/count:>6.1f}% {avg_loss:>8.4f} {100*bin_cer:>6.2f}% {ne_count_bin:>4}", flush=True)
                bucket_printed = True
                del ne_src_lens

            del cer_out, cer_work

        # WER: transfer to CPU, return args for async computation (overlaps with AR eval).
        print(f"  CER done ({total_cer_edits} edits). WER deferred ({n_nonexact} pairs).", flush=True)
        pred_lens_cpu = pred_byte_lens.tolist()
        tgt_lens_cpu = tgt_byte_lens.tolist()
        pred_bytes_cpu = pred_bytes_buf.cpu().numpy()
        tgt_bytes_cpu = tgt_bytes_buf.cpu().numpy()

        # Save non-exact pred/tgt pairs for inspection.
        ne_path = os.path.join(output_dir, "val_nonexact.jsonl") if output_dir else None
        if ne_path:
            with open(ne_path, "w") as f:
                for i in range(n_nonexact):
                    pred_str = pred_bytes_cpu[i][:pred_lens_cpu[i]].tobytes().decode("utf-8", errors="replace")
                    tgt_str = tgt_bytes_cpu[i][:tgt_lens_cpu[i]].tobytes().decode("utf-8", errors="replace")
                    json.dump({"pred": pred_str, "target": tgt_str}, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  Saved {n_nonexact} non-exact pairs to {ne_path}", flush=True)

        del pred_bytes_buf, tgt_bytes_buf, pred_byte_lens, tgt_byte_lens

        wer_args = [(pred_bytes_cpu[i], pred_lens_cpu[i],
                      tgt_bytes_cpu[i], tgt_lens_cpu[i])
                     for i in range(n_nonexact)]
    else:
        del ne_preds_buf, ne_tgts_buf, ne_lens_buf
        wer_args = None

    # Print bucket stats without CER if we didn't compute it above.
    if bucket_data and not bucket_printed:
        print(f"  Val by src tokens:  {'Bucket':>10} {'Count':>6} {'Exact%':>7} {'AvgLoss':>8}", flush=True)
        for name, count, exact_count, avg_loss in bucket_data:
            print(f"                      {name:>10} {count:>6} {100*exact_count/count:>6.1f}% {avg_loss:>8.4f}", flush=True)

    exact_rate = exact_matches / max(total_samples, 1)
    val_cer = total_cer_edits / max(ne_total_bytes, 1)
    return total_loss, total_tokens, exact_rate, val_cer, ne_total_bytes, wer_args


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
        # Used as Triton constexpr MAX_PIECE for decode kernel.
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
        # Account for leading space strip (sentencepiece convention)
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
    # Keep non-ws bytes, plus the first byte of each ws run
    prev_ws = torch.cat([torch.tensor([False], device=t.device), ws[:-1]])
    keep = ~ws | (ws & ~prev_ws)
    out = t[keep].clone()
    # Remaining ws bytes (tab/newline/CR that were first-in-run) → space
    out_ws = (out == 9) | (out == 10) | (out == 13)
    out[out_ws] = 32
    # Strip leading/trailing space
    if len(out) > 0 and out[0] == 32:
        out = out[1:]
    if len(out) > 0 and out[-1] == 32:
        out = out[:-1]
    return out


# ── Triton validation bookkeeping kernel ─────────────────────────────────────
#
# One launch over all val samples. Each program: exact match check, ref_len
# accumulation, non-exact buffer append. Zero Python in the hot path.

@triton.jit
def _val_bookkeep_kernel(
    preds_ptr, tgt_ptr, piece_lens_ptr,
    exact_count_ptr, ref_bytes_ptr,
    ne_preds_ptr, ne_tgts_ptr, ne_lens_ptr, ne_count_ptr,
    seq_len, ne_buf_stride, max_ne,
    BLOCK_S: tl.constexpr,
):
    bid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    valid = offs < seq_len
    base = bid * seq_len

    # int16 buffers — load and widen.
    p = tl.load(preds_ptr + base + offs, mask=valid, other=0).to(tl.int64)
    t = tl.load(tgt_ptr + base + offs, mask=valid, other=-100).to(tl.int64)

    # Mask: -100 is padding (int16 range covers -100).
    m = (t != -100) & valid

    # Exact match.
    has_mismatch = tl.sum(((p != t) & m).to(tl.int32)) > 0
    if not has_mismatch:
        tl.atomic_add(exact_count_ptr, 1)

    # Ref_len: sum piece_lengths for masked target tokens.
    safe_t = tl.where(m, t, 0)
    plens = tl.load(piece_lens_ptr + safe_t, mask=m, other=0)
    tl.atomic_add(ref_bytes_ptr, tl.sum(plens).to(tl.int64))

    # Non-exact: atomically append to buffer.
    if has_mismatch:
        slot = tl.atomic_add(ne_count_ptr, 1)
        if slot < max_ne:
            ne_base = slot * ne_buf_stride
            tl.store(ne_preds_ptr + ne_base + offs,
                     tl.where(m, p, 0).to(tl.int16), mask=valid)
            tl.store(ne_tgts_ptr + ne_base + offs,
                     tl.where(m, t, 0).to(tl.int16), mask=valid)
            tl.store(ne_lens_ptr + slot, tl.sum(m.to(tl.int32)))


# ── Triton batched decode + normalize kernel ─────────────────────────────────
#
# Fused token decode + whitespace normalization. One program per sample.
# Reads token IDs, scatters piece bytes to output, normalizes whitespace inline.
# Output: clean normalized byte sequences ready for Levenshtein.

@triton.jit
def _batched_decode_norm_kernel(
    tok_ptr, tok_lens_ptr,
    piece_bytes_ptr, piece_offsets_ptr, piece_lengths_ptr,
    out_ptr, out_lens_ptr,
    max_seq, max_out,
    MAX_TOK: tl.constexpr,
    MAX_PIECE: tl.constexpr,
):
    bid = tl.program_id(0)
    tlen = tl.load(tok_lens_ptr + bid)
    out_base = bid * max_out
    out_pos = 0
    prev_ws = True  # treat start as whitespace to strip leading space

    for t in tl.range(0, MAX_TOK):
        if t < tlen:
            tid = tl.load(tok_ptr + bid * max_seq + t).to(tl.int64)
            if tid > 0:
                poff = tl.load(piece_offsets_ptr + tid)
                plen = tl.load(piece_lengths_ptr + tid)
                for b in tl.range(0, MAX_PIECE):
                    if b < plen and out_pos < max_out:
                        bval = tl.load(piece_bytes_ptr + poff + b)
                        is_ws = (bval == 32) | (bval == 9) | (bval == 10) | (bval == 13)
                        if is_ws:
                            if not prev_ws:
                                tl.store(out_ptr + out_base + out_pos, 32)
                                out_pos += 1
                            prev_ws = True
                        else:
                            tl.store(out_ptr + out_base + out_pos, bval)
                            out_pos += 1
                            prev_ws = False

    # Strip trailing space.
    if out_pos > 0:
        last = tl.load(out_ptr + out_base + out_pos - 1)
        if last == 32:
            out_pos -= 1

    tl.store(out_lens_ptr + bid, out_pos)


# ── Triton Levenshtein kernel ────────────────────────────────────────────────
#
# Single kernel launch per pair (or per batch). The entire DP double-loop
# compiles to native GPU instructions — no Python loop, no kernel launch
# overhead per anti-diagonal. Batched version runs one program per pair.

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

    # Init row0[j] = j (prev row)
    for j in tl.range(0, ROW):
        tl.store(row0 + j, j)

    # Double-buffer: alternate prev/curr via flag, no copy.
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

    # Result is in whichever row was last written (curr).
    if use_row0_as_prev:
        tl.store(out_ptr + pid, tl.load(row0 + nb))
    else:
        tl.store(out_ptr + pid, tl.load(row1 + nb))


@triton.jit
def _fused_decode_cer_kernel(
    pred_tok_ptr, tgt_tok_ptr, tok_lens_ptr,
    piece_bytes_ptr, piece_offsets_ptr, piece_lengths_ptr,
    cer_out_ptr, tgt_len_out_ptr,
    a_buf_ptr, b_buf_ptr, work_ptr,
    max_seq, max_out, stride_work,
    MAX_TOK: tl.constexpr,
    MAX_PIECE: tl.constexpr,
    MAX_LEN: tl.constexpr,
):
    """Fused decode + Levenshtein CER. One program per sample, zero intermediate buffers."""
    pid = tl.program_id(0)
    tlen = tl.load(tok_lens_ptr + pid)
    a_base = a_buf_ptr + pid * max_out
    b_base = b_buf_ptr + pid * max_out

    # ── Decode pred tokens → bytes (a_buf), with whitespace normalization ──
    pred_base = pred_tok_ptr + pid * max_seq
    a_pos = 0
    prev_ws = True
    for t in tl.range(0, MAX_TOK):
        if t < tlen:
            tid = tl.load(pred_base + t).to(tl.int64)
            if tid > 0:
                poff = tl.load(piece_offsets_ptr + tid)
                plen = tl.load(piece_lengths_ptr + tid)
                for b in tl.range(0, MAX_PIECE):
                    if b < plen and a_pos < max_out:
                        bval = tl.load(piece_bytes_ptr + poff + b)
                        is_ws = (bval == 32) | (bval == 9) | (bval == 10) | (bval == 13)
                        if is_ws:
                            if not prev_ws:
                                tl.store(a_base + a_pos, 32)
                                a_pos += 1
                            prev_ws = True
                        else:
                            tl.store(a_base + a_pos, bval)
                            a_pos += 1
                            prev_ws = False
    if a_pos > 0:
        last = tl.load(a_base + a_pos - 1)
        if last == 32:
            a_pos -= 1
    na = a_pos

    # ── Decode tgt tokens → bytes (b_buf) ──
    tgt_base = tgt_tok_ptr + pid * max_seq
    b_pos = 0
    prev_ws = True
    for t in tl.range(0, MAX_TOK):
        if t < tlen:
            tid = tl.load(tgt_base + t).to(tl.int64)
            if tid > 0:
                poff = tl.load(piece_offsets_ptr + tid)
                plen = tl.load(piece_lengths_ptr + tid)
                for b in tl.range(0, MAX_PIECE):
                    if b < plen and b_pos < max_out:
                        bval = tl.load(piece_bytes_ptr + poff + b)
                        is_ws = (bval == 32) | (bval == 9) | (bval == 10) | (bval == 13)
                        if is_ws:
                            if not prev_ws:
                                tl.store(b_base + b_pos, 32)
                                b_pos += 1
                            prev_ws = True
                        else:
                            tl.store(b_base + b_pos, bval)
                            b_pos += 1
                            prev_ws = False
    if b_pos > 0:
        last = tl.load(b_base + b_pos - 1)
        if last == 32:
            b_pos -= 1
    nb = b_pos
    tl.store(tgt_len_out_ptr + pid, nb)

    # ── Levenshtein DP ──
    ROW: tl.constexpr = MAX_LEN + 1
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
        tl.store(cer_out_ptr + pid, tl.load(row0 + nb))
    else:
        tl.store(cer_out_ptr + pid, tl.load(row1 + nb))


def _next_pow2(x):
    p = 16
    while p < x:
        p *= 2
    return min(p, 8192)


def triton_levenshtein(a, b):
    """Single-pair Levenshtein via Triton. a, b: GPU uint8 tensors."""
    na, nb = len(a), len(b)
    if na == 0: return nb
    if nb == 0: return na
    device = a.device

    block = _next_pow2(max(na, nb))
    # Pad into [1, block] tensors
    a_pad = torch.zeros(1, block, dtype=torch.uint8, device=device)
    b_pad = torch.zeros(1, block, dtype=torch.uint8, device=device)
    a_pad[0, :na] = a
    b_pad[0, :nb] = b
    na_t = torch.tensor([na], dtype=torch.int32, device=device)
    nb_t = torch.tensor([nb], dtype=torch.int32, device=device)
    out = torch.zeros(1, dtype=torch.int32, device=device)
    row = block + 1
    work = torch.zeros(1, 2 * row, dtype=torch.int32, device=device)

    _lev_kernel[(1,)](a_pad, b_pad, na_t, nb_t, out, work,
                      block, 2 * row, MAX_LEN=block)
    return out[0].item()


def batched_triton_levenshtein(a_list, b_list, device):
    """Batch Levenshtein for multiple pairs. a_list/b_list: lists of GPU uint8 tensors.
    Returns list of int distances. Groups by sequence length for optimal kernel dispatch."""
    B = len(a_list)
    if B == 0:
        return []

    results = [0] * B

    # Group by power-of-2 block size
    groups = {}
    for idx in range(B):
        na, nb = len(a_list[idx]), len(b_list[idx])
        if na == 0 or nb == 0:
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

        # One bulk .tolist() per group instead of per-element .item().
        out_list = out_t.tolist()
        for li, gi in enumerate(indices):
            results[gi] = out_list[li]

    return results


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
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def xml_semantically_equal(a, b):
    """Compare two XML strings by flattening to canonical (element, text) tokens.

    Treats CDATA vs plain text as equivalent and ignores insignificant whitespace.
    """
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
    """Word-level Levenshtein where each edit is weighted by character length.

    A substitution costs max(len(ref_word), len(pred_word)) chars.
    A deletion costs len(ref_word) chars. An insertion costs len(pred_word) chars.
    """
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


def _wer_worker(args):
    """Picklable WER worker for ProcessPoolExecutor."""
    pred_bytes, pred_len, tgt_bytes, tgt_len = args
    pred_text = pred_bytes[:pred_len].tobytes().decode('utf-8', errors='replace')
    tgt_text = tgt_bytes[:tgt_len].tobytes().decode('utf-8', errors='replace')
    return char_weighted_wer(pred_text.split(), tgt_text.split())


_CDATA_OPEN_ID = 14   # <![CDATA[
_CDATA_CLOSE_ID = 15  # ]]>


def mrt_error_weighted_loss(logits, tgt_labels, vocab_size, error_weight, threshold,
                            cdata_weight=0.0):
    """Online MRT: upweight loss on near-miss samples + targeted CDATA weighting.

    Near-miss = sample with token accuracy >= threshold but < 1.0.
    All tokens in near-miss samples get error_weight multiplier.
    CDATA tokens (IDs 14, 15) get cdata_weight multiplier on ALL samples (not just near-misses).
    Zero extra kernel launches — just argmax + comparison on existing logits.
    """
    B, S = tgt_labels.shape
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = tgt_labels.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)

    valid = (tgt_labels != -100)                          # (B, S)
    preds = logits.argmax(dim=-1)                         # (B, S)
    correct = (preds == tgt_labels) & valid               # (B, S)
    token_acc = correct.sum(dim=1).float() / valid.sum(dim=1).float().clamp(min=1)
    near_miss = (token_acc >= threshold) & (token_acc < 1.0)

    # Build per-token weight.
    weights = torch.ones(B, S, device=tgt_labels.device, dtype=per_token.dtype)
    # MRT: upweight ALL tokens in near-miss samples.
    error_mask = near_miss.unsqueeze(1) & valid
    weights[error_mask] = error_weight
    # CDATA: upweight tokens 14/15 on ALL samples (dominant failure mode).
    if cdata_weight > 0:
        cdata_mask = ((tgt_labels == _CDATA_OPEN_ID) | (tgt_labels == _CDATA_CLOSE_ID)) & valid
        weights[cdata_mask] = torch.maximum(weights[cdata_mask],
                                            torch.tensor(cdata_weight, device=weights.device, dtype=weights.dtype))
    weights[~valid] = 0.0

    return (per_token * weights.reshape(-1)).sum() / weights.sum().clamp(min=1)


def weighted_content_loss(log_probs, tgt_labels, vocab_size, content_weight, structural_max_id,
                          structural_weight=1.0):
    """NLL loss with configurable weight on content and structural tokens.

    Input is raw logits from the model.
    Structural tokens (IDs 0..structural_max_id) get structural_weight.
    All other tokens (content being generated) get content_weight.
    Padding positions (label == -100) contribute 0.
    """
    flat_logits = log_probs.reshape(-1, vocab_size)
    flat_labels = tgt_labels.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)

    valid = flat_labels != -100
    structural = (flat_labels >= 0) & (flat_labels <= structural_max_id)
    weights = torch.where(structural, structural_weight, content_weight)
    weights = torch.where(valid, weights, 0.0)

    return (per_token * weights).sum() / weights.sum().clamp(min=1)


@torch.no_grad()
def autoregressive_eval(model, sp, n_samples=10,
                        augment_bin="/app/augment", data_dir="data",
                        max_src_len=1152, max_tgt_len=1536,
                        output_dir=None, epoch=0, stage=None,
                        device="cuda", gpu_decoder=None):
    """Run greedy autoregressive decoding on fresh augmented haiku samples.

    Uses the augment binary to generate fresh samples matching the current
    stage's type and difficulty. Writes per-sample results to
    output_dir/ar_inferences/epoch_N.jsonl for failure analysis.
    """
    model.eval()

    # Generate fresh samples matching current stage difficulty.
    seed = int(time.time()) % 2**32
    if stage is None:
        stage = max(HAIKU_STAGES.keys())
    params = HAIKU_STAGES[stage]
    # Generate enough samples to guarantee n_samples across all token-length bins.
    # Over-request by 10x to give the stratifier enough samples per bin.
    aug = max(params["aug_ratio"], 1)
    needed_haiku = max(1, (n_samples * 10 // aug) + 5)
    corpus_est = 50000 if params.get("type", "all") != "all" else 100000
    sample_pct = max(0.01, needed_haiku / (corpus_est / 100))

    # Find tokenizer path (same directory as augment_bin or from data_dir).
    tok_path = os.path.join(output_dir, "tokenizer.model") if output_dir else None
    cmd = [
        augment_bin,
        "-dir", os.path.join(data_dir, "haiku"),
        "-sample-pct", f"{sample_pct:.4f}",
        "-aug-ratio", str(params["aug_ratio"]),
        "-special-prob", str(params["special_prob"]),
        "-type", str(params.get("type", "all")),
        "-shorten-pct", str(params.get("shorten_pct", 0)),
        "-compact-pct", str(params.get("compact_pct", 50)),
        "-seed", str(seed),
    ]
    if tok_path and os.path.exists(tok_path):
        cmd.extend(["-tokenizer", tok_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    records = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))

    # Batched greedy AR decode on GPU — all samples in parallel.
    # Tokenize all records, filter by length, shuffle, then take n_samples.
    # Shuffle ensures balanced length distribution from stratified augment output.
    # Parallel tokenization — sp.encode is CPU-bound
    from multiprocessing.pool import ThreadPool
    def _tok(r):
        r["_src_ids"] = sp.encode(r["input"])
        return r
    with ThreadPool(12) as pool:
        records = list(pool.map(_tok, records))
    eligible = [r for r in records if len(r["_src_ids"]) <= max_src_len]
    random.shuffle(eligible)
    samples = eligible[:n_samples]

    if len(samples) < n_samples:
        print(f"  AR eval: only {len(samples)}/{n_samples} samples fit context "
              f"({len(records)} generated, {len(records) - len(samples)} skipped)", flush=True)
    src_ids_list = [r["_src_ids"] for r in samples]
    targets = [r["target"] for r in samples]

    print(f"AR eval: fanned-out decode of {len(samples)} samples on {device}...", flush=True)
    # Batch src_ids into a padded tensor for ar_decode_parallel
    from ar_parallel import ar_decode_parallel
    _max_src = max(len(ids) for ids in src_ids_list)
    _src_t = torch.zeros(len(src_ids_list), _max_src, dtype=torch.long, device=device)
    _src_mask = torch.ones(len(src_ids_list), _max_src, dtype=torch.bool, device=device)
    for i, ids in enumerate(src_ids_list):
        _src_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        _src_mask[i, :len(ids)] = False
    all_pred_ids = ar_decode_parallel(model, _src_t, sp, max_len=max_tgt_len,
                                      src_key_padding_mask=_src_mask)
    print(f"  Decode complete, scoring...", flush=True)

    # Score each sample.
    exact_count = 0
    semantic_count = 0
    xml_ok_count = 0
    total = len(samples)
    ne_cer_chars = 0
    total_cer_edits = 0
    ne_wer_words = 0
    total_wer_edits = 0
    inferences = []

    # Batch decode ALL predictions to strings in one pass (one bulk GPU→CPU transfer).
    if gpu_decoder is not None:
        all_pred_texts = [gpu_decoder.decode(ids) for ids in all_pred_ids]
    else:
        all_pred_texts = [sp.decode(ids) for ids in all_pred_ids]

    # All CPU scoring (regex, XML parse, exact check) — GPU is free.
    nonexact_indices = []
    for i, (pred, target) in enumerate(zip(all_pred_texts, targets)):
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

        if exact:
            exact_count += 1
            char_edits = 0
            wer_chars = 0
        elif semantic:
            semantic_count += 1
            char_edits = 0
            wer_chars = 0
        else:
            nonexact_indices.append(i)
            ne_cer_chars += char_ref_len
            ne_wer_words += char_ref_len
            char_edits = -1
            wer_chars = char_weighted_wer(norm_pred.split(), norm_tgt.split())
            total_wer_edits += wer_chars

        if xml_ok:
            xml_ok_count += 1

        inferences.append({
            "input": samples[i]["input"],
            "expected": target,
            "predicted": pred,
            "exact": exact,
            "semantic": semantic,
            "xml_ok": xml_ok,
            "cer": -1.0,
            "wer": round(wer_chars / max(char_ref_len, 1), 6),
            "char_edits": char_edits,
            "wer_chars": wer_chars,
        })

    # Batched Triton CER for non-exact pairs — GPU decode + Levenshtein.
    if nonexact_indices and gpu_decoder is not None:
        nonexact_pred_bytes = [gpu_normalize_ws(gpu_decoder.decode_to_bytes(
            list(all_pred_ids[i]))) for i in nonexact_indices]
        nonexact_tgt_bytes = [gpu_normalize_ws(gpu_decoder.decode_to_bytes(
            sp.encode(targets[i])[:max_tgt_len])) for i in nonexact_indices]
        distances = batched_triton_levenshtein(nonexact_pred_bytes, nonexact_tgt_bytes,
                                               gpu_decoder.device)
        for idx, dist in zip(nonexact_indices, distances):
            inferences[idx]["char_edits"] = dist
            ref = max(len(re.sub(r"\s+", " ", targets[idx].strip())), 1)
            inferences[idx]["cer"] = round(dist / ref, 6)
            total_cer_edits += dist

    # Fill CER for exact matches.
    for inf in inferences:
        if inf["cer"] < 0:
            inf["cer"] = 0.0

    overall_cer = total_cer_edits / max(ne_cer_chars, 1)
    overall_wer = total_wer_edits / max(ne_wer_words, 1)

    # Print per-sample results.
    for i, inf in enumerate(inferences):
        status = "exact" if inf["exact"] else ("semantic" if inf["semantic"] else ("xml_ok" if inf["xml_ok"] else "FAIL"))
        detail = "" if inf["exact"] else f" cer={inf['cer']:.1%} wer={inf['wer']:.1%}"
        print(f"  AR eval {i+1}/{total}: {len(src_ids_list[i])}→{len(all_pred_ids[i])} tokens [{status}]{detail}", flush=True)

    # Per-bucket AR stats by source token length.
    TOKEN_BINS = [(0, 64), (64, 128), (128, 256), (256, 384), (384, 512), (512, 768), (768, 1024), (1024, 9999)]
    BIN_NAMES = ["0-63", "64-127", "128-255", "256-383", "384-511", "512-767", "768-1023", "1024+"]
    ar_bucket_data = {}
    for i, inf in enumerate(inferences):
        src_len = len(src_ids_list[i])
        for (lo, hi), name in zip(TOKEN_BINS, BIN_NAMES):
            if lo <= src_len < hi:
                if name not in ar_bucket_data:
                    ar_bucket_data[name] = {"count": 0, "exact": 0}
                ar_bucket_data[name]["count"] += 1
                if inf["exact"]:
                    ar_bucket_data[name]["exact"] += 1
                break
    print(f"AR eval done: {exact_count}/{total} exact, {semantic_count}/{total} semantic, {xml_ok_count}/{total} xml_ok, CER={overall_cer:.2%}, WER={overall_wer:.2%}", flush=True)
    if ar_bucket_data:
        print(f"  AR by src tokens:   {'Bucket':>10} {'Count':>6} {'Exact%':>7}", flush=True)
        for name in BIN_NAMES:
            if name in ar_bucket_data:
                b = ar_bucket_data[name]
                print(f"                      {name:>10} {b['count']:>6} {100*b['exact']/b['count']:>6.1f}%", flush=True)

    # Write inference log.
    if output_dir:
        ar_dir = os.path.join(output_dir, "ar_inferences")
        os.makedirs(ar_dir, exist_ok=True)
        ar_path = os.path.join(ar_dir, f"epoch_{epoch}.jsonl")
        with open(ar_path, "w") as f:
            for inf in inferences:
                f.write(json.dumps(inf) + "\n")

    return exact_count, semantic_count, xml_ok_count, total, overall_cer, overall_wer, ar_bucket_data


def interrupt_filename():
    """Timestamped interrupt checkpoint filename (never overwrites previous)."""
    return f"interrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, output_dir, filename, epoch_complete=True, epoch_step=0, epoch_seed=None, stage=1, stage_good_epochs=0, training_done=False, val_state=None, train_loss=0.0, cw_boost=0.0, pf_boost=0.0, sw_boost=0.0, prev_val_loss=None, best_ar_exact=0):
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
        "cw_boost": cw_boost,
        "pf_boost": pf_boost,
        "sw_boost": sw_boost,
        "prev_val_loss": prev_val_loss,
        "best_ar_exact": best_ar_exact,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if val_state is not None:
        ckpt["val_state"] = val_state
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train transmutation model (haiku-first)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--tokenizer", default="models/tokenizer.model", help="Tokenizer model path")
    parser.add_argument("--output-dir", default="models", help="Output directory")

    # Model.
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=64)
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
    parser.add_argument("--lr-patience", type=int, default=5,
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
    parser.add_argument("--structural-weight", type=float, default=1.0,
                        help="Weight multiplier for structural tokens. Prevents copy mechanism "
                             "from dominating generation head. Try 5.0-10.0.")
    parser.add_argument("--token-noise", type=float, default=0.15,
                        help="Fallback probability of replacing a content token (used if stage not in PF_NOISE_SCHEDULE)")
    parser.add_argument("--professor-forcing", action="store_true", default=True,
                        help="Use model predictions instead of random tokens for noise (requires --token-noise > 0)")
    parser.add_argument("--pf-iterations", type=int, default=1,
                        help="Number of iterative PF passes (1=standard, 3+=cascading error approximation)")
    parser.add_argument("--ar-train-frac", type=float, default=0.0,
                        help="Fraction of training batches using full AR decode as input (0.0=off, 0.1=10%%)")
    parser.add_argument("--skip-data-gen", action="store_true", default=False,
                        help="Skip val/train data generation on first epoch (reuse existing data)")
    parser.add_argument("--force-resume", action="store_true", default=False,
                        help="Resume from specified checkpoint even if newer ones exist")

    # Haiku augmentation pipeline.
    parser.add_argument("--augment-bin", type=str, default="/app/augment",
                        help="Path to Go augment binary")
    parser.add_argument("--dict-word-pct", type=float, default=None,
                        help="Override dict-word-pct for all stages (0=shuffle only, 50=default, 100=all dict words)")
    parser.add_argument("--stage", type=int, default=1,
                        help="Starting curriculum stage (1-6)")
    parser.add_argument("--max-stage", type=int, default=6,
                        help="Maximum curriculum stage (auto-advance stops here)")
    parser.add_argument("--force-stage", type=int, default=None,
                        help="Override checkpoint stage on resume (one-time use, do NOT put in run.sh)")
    parser.add_argument("--stage-patience", type=int, default=2,
                        help="Consecutive epochs above threshold before advancing")

    parser.add_argument("--mrt-weight", type=float, default=0.0,
                        help="Error token weight multiplier for online MRT (0=disabled, 2.0=typical). "
                             "Near-miss samples (token acc >= threshold) get error tokens upweighted.")
    parser.add_argument("--mrt-threshold", type=float, default=0.90,
                        help="Min token accuracy for near-miss MRT gating (default 0.90)")
    parser.add_argument("--cdata-weight", type=float, default=0.0,
                        help="Extra weight on CDATA tokens 14/15 (0=disabled, 20.0=typical). "
                             "Targets the dominant failure mode: missing <![CDATA[ wrapper.")
    parser.add_argument("--max-src-len", type=int, default=1152)
    parser.add_argument("--max-tgt-len", type=int, default=1536)
    parser.add_argument("--max-epoch-samples", type=int, default=0,
                        help="Cap samples per epoch (0=no cap). Subsamples AFTER augmentation for fair coverage.")
    parser.add_argument("--num-workers", type=int, default=2)

    # Checkpointing.
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
