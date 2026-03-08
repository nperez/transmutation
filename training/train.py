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

"""Training loop for the transmutation model."""

import atexit
import argparse
import json
import os
import signal
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import create_dataloader
from model import build_model


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

    # Shared dataloader args.
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
    val_loader, _ = create_dataloader(data_dir=val_data_dir, shuffle=False, **dl_kwargs)
    # Train loader created per-epoch (for resumable seeded shuffle).
    tmp_loader, _ = create_dataloader(data_dir=train_data_dir, shuffle=True, **dl_kwargs)
    n_train = len(tmp_loader.dataset)
    n_batches_per_epoch = len(tmp_loader)
    del tmp_loader
    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    total_steps = n_batches_per_epoch * args.epochs // args.grad_accum
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    # Loss.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Mixed precision.
    scaler = GradScaler("cuda", enabled=args.fp16)

    # Training state.
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    log_entries = []
    global_step = 0
    start_epoch = 1
    resume_epoch_seed = None
    resume_epoch_step = 0

    # Resume from checkpoint.
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        completed_epoch = ckpt.get("epoch_complete", False)
        start_epoch = ckpt["epoch"] + 1 if completed_epoch else ckpt["epoch"]
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if not completed_epoch:
            resume_epoch_seed = ckpt.get("epoch_seed")
            resume_epoch_step = ckpt.get("epoch_step", 0)
        # Reload existing log.
        log_path = os.path.join(args.output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = json.load(f)
        print(f"Resuming from epoch {start_epoch}, step {resume_epoch_step}, global_step={global_step}, best_val_loss={best_val_loss:.4f}")

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
            save_checkpoint(model, optimizer, scheduler, scaler,
                            atexit_state["epoch"], global_step, best_val_loss,
                            args.output_dir, "interrupt.pt", epoch_complete=False,
                            epoch_step=actual_step, epoch_seed=atexit_state["epoch_seed"])
            print(f">>> atexit: saved at epoch {atexit_state['epoch']} batch {actual_step} <<<")
        except Exception as e:
            print(f">>> atexit: FAILED to save checkpoint: {e} <<<")
    atexit.register(atexit_save)

    print(f"\nTraining for {args.epochs} epochs, {total_steps} steps")
    print(f"Grad accumulation: {args.grad_accum}, effective batch: {args.batch_size * args.grad_accum}")
    print()

    for epoch in range(start_epoch, args.epochs + 1):
        # Create train loader with deterministic seed for this epoch.
        epoch_seed = epoch * 1000 + 42
        start_index = 0
        if epoch == start_epoch and resume_epoch_step > 0:
            if resume_epoch_seed is not None:
                epoch_seed = resume_epoch_seed
            start_index = resume_epoch_step
            print(f"Resuming epoch {epoch} from batch {start_index} (seed={epoch_seed})")
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

            with autocast("cuda", enabled=args.fp16):
                logits = model(src, tgt_in, src_mask)
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
                scheduler.step()
                global_step += 1

            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Handle signal-triggered checkpoint.
            if sig_state["save"]:
                sig_state["save"] = False
                actual_step = start_index + step + 1
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss,
                                args.output_dir, "interrupt.pt", epoch_complete=False,
                                epoch_step=actual_step, epoch_seed=epoch_seed)
                print(f"\n>>> Checkpoint saved: epoch {epoch} batch {actual_step}/{n_batches_per_epoch} <<<")
                if sig_state["stop"]:
                    print("Exiting cleanly.")
                    return model

        avg_train_loss = train_loss / max(train_tokens, 1)

        # Validate.
        val_loss, val_tokens, val_exact = validate(model, val_loader, criterion, vocab_size, device, args.fp16, sp)
        avg_val_loss = val_loss / max(val_tokens, 1)

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_exact_match": val_exact,
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        }
        log_entries.append(log_entry)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_exact={val_exact:.2%} lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, "best.pt")
            print(f"  -> New best model saved (val_loss={avg_val_loss:.4f})")

        # Save periodic checkpoint.
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, args.output_dir, f"epoch_{epoch}.pt")

        # Epoch complete — atexit not needed until next epoch starts.
        atexit_state["active"] = False

        # Clean up interrupt checkpoint after successful epoch completion.
        interrupt_path = os.path.join(args.output_dir, "interrupt.pt")
        if os.path.exists(interrupt_path):
            os.remove(interrupt_path)

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


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, output_dir, filename, epoch_complete=True, epoch_step=0, epoch_seed=None):
    path = os.path.join(output_dir, filename)
    torch.save({
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "epoch_step": epoch_step,
        "epoch_seed": epoch_seed,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train transmutation model")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")

    # Data.
    parser.add_argument("--max-src-len", type=int, default=2048)
    parser.add_argument("--max-tgt-len", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)

    # Checkpointing.
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
