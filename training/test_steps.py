"""Test discrete iterative refinement on multiple samples."""
import sys
sys.path.insert(0, "training")
import json
import subprocess
import torch
from infer import load_model
from model import LENGTH_BUCKETS

device = torch.device("cpu")

# Find latest checkpoint
import glob
ckpts = sorted(glob.glob("models/run7/epoch_*.pt"))
ckpt = ckpts[-1] if ckpts else "models/run7/best.pt"
model, sp = load_model(ckpt, device)
vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else sp.get_piece_size()

# Generate samples
result = subprocess.run(
    ["tmp/generate", "-stage", "3", "-stdout", "-train", "15", "-val", "0", "-seed", "12345"],
    capture_output=True, text=True, cwd="/app"
)
lines = [l for l in result.stdout.strip().split('\n') if l.startswith('{')]

for sample_idx, line in enumerate(lines[:5]):
    rec = json.loads(line)
    src_ids = torch.tensor([sp.encode(rec["input"])], device=device)
    if src_ids.shape[1] > 1152:
        continue
    src_mask = (src_ids == sp.pad_id())

    length_logits = model.predict_length(src_ids, src_mask)
    bucket_idx = min(length_logits.argmax(dim=-1).item() + 1, len(LENGTH_BUCKETS) - 1)
    tgt_len = LENGTH_BUCKETS[bucket_idx]

    print(f"\n=== Sample {sample_idx+1} (src={src_ids.shape[1]} tgt_bucket={tgt_len}) ===")

    torch.manual_seed(sample_idx * 1000 + 42)
    current_ids = torch.randint(0, vocab_size, (1, tgt_len))
    prev_tokens = None

    with torch.no_grad():
        for i in range(4):
            if i == 0:
                t_val = torch.ones(1)
                r_val = torch.ones(1)
            else:
                t_val = torch.full((1,), 0.125)
                r_val = torch.full((1,), 0.125)

            pred_tokens = model.predict_tokens(src_ids, current_ids, t_val, src_mask, None, r=r_val)

            changed = 0 if prev_tokens is None else (pred_tokens[0] != prev_tokens[0]).sum().item()
            prev_tokens = pred_tokens.clone()

            trimmed = []
            for tid in pred_tokens[0].tolist():
                if tid == sp.eos_id() or tid == sp.pad_id():
                    break
                trimmed.append(tid)
            text = sp.decode(trimmed)[:200]
            chg = f" Δ{changed}/{tgt_len}" if i > 0 else ""
            print(f"  step {i+1} (t={t_val[0]:.3f}){chg}: {text[:150]}")

            current_ids = pred_tokens
