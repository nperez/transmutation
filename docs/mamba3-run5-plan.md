# Mamba3 Architecture Upgrade — Run5 Plan

## Motivation

Run4 plateaued at 60.5% val_exact in stage 5. The core SSM (Mamba 1 / `mamba_simple`)
uses real-valued states that can only express exponential decay — they cannot track
oscillatory patterns like XML nesting depth. Mamba3 introduces complex-valued states
with rotational dynamics that solve bracketing/parity problems where Mamba 2 fails
entirely (0.88% → 87.75% on arithmetic with brackets).

Additionally, Mamba3 includes BCNorm (training stabilization), a three-term
exponential-trapezoidal discretization (replaces conv1d), and learnable B/C biases.
These come free as a drop-in replacement of the SSM block.

## Architecture Changes

### What Changes

Replace `Mamba` from `mamba_ssm.modules.mamba_simple` with `Mamba3` from
`mamba_ssm.modules.mamba3` in both encoder and decoder layers.

```python
# Before (model.py)
from mamba_ssm.modules.mamba_simple import Mamba
self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

# After
from mamba_ssm.modules.mamba3 import Mamba3
self.mamba = Mamba3(d_model=d_model, d_state=TBD, expand=2, headdim=64)
```

The `forward(x)` call signature is identical — `(batch, seqlen, d_model)` in/out.

### What Stays The Same

- Encoder-decoder architecture with cross-attention
- Copy gate (pointer-generator) on last decoder layer
- Weight-tied embedding + output projection
- Feedforward layers in decoder (Linear → GELU → Linear)
- Cross-attention (nn.MultiheadAttention) in decoder
- Loss function (NLLLoss with copy mechanism)
- Go inference with ONNX (updated state shapes, see below)

### What Mamba3 Gives Us (Built In)

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| Complex-valued states | Rotational dynamics via RoPE trick | Tracks XML nesting depth (bracketing problem) |
| BCNorm | RMSNorm on B/C projections | Training stability — may eliminate fp32 copy workaround |
| Trapezoidal discretization | Three-term recurrence: h_t = α h_{t-1} + β B_{t-1} x_{t-1} + γ B_t x_t | Replaces conv1d, second-order accuracy |
| B/C biases | Learnable data-independent SSM components | Implicit convolution without explicit conv1d |
| No conv1d | Eliminated entirely | Simpler ONNX export (no conv state cache) |

### What Mamba3 Does NOT Give Us

- Encoder-decoder support (Mamba3 is decoder-only in the paper)
- Copy mechanism (we keep our own)
- MIMO (multi-input multi-output) — save for later, significant complexity

## Training Strategy: From Scratch, Simplified Stages

No warm start. The SSM internals are completely different. Training from scratch
lets the complex-valued states learn clean representations without interference
from transferred real-valued state patterns.

### Why Fewer Stages

With Mamba 1, we needed 5 stages because:
1. The model couldn't handle complex inputs early on
2. We needed to layer in difficulty gradually to coax phase transitions
3. Real-valued states needed extensive curriculum to learn structure

Mamba3's complex states should learn structural patterns faster (100% parity vs
0.9%). We don't know where phase transitions will land, so a complicated curriculum
is likely to mistime stage advances. Simpler is better.

### Stage Design

**Stage 1: Full Haiku, Natural + Compact (learn the mapping)**

Train on the FULL haiku corpus (100% sample_pct). No augmentation.
Compact 50% of inputs to single-line JSON (doubles effective variety).
No corruption, no special chars, no shuffle augmentation.

The model learns the fundamental JSON → XML mapping on real data from the start.
Both answer-only and tool samples included — no type filtering.

```
sample_pct=100, aug_ratio=0, special_prob=0, corrupt_pct=0,
compact_pct=50, dict_word_pct=0, type=all
```

PF noise: 0.15 (light — let it learn the basics first)

**Advance signal:** AR exact on val set > 55% for 2 consecutive epochs.

**Stage 2: Augmentation with Content Shuffle (learn to generalize copying)**

Introduce shuffled augmented variants. Shuffle-only (dict_word_pct=0) preserves
real token complexity — code syntax, markdown, punctuation. The model learns that
content tokens must be copied faithfully regardless of their order/context.

Special char injection teaches CDATA handling on augmented samples.

```
sample_pct=100, aug_ratio=3, special_prob=0.20, corrupt_pct=0,
compact_pct=50, dict_word_pct=0, type=all
```

PF noise: 0.30 (the model needs to handle its own predictions on augmented content)

**Advance signal:** AR exact > 55% for 2 consecutive epochs.

**Stage 3: Corruption (learn robustness)**

Introduce JSON corruption (broken structure, missing brackets, truncated strings).
This teaches the model to handle real-world LLM output defects.
Higher special char probability for thorough CDATA practice.
Higher PF noise to harden AR decoding.

```
sample_pct=100, aug_ratio=3, special_prob=0.35, corrupt_pct=15,
compact_pct=50, dict_word_pct=0, type=all
```

PF noise: 0.50 (full AR hardening)

**Advance signal:** None — this is the final stage. Train until convergence.

### Why 100% sample_pct

In run4, we sampled 5% of the corpus per epoch with 10x augmentation — effectively
seeing augmented variants of a small slice. With 100% sample and lower aug_ratio (0-3),
every epoch sees the full corpus. Each epoch is longer but the model sees more
diverse real content per training step. No stratified sampling needed.

### Content Weight / CW Boost

Start with content_weight=1.0 (disabled). Monitor whether the copy gate learns
appropriate specialization. If content fidelity stalls while structure is good,
re-enable cw_boost. But with Mamba3's complex states potentially handling structure
better, the copy gate may find the right balance on its own.

## Implementation Checklist

### Docker Image (Dockerfile)

- [ ] Update mamba-ssm to latest from git (or pin commit with Mamba3 support)
  ```dockerfile
  RUN pip install git+https://github.com/state-spaces/mamba.git@COMMIT_HASH
  ```
- [ ] Verify Triton kernel compatibility on RTX 2060 (sm_75)
- [ ] Verify `einops` is installed (Mamba3 dependency)

### Feasibility Smoke Test (before any code changes)

- [ ] Import Mamba3, instantiate with our dimensions
- [ ] Measure parameter count at d_state=32, 64, 128
- [ ] Run forward pass on GPU, verify no kernel errors
- [ ] Measure peak VRAM with batch_size=2, seq_len=1152
- [ ] Test step() method for single-token decoding
- [ ] **CRITICAL**: If Triton kernels fail on sm_75, stop here

### model.py

- [ ] Change import: `Mamba` → `Mamba3`
- [ ] Update `MambaEncoderLayer.__init__`: replace Mamba constructor
- [ ] Update `MambaDecoderLayer.__init__`: replace Mamba constructor
- [ ] Add `headdim` parameter to `TransmutationModel.__init__`
- [ ] Remove `d_conv` parameter (Mamba3 has no conv1d)
- [ ] Update `_init_weights` if Mamba3 has different parameter naming

### train.py

- [ ] Simplify HAIKU_STAGES to 3 stages (see above)
- [ ] Update PF_NOISE_SCHEDULE to 3 stages
- [ ] Update VAL_MAX_BATCHES to 3 stages
- [ ] Update stage advance signal thresholds
- [ ] Remove `d_conv` from model constructor args
- [ ] Add `headdim` to model constructor args
- [ ] Remove `strict=False` checkpoint loading (fresh start, no legacy compat needed)

### export.py

- [ ] Write `_mamba3_step()`: pure PyTorch three-term recurrence with RoPE
  - No conv buffer — eliminated
  - New state: (angle_state, ssm_state, k_state, v_state) per layer
  - BCNorm + B/C bias application
  - RoPE rotation accumulation
- [ ] Update `SingleStepDecoderWrapper`:
  - Input: `tgt_token, all_k, all_v, all_angle, all_ssm, all_k_state, all_v_state, src_ids`
  - Output: `log_probs, all_angle_out, all_ssm_out, all_k_state_out, all_v_state_out`
- [ ] Update `mamba_forward_onnx()` for encoder (use Mamba3's chunked forward)
  - May need pure PyTorch selective scan replacement for ONNX traceability
- [ ] Update dummy input shapes and dynamic axes
- [ ] Update validation comparison

### cmd/infer/main.go

- [ ] Update decoder ONNX session: new input/output names and shapes
- [ ] Replace `all_h, all_conv` state management with `all_angle, all_ssm, all_k_state, all_v_state`
- [ ] Update `greedyDecode` and `beamDecode` state initialization
- [ ] Remove conv state allocation

### cmd/augment/main.go

- [ ] No changes needed (augmentation is architecture-agnostic)

## Open Questions

### d_state Selection

| d_state | nheads (d_inner=768, headdim=64) | State per layer | Notes |
|---------|----------------------------------|-----------------|-------|
| 32 | 12 | 12×64×32 = 24K floats | Minimal, fast |
| 64 | 12 | 12×64×64 = 49K floats | Balanced |
| 128 | 12 | 12×64×128 = 98K floats | Default, might stress VRAM |

Must measure actual VRAM with forward+backward pass at batch_size=2.

### MIMO Later?

MIMO (rank 4) would give ~4x more SSM capacity with minimal latency impact.
But it requires TileLang kernels, changes B/C projection shapes, and adds
significant implementation complexity. Defer to run6 if needed.

### LR Schedule

Fresh training with new architecture. Start with same LR as run4 (1e-4) and
ReduceLROnPlateau. If complex states converge faster, we may need to adjust.

### Validation Set

Generate val at stage 3 difficulty (with corruption + special chars) as the
fixed held-out set, same as run4 approach. This ensures we measure the hardest
task from the start even when training on easier stages.
