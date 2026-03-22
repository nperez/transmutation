# Copy Mechanism (Pointer-Generator) — Run4 Plan

## Motivation

The model gets XML structure right but makes minor content copying errors (dropped characters, duplicated words, token substitutions). These compound over long sequences. A dedicated copy mechanism lets the model explicitly switch between "generate structural XML token" and "copy content token from input," reducing the burden on the shared output projection.

## Architecture

Add a **copy gate** and **copy distribution** to the decoder output:

```
p_copy = sigmoid(copy_gate(decoder_hidden))     # per-position scalar
gen_probs = softmax(output_proj(decoder_hidden)) # standard vocab distribution
copy_probs = scatter_add(attn_weights, src_ids)  # attn weights → vocab space
final = (1 - p_copy) * gen_probs + p_copy * copy_probs
output = log(final + eps)
```

- **copy_gate**: `Linear(d_model, 1)` — 385 new parameters
- **Attention source**: last decoder layer's cross-attention weights, averaged across heads
- **Copy distribution**: attention weights scattered into vocab-sized vector using source token IDs

## Changes By File

### model.py

1. Add `self.copy_gate = nn.Linear(d_model, 1)` to `__init__`
2. Initialize copy_gate bias to -2.0 (sigmoid(-2) ≈ 0.12, gentle start)
3. Add `return_attn_weights=False` param to `MambaDecoderLayer.forward`
   - When True: pass `need_weights=True, average_attn_weights=True` to cross_attn
   - Only the last layer needs True (others skip for performance)
4. Modify `decode()` to accept `src_ids` parameter
   - All layers except last: call normally (no attn weights)
   - Last layer: get `x, attn_weights` with `return_attn_weights=True`
   - If `src_ids` provided and `copy_gate` exists: compute blended log-probs
   - If `src_ids` is None: return raw logits (backward compatible)
5. Modify `forward()` to pass `src_ids` through to `decode()`

### train.py

1. Detect copy: `use_copy = hasattr(model, 'copy_gate')`
2. Loss: `NLLLoss(ignore_index=-100)` when copy active, `CrossEntropyLoss` otherwise
3. `weighted_content_loss`: add `use_log_probs` param, switch from `cross_entropy` to `nll_loss`
4. Checkpoint loading: `strict=False`, initialize missing copy_gate to near-zero
   ```python
   nn.init.constant_(model.copy_gate.bias, -5.0)
   nn.init.zeros_(model.copy_gate.weight)
   ```
5. Professor forcing `.argmax(dim=-1)` works on both logits and log-probs — no change needed

### export.py

1. `_cross_attn_cached`: add `return_attn=False` param, return averaged attention weights when True
2. `SingleStepDecoderWrapper`:
   - Accept `src_ids` as 6th input `(1, src_len) int64`
   - Store `self.copy_gate` from model if present
   - Last layer's cross_attn returns weights
   - Compute blended log-probs, return instead of logits
3. `export_decoder`: add `src_ids` to input names, dynamic axes, dummy inputs
4. Output name: `log_probs` instead of `logits` when copy active
5. Validation: pass src_ids through both PyTorch and ONNX paths

### infer_cpu.py

1. `greedy_decode`: after last decoder layer, compute copy distribution
   - Get attn_weights from last layer's cross_attn (need_weights=True)
   - scatter_add into vocab using src_ids
   - Blend with gen_probs, argmax on blended
2. `beam_decode` / `_decoder_step`: same pattern
3. `load_model`: `strict=False` handling for old checkpoints

### cmd/infer/main.go

1. Add `-copy` flag (or auto-detect from ONNX model input count)
2. Create decoder session with 6 inputs: `tgt_token, all_k, all_v, all_h, all_conv, src_ids`
3. Output name: `log_probs` instead of `logits`
4. Create src_ids tensor once per sample, pass as 6th input each decode step
5. argmax still works on log-probs — greedy decode unchanged except for input passing
6. beam_decode: skip logSoftmax when output is already log-probs

## Memory Impact (RTX 2060 6GB)

Under fp16 autocast with batch_size=2:
- copy_gate: 1.5 KB (negligible)
- attn_weights: (2, 1536, 1152) × 2 bytes = ~7 MB
- copy_probs + gen_probs: (2, 1536, 8000) × 2 bytes × 2 = ~98 MB
- Total new: ~105 MB — fits within remaining headroom (~2GB free during training)

## Backward Compatibility

- Old checkpoints load with `strict=False`
- Missing copy_gate keys detected, initialized to near-zero (sigmoid(-5) ≈ 0.007)
- Model output with near-zero copy is numerically equivalent to pre-copy logits
- Can resume run3 checkpoints into run4 with copy mechanism

## Other Run4 Changes (already implemented)

- PF noise ramp (mechanism TBD — sawtooth vs ratchet vs manual)
- Stage 5 → 5/6 split (0.35/15% → 0.40/20%) — already in train.py
- Drop memory from 20% of augmented samples — already in augment
- Stale checkpoint safety check — already in train.py
- Go tokenizer whitespace fix — already in sentencepiece.go

## Testing Strategy

1. Load run3 best.pt with copy mechanism — verify near-identical output (p_copy ≈ 0)
2. Train a few steps — verify loss is finite and decreasing
3. Export to ONNX — run validation comparing PyTorch vs ONNX step-by-step
4. Go inference — verify matches Python on shared test file
