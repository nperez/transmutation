# Transmutation

Transforms broken JSON into valid XML using a neural network.

LLM agents instructed to respond in JSON frequently produce broken output — missing quotes, dropped commas/colons, `//` comments, explanatory text wrapping the JSON, and embedded code (SQL, Python, etc.) inside string values that creates ambiguous boundaries. Transmutation sits between LLM output and response parsing, converting the mess into valid XML that can be deterministically parsed downstream.

XML was chosen as the output format because it cleanly handles embedded content via CDATA sections, avoiding the escaping nightmare of trying to produce valid JSON containing embedded code.

## Architecture

**Model (run 7)**: DiT bidirectional transformer (~45.7M parameters)
- Diffusion Transformer with adaLN-Zero timestep conditioning
- 10 layers, d_model=512, 8 heads, d_ff=1536
- Factored embedding (16k vocab, rank-128 bottleneck)
- RoPE positional encoding, segment embeddings
- Subword tokenization (BPE, 16k vocab) via SentencePiece

**Model (runs 1-6)**: Mamba-based seq2seq encoder-decoder (~25M parameters)
- Mamba 3 (run 5-6) / Mamba 1 (runs 1-4) state space model
- 6 encoder layers, 6 decoder layers, d_model=384
- Cross-attention between decoder and encoder states
- Subword tokenization (BPE, 8k vocab) via SentencePiece

**Training**: Python + PyTorch + CUDA, runs entirely in Docker. Exports to ONNX.

**Inference**: ONNX Runtime. The ONNX models can be loaded from any language with an ONNX runtime — Go, Python, Java, C#, Rust, etc. A Go inference harness is included.

**Data generation**: Go. Two pipelines:
- **Haiku-first** (current) — samples from a corpus of ~140k real LLM haiku outputs with length-stratified sampling, augments by replacing string values with dictionary words or shuffled content while preserving JSON structure, with configurable XML special character injection, JSON corruption, and compact (single-line) JSON output
- **Synthetic** (legacy) — generates random agent response JSON with embedded code, markdown, and tool calls, applies configurable corruption and produces target XML

## Run 1 Results

Run 1 used synthetic data generation with a 5-stage curriculum. Training was done on an RTX 2060 (6GB VRAM) with batch_size=2 and gradient accumulation of 16 (effective batch size 32).

### Training Budget

| Metric | Value |
|--------|-------|
| Optimizer steps | 138,343 |
| Training samples | ~4.4M (138,343 × 32) |
| Training tokens (src+tgt) | ~5.2B |
| Target tokens (loss-contributing) | ~3.0B |
| Validation tokens | ~0.6B |
| Epochs | 53 |
| Steps/epoch (stages 1-2) | ~6,250 (200k samples) |
| Steps/epoch (stages 3-5) | ~2,000-3,000 (64k-96k samples) |

Token estimates based on ~1,170 tokens per sample (measured ~4,100 chars/sample at ~3.5 chars/token with 8k BPE vocab).

### Curriculum Stages

| Stage | Description | Epochs |
|-------|-------------|--------|
| 1 | Clean simple JSON (text answers, markdown) | 1-3 |
| 2 | Tool calls with embedded code (SQL, Python, JS, Go, Shell) | 4-6 |
| 3 | Full content mix + augmentation (special_prob=0.15) | 7-17 |
| 4 | Subtle/light corruption (~10% of samples) | 18-23 |
| 5 | Wrappers + heavier corruption (~20% of samples) | 24-53 |

Auto-advance triggered when AR exact match >= 70% for 2 consecutive epochs.

### Key Metrics

| Epoch | Stage | Train Loss | Val Exact | AR Exact | AR XML OK |
|-------|-------|-----------|-----------|----------|-----------|
| 1     | 1     | 4.823     | 0.0%      | —        | —         |
| 6     | 2     | 0.027     | 92.9%     | 10/10    | 10/10     |
| 18    | 3     | 0.010     | 62.0%     | 24/50    | 46/50     |
| 21    | 4     | 0.008     | 71.5%     | 34/50    | 50/50     |
| 24    | 5     | 0.006     | 76.6%     | 37/50    | 50/50     |
| 44    | 5     | 0.006     | 75.3%     | 24/50    | 42/50     |
| 46    | 5     | 0.005     | 90.1%     | 47/50    | 50/50     |
| 49    | 5     | 0.006     | 94.6%     | 48/50    | 50/50     |
| **51**| **5** | **0.006** | **95.5%** | **50/50**| **50/50** |
| 52    | 5     | 0.006     | 95.0%     | 49/50    | 50/50     |
| 53    | 5     | 0.006     | 94.8%     | 49/50    | 49/50     |

### Peak Performance (Epoch 51)

- **50/50 autoregressive exact match** — perfect on all samples the model can fully see
- **50/50 XML validity** — every output parses as valid XML
- **95.5% token-level val exact match**
- Train loss converged at ~0.006

### Remaining Failures

All failures after epoch 51 were traced to **input truncation** — inputs exceeding the 1536-token max source length. Epoch 52's failure had 1568 tokens (2% truncated); epoch 53's had 2198 tokens (30% truncated). The model produces perfect output for any input it can fully see.

### Lessons Learned

- **CDATA wrapping** required heavy special character injection (0.40 probability at word boundaries) before the model reliably learned `<![CDATA[...]]>` rules. At the default 0.15, CDATA failures persisted for many epochs.
- **Content-weighted loss** (10x weight on string/number tokens vs structural XML tokens) was critical — without it the model would copy structural tokens perfectly but mangle the actual data values. Run 2 improved on this with adaptive sawtooth weighting (see above).
- **Structure before content** — starting with content_weight=1.0 and ramping adaptively is far more efficient than starting at 10x. The model needs to learn where XML tags go before it can learn to copy content into them accurately.
- **fp16 NaN at high LR** — LR 3e-4 caused fp16 overflow with small epochs (~160 steps). Lowered to 2e-4 with 500-step warmup. Larger epochs (~776 steps) also help by stabilizing gradient estimates.
- **Professor forcing** (teacher forcing with scheduled sampling) improved AR eval performance significantly vs pure teacher forcing.
- **Token noise** (0.15 probability of random token substitution in inputs during training) acted as regularization and improved robustness.
- **Batch size 2 + grad_accum 16** was the practical max for 6GB VRAM with mixed precision.
- **Validation budget scaling** — full validation set at every epoch wastes GPU time in early stages. Stage 1 needs only ~100 val samples for a coarse signal; scale up as stages advance.
- **AR eval is the bottleneck** — CPU-based autoregressive decoding (required because Mamba CUDA kernels don't support single-step mode) takes ~15 min for 50 samples. Consider reducing AR eval samples or running less frequently.

## Run 2 Results

Run 2 used real haiku LLM outputs (~140k samples) with an 8-stage answer-first curriculum. Same hardware (RTX 2060 6GB), same model architecture. Best model: epoch 32 (stage 6).

### Training Budget

| Metric | Run 1 | Run 2 | Run 2 / Run 1 |
|--------|-------|-------|---------------|
| Optimizer steps | 138,343 | 50,795 | 37% |
| Epochs | 53 | 39 | 74% |
| Steps to 50/50 AR | 131,248 | 32,651 | **4x faster** |
| Best val exact | 95.5% | 91.4% | — |
| Real reject xml_ok | untested | 4/20 (20%) | — |

### Key Metrics

| Epoch | Stage | Train Loss | Val Exact | AR Exact | AR XML OK |
|-------|-------|-----------|-----------|----------|-----------|
| 1     | 1     | 81.57     | 0.0%      | 0/50     | 0/50      |
| 11    | 1     | 0.44      | 1.0%      | 11/50    | 26/50     |
| 21    | 1→2   | 0.12      | 5.0%      | 41/50    | 47/50     |
| 24    | 3→4   | 0.10      | 73.5%     | 34/50    | 39/50     |
| 27    | 5     | 0.11      | 80.8%     | 49/50    | 50/50     |
| 29    | 6     | 0.15      | 90.1%     | 39/50    | 50/50     |
| **32**| **6** | **0.18**  | **91.4%** | **50/50**| **50/50** |
| 36    | 7     | 0.18      | 90.9%     | 46/50    | 50/50     |
| 38    | 8     | 1.02      | 14.7%     | 49/50    | 50/50     |

### What Worked

- **Sawtooth content weight** — adaptive cw that ramps when val improvement stalls, resets on stage advance. Drove val exact from 80.8%→91.4% without manual intervention.
- **Curriculum learning** — 4x faster to 50/50 AR than run 1. Staged difficulty (answer→tool→mixed→augmented→corrupted) is dramatically more efficient than monolithic training.
- **Bracket swap corruption** — added mid-run when real reject analysis showed `}`↔`]` swaps are the #1 real LLM failure pattern. AR exact jumped 39→48 in one epoch after adding it.
- **Compact JSON augmentation** — compacting pretty-printed JSON to single-line format, introduced at stage 7. Model absorbed it within 3 epochs.

### What Didn't Work

- **Stage 8 (long samples only)** — training exclusively on samples >4000 chars at LR 1.25e-5 produced no convergence. Losses stayed at 0.8-1.5 across two full epochs. The 384d/d_state=16 Mamba architecture hits a representational capacity ceiling at ~1000 tokens.
- **Real reject inference** — 0/20 exact match on actual broken LLM output across all checkpoints. The model handles synthetic corruption well but fails on real patterns: multi-bracket runs (`]}}}`), escaped `\n` in strings, ambiguous nesting boundaries.
- **Late compact introduction** — compact JSON wasn't added until stage 7 (epoch 33). The model spent 32 epochs on pretty-printed only, then had to learn a new format. Should be present from stage 1.

### Lessons Learned (new in run 2)

- **best_ar_exact must be persisted in checkpoints** — a stop/resume cycle reset the tracker, causing best.pt to be overwritten with a worse model.
- **Val set must not change with max-stage** — regenerating the val set with new stage params breaks metric comparability across the run.
- **LR warm restart on stage advance** — the plateau scheduler halved LR 4 times by stage 8, leaving it at 1.25e-5 which was too low to learn new patterns. Implemented LR restart to half-base on advance.
- **AR eval is noisy at later stages** — variable sample lengths cause AR exact to oscillate 32-50 between epochs. Val exact is more stable.
- **Effective sequence length ceiling ~1000 tokens** — the model produces exact matches up to ~1000 tokens, whitespace-collapsed xml_ok up to ~1200, and fails/truncates beyond ~1500.
- **`--no-session-persistence`** — Claude CLI generates ~3GB/hour of session logs without this flag. Essential for batch generation.

## Run 3 Results

Run 3 continued with the haiku-first pipeline, same 5-stage curriculum, same hardware. Key changes from run 2: reduced stages to 5 (dropped stages 6-8), stage advance threshold lowered to 55% AR, content weight sawtooth from 1.0 with adaptive ramp, professor forcing at 15% token noise.

### Training Budget

| Metric | Run 2 | Run 3 | Run 3 / Run 2 |
|--------|-------|-------|---------------|
| Optimizer steps | 50,795 | ~63,500 | 125% |
| Epochs | 39 | 40 | ~same |
| Best val exact | 91.4% | 60.5% | — (different val set) |
| Best AR exact | 50/50 | 50/50 | same |
| Real reject xml_ok | 4/20 (20%) | 5/20 (25%) | +5% |

Note: val_exact is not comparable between runs due to a training restart that regenerated the val set mid-run.

### Key Metrics

| Epoch | Stage | Train Loss | Val Exact | AR Exact | AR XML OK |
|-------|-------|-----------|-----------|----------|-----------|
| 1     | 1     | 63.92     | 0.0%      | 0/50     | 0/50      |
| 10    | 1     | 0.62      | 0.0%      | 0/50     | 0/50      |
| 15    | 1     | 0.45      | 0.0%      | 0/50     | 0/50      |
| 20    | 2     | 0.47      | —         | —        | —         |
| 25    | 5     | 0.52      | —         | 48/50    | —         |
| 30    | 5     | 0.52      | 57.2%     | 46/50    | 47/50     |
| 33    | 5     | 0.51      | 58.4%     | **50/50**| 50/50     |
| **36**| **5** | **0.48**  | **60.5%** | **50/50**| **50/50** |
| 40    | 5     | 0.48      | 60.6%     | 45/50    | 49/50     |

### What Worked

- **Professor forcing noise bump** — increasing token noise from 0.15 to 0.25 at epoch 36 produced the best single-epoch improvement (val_exact 58.6%->60.5%, val_loss 0.508->0.483). The model was starved for self-correction practice.
- **Go tokenizer whitespace fix** — the Go sentencepiece implementation wasn't normalizing `\n`/`\t` to spaces before tokenization, causing completely different token IDs from Python. Fixing this 2-line bug brought Go ONNX inference from 0/20 exact to matching Python exactly.
- **Semantic XML comparison** — added a `SEMANTIC` tier to inference that parses both XML trees and compares canonically, catching cases where output is correct but has CDATA/whitespace differences.
- **Stale checkpoint safety** — added a check that refuses to resume from an old checkpoint when newer ones exist, preventing Docker restart from overwriting progress (happened twice).

### What Didn't Work

- **Late PF noise bump** — bumping professor forcing noise on a converged model (epoch 36+) gave one good epoch then plateaued. PF noise needs to ramp during training, not be bolted on at the end.
- **CW boost vs AR** — content weight ramping to 5.2+ improved val_exact but hurt AR exact (dropped from 50 to 42). High CW pushes teacher-forced accuracy at the expense of autoregressive coherence.
- **Real reject inference** — 0/30 exact match on repaired reject samples regardless of input token limit (tested 400-1060). Root cause: 99.98% of training data has a `memory` field; rejects without memory cause the model to hallucinate a memory section and collapse.

### Lessons Learned (new in run 3)

- **Go sentencepiece must normalize whitespace** — the C++ sentencepiece library applies NFKC normalization (including `\n`->`space`) before tokenization. The Go implementation must do the same or tokenization diverges on any input with newlines.
- **ONNX export numerical validation** — step-by-step comparison of PyTorch vs ONNX outputs caught that the export was correct; the real bug was in tokenization. Always validate the full pipeline end-to-end.
- **Training data distribution gaps kill generalization** — the model's reject failures weren't about context length or model capacity, they were about never seeing inputs without a `memory` field. Added `--drop-memory-pct 20` to augment.
- **Docker container restart hazard** — `docker run -d` bakes the command at creation time. If systemd restarts the container, it replays the original command (including the original `--resume` checkpoint). Added stale checkpoint detection to train.py.
- **Professor forcing noise schedule** — static per-stage schedule is simpler and more predictable than dynamic ramping tied to val loss stalls.

### ONNX Inference Performance

| Model | Size | Tokens/sec (12 threads) |
|-------|------|------------------------|
| fp32  | 120 MB | ~54 tok/s |
| int8  | 31 MB  | ~69 tok/s |

Int8 quantization: 28% faster, 4x smaller, slight quality degradation (1 fewer exact match in 10 samples).

## Run 4 Results

Run 4 added a pointer-generator copy mechanism to the decoder, allowing the model to explicitly copy content tokens from the input instead of regenerating them through the output vocabulary. Same hardware (RTX 2060 6GB), same 5-stage curriculum, same haiku corpus.

### Architecture Changes

- **Copy gate**: `Linear(d_model, 1)` with sigmoid, 385 new parameters. Blends generate distribution with copy distribution from cross-attention weights: `blended = (1-p_copy) * gen_probs + p_copy * copy_probs`
- **NLLLoss**: switched from CrossEntropyLoss since output is now log-probabilities
- **Float32 copy blending**: entire copy block wrapped in `torch.autocast("cuda", enabled=False)` to prevent fp16 NaN

### Training Budget

| Metric | Run 3 | Run 4 | Run 4 / Run 3 |
|--------|-------|-------|---------------|
| Optimizer steps | ~63,500 | ~63,800 | ~same |
| Epochs | 40 | 43 | +7% |
| Best val exact | 60.5% | 80.2% | +33% |
| Best AR exact | 50/50 | 47/50 | -6% (harder val set) |
| Best AR CER | — | 0.02% | new metric |
| Real reject xml_ok | 5/20 (25%) | 17/30 (57%) | +128% |
| Real reject CER | — | 6.1% | new metric |

### Key Metrics

| Epoch | Stage | Train Loss | Val Exact | AR Exact | AR XML OK | CER | WER |
|-------|-------|-----------|-----------|----------|-----------|-----|-----|
| 20    | 1     | —         | —         | 0/50     | 0/50      | —   | —   |
| 30    | 3     | —         | —         | —        | —         | —   | —   |
| 34    | 4     | 0.53      | 95.0%     | 33/50    | 50/50     | —   | —   |
| 35    | 5     | 0.63      | 58.7%     | 46/50    | 50/50     | —   | —   |
| 38    | 5     | 0.62      | 60.5%     | 40/50    | 49/50     | —   | —   |
| **39**| **5** | **0.24**  | **78.2%** | **46/50**| **50/50** | —   | —   |
| **40**| **5** | **0.21**  | **80.2%** | **47/50**| **50/50** | 0.07%| 0.15%|
| 41    | 5     | 0.31      | 73.3%     | 46/50    | 50/50     | 0.02%| 0.13%|
| 42    | 5     | 0.31      | 73.6%     | 45/50    | 50/50     | 0.28%| 0.38%|

Epoch 39 breakthrough: switching from random dictionary word augmentation to shuffle-only (`dict_word_pct=0`) broke through the 60.5% plateau by 17.7 points in a single epoch.

Epochs 41-42 introduced truncation augmentation and higher drop-memory rate, which made the training harder (val_exact dropped) but dramatically improved real reject performance.

### What Worked

- **Copy mechanism** — pointer-generator copy gate lets the model explicitly copy content tokens from source. Content fidelity improved dramatically, especially on code blocks and markdown.
- **Shuffle-only augmentation** (`dict_word_pct=0`) — random dictionary words were training the copy mechanism on easy targets. Shuffling real content words preserves code syntax, markdown, and punctuation complexity. Single biggest improvement: 60.5% → 78.2% val_exact.
- **Truncation augmentation** — `corrupt.DropKeys()` + `corrupt.TruncateJSON()` teaches the model to handle truncated inputs without hallucinating missing fields. Reject xml_ok went from 4/30 to 16/30 in one epoch.
- **Drop memory 50%** — increasing from 20% to 50% reduced memory hallucination on inputs without a memory key.
- **CER/WER metrics** — character error rate and character-weighted word error rate provide continuous quality measurement beyond binary exact/fail. Levenshtein-based, computed per-sample and aggregated.
- **Professor forcing at 0.50-0.65** — higher PF rates in stage 5 close the train/eval gap by training on the model's own predictions.

### What Didn't Work

- **CW boost with copy mechanism** — content weight ramping fought the copy gate. The model reduced loss by increasing p_copy instead of learning structure. Disabled entirely (`content_weight=1.0`).
- **NaN from fp16 copy blending** — `log(blended + 1e-10)` underflows in fp16 (1e-10 below fp16 minimum). Required wrapping entire copy block in float32 with autocast disabled.
- **best_val_loss reset on stage advance** — resetting to infinity caused best.pt to be overwritten with worse models on resume. Removed the reset.

### Remaining Failure Modes

- **Field skipping on long inputs** — model drops entire XML entries (answer, memory) on 1000+ token inputs despite all fields being present in input. Likely an encoder state decay issue (real-valued SSM states lose early-field information).
- **CDATA lookahead** — model must decide `<![CDATA[` before seeing content tokens. If it emits bare `<value>` and content has `<` or `&`, XML is broken. Fix implemented (always-CDATA) but not yet trained.
- **Copy mechanism variable confusion** — copy gate attends to wrong source position, substituting nearby variable names (`state`→`time`, `high_efficiency`→`]_efficiency`).

### Conclusions

The copy mechanism and shuffle-only augmentation were transformative for content fidelity. However, the model hits a ceiling on long inputs where the Mamba 1 real-valued SSM states cannot maintain field-level state tracking. The dominant remaining failures are structural (field skipping, CDATA decisions) rather than content-level — exactly the class of problems that Mamba 3's complex-valued states are designed to address.

## Run 5 Results

Run 5 switched from Mamba 1 to Mamba 3 (complex-valued SSM with RoPE), removed the copy mechanism entirely, and introduced length-stratified token binning with balanced distribution. Same hardware (RTX 2060 6GB). Best model: epoch 69.

### Architecture Changes

- **Mamba 3 encoder-decoder** — complex-valued SSM states with rotary position embeddings, replacing Mamba 1's real-valued states. d_model=384, 6+6 layers, 12 heads, 64 headdim, 64 d_state, ~25M params
- **Copy mechanism removed** — Mamba 3's RoPE provides positional alignment natively. The copy gate in run 4 caused generation head collapse (p_copy→1.0, structural tokens ignored). Removing it freed ~3GB VRAM
- **NormedInProj** — RMSNorm on Mamba 3 in_proj outputs (x and z) required for fp16 stability. Without it, SSM scan overflows at epoch 6
- **QKNormCrossAttention** — RMSNorm on Q and K in cross-attention required for fp16 stability. Without it, attention scores overflow at epoch 4
- **8000 BPE vocab** — IDs 0-15 reserved for structural tokens (pad/BOS/EOS/UNK + XML tags)

### Training Budget

| Metric | Run 4 | Run 5 |
|--------|-------|-------|
| Epochs | 43 | 75 |
| Best val exact | 80.2% | 92.2% |
| Best AR exact | 47/50 | 162/250 (64.8%) |
| Holdout exact | untested | 76/239 (31.8%) |
| Holdout exact+semantic | untested | 109/239 (45.6%) |

### Key Metrics

| Epoch | Stage | Train Loss | Val Exact | AR Exact | AR XML OK | CER | WER | LR |
|-------|-------|-----------|-----------|----------|-----------|-----|-----|----|
| 5     | 1→2   | 0.138     | 25.3%     | —        | —         | —   | —   | 2e-4 |
| 20    | 2     | 0.004     | 85.7%     | —        | —         | —   | —   | 2.5e-5 |
| 29    | 2     | 0.002     | 88.2%     | —        | —         | 0.06% | 0.18% | 2.5e-5 |
| 40    | 6     | —         | 88.9%     | 50/50    | 50/50     | —   | —   | — |
| 53    | 4     | 0.055     | 90.4%     | 251/500  | 278/500   | 1.01% | 1.26% | 2.5e-5 |
| **67**| **4** | **0.056** | **92.2%** | 132/250  | 146/250   | 0.87% | 1.00% | 6.25e-6 |
| **69**| **4** | **0.059** | 92.0%     | **162/250** | **172/250** | **0.88%** | **1.10%** | 2.5e-5 |
| 74    | 4     | 0.054     | 91.7%     | 120/250  | 134/250   | 1.03% | 1.12% | 2.5e-5 |

### Curriculum

| Stage | Description | Gate |
|-------|-------------|------|
| 1 | complexity<=5, fp16 warmup | val_loss < 2.0 |
| 2 | full complexity, clean, CDATA (special_prob=0.10) | val_exact >= 90% |
| 3 | light augmentation (aug=1, corrupt=2%) | val_exact >= 90% |
| 4 | heavy augmentation (aug=2, corrupt=5%, compact=50%) | AR exact >= 90% |
| 5 | heavier (aug=3, corrupt=10%) | AR exact >= 90% |
| 6 | maximum (aug=3, corrupt=15%) | AR exact >= 90% |

Phase transition 3x faster than previous runs due to balanced length distribution exercising the full RoPE position spectrum from epoch 1.

### What Worked

- **Copy mechanism removal** — eliminated generation head collapse. The model learned to generate both structural and content tokens through cross-attention alone.
- **Token-length binning** — replaced char-based length bins with actual BPE token bins in the augment pipeline. Required rewriting Go sentencepiece Encode() from O(n^2) to O(n log n) with doubly-linked list + heap priority queue (37x faster). Unlocked 5+pp val_exact improvement.
- **Balanced distribution** — shorten-pct=30 + compact-pct=50 flattened training length distribution across all token buckets. Previous runs had 74% long samples, 0% short.
- **Online MRT** — 10x loss weighting on full samples with 90-99% token accuracy. Per-token focal loss was too dilute (0.2% of loss). Full-sample weighting worked.
- **Fixed val seed** — VAL_SEED=7777777 eliminated 3-4pp epoch-to-epoch oscillation from random val sampling.
- **Professor forcing schedule** — PF noise ramped from 0.05 (stage 1) to 1.00 (stage 4+). At PF 1.0 all content tokens replaced with model predictions, structural tokens (IDs 0-15) remain ground truth. AR exact improved from 46% to 65%.
- **Triton validation kernel** — single kernel for exact match + ref_len + non-exact buffer across 135K val samples. ~50ms vs minutes of Python.
- **Fanned-out parallel AR decode** — per-operation Triton kernels with grid=(batch, n_tiles) for full 30-SM utilization. 6x faster than the persistent single-SM-per-sample kernel. Fused norm+VMM, VMM+residual, quad-RMSNorm kernels.

### What Didn't Work

- **100% AR training at epoch 74** — positive feedback loop: bad AR decode → large gradients → worse model → worse AR decode. Destabilized within ~4000 batches. Loss spike frequency went 25% → 45% → 75%, with individual batch losses hitting 6.0+. Holdout eval on the interrupt checkpoint: 3.3% exact (destroyed). AR training cannot be bolted on to a model with 74 epochs of teacher-forced weight patterns.
- **Zero-copy weight blob** — replacing param.data with fp16 views into a flat buffer breaks GradScaler (fp16 gradients). Training uses fp32 params with autocast for fp16 forward. The fp32→fp16 copy for the decode kernel is inherent to this split.

### Holdout Eval (epoch 74, 239 unseen samples)

| Bucket | Total | Exact | Sem | XmlOk | Fail | Exact% |
|--------|-------|-------|-----|-------|------|--------|
| 0-250 | 23 | 12 | 0 | 0 | 11 | 52.2% |
| 251-500 | 109 | 20 | 24 | 6 | 59 | 18.3% |
| 501-750 | 60 | 26 | 1 | 7 | 26 | 43.3% |
| 751-1000 | 35 | 15 | 6 | 5 | 9 | 42.9% |
| 1001+ | 12 | 3 | 2 | 4 | 3 | 25.0% |

### Token Accuracy by Position (epoch 75 val)

```
0-99:    99.8%
100-299: 99.4%
300-499: 99.1%
500-799: 98.5%
800-1199: 96.3%
1200+:   28.3%
```

### Infrastructure Built

- Fanned-out parallel Triton decoder (ar_parallel.py) with per-operation kernel launches
- Fused kernels: norm+VMM, VMM+residual, quad-RMSNorm
- Persistent single-SM kernel (ar_kernel.py) for comparison/inference
- Idempotent data generation with marker files and resume support
- Token cache with padded 2D tensor format for fast loading
- Val dataset caching across epochs
- Signal propagation to child subprocesses
- `--force-resume` for checkpoint rollback, `--ckpt` for eval checkpoint override

### Lessons Learned (new in run 5)

- **Binning is data engineering** — changing 5 numbers in a Go slice (length bins) unlocked 5+pp val_exact improvement
- **Val seed matters** — random val sampling created 3-4pp oscillation that masked real improvement
- **Focal-loss MRT too dilute** — 2x on ~5 wrong tokens out of 4500 is unmeasurable. Full-sample 10x weighting needed
- **AR training must be integrated from the start** — the exposure bias gap (92% TF vs 65% AR) cannot be closed by bolting AR training onto a converged TF model. The weight manifold becomes too specialized for always-correct decoder input
- **fp16 stability requires NormedInProj + QKNormCrossAttention** — non-negotiable for Mamba 3. Without both, NaN within 6 epochs
- **Copy mechanism is harmful with RoPE** — Mamba 3 doesn't need it and the copy gate collapses training dynamics

## Run 6 Results

Run 6 tested whether pre-generated data + separated training phases (teacher forcing then autoregressive) could break past the 45% AR holdout ceiling from run 5. Same hardware (RTX 2060 6GB), same architecture (Mamba 3, 384d, 6+6 layers, ~25M params).

### What Changed from Run 5

- **Pre-generated dataset** — all training data generated and tokenized once upfront (`prepare_data.py`), stored as `dataset.pt`. Eliminates per-epoch data generation overhead.
- **Logit soft cap (8.0)** — `cap * tanh(logits / cap)` applied in `decode()`, borrowed from Gemma 2. Prevents fp16 NaN from tied embedding weight scaling inflating logits to [-93, +386].
- **BucketedBatchSampler** — stratified by `max(src_len, tgt_len)` with batch size scaling inversely with bucket max length. Equal representation across length buckets.
- **Simplified curriculum** — 2 stages (clean only, then +corrupt) vs run 5's 6 stages.
- **Removed** copy mechanism, PF noise schedule, content/structural weight boosting, 6-stage curriculum.

### Training Phases

**Phase 1: Full AR (epochs 1-16)**
100% AR decode from step 1, no teacher forcing. Hypothesis: skip the TF trap entirely.

Result: loss dropped from 7.87 to 4.37 over 16 epochs but zero AR exact, zero valid XML, CER stuck at 440-500%. The model couldn't bootstrap from its own garbage outputs. Abandoned.

**Phase 2: Teacher Forcing (epochs 17-32)**
Switched to 0% AR (pure TF). 6x faster training (5.9 it/s vs 0.5 it/s).

| Epoch | Train Loss | CER | WER |
|-------|-----------|-----|-----|
| 17 | 3.17 | 460% | 566% |
| 20 | 1.10 | 206% | 303% |
| 23 | 0.97 | 133% | 210% |
| 28 | 0.24 | 12.2% | 11.0% |
| 29 | 0.18 | **0.86%** | 1.7% |
| 32 | 0.17 | 6.8% | 6.6% |

TF loss floored at ~0.17. CER best was 0.86% (epoch 29) but zero AR exact, zero valid XML across all TF epochs. CER regressed after epoch 29 (0.86% -> 4.8% -> 6.8%), suggesting TF overfitting was degrading AR quality.

**Phase 3: Full AR at low LR (epoch 30, from epoch 29 checkpoint)**
Switched to 100% AR with lr=2e-5 (10x lower than TF phase). Hypothesis: low LR prevents the feedback spiral.

Result: loss bounded at 5-8 (no spiral), but not learning. Batch losses uncorrelated with sequence length — driven by structural complexity of the content, not length. Model either nailed the structure (loss ~0.001) or completely failed (loss ~8.0), regardless of sequence length.

### Inference Analysis (epoch 29 checkpoint)

GPU inference on 10 unseen samples: **4/10 valid XML, 0/10 exact match**.

Failure patterns — all structural, not content:
- Missing root `<object>` tag
- Flattened nested objects (pulls inner keys to top level)
- Cross-contamination between fields (answer text leaking into thought)
- Dropped array elements

Content was nearly perfect — correct CDATA values, correct text, correct escaping. The model knows WHAT to output but fails at WHERE to put structural delimiters.

### Key Metrics

| Epoch | Stage | Train Loss | AR Exact | AR XML | CER | WER | LR | Mode |
|-------|-------|-----------|----------|--------|-----|-----|----|------|
| 16 | 1 | 4.52 | 0/110 | 0/110 | 450% | 556% | 2e-4 | AR |
| 17 | 1 | 3.17 | 0/157 | 0/157 | 460% | 566% | 2e-4 | TF |
| 20 | 1 | 1.10 | 0/452 | 0/452 | 206% | 303% | 2e-4 | TF |
| 29 | 1 | 0.18 | 0/50 | 0/50 | 0.86% | 1.7% | 2e-4 | TF |
| 32 | 1 | 0.17 | 0/50 | 0/50 | 6.8% | 6.6% | 2e-4 | TF |

### Infrastructure Built

- `gpu-infer` command — GPU inference with `--ckpt` override
- `--override-ar-frac` flag — force AR training fraction on resume (checkpoint-authoritative by default, like `--override-lr`)
- Phase transition auto-switch — `ar_train_frac` automatically set to 1.0 when first `ar_xml_ok > 0` (never triggered)
- AR eval sample cap — 50 samples until first valid XML, then scale with loss
- Levenshtein overflow guard — skip pairs > 8192 bytes with max-distance
- Post-eval signal check — stops cleanly after epoch completion instead of starting next epoch
- Batch/seq logging — `B=<batch_size> seq=<max_seq>` in progress lines for bucket visibility

### What Didn't Work

- **Full AR from scratch** — 16 epochs, loss plateau at 4.3, zero AR exact. Can't bootstrap.
- **Pure TF then cold AR switch** — TF drove CER to 0.86% but zero valid XML. Switching to 100% AR at lr=2e-4 caused immediate loss explosion (1.15 -> 7.0 in 500 batches).
- **Full AR at low LR (2e-5)** — prevented spiral but didn't learn. Loss bounded at 5-8 on complex structures.
- **The phase transition approach** — gating AR switch on first valid XML never triggered because TF alone cannot produce valid AR output.

### Conclusion

Run 6 confirmed run 5's finding: the SSM decoder cannot reliably maintain structural state during autoregressive generation. The 45% AR holdout ceiling from run 5 is an architectural limitation, not a training regime problem.

The SSM state is the only mechanism for tracking output structure (nesting depth, open tags, current context). When a structural token is wrong, the state is corrupted for all subsequent positions with no recovery path. A transformer decoder handles this via self-attention over all previous tokens — any position can "look back" and see the open tags. The SSM cannot.

TF trains the model to produce perfect output given perfect context. AR training requires the model to recover from structural errors in its own output. The SSM's compressed state makes error recovery fundamentally harder than for attention-based decoders.

## Run 7 Results

Run 7 replaced the Mamba3 SSM encoder-decoder with a DiT (Diffusion Transformer) to eliminate the autoregressive bottleneck. Instead of sequential token generation, the model sees all positions simultaneously via bidirectional attention and iteratively refines the output. Same hardware (RTX 2060 6GB).

### Architecture Changes

- **DiT bidirectional transformer** — 512d, 10 layers, 8 heads, d_ff=1536, ~45.7M params. adaLN-Zero timestep conditioning (gate params initialized to zero so each layer starts as identity).
- **Factored embedding** — rank-128 bottleneck: (16k, 128) @ (128, 512). Reduces embedding params from 8.2M to 2.2M. `project_to_vocab()` inverts the factorization for output logits.
- **16k BPE vocab** — up from 8k in runs 5/6.
- **RoPE** on Q, K in every layer. Segment embeddings (JSON=0, XML=1).
- **Prefix conditioning** — input is [JSON tokens | corrupted XML tokens], model outputs logits over XML positions only.
- **Length predictor** — mean-pool JSON embeddings → 8-bucket classifier [64, 128, 256, 384, 512, 768, 1024, 1536].
- **Removed** Mamba SSM, cross-attention, copy mechanism, AR decode, professor forcing, all Triton AR kernels.

### Pipeline Evolution

The diffusion pipeline went through five iterations before settling on the final architecture:

1. **CDCD continuous diffusion** — Gaussian noise in embedding space, MSE loss, consistency training with EMA. Loss dropped but CER regressed (265%→679%). Model learned manifold detection instead of contextual correction.
2. **Direct denoising with MSE** — removed consistency training. MSE doesn't track CER — model gets close in embedding space but discretizes to wrong tokens (Voronoi boundary problem).
3. **Cross-entropy on vocab projection** — CE loss aligned with CER for the first time. Required gradient checkpointing for (B, S, 16000) logits tensor. But continuous noise still created off-manifold inputs.
4. **Token-manifold noise** — interpolation between clean and random token embeddings. On-manifold perturbations still didn't cross Voronoi boundaries. Refinement steps produced Δ0.
5. **Fully discrete corruption** (final) — replace t% of token IDs with random tokens, model predicts clean tokens via CE. Inference: random tokens → model → argmax → feed back.

### Training Budget

| Metric | Run 6 | Run 7 |
|--------|-------|-------|
| Epochs | 32 | 44 |
| Gradient steps | — | 28,292 |
| Wall time | — | 16.7h |
| Best eval exact | 0/50 (0%) | 0/500 (0%) |
| Best eval XML OK | 0/50 | 5/500 |
| Best eval CER | 0.86% (TF) | 86.7% (diffusion) |

### Key Metrics

| Epoch | Train Loss | Eval Exact | XML OK | CER | WER |
|-------|-----------|-----------|--------|------|------|
| 1 | 7.59 | 0/500 | 0 | 100.0% | 100.0% |
| 5 | 3.89 | 0/500 | 1 | 87.6% | 92.0% |
| 6 | 3.58 | 0/500 | 0 | **86.7%** | 99.1% |
| 8 | 2.77 | 0/500 | 0 | 87.0% | 101.9% |
| 10 | 1.87 | 0/500 | 0 | 112.9% | 173.2% |
| 20 | 0.98 | 0/500 | 0 | 188.7% | 202.7% |
| 30 | 0.85 | 0/500 | 0 | 231.9% | 288.2% |
| 44 | 0.74 | 0/500 | 0 | 268.7% | 273.5% |

Train loss dropped steadily from 7.59 to 0.74 (perplexity ~2). Eval CER hit a floor at 87% around epoch 6, then diverged catastrophically — reaching 200-500% by epoch 20+ while train loss continued improving.

### Three Phases of Eval Behavior

**Epochs 1-8 (learning)**: CER dropped 100%→87%. Model learned XML structural tokens. Refinement steps showed Δ1-7 token changes. Output had recognizable structure.

**Epochs 9-20 (divergence)**: Train loss dropped 2.5→1.0 but CER spiked to 200-400%. Repetition collapse began — high-frequency tokens amplified over denoising iterations.

**Epochs 20-44 (plateau)**: Train loss plateaued 0.74-0.85. Eval CER oscillated 150-590% per epoch. Output was repetitive tokens with occasional structural fragments. More denoising steps (20-50) made output worse.

### Root Cause: Train/Inference Distribution Mismatch

The model trains on `(corrupted_target, t)` where the corrupted target is always a perturbation of real XML. At inference, the model's input is its own previous output — a completely different error distribution.

At t=0.125 (refinement steps), the model treats input as "87.5% correct." When fed its own predictions, it barely changes anything (Δ0-2), and what it does change trends toward high-frequency tokens. Over multiple steps, this self-reinforcing feedback loop produces pure repetition (`pullpullpull`, `thethethe`).

Testing with 20 denoising steps confirmed: step 1 produces partial structure (correct `<object><entry><key>answer</key>`), but subsequent steps progressively destroy it by amplifying common tokens.

### What Worked

- **Factored embedding** — rank-128 bottleneck reduced params, enabled `project_to_vocab()` for output
- **adaLN-Zero** — clean timestep conditioning, identity initialization
- **`predict_tokens()` method** — chunked vocab projection for VRAM-efficient eval (peak: B×256×16k instead of B×1536×16k)
- **Discrete corruption aligned loss with eval** (epochs 1-8) — CE on token prediction directly measures what eval measures
- **Resumable eval** — eval progress saved to checkpoint via `val_state`, survives interrupts
- **`torch.compile(dynamic=True)`** — works after moving compile after checkpoint load

### What Didn't Work

- **All five continuous/manifold noise variants** — Gaussian noise creates off-manifold inputs. Token-manifold interpolation stays within Voronoi cells. Neither teaches contextual correction.
- **Multi-step discrete refinement** — self-reinforcing feedback loop. The model's own output becomes the input distribution it never trained on.
- **ReduceLROnPlateau** — halved LR 3 times, killed training. Constant LR is standard for diffusion.
- **Consistency training** — vacuous loss, slow signal propagation.

### Conclusion

Discrete diffusion cannot produce valid structured output for this task. The fundamental limitation is threefold:

1. **Independent per-position prediction** — each token is classified independently. The model cannot enforce that `<entry>` at position 40 requires `</entry>` at position 55. XML requires coordinated structural decisions that per-position argmax cannot provide.
2. **Train/inference distribution mismatch** — training sees corrupted versions of real XML. Inference sees the model's own (wrong) predictions. The model optimizes the training task (lower loss) without improving the inference task (lower CER).
3. **Pointwise discretization bottleneck** — confirmed by CoDAR (Shen et al. 2026): pointwise projection from embeddings to tokens has an irreducible optimality gap due to conditional total correlation between positions.

The 45% AR ceiling from runs 5-6 was an SSM state limitation. The 0% diffusion ceiling from run 7 is a more fundamental limitation of independent per-position token prediction without holistic sequence representation.

## Data Pipelines

### Haiku-First Pipeline (Run 2, current)

Uses real LLM outputs (~100k haiku samples) as the data source. Each epoch samples a percentage of the corpus and augments:

```bash
# Build augment binary and start training
./training/run.sh train
```

The `cmd/augment` tool handles sampling and augmentation:
- Loads haiku JSONL from `data/haiku/` (corpus.jsonl or individual shards)
- Length-stratified sampling: bins samples by character length, samples equally from each bin
- For each sample, emits the natural pair + N augmented variants
- Augmented variants replace string values with dictionary words or shuffled content
- Configurable XML special character injection (`-special-prob`)
- Configurable JSON corruption (`-corrupt-pct`) with bracket swaps, drops, and multi-bracket runs
- Configurable compact JSON output (`-compact-pct`) — single-line with escaped newlines
- Minimum character length filter (`-min-chars`)

**Haiku curriculum stages (5-stage, run 3):**

| Stage | Type | Aug Ratio | Special Prob | Corrupt % | Compact % | Sample % |
|-------|------|-----------|-------------|-----------|-----------|----------|
| 1     | answer | 0 (natural only) | 0.0  | 0   | 50 | 50       |
| 2     | tool   | 0 (natural only) | 0.0  | 0   | 50 | 50       |
| 3     | all    | 1:5       | 0.15        | 0         | 50 | 5        |
| 4     | all    | 1:10      | 0.30        | 10        | 50 | 5        |
| 5     | all    | 1:10      | 0.35        | 15        | 50 | 5        |

Auto-advance when AR exact match >= 55% for 2 consecutive epochs. LR resets to half-base on stage advance.

**Professor forcing noise schedule** (static, per-stage):

| Stage | PF Noise |
|-------|----------|
| 1     | 0.30     |
| 2     | 0.30     |
| 3     | 0.35     |
| 4     | 0.40     |
| 5     | 0.50     |

**Key training innovations:**

- **Sawtooth content weight** — content token loss weight starts at 1.0 (learn structure first) and ramps up adaptively when val improvement stalls. Resets to 1.0 on stage advance.
- **Length-stratified sampling** — bins corpus by character length and samples equally from each bin, ensuring long samples (~10-26% of training) aren't drowned out by the short majority.
- **Compact JSON from stage 1** — 50% of samples are single-line compact JSON throughout training. Real LLM output is compact; the model learns both formats from the start.
- **Multi-bracket corruption** — generates `]}}}}` runs matching real LLM failure patterns, not just single bracket swaps.
- **LR warm restart on stage advance** — resets learning rate to half-base and rebuilds the plateau scheduler, preventing accumulated LR decay from blocking learning on new data distributions.
- **Adaptive validation budget** — early stages validate on a small prefix (50-5000 batches). Full validation runs only at stage 5.
- **Variable schema** — 20% of augmented samples have their `memory` field dropped (`--drop-memory-pct 20`), teaching the model to handle varied JSON schemas. Without this, the model hallucinated a memory section on real inputs that lacked one.

### Synthetic Pipeline (Run 1, legacy)

Generates random agent response JSON with configurable structure depth, embedded code in multiple languages, and progressive corruption:

```bash
go run ./cmd/generate -stage 1 -train 200000 -val 10000
```

## Input Schema

Training data follows a fixed agent response schema:

```json
{
  "thought": "reasoning about the user's request...",
  "answer": "response text, often markdown with code blocks...",
  "tool": {
    "tool_name": "execute_sql",
    "arguments": {
      "query": "SELECT * FROM users WHERE active = true"
    }
  },
  "memory": [
    "User prefers Python for scripting tasks.",
    "The database is PostgreSQL on port 5432."
  ]
}
```

- `thought` — the agent's reasoning (always present)
- `answer` — text response, may contain markdown with fenced code blocks, tables, and lists (null when a tool is called)
- `tool` — tool invocation with nested `tool_name` and `arguments`; arguments vary from single-field (`{"query": "..."}`) to multi-field with nested objects, arrays, and mixed types. Code execution tools embed realistic snippets. (null when an answer is given)
- `memory` — contextual notes carried across interactions (optional — some samples omit this field entirely)

## XML Schema

Six element names. No attributes, no declarations, no namespaces.

```xml
<object>
  <entry>
    <key>query</key>
    <value><![CDATA[SELECT * FROM users WHERE name = 'alice']]></value>
  </entry>
  <entry>
    <key>count</key>
    <value>42</value>
  </entry>
  <entry>
    <key>tags</key>
    <value>
      <array>
        <value>admin</value>
        <value>active</value>
      </array>
    </value>
  </entry>
</object>
```

- **Object** -> `<object>` containing `<entry>` children
- **Entry** -> `<entry>` containing `<key>` + `<value>`
- **Array** -> `<array>` containing `<value>` children
- **String values** containing `<`, `>`, `&`, or `]]>` are wrapped in `<![CDATA[...]]>`
- **Numbers, booleans, null** -> text content inside `<value>`

## Project Structure

```
transmutation/
├── cmd/
│   ├── augment/       # Haiku augmentation pipeline CLI
│   ├── enrich/        # Tool-call argument enrichment
│   ├── generate/      # Synthetic training data generator + haiku wrapper
│   ├── repair/        # Reject repair validation (dual-LLM agreement)
│   ├── infer/         # Go ONNX inference CLI + Dockerfile
│   └── collage/       # Visual sample collage generator
├── pkg/
│   ├── agent/         # Agent response schema generator (curriculum stages)
│   ├── jsongen/       # Random JSON structure builder
│   ├── languages/     # Embedded code snippet generators (SQL, Python, JS, Go, etc.)
│   ├── corrupt/       # JSON corrupter (quotes, commas, comments, wrappers, etc.)
│   ├── randtext/      # Random text + augmentation helpers (dict words, special chars)
│   ├── xmlconv/       # Deterministic JSON -> XML converter
│   └── sentencepiece/ # SentencePiece BPE tokenizer (pure Go)
├── training/          # Python training code (runs in Docker)
│   ├── Dockerfile
│   ├── model.py       # Mamba encoder-decoder
│   ├── train.py       # Training loop with content-weighted loss
│   ├── export.py      # ONNX export (single-step decoder)
│   ├── infer_cpu.py   # Python CPU inference
│   ├── run.sh         # Orchestrates all training steps
│   └── wheels/        # Pre-built mamba_ssm + causal_conv1d wheels
├── models/
│   ├── run1/          # Archived run 1 (synthetic data, 53 epochs)
│   ├── run2/          # Archived run 2 (haiku data, 8-stage, 39 epochs)
│   ├── run3/          # Archived run 3 (haiku data, 5-stage, 40 epochs)
│   ├── run4/          # Archived run 4 (copy mechanism, 43 epochs)
│   └── run5/          # Current run (Mamba 3, no copy, 75 epochs)
├── data/
│   └── haiku/         # ~140k real LLM haiku outputs (corpus.jsonl)
├── scripts/
│   ├── gen_haiku.sh   # Generate haiku samples via Claude CLI
│   └── repair_rejects.sh  # Repair broken samples via dual-LLM passes
├── go.mod
└── go.sum
```

## Usage

### Prerequisites

- Go 1.24+
- Docker with NVIDIA Container Toolkit (for training)
- GPU with CUDA support (for training; inference is CPU)

### Train

```bash
./training/run.sh train
```

Runs tokenizer training (if needed), then model training in a Docker container with GPU passthrough. Checkpoints are saved to `models/`. Supports auto-resume from interrupts (SIGUSR1 saves a mid-epoch checkpoint).

See `./training/run.sh` for all commands: `tokenizer`, `train`, `stop`, `checkpoint`, `export`, `infer`, `go-infer`, `status`, `logs`.

### Export to ONNX

```bash
./training/run.sh export
```

Exports encoder and decoder to `models/onnx/`. The decoder uses a single-step API with explicit Mamba state — the autoregressive loop runs in the caller, not in the ONNX graph.

### Run Inference

```bash
# Python CPU inference (10 samples)
./training/run.sh infer 10

# Go ONNX inference (10 samples)
./training/run.sh go-infer 10
```

### Training Management

```bash
./training/run.sh status       # Show checkpoints, metrics, container state
./training/run.sh logs         # Follow training output
./training/run.sh checkpoint   # Save checkpoint without stopping (SIGUSR1)
./training/run.sh stop         # Graceful stop (saves checkpoint, 120s timeout)
```

## Corruption Types

The corrupter applies a random subset of these to valid JSON:

- **Quote stripping** — remove quotes from keys or identifier-like values
- **Comma dropping** — remove random commas between elements
- **Colon dropping** — remove colons between key-value pairs
- **Comment insertion** — `//` line comments and `/* */` block comments
- **Preamble/postamble** — wrapping text ("Here is the JSON response:", etc.)
- **Trailing commas** — after last elements
- **Whitespace mangling** — inconsistent indentation
- **Bracket issues** — single swaps (`}`↔`]`), drops, duplicates, and multi-bracket runs (`]}}}}`) matching real LLM failure patterns

## Embedded Languages

Code generators produce syntactically realistic snippets with high combinatorial entropy via compositional identifier generation. Supported languages: SQL, Python, JavaScript, Go, Shell, HTML, Markdown, CSS, YAML, and nested JSON-as-string.

## ONNX Model API

**Encoder** — called once per input:
- Input: `src_ids` (1, src_len) int64
- Outputs: `all_k` (6, 6, src_len, 64) float32, `all_v` (6, 6, src_len, 64) float32

The encoder pre-computes cross-attention K/V projections for all 6 decoder layers (6 heads, head_dim=64), so the decoder only needs Q projection per step.

**Decoder** — called once per output token (autoregressive loop in caller):
- Inputs: `tgt_token` (1, 1) int64, `all_k` (6, 6, src_len, 64) float32, `all_v` (6, 6, src_len, 64) float32, `all_h` (6, 768, 16) float32, `all_conv` (6, 768, 3) float32
- Outputs: `logits` (1, 8000) float32, `all_h_out` (6, 768, 16) float32, `all_conv_out` (6, 768, 3) float32

Initialize `all_h` and `all_conv` to zeros. Feed BOS token first. Greedy decode: take argmax of logits, stop at EOS. Copy `all_h_out`/`all_conv_out` back into `all_h`/`all_conv` each step. `all_k`/`all_v` are read-only.

## License

Copyright (C) 2026 Nicholas Perez

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE) for details.
