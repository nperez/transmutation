# Transmutation

Transforms broken JSON into valid XML using a neural network.

LLM agents instructed to respond in JSON frequently produce broken output — missing quotes, dropped commas/colons, `//` comments, explanatory text wrapping the JSON, and embedded code (SQL, Python, etc.) inside string values that creates ambiguous boundaries. Transmutation sits between LLM output and response parsing, converting the mess into valid XML that can be deterministically parsed downstream.

XML was chosen as the output format because it cleanly handles embedded content via CDATA sections, avoiding the escaping nightmare of trying to produce valid JSON containing embedded code.

## Architecture

**Model**: Mamba-based seq2seq encoder-decoder (~25M parameters)
- Mamba (state space model) for linear-time inference on long sequences
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
│   └── run4/          # Current run
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
