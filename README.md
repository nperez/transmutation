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
- **Haiku-first** (current) — samples from a corpus of ~100k real LLM haiku outputs, augments by replacing string values with dictionary words or shuffled content while preserving JSON structure, with configurable XML special character injection and corruption
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
- **Content-weighted loss** (10x weight on string/number tokens vs structural XML tokens) was critical — without it the model would copy structural tokens perfectly but mangle the actual data values.
- **Professor forcing** (teacher forcing with scheduled sampling) improved AR eval performance significantly vs pure teacher forcing.
- **Token noise** (0.15 probability of random token substitution in inputs during training) acted as regularization and improved robustness.
- **Batch size 2 + grad_accum 16** was the practical max for 6GB VRAM with mixed precision.

## Data Pipelines

### Haiku-First Pipeline (Run 2, current)

Uses real LLM outputs (~100k haiku samples) as the data source. Each epoch samples a percentage of the corpus and augments:

```bash
# Build augment binary and start training
./training/run.sh train
```

The `cmd/augment` tool handles sampling and augmentation:
- Loads haiku JSONL from `data/haiku/`
- Samples N% of corpus per epoch (disjoint train/val via seed offset)
- For each sample, emits the natural pair + N augmented variants
- Augmented variants replace string values with dictionary words or shuffled content
- Configurable XML special character injection (`-special-prob`)
- Configurable JSON corruption (`-corrupt-pct`)

**Haiku curriculum stages:**

| Stage | Aug Ratio | Special Prob | Corrupt % | Sample % |
|-------|-----------|-------------|-----------|----------|
| 1     | 0 (natural only) | 0.0  | 0         | 10       |
| 2     | 1:5       | 0.15        | 0         | 5        |
| 3     | 1:10      | 0.30        | 0         | 5        |
| 4     | 1:10      | 0.40        | 10        | 5        |
| 5     | 1:10      | 0.40        | 20        | 5        |

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
- `tool` — tool invocation with arguments; code execution tools (`execute_sql`, `execute_python`, `execute_javascript`, `execute_shell`, `execute_go`) embed realistic code snippets; utility tools (`search`, `read_file`, `write_file`, `http_request`) have simple arguments (null when an answer is given)
- `memory` — contextual notes carried across interactions

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
│   ├── generate/      # Synthetic training data generator CLI (legacy)
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
│   └── run1/          # Archived run 1 (synthetic data, 53 epochs)
├── data/
│   └── haiku/         # ~100k real LLM haiku outputs (JSONL)
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
- **Bracket issues** — mismatched or missing brackets

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
