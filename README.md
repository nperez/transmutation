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

**Data generation**: Go. Generates agent response JSON with embedded code, markdown, and tool calls. Applies configurable corruption and produces target XML.

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
│   ├── generate/       # Training data generator CLI
│   ├── infer/          # Go ONNX inference CLI + Dockerfile
│   └── collage/        # Visual sample collage generator
├── pkg/
│   ├── agent/          # Agent response schema generator (curriculum stages)
│   ├── jsongen/        # Random JSON structure builder (legacy)
│   ├── languages/      # Embedded code snippet generators (SQL, Python, JS, Go, etc.)
│   ├── corrupt/        # JSON corrupter (quotes, commas, comments, wrappers, etc.)
│   ├── xmlconv/        # Deterministic JSON -> XML converter
│   └── sentencepiece/  # SentencePiece BPE tokenizer (pure Go)
├── training/           # Python training code (runs in Docker)
│   ├── Dockerfile
│   ├── model.py        # Mamba encoder-decoder
│   ├── train.py        # Training loop with content-weighted loss
│   ├── export.py       # ONNX export (single-step decoder)
│   ├── infer_cpu.py    # Python CPU inference
│   ├── run.sh          # Orchestrates all training steps
│   └── wheels/         # Pre-built mamba_ssm + causal_conv1d wheels
├── go.mod
└── go.sum
```

## Usage

### Prerequisites

- Go 1.24+
- Docker with NVIDIA Container Toolkit (for training)
- GPU with CUDA support (for training; inference is CPU)

### Generate Training Data

```bash
# Agent schema (curriculum stage 1-5)
go run ./cmd/generate -stage 1 -train 200000 -val 10000

# Legacy random JSON (stage 0)
go run ./cmd/generate -train 200000 -val 10000
```

Writes JSONL shards to `data/train/` and `data/val/`.

### Train

```bash
./training/run.sh train
```

Runs tokenizer training (if needed), then model training in a Docker container with GPU passthrough. Checkpoints are saved to `models/`. Supports auto-resume from interrupts (SIGUSR1 saves a mid-epoch checkpoint).

See `./training/run.sh` for all commands: `tokenizer`, `train`, `infer`, `export`, `go-infer`, `all`.

### Export to ONNX

```bash
./training/run.sh export
```

Exports encoder and decoder to `models/onnx/`. The decoder uses a single-step API with explicit Mamba state — the autoregressive loop runs in the caller, not in the ONNX graph.

### Run Inference (Go + ONNX)

```bash
go run ./cmd/generate -stdout -train 30 -val 0 | ./training/run.sh go-infer 10
```

Or build and run the inference Docker image directly:

```bash
docker build -t transmutation-infer -f cmd/infer/Dockerfile .
echo '{"input": "{\"name\": \"Alice\"}", "target": ""}' | \
  docker run --rm -i -v ./models:/app/models:ro transmutation-infer \
    -encoder models/onnx/encoder.onnx \
    -decoder models/onnx/decoder.onnx \
    -tokenizer models/tokenizer.model \
    -ort-lib /usr/local/lib/libonnxruntime.so \
    -n 1
```

### Run Inference (Python, CPU)

```bash
go run ./cmd/generate -stdout -train 30 -val 0 | ./training/run.sh infer 10
```

## Curriculum Training

Training follows a staged curriculum that progressively increases complexity:

1. **Clean simple** — text answers and markdown, clean JSON, learn the schema
2. **Tool calls** — add code execution and utility tool invocations with embedded code
3. **Full mix** — all content types together (simple, markdown, code tools, utility tools)
4. **Subtle corruption** — introduce light JSON corruption (~5% of samples)
5. **Wrappers** — add preamble/postamble text around the JSON

The loss function weights content tokens (strings, numbers) higher than structural XML tokens to penalize mistakes in the actual data the model is copying.

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
