#!/bin/bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_IMAGE="transmutation-train"
INFER_IMAGE="transmutation-infer"
CONTAINER_NAME="transmutation-train"

# ── Wheels ───────────────────────────────────────────────────────────────────

WHEELS_DIR="$SCRIPT_DIR/wheels"
CAUSAL_CONV1D_VER="1.5.0.post8"
MAMBA_SSM_VER="2.2.4"
WHL_SUFFIX="cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64"

fetch_wheels() {
    mkdir -p "$WHEELS_DIR"
    local cc_whl="causal_conv1d-${CAUSAL_CONV1D_VER}+${WHL_SUFFIX}.whl"
    local mm_whl="mamba_ssm-${MAMBA_SSM_VER}+${WHL_SUFFIX}.whl"

    if [ ! -f "$WHEELS_DIR/$cc_whl" ]; then
        echo "Downloading causal-conv1d wheel..."
        curl -fSL -o "$WHEELS_DIR/$cc_whl" \
            "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VER}/$(echo "$cc_whl" | sed 's/+/%2B/g')"
    fi
    if [ ! -f "$WHEELS_DIR/$mm_whl" ]; then
        echo "Downloading mamba-ssm wheel..."
        curl -fSL -o "$WHEELS_DIR/$mm_whl" \
            "https://github.com/state-spaces/mamba/releases/download/v${MAMBA_SSM_VER}/$(echo "$mm_whl" | sed 's/+/%2B/g')"
    fi
}

# ── Build ────────────────────────────────────────────────────────────────────

GENERATE_BIN="$PROJECT_DIR/tmp/generate"

build_generator() {
    local bin="$GENERATE_BIN"
    if [ ! -f "$bin" ] || [ "$(find "$PROJECT_DIR/cmd/generate" "$PROJECT_DIR/pkg" -newer "$bin" 2>/dev/null)" ]; then
        echo "Building generator binary..."
        (cd "$PROJECT_DIR" && CGO_ENABLED=0 go build -o "$bin" ./cmd/generate/)
    fi
}

build_train() {
    fetch_wheels
    build_generator
    echo "Building training image..."
    docker build -t "$TRAIN_IMAGE" "$SCRIPT_DIR"
}

build_infer() {
    echo "Building inference image..."
    docker build -t "$INFER_IMAGE" -f "$PROJECT_DIR/cmd/infer/Dockerfile" "$PROJECT_DIR"
}

# ── Docker helpers ───────────────────────────────────────────────────────────

# Run a GPU container in the foreground (blocking).
run_gpu() {
    docker run --rm --gpus all \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        -v "$PROJECT_DIR/tmp/generate:/app/generate:ro" \
        "$TRAIN_IMAGE" \
        "$@"
}

# Run a GPU container detached. Returns container ID.
run_gpu_detached() {
    docker run -d --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        -v "$PROJECT_DIR/tmp/generate:/app/generate:ro" \
        "$TRAIN_IMAGE" \
        "$@"
}

# Run a CPU container with stdin passthrough (for piped input).
run_cpu_stdin() {
    docker run --rm -i \
        -v "$PROJECT_DIR/data:/app/data:ro" \
        -v "$PROJECT_DIR/models:/app/models:ro" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        "$TRAIN_IMAGE" \
        "$@"
}

# Find the training container (running or stopped).
find_train_container() {
    docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.ID}}' | head -1 || true
}

# ── Auto-resume logic ───────────────────────────────────────────────────────

find_resume_flag() {
    # Find the most recent checkpoint of any type by modification time.
    local latest
    latest=$(ls -t "$PROJECT_DIR/models"/interrupt_*.pt \
                    "$PROJECT_DIR/models"/interrupt.pt \
                    "$PROJECT_DIR/models"/best.pt \
                    "$PROJECT_DIR/models"/epoch_*.pt \
                    2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then
        echo "--resume models/$(basename "$latest")"
    fi
}

# ── Commands ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    build)
        build_train
        ;;

    train)
        build_train
        shift

        # Bail if already running.
        if docker ps --filter "name=$CONTAINER_NAME" --format '{{.ID}}' | grep -q .; then
            echo "Training is already running. Use 'stop' first."
            exit 1
        fi

        # Clean up stopped container with same name if present.
        docker rm "$CONTAINER_NAME" 2>/dev/null || true

        RESUME_FLAG=$(find_resume_flag)
        if [ -n "$RESUME_FLAG" ]; then
            LR="1e-4"
            WARMUP="500"
            echo "Resuming: $RESUME_FLAG (lr=$LR, warmup=$WARMUP)"
        else
            LR="3e-4"
            WARMUP="2000"
            echo "Starting fresh training (lr=$LR, warmup=$WARMUP)"
        fi

        CID=$(run_gpu_detached training/train.py \
            --data-dir data \
            --tokenizer models/tokenizer.model \
            --output-dir models \
            --batch-size 2 \
            --grad-accum 16 \
            --max-src-len 1536 \
            --max-tgt-len 2048 \
            --epochs 100 \
            --train-samples 50000 \
            --val-samples 2500 \
            --lr "$LR" \
            --warmup-steps "$WARMUP" \
            --save-every 5 \
            --fp16 \
            --max-stage 5 \
            --stage-advance-ar 0.5 \
            --stage-patience 3 \
            --content-weight 10.0 \
            --token-noise 0.15 \
            --ar-eval-samples 50 \
            $RESUME_FLAG \
            "$@")

        echo "Container: $CID"
        echo "Use './training/run.sh logs' to follow output."
        echo "Use './training/run.sh status' to check progress."
        ;;

    stop)
        CID=$(find_train_container)
        if [ -z "$CID" ]; then
            echo "No training container found."
            exit 0
        fi
        echo "Sending SIGTERM (will checkpoint and exit)..."
        docker stop -t 120 "$CID"
        echo "Stopped."
        ;;

    kill)
        CID=$(find_train_container)
        if [ -z "$CID" ]; then
            echo "No training container found."
            exit 0
        fi
        echo "Killing training container..."
        docker kill "$CID"
        docker rm "$CID" 2>/dev/null || true
        echo "Killed."
        ;;

    checkpoint)
        CID=$(find_train_container)
        if [ -z "$CID" ]; then
            echo "No training container found."
            exit 1
        fi
        echo "Sending SIGUSR1 (save checkpoint, keep training)..."
        docker kill -s USR1 "$CID"
        echo "Signal sent. Check logs for confirmation."
        ;;

    logs)
        CID=$(find_train_container)
        if [ -z "$CID" ]; then
            echo "No training container found."
            exit 1
        fi
        docker logs -f "$CID" 2>&1 | tr '\r' '\n'
        ;;

    status)
        echo "=== Checkpoints ==="
        if ls "$PROJECT_DIR/models"/*.pt 1>/dev/null 2>&1; then
            docker run --rm \
                -v "$PROJECT_DIR/models:/app/models:ro" \
                "$TRAIN_IMAGE" -c "
import torch, os
for f in sorted(os.listdir('models')):
    if not f.endswith('.pt'): continue
    path = os.path.join('models', f)
    sz_mb = os.path.getsize(path) / 1024 / 1024
    c = torch.load(path, map_location='cpu', weights_only=True)
    ep = c.get('epoch', '?')
    gs = c.get('global_step', '?')
    vl = c.get('best_val_loss', None)
    ec = c.get('epoch_complete', False)
    es = c.get('epoch_step', 0)
    stg = c.get('stage', '?')
    parts = [f'epoch={ep}', f'step={gs}', f'stage={stg}']
    if vl is not None: parts.append(f'best_val={vl:.4f}')
    if not ec and es: parts.append(f'batch={es}')
    print(f'  {f:<40s} {sz_mb:.0f}MB  {\" \".join(parts)}')
"
        else
            echo "  (none)"
        fi
        echo
        echo "=== ONNX Models ==="
        ls -lh "$PROJECT_DIR/models/onnx"/*.onnx 2>/dev/null || echo "  (none)"
        echo
        echo "=== Training Log ==="
        if [ -f "$PROJECT_DIR/models/training_log.json" ]; then
            python3 -c "
import json, sys
entries = json.load(open('$PROJECT_DIR/models/training_log.json'))
for e in entries[-5:]:
    ar = f\"ar={e.get('ar_exact','?')}/{e.get('ar_total','?')}exact {e.get('ar_xml_ok','?')}/{e.get('ar_total','?')}xml\" if 'ar_total' in e else ''
    print(f\"  epoch={e['epoch']} train={e['train_loss']:.4f} val={e['val_loss']:.4f} exact={e.get('val_exact_match',0):.1%} {ar} lr={e['lr']:.2e}\")
" 2>/dev/null || echo "  (empty or parse error)"
        else
            echo "  (no training_log.json yet)"
        fi
        echo
        echo "=== Container ==="
        CID=$(find_train_container)
        if [ -n "$CID" ]; then
            STATE=$(docker inspect --format '{{.State.Status}}' "$CID" 2>/dev/null || echo "unknown")
            echo "  $CONTAINER_NAME ($CID): $STATE"
            if [ "$STATE" = "running" ]; then
                echo
                echo "=== Recent Output ==="
                docker logs --tail 5 "$CID" 2>&1 | tr '\r' '\n' | grep -v '^$' | tail -5
            fi
        else
            echo "  (not running)"
        fi
        ;;

    tokenizer)
        build_train
        echo "Training tokenizer..."
        run_gpu training/tokenizer_train.py \
            --data-dir data \
            --output-dir models \
            --vocab-size 8000
        ;;

    export)
        build_train
        shift
        if [ -n "${1:-}" ]; then
            CKPT="$1"
        else
            CKPT=$(find_resume_flag | sed 's/--resume //')
            CKPT="${CKPT:-models/best.pt}"
        fi
        echo "Exporting $CKPT to ONNX..."
        run_gpu training/export.py \
            --checkpoint "$CKPT" \
            --tokenizer models/tokenizer.model \
            --output-dir models/onnx
        ;;

    infer)
        build_train
        build_generator
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        STAGE="${1:-3}"
        if [ -n "${1:-}" ]; then shift; fi

        RESUME_CKPT=$(find_resume_flag | sed 's/--resume //')
        CHECKPOINT="${RESUME_CKPT:-models/best.pt}"

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        TMPFILE="$PROJECT_DIR/tmp/infer_input_$$.jsonl"
        trap "rm -f '$TMPFILE'" EXIT

        echo "Generating $GEN_COUNT candidates (stage $STAGE)..."
        "$GENERATE_BIN" -stage "$STAGE" -stdout -train "$GEN_COUNT" -val 0 -seed "$$" > "$TMPFILE"

        echo "CPU inference: $CHECKPOINT ($N_SAMPLES samples, stage $STAGE)..."
        cat "$TMPFILE" | run_cpu_stdin training/infer_cpu.py "$CHECKPOINT" "$N_SAMPLES" "$@"
        ;;

    go-infer)
        build_infer
        build_generator
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        STAGE="${1:-3}"
        if [ -n "${1:-}" ]; then shift; fi

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        TMPFILE="$PROJECT_DIR/tmp/infer_input_$$.jsonl"
        trap "rm -f '$TMPFILE'" EXIT

        echo "Generating $GEN_COUNT candidates (stage $STAGE)..."
        "$GENERATE_BIN" -stage "$STAGE" -stdout -train "$GEN_COUNT" -val 0 -seed "$$" > "$TMPFILE"

        echo "Go ONNX inference ($N_SAMPLES samples from stage $STAGE)..."
        docker run --rm \
            -v "$PROJECT_DIR/models:/app/models:ro" \
            -v "$TMPFILE:/app/input.jsonl:ro" \
            --entrypoint sh \
            "$INFER_IMAGE" \
            -c "cat /app/input.jsonl | infer \
                -encoder models/onnx/encoder_int8.onnx \
                -decoder models/onnx/decoder_int8.onnx \
                -tokenizer models/tokenizer.model \
                -ort-lib /usr/local/lib/libonnxruntime.so \
                -n $N_SAMPLES \
                $*"
        ;;

    all)
        build_train
        echo "=== Step 1: Tokenizer ==="
        run_gpu training/tokenizer_train.py \
            --data-dir data \
            --output-dir models \
            --vocab-size 8000

        echo "=== Step 2: Train ==="
        run_gpu training/train.py \
            --data-dir data \
            --tokenizer models/tokenizer.model \
            --output-dir models \
            --batch-size 2 \
            --grad-accum 16 \
            --max-src-len 1536 \
            --max-tgt-len 2048 \
            --epochs 30 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16

        echo "=== Step 3: Export ==="
        run_gpu training/export.py \
            --checkpoint models/best.pt \
            --tokenizer models/tokenizer.model \
            --output-dir models/onnx

        echo "=== Done ==="
        ;;

    haiku-gen)
        build_generator
        shift
        N="${1:-100}"
        echo "Generating $N samples via Haiku..."
        "$PROJECT_DIR/scripts/gen_haiku.sh" "$N" "$PROJECT_DIR/data/haiku"
        ;;

    haiku-clean)
        echo "Cleaning haiku corpus..."
        rm -rf "$PROJECT_DIR/data/haiku"
        echo "Done."
        ;;

    clean-generated)
        echo "Cleaning generated train/val data (preserving haiku)..."
        rm -rf "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val"
        echo "Done."
        ;;

    clean-all)
        echo "Cleaning all data (train, val, haiku)..."
        rm -rf "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val" "$PROJECT_DIR/data/haiku"
        echo "Done."
        ;;

    help|*)
        cat <<'USAGE'
Usage: ./training/run.sh <command> [args...]

Training:
  train [flags]     Start training (detached). Auto-resumes from checkpoint.
  stop              Graceful stop (checkpoints, then exits).
  kill              Force kill (no checkpoint).
  checkpoint        Save a checkpoint without stopping.
  logs              Follow training output.
  status            Show checkpoints, metrics, container state.

Inference:
  infer [N] [stage]     CPU inference on N samples (default 10, stage 3).
  go-infer [N] [stage]  Go ONNX inference on N samples (default 10, stage 3).

Data:
  haiku-gen [N]     Generate N samples via Claude Haiku (default 100).
  haiku-clean       Remove haiku corpus.
  clean-generated   Remove train/val data (preserves haiku).
  clean-all         Remove all data (train, val, haiku).

Other:
  export [ckpt]     Export checkpoint to ONNX (default: best.pt).
  tokenizer         Train the tokenizer.
  build             Build the training Docker image.
  all               Full pipeline: tokenizer → train → export.
  help              This message.
USAGE
        ;;
esac
