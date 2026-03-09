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

build_train() {
    fetch_wheels
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
        -v "$PROJECT_DIR/data:/app/data:ro" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        "$TRAIN_IMAGE" \
        "$@"
}

# Run a GPU container detached. Returns container ID.
run_gpu_detached() {
    docker run -d --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_DIR/data:/app/data:ro" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
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
    if [ -f "$PROJECT_DIR/models/interrupt.pt" ]; then
        echo "--resume models/interrupt.pt"
    else
        local latest
        latest=$(ls -v "$PROJECT_DIR/models"/epoch_*.pt 2>/dev/null | tail -1 || true)
        if [ -n "$latest" ]; then
            echo "--resume models/$(basename "$latest")"
        fi
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
            echo "Resuming: $RESUME_FLAG"
        else
            echo "Starting fresh training."
        fi

        CID=$(run_gpu_detached training/train.py \
            --data-dir data \
            --tokenizer models/tokenizer.model \
            --output-dir models \
            --batch-size 1 \
            --grad-accum 32 \
            --max-src-len 1536 \
            --max-tgt-len 2048 \
            --epochs 30 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16 \
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
        ls -lh "$PROJECT_DIR/models"/*.pt 2>/dev/null || echo "  (none)"
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
        CKPT="${1:-models/best.pt}"
        echo "Exporting $CKPT to ONNX..."
        run_gpu training/export.py \
            --checkpoint "$CKPT" \
            --tokenizer models/tokenizer.model \
            --output-dir models/onnx
        ;;

    infer)
        build_train
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        if [ -f "$PROJECT_DIR/models/best.pt" ]; then
            CHECKPOINT="models/best.pt"
        else
            LATEST=$(ls -v "$PROJECT_DIR/models"/epoch_*.pt 2>/dev/null | tail -1 || true)
            if [ -n "$LATEST" ]; then
                CHECKPOINT="models/$(basename "$LATEST")"
            elif [ -f "$PROJECT_DIR/models/interrupt.pt" ]; then
                CHECKPOINT="models/interrupt.pt"
            else
                echo "Error: no checkpoint found."
                exit 1
            fi
        fi

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        echo "CPU inference: $CHECKPOINT ($N_SAMPLES samples)..."
        cd "$PROJECT_DIR" && go run ./cmd/generate -stage 1 -stdout -train "$GEN_COUNT" -val 0 -seed "$RANDOM" \
            | run_cpu_stdin training/infer_cpu.py "$CHECKPOINT" "$N_SAMPLES" "$@"
        ;;

    go-infer)
        build_infer
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        echo "Go ONNX inference ($N_SAMPLES samples)..."
        cd "$PROJECT_DIR" && go run ./cmd/generate -stage 1 -stdout -train "$GEN_COUNT" -val 0 -seed "$RANDOM" \
            | docker run --rm -i \
                -v "$PROJECT_DIR/models:/app/models:ro" \
                "$INFER_IMAGE" \
                -encoder models/onnx/encoder_int8.onnx \
                -decoder models/onnx/decoder_int8.onnx \
                -tokenizer models/tokenizer.model \
                -ort-lib /usr/local/lib/libonnxruntime.so \
                -n "$N_SAMPLES" \
                "$@"
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
            --batch-size 1 \
            --grad-accum 32 \
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
  infer [N]         CPU inference on N samples (default 10).
  go-infer [N]      Go ONNX inference on N samples (default 10).

Other:
  export [ckpt]     Export checkpoint to ONNX (default: best.pt).
  tokenizer         Train the tokenizer.
  build             Build the training Docker image.
  all               Full pipeline: tokenizer → train → export.
  help              This message.
USAGE
        ;;
esac
