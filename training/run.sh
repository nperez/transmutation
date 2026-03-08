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

# Usage: ./training/run.sh [tokenizer|train|infer|export|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="transmutation-train"

# Ensure pre-built CUDA wheels are downloaded.
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

# Build the Docker image.
build() {
    fetch_wheels
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
}

# Run a Python script inside the container with GPU access.
run_in_docker() {
    docker run --rm --gpus all \
        -v "$PROJECT_DIR/data:/app/data:ro" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        "$IMAGE_NAME" \
        "$@"
}

# Run a Python script inside the container (CPU only, stdin passthrough).
run_in_docker_cpu() {
    docker run --rm -i \
        -v "$PROJECT_DIR/data:/app/data:ro" \
        -v "$PROJECT_DIR/models:/app/models:ro" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        "$IMAGE_NAME" \
        "$@"
}

case "${1:-all}" in
    build)
        build
        ;;
    tokenizer)
        build
        echo "Training tokenizer..."
        run_in_docker training/tokenizer_train.py \
            --data-dir data \
            --output-dir models \
            --vocab-size 8000
        ;;
    train)
        build
        echo "Training model..."
        shift  # remove "train" from $@

        # Auto-resume: interrupt.pt (mid-epoch) takes priority, then latest epoch checkpoint.
        RESUME_FLAG=""
        if [ -f "$PROJECT_DIR/models/interrupt.pt" ]; then
            RESUME_FLAG="--resume models/interrupt.pt"
            echo "Resuming from mid-epoch checkpoint (interrupt.pt)"
        else
            LATEST=$(ls -v "$PROJECT_DIR/models"/epoch_*.pt 2>/dev/null | tail -1)
            if [ -n "$LATEST" ]; then
                RESUME_FLAG="--resume models/$(basename "$LATEST")"
                echo "Resuming from $(basename "$LATEST")"
            fi
        fi

        run_in_docker training/train.py \
            --data-dir data \
            --tokenizer models/tokenizer.model \
            --output-dir models \
            --batch-size 2 \
            --grad-accum 16 \
            --max-src-len 1024 \
            --max-tgt-len 2048 \
            --epochs 5 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16 \
            $RESUME_FLAG \
            "$@"
        ;;
    infer)
        build
        shift  # remove "infer" from $@

        # Default to best.pt, then latest epoch checkpoint.
        CHECKPOINT="${1:-}"
        if [ -n "$CHECKPOINT" ]; then
            shift
        elif [ -f "$PROJECT_DIR/models/best.pt" ]; then
            CHECKPOINT="models/best.pt"
        else
            LATEST=$(ls -v "$PROJECT_DIR/models"/epoch_*.pt 2>/dev/null | tail -1)
            if [ -n "$LATEST" ]; then
                CHECKPOINT="models/$(basename "$LATEST")"
            else
                echo "Error: no checkpoint found. Train a model first."
                exit 1
            fi
        fi

        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        # Generate more than needed (some get filtered by token length).
        GEN_COUNT=$(( N_SAMPLES * 3 ))

        echo "Running CPU inference with $CHECKPOINT ($N_SAMPLES fresh samples)..."
        cd "$PROJECT_DIR" && go run ./cmd/generate -stdout -train "$GEN_COUNT" -val 0 -seed "$RANDOM" \
            | run_in_docker_cpu training/infer_cpu.py "$CHECKPOINT" "$N_SAMPLES" "$@"
        ;;
    go-infer)
        shift  # remove "go-infer" from $@
        GO_INFER_IMAGE="transmutation-infer"

        echo "Building Go inference image..."
        docker build -t "$GO_INFER_IMAGE" -f "$PROJECT_DIR/cmd/infer/Dockerfile" "$PROJECT_DIR"

        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        GEN_COUNT=$(( N_SAMPLES * 3 ))

        echo "Running Go ONNX inference ($N_SAMPLES fresh samples)..."
        cd "$PROJECT_DIR" && go run ./cmd/generate -stdout -train "$GEN_COUNT" -val 0 -seed "$RANDOM" \
            | docker run --rm -i \
                -v "$PROJECT_DIR/models:/app/models:ro" \
                "$GO_INFER_IMAGE" \
                -encoder models/onnx/encoder.onnx \
                -decoder models/onnx/decoder.onnx \
                -tokenizer models/tokenizer.model \
                -ort-lib /usr/local/lib/libonnxruntime.so \
                -n "$N_SAMPLES" \
                "$@"
        ;;
    export)
        build
        shift  # remove "export" from $@
        echo "Exporting to ONNX..."
        run_in_docker training/export.py \
            --checkpoint "${1:-models/best.pt}" \
            --tokenizer models/tokenizer.model \
            --output-dir models/onnx
        ;;
    all)
        build
        echo "=== Step 1: Train tokenizer ==="
        run_in_docker training/tokenizer_train.py \
            --data-dir data \
            --output-dir models \
            --vocab-size 8000

        echo "=== Step 2: Train model ==="
        run_in_docker training/train.py \
            --data-dir data \
            --tokenizer models/tokenizer.model \
            --output-dir models \
            --batch-size 2 \
            --grad-accum 16 \
            --max-src-len 1024 \
            --max-tgt-len 2048 \
            --epochs 5 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16

        echo "=== Step 3: Export to ONNX ==="
        run_in_docker training/export.py \
            --checkpoint models/best.pt \
            --tokenizer models/tokenizer.model \
            --output-dir models/onnx

        echo "=== Done ==="
        ;;
    *)
        echo "Usage: $0 [build|tokenizer|train|infer|go-infer|export|all]"
        exit 1
        ;;
esac
