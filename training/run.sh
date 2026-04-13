#!/bin/bash
# ── Destructive operation authorization ─────────────────────────────────────
# Destructive commands (clean-run, clean-generated, clean-all, kill) require
# an authorization token created from an interactive terminal.
#
# Workflow:
#   1. Claude asks user to run: ! ./training/run.sh authorize <action>
#   2. User runs it (TTY check passes), token created (valid 5 minutes)
#   3. Claude runs the destructive command, token consumed
#
# Claude cannot bypass this: its bash tool is never a TTY.
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
AUTH_TOKEN_FILE="$PROJECT_DIR/.destructive-auth"
AUTH_MAX_AGE=300

require_auth() {
    local action="$1"
    if [ ! -f "$AUTH_TOKEN_FILE" ]; then
        echo "ERROR: No authorization token."
        echo "  Run:  ! ./training/run.sh authorize $action"
        exit 1
    fi
    local token_time token_action now age
    token_time=$(awk '{print $1}' "$AUTH_TOKEN_FILE")
    token_action=$(cut -d' ' -f2- "$AUTH_TOKEN_FILE")
    now=$(date +%s)
    age=$(( now - token_time ))
    if [ "$age" -gt "$AUTH_MAX_AGE" ]; then
        rm -f "$AUTH_TOKEN_FILE"
        echo "ERROR: Authorization expired (${age}s old, max ${AUTH_MAX_AGE}s)."
        echo "  Run:  ! ./training/run.sh authorize $action"
        exit 1
    fi
    if [ "$token_action" != "$action" ]; then
        echo "ERROR: Authorization is for '$token_action', not '$action'."
        echo "  Run:  ! ./training/run.sh authorize $action"
        exit 1
    fi
    rm -f "$AUTH_TOKEN_FILE"
}

# ── Current run ──────────────────────────────────────────────────────────────
# Each run gets its own directory under models/ (checkpoints, tokenizer, ONNX,
# training log, AR inferences). Override with TRANSMUTATION_RUN=runN.
if [ -n "${TRANSMUTATION_RUN:-}" ]; then
    RUN="$TRANSMUTATION_RUN"
else
    # Auto-detect: highest numbered models/runN directory.
    RUN=$(for d in "$PROJECT_DIR/models"/run*/; do basename "$d"; done 2>/dev/null | sort -V | tail -1)
    if [ -z "$RUN" ]; then
        echo "Error: no run directories found in models/. Set TRANSMUTATION_RUN=runN" >&2
        exit 1
    fi
fi
RUN_DIR="models/$RUN"

# ── Build ────────────────────────────────────────────────────────────────────

GENERATE_BIN="$PROJECT_DIR/tmp/generate"
AUGMENT_BIN="$PROJECT_DIR/tmp/augment"

build_generator() {
    local bin="$GENERATE_BIN"
    if [ ! -f "$bin" ] || [ "$(find "$PROJECT_DIR/cmd/generate" "$PROJECT_DIR/pkg" -newer "$bin" 2>/dev/null)" ]; then
        echo "Building generator binary..."
        (cd "$PROJECT_DIR" && CGO_ENABLED=0 go build -o "$bin" ./cmd/generate/)
    fi
}

build_augment() {
    local bin="$AUGMENT_BIN"
    if [ ! -f "$bin" ] || [ "$(find "$PROJECT_DIR/cmd/augment" "$PROJECT_DIR/pkg" -newer "$bin" 2>/dev/null)" ]; then
        echo "Building augment binary..."
        (cd "$PROJECT_DIR" && CGO_ENABLED=0 go build -o "$bin" ./cmd/augment/)
    fi
}

build_train() {
    build_generator
    build_augment
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
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    docker run --rm --gpus all \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        -v "$PROJECT_DIR/tmp/generate:/app/generate:ro" \
        -v "$PROJECT_DIR/tmp/augment:/app/augment:ro" \
        -v "$PROJECT_DIR/tmp/triton_cache:/home/trainer/.triton" \
        "$TRAIN_IMAGE" \
        "$@"
}

# Run a GPU container detached. Returns container ID.
run_gpu_detached() {
    docker run -d --gpus all \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$SCRIPT_DIR:/app/training:ro" \
        -v "$PROJECT_DIR/tmp/generate:/app/generate:ro" \
        -v "$PROJECT_DIR/tmp/augment:/app/augment:ro" \
        -v "$PROJECT_DIR/tmp/triton_cache:/home/trainer/.triton" \
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
    latest=$(ls -t "$PROJECT_DIR/$RUN_DIR"/interrupt_*.pt \
                    "$PROJECT_DIR/$RUN_DIR"/interrupt.pt \
                    "$PROJECT_DIR/$RUN_DIR"/best.pt \
                    "$PROJECT_DIR/$RUN_DIR"/epoch_*.pt \
                    2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then
        echo "--resume $RUN_DIR/$(basename "$latest")"
    fi
}

# ── Commands ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    build)
        build_train
        ;;

    prepare-data)
        build_train
        build_generator
        build_augment

        mkdir -p "$PROJECT_DIR/data/run7/train" "$PROJECT_DIR/data/run7/val"

        # Step 1: Short synthetic samples (idempotent — skip if shards exist).
        if ls "$PROJECT_DIR/data/run7/train"/shard_*.jsonl 1>/dev/null 2>&1; then
            echo "Short synthetic samples already exist, skipping."
        else
            echo "Generating short synthetic samples..."
            "$GENERATE_BIN" -stage 1 -short -train 50000 -val 5000 \
                -seed 42 -out "$PROJECT_DIR/data/run7"
        fi

        # Step 2: Haiku augmentation (idempotent — skip if haiku_all.jsonl exists and is non-empty).
        if [ -s "$PROJECT_DIR/data/run7/train/haiku_all.jsonl" ]; then
            echo "Haiku train augmentation already exists, skipping."
        else
            echo "Augmenting haiku corpus (train)..."
            "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
                -sample-pct 100 -aug-ratio 5 \
                -special-prob 0.25 -corrupt-pct 15 \
                -compact-pct 50 -truncate-pct 10 \
                -shorten-pct 30 -drop-memory-pct 30 \
                -tokenizer "$PROJECT_DIR/$RUN_DIR/tokenizer.model" \
                -seed 42 \
                > "$PROJECT_DIR/data/run7/train/haiku_all.jsonl"
        fi

        if [ -s "$PROJECT_DIR/data/run7/val/haiku_all.jsonl" ]; then
            echo "Haiku val augmentation already exists, skipping."
        else
            echo "Augmenting haiku corpus (val)..."
            "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
                -sample-pct 100 -aug-ratio 5 \
                -special-prob 0.25 -corrupt-pct 15 \
                -compact-pct 50 -truncate-pct 10 \
                -shorten-pct 30 -drop-memory-pct 30 \
                -tokenizer "$PROJECT_DIR/$RUN_DIR/tokenizer.model" \
                -seed 42 -val \
                > "$PROJECT_DIR/data/run7/val/haiku_all.jsonl"
        fi

        # Step 3: Tokenize (idempotent — skip if dataset.pt exists and is newer than all JSONL).
        if [ -f "$PROJECT_DIR/data/run7/train/dataset.pt" ] && \
           [ -f "$PROJECT_DIR/data/run7/val/dataset.pt" ] && \
           ! find "$PROJECT_DIR/data/run7" -name "*.jsonl" -newer "$PROJECT_DIR/data/run7/train/dataset.pt" 2>/dev/null | grep -q .; then
            echo "dataset.pt already up to date, skipping tokenization."
        else
            echo "Tokenizing and preparing dataset..."
            run_gpu training/prepare_data.py \
                --tokenizer "$RUN_DIR/tokenizer.model" \
                --data-dir data/run7 \
                --max-src-len 1152 --max-tgt-len 1536
        fi

        echo "Data preparation complete."
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

        # Skip auto-resume if user passed --resume explicitly.
        if echo "$@" | grep -q -- '--resume'; then
            RESUME_FLAG=""
            echo "Using explicit --resume from args"
        else
            RESUME_FLAG=$(find_resume_flag)
            if [ -n "$RESUME_FLAG" ]; then
                echo "Resuming: $RESUME_FLAG"
            else
                echo "Starting fresh training"
            fi
        fi

        mkdir -p "$PROJECT_DIR/$RUN_DIR"

        # Train tokenizer if missing.
        if [ ! -f "$PROJECT_DIR/$RUN_DIR/tokenizer.model" ]; then
            echo "No tokenizer found — training one..."
            build_generator
            # Generate augmented corpus for tokenizer training.
            mkdir -p "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val"
            "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
                -sample-pct 30 -aug-ratio 5 -special-prob 0.30 -compact-pct 50 -seed 42 \
                > "$PROJECT_DIR/data/train/haiku_augmented.jsonl"
            "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
                -sample-pct 10 -aug-ratio 5 -special-prob 0.30 -compact-pct 50 -seed 42 -val \
                > "$PROJECT_DIR/data/val/haiku_augmented.jsonl"
            run_gpu training/tokenizer_train.py \
                --data-dir data \
                --output-dir "$RUN_DIR" \
                --vocab-size 16000
            echo "Tokenizer trained."
        fi

        echo "Run: $RUN ($RUN_DIR)"
        CID=$(run_gpu_detached training/train.py \
            --data-dir data/run7 \
            --tokenizer "$RUN_DIR/tokenizer.model" \
            --output-dir "$RUN_DIR" \
            --batch-size 4 \
            --grad-accum 8 \
            --d-model 512 \
            --n-layers 10 \
            --n-heads 8 \
            --d-ff 1536 \
            --emb-rank 128 \
            --max-src-len 1152 \
            --max-tgt-len 1536 \
            --epochs 100 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16 \
            --eval-denoise-steps 4 \
            --override-lr 3e-4 \
            --stage 1 \
            --max-stage 2 \
            --stage-patience 2 \
            --max-epoch-samples 33000 \
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
        if [ -n "${2:-}" ]; then
            docker logs --tail "$2" "$CID" 2>&1 | tr '\r' '\n'
        else
            docker logs -f "$CID" 2>&1 | tr '\r' '\n'
        fi
        ;;

    status)
        echo "=== Run: $RUN ($RUN_DIR) ==="
        echo
        echo "=== Checkpoints ==="
        if ls "$PROJECT_DIR/$RUN_DIR"/*.pt 1>/dev/null 2>&1; then
            ls -1 "$PROJECT_DIR/$RUN_DIR"/*.pt | while read f; do
                name=$(basename "$f")
                sz=$(du -h "$f" | cut -f1)
                echo "  $name  $sz"
            done
        else
            echo "  (none)"
        fi
        echo
        echo "=== ONNX Models ==="
        ls -lh "$PROJECT_DIR/$RUN_DIR/onnx"/*.onnx 2>/dev/null || echo "  (none)"
        echo
        echo "=== Training Log ==="
        if [ -f "$PROJECT_DIR/$RUN_DIR/training_log.json" ]; then
            python3 -c "
import json, sys
entries = json.load(open('$PROJECT_DIR/$RUN_DIR/training_log.json'))
for e in entries[-10:]:
    ev = f\"eval={e.get('eval_exact','?')}/{e.get('eval_total','?')}exact {e.get('eval_xml_ok','?')}/{e.get('eval_total','?')}xml\" if 'eval_total' in e else (f\"ar={e.get('ar_exact','?')}/{e.get('ar_total','?')}exact {e.get('ar_xml_ok','?')}/{e.get('ar_total','?')}xml\" if 'ar_total' in e else '')
    cer_key = 'eval_cer' if 'eval_cer' in e else 'ar_cer'
    wer_key = 'eval_wer' if 'eval_wer' in e else 'ar_wer'
    er = f\" CER={e[cer_key]:.2%} WER={e[wer_key]:.2%}\" if cer_key in e else ''
    stg = f\"s{e['stage']}\" if 'stage' in e else ''
    wc = f\" {e['wallclock']/3600:.1f}h\" if 'wallclock' in e else ''
    ds = f\" {e['denoise_steps']}step\" if 'denoise_steps' in e else ''
    print(f\"  epoch={e['epoch']} {stg} train={e['train_loss']:.4f} {ev}{er} lr={e['lr']:.2e}{wc}{ds}\")
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
        echo "Generating haiku corpus for tokenizer training..."
        # Generate a large augmented sample at max difficulty for tokenizer training.
        mkdir -p "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val"
        "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
            -sample-pct 30 -aug-ratio 5 -special-prob 0.30 -seed 42 \
            > "$PROJECT_DIR/data/train/haiku_augmented.jsonl"
        "$AUGMENT_BIN" -dir "$PROJECT_DIR/data/haiku" \
            -sample-pct 10 -aug-ratio 5 -special-prob 0.30 -seed 42 -val \
            > "$PROJECT_DIR/data/val/haiku_augmented.jsonl"
        mkdir -p "$PROJECT_DIR/$RUN_DIR"
        echo "Training tokenizer on haiku corpus (-> $RUN_DIR)..."
        run_gpu training/tokenizer_train.py \
            --data-dir data \
            --output-dir "$RUN_DIR" \
            --vocab-size 16000
        ;;

    export)
        build_train
        shift
        if [ -n "${1:-}" ]; then
            CKPT="$1"
        else
            CKPT=$(find_resume_flag | sed 's/--resume //')
            CKPT="${CKPT:-$RUN_DIR/best.pt}"
        fi
        echo "Exporting $CKPT to ONNX (CPU)..."
        # Run on CPU so export works while training holds the GPU.
        docker run --rm \
            -v "$PROJECT_DIR/data:/app/data:ro" \
            -v "$PROJECT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR:/app/training:ro" \
            -v "$PROJECT_DIR/tmp/generate:/app/generate:ro" \
            "$TRAIN_IMAGE" \
            training/export.py \
            --checkpoint "$CKPT" \
            --tokenizer "$RUN_DIR/tokenizer.model" \
            --output-dir "$RUN_DIR/onnx"
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
        CHECKPOINT="${RESUME_CKPT:-$RUN_DIR/best.pt}"

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        TMPFILE="$PROJECT_DIR/tmp/infer_input_$$.jsonl"
        trap "rm -f '$TMPFILE'" EXIT

        echo "Generating $GEN_COUNT candidates (stage $STAGE)..."
        "$GENERATE_BIN" -stage "$STAGE" -stdout -train "$GEN_COUNT" -val 0 -seed "$$" > "$TMPFILE"

        echo "CPU inference: $CHECKPOINT ($N_SAMPLES samples, stage $STAGE)..."
        shuf -n "$N_SAMPLES" "$TMPFILE" | run_cpu_stdin training/infer.py "$CHECKPOINT" \
            -n "$N_SAMPLES" "$@"
        ;;

    gpu-infer)
        build_train
        build_generator
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        STAGE="${1:-3}"
        if [ -n "${1:-}" ]; then shift; fi

        # Accept --ckpt override.
        RESUME_CKPT=$(find_resume_flag | sed 's/--resume //')
        CHECKPOINT="${RESUME_CKPT:-$RUN_DIR/best.pt}"
        _prev=""
        _remaining=()
        for _arg in "$@"; do
            if [ "$_prev" = "--ckpt" ]; then CHECKPOINT="$_arg"; _prev=""; continue; fi
            if [ "$_arg" = "--ckpt" ]; then _prev="$_arg"; continue; fi
            _remaining+=("$_arg")
        done
        set -- "${_remaining[@]+"${_remaining[@]}"}"

        GEN_COUNT=$(( N_SAMPLES * 3 ))
        TMPFILE="$PROJECT_DIR/tmp/infer_input_$$.jsonl"
        trap "rm -f '$TMPFILE'" EXIT

        echo "Generating $GEN_COUNT candidates (stage $STAGE)..."
        "$GENERATE_BIN" -stage "$STAGE" -stdout -train "$GEN_COUNT" -val 0 -seed "$$" > "$TMPFILE"
        shuf -n "$N_SAMPLES" "$TMPFILE" > "${TMPFILE}.shuf"
        mv "${TMPFILE}.shuf" "$TMPFILE"

        echo "GPU inference: $CHECKPOINT ($N_SAMPLES samples, stage $STAGE)..."
        docker run --rm --gpus all \
            -v "$PROJECT_DIR/data:/app/data" \
            -v "$PROJECT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR:/app/training:ro" \
            -v "$PROJECT_DIR/tmp:/app/tmp" \
            "$TRAIN_IMAGE" \
            training/infer.py "$CHECKPOINT" \
            -n "$N_SAMPLES" --gpu --input "tmp/infer_input_$$.jsonl" "$@"
        ;;

    infer-rejects)
        build_train
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        PAIRS_FILE="$PROJECT_DIR/data/rejects/repaired_pairs.jsonl"
        if [ ! -f "$PAIRS_FILE" ]; then
            echo "No repaired pairs file at $PAIRS_FILE"
            exit 1
        fi

        RESUME_CKPT=$(find_resume_flag | sed 's/--resume //')
        CHECKPOINT="${RESUME_CKPT:-$RUN_DIR/best.pt}"

        TOTAL=$(wc -l < "$PAIRS_FILE")
        echo "CPU inference on repaired rejects: $CHECKPOINT ($N_SAMPLES of $TOTAL pairs)..."
        shuf -n "$N_SAMPLES" "$PAIRS_FILE" | run_cpu_stdin training/infer.py "$CHECKPOINT" \
            -n "$N_SAMPLES" "$@"
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
                -model $RUN_DIR/onnx/diffusion_int8.onnx \
                -length-model $RUN_DIR/onnx/length_predictor_int8.onnx \
                -emb-down $RUN_DIR/onnx/emb_down.npy \
                -emb-up $RUN_DIR/onnx/emb_up.npy \
                -tokenizer $RUN_DIR/tokenizer.model \
                -ort-lib /usr/local/lib/libonnxruntime.so \
                -n $N_SAMPLES \
                $*"
        ;;

    go-infer-rejects)
        build_infer
        shift
        N_SAMPLES="${1:-10}"
        if [ -n "${1:-}" ]; then shift; fi

        PAIRS_FILE="$PROJECT_DIR/data/rejects/repaired_pairs.jsonl"
        if [ ! -f "$PAIRS_FILE" ]; then
            echo "No repaired pairs file at $PAIRS_FILE"
            exit 1
        fi

        TOTAL=$(wc -l < "$PAIRS_FILE")
        echo "Go ONNX inference on repaired rejects ($N_SAMPLES of $TOTAL pairs)..."
        docker run --rm \
            -v "$PROJECT_DIR/models:/app/models:ro" \
            -v "$PAIRS_FILE:/app/input.jsonl:ro" \
            --entrypoint sh \
            "$INFER_IMAGE" \
            -c "cat /app/input.jsonl | infer \
                -model $RUN_DIR/onnx/diffusion_int8.onnx \
                -length-model $RUN_DIR/onnx/length_predictor_int8.onnx \
                -emb-down $RUN_DIR/onnx/emb_down.npy \
                -emb-up $RUN_DIR/onnx/emb_up.npy \
                -tokenizer $RUN_DIR/tokenizer.model \
                -ort-lib /usr/local/lib/libonnxruntime.so \
                -n $N_SAMPLES \
                $*"
        ;;

    all)
        build_train
        mkdir -p "$PROJECT_DIR/$RUN_DIR"
        echo "=== Step 1: Tokenizer ==="
        run_gpu training/tokenizer_train.py \
            --data-dir data \
            --output-dir "$RUN_DIR" \
            --vocab-size 16000

        echo "=== Step 2: Train ==="
        run_gpu training/train.py \
            --data-dir data/run7 \
            --tokenizer "$RUN_DIR/tokenizer.model" \
            --output-dir "$RUN_DIR" \
            --batch-size 4 \
            --grad-accum 8 \
            --d-model 512 \
            --n-layers 10 \
            --n-heads 8 \
            --d-ff 1536 \
            --emb-rank 128 \
            --max-src-len 1152 \
            --max-tgt-len 1536 \
            --epochs 100 \
            --lr 3e-4 \
            --warmup-steps 2000 \
            --save-every 1 \
            --fp16 \
            --eval-denoise-steps 4 \
            --override-lr 3e-4 \
            --stage 1 \
            --max-stage 2

        echo "=== Step 3: Export ==="
        run_gpu training/export.py \
            --checkpoint "$RUN_DIR/best.pt" \
            --tokenizer "$RUN_DIR/tokenizer.model" \
            --output-dir "$RUN_DIR/onnx"

        echo "=== Done ==="
        ;;

    haiku-gen)
        build_generator
        shift
        N="${1:-100}"
        echo "Generating $N samples via Haiku..."
        "$PROJECT_DIR/scripts/gen_haiku.sh" "$N" "$PROJECT_DIR/data/haiku"
        ;;

    haiku-split)
        HAIKU_DIR="$PROJECT_DIR/data/haiku"
        TRAIN_DIR="$PROJECT_DIR/data/haiku_train"
        VAL_DIR="$PROJECT_DIR/data/haiku_val"

        if [ ! -d "$HAIKU_DIR" ]; then
            echo "Error: $HAIKU_DIR does not exist."
            exit 1
        fi

        mkdir -p "$TRAIN_DIR" "$VAL_DIR"

        # Concatenate all JSONL, sort for determinism, split 90/10 by line number.
        TOTAL=$(cat "$HAIKU_DIR"/*.jsonl | wc -l)
        VAL_COUNT=$(( TOTAL / 10 ))
        TRAIN_COUNT=$(( TOTAL - VAL_COUNT ))

        cat "$HAIKU_DIR"/*.jsonl | sort > "$PROJECT_DIR/tmp/haiku_sorted.jsonl"
        head -n "$TRAIN_COUNT" "$PROJECT_DIR/tmp/haiku_sorted.jsonl" > "$TRAIN_DIR/haiku.jsonl"
        tail -n "$VAL_COUNT" "$PROJECT_DIR/tmp/haiku_sorted.jsonl" > "$VAL_DIR/haiku.jsonl"
        rm -f "$PROJECT_DIR/tmp/haiku_sorted.jsonl"

        echo "Split $TOTAL haiku samples: $TRAIN_COUNT train, $VAL_COUNT val"
        echo "  $TRAIN_DIR/haiku.jsonl"
        echo "  $VAL_DIR/haiku.jsonl"
        ;;

    haiku-clean)
        echo "Cleaning haiku corpus..."
        rm -rf "$PROJECT_DIR/data/haiku" "$PROJECT_DIR/data/haiku_train" "$PROJECT_DIR/data/haiku_val"
        echo "Done."
        ;;

    clean-generated)
        require_auth "clean-generated"
        echo "Cleaning generated train/val data (preserving haiku)..."
        rm -rf "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val"
        echo "Done."
        ;;

    clean-all)
        require_auth "clean-all"
        echo "Cleaning all data (train, val, haiku)..."
        rm -rf "$PROJECT_DIR/data/train" "$PROJECT_DIR/data/val" "$PROJECT_DIR/data/haiku" "$PROJECT_DIR/data/haiku_train" "$PROJECT_DIR/data/haiku_val"
        echo "Done."
        ;;

    # Clean checkpoints, logs, ONNX, and AR inferences for a run.
    # Does NOT remove the tokenizer — that requires retraining.
    clean-run)
        require_auth "clean-run"
        shift
        TARGET="${1:-$RUN}"
        TARGET_DIR="$PROJECT_DIR/models/$TARGET"
        if [ ! -d "$TARGET_DIR" ]; then
            echo "Error: $TARGET_DIR does not exist"
            exit 1
        fi
        echo "Cleaning $TARGET ($TARGET_DIR)..."
        rm -f "$TARGET_DIR"/*.pt "$TARGET_DIR"/training_log.json
        rm -rf "$TARGET_DIR"/eval_inferences "$TARGET_DIR"/onnx
        echo "Done. Tokenizer preserved."
        ;;

    new-run)
        # Create the next run directory.
        LAST=$(for d in "$PROJECT_DIR/models"/run*/; do basename "$d"; done 2>/dev/null | sort -V | tail -1)
        LAST_NUM=${LAST#run}
        NEXT_NUM=$(( LAST_NUM + 1 ))
        NEXT="run$NEXT_NUM"
        mkdir -p "$PROJECT_DIR/models/$NEXT"
        echo "Created models/$NEXT"
        echo "Set TRANSMUTATION_RUN=$NEXT or it will auto-detect as current run."
        ;;

    haiku-collapse)
        HAIKU_DIR="$PROJECT_DIR/data/haiku"
        CORPUS="$HAIKU_DIR/corpus.jsonl"
        echo "Collapsing shards into corpus.jsonl..."
        cat "$HAIKU_DIR"/haiku_shard_*.jsonl > "$CORPUS"
        LINES=$(wc -l < "$CORPUS")
        echo "$LINES samples -> $CORPUS"
        echo "Removing shards..."
        find "$HAIKU_DIR" -name 'haiku_shard_*.jsonl' -delete
        echo "Done."
        ;;

    enrich-tools)
        go build -o "$PROJECT_DIR/tmp/enrich" "$PROJECT_DIR/cmd/enrich/"
        shift
        PCT="${1:-25}"
        HAIKU_DIR="$PROJECT_DIR/data/haiku"
        CORPUS="$HAIKU_DIR/corpus.jsonl"
        if [ ! -f "$CORPUS" ]; then
            echo "No corpus.jsonl found. Run './training/run.sh haiku-collapse' first."
            exit 1
        fi
        echo "Enriching tool-call samples (${PCT}% of tools)..."
        "$PROJECT_DIR/tmp/enrich" -pct "$PCT" -seed 42 < "$CORPUS" > "${CORPUS}.enriched" 2>&1
        mv "${CORPUS}.enriched" "$CORPUS"
        echo "Done."
        ;;

    repair-rejects)
        go build -o "$PROJECT_DIR/tmp/repair" "$PROJECT_DIR/cmd/repair/"
        shift
        "$PROJECT_DIR/scripts/repair_rejects.sh" "$@"
        ;;

    eval)
        build_train
        build_generator
        shift
        N_SAMPLES="${1:-500}"
        if [ -n "${1:-}" ]; then shift; fi

        HOLDOUT_DIR="$PROJECT_DIR/data/holdout"
        EVAL_LOG="$PROJECT_DIR/tmp/eval.log"
        EVAL_PID="$PROJECT_DIR/tmp/eval.pid"
        mkdir -p "$PROJECT_DIR/tmp"

        # Reject if already running.
        if [ -f "$EVAL_PID" ] && kill -0 "$(cat "$EVAL_PID")" 2>/dev/null; then
            echo "Eval already running (PID $(cat "$EVAL_PID")). Use 'eval-logs' to monitor."
            exit 1
        fi

        # Accept explicit checkpoint: ./training/run.sh eval 250 --ckpt models/run5/epoch_74.pt
        RESUME_CKPT=$(find_resume_flag | sed 's/--resume //')
        CHECKPOINT="${RESUME_CKPT:-$RUN_DIR/best.pt}"
        _prev=""
        _remaining=()
        for _arg in "$@"; do
            if [ "$_prev" = "--ckpt" ]; then CHECKPOINT="$_arg"; _prev=""; continue; fi
            if [ "$_arg" = "--ckpt" ]; then _prev="$_arg"; continue; fi
            _remaining+=("$_arg")
        done
        set -- "${_remaining[@]+"${_remaining[@]}"}"
        EVAL_OUT="$PROJECT_DIR/tmp/eval_results.jsonl"

        # Run the full pipeline in a background subshell, logging to file.
        (
            exec > "$EVAL_LOG" 2>&1
            set -e
            trap 'echo "EVAL FAILED at line $LINENO (exit $?)"' ERR

            # Step 1: Generate fresh holdout data if not present.
            HOLDOUT_HAS=$(find "$HOLDOUT_DIR" -name '*.jsonl' 2>/dev/null | head -1 || true)
            if [ -z "$HOLDOUT_HAS" ]; then
                echo "No holdout data found. Generating $N_SAMPLES fresh samples via Haiku..."
                mkdir -p "$HOLDOUT_DIR"
                "$PROJECT_DIR/scripts/gen_haiku.sh" "$N_SAMPLES" "$HOLDOUT_DIR"
                echo "Holdout generation complete."
            else
                EXISTING=$(cat "$HOLDOUT_DIR"/*.jsonl 2>/dev/null | wc -l)
                echo "Using existing holdout set: $EXISTING samples in $HOLDOUT_DIR"
            fi

            # Step 2: Augment holdout into clean input/target pairs (no corruption).
            echo "Augmenting holdout to input/target pairs..."
            TMPFILE=$(mktemp)
            trap 'rm -f "$TMPFILE"' EXIT
            "$PROJECT_DIR/tmp/augment" \
                -dir "$HOLDOUT_DIR" \
                -sample-pct 100 \
                -aug-ratio 0 \
                -special-prob 0 \
                -corrupt-pct 0 \
                -compact-pct 50 \
                -type all \
                -seed 77777 > "$TMPFILE" 2>/dev/null

            TOTAL=$(wc -l < "$TMPFILE")
            echo "Augmented $TOTAL holdout pairs."

            # Step 3: Run inference with --json output (GPU if available).
            echo "Running inference: $CHECKPOINT ($N_SAMPLES of $TOTAL samples, --json)..."
            EVAL_INPUT="$PROJECT_DIR/data/eval_input.jsonl"
            EVAL_RAW="$PROJECT_DIR/data/eval_raw.jsonl"
            shuf -n "$N_SAMPLES" "$TMPFILE" > "$EVAL_INPUT"
            docker run --rm --gpus all \
                --name "${CONTAINER_NAME}-eval" \
                -v "$PROJECT_DIR/data:/app/data" \
                -v "$PROJECT_DIR/models:/app/models" \
                -v "$SCRIPT_DIR:/app/training:ro" \
                -v "$PROJECT_DIR/tmp/triton_cache:/home/trainer/.triton" \
                "$TRAIN_IMAGE" \
                training/infer.py "$CHECKPOINT" \
                -n "$N_SAMPLES" --json --gpu --input data/eval_input.jsonl "$@" > "$EVAL_RAW" 2>&1
            grep '^{' "$EVAL_RAW" > "$EVAL_OUT" || true
            rm -f "$EVAL_INPUT" "$EVAL_RAW"

            # Step 4: Display results bucketed by source token length.
            echo ""
            echo "=== Eval Summary ==="
            TOTAL_EVAL=$(wc -l < "$EVAL_OUT")
            EXACT=$(grep -c '"exact": true' "$EVAL_OUT" || true)
            SEMANTIC=$(grep -c '"semantic": true' "$EVAL_OUT" || true)
            XML_OK=$(grep -c '"xml_ok": true' "$EVAL_OUT" || true)
            FAIL=$((TOTAL_EVAL - XML_OK))
            echo "Total: $TOTAL_EVAL | Exact: $EXACT | Semantic: $SEMANTIC | XML_OK: $((XML_OK - EXACT - SEMANTIC)) | Fail: $FAIL"
            echo ""

            echo "=== By Source Token Length ==="
            printf "%-16s %6s %6s %6s %6s %6s\n" "Bucket" "Total" "Exact" "Sem" "XmlOk" "Fail"
            for BUCKET in "0-250" "251-500" "501-750" "751-1000" "1001+"; do
                case "$BUCKET" in
                    0-250)    FILTER='select(.src_tokens <= 250)' ;;
                    251-500)  FILTER='select(.src_tokens > 250 and .src_tokens <= 500)' ;;
                    501-750)  FILTER='select(.src_tokens > 500 and .src_tokens <= 750)' ;;
                    751-1000) FILTER='select(.src_tokens > 750 and .src_tokens <= 1000)' ;;
                    1001+)    FILTER='select(.src_tokens > 1000)' ;;
                esac
                STATS=$(jq -s -r "[.[] | $FILTER] | {t: length, e: [.[] | select(.exact)] | length, s: [.[] | select(.semantic)] | length, x: [.[] | select(.xml_ok)] | length} | \"\(.t) \(.e) \(.s) \(.x)\"" "$EVAL_OUT")
                BT=$(echo "$STATS" | awk '{print $1}')
                BE=$(echo "$STATS" | awk '{print $2}')
                BS=$(echo "$STATS" | awk '{print $3}')
                BX=$(echo "$STATS" | awk '{print $4}')
                BF=$((BT - BX))
                BXO=$((BX - BE - BS))
                if [ "$BT" -gt 0 ]; then
                    PCT=$(awk "BEGIN {printf \"%.1f\", ($BE/$BT)*100}")
                    printf "%-16s %6d %6d %6d %6d %6d  (%s%% exact)\n" "$BUCKET" "$BT" "$BE" "$BS" "$BXO" "$BF" "$PCT"
                fi
            done

            echo ""
            echo "=== Eval complete ==="
            rm -f "$EVAL_PID"
        ) &
        BGPID=$!
        echo "$BGPID" > "$EVAL_PID"
        echo "Eval started (PID $BGPID). Monitor with: ./training/run.sh eval-logs"
        ;;

    eval-logs)
        EVAL_LOG="$PROJECT_DIR/tmp/eval.log"
        EVAL_PID="$PROJECT_DIR/tmp/eval.pid"
        EVAL_CONTAINER="${CONTAINER_NAME}-eval"
        if [ ! -f "$EVAL_LOG" ]; then
            echo "No eval log found. Run 'eval' first."
            exit 1
        fi
        if [ -n "${2:-}" ]; then
            # Show eval log, plus live container logs if inference is running.
            tail -n "$2" "$EVAL_LOG"
            if docker ps -q -f "name=$EVAL_CONTAINER" 2>/dev/null | grep -q .; then
                echo "--- inference (live) ---"
                docker logs --tail "$2" "$EVAL_CONTAINER" 2>&1
            fi
        else
            if [ -f "$EVAL_PID" ] && kill -0 "$(cat "$EVAL_PID")" 2>/dev/null; then
                tail -f "$EVAL_LOG"
            else
                cat "$EVAL_LOG"
            fi
        fi
        ;;

    eval-kill)
        EVAL_PID="$PROJECT_DIR/tmp/eval.pid"
        EVAL_CONTAINER="${CONTAINER_NAME}-eval"
        # Stop the docker container first (the actual work).
        if docker ps -q -f "name=$EVAL_CONTAINER" 2>/dev/null | grep -q .; then
            echo "Stopping eval container..."
            docker stop -t 5 "$EVAL_CONTAINER" >/dev/null 2>&1 || true
            docker rm -f "$EVAL_CONTAINER" >/dev/null 2>&1 || true
        fi
        # Then kill the background subshell.
        if [ -f "$EVAL_PID" ] && kill -0 "$(cat "$EVAL_PID")" 2>/dev/null; then
            kill -- -"$(cat "$EVAL_PID")" 2>/dev/null || kill "$(cat "$EVAL_PID")" 2>/dev/null || true
        fi
        rm -f "$EVAL_PID"
        echo "Eval killed."
        ;;

    eval-regen)
        build_generator
        shift
        N_SAMPLES="${1:-500}"
        HOLDOUT_DIR="$PROJECT_DIR/data/holdout"
        echo "Regenerating holdout set: $N_SAMPLES fresh samples..."
        rm -rf "$HOLDOUT_DIR"
        mkdir -p "$HOLDOUT_DIR"
        "$PROJECT_DIR/scripts/gen_haiku.sh" "$N_SAMPLES" "$HOLDOUT_DIR"
        echo "Done. Run './training/run.sh eval' to evaluate."
        ;;

    authorize)
        if [ ! -t 0 ]; then
            echo "ERROR: authorize must be run from an interactive terminal."
            exit 1
        fi
        shift
        ACTION="${*:?Usage: ./training/run.sh authorize <action>}"
        echo "$(date +%s) $ACTION" > "$AUTH_TOKEN_FILE"
        echo "Authorized '$ACTION' (valid ${AUTH_MAX_AGE}s)."
        ;;

    help|*)
        cat <<'USAGE'
Usage: ./training/run.sh <command> [args...]

Training:
  train [flags]     Start training (detached). Auto-resumes from checkpoint.
  stop              Graceful stop (checkpoints, then exits).
  kill              Force kill (no checkpoint).
  checkpoint        Save a checkpoint without stopping.
  logs [N]          Follow training output, or show last N lines.
  status            Show checkpoints, metrics, container state.

Inference:
  infer [N] [stage] [--denoise-steps N]
                        CPU inference on N samples (default 10, stage 3).
  infer-rejects [N]    CPU inference on N rejected haiku samples (default 10).
  eval [N]             Holdout eval (background): generate unseen data, infer, bucket by token length.
  eval-logs [N]        Follow eval log, or show last N lines.
  eval-kill            Kill running eval.
  eval-regen [N]       Regenerate holdout set (N samples, default 500).
  go-infer [N] [stage] [-denoise-steps N]
                        Go ONNX inference on N samples (default 10, stage 3).
  go-infer-rejects [N]  Go ONNX inference on N rejected haiku samples (default 10).

Data:
  haiku-gen [N]     Generate N samples via Claude Haiku (default 100).
  haiku-split       Split haiku corpus into train/val (90/10).
  haiku-clean       Remove haiku corpus (all splits).
  haiku-collapse    Collapse haiku shards into single corpus.jsonl.
  enrich-tools [pct] Enrich tool-call argument structures (default 25%).
  repair-rejects    Repair rejected haiku samples via dual LLM passes.
  clean-generated   Remove train/val data (preserves haiku).
  clean-all         Remove all data (train, val, haiku).

Runs:
  new-run           Create next run directory (models/runN+1).
  clean-run [name]  Clean checkpoints/logs/ONNX for a run (keeps tokenizer).
  status            Show current run's checkpoints, metrics, container state.

  Current run auto-detected as highest models/runN.
  Override with TRANSMUTATION_RUN=runN.

Other:
  prepare-data      Pre-generate and tokenize run7 dataset.
  export [ckpt]     Export checkpoint to ONNX (default: best.pt).
  tokenizer         Train the tokenizer.
  build             Build the training Docker image.
  all               Full pipeline: tokenizer → train → export.
  help              This message.
USAGE
        ;;
esac
