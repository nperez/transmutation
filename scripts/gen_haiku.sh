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

# Generate training samples using Claude Haiku via the claude CLI.
#
# Usage: ./scripts/gen_haiku.sh [count] [output_dir]
#   count      - number of samples to generate (default: 100)
#   output_dir - where to save JSONL (default: data/haiku)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GENERATE_BIN="$PROJECT_DIR/tmp/generate"

COUNT="${1:-100}"
OUT_DIR="${2:-$PROJECT_DIR/data/haiku}"
BATCH_SIZE=25
PARALLEL=8
SAMPLES_PER_ROUND=$((BATCH_SIZE * PARALLEL))

DICT="/usr/share/dict/american-english"

# Tool types and their distribution. The script picks the mode, not the LLM.
TOOL_NAMES=(execute_sql execute_python execute_javascript execute_shell execute_go search read_file write_file http_request)

mkdir -p "$OUT_DIR"

if [ ! -f "$GENERATE_BIN" ]; then
    echo "Generator binary not found. Run: ./training/run.sh build"
    exit 1
fi

claude_cmd() {
    CLAUDECODE= MAX_THINKING_TOKENS=0 claude -p \
        --model=haiku \
        --tools='' \
        --disable-slash-commands \
        --setting-sources='' \
        --system-prompt='' \
        "$@"
}

# --- Two prompt templates: answer mode vs tool mode ---

ANSWER_PROMPT='You are a data generator. Output exactly %d JSON objects, one per line.

Schema: {"thought": "...", "answer": "...", "tool": null, "memory": ["...", ...]}

Every object MUST have "tool": null and "answer" as a non-null string.

Constraints:
- thought: 3-10 sentences of detailed technical reasoning
- answer: substantial response text, 5+ sentences. May include markdown, code blocks, tables, lists.
- tool: MUST be null
- memory: 4-8 short context strings about user preferences, system state, or session history
- Every sample must cover a DIFFERENT topic. Vary widely across: databases, APIs, deployments, monitoring, security, ML, networking, CI/CD, frontend, backend, DevOps, data engineering, testing, performance, architecture, etc.
- Each line must be valid JSON. Output ONLY JSON lines, nothing else.

SEED WORDS: %s'

TOOL_PROMPT='You are a data generator. Output exactly %d JSON objects, one per line.

Every object MUST use the tool "%s". The answer field MUST be null.

Schema: {"thought": "...", "answer": null, "tool": {"tool_name": "%s", "arguments": {%s}}, "memory": ["...", ...]}

Constraints:
- thought: 3-10 sentences of reasoning about why this tool is needed
- answer: MUST be null
- tool: MUST be {"tool_name": "%s", "arguments": {<realistic args>}}
- memory: 4-8 short context strings
- Every sample must cover a DIFFERENT topic
- Tool arguments must contain REALISTIC content, not placeholders
- Each line must be valid JSON. Output ONLY JSON lines, nothing else.

SEED WORDS: %s'

# Argument hints per tool type so Haiku knows the schema.
declare -A TOOL_ARG_HINTS
TOOL_ARG_HINTS[execute_sql]='"query": "<full SQL query>"'
TOOL_ARG_HINTS[execute_python]='"code": "<python code>"'
TOOL_ARG_HINTS[execute_javascript]='"code": "<javascript code>"'
TOOL_ARG_HINTS[execute_shell]='"command": "<shell command>"'
TOOL_ARG_HINTS[execute_go]='"code": "<go code>"'
TOOL_ARG_HINTS[search]='"query": "<search query>"'
TOOL_ARG_HINTS[read_file]='"path": "<file path>"'
TOOL_ARG_HINTS[write_file]='"path": "<file path>", "content": "<file content>"'
TOOL_ARG_HINTS[http_request]='"url": "<url>", "method": "<GET|POST|PUT|DELETE>", "body": "<optional json body>"'

# Generate one shard. Args: shard_num, mode (answer|tool_name), batch_size
generate_shard() {
    local shard_num="$1"
    local mode="$2"
    local batch="$3"
    local shard_file="$OUT_DIR/haiku_shard_$(printf '%04d' "$shard_num").jsonl"
    local seed_words
    seed_words=$(shuf -n 30 "$DICT" | tr '\n' ' ')

    local filled_prompt
    if [ "$mode" = "answer" ]; then
        # shellcheck disable=SC2059
        filled_prompt=$(printf "$ANSWER_PROMPT" "$batch" "$seed_words")
    else
        local tool_name="$mode"
        local arg_hint="${TOOL_ARG_HINTS[$tool_name]}"
        # shellcheck disable=SC2059
        filled_prompt=$(printf "$TOOL_PROMPT" "$batch" "$tool_name" "$tool_name" "$arg_hint" "$tool_name" "$seed_words")
    fi

    RAW=$(claude_cmd "$filled_prompt" 2>/dev/null || true)

    if [ -z "$RAW" ]; then
        echo "    shard $shard_num ($mode): empty response" >&2
        return
    fi

    VALID=$(echo "$RAW" | grep -E '^\s*\{' || true)
    if [ -z "$VALID" ]; then
        echo "    shard $shard_num ($mode): no JSON lines" >&2
        return
    fi

    WRAPPED=$(echo "$VALID" | "$GENERATE_BIN" -wrap 2>"$OUT_DIR/wrap_${shard_num}.log" || true)
    if [ -n "$WRAPPED" ]; then
        echo "$WRAPPED" > "$shard_file"
        N=$(echo "$WRAPPED" | wc -l)
        echo "    shard $shard_num ($mode): $N accepted" >&2
    else
        echo "    shard $shard_num ($mode): all rejected" >&2
    fi
}

TOTAL_REQUESTED=0
ROUND=0

# Resume shard numbering from existing files.
SHARD=0
if ls "$OUT_DIR"/haiku_shard_*.jsonl 1>/dev/null 2>&1; then
    LAST=$(ls "$OUT_DIR"/haiku_shard_*.jsonl | sort | tail -1 | grep -oP '\d{4}')
    SHARD=$((10#$LAST + 1))
    echo "Resuming from shard $SHARD (found existing shards)"
fi
NUM_ROUNDS=$(( (COUNT + SAMPLES_PER_ROUND - 1) / SAMPLES_PER_ROUND ))

echo "Generating $COUNT samples via haiku (batch=$BATCH_SIZE, parallel=$PARALLEL, ~$NUM_ROUNDS rounds)"
echo "  Mix: 50% answer, 50% tool calls (script-controlled)"

while [ "$TOTAL_REQUESTED" -lt "$COUNT" ]; do
    ROUND=$((ROUND + 1))

    REMAINING=$((COUNT - TOTAL_REQUESTED))
    JOBS=$((REMAINING / BATCH_SIZE))
    if [ $((REMAINING % BATCH_SIZE)) -ne 0 ]; then
        JOBS=$((JOBS + 1))
    fi
    if [ "$JOBS" -gt "$PARALLEL" ]; then
        JOBS=$PARALLEL
    fi

    echo "  round $ROUND/$NUM_ROUNDS: launching $JOBS jobs..."

    PIDS=()
    for J in $(seq 1 "$JOBS"); do
        THIS_BATCH=$BATCH_SIZE
        REMAINING=$((COUNT - TOTAL_REQUESTED))
        if [ "$THIS_BATCH" -gt "$REMAINING" ]; then
            THIS_BATCH=$REMAINING
        fi
        if [ "$THIS_BATCH" -le 0 ]; then
            break
        fi

        # Alternate: even jobs = answer, odd jobs = random tool.
        if [ $((J % 2)) -eq 0 ]; then
            MODE="answer"
        else
            # Pick a random tool type.
            MODE="${TOOL_NAMES[$((RANDOM % ${#TOOL_NAMES[@]}))]}"
        fi

        generate_shard "$SHARD" "$MODE" "$THIS_BATCH" &
        PIDS+=($!)

        SHARD=$((SHARD + 1))
        TOTAL_REQUESTED=$((TOTAL_REQUESTED + THIS_BATCH))
    done

    for PID in "${PIDS[@]}"; do
        wait "$PID" 2>/dev/null || true
    done

    ACTUAL=0
    if ls "$OUT_DIR"/haiku_shard_*.jsonl 1>/dev/null 2>&1; then
        ACTUAL=$(wc -l "$OUT_DIR"/haiku_shard_*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
    fi
    echo "  round $ROUND done. $ACTUAL accepted so far (requested $TOTAL_REQUESTED/$COUNT)"
done

ACTUAL=0
if ls "$OUT_DIR"/haiku_shard_*.jsonl 1>/dev/null 2>&1; then
    ACTUAL=$(wc -l "$OUT_DIR"/haiku_shard_*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
fi
echo "Done. $ACTUAL valid training pairs in $OUT_DIR"
