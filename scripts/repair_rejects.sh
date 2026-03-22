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

# Repair rejected haiku samples using dual LLM passes.
#
# For each rejected sample, asks Claude to fix the JSON twice independently.
# If both fixes agree (low edit distance) and the result parses, generates
# a training pair: broken JSON → correct XML.
#
# Incrementally updates the rejects file — successfully repaired items are
# removed after each batch, so the process is resumable.
#
# Usage: ./scripts/repair_rejects.sh [rejects_file] [output_file]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPAIR_BIN="$PROJECT_DIR/tmp/repair"

REJECTS="${1:-$PROJECT_DIR/data/rejects/rejects.jsonl}"
OUTPUT="${2:-$PROJECT_DIR/data/rejects/repaired_pairs.jsonl}"
PARALLEL=8

if [ ! -f "$REPAIR_BIN" ]; then
    echo "Repair binary not found. Run: go build -o tmp/repair ./cmd/repair/"
    exit 1
fi

if [ ! -f "$REJECTS" ]; then
    echo "Rejects file not found: $REJECTS"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

# Snapshot the rejects file so line numbers stay stable across rewrites.
SNAPSHOT="$PROJECT_DIR/tmp/repair_snapshot_$$.jsonl"
cp "$REJECTS" "$SNAPSHOT"
TOTAL=$(wc -l < "$SNAPSHOT")

if [ "$TOTAL" -eq 0 ]; then
    echo "No rejects to repair."
    rm -f "$SNAPSHOT"
    exit 0
fi

echo "Repairing $TOTAL rejected samples from $REJECTS"
echo "  Output: $OUTPUT"
echo "  Parallel: $PARALLEL"

REPAIR_SYSTEM='You are a JSON repair tool. You will receive broken JSON. Fix it so it parses correctly.

Rules:
- Make MINIMAL changes. Only fix structural issues (missing/extra brackets, commas, colons, quotes).
- Do NOT change, add, or remove any content/values.
- Do NOT reformat or pretty-print. Keep the same format as the input.
- Output ONLY the fixed JSON. No explanation, no markdown, no wrapping.'

claude_repair() {
    echo "$1" | CLAUDECODE= MAX_THINKING_TOKENS=0 claude -p \
        --no-session-persistence \
        --model=sonnet \
        --tools='' \
        --disable-slash-commands \
        --setting-sources='' \
        --system-prompt "$REPAIR_SYSTEM" \
        "Fix the broken JSON provided on stdin:"
}

# Process one reject: two independent repairs, output repair record.
repair_one() {
    local line_num="$1"
    local broken="$2"
    local out_dir="$3"

    # Two independent repair passes via stdin.
    local fix_a fix_b
    fix_a=$(claude_repair "$broken" 2>/dev/null || true)
    fix_b=$(claude_repair "$broken" 2>/dev/null || true)

    if [ -z "$fix_a" ] || [ -z "$fix_b" ]; then
        echo "  line $line_num: empty LLM response" >&2
        return
    fi

    # Strip markdown code fences, take first non-empty line only.
    fix_a=$(echo "$fix_a" | sed '/^```/d' | sed '/^$/d' | head -1)
    fix_b=$(echo "$fix_b" | sed '/^```/d' | sed '/^$/d' | head -1)

    # Build repair record as JSONL. Escape for JSON embedding.
    # Each job writes to its own file to avoid interleaving.
    local orig_escaped fix_a_escaped fix_b_escaped
    orig_escaped=$(echo "$broken" | jq -Rs '.')
    fix_a_escaped=$(echo "$fix_a" | jq -Rs '.')
    fix_b_escaped=$(echo "$fix_b" | jq -Rs '.')

    echo "{\"original\": $orig_escaped, \"fix_a\": $fix_a_escaped, \"fix_b\": $fix_b_escaped}" > "$out_dir/record_${line_num}.jsonl"
}

# Validate a batch of repair records, append accepted pairs to output,
# and update the rejects file to remove successfully repaired items.
process_batch() {
    local batch_records
    batch_records=$(ls "$RECORDS_DIR"/record_*.jsonl 2>/dev/null || true)
    if [ -z "$batch_records" ]; then
        return
    fi

    local batch_accepted=0
    local batch_failed=0

    for record_file in $batch_records; do
        local lnum
        lnum=$(basename "$record_file" | grep -oP '\d+')
        local result
        result=$("$REPAIR_BIN" -max-dist 50 -max-change-pct 10 < "$record_file" 2>/dev/null || true)
        if [ -n "$result" ]; then
            echo "$result" >> "$OUTPUT"
            echo "$lnum" >> "$ACCEPTED_FILE"
            batch_accepted=$((batch_accepted + 1))
            TOTAL_ACCEPTED=$((TOTAL_ACCEPTED + 1))
        else
            batch_failed=$((batch_failed + 1))
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
        rm -f "$record_file"
    done

    # Rewrite rejects file: snapshot minus all accepted lines so far.
    awk 'NR==FNR{skip[$1]; next} !(FNR in skip)' "$ACCEPTED_FILE" "$SNAPSHOT" > "$REJECTS.tmp"
    mv "$REJECTS.tmp" "$REJECTS"

    local remaining
    remaining=$(wc -l < "$REJECTS")
    echo "  batch: +$batch_accepted repaired, $batch_failed failed | totals: $TOTAL_ACCEPTED repaired, $remaining remaining"
}

RECORDS_DIR="$PROJECT_DIR/tmp/repair_records_$$"
mkdir -p "$RECORDS_DIR"

ACCEPTED_FILE="$PROJECT_DIR/tmp/repair_accepted_$$"
touch "$ACCEPTED_FILE"

PROCESSED=0
LINE_NUM=0
TOTAL_ACCEPTED=0
TOTAL_FAILED=0

while IFS= read -r line; do
    LINE_NUM=$((LINE_NUM + 1))
    line=$(echo "$line" | tr -d '\r')

    if [ -z "$line" ]; then
        continue
    fi

    repair_one "$LINE_NUM" "$line" "$RECORDS_DIR" &
    PROCESSED=$((PROCESSED + 1))

    # Throttle parallelism — validate and update after each batch.
    if [ $((PROCESSED % PARALLEL)) -eq 0 ]; then
        wait
        process_batch
        echo "  processed $PROCESSED/$TOTAL..."
    fi
done < "$SNAPSHOT"

# Wait for remaining jobs and process final batch.
wait
process_batch

# Cleanup temp files.
rm -f "$SNAPSHOT" "$ACCEPTED_FILE"
rmdir "$RECORDS_DIR" 2>/dev/null || true

PAIRS=0
if [ -f "$OUTPUT" ]; then
    PAIRS=$(wc -l < "$OUTPUT")
fi
REMAINING=$(wc -l < "$REJECTS")
echo ""
echo "Done. $TOTAL_ACCEPTED repaired this run ($PAIRS total pairs in $OUTPUT), $REMAINING remaining rejects"
