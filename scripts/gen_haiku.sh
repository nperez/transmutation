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

# Domain-specific word pools for non-tech diversity.
DOMAIN_POOLS=(
    # Medicine & healthcare
    "patient diagnosis treatment symptom prescription dosage clinical trial placebo immunology oncology radiology pathology triage anesthesia biopsy catheter dialysis edema fracture gangrene hematoma intubation jaundice lesion metastasis neonatal orthopedic prognosis quarantine resuscitation sepsis ultrasound vasculature"
    # Law & legal
    "plaintiff defendant statute jurisdiction verdict deposition subpoena arbitration liability negligence compliance injunction fiduciary indictment acquittal appellate bailiff clemency counterclaim damages estoppel felony garnishment habeas hearsay impeachment jurisprudence lien magistrate notarize perjury recusal stipulation tort waiver"
    # Finance & economics
    "portfolio dividend amortization hedge derivative futures commodity valuation arbitrage liquidity solvency underwriting actuarial annuity collateral depreciation equity foreclosure garnishment hypothecation insolvency junk leverage margin notional options premium receivable securitization tranche vesting warrant yield"
    # Education & academia
    "curriculum pedagogy syllabus enrollment accreditation dissertation thesis faculty tenure adjunct prerequisite remedial rubric sabbatical valedictorian alumnus bursar commencement dean elective fellowship matriculation probation provost registrar scholarship seminar transcription tutorial"
    # Chemistry & biology
    "photosynthesis catalyst reagent titration isotope chromatography spectroscopy fermentation distillation polymer substrate enzyme mitosis osmosis ribosome cytoplasm organelle nucleotide peptide genotype phenotype allele mutation transcription replication centrifuge precipitate molar"
    # Agriculture & farming
    "harvest irrigation fertilizer compost pesticide greenhouse cultivar pollination germination topsoil aquifer watershed agronomy fallow humus mulch perennial silage terracing trellis tuber vermiculture windbreak drought canopy rhizome grafting"
    # Music & audio
    "tempo melody harmony chord progression reverb equalization mastering synthesizer arpeggio modulation timbre dynamics crescendo diminuendo fortissimo staccato legato vibrato counterpoint dissonance consonance octave resonance frequency amplitude waveform"
    # Culinary arts
    "calorie emulsify sautee braise julienne blanch deglaze reduction roux brine marinate ferment caramelize flambe infuse macerate poach render temper confit ceviche carpaccio chiffonade concasse dredge fold"
    # Automotive & mechanical
    "velocity trajectory aerodynamic torque propulsion combustion turbine throttle manifold crankshaft differential camshaft piston cylinder carburetor supercharger intercooler catalytic drivetrain suspension caliper flywheel gasket solenoid"
    # Urban planning & architecture
    "census zoning ordinance easement setback variance corridor density transit watershed stormwater brownfield charrette cornice dormer facade fenestration lintel parapet pilaster portico soffit terrazzo vestibule cantilever"
    # Journalism & media
    "headline byline editorial syndicate circulation masthead attribution corroboration retraction embargo deadline dateline exclusive exposé freelance infographic masthead op-ed plagiarism redaction scoop sensationalism tabloid verification"
    # Game design & development
    "sprite tilemap hitbox collider shader viewport frustum quaternion raycast navmesh procedural voxel parallax culling instancing skeletal rigidbody ragdoll pathfinding heuristic minimax permadeath roguelike sandbox"
    # Environmental science
    "emissions sequestration biodiversity ecosystem watershed runoff turbidity salinity aquifer permafrost albedo deforestation eutrophication biome estuary mangrove phytoplankton zooplankton thermocline desertification remediation leachate"
    # Linguistics
    "morpheme phoneme syntax semantics pragmatics lexicon pidgin creole dialect orthography etymology affix conjugation declension diphthong fricative glottal intonation palatal sibilant uvular velar phonology"
    # Astronomy & space
    "latitude longitude azimuth declination ephemeris parallax redshift luminosity spectrograph interferometer aphelion perihelion ecliptic magnetosphere chromosphere corona nebula pulsar quasar supernova asteroid occultation"
    # Maritime & naval
    "starboard portside bulkhead keel rudder ballast bilge berth gangway mooring windlass capstan draught fathom knot nautical heading bearing tonnage freeboard gunwale"
    # Textiles & fashion
    "warp weft selvage bobbin loom spindle distaff mordant indigo batik jacquard damask organza chiffon taffeta muslin serge twill herringbone chambray"
    # Psychology & neuroscience
    "cognition amygdala hippocampus synapse neurotransmitter dopamine serotonin cortisol adrenaline prefrontal temporal parietal occipital limbic autonomic parasympathetic sympathetic habituation"
    # Philosophy & logic
    "epistemology ontology metaphysics dialectic syllogism deduction induction abduction axiom theorem postulate corollary lemma tautology paradox fallacy heuristic empiricism rationalism"
    # Geology & earth science
    "stratigraphy tectonic magma igneous sedimentary metamorphic basalt granite obsidian feldspar quartz mica schist gneiss moraine esker drumlin alluvial floodplain"
    # Veterinary & zoology
    "mammal avian reptilian amphibian invertebrate vertebrate marsupial primate cetacean ungulate carnivore herbivore omnivore pheromone symbiosis parasitism mutualism"
)

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
- thought: 3-12 sentences of detailed reasoning. Use varied vocabulary and sentence structures. Mix short punchy sentences with longer analytical ones.
- answer: substantial response text, 5-20 sentences. MUST include a mix of: markdown headers, code blocks, bullet lists, tables, inline code, comparisons using < > operators, HTML/XML tag references, URLs with & parameters, and mathematical expressions. Use apostrophes, quotes, and special characters naturally.
- tool: MUST be null
- memory: 4-10 short context strings. Include things like threshold rules (e.g. "Alert if latency > 500ms && error_rate < 0.1%%"), tag references (e.g. "Uses <config> & <auth> modules"), user preferences with apostrophes, and version/path strings.
- Topics: cover DIFFERENT domains each sample. Go beyond tech: include medicine, law, finance, logistics, education, science, cooking, sports analytics, music production, agriculture, urban planning, journalism, game design, environmental science, linguistics, etc. The agent is a general-purpose assistant, not just a DevOps bot.
- Use the seed words below as inspiration for topics and vocabulary. Do NOT repeat words across samples.
- Each line must be valid JSON. Output ONLY JSON lines, nothing else.

SEED WORDS: %s

DOMAIN WORDS: %s'

TOOL_PROMPT='You are a data generator. Output exactly %d JSON objects, one per line.

Every object MUST use the tool "%s". The answer field MUST be null.

Schema: {"thought": "...", "answer": null, "tool": {"tool_name": "%s", "arguments": {%s}}, "memory": ["...", ...]}

Constraints:
- thought: 3-12 sentences of reasoning about why this tool is needed. Use varied vocabulary and natural language. Include comparisons with < > symbols, references to thresholds, and technical detail.
- answer: MUST be null
- tool: MUST be {"tool_name": "%s", "arguments": {<realistic args>}}. Tool arguments should contain substantial, realistic content — full queries, real code, genuine file paths. Code should include comments, string literals with special chars, and varied complexity.
- memory: 4-10 short context strings. Include threshold rules with < > && operators, tag/element references, user preferences with apostrophes, and technical notes.
- Topics: cover DIFFERENT domains each sample. Go beyond tech: medicine, finance, science, logistics, education, cooking, sports, music, agriculture, law, journalism, game design, environmental science, etc.
- Use the seed words below as inspiration. Do NOT repeat vocabulary across samples.
- Each line must be valid JSON. Output ONLY JSON lines, nothing else.

SEED WORDS: %s

DOMAIN WORDS: %s'

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
    local seed_words domain_words
    seed_words=$(shuf -n 50 "$DICT" | tr '\n' ' ')
    # Pick 2 random domain pools and combine them for cross-domain variety.
    local pool1 pool2
    pool1="${DOMAIN_POOLS[$((RANDOM % ${#DOMAIN_POOLS[@]}))]}"
    pool2="${DOMAIN_POOLS[$((RANDOM % ${#DOMAIN_POOLS[@]}))]}"
    domain_words=$(echo "$pool1 $pool2" | tr ' ' '\n' | shuf -n 20 | tr '\n' ' ')

    local filled_prompt
    if [ "$mode" = "answer" ]; then
        # shellcheck disable=SC2059
        filled_prompt=$(printf "$ANSWER_PROMPT" "$batch" "$seed_words" "$domain_words")
    else
        local tool_name="$mode"
        local arg_hint="${TOOL_ARG_HINTS[$tool_name]}"
        # shellcheck disable=SC2059
        filled_prompt=$(printf "$TOOL_PROMPT" "$batch" "$tool_name" "$tool_name" "$arg_hint" "$tool_name" "$seed_words" "$domain_words")
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
