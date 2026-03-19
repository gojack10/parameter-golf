#!/usr/bin/env bash
# run_probe.sh — Load config.yml as env var defaults, apply overrides, run training.
# Ensures every run inherits the full current-best config. No more forgotten env vars.
#
# Usage:
#   ./run_probe.sh [--3070|--4090|--h100|--m2] [KEY=VALUE ...] [-- extra_args...]
#
# Examples:
#   ./run_probe.sh --3070 MATRIX_LR=0.02
#   ./run_probe.sh --h100 EARLY_STOP=0
#   ./run_probe.sh --3070 MATRIX_LR=0.05 SCALAR_LR=0.05
#   ./run_probe.sh --m2                          # smoke test on Apple Silicon
#   ./run_probe.sh --3070 -- --some-extra-flag   # pass extra args to training script
#
# Priority (highest wins):
#   1. Command-line KEY=VALUE overrides
#   2. Hardware preset (--3070/--4090/--h100/--m2)
#   3. config.yml (all UPPER_CASE keys)
#
# Loading order: CLI overrides exported first, then preset (uses :- so CLI wins),
# then config.yml fills in everything else (only sets unset vars).

set -euo pipefail
cd "$(dirname "$0")"

CONFIG_FILE="config.yml"

# --- Parse config.yml: extract UPPER_CASE key: value pairs as env defaults ---
load_config() {
    while IFS= read -r line; do
        # Skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        # Match KEY: VALUE where KEY is UPPER_CASE (with underscores/digits)
        if [[ "$line" =~ ^([A-Z][A-Z0-9_]*):\ *(.*) ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            # Strip inline comments (but preserve quoted strings)
            val="${val%%#*}"
            # Strip trailing whitespace
            val="${val%"${val##*[![:space:]]}"}"
            # Strip surrounding quotes
            val="${val%\"}"
            val="${val#\"}"
            # Only set if not already in environment (env beats config)
            if [[ -z "${!key+x}" ]]; then
                export "$key=$val"
            fi
        fi
    done < "$CONFIG_FILE"
}

# --- Hardware presets ---
preset_3070() {
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
    export ITERATIONS="${ITERATIONS:-5000}"
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-250}"
    export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
    [[ -f reference_bpb_3070.json ]] && export EARLY_STOP_REF="${EARLY_STOP_REF:-reference_bpb_3070.json}"
    TRAIN_CMD="uv run python train_gpt.py"
}

preset_4090() {
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
    export ITERATIONS="${ITERATIONS:-10000}"
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
    export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
    [[ -f reference_bpb_4090.json ]] && export EARLY_STOP_REF="${EARLY_STOP_REF:-reference_bpb_4090.json}"
    TRAIN_CMD="uv run python train_gpt.py"
}

preset_h100() {
    # Uses config.yml defaults (already tuned for 8xH100)
    export EARLY_STOP_REF="${EARLY_STOP_REF:-reference_bpb_8xh100.json}"
    TRAIN_CMD="uv run torchrun --nproc_per_node=8 train_gpt.py"
}

preset_m2() {
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
    export ITERATIONS="${ITERATIONS:-500}"
    export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-16384}"
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-512}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
    export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
    TRAIN_CMD="uv run python train_gpt_mlx.py"
}

# --- Parse arguments ---
HARDWARE=""
OVERRIDES=()
EXTRA_ARGS=()
PAST_SEPARATOR=false

for arg in "$@"; do
    if $PAST_SEPARATOR; then
        EXTRA_ARGS+=("$arg")
    elif [[ "$arg" == "--" ]]; then
        PAST_SEPARATOR=true
    elif [[ "$arg" == "--3070" ]]; then
        HARDWARE="3070"
    elif [[ "$arg" == "--4090" ]]; then
        HARDWARE="4090"
    elif [[ "$arg" == "--h100" ]]; then
        HARDWARE="h100"
    elif [[ "$arg" == "--m2" ]]; then
        HARDWARE="m2"
    elif [[ "$arg" =~ ^[A-Z][A-Z0-9_]*= ]]; then
        OVERRIDES+=("$arg")
    else
        echo "Unknown argument: $arg" >&2
        echo "Usage: ./run_probe.sh [--3070|--4090|--h100|--m2] [KEY=VALUE ...] [-- extra_args...]" >&2
        exit 1
    fi
done

# --- Apply layers: CLI overrides > hardware preset > config.yml ---

# 1. CLI overrides first (highest priority — survives preset's :- and config's unset check)
for override in "${OVERRIDES[@]}"; do
    export "$override"
done

# 2. Hardware preset (uses :- so CLI overrides win, but beats config.yml)
TRAIN_CMD="uv run python train_gpt.py"  # default
case "$HARDWARE" in
    3070) preset_3070 ;;
    4090) preset_4090 ;;
    h100) preset_h100 ;;
    m2)   preset_m2 ;;
    "")   ;; # no preset, use config.yml as-is
esac

# 3. config.yml fills in everything else (only sets vars not already set)
load_config

# --- Summary ---
echo "=== run_probe.sh ==="
echo "Hardware: ${HARDWARE:-default}"
echo "Config:   $CONFIG_FILE"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    echo "Overrides: ${OVERRIDES[*]}"
fi
echo "Command:  $TRAIN_CMD ${EXTRA_ARGS[*]:-}"
echo ""
if [[ -n "${PROBE_HYPOTHESIS:-}" ]]; then
    echo "Hypothesis: $PROBE_HYPOTHESIS"
fi
if [[ -n "${PROBE_NOTES:-}" ]]; then
    echo "Notes: $PROBE_NOTES"
fi
echo "Key env vars:"
echo "  MATRIX_LR=$MATRIX_LR  SCALAR_LR=$SCALAR_LR  EMBED_LR=$EMBED_LR"
echo "  MUON_MOMENTUM=$MUON_MOMENTUM  EVAL_STRIDE=$EVAL_STRIDE"
echo "  MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS  ITERATIONS=$ITERATIONS"
echo "  TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS  EARLY_STOP_REF=${EARLY_STOP_REF:-<none>}"
echo "===================="
echo ""

# --- Run ---
exec $TRAIN_CMD "${EXTRA_ARGS[@]}"
