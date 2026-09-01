#!/bin/bash
# Run the full GSM8K 20-shot evaluation and preserve aggregate + sample logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"

PORT="${1:?usage: $0 PORT RUN_DIR [SCENARIO]}"
RUN_DIR="${2:?usage: $0 PORT RUN_DIR [SCENARIO]}"
SCENARIO="${3:-unknown}"
MODEL="${MODEL:-/mnt/models/GLM-5.2-MXFP4}"
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
EVAL_DIR="${RUN_DIR}/eval"
SAMPLES_DIR="${EVAL_DIR}/samples"
mkdir -p "$SAMPLES_DIR"

{
  echo "timestamp=$(date -Is)"
  echo "scenario=$SCENARIO"
  echo "model=$MODEL"
  echo "port=$PORT"
  echo "num_fewshot=20"
  echo "num_concurrent=$NUM_CONCURRENT"
  echo "gen_kwargs=max_tokens=16384,temperature=0,top_p=1"
  echo "samples_dir=$SAMPLES_DIR"
} | tee "${EVAL_DIR}/config.txt"

CHAT=1 \
NUM_CONCURRENT="$NUM_CONCURRENT" \
GEN_KWARGS="max_tokens=16384,temperature=0,top_p=1" \
LOG_SAMPLES_DIR="$SAMPLES_DIR" \
bash scripts/run_gsm8k_eval.sh "$MODEL" "$PORT" 20 \
  2>&1 | tee "${EVAL_DIR}/gsm8k_20shot.log"
