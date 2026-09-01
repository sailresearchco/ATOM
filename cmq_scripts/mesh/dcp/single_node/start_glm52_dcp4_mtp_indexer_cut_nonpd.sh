#!/bin/bash
# GLM-5.2 single-node TP4 + DCP4 + MTP3 with the indexer sharded across DCP
# ranks (ATOM_DCP_REPLICATE_INDEX_CACHE=0).
#
# No PD, no Mooncake: this isolates the decode-side indexer path from the KV
# transfer, which is the other thing that has to hold for the PD variant in
# start_glm52_pd_cpp4_dcp4_indexer_cut.sh.
#
# Usage:
#   LOG_DIR=... bash cmq_scripts/mesh/dcp/single_node/start_glm52_dcp4_mtp_indexer_cut_nonpd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/mnt/models/GLM-5.2-MXFP4}"
PORT="${PORT:-8020}"
GPUS="${GPUS:-4,5,6,7}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
REPLICATE_INDEX_CACHE="${REPLICATE_INDEX_CACHE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
SPEC_ACCEPT_RATE="${SPEC_ACCEPT_RATE:-off}"
TIMESTAMP="${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/it-share/mengqingcao/log/atom_experiments/dcp4_mtp_indexer_cut_${TIMESTAMP}}"
WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-3600}"
mkdir -p "$LOG_DIR"

ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-[1,2,4,8,16,32,64,128,256]}"

export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export ATOM_MLA_PAGE_SIZE=1
export ATOM_ONLINE_QUANT_STREAMING=0
export ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047
export ATOM_USE_TRITON_MLA=0
export MAX_JOBS=16
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

MODE_ARGS=(--cudagraph-mode FULL --cudagraph-capture-sizes "$CUDAGRAPH_SIZES")
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  MODE_ARGS=(--enforce-eager)
fi
PREFIX_ARGS=()
if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  PREFIX_ARGS=(--enable_prefix_caching)
fi
SPEC_ACCEPT_ARGS=()
if [[ "$SPEC_ACCEPT_RATE" != "off" ]]; then
  SPEC_ACCEPT_ARGS=(--spec-decode-acceptance-rate "$SPEC_ACCEPT_RATE")
fi

{
  echo "timestamp=$(date -Is)"
  echo "model=$MODEL"
  echo "gpus=$GPUS"
  echo "num_speculative_tokens=$NUM_SPEC_TOKENS"
  echo "replicate_index_cache=$REPLICATE_INDEX_CACHE"
  echo "enforce_eager=$ENFORCE_EAGER"
  echo "enable_prefix_caching=$ENABLE_PREFIX_CACHING"
  echo "spec_accept_rate=$SPEC_ACCEPT_RATE"
  echo "log_dir=$LOG_DIR"
} | tee "${LOG_DIR}/config.txt"

# Deliberately not exported: only this server should see it, matching the PD
# script where the prefill node must not inherit the flag.
HIP_VISIBLE_DEVICES="$GPUS" \
ATOM_DCP_REPLICATE_INDEX_CACHE="$REPLICATE_INDEX_CACHE" \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --host 0.0.0.0 --server-port "$PORT" --trust-remote-code \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
  "${PREFIX_ARGS[@]}" --max-num-seqs 512 \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --level 3 --method mtp --num-speculative-tokens "$NUM_SPEC_TOKENS" \
  "${SPEC_ACCEPT_ARGS[@]}" \
  -tp 4 --decode-context-parallel-size 4 \
  "${MODE_ARGS[@]}" \
  > "${LOG_DIR}/server.log" 2>&1 &
echo $! > "${LOG_DIR}/server.pid"

echo ">>> Waiting for http://127.0.0.1:${PORT}/v1/models (timeout ${WAIT_SERVER_TIMEOUT}s)"
deadline=$(( $(date +%s) + WAIT_SERVER_TIMEOUT ))
until curl -sf --max-time 10 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if grep -qE 'SHUTDOWN|proc died|AssertionError|Traceback \(most recent call last\)' \
      "${LOG_DIR}/server.log" 2>/dev/null; then
    if ! curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "ERROR: server failed; last 40 lines of ${LOG_DIR}/server.log:"
      tail -40 "${LOG_DIR}/server.log"
      exit 1
    fi
  fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "ERROR: server not ready after ${WAIT_SERVER_TIMEOUT}s"
    tail -40 "${LOG_DIR}/server.log"
    exit 1
  fi
  sleep 10
done

echo ""
echo "=== DCP4 + MTP${NUM_SPEC_TOKENS} (replicate_index_cache=${REPLICATE_INDEX_CACHE}) up ==="
echo "  api:  http://127.0.0.1:${PORT}/v1/chat/completions"
echo "  logs: ${LOG_DIR}/"
