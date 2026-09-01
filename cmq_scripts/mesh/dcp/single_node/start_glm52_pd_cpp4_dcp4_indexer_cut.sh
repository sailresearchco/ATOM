#!/bin/bash
# GLM-5.2 1P1D CPP4+DCP4 matching
#   glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-1m-c32
# from .github/benchmark/models_atomesh.yaml
#
# Usage:
#   LOG_DIR=... ENABLE_LMCACHE=1 bash cmq_scripts/mesh/dcp/single_node/start_glm52_pd_cpp4_dcp4_ci.sh
#   LOG_DIR=... ENABLE_LMCACHE=0 bash cmq_scripts/mesh/dcp/single_node/start_glm52_pd_cpp4_dcp4_ci.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/mnt/models/GLM-5.2-MXFP4}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
MESH_PORT="${MESH_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29100}"
ENABLE_LMCACHE="${ENABLE_LMCACHE:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
SPEC_ACCEPT_RATE="${SPEC_ACCEPT_RATE:-off}"
REPLICATE_INDEX_CACHE="${REPLICATE_INDEX_CACHE:-0}"
HOST_IP="${ATOM_HOST_IP:-127.0.0.1}"
TIMESTAMP="${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-cpp4_dcp4}"
LOG_DIR="${LOG_DIR:-/it-share/mengqingcao/log/atom_experiments/${EXPERIMENT_CONFIG}_${TIMESTAMP}}"
WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-4500}"
PREFILL_TRACE_DIR="${PREFILL_TRACE_DIR:-}"
DECODE_TRACE_DIR="${DECODE_TRACE_DIR:-}"
ATOM_PROFILER_MORE="${ATOM_PROFILER_MORE:-0}"
ATOM_PROFILER_TIMEOUT="${ATOM_PROFILER_TIMEOUT:-600}"
mkdir -p "$LOG_DIR"

PREFILL_TRACE_ARGS=()
DECODE_TRACE_ARGS=()
if [[ -n "${PREFILL_TRACE_DIR}" ]]; then
  rm -rf "${PREFILL_TRACE_DIR}"
  mkdir -p "${PREFILL_TRACE_DIR}"
  PREFILL_TRACE_ARGS=(--torch-profiler-dir "${PREFILL_TRACE_DIR}")
fi
if [[ -n "${DECODE_TRACE_DIR}" ]]; then
  rm -rf "${DECODE_TRACE_DIR}"
  mkdir -p "${DECODE_TRACE_DIR}"
  DECODE_TRACE_ARGS=(--torch-profiler-dir "${DECODE_TRACE_DIR}")
fi
if [[ ${#PREFILL_TRACE_ARGS[@]} -gt 0 || ${#DECODE_TRACE_ARGS[@]} -gt 0 ]]; then
  export ATOM_PROFILER_MORE ATOM_PROFILER_TIMEOUT
fi

MOONCAKE_LIB="$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake"

ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'
DECODE_CUDAGRAPH='[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]'
PREFIX_ARGS=()
if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  PREFIX_ARGS=(--enable_prefix_caching)
fi
SPEC_ACCEPT_ARGS=()
if [[ "$SPEC_ACCEPT_RATE" != "off" ]]; then
  SPEC_ACCEPT_ARGS=(--spec-decode-acceptance-rate "$SPEC_ACCEPT_RATE")
fi

if [[ "$ENABLE_LMCACHE" == "1" ]]; then
  PREFILL_KV_TRANSFER_CONFIG="{\"kv_connector\":\"multi\",\"connectors\":[{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"proxy_ip\":\"${HOST_IP}\",\"handshake_port\":${HANDSHAKE_PORT},\"protocol\":\"rdma\"},{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}"
else
  PREFILL_KV_TRANSFER_CONFIG="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"proxy_ip\":\"${HOST_IP}\",\"handshake_port\":${HANDSHAKE_PORT},\"protocol\":\"rdma\"}"
fi
DECODE_KV_TRANSFER_CONFIG="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_consumer\",\"proxy_ip\":\"${HOST_IP}\",\"handshake_port\":${HANDSHAKE_PORT},\"protocol\":\"rdma\"}"

{
  echo "timestamp=$(date -Is)"
  echo "host=$(hostname)"
  echo "model=$MODEL"
  echo "enable_lmcache=$ENABLE_LMCACHE"
  echo "enable_prefix_caching=$ENABLE_PREFIX_CACHING"
  echo "spec_accept_rate=$SPEC_ACCEPT_RATE"
  echo "replicate_index_cache=$REPLICATE_INDEX_CACHE"
  echo "log_dir=$LOG_DIR"
  echo "prefill_kv=$PREFILL_KV_TRANSFER_CONFIG"
  echo "decode_kv=$DECODE_KV_TRANSFER_CONFIG"
  echo "prefill_trace_dir=${PREFILL_TRACE_DIR:-<none>}"
  echo "decode_trace_dir=${DECODE_TRACE_DIR:-<none>}"
  echo "atom_profiler_more=${ATOM_PROFILER_MORE}"
} | tee "${LOG_DIR}/config.txt"

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
export ATOM_HOST_IP="$HOST_IP"
export LD_LIBRARY_PATH="${MOONCAKE_LIB}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

pkill -f 'atom.entrypoints.openai_server' 2>/dev/null || true
pkill -f 'atomesh launch' 2>/dev/null || true
sleep 2
rm -rf /root/.cache/atom/*

echo ">>> Prefill PP4 TP1 on GPU 0-3, port ${PREFILL_PORT}, lmcache=${ENABLE_LMCACHE}"
PREFILL_ENV=(
  HIP_VISIBLE_DEVICES=0,1,2,3
  VLLM_PP_LAYER_PARTITION=20,20,20,18
)
if [[ "$ENABLE_LMCACHE" == "1" ]]; then
  PREFILL_ENV+=(
    LMCACHE_LOCAL_CPU=True
    LMCACHE_MAX_LOCAL_CPU_SIZE=256
    LMCACHE_CHUNK_SIZE=256
    OFFLOAD_PROFILE=1
    OFFLOAD_MIN_LOAD_TOKENS=0
  )
fi
env "${PREFILL_ENV[@]}" \
  nohup python -m atom.entrypoints.openai_server \
    --model "$MODEL" --host 0.0.0.0 --server-port "$PREFILL_PORT" --trust-remote-code \
    --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
    "${PREFIX_ARGS[@]}" --max-num-seqs 512 \
    --online_quant_config "$ONLINE_QUANT_CONFIG" \
    --level 3 --method mtp --num-speculative-tokens 3 "${SPEC_ACCEPT_ARGS[@]}" \
    -tp 1 --pipeline-parallel-size 4 --enforce-eager --max-num-batched-tokens 8192 \
    --kv-transfer-config "$PREFILL_KV_TRANSFER_CONFIG" \
    "${PREFILL_TRACE_ARGS[@]}" \
    > "${LOG_DIR}/prefill.log" 2>&1 &
echo $! > "${LOG_DIR}/prefill.pid"

echo ">>> Decode TP4 DCP4 on GPU 4-7, port ${DECODE_PORT}"
HIP_VISIBLE_DEVICES=4,5,6,7 \
ATOM_DCP_REPLICATE_INDEX_CACHE="$REPLICATE_INDEX_CACHE" \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --host 0.0.0.0 --server-port "$DECODE_PORT" --trust-remote-code \
  --kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85 \
  "${PREFIX_ARGS[@]}" --max-num-seqs 512 \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --level 3 --method mtp --num-speculative-tokens 3 "${SPEC_ACCEPT_ARGS[@]}" \
  -tp 4 --decode-context-parallel-size 4 --cudagraph-mode FULL \
  --cudagraph-capture-sizes "$DECODE_CUDAGRAPH" \
  --kv-transfer-config "$DECODE_KV_TRANSFER_CONFIG" \
  "${DECODE_TRACE_ARGS[@]}" \
  > "${LOG_DIR}/decode.log" 2>&1 &
echo $! > "${LOG_DIR}/decode.pid"

wait_http() {
  local url="$1" name="$2" log="$3"
  local deadline=$(( $(date +%s) + WAIT_SERVER_TIMEOUT ))
  echo ">>> Waiting for ${name} ${url} (timeout ${WAIT_SERVER_TIMEOUT}s)"
  until curl -sf --max-time 10 "${url}" >/dev/null 2>&1; do
    if grep -qE 'SHUTDOWN|proc died|AssertionError|Traceback \(most recent call last\)' "$log" 2>/dev/null; then
      if ! curl -sf --max-time 5 "${url}" >/dev/null 2>&1; then
        echo "ERROR: ${name} appears to have failed; last 40 lines of ${log}:"
        tail -40 "$log"
        exit 1
      fi
    fi
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      echo "ERROR: ${name} not ready after ${WAIT_SERVER_TIMEOUT}s"
      tail -40 "$log"
      exit 1
    fi
    sleep 10
  done
  echo "    ${name} READY"
}

wait_http "http://127.0.0.1:${PREFILL_PORT}/v1/models" "prefill" "${LOG_DIR}/prefill.log"
wait_http "http://127.0.0.1:${DECODE_PORT}/v1/models" "decode" "${LOG_DIR}/decode.log"

echo ">>> Mesh on port ${MESH_PORT}"
nohup atomesh launch \
  --host 0.0.0.0 --port "$MESH_PORT" --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" "${HANDSHAKE_PORT}" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --policy random --backend atom --log-level info \
  --disable-circuit-breaker --prometheus-port "$PROMETHEUS_PORT" \
  > "${LOG_DIR}/mesh.log" 2>&1 &
echo $! > "${LOG_DIR}/mesh.pid"

wait_http "http://127.0.0.1:${MESH_PORT}/v1/models" "mesh" "${LOG_DIR}/mesh.log"

echo ""
echo "=== PD CPP4 + DCP4 up (lmcache=${ENABLE_LMCACHE}) ==="
echo "  mesh:  http://127.0.0.1:${MESH_PORT}/v1/chat/completions"
echo "  logs:  ${LOG_DIR}/"
