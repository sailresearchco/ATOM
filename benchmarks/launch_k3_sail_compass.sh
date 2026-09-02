#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 plain|dcp|dspark|dspark-N|dcp-dspark|dcp-dspark-N}"
task_root="${ATOM_ABLATION_ROOT:-/home/sr/work/atom-k3-ablation-20260901}"
cache_root="${ATOM_HF_CACHE_ROOT:-/home/sr/.cache}"
image="${ATOM_BENCHMARK_IMAGE:-sha256:5e406e193929971848a9d49cabf95d7231f6f92b7892b9d3612ca0d01fa643cb}"
max_model_len="${ATOM_MAX_MODEL_LEN:-16384}"
target_snapshot="/root/.cache/huggingface/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569-materialized"
draft_snapshot="/root/.cache/huggingface/hub/models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6"
container="atom-k3-ablation-atom-${variant}"

quant_config='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'

common=(
  --model "${target_snapshot}"
  --served-model-name moonshotai/Kimi-K3
  --server-port 31000
  --port 31100
  --kv_cache_dtype fp8
  -tp 8
  --trust-remote-code
  --max-model-len "${max_model_len}"
  --max-num-seqs 64
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.93
  --block-size 128
  --no-enable_prefix_caching
  --disable-uvicorn-access-log
  --online_quant_config "${quant_config}"
)

case "${variant}" in
  plain)
    delta=()
    ;;
  dcp)
    delta=(-dcp 8)
    ;;
  dspark)
    delta=(
      --draft-model "${draft_snapshot}"
      --method dspark
      --num-speculative-tokens 7
    )
    ;;
  dspark-*)
    dspark_tokens="${variant#dspark-}"
    if ! [[ "${dspark_tokens}" =~ ^[1-9][0-9]*$ ]]; then
      echo "invalid DSpark proposal depth: ${dspark_tokens}" >&2
      exit 2
    fi
    delta=(
      --draft-model "${draft_snapshot}"
      --method dspark
      --num-speculative-tokens "${dspark_tokens}"
    )
    ;;
  dcp-dspark)
    delta=(
      -dcp 8
      --draft-model "${draft_snapshot}"
      --method dspark
      --num-speculative-tokens 2
    )
    ;;
  dcp-dspark-*)
    dspark_tokens="${variant#dcp-dspark-}"
    if ! [[ "${dspark_tokens}" =~ ^[1-9][0-9]*$ ]]; then
      echo "invalid DCP+DSpark proposal depth: ${dspark_tokens}" >&2
      exit 2
    fi
    delta=(
      -dcp 8
      --draft-model "${draft_snapshot}"
      --method dspark
      --num-speculative-tokens "${dspark_tokens}"
    )
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

mkdir -p "${task_root}/logs" "${task_root}/manifests"
sudo docker rm -f "${container}" >/dev/null 2>&1 || true
if ss -H -ltn 'sport = :31000' | grep -q .; then
  echo "port 31000 is already in use before launching ${container}" >&2
  exit 1
fi
sudo docker run -d \
  --name "${container}" \
  --privileged \
  --network host \
  --ipc host \
  --shm-size 128g \
  --device /dev/kfd \
  --device /dev/dri \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${cache_root}:/root/.cache" \
  "${image}" \
  python -m atom.entrypoints.openai_server \
  "${common[@]}" \
  "${delta[@]}"

sudo docker inspect "${container}" \
  --format '{{json .Config}}' \
  > "${task_root}/manifests/${variant}-container-config.json"
