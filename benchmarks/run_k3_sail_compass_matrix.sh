#!/usr/bin/env bash
set -euo pipefail

config="${1:?usage: $0 CONFIG RESULT_DIR}"
result_dir="${2:?usage: $0 CONFIG RESULT_DIR}"
variants="${ATOM_MATRIX_VARIANTS:-plain dcp dspark dcp-dspark}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${result_dir}"

previous=""
for variant in ${variants}; do
  if [[ -n "${previous}" ]]; then
    sudo docker rm -f "atom-k3-ablation-atom-${previous}" >/dev/null 2>&1 || true
  fi
  "${script_dir}/run_k3_sail_compass_variant.sh" \
    "${variant}" \
    "${config}" \
    "${result_dir}/atom-${variant}-stock-equal.json"
  previous="${variant}"
done

if [[ -n "${previous}" ]]; then
  sudo docker rm -f "atom-k3-ablation-atom-${previous}" >/dev/null 2>&1 || true
fi
