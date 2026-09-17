#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 plain|dcp|dspark|dspark-N|dcp-dspark|dcp-dspark-N CONFIG EMIT_RUN_INFO}"
config="${2:?usage: $0 VARIANT CONFIG EMIT_RUN_INFO}"
emit_run_info="${3:?usage: $0 VARIANT CONFIG EMIT_RUN_INFO}"

task_root="${ATOM_ABLATION_ROOT:-/home/sr/work/atom-k3-ablation-20260901}"
compass_root="${SAIL_COMPASS_ROOT:-${task_root}/Sail-Compass}"
compass_python="${SAIL_COMPASS_PYTHON:-/home/sr/work/sail-compass/.venv/bin/python}"
server_log="${task_root}/logs/current-server.log"
container="atom-k3-ablation-atom-${variant}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/launch_k3_sail_compass.sh" "${variant}"

ready=0
for _ in $(seq 1 90); do
  if ! sudo docker inspect --format '{{.State.Running}}' "${container}" \
    2>/dev/null | grep -qx true; then
    sudo docker logs --tail 200 "${container}" >&2 || true
    exit 1
  fi
  if curl --fail --silent --show-error \
    http://127.0.0.1:31000/health >/dev/null 2>&1 && \
    curl --fail --silent --show-error \
      http://127.0.0.1:31000/v1/models \
      | jq -e '.data | any(.id == "moonshotai/Kimi-K3")' >/dev/null; then
    sleep 2
    if sudo docker inspect --format '{{.State.Running}}' "${container}" \
      2>/dev/null | grep -qx true; then
      ready=1
      break
    fi
  fi
  sleep 10
done

if [[ "${ready}" != 1 ]]; then
  sudo docker logs --tail 200 "${container}" >&2 || true
  echo "${container} did not become ready within 15 minutes" >&2
  exit 1
fi

sudo sh -c ": > '${server_log}'"
sudo sh -c "nohup docker logs -f '${container}' > '${server_log}' 2>&1 < /dev/null &"

cd "${compass_root}"
exec "${compass_python}" src/main.py "${config}" \
  --external-server \
  --strict-outcome \
  --emit-run-info "${emit_run_info}"
