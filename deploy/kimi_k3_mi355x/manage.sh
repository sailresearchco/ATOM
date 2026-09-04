#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
env_file="${ATOM_ENV_FILE:-${script_dir}/.env}"
compose_file="${script_dir}/compose.yaml"
manifest_root="${ATOM_MANIFEST_ROOT:-${script_dir}/manifests}"

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
compose=("${docker_cmd[@]}" compose --env-file "${env_file}" -f "${compose_file}")

action="${1:-status}"
container="${ATOM_CONTAINER_NAME:-sail-atom-k3}"
server_port="${ATOM_SERVER_PORT:-31000}"
served_model="${ATOM_SERVED_MODEL:-moonshotai/Kimi-K3}"

require_env() {
  if [[ ! -f "${env_file}" ]]; then
    echo "missing ${env_file}; copy .env.example and set ATOM_IMAGE" >&2
    exit 2
  fi
}

validate_host() {
  require_env
  : "${ATOM_IMAGE:?ATOM_IMAGE must be set}"
  "${docker_cmd[@]}" image inspect "${ATOM_IMAGE}" >/dev/null
  if [[ ! -e /dev/kfd ]]; then
    echo "/dev/kfd is missing" >&2
    exit 1
  fi
  local gpu_count
  gpu_count="$(ls -d /sys/class/drm/card[0-9]* 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${gpu_count}" -lt 8 ]]; then
    echo "expected at least 8 GPUs, found ${gpu_count}" >&2
    exit 1
  fi
  if ss -H -ltn "sport = :${server_port}" | grep -q . && \
     ! "${docker_cmd[@]}" inspect "${container}" >/dev/null 2>&1; then
    echo "port ${server_port} is owned by another process" >&2
    exit 1
  fi
}

record_manifest() {
  mkdir -p "${manifest_root}"
  local stamp output
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output="${manifest_root}/${stamp}-$(hostname)-${container}.json"
  "${docker_cmd[@]}" inspect "${container}" > "${output}"
  echo "manifest=${output}"
}

wait_ready() {
  local deadline state
  deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    state="$("${docker_cmd[@]}" inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container}" 2>/dev/null || true)"
    if [[ "${state}" == "healthy" ]]; then
      python3 "${script_dir}/smoke.py" \
        --base-url "http://127.0.0.1:${server_port}" \
        --model "${served_model}"
      return
    fi
    if [[ "${state}" == "exited" || "${state}" == "dead" || "${state}" == "unhealthy" ]]; then
      "${compose[@]}" logs --tail 300 server >&2 || true
      echo "ATOM entered state ${state}" >&2
      exit 1
    fi
    sleep 10
  done
  "${compose[@]}" logs --tail 300 server >&2 || true
  echo "ATOM did not become healthy within 30 minutes" >&2
  exit 1
}

case "${action}" in
  build)
    require_env
    revision="$(git -C "${repo_root}" rev-parse HEAD)"
    version="0.1.6rc1.dev382+g${revision:0:8}.sail"
    image="${ATOM_IMAGE:?ATOM_IMAGE must be set to the output tag}"
    "${docker_cmd[@]}" build \
      --build-arg "ATOM_BASE_IMAGE=${ATOM_BASE_IMAGE:-sailresearchco/atom@sha256:5e406e193929971848a9d49cabf95d7231f6f92b7892b9d3612ca0d01fa643cb}" \
      --build-arg "ATOM_SOURCE_COMMIT=${revision}" \
      --build-arg "ATOM_SOURCE_VERSION=${version}" \
      --file "${script_dir}/Dockerfile.runtime" \
      --tag "${image}" \
      "${repo_root}"
    actual="$("${docker_cmd[@]}" image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "${image}")"
    [[ "${actual}" == "${revision}" ]] || { echo "image revision mismatch" >&2; exit 1; }
    "${docker_cmd[@]}" image inspect --format 'image_id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}' "${image}"
    ;;
  validate)
    validate_host
    echo "host validation passed"
    ;;
  up)
    validate_host
    "${compose[@]}" up --detach --remove-orphans
    record_manifest
    wait_ready
    ;;
  wait)
    require_env
    wait_ready
    ;;
  smoke)
    python3 "${script_dir}/smoke.py" \
      --base-url "http://127.0.0.1:${server_port}" \
      --model "${served_model}"
    ;;
  status)
    require_env
    "${compose[@]}" ps
    ;;
  logs)
    require_env
    "${compose[@]}" logs --follow --tail 200 server
    ;;
  down)
    require_env
    "${compose[@]}" down --timeout 300
    ;;
  *)
    echo "usage: $0 build|validate|up|wait|smoke|status|logs|down" >&2
    exit 2
    ;;
esac
