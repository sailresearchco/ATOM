#!/bin/bash
# PD-mixed control: DCP4 + MTP3 + sharded indexer, no Mooncake transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR}"
mkdir -p "$RUN_DIR"

export ATOM_EXPERIMENT_SCENARIO=pd_mix
export EXPERIMENT_CONFIG=gsm8k20_pd_mix
export EXPERIMENT_TIMESTAMP="${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${RUN_DIR}/server"
export GPUS="${GPUS:-0,1,2,3}"
export PORT="${PORT:-8020}"
export REPLICATE_INDEX_CACHE=0

exec bash cmq_scripts/mesh/dcp/single_node/start_glm52_dcp4_mtp_indexer_cut_nonpd.sh
