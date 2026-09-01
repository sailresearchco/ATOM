#!/bin/bash
# PD-separated: PP4 prefill + DCP4 MTP3 decode + sharded index staging/RDMA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR}"
mkdir -p "$RUN_DIR"

export ATOM_EXPERIMENT_SCENARIO=pd_separate
export EXPERIMENT_CONFIG=gsm8k20_pd_separate
export EXPERIMENT_TIMESTAMP="${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${RUN_DIR}/server"
export ENABLE_LMCACHE=1
export REPLICATE_INDEX_CACHE=0

exec bash cmq_scripts/mesh/dcp/single_node/start_glm52_pd_cpp4_dcp4_indexer_cut.sh
