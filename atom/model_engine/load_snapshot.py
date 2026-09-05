# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang-compatible admission snapshots for the standalone server."""

from __future__ import annotations

import math
import time
from typing import Any

LOAD_SNAPSHOT_MAX_AGE_SECONDS = 5.0


def build_sglang_loads(
    *,
    config: Any,
    max_pool_tokens: int | None,
    rank_stats: dict[int, dict],
    now: float | None = None,
) -> dict[str, list[dict[str, int | float]]]:
    """Build ``/v1/loads`` for the supported one-DP standalone topology.

    ATOM's BlockManager counts physical, DCP-local KV blocks.  Its
    ``max_pool_tokens`` startup fact is already the corresponding global-token
    capacity, so it is the authoritative virtual total and must not be widened
    a second time.
    """
    dp_size = int(config.parallel_config.data_parallel_size)
    if dp_size != 1:
        raise ValueError(f"/v1/loads currently supports DP1, got DP{dp_size}")

    stats = rank_stats.get(0)
    if not stats or not stats.get("enabled", False):
        raise RuntimeError("engine load snapshot is not available yet")
    timestamp = stats.get("timestamp")
    if (
        not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(timestamp)
    ):
        raise RuntimeError("engine load snapshot has no valid timestamp")
    age = (time.time() if now is None else now) - timestamp
    if age < -5.0 or age > LOAD_SNAPSHOT_MAX_AGE_SECONDS:
        raise RuntimeError(f"engine load snapshot is stale ({age:.1f}s old)")

    dcp_size = max(int(config.decode_context_parallel_size), 1)
    block_size = int(config.kv_cache_block_size)
    blocks_total = int(stats.get("kv_blocks_total", 0))
    blocks_used = int(stats.get("kv_blocks_used", 0))
    blocks_free = int(stats.get("kv_blocks_free", max(blocks_total - blocks_used, 0)))
    effective_total = int(max_pool_tokens or 0)
    physical_total = blocks_total * block_size
    expected_effective = physical_total * dcp_size
    if physical_total <= 0 or effective_total <= 0:
        raise RuntimeError("engine KV capacity is not available yet")
    if effective_total != expected_effective:
        raise RuntimeError(
            "engine KV capacity is inconsistent: "
            f"max_pool_tokens={effective_total}, physical={physical_total}, "
            f"dcp_size={dcp_size}"
        )

    used_virtual = min(max(blocks_used, 0) * block_size * dcp_size, effective_total)
    running = max(int(stats.get("requests_running", 0)), 0)
    waiting = max(int(stats.get("requests_waiting", 0)), 0)
    max_num_seqs = max(int(config.max_num_seqs), 1)
    return {
        "loads": [
            {
                "timestamp": float(timestamp),
                "dp_rank": 0,
                "num_waiting_uncached_tokens": max(
                    int(stats.get("waiting_uncached_tokens", 0)), 0
                ),
                # Sail treats this value as already widened allocator units.
                "num_used_tokens": used_virtual,
                # Preserve the SGLang contract: physical plus explicit virtual.
                "max_total_num_tokens": physical_total,
                "effective_max_total_num_tokens_per_dp": effective_total,
                "total_prefill_uncached_tokens": max(
                    int(stats.get("total_prefill_uncached_tokens", 0)), 0
                ),
                "total_prefill_busy_us": max(
                    int(stats.get("total_prefill_busy_us", 0)), 0
                ),
                "slots_total": max_num_seqs,
                "slots_free": max(max_num_seqs - running, 0),
                "running_reqs": running,
                "waiting_reqs": waiting,
                "max_available_kv_blocks": max(blocks_free, 0),
                "kv_block_tokens": block_size * dcp_size,
            }
        ]
    }
