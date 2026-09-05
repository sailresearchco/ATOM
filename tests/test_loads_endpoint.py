import asyncio
import time
from types import SimpleNamespace

import pytest

from atom.model_engine.load_snapshot import build_sglang_loads


def _config(*, dcp=8, dp=1, block_size=128, max_num_seqs=32):
    return SimpleNamespace(
        decode_context_parallel_size=dcp,
        kv_cache_block_size=block_size,
        max_num_seqs=max_num_seqs,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )


def _stats(**overrides):
    values = {
        "enabled": True,
        "timestamp": 1_788_000_000.25,
        "kv_blocks_total": 952,
        "kv_blocks_used": 200,
        "kv_blocks_free": 752,
        "requests_running": 7,
        "requests_waiting": 3,
        "waiting_uncached_tokens": 12_345,
        "total_prefill_uncached_tokens": 456_789,
        "total_prefill_busy_us": 9_876_543,
    }
    values.update(overrides)
    return {0: values}


def test_dcp8_load_uses_physical_and_virtual_units_once():
    physical = 952 * 128
    virtual = physical * 8
    payload = build_sglang_loads(
        config=_config(),
        max_pool_tokens=virtual,
        rank_stats=_stats(),
        now=1_788_000_001.0,
    )

    assert len(payload["loads"]) == 1
    load = payload["loads"][0]
    assert load["dp_rank"] == 0
    assert load["timestamp"] == 1_788_000_000.25
    assert load["max_total_num_tokens"] == physical
    assert load["effective_max_total_num_tokens_per_dp"] == virtual
    assert load["num_used_tokens"] == 200 * 128 * 8
    assert virtual - load["num_used_tokens"] == 770_048
    assert load["num_waiting_uncached_tokens"] == 12_345
    assert load["total_prefill_uncached_tokens"] == 456_789
    assert load["total_prefill_busy_us"] == 9_876_543
    assert load["slots_free"] == 25
    assert load["max_available_kv_blocks"] == 752


def test_load_fails_closed_for_non_dp1_or_inconsistent_capacity():
    with pytest.raises(ValueError, match="supports DP1"):
        build_sglang_loads(
            config=_config(dp=2),
            max_pool_tokens=952 * 128 * 8,
            rank_stats=_stats(),
            now=1_788_000_001.0,
        )
    with pytest.raises(RuntimeError, match="inconsistent"):
        build_sglang_loads(
            config=_config(),
            max_pool_tokens=1_000_000,
            rank_stats=_stats(),
            now=1_788_000_001.0,
        )


def test_stale_engine_snapshot_is_not_retimestamped_by_api():
    with pytest.raises(RuntimeError, match="stale"):
        build_sglang_loads(
            config=_config(),
            max_pool_tokens=952 * 128 * 8,
            rank_stats=_stats(timestamp=100.0),
            now=106.0,
        )


def test_scheduler_queue_includes_waiting_and_partial_prefill_only():
    from atom.model_engine.scheduler import Scheduler

    def seq(prompt, cached, partial=False):
        return SimpleNamespace(
            num_prompt_tokens=prompt,
            num_cached_tokens=cached,
            is_partial_prefill=partial,
        )

    scheduler = SimpleNamespace(
        # A newly arrived prefix hit is conservatively full-sized until the
        # scheduler probes the cache and updates num_cached_tokens.
        waiting=[seq(100, 0), seq(100, 64)],
        running=[seq(200, 128, True), seq(300, 0, False)],
    )
    assert Scheduler.waiting_uncached_tokens(scheduler) == 208


def test_prefill_counters_are_monotonic_and_ignore_non_prefill_steps():
    from atom.model_engine.scheduler import Scheduler

    scheduler = SimpleNamespace(
        total_prefill_uncached_tokens=10,
        total_prefill_busy_us=20,
    )
    Scheduler.record_prefill_work(scheduler, 128, 1_000)
    Scheduler.record_prefill_work(scheduler, 0, 2_000)
    assert scheduler.total_prefill_uncached_tokens == 138
    assert scheduler.total_prefill_busy_us == 1_020


def test_http_route_returns_compatible_payload(monkeypatch):
    from atom.entrypoints.openai import api_server

    fake = SimpleNamespace(
        config=_config(),
        core_mgr=SimpleNamespace(
            max_pool_tokens=952 * 128 * 8,
            latest_loads=_stats(timestamp=time.time()),
        ),
    )
    monkeypatch.setattr(api_server, "engine", fake)

    route = next(route for route in api_server.app.routes if route.path == "/v1/loads")
    assert "GET" in route.methods
    response = asyncio.run(api_server.loads(include="core"))
    assert response["loads"][0]["effective_max_total_num_tokens_per_dp"] == (
        952 * 128 * 8
    )
