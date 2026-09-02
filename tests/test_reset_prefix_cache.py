import asyncio
import array
import json
import queue
from types import SimpleNamespace

from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.engine_utility import EngineUtilityHandler


def _indexed_block_manager() -> BlockManager:
    bm = BlockManager(
        MockConfig(
            enable_prefix_caching=True,
            pool_entries={"state": 2},
            pool_entries_per_req={"state": 1},
        )
    )
    bm.kv.publish(0, 101, array.array("i", [1, 2, 3, 4]))
    bm.state._index(101, 0)
    return bm


def _handler(bm: BlockManager, counts=(0, 0)):
    output = queue.Queue()
    scheduler = SimpleNamespace(
        block_manager=bm,
        get_request_counts=lambda: counts,
        engine_stats=SimpleNamespace(engine_index=0),
        _METRICS_ROLE="",
    )
    return EngineUtilityHandler(None, output, scheduler=scheduler), output


def test_block_manager_reset_reports_and_clears_prefix_and_state_indexes():
    bm = _indexed_block_manager()
    assert bm.cache_index_counts() == {
        "prefix_blocks_indexed": 1,
        "state_checkpoints_indexed": 1,
        "state_checkpoints_pending": 0,
    }
    bm.clear_cache()
    assert not any(bm.cache_index_counts().values())


def test_reset_utility_refuses_busy_scheduler_without_mutation():
    bm = _indexed_block_manager()
    handler, output = _handler(bm, counts=(1, 2))
    handler._execute_utility_command("reset_prefix_cache", {})
    _, response = output.get_nowait()
    result = response["result"]
    assert result["cleared"] is False
    assert result["before"]["running_requests"] == 1
    assert result["before"]["waiting_requests"] == 2
    assert bm.cache_index_counts()["prefix_blocks_indexed"] == 1
    assert bm.cache_index_counts()["state_checkpoints_indexed"] == 1


def test_reset_utility_returns_verifiable_before_after_counts():
    handler, output = _handler(_indexed_block_manager())
    handler._execute_utility_command("reset_prefix_cache", {})
    _, response = output.get_nowait()
    result = response["result"]
    assert result["cleared"] is True
    assert result["before"]["prefix_blocks_indexed"] == 1
    assert result["before"]["state_checkpoints_indexed"] == 1
    assert result["after"]["prefix_blocks_indexed"] == 0
    assert result["after"]["state_checkpoints_indexed"] == 0


def test_llm_engine_preflight_does_not_reset_when_any_engine_is_busy():
    from atom.model_engine.llm_engine import LLMEngine

    calls = []

    def broadcast(cmd, **kwargs):
        calls.append(cmd)
        return [
            {
                "result": {
                    "engine_index": 0,
                    "supported": True,
                    "running_requests": 1,
                    "waiting_requests": 0,
                }
            }
        ]

    fake = SimpleNamespace(
        core_mgr=SimpleNamespace(broadcast_utility_command_sync=broadcast)
    )
    result = LLMEngine.reset_prefix_cache(fake)
    assert result["status"] == "busy"
    assert calls == ["inspect_prefix_cache"]


def test_llm_engine_reports_busy_if_engine_becomes_active_during_reset():
    from atom.model_engine.llm_engine import LLMEngine

    calls = []

    def broadcast(cmd, **kwargs):
        calls.append(cmd)
        if cmd == "inspect_prefix_cache":
            return [
                {
                    "result": {
                        "engine_index": 0,
                        "supported": True,
                        "running_requests": 0,
                        "waiting_requests": 0,
                    }
                }
            ]
        return [
            {
                "result": {
                    "cleared": False,
                    "before": {
                        "supported": True,
                        "running_requests": 1,
                        "waiting_requests": 0,
                    },
                    "after": {"prefix_blocks_indexed": 1},
                }
            }
        ]

    fake = SimpleNamespace(
        core_mgr=SimpleNamespace(broadcast_utility_command_sync=broadcast)
    )
    result = LLMEngine.reset_prefix_cache(fake)
    assert result["status"] == "busy"
    assert result["cleared"] is False
    assert calls == ["inspect_prefix_cache", "reset_prefix_cache"]


def test_http_endpoint_maps_busy_reset_to_conflict(monkeypatch):
    from atom.entrypoints.openai import api_server

    fake = SimpleNamespace(
        reset_prefix_cache=lambda: {
            "status": "busy",
            "cleared": False,
            "engines": [],
        }
    )
    monkeypatch.setattr(api_server, "engine", fake)
    response = asyncio.run(api_server.reset_prefix_cache())
    assert response.status_code == 409
    assert json.loads(response.body)["cleared"] is False
