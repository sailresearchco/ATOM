"""Control-flow coverage for bounded decode-only EngineCore steps."""

import argparse
import queue
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from atom.config import Config
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.engine_core import EngineCore


def _core(steps=4):
    core = EngineCore.__new__(EngineCore)
    core.num_continuous_decode_steps = steps
    core.scheduler = mock.Mock()
    core.scheduler.is_finished.return_value = False
    core.input_queue = queue.Queue()
    core._has_pending_utility = False
    core._last_step_was_decode_only = True
    core._continuous_decode_bursts = 0
    core._continuous_decode_extra_steps = 0
    core.label = "test core"
    return core


def test_continuous_decode_runs_configured_group():
    core = _core()
    core._process_engine_step = mock.Mock(return_value=True)

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 3
    assert core._process_engine_step.call_args_list == [
        mock.call(decode_only=True),
        mock.call(decode_only=True),
        mock.call(decode_only=True),
    ]
    assert core._continuous_decode_bursts == 1
    assert core._continuous_decode_extra_steps == 3


def test_one_step_preserves_original_loop_cadence():
    core = _core(steps=1)
    core._process_engine_step = mock.Mock(return_value=True)

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 0
    core._process_engine_step.assert_not_called()


def test_prefill_does_not_start_continuous_group():
    core = _core()
    core._last_step_was_decode_only = False
    core._process_engine_step = mock.Mock(return_value=True)

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 0
    core._process_engine_step.assert_not_called()


def test_delayer_hold_can_form_continuous_decode_group():
    core = _core()
    core._process_engine_step = mock.Mock(return_value=True)

    # An already-queued prefill was deliberately held by the normal scheduler
    # pass that produced the preceding decode. It must not cause a stateful
    # second cache-admission probe or defeat the bounded decode group.
    core.scheduler.waiting = [object()]

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 3
    assert core._process_engine_step.call_count == 3


def test_group_yields_to_new_input():
    core = _core()
    core.input_queue.put_nowait(object())
    core._process_engine_step = mock.Mock(return_value=True)

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 0
    core._process_engine_step.assert_not_called()


def test_group_yields_to_utility_and_metrics():
    core = _core()
    core._has_pending_utility = True

    assert core._should_interrupt_continuous_decode(time.monotonic() + 60)

    core._has_pending_utility = False
    assert core._should_interrupt_continuous_decode(time.monotonic() - 1)


def test_group_stops_when_decode_work_finishes():
    core = _core()

    def finish_after_one_step(*, decode_only):
        core._last_step_was_decode_only = False
        return True

    core._process_engine_step = mock.Mock(side_effect=finish_after_one_step)

    assert core._run_continuous_decode_steps(time.monotonic() + 60) == 1
    core._process_engine_step.assert_called_once_with(decode_only=True)


def test_engine_step_publishes_kv_events_per_forward():
    core = _core()
    core._process_engine_step_inner = mock.Mock(return_value=True)

    assert core._process_engine_step() is True
    core._process_engine_step_inner.assert_called_once_with(decode_only=False)
    core.scheduler.publish_kv_events.assert_called_once_with()


def test_decode_only_scheduler_pass_leaves_prefill_queued(scheduler, seq_factory):
    decode_seq = seq_factory([1, 2, 3, 4])
    scheduler.add(decode_seq)
    scheduler.schedule()
    decode_seq.num_cached_tokens = decode_seq.num_prompt_tokens
    decode_seq.append_token(5)

    prefill_seq = seq_factory([6, 7, 8, 9])
    scheduler.add(prefill_seq)

    batch, _ = scheduler.schedule(decode_only=True)

    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 1
    assert prefill_seq in scheduler.waiting


def test_cli_exposes_continuous_decode_steps():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)

    args = EngineArgs.from_cli_args(
        parser.parse_args(["--num-continuous-decode-steps", "4"])
    )

    assert args.num_continuous_decode_steps == 4
    assert args._get_engine_kwargs()["num_continuous_decode_steps"] == 4


def test_config_rejects_non_positive_continuous_decode_steps():
    with pytest.raises(ValueError, match="must be at least 1"):
        Config.validate_continuous_decode_topology(
            SimpleNamespace(num_continuous_decode_steps=0)
        )


@pytest.mark.parametrize(
    ("dp_size", "pp_size", "rapidserve"),
    [(2, 1, False), (1, 2, False), (1, 1, True)],
)
def test_config_rejects_unsupported_continuous_decode_topologies(
    dp_size, pp_size, rapidserve
):
    config = SimpleNamespace(
        num_continuous_decode_steps=2,
        parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        pipeline_parallel_size=pp_size,
        enable_rapidserve=rapidserve,
    )

    with pytest.raises(ValueError, match="requires monolithic DP1/PP1"):
        Config.validate_continuous_decode_topology(config)


def test_topology_validation_rechecks_legacy_dp_mutation():
    config = SimpleNamespace(
        num_continuous_decode_steps=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        enable_rapidserve=False,
    )
    Config.validate_continuous_decode_topology(config)

    # LLMEngine's legacy loose `data_parallel_size` path mutates this after
    # Config construction and must validate the effective value again.
    config.parallel_config.data_parallel_size = 2
    with pytest.raises(ValueError, match="requires monolithic DP1/PP1"):
        Config.validate_continuous_decode_topology(config)
