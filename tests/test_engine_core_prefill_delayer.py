"""Construction coverage for EngineCore's shared prefill-delayer wiring."""

from types import SimpleNamespace
from unittest import mock

from atom.model_engine.engine_core import EngineCore
from atom.utils import envs


def _config(dp_size=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        max_num_batched_tokens=16384,
    )


def test_dp1_attachment_uses_local_decisions():
    core = EngineCore.__new__(EngineCore)
    core.scheduler = mock.Mock()

    with (
        mock.patch.object(envs, "ATOM_ENABLE_PREFILL_DELAYER", True),
        mock.patch("atom.model_engine.prefill_delayer.PrefillDelayer") as delayer_cls,
    ):
        core._maybe_attach_prefill_delayer(_config(), cpu_group=None)

    assert delayer_cls.call_args.kwargs["dp_size"] == 1
    assert delayer_cls.call_args.kwargs["cpu_group"] is None
    core.scheduler.set_prefill_delayer.assert_called_once_with(delayer_cls.return_value)


def test_attachment_preserves_dp_group():
    core = EngineCore.__new__(EngineCore)
    core.scheduler = mock.Mock()
    cpu_group = object()

    with (
        mock.patch.object(envs, "ATOM_ENABLE_PREFILL_DELAYER", True),
        mock.patch("atom.model_engine.prefill_delayer.PrefillDelayer") as delayer_cls,
    ):
        core._maybe_attach_prefill_delayer(_config(dp_size=8), cpu_group=cpu_group)

    assert delayer_cls.call_args.kwargs["dp_size"] == 8
    assert delayer_cls.call_args.kwargs["cpu_group"] is cpu_group


def test_disabled_delayer_is_not_attached():
    core = EngineCore.__new__(EngineCore)
    core.scheduler = mock.Mock()

    with (
        mock.patch.object(envs, "ATOM_ENABLE_PREFILL_DELAYER", False),
        mock.patch("atom.model_engine.prefill_delayer.PrefillDelayer") as delayer_cls,
    ):
        core._maybe_attach_prefill_delayer(_config(), cpu_group=None)

    delayer_cls.assert_not_called()
    core.scheduler.set_prefill_delayer.assert_not_called()


def test_deferred_scheduler_is_not_attached():
    core = EngineCore.__new__(EngineCore)
    core.scheduler = None

    with (
        mock.patch.object(envs, "ATOM_ENABLE_PREFILL_DELAYER", True),
        mock.patch("atom.model_engine.prefill_delayer.PrefillDelayer") as delayer_cls,
    ):
        core._maybe_attach_prefill_delayer(_config(), cpu_group=None)

    delayer_cls.assert_not_called()
