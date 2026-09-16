import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.reasoning_dialects import DIALECTS


def channel(dialect, starts_open):
    # Multi-token markers share framing IDs as K3's real markers do. Content
    # IDs need not encode to a single character, and are never re-tokenized.
    markers = dict.fromkeys(
        (dialect.output_open_marker, *dialect.end_markers, *dialect.content_framing)
    )
    encoded = {
        marker: (90, index + 10, 91) for index, marker in enumerate(markers) if marker
    }
    return ReasoningChannel(dialect, starts_open, encoded.__getitem__), encoded


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("starts_open", [False, True])
def test_generated_ids_and_boundaries_are_counted_across_every_chunk_split(
    dialect, starts_open
):
    reasoning, markers = channel(dialect, starts_open)
    opener = [] if starts_open else list(markers[dialect.output_open_marker])
    for end in dialect.end_markers:
        ids = [*opener, 100, 101, *markers[end], 102, 103]
        expected = len(ids) - 2
        for split in range(len(ids) + 1):
            counter = reasoning.token_counter()
            counter.update(ids[:split])
            counter.update(ids[split:])
            assert counter.count == expected
        counter = reasoning.token_counter()
        for token_id in ids:
            counter.update([token_id])
        assert counter.count == expected


@pytest.mark.parametrize("dialect", DIALECTS)
def test_truncation_counts_generated_reasoning_but_not_prompt_tokens(dialect):
    reasoning, markers = channel(dialect, True)
    close = markers[dialect.think_end_marker]
    for ids in ([], [100, 101], [100, *close[:-1]]):
        assert reasoning.usage_details([ids]) == {
            "completion_tokens_details": {"reasoning_tokens": len(ids)}
        }


@pytest.mark.parametrize("dialect", DIALECTS)
def test_no_reasoning_and_immediate_close(dialect):
    reasoning, markers = channel(dialect, False)
    for ids in ([100, 101], [*markers[dialect.think_end_marker], 100, 101]):
        counter = reasoning.token_counter()
        counter.update(ids)
        assert counter.count == 0


def test_fanout_counts_each_sibling_independently():
    reasoning, markers = channel(DIALECTS[0], True)
    close = markers[DIALECTS[0].think_end_marker]
    assert reasoning.usage_details([[100, *close, 102], [101, 102]]) == {
        "completion_tokens_details": {"reasoning_tokens": 6}
    }


def test_unknown_tokenizer_does_not_invent_a_zero_subtotal():
    assert ReasoningChannel(starts_open=True).usage_details([[100, 101]]) == {}
    reasoning, _ = channel(DIALECTS[0], True)
    assert reasoning.usage_details([None]) == {}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("siblings", [1, 2])
def test_chat_usage_matches_across_delivery_modes(stream, siblings, monkeypatch):
    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.serving_chat import (
        build_chat_response,
        build_chat_response_multi,
        stream_chat_response,
        stream_chat_response_fanout,
    )

    dialect = DIALECTS[0]
    close = [90, 10, 91]
    monkeypatch.setattr(api_server, "reasoning_dialect", dialect)
    monkeypatch.setattr(
        api_server,
        "tokenizer",
        SimpleNamespace(
            encode=lambda marker, *, add_special_tokens: (
                close if marker == dialect.think_end_marker else [92, 11, 93]
            )
        ),
    )
    reasoning = api_server.reasoning_channel(True, template_kwargs={})
    outputs = [
        {
            "text": "trace<|close|>think<|sep|>answer",
            "token_ids": [100, *close, 101],
            "num_tokens_input": 7,
            "num_tokens_output": 5,
            "finish_reason": "eos",
        },
        {
            "text": "more trace",
            "token_ids": [102, 103],
            "num_tokens_input": 7,
            "num_tokens_output": 2,
            "finish_reason": "max_tokens",
        },
    ][:siblings]
    if not stream:
        if siblings == 1:
            result = build_chat_response(
                "req", "model", outputs[0]["text"], outputs[0], reasoning=reasoning
            )
        else:
            result = build_chat_response_multi(
                "req", "model", outputs, reasoning=reasoning
            )
        usage = result.usage
    else:
        # Interleave siblings and split the three-ID closer across chunks.
        # Text length intentionally differs from token count.
        pending = []
        for offset in range(5):
            for index, out in enumerate(outputs):
                if offset >= len(out["token_ids"]):
                    continue
                chunk = {
                    "text": out["text"] if offset == 0 else "",
                    "token_ids": [out["token_ids"][offset]],
                    "finished": offset == len(out["token_ids"]) - 1,
                    "finish_reason": (
                        out["finish_reason"]
                        if offset == len(out["token_ids"]) - 1
                        else None
                    ),
                }
                pending.append((index, chunk) if siblings > 1 else chunk)

        class Collector:
            async def get(self):
                return pending.pop(0)

        async def collect():
            generate = (
                stream_chat_response if siblings == 1 else stream_chat_response_fanout
            )
            gen = generate(
                "req",
                "model",
                Collector(),
                0 if siblings == 1 else [0, 1],
                7,
                lambda *args, **kwargs: None,
                lambda *args: None,
                reasoning=reasoning,
            )
            return "".join([frame async for frame in gen])

        frames = asyncio.run(collect())
        usage = next(
            json.loads(line[6:])["usage"]
            for line in frames.splitlines()
            if line.startswith("data: {") and '"usage"' in line
        )
    assert usage["completion_tokens"] == (5 if siblings == 1 else 7)
    assert usage["completion_tokens_details"]["reasoning_tokens"] == (
        4 if siblings == 1 else 6
    )


@pytest.mark.parametrize("siblings", [1, 2])
@pytest.mark.parametrize("max_items", [1, 16])
def test_standalone_stream_usage_counts_split_markers_and_fanout(siblings, max_items):
    from atom.entrypoints.atomesh.atom_standalone_service import (
        AtomStandaloneService,
        ChatCompletionStreamState,
    )

    dialect = DIALECTS[0]
    close = [90, 10, 91]
    tokenizer = SimpleNamespace(
        encode=lambda marker, **kwargs: (
            close if marker == dialect.think_end_marker else [92, 11, 93]
        )
    )
    service = SimpleNamespace(
        tokenizer=tokenizer,
        reasoning_dialect=dialect,
        model_starts_in_reasoning=True,
        reasoning_toggle=None,
    )
    reasoning = AtomStandaloneService._reasoning_channel(service, "prompt", {})
    events = queue.Queue()
    outputs = [[100, *close, 101], [102, 103]][:siblings]
    for offset in range(5):
        for index, ids in enumerate(outputs):
            if offset < len(ids):
                events.put(
                    {
                        "index": index,
                        "text": "trace" if offset == 0 else "",
                        "token_ids": [ids[offset]],
                        "finished": offset == len(ids) - 1,
                    }
                )
    events.put({"done": True})
    state = ChatCompletionStreamState(
        "req", "model", "prompt", tokenizer, events, siblings, reasoning=reasoning
    )
    frames = []
    for _ in range(64):
        frames.extend(state.drain(max_items=max_items, timeout=0.0))
        if state.completed:
            break
    assert state.completed
    usage = next(
        payload["usage"]
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: {")
        for payload in [json.loads(line[6:])]
        if "usage" in payload
    )
    assert usage["completion_tokens"] == sum(map(len, outputs))
    assert usage["completion_tokens_details"]["reasoning_tokens"] == (
        4 if siblings == 1 else 6
    )
