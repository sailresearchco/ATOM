# SPDX-License-Identifier: MIT
"""CPU checks for startup request construction and failure propagation."""

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from atom.entrypoints.openai.image_warmup import warm_image_serving
from atom.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest


def warmup_dependencies(endpoint):
    return {
        "model_name": "moonshotai/Kimi-K3",
        "tokenizer": SimpleNamespace(
            encode=lambda *a, **k: [7], decode=lambda ids: "warm" * len(ids)
        ),
        "completions": endpoint,
        "chat_completions": endpoint,
    }


def test_warmup_uses_valid_requests_and_mixed_batches():
    requests = []
    active = 0
    peak = 0

    async def endpoint(payload, request):
        nonlocal active, peak
        requests.append(payload)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return SimpleNamespace(status_code=200)

    asyncio.run(warm_image_serving(**warmup_dependencies(endpoint)))
    texts = [r for r in requests if isinstance(r, CompletionRequest)]
    images = [r for r in requests if isinstance(r, ChatCompletionRequest)]
    assert len(texts) == 8
    assert all(isinstance(r.prompt, str) for r in texts)
    assert len(images) == 20
    assert peak == 8
    sizes = set()
    for r in images:
        for block in r.messages[0].content:
            if block["type"] == "image_url":
                payload = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
                sizes.add(Image.open(io.BytesIO(payload)).size)
    assert sizes == {(64, 64), (256, 256), (800, 600), (1600, 1200), (2048, 1536)}


def test_warmup_propagates_engine_failure():
    async def endpoint(*args):
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(warm_image_serving(**warmup_dependencies(endpoint)))
