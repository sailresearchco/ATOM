"""Opt-in image serving warmup, completed before ASGI readiness."""

import asyncio
import base64
import io
import logging
import time

from fastapi import Request
from PIL import Image

from .protocol import ChatCompletionRequest, CompletionRequest

logger = logging.getLogger(__name__)


async def warm_image_serving(*, tokenizer, model_name, completions, chat_completions):

    started = time.monotonic()

    async def receive():
        # Internal requests have no client that can disconnect. Cancellation
        # from the request cleanup still stops this receive immediately.
        await asyncio.Event().wait()

    def request():
        return Request({"type": "http", "headers": []}, receive=receive)

    async def run_chat(urls):
        result = await chat_completions(
            ChatCompletionRequest(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image briefly."},
                            *[
                                {"type": "image_url", "image_url": {"url": url}}
                                for url in urls
                            ],
                        ],
                    }
                ],
                max_tokens=2,
                temperature=0,
                ignore_eos=True,
                chat_template_kwargs={"thinking": False},
            ),
            request(),
        )
        if getattr(result, "status_code", 200) >= 400:
            raise RuntimeError(f"Image warmup failed: {result.status_code}")

    # Exercise text kernels at several prefill lengths as well as the vision
    # encoder. Startup's existing synthetic warmup misses small-prefill kernels.
    token = tokenizer.encode("warm", add_special_tokens=False)[0]
    for length in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        await asyncio.wait_for(
            completions(
                CompletionRequest(
                    model=model_name,
                    prompt=tokenizer.decode([token] * length),
                    max_tokens=2,
                    temperature=0,
                    ignore_eos=True,
                ),
                request(),
            ),
            timeout=600,
        )
    urls = []
    for width, height in [(64, 64), (256, 256), (800, 600), (1600, 1200), (2048, 1536)]:
        pixels = Image.new("RGB", (width, height), (180, 40, 60))
        buf = io.BytesIO()
        pixels.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        urls.append(url)
        await asyncio.wait_for(run_chat([url]), timeout=600)
    await asyncio.wait_for(run_chat(urls[:3]), timeout=600)
    for batch in [2, 4, 8]:
        await asyncio.wait_for(
            asyncio.gather(*[run_chat([urls[i % len(urls)]]) for i in range(batch)]),
            timeout=600,
        )
    logger.info("Image serving warmup complete: %.3fs", time.monotonic() - started)
