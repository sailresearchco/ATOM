#!/usr/bin/env python3
"""Exercise K3 generation, metrics, and the verified cache-reset path."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request_json(url: str, payload: object | None = None, timeout: int = 180) -> object:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model", default="moonshotai/Kimi-K3")
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    models = request_json(f"{base}/v1/models")
    assert any(item.get("id") == args.model for item in models.get("data", []))

    response = request_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with the single word READY."}],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        },
    )
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise RuntimeError(f"invalid chat completion: {response!r}")
    message = choices[0]["message"]
    if not (message.get("content") or message.get("reasoning_content")):
        raise RuntimeError(f"empty chat completion: {response!r}")

    cache = request_json(f"{base}/debug/cache_stats")
    if not isinstance(cache, dict):
        raise RuntimeError(f"invalid cache stats: {cache!r}")
    if not args.skip_reset:
        reset = request_json(f"{base}/reset_prefix_cache", {})
        if reset.get("status") != "success" or not reset.get("cleared"):
            raise RuntimeError(f"cache reset did not verify: {reset!r}")

    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as metrics_response:
        metrics = metrics_response.read().decode()
    required = ("atom:metrics_snapshot_available", "atom:requests_finished")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(f"missing Prometheus metrics: {missing}")

    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                "finish_reason": choices[0].get("finish_reason"),
                "content": message.get("content"),
                "reasoning_content": message.get("reasoning_content"),
                "cache_reset": not args.skip_reset,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
