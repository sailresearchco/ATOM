#!/usr/bin/env python3
"""Dependency-free liveness/readiness probe for the ATOM K3 service."""

from __future__ import annotations

import argparse
import json
import urllib.request


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    get_json(f"{base}/health")
    models = get_json(f"{base}/v1/models")
    if not isinstance(models, dict) or not any(
        item.get("id") == args.model for item in models.get("data", [])
    ):
        raise RuntimeError(f"served model {args.model!r} is absent from /v1/models")


if __name__ == "__main__":
    main()
