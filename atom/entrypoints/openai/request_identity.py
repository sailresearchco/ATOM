"""Preserve caller identity across Sail's OpenAI gateway and direct ingress."""

from __future__ import annotations

import uuid


def resolve_request_id(
    body_rid: str | None, header_request_id: str | None, prefix: str
) -> str:
    # Match Sail's SGLang contract: explicit body identity wins; a typed
    # gateway can drop that extension but forwards X-Request-Id unchanged.
    if body_rid is not None:
        return body_rid
    if header_request_id is not None:
        return header_request_id
    return f"{prefix}-{uuid.uuid4().hex}"
