"""HTTP contract tests; only tokenization and engine execution are mocked."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from atom.entrypoints.openai import api_server as api


OUTPUT = dict(
    text="hello",
    finish_reason="stop",
    num_tokens_input=7,
    num_tokens_output=1,
    num_cached_tokens=3,
)


async def generate(*args, **kwargs):
    await asyncio.sleep(0)
    yield dict(OUTPUT)


async def setup_stream(*args, **kwargs):
    collector = asyncio.Queue()
    collector.put_nowait(
        dict(
            text="hello",
            token_ids=[1],
            finished=True,
            finish_reason="stop",
            num_cached_tokens=3,
        )
    )
    return 42, collector, 7


class SailRequestIdentityHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch.object(api, "model_name", "test-model"),
            patch.object(api, "apply_chat_template", return_value="hello"),
            patch.object(api, "generate_async", generate),
            patch.object(api, "setup_streaming_request", setup_stream),
            patch.object(api, "cleanup_stream"),
            patch.object(api, "cleanup_request"),
        ]
        for p in self.patches:
            p.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://atom.test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for p in reversed(self.patches):
            p.stop()

    async def check_response(self, path, payload, body_id, header_id, stream):
        payload = dict(payload, model="test-model", max_tokens=8, stream=stream)
        if body_id is not None:
            payload["rid"] = body_id
        headers = {"X-Request-Id": header_id} if header_id is not None else {}
        response = await self.client.post(path, json=payload, headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        if stream:
            self.assertIn("data: [DONE]", response.text)
            frames = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
        else:
            frames = [response.json()]
        ids = {frame["id"] for frame in frames}
        expected = body_id if body_id is not None else header_id
        if expected is not None:
            self.assertEqual(ids, {expected})
        else:
            self.assertEqual(len(ids), 1)
            prefix = "chatcmpl-" if "chat" in path else "cmpl-"
            self.assertTrue(next(iter(ids)).startswith(prefix))
        usage = next(frame["usage"] for frame in reversed(frames) if frame.get("usage"))
        self.assertEqual(usage["prompt_tokens"], 7)
        self.assertEqual(usage["completion_tokens"], 1)
        if "chat" in path:
            self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 3)
        return ids

    async def test_http_identity_and_usage_matrix(self):
        for path, payload in [
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/completions", {"prompt": "hi"}),
        ]:
            for stream in (False, True):
                for body_id, header_id in [
                    ("resp_body:2", "resp_header:1"),
                    (None, "resp_gateway:3"),
                    (None, None),
                ]:
                    with self.subTest(
                        path=path, stream=stream, body=body_id, header=header_id
                    ):
                        await self.check_response(
                            path, payload, body_id, header_id, stream
                        )

    async def test_attempt_ids_stay_distinct_under_concurrency(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        results = await asyncio.gather(
            *[
                self.check_response(
                    "/v1/chat/completions", payload, f"resp_retry:{i}", None, False
                )
                for i in range(1, 5)
            ]
        )
        self.assertEqual(len(set.union(*results)), 4)


if __name__ == "__main__":
    unittest.main()
