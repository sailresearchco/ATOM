"""Image decoding failures must be client errors before engine dispatch."""

import asyncio
import base64
import io
import urllib.error
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from atom.entrypoints.openai import api_server as api


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), (11, 22, 33, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def truncated_image():
    raw = image_bytes()
    # Retain the PNG header so Image.open succeeds; pixel decoding must fail.
    return raw[: raw.index(b"IDAT") + 5]


@pytest.mark.parametrize("source", ["data", "http", "file"])
@pytest.mark.parametrize("content", ["valid", "unidentified", "truncated"])
def test_image_sources_validate_decoded_pixels(monkeypatch, tmp_path, source, content):
    raw = {
        "valid": image_bytes(),
        "unidentified": b"not-an-image",
        "truncated": truncated_image(),
    }[content]
    if source == "data":
        url = data_url(raw)
    elif source == "http":
        url = "https://images.test/image.png"
        monkeypatch.setattr(
            api.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(raw)
        )
    else:
        path = tmp_path / "image.png"
        path.write_bytes(raw)
        url = path.as_uri()
    if content == "valid":
        image = api._load_image_from_url(url)
        assert image.mode == "RGB"
        assert image.size == (16, 16)
        assert image.getpixel((0, 0)) == (11, 22, 33)
    else:
        with pytest.raises(ValueError, match="Invalid or unsupported image data"):
            api._load_image_from_url(url)


def test_oversized_image_is_a_validation_error(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ValueError, match="Invalid or unsupported image data"):
        api._load_image_from_url(data_url(image_bytes()))


def test_acquisition_failures_keep_their_original_type(monkeypatch, tmp_path):
    error = urllib.error.URLError("unavailable")
    monkeypatch.setattr(api.urllib.request, "urlopen", Mock(side_effect=error))
    with pytest.raises(urllib.error.URLError) as caught:
        api._load_image_from_url("https://images.test/image.png")
    assert caught.value is error
    with pytest.raises(FileNotFoundError):
        api._load_image_from_url(str(tmp_path / "missing.png"))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", ["unidentified", "truncated", "bad-base64"])
def test_invalid_image_returns_http_400_before_dispatch(monkeypatch, stream, content):
    monkeypatch.setattr(api, "model_name", "test-model")
    monkeypatch.setattr(api, "_get_multimodal_processor", Mock())
    dispatch = Mock(side_effect=AssertionError("invalid image reached engine"))
    monkeypatch.setattr(api, "build_multimodal_inputs", dispatch)
    url = (
        "data:image/png;base64,%%%"
        if content == "bad-base64"
        else data_url(truncated_image() if content == "truncated" else b"not-an-image")
    )

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://atom.test"
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "stream": stream,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}}
                            ],
                        }
                    ],
                },
            )

    response = asyncio.run(request())
    assert response.status_code == 400, response.text
    assert "image_url" in response.json()["detail"]
    dispatch.assert_not_called()
