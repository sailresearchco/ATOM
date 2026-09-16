# SPDX-License-Identifier: MIT
"""Pre-patchified vision inputs require only a linear projection."""

import pytest
import torch
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.models.kimi_k3_vl import KimiK3PatchEmbed


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("grid", [(2, 2), (3, 4)])
def test_patch_projection_matches_convolution_without_calling_it(
    monkeypatch, bias, grid
):
    torch.manual_seed(7)
    height, width = grid
    layer = KimiK3PatchEmbed(out_dim=32, patch_embed_proj_bias=bias).double()
    pixels = torch.randn(height * width, 3, 14, 14, dtype=torch.float64)
    grids = torch.tensor([[1, height, width]])
    expected = layer.pos_emb(layer.proj(pixels).flatten(1), grids)

    def unexpected_convolution(*args, **kwargs):
        raise AssertionError("Patch projection must not invoke convolution search")

    monkeypatch.setattr(layer.proj, "forward", unexpected_convolution)
    actual = layer(pixels, grids)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
