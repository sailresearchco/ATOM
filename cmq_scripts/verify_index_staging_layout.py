"""Byte-level check of gather_dcp_preshuffled_index_pages against ATOM's real
DSA index-cache layout.

ATOM always writes the index cache with ``preshuffle=True`` at
``kv_cache_block_size`` granularity (deepseek_v2.py re-views the cache as
``[-1, index_block_size, aligned_index_dim]`` before every write and read), so a
scheduler block is one MFMA-tiled key plane followed by an fp32 scale plane.
``index_cache[layer].shape[1]`` is the *MLA latent* page size and is unrelated to
that granularity.

Runs on CPU; no GPU or aiter needed.
"""

import numpy as np
import torch

B = 16  # kv_cache_block_size / scheduler block, tokens
HD = 128  # index_head_dim
ALIGNED = 144  # ((HD + 4 + 15) // 16) * 16
MLA_PAGE = 1  # ATOM_MLA_PAGE_SIZE, so index_cache.shape[1] == 1
BLOCK_RATIO = B // MLA_PAGE
W, R = 4, 1  # dcp world size, this rank
NBLK = 8  # scheduler blocks held by the producer
COL_TILES = HD // 16


def _load_gather_fn():
    """Exec just the gather function; aiter_mla's imports need a GPU box."""
    import ast
    import pathlib

    src = pathlib.Path("atom/model_ops/attentions/aiter_mla.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "gather_dcp_preshuffled_index_pages"
        ):
            ns = {"torch": torch, "cdiv": lambda a, b: (a + b - 1) // b}
            exec(ast.unparse(node), ns)  # noqa: S102
            return ns["gather_dcp_preshuffled_index_pages"]
    raise RuntimeError("gather_dcp_preshuffled_index_pages not found")


def build_preshuffled_source(nblk):
    """One scheduler block = tiled key plane, then fp32 scales, then padding."""
    page = np.zeros((nblk, B * ALIGNED), dtype=np.uint8)
    keys = np.zeros((nblk, COL_TILES, B, 16), dtype=np.uint8)
    scales = np.zeros((nblk, B), dtype=np.float32)
    for b in range(nblk):
        for t in range(B):
            g = b * B + t
            for c in range(HD):
                # Byte c of global token g, tagged so a misread is visible.
                keys[b, c // 16, t, c % 16] = (g * 7 + c * 3) % 251 + 1
            scales[b, t] = 1000.0 + g
    page[:, : B * HD] = keys.reshape(nblk, B * HD)
    page[:, B * HD : B * HD + B * 4] = scales.view(np.uint8).reshape(nblk, B * 4)
    return page, keys, scales


def decode_staged(staged_bytes, dst_pages):
    keys = staged_bytes[:dst_pages, : B * HD].reshape(dst_pages, COL_TILES, B, 16)
    scales = (
        staged_bytes[:dst_pages, B * HD : B * HD + B * 4]
        .copy()
        .view(np.float32)
        .reshape(dst_pages, B)
    )
    return keys, scales


def main():
    gather_dcp_preshuffled_index_pages = _load_gather_fn()

    page, src_keys, src_scales = build_preshuffled_source(NBLK)
    src_block_ids = list(range(NBLK))
    dst_pages = -(-NBLK // W)

    # As allocated: [num_physical_blocks, MLA_PAGE, ALIGNED], one row per
    # MLA latent page. int8 stands in for fp8 (same itemsize).
    source_as_allocated = torch.from_numpy(
        page.reshape(NBLK * BLOCK_RATIO, MLA_PAGE, ALIGNED).copy()
    ).view(torch.int8)
    # As every real reader/writer views it: [num_scheduler_blocks, B, ALIGNED].
    source_reviewed = torch.from_numpy(page.reshape(NBLK, B, ALIGNED).copy()).view(
        torch.int8
    )

    def run(source, page_size, block_ratio, label):
        staging = torch.zeros(64, B * ALIGNED, dtype=torch.uint8)
        n = gather_dcp_preshuffled_index_pages(
            source, staging, src_block_ids, W, R, HD, B, block_ratio
        )
        got_keys, got_scales = decode_staged(staging.numpy(), n)
        bad_keys = bad_scales = 0
        for i in range(n * B):
            g = i * W + R
            b, t = divmod(i, B)
            if g // B >= NBLK:
                continue
            want_k = src_keys[g // B, :, g % B, :]
            want_s = src_scales[g // B, g % B]
            if not np.array_equal(got_keys[b, :, t, :], want_k):
                bad_keys += 1
            if got_scales[b, t] != want_s:
                bad_scales += 1
        total = min(n * B, (NBLK * B - R + W - 1) // W)
        print(
            f"{label}: pages={n} checked={total} "
            f"wrong_key_tokens={bad_keys} wrong_scale_tokens={bad_scales}"
        )
        return bad_keys, bad_scales

    print(f"config: B={B} HD={HD} ALIGNED={ALIGNED} W={W} rank={R} src_blocks={NBLK}")
    print(f"expected dst_pages={dst_pages}\n")
    run(source_as_allocated, MLA_PAGE, BLOCK_RATIO, "current call (shape[1]==1)")
    run(source_reviewed, B, 1, "re-viewed at kv_cache_block_size")

    # Tile-order-independent proof. The taken branch reads token t of a
    # scheduler block as bytes [t*ALIGNED, t*ALIGNED+HD) of that block, but the
    # block's key plane is only B*HD bytes and its scale plane ends at
    # B*HD + B*4; past that is padding, which is always zero.
    print(
        f"\nkey plane ends at {B * HD}, scale plane at {B * HD + B * 4}, "
        f"page is {B * ALIGNED} bytes"
    )
    for t in (0, 13, 14, 15):
        lo, hi = t * ALIGNED, t * ALIGNED + HD
        where = (
            "inside key plane"
            if hi <= B * HD
            else (
                "pure padding (reads zeros)"
                if lo >= B * HD + B * 4
                else "straddles key/scale planes"
            )
        )
        print(f"  token {t:2d} -> source bytes [{lo}, {hi}) : {where}")


if __name__ == "__main__":
    main()
