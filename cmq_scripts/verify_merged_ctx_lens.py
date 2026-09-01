"""CPU validation of the merged dcp_local_context_lens (next_n dispatch).

The repo's own tests for this function skip without a GPU / triton, so the
next_n=1 + next_n>1 unification currently has no automated cover. This exercises
it against brute-force token ownership on CPU.
"""

import ast
import pathlib

import numpy as np
import torch


def load_fn():
    """Extract the function, dropping any `# region agent log` block.

    The debug block writes to a hardcoded absolute path, which raises OSError on
    a read-only filesystem, so it has to go before the logic can be exercised.
    """
    lines = pathlib.Path("atom/model_ops/dcp_ops.py").read_text().splitlines()
    kept, skipping, had_region = [], False, False
    for line in lines:
        if line.strip() == "# region agent log":
            skipping = had_region = True
        elif line.strip() == "# endregion":
            skipping = False
        elif not skipping:
            kept.append(line)
    for node in ast.parse("\n".join(kept)).body:
        if isinstance(node, ast.FunctionDef) and node.name == "dcp_local_context_lens":
            ns = {"torch": torch}
            exec(ast.unparse(node), ns)  # noqa: S102
            return ns["dcp_local_context_lens"], had_region
    raise RuntimeError("dcp_local_context_lens not found")


class FakeMeta:
    def __init__(self, ctx, **published):
        self.context_lens = torch.tensor(ctx, dtype=torch.int32)
        for k, v in published.items():
            setattr(self, k, v)


def brute(prefix_len, rank, world, interleave):
    return sum(1 for g in range(prefix_len) if (g // interleave) % world == rank)


def main():
    fn, has_debug_region = load_fn()
    if has_debug_region:
        print("NOTE: stripped a `# region agent log` block from the function; as")
        print("      shipped it raises OSError where that path is not writable.\n")

    failures = 0
    cases = [
        # (world, interleave, next_n, ctx)
        (4, 1, 1, [10, 11, 7]),
        (4, 1, 4, [10, 11, 7]),
        (4, 1, 4, [948, 949, 950, 951]),
        (4, 1, 2, [0, 1, 3, 4]),  # padded cudagraph tail: ctx < next_n
        (8, 1, 4, [1000, 37]),
        (4, 16, 4, [1000, 37, 129]),  # interleave > 1
        (2, 64, 3, [500, 64, 65]),
    ]

    for world, S, next_n, ctx in cases:
        for rank in range(world):
            got = fn(FakeMeta(ctx), rank, world, S, len(ctx), next_n).numpy()
            want = np.array(
                [
                    brute(max(0, c - next_n + 1 + j), rank, world, S)
                    for c in ctx
                    for j in range(next_n)
                ],
                dtype=np.int32,
            )
            ok = got.shape == want.shape and (got == want).all()
            failures += not ok
            if not ok:
                print(
                    f"FAIL W={world} S={S} N={next_n} rank={rank} ctx={ctx}\n"
                    f"  got ={got.tolist()}\n  want={want.tolist()}"
                )

    print(f"formula vs brute force: {len(cases) * 4} groups, {failures} failures")

    # next_n=1 must be bit-identical to the pre-refactor per-request path.
    def old_per_request(seq_lens, world, rank, S):
        full = seq_lens // (S * world)
        base = full * S
        rem = np.clip(seq_lens - base * world - rank * S, 0, S)
        return base + rem

    mismatch = 0
    for world, S in [(4, 1), (8, 1), (4, 16), (2, 64)]:
        ctx = [0, 1, 7, 63, 64, 65, 255, 948, 1000]
        for rank in range(world):
            got = fn(FakeMeta(ctx), rank, world, S, len(ctx)).numpy()
            want = old_per_request(np.array(ctx), world, rank, S)
            mismatch += not (got == want).all()
    print(f"next_n=1 vs pre-refactor formula: {mismatch} mismatches")

    # Unified published-buffer dispatch: the shape guard selects the current
    # query width and rejects stale views from another graph.
    bs, next_n = 3, 4
    ctx = [10, 11, 7]
    tagged = torch.arange(100, 100 + bs * next_n, dtype=torch.int32)
    meta = FakeMeta(ctx, dcp_local_context_lens=tagged)
    out = fn(meta, 0, 4, 1, bs, next_n)
    print(
        f"next_n>1 uses unified buffer (zero-copy):   {out.data_ptr() == tagged.data_ptr()}"
    )

    meta = FakeMeta(ctx, dcp_local_context_lens=torch.arange(bs, dtype=torch.int32))
    out = fn(meta, 0, 4, 1, bs, next_n)
    print(f"next_n>1 ignores a stale [B] view:           {out.shape[0] == bs * next_n}")

    per_req = torch.tensor([3, 3, 2], dtype=torch.int32)
    meta = FakeMeta(ctx, dcp_local_context_lens=per_req)
    out = fn(meta, 0, 4, 1, bs)
    print(
        f"next_n=1 uses the [B] buffer (zero-copy):   {out.data_ptr() == per_req.data_ptr()}"
    )

    # The old signature took num_rows and returned the FULL context_lens in the
    # fallback; running_bs > scheduled_bs then produced a too-long tensor.
    meta = FakeMeta([10, 11, 7, 0, 0, 0])  # running_bs=6, scheduled_bs=3
    out = fn(meta, 0, 4, 1, 3, 4)
    print(
        f"fallback honours batch_size (rows={out.shape[0]}, want 12):  {out.shape[0] == 12}"
    )

    return failures + mismatch


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
