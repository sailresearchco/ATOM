"""Print the three DCP local-context-length quantities for one small batch.

Every number is cross-checked against brute-force enumeration of token
ownership, so the table can be trusted as documentation.
"""

import numpy as np

W = 4  # dcp_world_size
S = 1  # cp_kv_cache_interleave_size
N = 4  # next_n = mtp_k + 1, the verify window
CTX = np.array([10, 11, 7], dtype=np.int64)  # context_lens, bs=3
B = len(CTX)


def owner(g):
    return (g // S) % W


def brute_local_len(prefix_len, rank):
    """Tokens this rank owns among global positions [0, prefix_len)."""
    return sum(1 for g in range(prefix_len) if owner(g) == rank)


def get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, interleave=1):
    """Verbatim copy of dcp_ops.get_dcp_local_seq_lens."""
    full_chunks = seq_lens // (interleave * dcp_size)
    base = full_chunks * interleave
    remainder_total = seq_lens - base * dcp_size
    remainder = np.clip(remainder_total - dcp_rank * interleave, 0, interleave)
    return base + remainder


print(f"W={W} S={S} N={N} ctx={CTX.tolist()}\n")

print("global token ownership (rank that stores each token):")
for b in range(B):
    row = " ".join(f"{owner(g)}" for g in range(CTX[b]))
    print(f"  req {b} (ctx={CTX[b]:2d}): {row}")

print("\nquery-token global positions in the verify window:")
positions = np.tile(np.arange(N), B) + np.repeat(CTX - N, N)
for b in range(B):
    print(f"  req {b}: {positions[b * N:(b + 1) * N].tolist()}")

# visible[b, j] = how many global tokens query token (b, j) may attend to.
visible = np.clip(CTX[:, None] - N + 1 + np.arange(N)[None, :], 0, None)
print(f"\nvisible global prefix length per (req, j):\n{visible}")

for rank in range(W):
    print(f"\n{'=' * 62}\nrank {rank}")

    per_request = get_dcp_local_seq_lens(CTX, W, rank, S)
    assert (per_request == [brute_local_len(c, rank) for c in CTX]).all()
    print(f"  dcp_local_context_lens (N=1)    [B]     = {per_request.tolist()}")

    local = get_dcp_local_seq_lens(visible, W, rank, S)
    for b in range(B):
        for j in range(N):
            assert local[b, j] == brute_local_len(visible[b, j], rank)

    per_token = local.reshape(-1)
    print(f"  dcp_local_context_lens (N={N})    [B*N]   = {per_token.tolist()}")

    # Structural anchor: the last verify position sees the whole context.
    assert (local[:, N - 1] == per_request).all()

    naive = np.clip(per_request[:, None] - N + 1 + np.arange(N)[None, :], 0, None)
    print(f"  (wrong) per_request - N + 1 + j         = {naive.reshape(-1).tolist()}")
    print(f"  per-row shortfall of the wrong form     = {(local - naive).tolist()}")

print(f"\n{'=' * 62}")
print("historical rank-1 comparison at global context 948:")
ctx948 = np.array([948])
vis948 = np.clip(ctx948[:, None] - N + 1 + np.arange(N)[None, :], 0, None)
print(f"  exactLocalByQuery = {get_dcp_local_seq_lens(vis948, W, 1, S).tolist()}")
print(
    f"  oldKernelImplied  = "
    f"{np.clip(get_dcp_local_seq_lens(ctx948, W, 1, S)[:, None] - N + 1 + np.arange(N)[None, :], 0, None).tolist()}"
)
