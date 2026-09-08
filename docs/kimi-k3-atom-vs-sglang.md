# Kimi-K3 on MI355X: ATOM versus SGLang

This document records Sail's September 2026 production-trace ablation of ATOM
against the Kimi-K3 SGLang deployment on eight-GPU AMD MI355X workers. The
decision result is the replicated six-node, one-hour replay. Smaller single-node
runs are included only where they explain a configuration choice or qualify the
result.

## Decision

Use ATOM with TP8, DCP8, DSpark-4, four continuous decode steps, prefix caching,
an 8,192-token prefill budget, a 32-request scheduler cap, and the DP1 prefill
delayer disabled.

On the replicated production replay, ATOM preserved SGLang-level decode speed
while improving request latency, TTFT, queueing, prefill throughput, power, and
allocated VRAM. The important regressions were p99 TPOT and final drain time.
DCP8 also supplied substantially more token-admission headroom, although this
campaign did not drive either engine to its maximum sustainable TPM and did not
measure the concurrency ceiling directly.

This was an EP1 comparison, not EP8:

- SGLang: TP8, EP1, no DCP, DSpark-3.
- ATOM: TP8, EP1, DCP8, DSpark-4.

## Primary result: six-node one-hour replay

Three otherwise idle MI355X workers ran each engine simultaneously from
2026-09-04 23:18 UTC through 2026-09-05 00:19 UTC. Every worker replayed the
same 1,984-request, one-hour workload. It was derived by repeating the
124-request dispatchable production cohort 16 times, with a unique cycle
marker to preserve within-cycle multi-turn prefix reuse without permitting
identical full-cycle cache hits.

Each cell completed all 1,984 raw requests and all 1,970 scored requests,
generated 2,031,616 raw and 2,017,280 scored tokens, and recorded zero request
failures, preemptions, container restarts, GPU faults, host OOMs, or collective
errors. Values below are arithmetic means across the three independent workers
per engine. Lower is better for latency, queueing, power, VRAM, and drain.

| Metric | SGLang TP8, DSpark-3 | ATOM TP8+DCP8, DSpark-4 | ATOM delta |
|---|---:|---:|---:|
| Per-request decode | 37.97 tok/s | 38.46 tok/s | +1.30% |
| Request latency | 32.06 s | 29.76 s | -7.15% |
| TTFT mean | 5.114 s | 3.164 s | -38.13% |
| TTFT p50 / p95 / p99 | 1.700 / 23.058 / 26.495 s | 0.750 / 17.424 / 19.611 s | -55.90% / -24.44% / -25.98% |
| TPOT mean / p95 / p99 | 26.34 / 38.57 / 43.37 ms | 26.00 / 38.20 / 48.14 ms | -1.28% / -0.95% / +11.01% |
| Prefill, all / noncached | 2,861.66 / 2,411.03 tok/s | 3,766.85 / 2,897.40 tok/s | +31.63% / +20.17% |
| Prefix-cache hit rate | 12.12% | 18.80% | +6.67 pp |
| Queue mean / peak | 2.26 / 23.33 requests | 1.39 / 19.33 requests | -38.59% / -17.14% |
| Final drain | 13.50 s | 17.21 s | +27.51% |
| Eight-GPU power | 6,652.8 W | 6,315.9 W | -5.06% |
| Peak allocated VRAM | 90.29% | 87.13% | -3.16 pp |
| Accepted tokens per forward | 2.4045 | 2.6979 | +12.20% |

The fixed arrival schedule and fixed 1,024-token output length pin successful
wall-clock output throughput at 562.62 tok/s in every cell. Per-request decode
speed, latency, queueing, and prefill are therefore the useful discriminators;
equal aggregate output throughput is not evidence that the stacks have equal
saturation capacity.

### Cache and TTFT interpretation

The lower ATOM TTFT is real for the compared production recipes, but it is not
an engine-only prefill claim. ATOM reused 6.67 percentage points more prompt
tokens and processed 8.50% fewer fresh prompt tokens. This was not leakage from
an earlier run: each server started measurement after a verified cache reset,
and every repeated trace cycle carried a unique marker. Even after removing
cached work, ATOM's reported prefill throughput was 20.17% higher, and that
advantage replicated closely on all three ATOM workers.

The campaign therefore supports an admission/prefill advantage but does not
attribute the full 38.13% mean-TTFT reduction to raw engine execution. Future
engine-only studies should pair cache-off cells or force an identical cache-hit
schedule in both engines.

### Fairness contract

The replay held these controls common:

- identical MI355X eight-GPU hardware and trace arrival timing;
- the same Kimi-K3 target revision and tokenizer;
- a dynamic 32-request scheduler cap (not a fixed batch size of 32);
- an 8,192-token chunked-prefill/batched-token budget;
- four continuous decode steps, FP8 attention KV, and prefix caching;
- 32 warmup requests followed by a synchronous, verified cache reset;
- exactly 1,024 generated tokens per request with EOS ignored;
- the same Sail-Compass revision and client-side measurement path.

All nine cross-arm pairings passed Sail-Compass's workload, hardware,
provenance, and health comparison without a failure or warning after only the
engine-specific fields were declared as expected differences. Realized scored
input work differed by 0.9783%, within the 1% gate, and output work was exact.

## Corroborating three-pair campaign

Before the one-hour soak, one MI355X worker ran three counterbalanced pairs in
SGLang/ATOM, ATOM/SGLang, and SGLang/ATOM order. Each cell scored 110 requests
from the same production cohort at 8x recorded arrival speed after 32 warmups
and a verified cache reset.

| Median metric | SGLang | ATOM | Result |
|---|---:|---:|---:|
| Decode | 40.04 tok/s | 39.97 tok/s | -0.17% paired median |
| Request latency | 31.58 s | 28.91 s | -7.93% paired median |
| Mean TTFT | 6.204 s | 3.336 s | -43.39% paired median |
| p95 TTFT | 24.637 s | 17.809 s | -25.25% paired median |
| Mean TPOT | 24.98 ms | 25.02 ms | directionally flat |
| Mean queue | 2.60 requests | 1.26 requests | lower with ATOM |
| Peak KV gauge | 40.00% | 19.61% | lower with ATOM |
| Eight-GPU power | 6,583 W | 6,271 W | -4.74% |

Paired ATOM decode deltas ranged from -2.16% to +2.83%, so the correct read is
directionally flat decode, not a small deterministic win. Cache rates were
close in these pairs (13.55% SGLang versus 13.90% ATOM by medians), which is
why they are useful corroboration for the larger soak's TTFT result.

## Why the selected ATOM controls look this way

### DCP8 and token concurrency

Kimi-K3 has 24 MLA layers and 69 KDA layers. DCP shards the MLA paged KV and
attention work; the recurrent KDA state remains replicated. DCP can therefore
greatly reduce per-rank MLA KV pressure and increase concurrent-token headroom
without guaranteeing a per-token speedup at sparse load.

In the earlier single-worker factorial, DCP8 alone was directionally flat on
decode (-0.32%) but reduced ATOM's reported peak KV pressure from 2.20% to
0.36% at the same peak of 18 running requests. After the ablation, the new
admission endpoint reported 1,527,168 physical KV tokens and 12,217,344 DCP8
virtual tokens on the same topology. Those figures validate the accounting and
headroom mechanism, not a maximum-load throughput claim.

### DSpark proposal depth

Sail production traffic normally accepts roughly 2.3--2.5 tokens per target
forward, making very deep proposals wasteful. The final SGLang recipe used
DSpark-3, while the combined ATOM+DCP recipe used DSpark-4. In the replicated
soak, they emitted 2.4045 and 2.6979 accepted tokens per forward respectively.

An earlier ATOM depth sweep found that DCP8+DSpark-3 had better decode tails and
more memory margin, DSpark-5 had the highest mean decode speed, and DSpark-4 was
the stable middle point used for production validation. Proposal depth and DCP
are not independently additive: speculative verification changes the DCP
query shape and communication path, so their combination must be measured as a
recipe.

### Continuous decode

ATOM's DP1 continuous-decode implementation was tested with a ceiling of four
steps. A precursor high-TPM run with the prefill delayer enabled improved
decode rate by 6.22% and mean TPOT by 5.83% when moving from one to four steps.
The engine yields early for new requests, utility work, metrics deadlines, or
completed decode work, so four is a ceiling rather than a guaranteed burst
length.

### Prefill delayer

This branch wires ATOM's prefill coalescer into DP1 as well as data-parallel
serving, but it is disabled in the selected K3 profile. At four continuous
decode steps, disabling it in the precursor high-TPM run improved decode by
9.46%, mean latency by 17.52%, mean TTFT by 22.95%, and final drain by 17.05%.
The current DP1 defaults over-coalesced this already-bursty workload.

### Prefix caching

ATOM prefix caching works with K3 TP8+DCP8+DSpark-4. A dedicated activation
smoke crossed the 8,192-token KDA checkpoint interval, returned 30/30 consistent
arithmetic answers, and recorded 172,032 admitted cached tokens plus two KDA
checkpoints. During the final soak, an idle reset on each ATOM worker cleared
1,811--1,814 prefix blocks and 359 KDA checkpoints to zero.

A short shared prefix that does not cross the checkpoint interval can produce
zero reusable KDA checkpoints; that is expected and must not be interpreted as
"ATOM cannot cache." Cache counters and before/after reset receipts should be
part of every future cache-sensitive comparison.

## K3 serving feature matrix at the measured revisions

This table describes the source revisions used in the campaign. "Stronger"
means broader or more mature support in the measured tree, not that the other
engine can never implement the feature.

| Shared | ATOM-only or stronger here | SGLang-only or stronger here |
|---|---|---|
| Kimi-K3 on MI355X; TP8; FP8 MLA KV; chunked prefill; prefix caching with KDA-aware state; DSpark and ReplaySSM; continuous decode; streaming Chat Completions; K3 reasoning/tool parsing; health and metrics endpoints | Measured TP8+DCP8 and DCP8+DSpark on one eight-GPU worker; DCP-aware virtual token admission; synchronous prefix/KDA cache reset; branch implementation of DP1 prefill coalescing; compact production wrapper | Broader OpenAI serving surface; hard grammar-constrained decoding; more mature hybrid radix and HiCache/offload tiers; K3-complete PD/EPD state transfer; broader production operational history |

Important boundaries:

- Both measured stacks were EP1. This campaign says nothing about an EP8
  comparison or an EP+DCP interaction.
- ATOM accepted the production Chat Completions workload, but its validator
  rejected seven requests whose tool names fell outside
  `^\w[\w.\-]*\Z`. Those requests were removed byte-for-byte from both arms;
  names were not rewritten.
- ATOM accepts `response_format` and `tool_choice`, but this is not equivalent
  to SGLang's hard grammar-constrained decoding for every schema.
- ATOM's generic PD machinery did not transfer K3's KDA state with MLA pages at
  the measured revision. SGLang's K3-specific path did.
- The performance lane forced fixed output work. It validates serving
  performance and protocol acceptance, not answer quality, tool-call semantic
  correctness, or JSON-schema adherence.

## Sail-Compass integration status

Sail-Compass did not initially have native ATOM labeling, endpoint validation,
or metric aliases. The pinned campaign branch added an `inference_engine:
atom` adapter, preserved captured Chat Completions bodies, mapped ATOM
scheduler/KV/speculation counters into the common schema, and taught the
comparator which engine-specific fields may differ.

Native ATOM profiling remained unsupported. All scored cells therefore used
external-server mode without profiling. Replaying this campaign from a Compass
revision that lacks the adapter is a real blocker, not merely a missing label:
request adaptation and unsupported zero-valued metric fallbacks can invalidate
the comparison.

## Reproducibility pins

- Production-trace dataset SHA-256:
  `23bad9cfcffccd767d70543e5386f82685ad0c0c038928871bcdd031395b96e8`
- Kimi-K3 target revision:
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- SGLang source:
  `87077ed46835622144e005cf38ffb2354a9fcdb6`
- SGLang equal-work overlay:
  `cbf8dc0c5e885cae1ae54f7b14190d631f046acb`
- SGLang image:
  `sha256:273962ddc80483187e5a19d42d15e594d5b6f6bd4d2d7800c259845225c2e5e6`
- ATOM source:
  `55811c6453a32768e7e90e757397a40edf986e9e`
- ATOM image:
  `sha256:c2f0b35ff25aaf3b87a45de2ba9a3443b8d9c2c4dfaf4e7ebda36a81d3bf36dd`
- Sail-Compass:
  `0479b1acf5eddfb9f34611c172f7a86c9a8161b9`

The equal-work SGLang overlay only retains supplied Kimi tool schemas when the
benchmark fixes `tool_choice=none`; it leaves tool parsing disabled. Without
that overlay, realized prompt work differs materially and the comparison fails
its fairness contract.

SGLang's scored image also set `SGLANG_DSPARK_TP_SYNC=0`. The branch-default
rank-0 DSpark decision synchronization reached healthy status but did not
complete a request in the initial attempt. The sync-off run is valid
performance evidence for the known working path, not proof that the branch
default was production-ready.

The production trace and per-request responses contain sensitive request data.
They are intentionally absent from this repository. Only aggregate metrics,
hashes, configs, and non-payload logs should be published.

Relevant checked-in ATOM entry points are:

- `benchmarks/launch_k3_sail_compass.sh` for the pinned K3 server launch;
- `benchmarks/run_k3_sail_compass_variant.sh` for one Compass cell;
- `benchmarks/run_k3_sail_compass_matrix.sh` for the ATOM factorial;
- `deploy/kimi_k3_mi355x/` for the production-style runtime profile.

## What this result does not establish

- Maximum sustainable TPM or maximum concurrent-token capacity.
- EP8 performance or correctness for either engine.
- Answer quality, tool correctness, or constrained-decoding parity.
- A cache-independent 38% engine TTFT advantage.
- That ATOM's worse p99 TPOT and final drain are acceptable under every
  production load shape.
- That later ATOM or SGLang commits preserve these numbers without rerunning
  the campaign.

Treat the selected recipe as a production candidate backed by a matched,
replicated trace replay. Keep an SGLang rollback path until the missing semantic,
saturation, and long-context checks are covered.
