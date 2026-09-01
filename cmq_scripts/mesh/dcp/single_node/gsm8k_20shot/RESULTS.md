# GSM8K 20-shot results

Model: `/mnt/models/GLM-5.2-MXFP4`
Indexer: DCP4 sharded (`ATOM_DCP_REPLICATE_INDEX_CACHE=0`)
Speculation: MTP3
Generation: `max_tokens=16384,temperature=0,top_p=1`

## Full runs (1319 samples)

- PD mix
  - Run: `/it-share/mengqingcao/log/atom_experiments/gsm8k20_pd_mix_20260901`
  - Flexible extract: `0.9712 ± 0.0046`
  - Strict match: `0.9719 ± 0.0045`
  - Runtime: `13m19s`
- PD separate, before the producer-worker device fix
  - Run: `/it-share/mengqingcao/log/atom_experiments/gsm8k20_pd_separate_20260901`
  - Flexible extract: `0.9227 ± 0.0074`
  - Strict match: `0.9234 ± 0.0073`
  - Runtime: `11m09s`
- PD separate, after the producer-worker device fix
  - Run:
    `/it-share/mengqingcao/log/atom_experiments/gsm8k20_pd_separate_postfix_full_20260901`
  - Flexible extract: `0.9644 ± 0.0051`
  - Strict match: `0.9636 ± 0.0052`
  - Runtime: `6m28s`

The post-fix PD-separate result is within sampling error of PD mix: the
flexible-extract difference is `-0.0068` (0.99 combined standard errors), and
the strict-match difference is `-0.0083` (1.21 combined standard errors).
All requests completed and neither prefill nor decode logged a traceback,
assertion, runtime error, process death, or shutdown.

## Runtime diagnosis

Before the fix, the prefill PP processes initialized on CUDA devices 0/1/2/3,
but every `mooncake-send-worker` started with CUDA device 0. The index gather
kernel runs on the source tensor's device, while the following no-argument
`torch.cuda.current_stream().synchronize()` used the worker's current device.
Ranks 1/2/3 therefore synchronized the wrong stream and could start RDMA before
the staged index pages were complete.

The producer executor now uses `torch.cuda.set_device` as its thread
initializer. Post-fix runtime evidence showed worker device 0/1/2/3 matching
the owning PP process, all staged page counts matching destination page counts,
all four producer-stage notifications arriving before consumer completion, and
all four DCP ranks using interleave 1 with the published MTP N=4 context
metadata.
