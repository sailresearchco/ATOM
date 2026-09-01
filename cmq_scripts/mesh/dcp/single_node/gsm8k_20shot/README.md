# GLM-5.2 DCP + MTP GSM8K 20-shot

This suite validates the sharded sparse indexer with DCP4 and MTP3 in:

- `pd_mix`: one DCP4 server, without KV transfer.
- `pd_separate`: PP4 prefill plus DCP4 decode, with interleave-1 index staging
  and Mooncake RDMA.

The scripts keep server configuration, aggregate evaluation output, and
per-sample prompts/generations under one run directory:

```text
<run-dir>/
├── server/
│   ├── config.txt
│   ├── server.log                 # pd_mix
│   ├── prefill.log                # pd_separate
│   └── decode.log                 # pd_separate
└── eval/
    ├── config.txt
    ├── gsm8k_20shot.log
    └── samples/
```

Run each lifecycle step separately so failures stop at a visible boundary:

```bash
# 1. Stop
bash scripts/stop_atom_server.sh

# 2. Start one scenario
bash cmq_scripts/mesh/dcp/single_node/gsm8k_20shot/start_pd_mix.sh <run-dir>
# or:
bash cmq_scripts/mesh/dcp/single_node/gsm8k_20shot/start_pd_separate.sh <run-dir>

# 3. Evaluate (mesh port 8000 for PD-separated; server port 8020 for pd_mix)
bash cmq_scripts/mesh/dcp/single_node/gsm8k_20shot/eval_gsm8k_20shot.sh \
  <port> <run-dir> <scenario>

# 4. Monitor while the evaluator runs
bash scripts/wait_infer_drain.sh <port> 30 10 [server-log] 6

# 5. Stop
bash scripts/stop_atom_server.sh
```

Actual run paths and final scores are recorded in `RESULTS.md`.
