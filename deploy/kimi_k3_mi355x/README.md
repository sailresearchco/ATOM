# Kimi-K3 on one 8x MI355X node

This is Sail's self-contained production profile for the ATOM configuration
selected by the K3 trace ablation:

- TP8, DCP8, EP1, DSpark proposal depth 4
- 8,192-token chunked-prefill budget and 32 running sequences
- 4 continuous decode steps, FP8 KV cache, prefix caching enabled
- 16,384-token context and 85% GPU-memory target
- prefill delayer available but explicitly disabled to match the current MiTAC
  SGLang scheduling policy

The container has a bounded Docker healthcheck, automatic restart, high file
descriptor limits, graceful shutdown, and rotated logs. Runtime source is baked
into the image and labeled with its full Git revision; no source checkout is
mounted into the container.

## Build and stage

The base image digest contains the validated gfx950 AITER build. On one build
host with that image available:

```bash
cp deploy/kimi_k3_mi355x/.env.example deploy/kimi_k3_mi355x/.env
# Set ATOM_IMAGE in .env to a commit-addressed Sail tag.
deploy/kimi_k3_mi355x/manage.sh build
docker push "$ATOM_IMAGE"
```

On each serving node, copy `.env`, pull the exact image, and run:

```bash
deploy/kimi_k3_mi355x/manage.sh validate
deploy/kimi_k3_mi355x/manage.sh up
```

`up` waits for model readiness, runs a real chat completion, checks Prometheus
metrics, verifies an idle cache reset on every engine, and writes the resolved
container/image configuration to `manifests/`.

The target and draft snapshots must already exist beneath
`ATOM_HF_CACHE_ROOT`. The cache is mounted read-only and both Hugging Face
offline modes are forced, preventing a replica from silently fetching or
changing model artifacts during rollout.

## Rollout boundary

The API binds to all interfaces by default because Sail's worker/Tailscale
network namespace supplies the service boundary. Do not expose port 31000 on an
untrusted network: ATOM's standalone API and administrative cache-reset endpoint
do not provide authentication themselves.

Drain a replica before `down` or replacement. For a production rollout, use one
canary, verify `status`, `smoke`, `/metrics`, and logs, then proceed one replica
at a time. The one-hour comparison harness intentionally runs outside this
directory so serving configuration and workload generation remain independent.
