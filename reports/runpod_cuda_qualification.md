# RunPod CUDA qualification guide

This guide records the bounded qualification procedure and its current status.
An RTX 4090 RunPod qualification has been attempted, but Phase 4D GPU
qualification has not passed. No production-data or 500-million-token training
run has started.

## Safety boundary

The qualification uses the real 49,295,872-parameter GPT and a 1,024-token
sequence, but only locally generated synthetic token IDs. The fixture is not
FineWeb-Edu, does not use a production tokenizer, and must never support
model-quality claims. The runner enforces at most 20 optimizer steps and never
automatically changes batch size or retries after an out-of-memory failure.

Use a RunPod PyTorch template rather than a checked-in Dockerfile for now.
ForgeLM's lock resolves a specific PyTorch and Linux CUDA dependency set, but
this repository has not verified a matching RunPod image tag and host-driver
combination. Guessing an image tag would be less reproducible than selecting a
current RunPod PyTorch template, installing the lock, and requiring the
preflight to prove that CUDA actually works.

## Create the Pod and persistent storage

1. In RunPod, create one Pod with one NVIDIA GPU. Do not select multi-GPU for
   this qualification.
2. Attach a persistent network volume or persistent Pod volume with generous
   headroom and mount it at a path such as `/workspace`.
3. Use a current RunPod PyTorch template. Record the template name, GPU model,
   volume size, and region outside the repository.
4. Do not paste API keys, Hugging Face tokens, SSH private keys, or other
   credentials into configuration files, shell history examples, reports, or
   commits.

Set paths in the Pod shell:

```bash
export FORGELM_VOLUME=/workspace
export FORGELM_ROOT="$FORGELM_VOLUME/ForgeLM"
export FORGELM_QUAL_DATA="$FORGELM_VOLUME/forgelm-data/cuda-qualification"
export FORGELM_QUAL_OUTPUT="$FORGELM_VOLUME/forgelm-output/cuda-eager"
mkdir -p "$FORGELM_VOLUME/forgelm-data" "$FORGELM_VOLUME/forgelm-output"
```

Clone the public repository:

```bash
git clone <PUBLIC_REPOSITORY_URL> "$FORGELM_ROOT"
cd "$FORGELM_ROOT"
git rev-parse HEAD
```

## Install the locked environment

Install `uv` using its current official instructions if the template does not
already provide it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install exactly the checked-in lock and verify the resulting PyTorch:

```bash
bash scripts/setup_runpod.sh
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

Stop here if `uv sync --frozen` changes the lock, CUDA is false, or the locked
PyTorch cannot use the template's driver. Do not fix that by silently changing
dependencies during a paid session.

## Create the synthetic fixture

```bash
uv run python scripts/create_cuda_qualification_data.py \
  --output-dir "$FORGELM_QUAL_DATA"
```

The command writes two small training shards, one validation shard, checksums,
and a synthetic-data warning. Re-running without `--force` is refused.

## Run environment preflight

```bash
uv run python scripts/cloud_preflight.py \
  --config configs/pretrain_gpt_cuda_qualification.yaml \
  --manifest "$FORGELM_QUAL_DATA/manifest.json" \
  --output-dir "$FORGELM_QUAL_OUTPUT" \
  --json-output "$FORGELM_VOLUME/forgelm-output/cloud_preflight.json"
```

Review every required check before training. The report includes PyTorch/CUDA,
cuDNN, GPU, VRAM, BF16, SDPA controls, compilation, NCCL, disk, model,
configuration, manifest, and fixture identities. It does not enumerate
environment variables.

## Run eager qualification

```bash
uv run python scripts/qualify_cuda_training.py \
  --config configs/pretrain_gpt_cuda_qualification.yaml \
  --manifest "$FORGELM_QUAL_DATA/manifest.json" \
  --output-dir "$FORGELM_QUAL_OUTPUT" \
  --max-optimizer-steps 20 \
  --no-compile
```

The runner performs a forward/backward probe, trains to an intermediate
checkpoint, reconstructs the engine from that checkpoint, and completes at
least one post-resume update. It exits nonzero for CUDA/preflight failure,
non-finite values, OOM, failed resume, corrupt artifacts, or unmet mandatory
acceptance criteria.

## Checkpoint-resume troubleshooting and pending rerun

The first real run used an RTX 4090, PyTorch `2.14.1+cu130`, CUDA 13.0, BF16,
1,024-token sequences, two accumulation micro-steps, the full
49,295,872-parameter model, and eager execution. CUDA preflight and the
full-model forward/backward probe passed with finite loss and gradients. The
run completed five optimizer steps, ten micro-steps, 10,240 tokens, validation,
and checkpoint creation, then aborted while reconstructing the engine with:

```text
TypeError: RNG state must be a torch.ByteTensor
```

The checkpoint loader had applied `map_location=cuda` to the entire checkpoint.
That moved saved CUDA RNG-state ByteTensors, which PyTorch stores and restores
from CPU, onto CUDA before `torch.cuda.set_rng_state_all`. Checkpoints are now
staged on CPU; CPU, CUDA, and sampler RNG tensors are validated and normalized
to contiguous CPU ByteTensors before restoration. Model parameters remain on
the selected training device, and AdamW uses its device-aware state loader.

After pulling the correction, rerun the bounded qualification from a clean
qualification output directory:

```bash
cd "$FORGELM_ROOT"
git pull --ff-only origin main
uv run python scripts/qualify_cuda_training.py \
  --config configs/pretrain_gpt_cuda_qualification.yaml \
  --manifest "$FORGELM_QUAL_DATA/manifest.json" \
  --output-dir "$FORGELM_QUAL_OUTPUT" \
  --max-optimizer-steps 20 \
  --no-compile \
  --force
```

GPU qualification remains pending until this rerun passes every required
acceptance criterion. The partial run is diagnostic evidence only and must not
be reported as a Phase 4D GPU qualification pass.

## Optionally qualify compilation

Use a separate output directory so eager evidence remains intact:

```bash
export FORGELM_COMPILE_OUTPUT="$FORGELM_VOLUME/forgelm-output/cuda-compile"
uv run python scripts/qualify_cuda_training.py \
  --config configs/pretrain_gpt_cuda_qualification.yaml \
  --manifest "$FORGELM_QUAL_DATA/manifest.json" \
  --output-dir "$FORGELM_COMPILE_OUTPUT" \
  --max-optimizer-steps 20 \
  --compile
```

A compile failure is recorded as an optional capability failure; it is not
reported as a successful compiled run. Preserve the successful eager report
separately.

## Artifacts, resume, and shutdown

Each output directory contains:

- `qualification_report.json`
- `run_metadata.json`
- `metrics.jsonl`
- `latest.pt`, `best.pt`, and numbered checkpoints

Resume with the same configuration, manifest, precision, world size, and
output directory. The absolute step limit must exceed the saved step:

```bash
uv run python scripts/qualify_cuda_training.py \
  --config configs/pretrain_gpt_cuda_qualification.yaml \
  --manifest "$FORGELM_QUAL_DATA/manifest.json" \
  --output-dir "$FORGELM_QUAL_OUTPUT" \
  --resume-checkpoint "$FORGELM_QUAL_OUTPUT/latest.pt" \
  --max-optimizer-steps 20 \
  --no-compile
```

Before stopping the Pod:

1. Confirm the data and output paths are on persistent storage.
2. Copy or download the preflight, qualification report, metrics, metadata,
   checkpoints, manifest, and shard checksums.
3. Hash copied artifacts and compare them with the source.
4. Stop the Pod as soon as inspection or transfer is complete.
5. Verify persistent artifacts from another session before terminating the
   Pod. Termination can destroy ephemeral container storage.

The default fixture contains 73,728 `uint16` tokens, or 147,456 raw bytes,
plus a small manifest. A post-update checkpoint stores roughly 188 MiB of
FP32 model weights plus roughly 376 MiB of Adam moments, with serialization
overhead. Multiple numbered, best, and latest checkpoints can therefore use
several GiB. The Python environment and CUDA packages can use many additional
GiB. Allocate generous persistent-volume headroom and monitor it before every
run.
