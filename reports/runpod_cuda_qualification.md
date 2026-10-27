# RunPod CUDA qualification report and guide

## Qualification status

Phase 4D bounded single-GPU CUDA qualification passed on a real RunPod
RTX 4090 after commit `658e350` corrected CUDA RNG checkpoint restoration.
Both required eager mode and optional `torch.compile` mode passed every
required acceptance criterion, including exact checkpoint resumption from
optimizer step 5 through step 20. The compiled acceptance criterion also
passed.

This result qualifies the synthetic, 20-step, single-GPU training path. It
does not qualify multi-GPU execution, production data, a long training run, or
model quality. ForgeLM has not been pretrained and has not learned language.

## Qualified environment and run identity

- GPU: NVIDIA GeForce RTX 4090
- GPU memory: 25,252,724,736 bytes
- Compute capability: 8.9
- Visible GPUs: one
- PyTorch: `2.14.1+cu130`
- CUDA runtime: 13.0
- cuDNN: 92400
- Python: 3.12.3
- Precision: BF16, supported and selected
- Model: 49,295,872 trainable parameters
- Sequence length: 1,024
- Effective batch: `1 × 1,024 × 2 accumulation steps × 1 GPU = 2,048` tokens
- Bound: 20 optimizer steps, 40 micro-steps, 40,960 synthetic tokens
- Synthetic fixture identity:
  `b039daa5fdb33d71a6edd1537953d7345411a51987542585e84fd735cdeb6c6d`
- Manifest SHA-256:
  `9e0b0c9cd38d3cd050c505bb2e6d77be3b39c166c15a6999987746c7b4ab906f`

## Safety boundary

The qualification uses the real 49,295,872-parameter GPT and a 1,024-token
sequence, but only locally generated synthetic token IDs. The fixture is not
FineWeb-Edu, does not use a production tokenizer, and must never support
model-quality claims. The runner enforces at most 20 optimizer steps and never
automatically changes batch size or retries after an out-of-memory failure.

The recorded environment proves that ForgeLM's lock worked on the qualified
RunPod host. Continue to require preflight on every new host because a different
template, driver, CUDA stack, or GPU is not covered by this result.

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

### Eager qualification result

The eager run passed with `failure: null`. All required criteria passed:
CUDA detection, full-model loading, production sequence length, BF16,
finite forward loss, finite gradients, optimizer updates, validation,
checkpoint save/load, interruption/resumption, a post-resume optimizer step,
peak-VRAM recording, artifact integrity, and metrics/metadata output.

- Resume range: optimizer step 5 through step 20
- Validation records: 4
- Median optimizer-step latency: 0.04978756356285885 seconds
- Steady-state throughput: 41,693.00299628792 tokens/second
- Peak allocated VRAM: 1,515,574,784 bytes
- Peak reserved VRAM: 1,564,475,392 bytes
- Final training loss: 9.832122802734375
- Final validation loss: 9.8648681640625
- Final validation perplexity: 19,242.336644453517
- Final checkpoint SHA-256:
  `aaebd690a1436b75094a18c2da86989f320ced32a435f87b0fefea4e06028a92`

The token fixture is synthetic and was generated only to exercise systems
behavior. Its loss and perplexity do not measure language-model quality,
learning progress, or expected FineWeb-Edu performance.

## CUDA RNG restoration issue and correction

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

Commit `658e350` implemented that correction. The successful eager and compiled
runs both reconstructed the engine at optimizer step 5, restored exact state,
and completed through optimizer step 20. Exact single-GPU CUDA checkpoint
resumption is therefore qualified for this environment.

The successful eager rerun used:

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

## Compiled qualification

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

### Compiled qualification result

The compiled run passed with `failure: null`. All required criteria and the
optional compilation criterion passed. Resume again began at optimizer step 5
and completed through step 20.

- Compilation startup time: 28.73571403697133 seconds
- Median optimizer-step latency: 0.029048574448097497 seconds
- Steady-state throughput: 71,281.70456779329 tokens/second
- Peak allocated VRAM: 1,314,649,600 bytes
- Peak reserved VRAM: 1,436,549,120 bytes
- Final training loss: 9.830730438232422
- Final validation loss: 9.863788604736328
- Final validation perplexity: 19,221.57460941236
- Final checkpoint SHA-256:
  `17bc5d23657103b23552320e58c4b055e96fd89cfb60ac3c5fc3ff674c1948ac`

Compared with eager mode, compilation increased steady-state throughput by
approximately 70.97%, reduced median optimizer-step latency by approximately
41.65%, and reduced peak allocated VRAM by approximately 13.26%. It added a
one-time startup cost of approximately 28.74 seconds.

Compiled mode is the current candidate for the later production run because it
performed substantially better in this bounded test. Eager mode remains the
validated fallback if compilation fails or behaves differently with production
data or a longer run. These measurements are not a full-run cost or duration
estimate.

Multi-GPU/NCCL execution remains untested. The production tokenizer,
production tokenizer fingerprint, 500-million-token FineWeb-Edu shards, short
real-data training gate, and full paid pretraining run also remain incomplete.

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
