# Cloud-ready GPT pretraining engine

Phase 4C connects the Phase 4A packed token shards to the Phase 4B GPT model.
It implements deterministic sampling, optimization, evaluation, metrics,
distributed execution, and exact optimizer-boundary resumption. The engine is
validated with a tiny CPU model and synthetic local shards. It has not started
the planned 500-million-token run.

## From token shards to model inputs

The training configuration references a Phase 4A `manifest.json` and a GPT
model configuration. Startup validates the manifest schema, little-endian
`uint16` storage, every shard size and checksum, tokenizer fingerprint,
vocabulary size, sequence length, model block size, and RoPE capacity.

Train and validation splits are opened with NumPy memory maps. Each sampler
uses its own seeded `torch.Generator`; only the selected `(B, T + 1)` windows
are copied from the mapped shard and converted into shifted input and target
tensors shaped `(B, T)`. Only those sampled tensors move to the training
device. Complete shards are never loaded into RAM.

Training and validation samplers have separate rank-specific seeds. Fixed
validation saves and restores its generator state around evaluation, so every
evaluation point uses the same batches and validation cannot alter the
training stream.

## Micro-batches and effective global batches

One optimizer update consumes:

`micro_batch_size × sequence_length × gradient_accumulation_steps × world_size`

The production defaults are 8 examples, 1,024 tokens, and 16 accumulation
steps. On one process this is 131,072 tokens per optimizer step. World size
multiplies the effective global batch because every DDP rank samples a
different deterministic stream.

The engine divides each micro-batch loss by the accumulation count before
backward. Non-final DDP micro-steps run under `no_sync()`. Gradients are
unscaled for FP16 when necessary, clipped once immediately before the update,
and cleared with `set_to_none=True`. Optimizer step, micro-step, and global
token counters are tracked separately.

## AdamW and learning-rate schedule

Every trainable storage allocation appears in exactly one optimizer group.
Matrix parameters, including the tied token embedding/output matrix, receive
weight decay. RMSNorm scale vectors and any configured biases do not. Tied
storage is deduplicated before grouping. Fused AdamW is requested only when the
runtime is CUDA and the installed PyTorch exposes that option; CPU and other
devices use the normal implementation.

Learning rate is a stateless function of the completed optimizer-step number
and resolved configuration. Step zero is zero when warmup is enabled. Linear
warmup reaches the configured maximum exactly at the warmup boundary, then
cosine decay reaches the configured minimum exactly at the final calculated
step and never falls below it. Exact resume restores the optimizer step and
optimizer state, so the next learning rate is reproduced without a separate
scheduler object.

## Precision and compilation

`precision: auto` selects BF16 when a CUDA device reports BF16 support, FP16
with `torch.amp.GradScaler` on other CUDA devices, and FP32 on CPU or MPS.
Explicit FP16 or BF16 requests are rejected on non-CUDA devices. Training and
evaluation share the selected autocast policy.

The local smoke workflow selects FP32. CUDA mixed precision has not been tested
on this CPU-only machine.

`torch.compile` is configurable and can be overridden from the CLI. It wraps
the model boundary before DDP while checkpoints continue to use the original
unwrapped model. Compilation is disabled for CPU smoke tests. Setup errors
fail unless the configuration explicitly permits eager fallback; deferred
compiler errors still fail rather than silently changing execution.

## Reproducible validation

Evaluation uses evaluation mode and inference mode, preserving and restoring
the prior training mode. Loss is accumulated as total token negative
log-likelihood and divided by the exact global target-token count. DDP ranks
sum both values before calculating mean cross-entropy. Perplexity returns
infinity rather than overflowing for extremely large losses.

## Exact checkpoint resumption

Production checkpoints are separate from ForgeLM's educational helper. Each
checkpoint contains the unwrapped model, optimizer, optional gradient scaler,
optimizer and micro-step counters, global token count, best validation loss,
resolved training and model configurations, manifest identity, tokenizer
fingerprint, world size, and precision.

Each rank contributes its Python, NumPy, PyTorch CPU, CUDA, training-sampler,
and validation-sampler states. Rank zero writes the gathered checkpoint to a
temporary file, synchronizes it, and atomically replaces the destination.
Compatibility checks reject changes to the model, training configuration,
manifest, tokenizer, world size, or precision.

Checkpoints are captured only after a complete optimizer-step boundary.
Evaluation restores its validation sampler before checkpoint capture.
Serialization and repeated final `latest.pt` writes do not call random APIs.
The CPU test compares uninterrupted three-step training with one interrupted
step followed by exact resume through step three. Model tensors and the full
AdamW state are bitwise equal.

`latest.pt` is updated at safe periodic and final boundaries. Improvements
write `best.pt`; configured intervals write numbered files such as
`step_00000250.pt`. Old numbered checkpoints are not removed. SIGINT or
SIGTERM requests a synchronized stop after the current optimizer update and
writes an `interrupt_step_XXXXXXXX.pt` checkpoint before shutdown. A separate
weights-only mode deliberately restores no optimizer, counters, samplers, or
random state.

## DDP behavior

Without torchrun variables the engine runs as one process. With `RANK`,
`LOCAL_RANK`, and `WORLD_SIZE`, it assigns one CUDA device per process and uses
NCCL, or Gloo for CPU distributed execution. Training streams differ
deterministically by rank, validation totals are reduced globally, only rank
zero prints and writes artifacts, and checkpoint state is gathered from every
rank. Process groups are destroyed in the entry point's `finally` block.

Environment parsing and the single-process path are tested offline. Actual
multi-process and multi-GPU execution remains a cloud validation task.

## Metrics and run metadata

Rank zero appends JSON Lines records containing optimizer and micro-step,
tokens processed, learning rate, raw training loss, optional validation loss
and perplexity, gradient norm, elapsed and step time, tokens per second,
device, precision, and world size.

Atomic run metadata records both resolved configurations, unique model
parameter count, manifest and tokenizer identities, device, PyTorch version,
world size, UTC start time, resume status, and the Git commit when available.
No external experiment service is required.

## Offline smoke commands

If the ignored Phase 4A smoke shards are absent, regenerate them first:

```bash
uv run python scripts/prepare_pretraining_data.py \
  --config configs/pretraining_data_smoke.yaml \
  --local-jsonl tests/fixtures/pretraining_documents.jsonl \
  --force --retrain-tokenizer
```

Run two optimizer steps into a fresh output directory:

```bash
uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_smoke.yaml \
  --device cpu \
  --output-dir outputs/pretrain_gpt_smoke \
  --max-optimizer-steps 2
```

Resume exactly and complete the third step:

```bash
uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_smoke.yaml \
  --device cpu \
  --output-dir outputs/pretrain_gpt_smoke \
  --resume outputs/pretrain_gpt_smoke/latest.pt \
  --max-optimizer-steps 3
```

The smoke model has 30,880 parameters. Its effective batch is
`2 × 16 × 2 × 1 = 64` tokens, so three updates process exactly 192 tokens.

## Eventual cloud commands

After generating the real shards, pinning the dataset and tokenizer identities,
and choosing the CUDA precision/compile policy, a one-GPU command will be:

```bash
uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_50m.yaml \
  --device cuda
```

A same-node multi-GPU launch will use torchrun:

```bash
torchrun --standalone --nproc_per_node=4 scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_50m.yaml \
  --device cuda
```

These are planned commands, not completed runs.

## Remaining work before paid pretraining

The final FineWeb-Edu revision and tokenizer fingerprint must be pinned and the
500-million-token corpus prepared. CUDA BF16/FP16, fused AdamW,
`torch.compile`, NCCL DDP, cloud filesystem throughput, interruption delivery,
and checkpoint upload/recovery need a short unpaid or tightly bounded cloud
qualification run. Batch size, accumulation, compile behavior, validation
cost, and checkpoint cadence may need adjustment from measured accelerator
memory and throughput. The current token target rounds up to a complete global
optimizer batch if it is not exactly divisible.
