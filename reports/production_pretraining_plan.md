# Production pretraining launch plan

This is a gated plan, not a record of executed cloud work. Phase 4D has not
trained a production tokenizer, processed the 500-million-token corpus,
qualified CUDA, or started paid pretraining.

## Gates

1. **All local tests pass.** Run the complete offline test and lint suite from
   the locked environment.
2. **CUDA preflight passes.** Preserve the JSON report from a single-GPU
   RunPod host.
3. **Bounded full-model qualification passes.** Require the 49,295,872
   parameter model, 1,024-token sequences, mixed precision, finite loss and
   gradients, validation, checkpoints, memory, and throughput reporting.
4. **Checkpoint resumption passes.** Require an intermediate exact resume and
   at least one successful post-resume optimizer update.
5. **Tokenizer is trained only on production training documents.** Do not use
   validation documents or the synthetic qualification fixture.
6. **Tokenizer fingerprint is pinned.** Record the artifact SHA-256 identity
   in the production training configuration and preserve the tokenizer.
7. **500M-token shards and checksums are created.** Validate every shard and
   preserve the manifest before training.
8. **A short FineWeb training run passes.** Use production shards with an
   explicit small step cap and inspect numerical behavior, throughput,
   validation, storage, and resume.
9. **A human explicitly approves the full paid run.** No script may infer
   approval from earlier gates.
10. **Final training and evaluation run.** Preserve all configurations,
    identities, metrics, checkpoints, and evaluation outputs.

Failure at any gate stops progression. A later gate does not waive an earlier
one.

## Pinned data contract

The production source is `HuggingFaceFW/fineweb-edu`, configuration
`sample-10BT`, split `train`, streamed at immutable revision
`924804708136147342bc2390cd29abb1edaf0c67`. Preparation targets 500,000,000
training tokens using a 16,384-token production tokenizer trained from a
100 MB train-only sample. Shards use 10,000,000 little-endian `uint16` tokens
and `<|endoftext|>` document separators.

This implies 50 full training shards at 10 million tokens each, subject to the
final manifest behavior. Raw training-token storage is approximately 1 GB
(1,000,000,000 bytes). Validation tokens add approximately 10 MB with the
current five-million-token cap. These figures exclude manifests, tokenizer
files, checkpoints, the Python/CUDA environment, caches, and temporary files.

Run the read-only production preflight first:

```bash
uv run python scripts/prepare_pretraining_data.py \
  --config configs/pretraining_data.yaml \
  --preflight-only
```

If connectivity is available but the pinned revision cannot be opened by the
installed Datasets interface, stop and report that incompatibility. Never
substitute `main`.

After all prior gates and with explicit output paths:

```bash
uv run python scripts/prepare_pretraining_data.py \
  --config configs/pretraining_data.yaml \
  --output-dir <PERSISTENT_DATA_DIRECTORY> \
  --tokenizer-path <PERSISTENT_TOKENIZER_PATH> \
  --retrain-tokenizer
```

This template is not authorized for execution by this plan.

## Storage planning

The 49.3M model has 197,183,488 bytes of FP32 parameter storage. A checkpoint
after AdamW state initialization contains model weights and approximately two
same-sized optimizer moment tensors, plus counters, random states, and
serialization overhead. A practical planning floor is therefore roughly
600 MB per checkpoint, not a guaranteed final size. `latest`, `best`, periodic,
and interrupt checkpoints can coexist; retention policy determines aggregate
usage.

Use a persistent volume with generous headroom for:

- roughly 1.01 GB of raw production and validation token shards;
- tokenizer and manifest artifacts;
- source/library caches and temporary preparation paths;
- the locked Python and CUDA environment;
- multiple checkpoints and metrics;
- copied artifacts during atomic publication or rollback.

Measure actual free space during preflight and before each gate. Do not size a
volume exactly to the raw shard estimate.

## Short production-shard gate

After the manifest and tokenizer fingerprint are pinned, copy the production
training configuration to a separately reviewed short-run configuration. Keep
the real architecture and 1,024-token sequence, point it at the production
manifest, use a separate persistent output directory, and invoke an explicit
small absolute step limit:

```bash
uv run python scripts/pretrain_gpt.py \
  --config <REVIEWED_SHORT_FINEWEB_CONFIG> \
  --device cuda \
  --max-optimizer-steps <SMALL_APPROVED_LIMIT>
```

Do not repurpose the synthetic qualification report as evidence for this
FineWeb gate.

## Full-run template

Only after Gate 9:

```bash
uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_50m.yaml \
  --device cuda
```

For any actual launch, first pin
`expected_tokenizer_fingerprint`, use persistent absolute paths, review the
effective batch and calculated optimizer steps, and capture the Git commit.
This plan intentionally contains no runtime or cost estimate because no target
GPU has been measured.

## Cost controls and manual stop conditions

- Start with one GPU and the bounded qualification.
- Set provider spending alerts and inspect the provider price before launch.
- Keep the Pod stopped except during active setup, qualification, transfer, or
  approved training.
- Never enable an unbounded retry loop.
- Stop manually for non-finite values, OOM, repeated compiler failure,
  unexpected batch/sequence/model identity, checksum mismatch, declining free
  disk, missing validation, failed checkpoint, failed resume, abnormal step
  latency, or loss behavior requiring investigation.
- Treat provider interruption, storage errors, and changed dependency
  resolution as stop conditions, not reasons to bypass gates.

## Rollback and resume

At every checkpoint boundary, preserve the previous known-good checkpoint
until the new file is hashed and load-tested. If a run fails:

1. Stop compute after logs and diagnostics are safely written.
2. Preserve the failing output without editing it.
3. Verify the manifest, tokenizer, configuration, and last known-good
   checkpoint identities.
4. Resume only with the exact model/training configuration, manifest,
   tokenizer fingerprint, precision, and world size.
5. Use `--resume <PERSISTENT_CHECKPOINT>` for the normal pretraining engine.
6. If compatibility validation fails, investigate; do not bypass it with a
   weights-only load for an exact-resume claim.

## Artifact-preservation checklist

- Git commit and clean/dirty status
- model and training YAML files and their hashes
- dataset source revision and filtering settings
- tokenizer JSON and fingerprint
- manifest and manifest SHA-256
- every shard name, size, token count, maximum ID, and checksum
- preflight and qualification JSON reports
- run metadata and JSONL metrics
- latest, best, periodic, and interrupt checkpoints with hashes
- validation and final evaluation outputs
- provider template, GPU model, driver, CUDA, cuDNN, and PyTorch versions
- explicit human approval record for Gate 9, stored outside credentials
