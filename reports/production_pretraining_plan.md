# Production pretraining launch plan

This is a gated plan, not a record of executed cloud work. Phase 4D has not
trained a production tokenizer, processed the 500-million-token corpus, run on
real FineWeb-Edu shards, or started paid pretraining. It has completed bounded
single-GPU CUDA qualification on synthetic data.

## Gate status

1. **Passed — all local tests and lint checks.** The complete offline suite
   passes in the locked environment at Phase 4D closeout.
2. **Passed — CUDA preflight.** The preserved single-GPU RunPod evidence
   records an RTX 4090, PyTorch `2.14.1+cu130`, CUDA 13.0, cuDNN 92400, one
   visible GPU, and BF16 support.
3. **Passed — bounded full-model qualification.** The 49,295,872-parameter
   model completed 20 optimizer steps and 40 micro-steps at sequence length
   1,024 in both eager and compiled BF16 modes. Finite loss and gradients,
   validation, checkpoints, memory, throughput, artifacts, and metadata all
   passed their required acceptance checks.
4. **Passed — exact checkpoint resumption.** Both CUDA runs saved at optimizer
   step 5, reconstructed the engine, and completed through step 20 with a
   post-resume optimizer update. The earlier CUDA RNG device-restoration defect
   was corrected by commit `658e350` before these passing runs.
5. **Pending — production tokenizer training.** Train only on production
   training documents; do not use validation documents or the synthetic
   qualification fixture. Phase 4E-A now provides the staged, checksummed
   train-only corpus workflow, but it has not been run on FineWeb-Edu.
6. **Pending — production tokenizer fingerprint.** Record the artifact SHA-256
   identity in the production training configuration and preserve the
   tokenizer. The explicit review/approval command exists; no production
   fingerprint exists yet.
7. **Pending — 500M-token shards and checksums.** Validate every FineWeb-Edu
   shard and preserve the manifest before training. Restartable construction,
   full verification, and the final production-data-lock format are ready only
   on offline fixtures.
8. **Pending — short FineWeb-Edu training run.** Use production shards with an
   explicit small step cap and inspect numerical behavior, throughput,
   validation, storage, and resume.
9. **Pending — explicit human approval for the full paid run.** No script may
   infer approval from earlier gates.
10. **Pending — final training and evaluation.** Preserve all configurations,
    identities, metrics, checkpoints, and evaluation outputs.

Failure at any gate stops progression. A later gate does not waive an earlier
one. Progression is currently blocked at Gate 5.

Follow the
[production-data runbook](production_data_runbook.md) for non-cascading
Stages A–E. A later FineWeb-Edu training configuration must pin both the
reviewed tokenizer fingerprint and the verified production-data lock; the
runtime rejects production-source training without them.

The compiled single-GPU path is the current production candidate based on the
bounded qualification's higher throughput and lower latency and allocated
VRAM. Eager mode is the validated fallback. This choice must be rechecked
during the short real-data gate; synthetic qualification losses are not
model-quality evidence. Multi-GPU/NCCL qualification remains incomplete and
must be performed before any multi-GPU launch.

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
The bounded RTX 4090 measurements are insufficient for a full-run runtime or
cost estimate because they exclude production data loading, long-run behavior,
checkpoint retention, and provider variability.

## Cost controls and manual stop conditions

- Use the qualified single-GPU path for the short real-data gate.
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
