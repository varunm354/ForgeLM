# Production tokenizer and FineWeb-Edu build runbook

## Status and scope

Phase 4E-A artifacts were completed and independently verified on persistent
storage at `/workspace/forgelm-production/fineweb-edu-500m-v1`. The lock
identity is
`a973f330ba67708bd306156628266d763d6875a11d02e187613e6cf045efce1d`;
the tokenizer fingerprint is
`2f420b82fe239f8a80283ac8adb4abe956e3a7192321da1fbc4b36854144a740`.
The lock remains `complete_verified_not_training_approval`: it verifies data
but does not authorize either the Phase 4E-B canary or full pretraining.

CUDA qualification remains passed. No real-data model training has started.

## Existing implementation reused

Phase 4E-A retains the established implementations:

- strict YAML configuration and the pinned FineWeb-Edu revision;
- Hugging Face streaming with no fallback to `main`;
- deterministic SHA-256 train/validation assignment before tokenizer sampling;
- shared filtering, line-ending policy, and exact SHA-256 deduplication;
- Rust-backed byte-level BPE with complete byte fallback;
- canonical tokenizer fingerprints and explicit `<|endoftext|>`;
- little-endian `uint16` token shards, atomic shard publication, SHA-256
  checksums, and versioned manifests;
- memory-mapped Phase 4C readers and deterministic samplers;
- bounded, read-only source probing and overwrite checks.

The confirmed gaps were production corpus metadata, a machine-readable
tokenizer candidate and review boundary, restartable 500M-token shard writing,
a final production-data lock, detailed storage reporting, and explicit
non-cascading commands. Those additions wrap and reuse the Phase 4A
filtering, tokenizer, shard, and manifest formats.

## Immutable source and document policy

The only configured remote source is:

- dataset: `HuggingFaceFW/fineweb-edu`
- configuration: `sample-10BT`
- split: `train`
- revision: `924804708136147342bc2390cd29abb1edaf0c67`
- access mode: streaming

If that exact revision cannot be opened, stop. Never substitute `main`.

Each source pass extracts `text`, applies the configured line-ending policy,
rejects blank, shorter-than-minimum, and longer-than-maximum documents, then
computes the exact-text SHA-256 digest. Duplicate digests are rejected.
Train/validation assignment hashes the configured seed and document digest.
That assignment occurs before tokenizer-corpus collection, so validation text
cannot influence BPE merges.

The tokenizer corpus is a versioned length-prefixed UTF-8 file. It contains
only accepted, unique training documents and deterministically caps the final
document at a valid UTF-8 boundary. Its manifest distinguishes:

- extracted raw source UTF-8 bytes observed by filtering;
- accepted UTF-8 bytes by train and validation partition;
- tokenizer-corpus UTF-8 payload bytes;
- framed corpus file bytes;
- later `uint16` token-storage bytes.

The corpus manifest records the source identity, filtering, partition
algorithm and seed, deduplication policy, document counts, byte counts,
configuration identity, corpus checksum, and corpus identity.

## Persistent artifact layout

Choose a new root for each production build:

```text
<PRODUCTION_ROOT>/
  source_state/       # preserved Stage A report
  tokenizer_corpus/   # train-only framed corpus and manifest
  tokenizer/          # tokenizer.json, candidate manifest, explicit approval
  shards/             # uint16 shards and final shard manifest
  manifests/          # final production-data lock
  progress/           # versioned resumable shard-build state
  logs/               # operator-captured command logs
  verification/       # checksum verification report
```

The root must be on the persistent RunPod network volume. Generated artifacts
remain outside Git. Later repository review should pin the tokenizer
fingerprint, production-data-lock identity, source/config identity, and the
small tokenizer and shard manifests. The tokenizer JSON is also small enough
to review for copying if desired, but this local readiness phase copies
nothing. Binary shards and tokenizer-corpus contents must not be committed.

## Stage A — bounded read-only source probe

Use a CPU-oriented Pod with the existing persistent volume mounted. GPU access
is unnecessary.

```bash
export FORGELM_ROOT=/workspace/ForgeLM
export PRODUCTION_ROOT=/workspace/forgelm-production/fineweb-edu-500m-v1
cd "$FORGELM_ROOT"
mkdir -p "$PRODUCTION_ROOT/source_state" "$PRODUCTION_ROOT/logs"

FORGELM_VERIFY_CUDA=0 bash scripts/setup_runpod.sh

uv run python scripts/prepare_production_data.py \
  --stage probe \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  --preflight-documents 2 \
  | tee "$PRODUCTION_ROOT/source_state/preflight.json"
```

This probes at most two records and does not train a tokenizer or write shards.
Review the exact revision, source result, persistent root, free space, and
dependency checks. Stop if the pinned revision probe fails.

## Stage B — train the candidate production tokenizer

```bash
uv run python scripts/prepare_production_data.py \
  --stage tokenizer \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  | tee "$PRODUCTION_ROOT/logs/stage-b-tokenizer.log"
```

This stage creates only the train-only corpus and candidate tokenizer bundle.
It must reach the configured 100,000,000-byte corpus target, allowing only the
documented UTF-8 boundary cap. It trains exactly 16,384 tokens and validates
the complete byte alphabet, BPE model, ByteLevel components, vocabulary size,
and `<|endoftext|>` special-token ID.

Stop compute after artifact synchronization if review will not happen
immediately; the network volume preserves the stage.

## Stage C — review and pin the tokenizer

Inspect:

- `tokenizer_corpus/manifest.json`
- `tokenizer/manifest.json`
- `tokenizer/tokenizer.json`
- corpus and tokenizer checksums
- source revision, filters, split seed, corpus bytes, and document counts

Set the fingerprint only after independent review:

```bash
export REVIEWED_TOKENIZER_FINGERPRINT=<64-CHARACTER-SHA256>

uv run python scripts/prepare_production_data.py \
  --stage pin-tokenizer \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  --expected-tokenizer-fingerprint "$REVIEWED_TOKENIZER_FINGERPRINT" \
  | tee "$PRODUCTION_ROOT/logs/stage-c-pin-tokenizer.log"
```

A mismatch is fatal. Approval permits shard construction only; it is not
approval for model training. No shard build starts automatically.

## Stage D — build exact token targets

```bash
uv run python scripts/prepare_production_data.py \
  --stage shards \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  | tee "$PRODUCTION_ROOT/logs/stage-d-shards.log"
```

The build targets exactly 500,000,000 training tokens and 5,000,000 validation
tokens. Documents stay in their assigned split and receive
`<|endoftext|>` after encoding. Token IDs are checked against the vocabulary
and `uint16` range.

Training shards roll at 10,000,000 tokens. The configured training target is
nominally divisible into 50 full shards, while validation is nominally one
partial shard. These are expectations, not recorded results: the final
manifest reports the actual shard list, exact token counts, byte sizes,
maximum IDs, and checksums.

If interrupted, rerun only Stage D with:

```bash
uv run python scripts/prepare_production_data.py \
  --stage shards \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  --resume \
  | tee "$PRODUCTION_ROOT/logs/stage-d-resume.log"
```

Resumption never trusts a streaming offset. It reopens the same pinned source
from the beginning, rebuilds filtering/deduplication/split state, and verifies
every committed token against finalized shards before appending. A mismatch in
the committed prefix fails rather than skipping or duplicating data. Remote
tail stability relies on the immutable dataset revision; local fixture runs
also bind the complete source-file SHA-256, so any local tail change is
rejected before resume. Hidden partial shards are removed; finalized
checksum-verified shards are retained. This policy costs replay time but is
the safest exact policy available without a source-provided immutable cursor.

`--force` starts Stage D again only when no completed manifest or final lock
exists. It removes the partial shard set and versioned progress together, so
old and new artifacts cannot mix. A completed build is immutable; use a new
production root to rebuild it.

Stop for human review when Stage D completes. Do not start training.

## Stage E — verify and create the production-data lock

```bash
uv run python scripts/prepare_production_data.py \
  --stage verify \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  | tee "$PRODUCTION_ROOT/logs/stage-e-verify.log"
```

Stage E rereads every shard, verifies sizes, token counts, maximum IDs, and
SHA-256 checksums, then creates:

- `verification/production-data-verification.json`
- `manifests/production-data-lock.json`

The lock pins source dataset/configuration/split/revision, partition algorithm
and seed, filtering and deduplication, tokenizer and special-token IDs,
tokenizer-corpus checksum, shard-manifest checksum, exact token counts, every
shard path/size/checksum, configuration identity, Git commit where available,
completion timestamp, format version, and lock identity.

The Phase 4C runtime rejects FineWeb-Edu training unless both the tokenizer
fingerprint and verified production-data lock are configured and match the
manifest. Stage E still does not authorize training.

## Storage and resource review

Configured payloads require approximately:

- 1,010,000,000 bytes of final raw `uint16` token storage;
- up to 100,000,000 bytes of tokenizer-corpus UTF-8 payload, plus an 8-byte
  length prefix per selected document and a fixed header;
- up to 20,000,000 bytes for one active 10M-token temporary shard;
- a small tokenizer artifact;
- progress, manifests, logs, and verification metadata.

Preflight reports these components separately, uses a 10 MiB metadata/log
allowance, and recommends at least 10 GiB free or twice the conservative
artifact minimum, whichever is larger. A 50 GB persistent volume has ample
nominal capacity if preflight confirms the free-space threshold after
environment and cache usage. Monitor free space during the real build.

No runtime or throughput estimate is asserted. The pipeline is single-process.
RAM use grows with the Rust tokenizer trainer and the in-memory exact SHA-256
deduplication set; monitor both during Stage B and Stage D. A multi-core
CPU-oriented Pod is appropriate, and the Pod can be stopped between stages
without detaching or deleting the network volume.

## Review before a real-data canary

Before Phase 4E-B or any model training:

1. Preserve Stage A–E logs and reports off-Pod.
2. Review the tokenizer corpus and tokenizer identities.
3. Review exact train/validation counts and every shard checksum.
4. Review source rights, privacy, removal, and content-policy limitations.
5. Review `configs/pretrain_gpt_production_canary.yaml`.
6. Obtain explicit approval for the bounded canary only.

Do not pin or launch `configs/pretrain_gpt_50m.yaml` yet.

## Stage D shutdown failure and lifecycle correction

Stage D reached its exact shard totals and published its final manifest before
Python aborted during interpreter shutdown with a `PyGILState_Release` fatal
error. Stage E subsequently re-read and checksum-verified all 51 shards, so
the shutdown failure did not invalidate or conceal the artifact result.

The failure signature matches the upstream PyArrow partial-Parquet-stream
teardown race seen with Python 3.12. ForgeLM's locked environment uses
`datasets 5.1.0` and PyArrow `26.0.0`. Stage D necessarily stops the
FineWeb-Edu stream as soon as both exact token targets are met. Previously,
that partially consumed iterator could leave Arrow prefetch objects to be
destroyed during interpreter finalization, when a worker can no longer safely
release the GIL.

ForgeLM now closes every streamed iterator deterministically when a bounded
consumer stops and runs cyclic garbage collection before the production-data
CLI returns to interpreter shutdown. This is lifecycle cleanup, not error
suppression: source, tokenization, shard, checksum, and close failures still
propagate and produce a nonzero exit. No dependency was changed. On the next
remote streaming command, preserve stdout, stderr, and the shell exit code to
confirm clean process termination in that environment.

## Phase 4E-B — bounded production-data CUDA canary

The canary is a separate command and configuration. It cannot launch the
500-million-token schedule. Its immutable limits and required settings are:

- one CUDA rank with no CPU fallback;
- the full 49,295,872-parameter model;
- 1,024-token sequences;
- explicit BF16 autocast;
- micro-batch 1 and two accumulation micro-steps;
- four optimizer steps and 8,192 training tokens total;
- one fixed validation batch after every optimizer step;
- eager mode only;
- live verification of the production lock, tokenizer bundle, manifests, all
  shard checksums, data configuration, model configuration, and exact pinned
  identities before model construction;
- a simulated safe interruption after step 1, an interruption checkpoint,
  strict exact resume, and three post-resume steps.

The runner writes `canary_report.json`, `metrics.jsonl`, metadata, and hashed
checkpoints. Fresh execution refuses any non-empty output directory. There is
no force/delete option. Exact resume requires a checkpoint inside the same
output directory, and checkpoint compatibility binds the full resolved
training config, model config, manifest hash, tokenizer fingerprint, world
size, and precision.

### Exact RunPod commands

Use the same repository checkout that contains this phase. Keep shell
`pipefail` enabled so `tee` cannot convert a failing canary into a successful
shell status:

```bash
set -euo pipefail
export FORGELM_ROOT=/workspace/ForgeLM
export PRODUCTION_ROOT=/workspace/forgelm-production/fineweb-edu-500m-v1
export CANARY_PARENT=/workspace/forgelm-output
export CANARY_OUTPUT="$CANARY_PARENT/fineweb-edu-canary-$(date -u +%Y%m%dT%H%M%SZ)"
export EXPECTED_TOKENIZER_FINGERPRINT=2f420b82fe239f8a80283ac8adb4abe956e3a7192321da1fbc4b36854144a740
export EXPECTED_PRODUCTION_LOCK_IDENTITY=a973f330ba67708bd306156628266d763d6875a11d02e187613e6cf045efce1d

cd "$FORGELM_ROOT"
mkdir -p "$CANARY_PARENT" "$PRODUCTION_ROOT/logs"
git status --short
git rev-parse HEAD
bash scripts/setup_runpod.sh
```

Re-run Stage E as a read-only verification of the already locked artifacts:

```bash
uv run python scripts/prepare_production_data.py \
  --stage verify \
  --config configs/pretraining_data.yaml \
  --root "$PRODUCTION_ROOT" \
  2>&1 | tee "$PRODUCTION_ROOT/logs/stage-e-pre-canary-verify.log"
```

Confirm the checked-in canary configuration and launch exactly four steps:

```bash
uv run python scripts/run_production_canary.py run \
  --config configs/pretrain_gpt_production_canary.yaml \
  --data-config configs/pretraining_data.yaml \
  --output-dir "$CANARY_OUTPUT" \
  --max-optimizer-steps 4 \
  --expected-production-lock-identity \
    "$EXPECTED_PRODUCTION_LOCK_IDENTITY" \
  2>&1 | tee "$PRODUCTION_ROOT/logs/phase-4e-b-canary.log"
```

The command must exit `0` and the report must say `passed: true`. It must
report `main_training_started: false`, four optimizer steps, eight
micro-steps, 8,192 training tokens, at least one validation record, a resume
start at step 1, and at least one post-resume step.

### Safe resume

The normal command performs its own interruption/resume sequence. If the Pod
or process exits after the step-1 interruption checkpoint was published,
reuse the same output directory and exact config:

```bash
uv run python scripts/run_production_canary.py run \
  --config configs/pretrain_gpt_production_canary.yaml \
  --data-config configs/pretraining_data.yaml \
  --output-dir "$CANARY_OUTPUT" \
  --resume-checkpoint "$CANARY_OUTPUT/interrupt_step_00000001.pt" \
  --max-optimizer-steps 4 \
  --expected-production-lock-identity \
    "$EXPECTED_PRODUCTION_LOCK_IDENTITY" \
  2>&1 | tee "$PRODUCTION_ROOT/logs/phase-4e-b-canary-resume.log"
```

Do not use resume to rerun an already passing report. Verify it instead. To
repeat the canary, choose a new timestamped output directory.

### Independent report and artifact verification

```bash
uv run python scripts/run_production_canary.py verify \
  --report "$CANARY_OUTPUT/canary_report.json" \
  --expected-production-lock-identity \
    "$EXPECTED_PRODUCTION_LOCK_IDENTITY" \
  --expected-tokenizer-fingerprint \
    "$EXPECTED_TOKENIZER_FINGERPRINT" \
  | tee "$PRODUCTION_ROOT/logs/phase-4e-b-canary-verification.log"
```

The verifier rejects a failed report, changed hard bounds, an identity
mismatch, any failed acceptance criterion, or any changed/missing lock,
manifest, tokenizer, model config, metrics, metadata, or checkpoint.

### Acceptance criteria

Every required criterion in `canary_report.json` must pass:

1. exact production lock, tokenizer, data config, source, manifest, shard
   metadata, and live checksums;
2. distinct training and validation shard sets and
   `validation_used_for_training: false`;
3. one CUDA rank, 49,295,872 parameters, sequence length 1,024, and BF16 AMP;
4. finite probe/training loss and gradients;
5. optimizer updates, validation, interruption checkpoint, strict checkpoint
   load, exact counters/RNG/sampler resume, and post-resume updates;
6. complete loss, LR, gradient norm, throughput, token count, timing, VRAM,
   configuration, and artifact-identity evidence;
7. exactly four or fewer approved optimizer steps and no main-training launch.

Any missing evidence is a failure, not a warning.

### Expected duration and approximate RTX 4090 cost

The four updates themselves are sub-second on the previously qualified RTX
4090. End-to-end time is dominated by repeatedly hashing about 1.01 GB of
locked shards, model/optimizer construction, and several roughly 600 MB
checkpoint writes. Budget **5–10 minutes** on a healthy persistent volume,
excluding Pod provisioning and environment installation.

At the October 2026 listed RTX 4090 rates of approximately **$0.34/GPU-hour**
Community or **$0.89/GPU-hour** Secure, 5–10 minutes is roughly
**$0.03–$0.06 Community** or **$0.07–$0.15 Secure**. Storage, idle/provisioning
time, taxes, and current regional pricing are additional. Check the live rate
before launch and stop the Pod promptly after evidence is preserved.

## Gates remaining before main pretraining

Even after a passing canary, do not start the 500M-token run until all of the
following are complete:

1. preserve and independently verify the canary report, metrics, metadata,
   checkpoints, Stage E output, and shell exit codes;
2. review real-data loss, gradient norm, throughput, VRAM, and validation for
   anomalies without treating four steps as a quality result;
3. review source rights, privacy, takedown, and content-policy obligations;
4. decide eager versus compiled mode for the full run; this canary qualifies
   only eager BF16;
5. set and review the final single- or multi-GPU topology. Multi-GPU/NCCL
   remains unqualified;
6. estimate full-run duration, checkpoint storage, and spend from the canary
   evidence and set a budget/stop policy;
7. pin the production fingerprint and lock in the final full-run config,
   review its complete diff, and obtain explicit main-training approval.

The main 500M-token run remains unstarted and unapproved.
