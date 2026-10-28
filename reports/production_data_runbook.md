# Production tokenizer and FineWeb-Edu build runbook

## Status and scope

Phase 4E-A makes the existing Phase 4A pipeline ready for a staged production
execution on persistent storage. It does not contain production artifacts.
The 16,384-token tokenizer has not been trained, FineWeb-Edu has not been
processed, and no 500-million-token model-training run has started.

CUDA qualification remains passed. Data preparation is CPU-oriented and does
not need a GPU.

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
3. Pin the reviewed tokenizer fingerprint in `configs/pretrain_gpt_50m.yaml`.
4. Point `production_data_lock_path` at the verified persistent lock.
5. Review exact train/validation counts and every shard checksum.
6. Review source rights, privacy, removal, and content-policy limitations.
7. Create a separately bounded real-data canary configuration.
8. Obtain explicit approval for that canary only.

Multi-GPU qualification, the short FineWeb-Edu canary, and full paid
pretraining remain incomplete.
