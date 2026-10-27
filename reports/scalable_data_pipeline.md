# Scalable pretraining-data pipeline

Phase 4A prepares ForgeLM to move from a small educational corpus to a
reproducible, streaming pretraining corpus. It implements the pipeline and
validates it offline; it does **not** claim that the planned 500 million
training tokens have been processed.

## Why Tiny Shakespeare is not enough

Tiny Shakespeare is useful for fast model, tokenizer, checkpoint, and
evaluation experiments. Its roughly one million characters come from one
narrow source, however, so it cannot provide the volume, topical diversity, or
language distribution needed for ForgeLM's final pretraining experiment.
Repeatedly training on it also makes overfitting much more likely.

The planned source is
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
using its `sample-10BT` configuration. FineWeb-Edu is a filtered educational
subset of web data. Phase 4D pins immutable revision
`924804708136147342bc2390cd29abb1edaf0c67` so later upstream changes cannot
alter the configured source. The read-only preflight never falls back to
`main` if that revision cannot be opened.

## Streaming and deterministic document processing

The Hugging Face source is opened with `streaming=True`. Documents are read as
an iterator rather than downloading or materializing the complete dataset.
The same pipeline also accepts plain strings or mapping-like local records;
the preparation CLI uses this path for the synthetic offline JSONL fixture.

Each pass applies these steps:

1. Select the configured text field.
2. Reject missing, non-string, blank, too-short, and too-long values.
3. Normalize CRLF and CR line endings only when explicitly enabled.
4. Compute an exact-document SHA-256 digest over the accepted UTF-8 text.
5. Reject a digest already observed during that pass.
6. Assign train or validation with a second SHA-256 calculation over the seed
   and document digest.

The split is independent of Python's randomized `hash()` and independent of
iteration order. Document order still determines which accepted documents are
encountered before a finite token budget is filled, so source ordering and the
pinned revision remain part of the reproducibility contract.

Exact deduplication stores one SHA-256 value per valid unique document.
Document contents are not retained during sharding, but the digest set grows
with the number of unique documents. Python set overhead can therefore become
material at very large document counts. Phase 4A accepts that trade-off for the
planned scale; disk-backed or probabilistic deduplication is a future option.

## Train-only tokenizer fitting

Preparation uses two restartable source passes. The first pass takes only
documents assigned to training and retains at most the configured UTF-8 byte
budget. If the final sampled document crosses that budget, it is truncated at
a valid UTF-8 boundary. Validation documents never enter this sample and
cannot influence learned merges. The second, fresh pass performs filtering,
deduplication, splitting, tokenization, and shard writing.

ForgeLM intentionally has two BPE implementations:

- `BytePairTokenizer` is the from-scratch educational implementation used by
  the Tiny Shakespeare phases. It exposes merge mechanics and deterministic
  tie-breaking clearly.
- `ScalableByteLevelTokenizer` uses Hugging Face Tokenizers' Rust-backed BPE
  trainer and encoder. It supplies ByteLevel pre-tokenization and decoding,
  initializes all 256 byte representations, needs no unknown token for valid
  UTF-8, and reserves `<|endoftext|>` explicitly.

The scalable tokenizer is native Tokenizers JSON. ForgeLM validates its model,
pre-tokenizer, decoder, complete byte alphabet, vocabulary size, and special
token on load. Its fingerprint is SHA-256 over canonicalized tokenizer JSON,
so formatting changes do not alter identity. Encoding and decoding preserve
exact Unicode text, including a literal occurrence of the EOD spelling.

## Packed shards and document boundaries

Accepted documents are encoded and followed by the EOD token. This preserves
document boundaries without a separate offsets index. A hard token budget may
truncate the final encoded document; because no later document follows it,
there is no missing inter-document separator.

Tokens are stored as little-endian `uint16`. The writer rejects vocabularies
larger than 65,536 tokens and validates every token ID before writing. It keeps
at most the current shard-sized output plus a single encoded document in
working memory. Deterministic names use `train-00000.bin` and
`validation-00000.bin` sequences.

Shard rollover occurs exactly at `tokens_per_shard`; a document may cross a
shard boundary without losing or duplicating IDs. Each shard is first written
to a hidden temporary path, flushed and synchronized, then atomically renamed.
Empty shards are never created. Existing final or temporary paths cause an
error rather than an implicit resume.

Training stops exactly at `target_tokens`. Validation uses a bounded policy:
validation documents encountered before the training budget is reached are
preserved in stream order up to `validation_target_tokens`. The final
validation document may be truncated to hit that cap exactly. The default
five-million-token validation cap is one percent of the planned training
target.

## Versioned manifest

`manifest.json` records format version 1, source dataset identity and revision,
the creation limits, tokenizer path and fingerprint, vocabulary and EOD
identity, dtype and byte order, document/filter counters, raw UTF-8 bytes,
train and validation token totals, and per-shard names, token counts, byte
sizes, maximum token IDs, and SHA-256 checksums. The manifest is also written
through a temporary path and atomically finalized.

The output directory is assembled under a staging name and renamed only after
all shards and the manifest validate. A completed or stale staging directory
is refused unless `--force` is explicit. This is replacement, not partial-run
continuation.

## Memory-mapped training access

`MemoryMappedShardDataset` validates the manifest, dtype, byte order,
vocabulary metadata, tokenizer compatibility, file sizes, and optionally every
checksum. NumPy opens each shard read-only with `memmap`; complete shards are
not copied into RAM.

Random next-token windows are weighted across all valid starts and sampled
deterministically with a supplied PyTorch generator. Inputs and targets have
shape `(batch_size, block_size)` and targets are shifted by one ID. Windows do
not cross shard boundaries. A shard with fewer than `block_size + 1` tokens is
rejected, so production shard sizes must be selected with the future model
context length in mind.

## Offline reproduction

The synthetic fixture contains short generated notes and deliberate invalid
records. It has no private data or copied long-form work. Run the complete
offline path with:

```bash
uv run python scripts/prepare_pretraining_data.py \
  --config configs/pretraining_data_smoke.yaml \
  --local-jsonl tests/fixtures/pretraining_documents.jsonl \
  --force --retrain-tokenizer
```

Generated tokenizers, staging files, manifests, and binary shards live under
the already ignored `artifacts/` and `data/processed/` trees.

## Reproducibility, source rights, and limitations

Given the same immutable source revision and ordering, configuration, library
versions, and local records, filtering, deduplication, splitting, tokenizer
sample selection, shard naming, token counts, fingerprints, and checksums are
deterministic. The manifest makes these inputs and outputs inspectable.

FineWeb-Edu's dataset card and upstream FineWeb documentation must be reviewed
before the final run. A dataset-level license or terms statement does not
automatically clear every underlying web document; source content can retain
separate copyright, privacy, attribution, or removal obligations. ForgeLM does
not add content-level license classification, personally identifiable
information removal, language balancing, quality rescoring, or near-duplicate
detection in this phase.

Other remaining limitations include the in-memory exact-hash set, two source
passes when fitting a tokenizer, source-order dependence at finite budgets,
single-process shard preparation, no cross-shard training windows, and no
cloud object-store reader yet. Phase 4A only proves the scalable path with
bounded local data. The planned final run remains 500 million training tokens.
