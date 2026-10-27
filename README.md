# ForgeLM

ForgeLM is an end-to-end project for training, evaluating, fine-tuning, and serving small language models. The code is meant to stay readable while it grows into a complete training and serving stack.

## Planned phases

1. **Foundation** — package layout, configuration, and shared utilities.
2. **Data** — datasets, tokenization, and batching.
3. **Model** — a small Transformer language model.
4. **Training** — the training loop, checkpoints, and experiment tracking.
5. **Evaluation** — metrics and held-out evaluation.
6. **Fine-tuning** — adaptation of a trained checkpoint.
7. **Serving** — an inference service for trained models.

## Current status

The project foundation is complete, including the package layout and shared
seeding and device helpers. Phase 1A is complete with character-level
tokenization, reproducible language-model batching, and a Tiny Shakespeare
download script for pipeline validation.

Phase 1B is complete: the character-level bigram baseline was trained and
evaluated on Tiny Shakespeare. Phase 2A completed a single causal self-attention
head with explicit attention mathematics.

Phase 2B contains explicit multi-head causal self-attention, a position-wise
feed-forward network, layer normalization, residual connections, and a complete
pre-norm Transformer block.

Phase 2C assembled those components into a complete decoder-only Transformer
language model with token and learned positional embeddings, stacked Transformer
blocks, next-token cross-entropy training, autoregressive generation, and a
configurable training script. Its 5,000-step Tiny Shakespeare baseline run is
complete.

Phase 2D adds deterministic full-corpus evaluation so the best Bigram and
Transformer checkpoints can be compared over every validation target exactly
once.

Phase 3A adds a deterministic byte-level BPE tokenizer trained from scratch.
Its 256-byte base vocabulary supports arbitrary UTF-8 text without unknown
tokens, while 256 learned merges reduce Tiny Shakespeare from 1,115,394
character tokens to 691,315 BPE tokens.

Phase 3B integrates BPE with the Transformer training pipeline. Raw text is
split before tokenizer training, so merges are learned from the 90% training
portion only and validation text cannot leak into the vocabulary. Tokenizer
artifacts have stable SHA-256 fingerprints recorded in checkpoints for
compatibility validation. The BPE Transformer's 5,000-step Tiny Shakespeare
baseline run is complete.

Character-level and BPE token-level perplexities are not directly comparable:
their tokens represent different amounts of source text.

Phase 3C adds exact full-corpus bits-per-byte evaluation. Both Transformer
checkpoints score the same raw validation text, with every target token's
negative log-likelihood normalized by its exact represented UTF-8 bytes.

Phase 4A adds the scalable pretraining-data path. It streams FineWeb-Edu or a
local JSONL source, performs deterministic filtering, exact deduplication, and
hash-based train/validation assignment, fits a Rust-backed byte-level BPE on
training documents only, and writes checksummed `uint16` shards for
memory-mapped batching. The pipeline has been validated with the synthetic
offline fixture; the planned 500-million-token corpus has not been processed.

Phase 4B adds a separate production-oriented decoder-only GPT architecture
with RMSNorm, RoPE, fused QKV projections, PyTorch scaled dot-product
attention, SwiGLU, pre-normalized residual blocks, and tied token/output
weights. The default configuration has exactly 49,295,872 trainable parameters
and a 1,024-token context. CPU architecture checks are complete; the model has
not been pretrained.

Phase 4C connects memory-mapped shards to that GPT with intentional AdamW
grouping, warmup/cosine scheduling, gradient accumulation and clipping,
device-aware mixed precision, fixed validation, atomic exact-resume
checkpoints, JSONL metrics, optional compilation, and torchrun-compatible DDP.
The tiny offline CPU workflow passes interruption/resumption checks; no
500-million-token pretraining run has started.

See the [BPE tokenizer report](reports/bpe_tokenizer.md) for the algorithm,
compression results, and limitations, and the
[BPE Transformer report](reports/bpe_transformer_baseline.md) for training and
normalized evaluation results. The
[scalable data pipeline report](reports/scalable_data_pipeline.md) documents
the Phase 4A formats, reproducibility contract, and data-source considerations.
The [GPT 50M architecture report](reports/gpt_50m_architecture.md) documents
the production model design, tensor shapes, exact parameter count, and
remaining cloud-training work. The
[cloud pretraining engine report](reports/cloud_pretraining_engine.md)
documents batching, optimization, validation, exact checkpoints, DDP, and the
offline smoke workflow.

## Tokenizer-normalized Transformer comparison

| Model | Validation tokens | Total NLL | Scored bytes | Bits/byte |
| --- | ---: | ---: | ---: | ---: |
| Character Transformer | 111,540 | 193,360.159408863 | 111,539 | 2.501006312 |
| BPE Transformer | 70,184 | 180,316.900768768 | 111,539 | 2.332299003 |

On this exact benchmark, BPE improves bits per byte by **0.168707309**
(**6.7456%**). Context resets at non-overlapping 128-token window boundaries.

## Baseline comparison

These are the final sampled training-run metrics:

| Model | Parameters | Validation loss | Validation perplexity |
| --- | ---: | ---: | ---: |
| Bigram | 4,225 | 2.4597 | 11.70 |
| Transformer | 824,897 | 1.7348 | 5.67 |
| Improvement | — | 29.47% lower | 51.54% lower |

See the [Bigram report](reports/bigram_baseline.md) and
[Transformer report](reports/transformer_baseline.md). Run the separate
deterministic benchmark with:

```bash
uv run python scripts/compare_baselines.py --device cpu
```

## Layout

- `configs/` — experiment and model configuration
- `data/raw/`, `data/processed/` — local datasets, not committed
- `scripts/` — entry-point scripts
- `src/forgelm/` — library code for data, models, training, evaluation, serving, and utilities
- `tests/` — pytest suite

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync
uv run pytest
uv run ruff check .
```

Run the Phase 4A offline smoke pipeline without network access:

```bash
uv run python scripts/prepare_pretraining_data.py \
  --config configs/pretraining_data_smoke.yaml \
  --local-jsonl tests/fixtures/pretraining_documents.jsonl \
  --force --retrain-tokenizer
```

Inspect the Phase 4B model and run its bounded CPU smoke checks:

```bash
uv run python scripts/inspect_gpt_model.py --config configs/gpt_50m.yaml
```

Run two Phase 4C CPU smoke updates, then resume for the third:

```bash
uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_smoke.yaml \
  --device cpu --max-optimizer-steps 2

uv run python scripts/pretrain_gpt.py \
  --config configs/pretrain_gpt_smoke.yaml \
  --device cpu \
  --resume outputs/pretrain_gpt_smoke/latest.pt \
  --max-optimizer-steps 3
```
