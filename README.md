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
character tokens to 691,315 BPE tokens. The Transformer has not yet been
retrained with BPE.

See the [BPE tokenizer report](reports/bpe_tokenizer.md) for the algorithm,
compression results, and limitations.

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
