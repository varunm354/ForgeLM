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
evaluated on Tiny Shakespeare. Phase 2A is now implementing a single causal
self-attention head with explicit attention mathematics. A complete Transformer
has not been implemented or trained.

## Bigram baseline results

| Metric | Initial | Final |
| --- | ---: | ---: |
| Training loss / perplexity | 4.7455 / 115.06 | 2.4546 / 11.64 |
| Validation loss / perplexity | 4.7017 / 110.13 | 2.4597 / 11.70 |

The 4,225-parameter model trained for 3,000 steps on CPU in 9.51 seconds. See
the [complete baseline report](reports/bigram_baseline.md) for configuration,
interpretation, and generated text.

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
