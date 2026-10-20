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
seeding and device helpers. Phase 1A adds character-level tokenization,
reproducible language-model batching, and a Tiny Shakespeare download script
for pipeline validation. No model has been implemented or trained yet.

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
