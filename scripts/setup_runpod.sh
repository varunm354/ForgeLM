#!/usr/bin/env bash
set -euo pipefail

# Run from a public ForgeLM clone inside a RunPod PyTorch template.
# The template is responsible for supplying a CUDA-enabled PyTorch runtime.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

uv sync --frozen
uv run python -c \
  'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})'

if [[ "${FORGELM_VERIFY_CUDA:-1}" == "1" ]]; then
  uv run python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"'
fi
