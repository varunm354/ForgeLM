"""Tests for reproducible seeding."""

import random

import numpy as np
import torch

from forgelm.utils.seed import set_seed


def test_set_seed_reproduces_python_numpy_and_torch() -> None:
    set_seed(7)
    python_a = [random.random() for _ in range(3)]
    numpy_a = np.random.rand(3)
    torch_a = torch.rand(3)

    set_seed(7)
    python_b = [random.random() for _ in range(3)]
    numpy_b = np.random.rand(3)
    torch_b = torch.rand(3)

    assert python_a == python_b
    np.testing.assert_array_equal(numpy_a, numpy_b)
    assert torch.equal(torch_a, torch_b)


def test_different_seeds_change_torch_samples() -> None:
    set_seed(1)
    first = torch.rand(4)
    set_seed(2)
    second = torch.rand(4)

    assert not torch.equal(first, second)
