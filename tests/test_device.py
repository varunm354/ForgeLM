"""Tests for device selection order."""

from unittest.mock import patch

import torch

from forgelm.utils.device import get_device


def test_get_device_prefers_cuda() -> None:
    with (
        patch("forgelm.utils.device.torch.cuda.is_available", return_value=True),
        patch("forgelm.utils.device.torch.backends.mps.is_available", return_value=True),
    ):
        assert get_device() == torch.device("cuda")


def test_get_device_uses_mps_when_cuda_is_unavailable() -> None:
    with (
        patch("forgelm.utils.device.torch.cuda.is_available", return_value=False),
        patch("forgelm.utils.device.torch.backends.mps.is_available", return_value=True),
    ):
        assert get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu() -> None:
    with (
        patch("forgelm.utils.device.torch.cuda.is_available", return_value=False),
        patch("forgelm.utils.device.torch.backends.mps.is_available", return_value=False),
    ):
        assert get_device() == torch.device("cpu")
