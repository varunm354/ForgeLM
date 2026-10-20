"""Tests for training checkpoints."""

from pathlib import Path

import torch

from forgelm.models.bigram import BigramLanguageModel
from forgelm.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    model = BigramLanguageModel(vocab_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    input_ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 0]])
    _, loss = model(input_ids, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()

    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        step=12,
        config={"learning_rate": 0.01},
        metrics={"validation_loss": 1.25},
    )

    restored_model = BigramLanguageModel(vocab_size=3)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    checkpoint = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        device="cpu",
    )

    assert path.exists()
    assert checkpoint["step"] == 12
    assert checkpoint["config"] == {"learning_rate": 0.01}
    assert checkpoint["metrics"] == {"validation_loss": 1.25}
    assert restored_optimizer.state_dict()["state"]
    for name, parameter in restored_model.named_parameters():
        assert parameter.device.type == "cpu"
        assert torch.equal(parameter, expected_parameters[name])
