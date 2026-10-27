"""Cloud-ready GPT pretraining loop over memory-mapped token shards."""

import json
import math
import os
import random
import signal
import subprocess
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from forgelm.data.token_shards import MemoryMappedShardDataset
from forgelm.models.gpt import GPTConfig, GPTLanguageModel, parameter_report
from forgelm.training.pretrain_checkpoint import (
    TrainingProgress,
    capture_rank_state,
    load_exact_checkpoint,
    load_weights_only,
    save_pretrain_checkpoint,
)
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_runtime import (
    DistributedContext,
    ShardBatchSampler,
    create_adamw,
    learning_rate_at_step,
    select_precision,
    validate_training_compatibility,
)


class PretrainingEngine:
    """Own model, samplers, optimizer, evaluation, checkpoints, and metrics."""

    def __init__(
        self,
        config: PretrainConfig,
        distributed: DistributedContext,
        *,
        resume_checkpoint: Path | None = None,
        weights_only_checkpoint: Path | None = None,
    ) -> None:
        if resume_checkpoint is not None and weights_only_checkpoint is not None:
            raise ValueError("Exact resume and weights-only load are mutually exclusive.")
        self.config = config
        self.distributed = distributed
        self.device = distributed.device
        if config.warmup_steps > config.optimizer_steps(distributed.world_size):
            raise ValueError(
                "warmup_steps cannot exceed calculated optimizer steps."
            )
        self.model_config = GPTConfig.load(config.model_config_path)
        self.manifest, self.manifest_identity = validate_training_compatibility(
            config.manifest_path,
            self.model_config,
            config,
        )
        self.tokenizer_fingerprint = self.manifest["tokenizer"]["fingerprint"]
        self.precision = select_precision(config.precision, self.device)
        self.scaler = self.precision.create_grad_scaler()
        self._seed_process(config.seed + distributed.rank)

        self.training_dataset = MemoryMappedShardDataset(
            config.manifest_path,
            split="train",
            block_size=config.sequence_length,
            expected_vocabulary_size=self.model_config.vocab_size,
            expected_tokenizer_fingerprint=self.tokenizer_fingerprint,
            verify_checksums=config.verify_checksums,
        )
        self.validation_dataset = MemoryMappedShardDataset(
            config.manifest_path,
            split="validation",
            block_size=config.sequence_length,
            expected_vocabulary_size=self.model_config.vocab_size,
            expected_tokenizer_fingerprint=self.tokenizer_fingerprint,
            verify_checksums=config.verify_checksums,
        )
        rank_seed_offset = distributed.rank * 100_003
        self.training_sampler = ShardBatchSampler(
            self.training_dataset,
            batch_size=config.micro_batch_size,
            seed=config.seed + rank_seed_offset,
        )
        self.validation_sampler = ShardBatchSampler(
            self.validation_dataset,
            batch_size=config.micro_batch_size,
            seed=config.seed + 1_000_000 + rank_seed_offset,
        )

        self.model = GPTLanguageModel(self.model_config).to(self.device)
        optimizer_setup = create_adamw(
            self.model,
            config,
            device=self.device,
        )
        self.optimizer = optimizer_setup.optimizer
        self.optimizer_fused = optimizer_setup.fused
        self.progress = TrainingProgress()
        resume_status = "fresh"
        if resume_checkpoint is not None:
            self.progress = load_exact_checkpoint(
                resume_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                training_config=config,
                model_config=self.model_config,
                manifest_identity=self.manifest_identity,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                training_sampler=self.training_sampler,
                validation_sampler=self.validation_sampler,
                rank=distributed.rank,
                world_size=distributed.world_size,
                precision=self.precision.name,
                device=self.device,
            )
            resume_status = f"exact resume from {resume_checkpoint}"
        elif weights_only_checkpoint is not None:
            load_weights_only(
                weights_only_checkpoint,
                model=self.model,
                model_config=self.model_config,
                device=self.device,
            )
            resume_status = f"weights only from {weights_only_checkpoint}"

        self.execution_model: nn.Module = self._prepare_execution_model()
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.started_at = time.perf_counter()
        self.resume_status = resume_status
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        if self.distributed.is_rank_zero:
            self._write_run_metadata()

    @property
    def total_optimizer_steps(self) -> int:
        return self.config.optimizer_steps(self.distributed.world_size)

    @property
    def effective_batch_tokens(self) -> int:
        return self.config.effective_batch_tokens(self.distributed.world_size)

    def install_signal_handlers(self) -> None:
        """Request a checkpoint after the current optimizer-step boundary."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def run(self, *, maximum_optimizer_steps: int | None = None) -> dict[str, Any]:
        """Train to the configured or explicit absolute optimizer-step limit."""
        limit = self.total_optimizer_steps
        if maximum_optimizer_steps is not None:
            if (
                not isinstance(maximum_optimizer_steps, int)
                or isinstance(maximum_optimizer_steps, bool)
                or maximum_optimizer_steps <= 0
            ):
                raise ValueError("maximum_optimizer_steps must be positive.")
            limit = min(limit, maximum_optimizer_steps)
        if limit <= self.progress.optimizer_step:
            raise ValueError(
                "Requested optimizer-step limit has already been reached."
            )

        last_training_loss = float("nan")
        last_validation_loss: float | None = None
        last_validation_perplexity: float | None = None
        last_gradient_norm = float("nan")
        while self.progress.optimizer_step < limit:
            step_started = time.perf_counter()
            last_training_loss, last_gradient_norm, learning_rate = (
                self.train_one_optimizer_step()
            )
            step_time = time.perf_counter() - step_started
            last_validation_loss = None
            last_validation_perplexity = None
            should_validate = (
                self.progress.optimizer_step % self.config.validation_interval == 0
                or self.progress.optimizer_step == limit
            )
            if should_validate:
                last_validation_loss, last_validation_perplexity = self.evaluate()
                if last_validation_loss < self.progress.best_validation_loss:
                    self.progress.best_validation_loss = last_validation_loss
                    self._save_collective(
                        self.config.output_directory
                        / self.config.best_checkpoint_name
                    )

            should_log = (
                self.progress.optimizer_step % self.config.logging_interval == 0
                or should_validate
                or self.progress.optimizer_step == limit
            )
            if should_log:
                self._write_metrics(
                    training_loss=last_training_loss,
                    validation_loss=last_validation_loss,
                    validation_perplexity=last_validation_perplexity,
                    gradient_norm=last_gradient_norm,
                    learning_rate=learning_rate,
                    step_time=step_time,
                )
                self._console_status(
                    last_training_loss,
                    last_validation_loss,
                    learning_rate,
                    last_gradient_norm,
                    step_time,
                )

            if (
                self.progress.optimizer_step % self.config.checkpoint_interval
                == 0
            ):
                self._save_collective(
                    self.config.output_directory
                    / f"step_{self.progress.optimizer_step:08d}.pt"
                )
                self._save_collective(
                    self.config.output_directory
                    / self.config.latest_checkpoint_name
                )

            if self._synchronized_stop_requested():
                self._save_collective(
                    self.config.output_directory
                    / f"interrupt_step_{self.progress.optimizer_step:08d}.pt"
                )
                break

        self._save_collective(
            self.config.output_directory / self.config.latest_checkpoint_name
        )
        return {
            "optimizer_step": self.progress.optimizer_step,
            "micro_step": self.progress.micro_step,
            "tokens_processed": self.progress.tokens_processed,
            "training_loss": last_training_loss,
            "validation_loss": last_validation_loss,
            "validation_perplexity": last_validation_perplexity,
            "gradient_norm": last_gradient_norm,
        }

    def train_one_optimizer_step(self) -> tuple[float, float, float]:
        """Accumulate micro-batches and perform one clipped optimizer update."""
        self.execution_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        next_optimizer_step = self.progress.optimizer_step + 1
        learning_rate = learning_rate_at_step(
            next_optimizer_step,
            total_steps=self.total_optimizer_steps,
            warmup_steps=self.config.warmup_steps,
            maximum=self.config.learning_rate,
            minimum=self.config.minimum_learning_rate,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        raw_loss_total = 0.0
        for accumulation_index in range(
            self.config.gradient_accumulation_steps
        ):
            inputs, targets = self.training_sampler.next_batch()
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            targets = targets.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            synchronize = (
                accumulation_index
                == self.config.gradient_accumulation_steps - 1
            )
            sync_context = (
                nullcontext()
                if synchronize or not hasattr(self.execution_model, "no_sync")
                else self.execution_model.no_sync()  # type: ignore[union-attr]
            )
            with sync_context, self.precision.autocast(self.device):
                _, loss = self.execution_model(inputs, targets)
                if loss is None:
                    raise RuntimeError("GPT did not return a training loss.")
                scaled_loss = loss / self.config.gradient_accumulation_steps
            raw_loss_total += float(loss.detach())
            if self.scaler is None:
                scaled_loss.backward()
            else:
                self.scaler.scale(scaled_loss).backward()
            self.progress.micro_step += 1
            self.progress.tokens_processed += (
                inputs.numel() * self.distributed.world_size
            )

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clipping,
        )
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.progress.optimizer_step = next_optimizer_step
        mean_loss = raw_loss_total / self.config.gradient_accumulation_steps
        return (
            self._distributed_mean(mean_loss),
            self._distributed_mean(float(gradient_norm)),
            learning_rate,
        )

    def evaluate(self) -> tuple[float, float]:
        """Evaluate fixed sampler batches without perturbing either sampler."""
        was_training = self.execution_model.training
        validation_state = self.validation_sampler.state_dict()
        total_negative_log_likelihood = 0.0
        total_tokens = 0
        self.execution_model.eval()
        try:
            with torch.inference_mode():
                for _ in range(self.config.validation_batches):
                    inputs, targets = self.validation_sampler.next_batch()
                    inputs = inputs.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    targets = targets.to(
                        self.device,
                        non_blocking=self.device.type == "cuda",
                    )
                    with self.precision.autocast(self.device):
                        _, loss = self.execution_model(inputs, targets)
                    if loss is None:
                        raise RuntimeError("GPT did not return validation loss.")
                    total_negative_log_likelihood += (
                        float(loss) * targets.numel()
                    )
                    total_tokens += targets.numel()
        finally:
            if self.config.fixed_validation_batches:
                self.validation_sampler.load_state_dict(validation_state)
            self.execution_model.train(was_training)

        totals = torch.tensor(
            [total_negative_log_likelihood, total_tokens],
            dtype=torch.float64,
            device=self.device,
        )
        if self.distributed.distributed:
            torch.distributed.all_reduce(
                totals,
                op=torch.distributed.ReduceOp.SUM,
            )
        mean_loss = float(totals[0] / totals[1])
        perplexity = math.exp(mean_loss) if mean_loss < 700 else float("inf")
        return mean_loss, perplexity

    def _prepare_execution_model(self) -> nn.Module:
        execution_model: nn.Module = self.model
        if self.config.torch_compile:
            try:
                execution_model = torch.compile(self.model)
            except Exception:
                if not self.config.compile_allow_fallback:
                    raise
                self._rank_zero_print(
                    "warning: torch.compile setup failed; using eager mode"
                )
        if self.distributed.distributed:
            device_ids = (
                [self.distributed.local_rank]
                if self.device.type == "cuda"
                else None
            )
            execution_model = DistributedDataParallel(
                execution_model,
                device_ids=device_ids,
            )
        return execution_model

    def _save_collective(self, path: Path) -> None:
        local_state = capture_rank_state(
            self.training_sampler,
            self.validation_sampler,
        )
        if self.distributed.distributed:
            gathered: list[Any] | None = (
                [None] * self.distributed.world_size
                if self.distributed.is_rank_zero
                else None
            )
            torch.distributed.gather_object(
                local_state,
                gathered,
                dst=0,
            )
            rank_states = gathered
        else:
            rank_states = [local_state]
        if self.distributed.is_rank_zero:
            if rank_states is None:
                raise RuntimeError("Rank-zero checkpoint states were not gathered.")
            save_pretrain_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                progress=self.progress,
                training_config=self.config,
                model_config=self.model_config,
                manifest_identity=self.manifest_identity,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                rank_states=rank_states,
                world_size=self.distributed.world_size,
                precision=self.precision.name,
            )

    def _write_metrics(
        self,
        *,
        training_loss: float,
        validation_loss: float | None,
        validation_perplexity: float | None,
        gradient_norm: float,
        learning_rate: float,
        step_time: float,
    ) -> None:
        if not self.distributed.is_rank_zero:
            return
        elapsed = time.perf_counter() - self.started_at
        record = {
            "optimizer_step": self.progress.optimizer_step,
            "micro_step": self.progress.micro_step,
            "tokens_processed": self.progress.tokens_processed,
            "learning_rate": learning_rate,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_perplexity": validation_perplexity,
            "gradient_norm": gradient_norm,
            "elapsed_seconds": elapsed,
            "step_time_seconds": step_time,
            "tokens_per_second": self.effective_batch_tokens / step_time,
            "device_type": self.device.type,
            "precision": self.precision.name,
            "world_size": self.distributed.world_size,
        }
        metrics_path = (
            self.config.output_directory / self.config.metrics_filename
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _write_run_metadata(self) -> None:
        report = parameter_report(self.model)
        metadata = {
            "training_config": self.config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "model_parameters": report.total_parameters,
            "manifest_identity": self.manifest_identity,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "manifest_path": str(self.config.manifest_path),
            "device": str(self.device),
            "device_type": self.device.type,
            "pytorch_version": torch.__version__,
            "world_size": self.distributed.world_size,
            "precision": self.precision.name,
            "start_time_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "resume_status": self.resume_status,
        }
        _write_json_atomic(
            self.config.output_directory / self.config.metadata_filename,
            metadata,
        )

    def _console_status(
        self,
        training_loss: float,
        validation_loss: float | None,
        learning_rate: float,
        gradient_norm: float,
        step_time: float,
    ) -> None:
        if not self.distributed.is_rank_zero:
            return
        validation = (
            "n/a" if validation_loss is None else f"{validation_loss:.4f}"
        )
        print(
            f"step {self.progress.optimizer_step:6d} | "
            f"micro {self.progress.micro_step:7d} | "
            f"tokens {self.progress.tokens_processed:12,d} | "
            f"train {training_loss:.4f} | val {validation} | "
            f"lr {learning_rate:.3e} | grad {gradient_norm:.3f} | "
            f"{step_time:.3f}s"
        )

    def _distributed_mean(self, value: float) -> float:
        if not self.distributed.distributed:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return float(tensor / self.distributed.world_size)

    def _synchronized_stop_requested(self) -> bool:
        requested = torch.tensor(
            int(self.stop_requested),
            dtype=torch.int32,
            device=self.device,
        )
        if self.distributed.distributed:
            torch.distributed.all_reduce(
                requested,
                op=torch.distributed.ReduceOp.MAX,
            )
        return bool(requested.item())

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.stop_signal = signum

    def _rank_zero_print(self, message: str) -> None:
        if self.distributed.is_rank_zero:
            print(message)

    @staticmethod
    def _seed_process(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
