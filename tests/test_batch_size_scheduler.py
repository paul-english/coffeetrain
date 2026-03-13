"""Tests for BatchSizeSchedulerCallback."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from coffeetrain import (
    Trainer,
    Callback,
    BatchSizeSchedulerCallback,
)
from coffeetrain.state import State


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        return self.linear(batch["x"])

    def loss(self, outputs, batch):
        return nn.functional.mse_loss(outputs, batch["y"])


class GradAccumTracker(Callback):
    """Callback to track grad_accum changes."""

    def __init__(self):
        self.grad_accum_history = []

    def batch_end(self, state: State) -> None:
        self.grad_accum_history.append(state.grad_accum)


class TestBatchSizeSchedulerCallback:
    """Tests for the BatchSizeSchedulerCallback."""

    def test_initialization(self):
        callback = BatchSizeSchedulerCallback(
            microbatch_size=96,
            start_batch_size=768,
            final_batch_size=4608,
            warmup_steps=10000,
        )
        assert callback.microbatch_size == 96
        assert callback.start_batch_size == 768
        assert callback.final_batch_size == 4608
        assert callback.warmup_steps == 10000
        assert callback.start_grad_accum == 8
        assert callback.final_grad_accum == 48

    def test_initialization_validation(self):
        with pytest.raises(ValueError, match="start_batch_size.*must be divisible"):
            BatchSizeSchedulerCallback(
                microbatch_size=96,
                start_batch_size=100,
                final_batch_size=192,
                warmup_steps=100,
            )

        with pytest.raises(ValueError, match="final_batch_size.*must be divisible"):
            BatchSizeSchedulerCallback(
                microbatch_size=96,
                start_batch_size=96,
                final_batch_size=100,
                warmup_steps=100,
            )

        with pytest.raises(ValueError, match="start_batch_size.*must be <="):
            BatchSizeSchedulerCallback(
                microbatch_size=96,
                start_batch_size=192,
                final_batch_size=96,
                warmup_steps=100,
            )

    def test_compute_grad_accum_at_boundaries(self):
        callback = BatchSizeSchedulerCallback(
            microbatch_size=96,
            start_batch_size=768,
            final_batch_size=4608,
            warmup_steps=10000,
        )
        assert callback._compute_grad_accum(0) == 8
        assert callback._compute_grad_accum(10000) == 48
        assert callback._compute_grad_accum(100000) == 48

    def test_compute_grad_accum_midpoint(self):
        callback = BatchSizeSchedulerCallback(
            microbatch_size=96,
            start_batch_size=96,
            final_batch_size=960,
            warmup_steps=1000,
        )
        mid_accum = callback._compute_grad_accum(500)
        assert 4 <= mid_accum <= 6

    def test_state_dict(self):
        callback = BatchSizeSchedulerCallback(
            microbatch_size=96,
            start_batch_size=768,
            final_batch_size=4608,
            warmup_steps=10000,
        )
        state = callback.state_dict()

        assert state["microbatch_size"] == 96
        assert state["start_batch_size"] == 768
        assert state["final_batch_size"] == 4608
        assert state["warmup_steps"] == 10000

    def test_integration_with_trainer(self):
        x = torch.randn(100, 10)
        y = torch.randn(100, 2)
        dataset = TensorDataset(x, y)

        def collate_fn(batch):
            xs, ys = zip(*batch)
            return {"x": torch.stack(xs), "y": torch.stack(ys)}

        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        batch_scheduler = BatchSizeSchedulerCallback(
            microbatch_size=4,
            start_batch_size=4,
            final_batch_size=8,
            warmup_steps=10,
        )
        tracker = GradAccumTracker()

        trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            optimizers=optimizer,
            max_epochs=1,
            callbacks=[batch_scheduler, tracker],
            device="cpu",
            progress_bar=False,
        )
        trainer.fit()

        assert len(tracker.grad_accum_history) > 0
        assert tracker.grad_accum_history[0] == 1
        if len(tracker.grad_accum_history) > 10:
            assert tracker.grad_accum_history[-1] == 2


class TestEqualStepsSchedule:
    """Tests for the equal_steps schedule type."""

    def test_equal_steps_distribution(self):
        callback = BatchSizeSchedulerCallback(
            microbatch_size=10,
            start_batch_size=10,
            final_batch_size=50,
            warmup_steps=500,
            schedule_type="equal_steps",
        )
        steps_per_level = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for step in range(500):
            grad_accum = callback._compute_grad_accum(step)
            steps_per_level[grad_accum] += 1
        for level, count in steps_per_level.items():
            assert count == 100, f"Level {level} has {count} steps, expected 100"

