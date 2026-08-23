"""Tests for the plugin-based Trainer runtime."""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from coffeetrain import Trainer, ema_plugin, save_best_model_plugin
from coffeetrain.plugins.batch_size_scheduler import batch_size_scheduler
from coffeetrain.plugins.gradient_accumulator import gradient_accumulator


class TinyModel(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def _wire_toy_trainer(trainer: Trainer, batch_size: int = 8) -> Trainer:
    compute_device = torch.device("cpu")

    @trainer.system("DATA_BEFORE")
    def load_data():
        x = torch.randn(32, 4)
        y = torch.randn(32, 2)
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=batch_size)
        return {
            "train_dataloader": loader,
            "eval_dataloader": loader,
            "loss_fn": nn.MSELoss(),
            "batch_size": batch_size,
            "real_batch_size": batch_size,
            "device": None,
            "compute_device": compute_device,
            "train_metrics": {},
            "is_main_process": True,
        }

    @trainer.system("MODEL_BEFORE")
    def create_model():
        return {"model": TinyModel().to(compute_device)}

    @trainer.system("OPTIMIZER_BEFORE")
    def create_optimizer(model, lr=1e-3):
        return {"optimizer": torch.optim.Adam(model.parameters(), lr=lr)}

    @trainer.system("FORWARD_BEFORE")
    def prepare_forward_batch(batch, compute_device):
        x, y = batch
        return {"inputs": x.to(compute_device), "labels": y.to(compute_device)}

    @trainer.system("FORWARD")
    def forward_pass(model, inputs):
        return {"outputs": model(inputs)}

    @trainer.system("EVAL_FORWARD_BEFORE")
    def prepare_eval_batch(batch, compute_device):
        x, y = batch
        return {"inputs": x.to(compute_device), "labels": y.to(compute_device)}

    @trainer.system("EVAL_FORWARD")
    def eval_forward_pass(model, inputs):
        return {"outputs": model(inputs)}

    @trainer.system("BACKWARD_AFTER")
    def optimizer_step(optimizer):
        optimizer.step()
        optimizer.zero_grad()

    return trainer


async def _run_train_command(trainer: Trainer, max_epochs: int = 1):
    trainer.set_state({"max_epochs": max_epochs})
    sys.argv = ["test_trainer", "train"]
    await trainer.dispatch()


def test_trainer_smoke():
    trainer = _wire_toy_trainer(Trainer())
    asyncio.run(_run_train_command(trainer, max_epochs=1))
    assert trainer.get_state("global_step") > 0


def test_save_best_model_plugin_writes_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _wire_toy_trainer(Trainer())
        trainer.register_plugin(save_best_model_plugin)
        trainer.set_state(
            {
                "save_best_model_dir": tmpdir,
                "save_best_model_metric": "loss",
                "save_best_model_mode": "min",
                "save_best_model_save_steps": 1,
            }
        )
        asyncio.run(_run_train_command(trainer, max_epochs=1))

        checkpoint = Path(tmpdir) / "best_model.pt"
        assert checkpoint.exists()
        assert checkpoint.stat().st_size > 0


def test_ema_plugin_init_creates_shadow_model():
    init_fn = ema_plugin.registered_systems["init_ema_model"]["func"]
    model = TinyModel()
    result = init_fn(model=model, device=torch.device("cpu"))
    assert result["ema_model"] is not None


def test_gradient_accumulator_init():
    init_fn = gradient_accumulator.registered_systems["init_grad_accum"]["func"]
    result = init_fn(batch_size=768, real_batch_size=96)
    assert result["grad_accum"] == 8

    with pytest.raises(ValueError, match="must be divisible"):
        init_fn(batch_size=100, real_batch_size=96)


def test_batch_size_scheduler_init():
    init_fn = batch_size_scheduler.registered_systems["init_batch_size_scheduler"]["func"]
    result = init_fn(
        batch_size=768,
        real_batch_size=96,
        start_batch_size=768,
        final_batch_size=4608,
        batch_size_warmup_steps=100,
    )
    assert result["batch_size_start_grad_accum"] == 8
    assert result["batch_size_final_grad_accum"] == 48

    with pytest.raises(ValueError, match="start_batch_size.*must be divisible"):
        init_fn(
            batch_size=768,
            real_batch_size=96,
            start_batch_size=100,
            final_batch_size=4608,
        )

    with pytest.raises(ValueError, match="final_batch_size.*must be divisible"):
        init_fn(
            batch_size=768,
            real_batch_size=96,
            start_batch_size=768,
            final_batch_size=100,
        )

    with pytest.raises(ValueError, match="start_batch_size.*must be <="):
        init_fn(
            batch_size=768,
            real_batch_size=96,
            start_batch_size=4608,
            final_batch_size=768,
        )
