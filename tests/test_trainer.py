"""Tests for the plugin-based Trainer runtime."""

import asyncio
import sys
import tempfile
from enum import Enum
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


async def _run_train_command(trainer: Trainer, max_epochs: int = 1, extra_argv=None):
    trainer.set_state({"max_epochs": max_epochs})
    sys.argv = ["test_trainer", "train", *(extra_argv or [])]
    await trainer.dispatch()


def test_trainer_smoke():
    trainer = _wire_toy_trainer(Trainer())
    asyncio.run(_run_train_command(trainer, max_epochs=1))
    assert trainer.get_state("global_step") > 0


def test_cli_overrides_hyperparams():
    trainer = _wire_toy_trainer(Trainer())
    asyncio.run(
        _run_train_command(
            trainer,
            extra_argv=["--max_epochs", "2", "--lr=0.01"],
        )
    )
    assert trainer.get_state("max_epochs") == 2
    assert trainer.get_state("lr") == pytest.approx(0.01)


def test_cli_unknown_flag_is_ignored():
    trainer = _wire_toy_trainer(Trainer())
    asyncio.run(
        _run_train_command(trainer, max_epochs=1, extra_argv=["--nope", "x"])
    )
    assert trainer.get_state("global_step") > 0


def test_set_state_kwargs_forms():
    trainer = Trainer()
    trainer.set_state(name="foo", value=42)
    assert trainer.get_state("foo") == 42
    trainer.set_state("bar", "baz")
    assert trainer.get_state("bar") == "baz"
    trainer.set_state({"a": 1}, b=2)
    assert trainer.get_state("a") == 1
    assert trainer.get_state("b") == 2


def test_custom_coerce_list_and_literal():
    from typing import List, Literal

    from coffeetrain.trainer import custom_coerce

    assert custom_coerce(["1", "2"], List[int]) == [1, 2]
    assert custom_coerce("1", Literal[1, 2]) == 1
    with pytest.raises(ValueError, match="not one of"):
        custom_coerce("3", Literal[1, 2])


def test_override_block_replaces_body():
    from coffeetrain import Plugin

    plugin = Plugin(name="override_test", description="test")

    @plugin.event()
    class OverrideTestEvents(str, Enum):
        THING = "OVERRIDE_TEST_THING"

    calls = {"before": 0, "after": 0}
    ran = []

    @plugin.system("OVERRIDE_TEST_THING_BEFORE")
    def thing_before():
        calls["before"] += 1

    @plugin.system("OVERRIDE_TEST_THING_AFTER")
    def thing_after():
        calls["after"] += 1

    over = Plugin(name="override_over", description="test")

    @over.override_block("OVERRIDE_TEST_THING")
    def thing_override():
        ran.append("override")

    trainer = Trainer()
    trainer.register_plugin([plugin, over])

    async def _run():
        await trainer.execution_block("OVERRIDE_TEST_THING", call=True)

    asyncio.run(_run())
    assert ran == ["override"]
    # An override replaces the complete block while preserving its hooks.
    assert calls == {"before": 1, "after": 1}


def test_stop_training_stops_loop_gracefully():
    trainer = _wire_toy_trainer(Trainer())

    @trainer.system("BATCH_AFTER")
    def stop_after_first_batch(global_step):
        if global_step >= 1:
            return {"stop_training": True}

    asyncio.run(_run_train_command(trainer, max_epochs=10))
    # 4 batches/epoch at batch_size 8 over 32 samples; stopped after ~2 batches
    assert trainer.get_state("global_step") <= 4


def test_early_stopping_flags_plateau():
    import importlib

    from coffeetrain.plugins.interruptable_train import StopLoop

    # early_stopping is a default plugin: the first trainer already carries
    # the real one. Drive its systems directly instead of re-registering.
    es = importlib.import_module("coffeetrain.plugins.early_stopping").early_stopping

    trainer = _wire_toy_trainer(Trainer())
    trainer.set_state(
        {
            "early_stopping_patience": 1,
            "early_stopping_metric": "eval_loss",
            "early_stopping_mode": "min",
        }
    )
    asyncio.run(_run_train_command(trainer, max_epochs=3))
    # The metric is tracked through EVAL_AFTER. It remains available after
    # FIT_AFTER cleanup so callers can inspect the final run state.
    assert trainer.get_state("early_stopping_best") is not None

    # A flat metric must trigger the stop flag then StopLoop
    trainer2 = Trainer()
    trainer2.set_state(
        {
            "early_stopping_patience": 1,
            "early_stopping_metric": "eval_loss",
            "early_stopping_mode": "min",
            "early_stopping_best": 1.0,
            "early_stopping_bad_epochs": 0,
            "early_stopping_min_delta": 0.0,
            "eval_loss": 1.0,
        }
    )
    eval_end = es.registered_systems["early_stopping_eval_end"]["func"]
    should_stop = es.registered_systems["should_we_stop"]["func"]
    asyncio.run(trainer2._execute(eval_end, **trainer2._collect_kwargs_from_ctx(
        eval_end, es.registered_systems["early_stopping_eval_end"]["sigs"])))
    assert trainer2.get_state("stop_training") is True
    with pytest.raises(StopLoop):
        should_stop(get_state=trainer2.get_state)


def test_signal_handlers_roundtrip():
    import signal as signal_mod

    trainer = Trainer()
    before_sigint = signal_mod.getsignal(signal_mod.SIGINT)
    asyncio.run(_run_train_command(_wire_toy_trainer(trainer), max_epochs=1))
    after_sigint = signal_mod.getsignal(signal_mod.SIGINT)
    assert after_sigint == before_sigint


def test_hf_accelerator_importable_and_guarded():
    import importlib

    hf_accelerator = importlib.import_module(
        "coffeetrain.plugins.hf_accelerator"
    ).hf_accelerator

    assert hf_accelerator.name == "hf_accelerator"
    assert "BATCH" in hf_accelerator.registered_overrides
    prepare = hf_accelerator.registered_systems["prepare_hf_accelerator"]["func"]
    trainer = Trainer()
    try:
        import accelerate  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="accelerate"):
            prepare(
                get_state=trainer.get_state,
                set_state=trainer.set_state,
            )
    else:
        result = prepare(
            get_state=trainer.get_state,
            set_state=trainer.set_state,
        )
        assert result["hf_accelerator"] is not None


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
