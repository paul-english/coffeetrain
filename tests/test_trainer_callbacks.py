"""Generic Trainer + callback tests for coffeetrain."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from coffeetrain import (
    BestModelCheckpointer,
    CosineWarmupScheduler,
    EMACallback,
    HistoryCallback,
    Trainer,
)


class ToyDataset(Dataset):
    """Small deterministic dataset."""

    def __init__(self, n: int = 48, in_dim: int = 6, out_dim: int = 2, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        # Linear-ish target with small noise
        w = torch.randn(in_dim, out_dim, generator=g)
        self.y = self.x @ w + 0.01 * torch.randn(n, out_dim, generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return {"x": self.x[idx], "y": self.y[idx]}


def collate_fn(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


class SimpleModel(nn.Module):
    """Model exposing Trainer protocol methods."""

    def __init__(self, in_dim: int = 6, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, batch):
        return {"preds": self.net(batch["x"])}

    def loss(self, outputs, batch):
        return nn.functional.mse_loss(outputs["preds"], batch["y"])


def make_loaders(batch_size: int = 8):
    train_ds = ToyDataset(n=64, seed=1)
    eval_ds = ToyDataset(n=24, seed=2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, eval_loader


def test_trainer_runs_and_updates_state():
    model = SimpleModel()
    train_loader, _ = make_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizers=optimizer,
        max_epochs=2,
        device="cpu",
        progress_bar=False,
    )
    state = trainer.fit()

    assert state.global_step > 0
    assert state.epoch == 1  # zero-indexed after 2 epochs


def test_history_callback_persists_metrics():
    model = SimpleModel()
    train_loader, eval_loader = make_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        history = HistoryCallback(save_dir=tmpdir)
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizers=optimizer,
            max_epochs=2,
            callbacks=[history],
            device="cpu",
            progress_bar=False,
        )
        trainer.fit()

        history_file = Path(tmpdir) / "history.json"
        assert history_file.exists()
        assert len(history.history["train_loss"]) == 2
        assert len(history.history["val_loss"]) >= 1


def test_best_model_checkpointer_saves_file():
    model = SimpleModel()
    train_loader, eval_loader = make_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = BestModelCheckpointer(save_dir=tmpdir, metric_name="loss", mode="min")
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            optimizers=optimizer,
            max_epochs=2,
            callbacks=[ckpt],
            device="cpu",
            progress_bar=False,
        )
        trainer.fit()

        checkpoint_path = Path(tmpdir) / "best_model.pt"
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0


def test_ema_callback_creates_shadow_model():
    model = SimpleModel()
    train_loader, _ = make_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMACallback(decay=0.99)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizers=optimizer,
        max_epochs=2,
        callbacks=[ema],
        device="cpu",
        progress_bar=False,
    )
    trainer.fit()

    assert ema.ema_model is not None


def test_scheduler_integration_updates_lr():
    model = SimpleModel()
    train_loader, _ = make_loaders(batch_size=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    total_steps = len(train_loader) * 2
    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup_steps=1, total_steps=total_steps, min_lr=1e-5)
    initial_lr = optimizer.param_groups[0]["lr"]

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_epochs=2,
        device="cpu",
        progress_bar=False,
    )
    trainer.fit()

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr != initial_lr
    assert final_lr >= 1e-5

