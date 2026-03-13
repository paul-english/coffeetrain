"""Weights & Biases logging callback."""

import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from coffeetrain.callback import Callback
from coffeetrain.state import State


class WandbCallback(Callback):
    """Log training metrics to Weights & Biases.

    Logs training loss, learning rates, component losses, and validation
    metrics to W&B for experiment tracking and visualization.

    Features:
    - Metric logging with configurable intervals
    - Proper x-axis alignment via define_metric()
    - Summary statistics (min/max/best) for key metrics
    - Gradient/parameter histogram tracking via wandb.watch()
    - Loss spike and NaN alerts
    - Prediction tables for error analysis (via callback_state)

    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Pydantic config model or dict of hyperparameters to log
        log_interval: Log training metrics every N batches
        log_model: Whether to save model artifacts to W&B
        resume: Whether to resume from previous run if available
        debug_log: Write timestamped local JSONL file for debugging
        watch_model: Enable gradient/parameter histograms
        watch_log: What to log: "gradients", "parameters", or "all"
        watch_log_freq: Log gradients/params every N batches
        enable_alerts: Enable loss spike and NaN alerts
        log_predictions: Log prediction tables during evaluation
        max_prediction_samples: Max prediction samples to log

    Example:
        >>> callback = WandbCallback(
        ...     project="my_project",
        ...     config=my_training_config,
        ...     log_interval=10,
        ...     watch_model=True,
        ...     enable_alerts=True,
        ... )
        >>> trainer = Trainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        project: str = "coffeetrain",
        name: Optional[str] = None,
        config: Optional[Union[Any, Dict[str, Any]]] = None,
        log_interval: int = 10,
        log_model: bool = False,
        resume: bool = True,
        debug_log: bool = False,
        # Gradient/weight tracking
        watch_model: bool = False,
        watch_log: str = "gradients",
        watch_log_freq: int = 100,
        # Alerts
        enable_alerts: bool = True,
        # Prediction logging
        log_predictions: bool = False,
        max_prediction_samples: int = 50,
    ):
        self.project = project
        self.name = name
        # Convert Pydantic model to dict for wandb
        if config is None:
            self.config = {}
        elif hasattr(config, "model_dump"):
            self.config = config.model_dump()
        else:
            self.config = config if isinstance(config, dict) else {}
        self.log_interval = log_interval
        self.log_model = log_model
        self.resume_enabled = resume
        self.debug_log = debug_log

        self.watch_model = watch_model
        self.watch_log = watch_log
        self.watch_log_freq = watch_log_freq
        self.enable_alerts = enable_alerts
        self.log_predictions = log_predictions
        self.max_prediction_samples = max_prediction_samples

        self.run = None
        self.run_id: Optional[str] = None
        self._wandb = None  # Lazy import
        self._debug_file: Optional[TextIO] = None
        self._debug_path: Optional[Path] = None

        # Loss tracking for alerts and best metrics
        self._loss_window: deque = deque(maxlen=100)
        self._best_train_loss: Optional[float] = None
        self._best_val_loss: Optional[float] = None

    def _get_wandb(self):
        """Lazy import wandb to avoid import errors when not installed."""
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "wandb is not installed. Install it with: pip install wandb"
                )
        return self._wandb

    def init(self, state: State) -> None:
        """Initialize wandb run."""
        if not state.is_main_process:
            return

        wandb = self._get_wandb()

        # Check if W&B run already active (e.g., from init_run_directory)
        if wandb.run is not None:
            self.run = wandb.run
            self.run_id = self.run.id
            self._define_metrics(wandb)

            if self.debug_log:
                self._debug_path = Path(f"wandb_debug_{self.run_id}.jsonl")
                self._debug_file = open(self._debug_path, "a")
                print(f"[WandbCallback] Debug logging enabled: {self._debug_path}")

            print(f"[WandbCallback] Attached to existing run: {self.run.name}")
            return

        # Determine resume mode
        resume_mode = None
        if self.resume_enabled and self.run_id:
            resume_mode = "allow"

        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            id=self.run_id,
            resume=resume_mode,
        )
        self.run_id = self.run.id

        self._define_metrics(wandb)

        if self.debug_log:
            self._debug_path = Path(f"wandb_debug_{self.run_id}.jsonl")
            self._debug_file = open(self._debug_path, "a")
            print(f"[WandbCallback] Debug logging enabled: {self._debug_path}")

    def _define_metrics(self, wandb) -> None:
        """Configure metrics for proper x-axis alignment and summary stats."""
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")

        wandb.define_metric("train/loss", summary="min", goal="minimize")
        wandb.define_metric("val/loss", summary="min", goal="minimize")
        wandb.define_metric("train/lr", summary="last")

        wandb.define_metric("train/classification_loss", summary="min")
        wandb.define_metric("train/structure_loss", summary="min")
        wandb.define_metric("train/count_loss", summary="min")

        wandb.define_metric("train/samples_per_sec", summary="mean")
        wandb.define_metric("train/epoch_time", summary="last")

    def fit_start(self, state: State) -> None:
        """Start of training - enable gradient/parameter tracking."""
        if not state.is_main_process:
            return
        if self.watch_model and self.run:
            self._wandb.watch(
                state.model,
                log=self.watch_log,
                log_freq=self.watch_log_freq,
                log_graph=True,
            )

    def _write_debug_log(self, metrics: Dict[str, Any], step: int) -> None:
        """Write metrics to debug log file."""
        if self._debug_file is None:
            return

        record = {
            "ts": datetime.now().isoformat(),
            "step": step,
            **metrics,
        }
        self._debug_file.write(json.dumps(record) + "\n")
        self._debug_file.flush()

    def batch_end(self, state: State) -> None:
        """Log training metrics at specified interval."""
        if self.run is None:
            return

        if state.loss is not None:
            loss_val = state.loss.item()
            self._track_loss(loss_val, state.global_step)

        if state.global_step % self.log_interval != 0:
            return

        metrics: Dict[str, Any] = {
            "global_step": state.global_step,
        }

        if state.loss is not None:
            metrics["train/loss"] = state.loss.item()

        lr = state.get_lr()
        if lr is not None:
            metrics["train/lr"] = lr

        for key, value in state.train_metrics.items():
            metrics[f"train/{key}"] = value

        self._write_debug_log(metrics, state.global_step)

        self.run.log(metrics, step=state.global_step)

    def _track_loss(self, loss_val: float, step: int) -> None:
        """Track loss for alerts and best metrics."""
        if not math.isfinite(loss_val):
            if self.enable_alerts and self.run:
                self.run.alert(
                    title="NaN/Inf Loss Detected",
                    text=f"Loss became {loss_val} at step {step}",
                    level="ERROR",
                )
            return

        if self._best_train_loss is None or loss_val < self._best_train_loss:
            self._best_train_loss = loss_val

        if self._loss_window:
            avg_loss = sum(self._loss_window) / len(self._loss_window)
            if self.enable_alerts and loss_val > 10 * avg_loss and self.run:
                self.run.alert(
                    title="Loss Spike Detected",
                    text=f"Loss {loss_val:.4f} > 10x avg {avg_loss:.4f} at step {step}",
                    level="WARN",
                    wait_duration=300,
                )

        self._loss_window.append(loss_val)

    def epoch_end(self, state: State) -> None:
        """Log epoch marker."""
        if self.run is None:
            return

        self.run.log({"epoch": state.epoch}, step=state.global_step)

    def eval_end(self, state: State) -> None:
        """Log validation metrics and prediction tables."""
        if self.run is None:
            return

        metrics: Dict[str, Any] = {}
        for key, value in state.eval_metrics.items():
            metrics[f"val/{key}"] = value

            if key == "loss" and isinstance(value, (int, float)):
                if self._best_val_loss is None or value < self._best_val_loss:
                    self._best_val_loss = value

        if metrics:
            self.run.log(metrics, step=state.global_step)

        if self.log_predictions:
            self._log_prediction_table(state)

    def _log_prediction_table(self, state: State) -> None:
        """Log prediction samples as a wandb Table."""
        if self.run is None:
            return

        predictions = state.callback_state.get("predictions", [])
        if not predictions:
            return

        wandb = self._get_wandb()
        table = wandb.Table(
            columns=["text", "ground_truth", "prediction", "correct"],
            data=predictions[: self.max_prediction_samples],
        )
        self.run.log({"val/predictions": table}, step=state.global_step)

    def fit_end(self, state: State) -> None:
        """Finish wandb run with summary metrics."""
        if self._debug_file is not None:
            self._debug_file.close()
            self._debug_file = None
            print(f"[WandbCallback] Debug log saved: {self._debug_path}")

        if self.run is not None:
            self.run.summary["final_epoch"] = state.epoch
            self.run.summary["total_steps"] = state.global_step

            if self._best_train_loss is not None:
                self.run.summary["best_train_loss"] = self._best_train_loss
            if self._best_val_loss is not None:
                self.run.summary["best_val_loss"] = self._best_val_loss

            self.run.finish()
            self.run = None

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "run_id": self.run_id,
            "best_train_loss": self._best_train_loss,
            "best_val_loss": self._best_val_loss,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.run_id = state_dict.get("run_id")
        self._best_train_loss = state_dict.get("best_train_loss")
        self._best_val_loss = state_dict.get("best_val_loss")
