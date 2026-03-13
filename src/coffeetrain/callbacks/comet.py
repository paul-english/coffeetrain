"""Comet.ml logging callback."""

from typing import Any, Dict, Optional, Union

from coffeetrain.callback import Callback
from coffeetrain.state import State


class CometCallback(Callback):
    """Log training metrics to Comet.ml.

    Logs training loss, learning rates, component losses, and validation
    metrics to Comet for experiment tracking and visualization.

    This is a simpler alternative to WandbCallback that focuses on core
    metrics logging without alerts, prediction tables, or model watching.

    Args:
        project: Comet project name
        workspace: Comet workspace (uses default if None)
        name: Experiment name (auto-generated if None)
        config: Pydantic config model or dict of hyperparameters to log
        log_interval: Log training metrics every N batches
        resume: Whether to resume from previous experiment if available

    Example:
        >>> callback = CometCallback(
        ...     project="my_project",
        ...     config=my_training_config,
        ...     log_interval=10,
        ... )
        >>> trainer = Trainer(model, callbacks=[callback])
    """

    def __init__(
        self,
        project: str = "coffeetrain",
        workspace: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Union[Any, Dict[str, Any]]] = None,
        log_interval: int = 10,
        resume: bool = True,
    ):
        self.project = project
        self.workspace = workspace
        self.name = name
        # Convert Pydantic model to dict for comet
        if config is None:
            self.config = {}
        elif hasattr(config, "model_dump"):
            self.config = config.model_dump()
        else:
            self.config = config if isinstance(config, dict) else {}
        self.log_interval = log_interval
        self.resume_enabled = resume

        self.experiment = None
        self.experiment_key: Optional[str] = None
        self._comet = None  # Lazy import

        # Loss tracking for best metrics
        self._best_train_loss: Optional[float] = None
        self._best_val_loss: Optional[float] = None

    def _get_comet(self):
        """Lazy import comet_ml to avoid import errors when not installed."""
        if self._comet is None:
            try:
                import comet_ml

                self._comet = comet_ml
            except ImportError:
                raise ImportError(
                    "comet_ml is not installed. Install it with: pip install comet-ml"
                )
        return self._comet

    def init(self, state: State) -> None:
        """Initialize comet experiment."""
        comet = self._get_comet()

        # Create or resume experiment
        if self.resume_enabled and self.experiment_key:
            self.experiment = comet.ExistingExperiment(
                previous_experiment=self.experiment_key,
                project_name=self.project,
                workspace=self.workspace,
            )
        else:
            self.experiment = comet.Experiment(
                project_name=self.project,
                workspace=self.workspace,
            )
            if self.name:
                self.experiment.set_name(self.name)

        self.experiment_key = self.experiment.get_key()

        # Log hyperparameters
        if self.config:
            self.experiment.log_parameters(self.config)

    def batch_end(self, state: State) -> None:
        """Log training metrics at specified interval."""
        if self.experiment is None:
            return

        # Only log at specified interval
        if state.global_step % self.log_interval != 0:
            return

        # Training loss
        if state.loss is not None:
            loss_val = state.loss.item()
            self.experiment.log_metric(
                "train/loss", loss_val, step=state.global_step
            )

            # Track best train loss
            if self._best_train_loss is None or loss_val < self._best_train_loss:
                self._best_train_loss = loss_val

        # Learning rate
        lr = state.get_lr()
        if lr is not None:
            self.experiment.log_metric("train/lr", lr, step=state.global_step)

        # Component metrics from other callbacks
        for key, value in state.train_metrics.items():
            self.experiment.log_metric(
                f"train/{key}", value, step=state.global_step
            )

    def epoch_end(self, state: State) -> None:
        """Log epoch marker."""
        if self.experiment is None:
            return

        self.experiment.log_metric("epoch", state.epoch, step=state.global_step)

    def eval_end(self, state: State) -> None:
        """Log validation metrics."""
        if self.experiment is None:
            return

        for key, value in state.eval_metrics.items():
            self.experiment.log_metric(
                f"val/{key}", value, step=state.global_step
            )

            # Track best val loss
            if key == "loss" and isinstance(value, (int, float)):
                if self._best_val_loss is None or value < self._best_val_loss:
                    self._best_val_loss = value

    def fit_end(self, state: State) -> None:
        """Finish comet experiment with summary metrics."""
        if self.experiment is not None:
            # Log summary metrics
            self.experiment.log_metric("final_epoch", state.epoch)
            self.experiment.log_metric("total_steps", state.global_step)

            if self._best_train_loss is not None:
                self.experiment.log_metric("best_train_loss", self._best_train_loss)
            if self._best_val_loss is not None:
                self.experiment.log_metric("best_val_loss", self._best_val_loss)

            self.experiment.end()
            self.experiment = None

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "experiment_key": self.experiment_key,
            "best_train_loss": self._best_train_loss,
            "best_val_loss": self._best_val_loss,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.experiment_key = state_dict.get("experiment_key")
        self._best_train_loss = state_dict.get("best_train_loss")
        self._best_val_loss = state_dict.get("best_val_loss")
