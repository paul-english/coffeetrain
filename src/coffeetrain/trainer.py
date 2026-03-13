"""Main Trainer class with event-driven training loop."""

import signal
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from coffeetrain.events import Event
from coffeetrain.state import State
from coffeetrain.callback import Callback
from coffeetrain.model import TrainerModel

if TYPE_CHECKING:
    from accelerate import Accelerator


class Trainer:
    """Event-driven trainer for PyTorch models.

    Manages the training loop with configurable callbacks for extensibility.
    Supports gradient clipping, gradient accumulation, and evaluation.

    Example:
        model = MyModel()
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            optimizers=optimizer,
            max_epochs=10,
            callbacks=[EMACallback(), HistoryCallback()],
        )
        trainer.fit()

    Attributes:
        model: Model implementing TrainerModel protocol
        state: Training state container
        callbacks: List of registered callbacks
    """

    def __init__(
        self,
        model: TrainerModel,
        train_dataloader: DataLoader,
        optimizers: Union[Optimizer, List[Optimizer]],
        max_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        schedulers: Optional[Union[LRScheduler, List[LRScheduler]]] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = 'cuda',
        grad_clip_norm: Optional[float] = None,
        grad_accum: int = 1,
        eval_interval: int = 1,
        progress_bar: bool = True,
        log_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        accelerator: Optional["Accelerator"] = None,
    ):
        """Initialize the Trainer.

        Args:
            model: Model implementing TrainerModel protocol
            train_dataloader: Training data loader
            optimizers: Optimizer or list of optimizers
            max_epochs: Number of epochs to train
            eval_dataloader: Optional validation data loader
            schedulers: Optional LR scheduler(s)
            callbacks: List of callbacks
            device: Device to train on ('cuda', 'cpu', etc.)
            grad_clip_norm: Max gradient norm for clipping (None = no clipping)
            grad_accum: Gradient accumulation steps
            eval_interval: Evaluate every N epochs
            progress_bar: Show tqdm progress bar
            log_interval: Log every N batches
            checkpoint_dir: Directory for auto-saving checkpoints on interrupt
            accelerator: Optional HuggingFace Accelerator for distributed training
        """
        self.model = model
        self.accelerator = accelerator
        self.is_main_process = accelerator is None or accelerator.is_main_process
        self.grad_clip_norm = grad_clip_norm
        self._initial_grad_accum = grad_accum  # Store for state initialization
        self.eval_interval = eval_interval
        self.progress_bar = progress_bar
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._interrupted = False
        self._original_sigint = None
        self._original_sigterm = None

        # Normalize to lists
        if isinstance(optimizers, Optimizer):
            optimizers = [optimizers]
        if schedulers is None:
            schedulers = []
        elif isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]

        # Setup device
        if accelerator is not None:
            # Accelerator handles device placement
            device = accelerator.device
        else:
            device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
            model.to(device)

        # Prepare with Accelerator (wraps model, optimizer, dataloader for DDP)
        if accelerator is not None:
            # Prepare all components - accelerator handles DDP wrapping
            prepared = accelerator.prepare(
                model,
                *optimizers,
                train_dataloader,
            )
            # Unpack prepared objects
            model = prepared[0]
            optimizers = list(prepared[1:1+len(optimizers)])
            train_dataloader = prepared[1+len(optimizers)]
            self.model = model  # Update reference to wrapped model

        # Create state
        self.state = State(
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_epochs=max_epochs,
            device=device,
            grad_accum=self._initial_grad_accum,
            is_main_process=self.is_main_process,
        )

        # Setup callbacks
        self.callbacks = callbacks or []

        # Run init event
        self._run_event(Event.INIT)

    def fit(self) -> State:
        """Run the training loop.

        Returns:
            Final training state
        """
        self._setup_signal_handlers()
        try:
            self._run_event(Event.FIT_START)

            for epoch in range(self.state.max_epochs):
                self.state.epoch = epoch
                self._train_epoch()

                # Check for interrupt/early stopping after epoch
                if self.state.stop_training:
                    break

                # Evaluation
                if self.state.eval_dataloader is not None:
                    if (epoch + 1) % self.eval_interval == 0:
                        self._eval_epoch()

                # Early stopping check (set by callbacks like EarlyStoppingCallback)
                if self.state.stop_training:
                    break

            # Fire interrupted event if stopped by signal
            if self._interrupted:
                self._run_event(Event.TRAINING_INTERRUPTED)
                # Auto-save checkpoint on interrupt if checkpoint_dir is set
                if self.checkpoint_dir:
                    ckpt_path = self.checkpoint_dir / "checkpoint_interrupted.pt"
                    self.save_checkpoint(str(ckpt_path))
                    print(f"Checkpoint saved to: {ckpt_path}")

            self._run_event(Event.FIT_END)
        finally:
            self._restore_signal_handlers()
        return self.state

    def _train_epoch(self) -> None:
        """Run a single training epoch."""
        self.model.train()
        self._run_event(Event.EPOCH_START)

        dataloader = self.state.train_dataloader
        if self.progress_bar and self.is_main_process:
            dataloader = tqdm(dataloader, desc=f"Epoch {self.state.epoch + 1}/{self.state.max_epochs}")

        # Zero gradients at epoch start
        for optimizer in self.state.optimizers:
            optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            self.state.batch_idx = batch_idx
            self._train_batch(batch)

            # Check for interrupt/early stopping after each batch
            if self.state.stop_training:
                break

            # Update progress bar
            if self.progress_bar and hasattr(dataloader, 'set_postfix'):
                postfix = {
                    'loss': f"{self.state.loss.item():.4f}",
                    'lr': f"{self.state.get_lr():.2e}",
                    'step': self.state.global_step,
                }
                dataloader.set_postfix(postfix)

        self._run_event(Event.EPOCH_END)

    def _train_batch(self, batch: Any) -> None:
        """Process a single training batch."""
        # Signal CUDAGraphs step boundary at start of each iteration
        # Must be called before any computation for torch.compile with max-autotune
        torch.compiler.cudagraph_mark_step_begin()

        self.state.batch = batch
        self._run_event(Event.BATCH_START)

        # Move batch to device if it's a dict of tensors
        batch = self._to_device(batch)
        self.state.batch = batch

        # Use Accelerate's accumulate context manager if available
        context_manager = (
            self.accelerator.accumulate(self.model)
            if self.accelerator is not None
            else nullcontext()
        )

        with context_manager:
            # Forward pass
            self._run_event(Event.BEFORE_FORWARD)
            outputs = self.model.forward(batch)
            self.state.outputs = outputs
            self._run_event(Event.AFTER_FORWARD)

            # Compute loss
            self._run_event(Event.BEFORE_LOSS)
            loss = self.model.loss(outputs, batch)
            self.state.loss = loss
            self._run_event(Event.AFTER_LOSS)

            # Backward pass
            self._run_event(Event.BEFORE_BACKWARD)
            if self.accelerator is not None:
                # Accelerator handles loss scaling for accumulation
                self.accelerator.backward(loss)
            else:
                # Manual gradient accumulation
                scaled_loss = loss / self.state.grad_accum
                scaled_loss.backward()
            self._run_event(Event.AFTER_BACKWARD)

            # Optimizer step
            if self.accelerator is not None:
                # Accelerator's accumulate context handles when to sync gradients
                if self.accelerator.sync_gradients:
                    if self.grad_clip_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_norm
                        )
                    for optimizer in self.state.optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in self.state.schedulers:
                        scheduler.step()
                    self.state.global_step += 1
            else:
                # Manual gradient accumulation
                if (self.state.batch_idx + 1) % self.state.grad_accum == 0:
                    if self.grad_clip_norm is not None:
                        for optimizer in self.state.optimizers:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.grad_clip_norm
                            )
                    for optimizer in self.state.optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in self.state.schedulers:
                        scheduler.step()
                    self.state.global_step += 1

        self._run_event(Event.BATCH_END)

    def _eval_epoch(self) -> None:
        """Run evaluation on the validation set."""
        self.model.eval()
        self._run_event(Event.EVAL_START)

        # Reset eval metrics
        self.state.eval_metrics = {}
        total_loss = 0.0
        num_batches = 0

        dataloader = self.state.eval_dataloader
        if self.progress_bar:
            dataloader = tqdm(dataloader, desc="Validation")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Signal CUDAGraphs step boundary at start of each iteration
                torch.compiler.cudagraph_mark_step_begin()

                self.state.batch_idx = batch_idx
                self.state.batch = batch
                self._run_event(Event.EVAL_BATCH_START)

                # Move to device
                batch = self._to_device(batch)
                self.state.batch = batch

                # Forward
                self._run_event(Event.EVAL_BEFORE_FORWARD)
                outputs = self.model.forward(batch)
                self.state.outputs = outputs
                self._run_event(Event.EVAL_AFTER_FORWARD)

                # Compute loss
                loss = self.model.loss(outputs, batch)
                self.state.loss = loss
                total_loss += loss.item()
                num_batches += 1

                self._run_event(Event.EVAL_BATCH_END)

                # Update progress bar
                if self.progress_bar and hasattr(dataloader, 'set_postfix'):
                    dataloader.set_postfix({'loss': f"{loss.item():.4f}"})

        # Store eval loss
        if num_batches > 0:
            self.state.eval_metrics['loss'] = total_loss / num_batches

        self._run_event(Event.EVAL_END)

        # Return to training mode
        self.model.train()

    def _run_event(self, event: Event) -> None:
        """Dispatch an event to all callbacks.

        Args:
            event: Event to dispatch
        """
        method_name = event.value
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method is not None:
                method(self.state)

    def _to_device(self, batch: Any) -> Any:
        """Move batch to training device.

        Handles dicts, lists, tensors, and objects with .to() method.
        Leaves other types unchanged.

        Args:
            batch: Batch data

        Returns:
            Batch on device
        """
        # Handle objects with .to() method (e.g., PackedBatch)
        if hasattr(batch, 'to') and callable(getattr(batch, 'to')):
            return batch.to(self.state.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            # Don't move lists of records (NER / multi-task format)
            if batch and isinstance(batch[0], dict) and ('input' in batch[0] or 'text' in batch[0]):
                return batch  # Leave as-is for NER / multi-task records
            return type(batch)(self._to_device(v) for v in batch)
        elif isinstance(batch, Tensor):
            return batch.to(self.state.device)
        return batch

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        # Only save on main process in distributed training
        if not self.is_main_process:
            return

        # Unwrap model if using Accelerate
        model_to_save = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )

        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in self.state.optimizers],
            'scheduler_state_dicts': [sch.state_dict() for sch in self.state.schedulers],
            'callback_state_dicts': {
                type(cb).__name__: cb.state_dict() for cb in self.callbacks
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.state.device)

        self.state.epoch = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']

        # Handle _orig_mod. prefix from torch.compile
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        for opt, state_dict in zip(self.state.optimizers, checkpoint['optimizer_state_dicts']):
            opt.load_state_dict(state_dict)

        for sch, state_dict in zip(self.state.schedulers, checkpoint['scheduler_state_dicts']):
            sch.load_state_dict(state_dict)

        for cb in self.callbacks:
            cb_name = type(cb).__name__
            if cb_name in checkpoint['callback_state_dicts']:
                cb.load_state_dict(checkpoint['callback_state_dicts'][cb_name])

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful stopping."""
        self._interrupted = False
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame) -> None:
        """Handle interrupt signal (SIGINT/SIGTERM)."""
        if self._interrupted:
            # Second interrupt - force exit
            print("\nForced exit requested.")
            raise KeyboardInterrupt
        self._interrupted = True
        self.state.stop_training = True
        print("\nGraceful stop requested. Finishing current batch...")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
