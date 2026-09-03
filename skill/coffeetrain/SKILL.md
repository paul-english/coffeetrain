---
name: coffeetrain
description: Use when writing or modifying PyTorch training code built on the coffeetrain library — wiring systems to events, registering plugins, configuring hyperparameters/CLI, or creating custom plugins for the Trainer runtime.
---

# coffeetrain

Lightweight event-driven PyTorch training runtime with composable plugins. You write
small functions (systems) tied to lifecycle events; the trainer fills their parameters
from shared context state. Inspired by MosaicML Composer, but with fewer dependencies.

## Mental model

- **Trainer** — the runtime. `trainer()` dispatches a CLI command (default: `train`).
- **Plugin** — a named bundle of events, systems, and commands. `Trainer()` registers a
  default plugin set; add more with `trainer.register_plugin(...)`.
- **System** — a function hooked onto one or more events via `@trainer.system('EVENT')`.
  Parameters are resolved (in order) from injected helpers (`log`, `run_event`,
  `get_state`, `set_state`, `execution_block`), then hyperparameter defaults from the
  function signature, then keys in shared context. Return a `dict` to update context
  (or `None`).
- **Context** — flat `trainer.context` dict holding all state (`model`, `loss`, `epoch`,
  `global_step`, ...). Systems communicate through it.
- **Command** — CLI entry point (`python train.py train --max_epochs 5`). Any parameter
  default on a system/command becomes a CLI-overridable hyperparameter.

## Minimal training script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from coffeetrain import Trainer

trainer = Trainer()

@trainer.system('DATA_BEFORE')
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return {
        'train_dataloader': DataLoader(train_ds, batch_size=64, shuffle=True),
        'eval_dataloader': DataLoader(test_ds, batch_size=256, shuffle=False),
        'loss_fn': nn.CrossEntropyLoss(),
    }

@trainer.system('MODEL_BEFORE')
def create_model():
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistCNN().to(compute_device)
    return {'model': model, 'compute_device': compute_device}

@trainer.system('OPTIMIZER_BEFORE')
def create_optimizer(model, lr=5e-4):
    return {'optimizer': torch.optim.Adam(model.parameters(), lr=lr), 'lr': lr}

@trainer.system('FORWARD_BEFORE')
def prepare_forward_batch(batch, compute_device):
    inputs, labels = batch
    return {'inputs': inputs.to(compute_device), 'labels': labels.to(compute_device)}

@trainer.system('FORWARD')
def forward_pass(model, inputs):
    return {'outputs': model(inputs)}

@trainer.system('BACKWARD_AFTER')
def optimizer_step(optimizer):
    optimizer.step()
    optimizer.zero_grad()

if __name__ == '__main__':
    trainer()
```

Run it:

```bash
python train.py train --max_epochs 10 --lr 5e-4
```

Note: the core loop calls `loss.backward()` for you (`BACKWARD` event), so the user only
steps the optimizer on `BACKWARD_AFTER`. Loss is computed automatically on `LOSS`/
`EVAL_LOSS` if `loss_fn`, `outputs`, and `labels` are in context.

## Event lifecycle

The default `train` plugin uses block-oriented events. Each block fires `{EVENT}_BEFORE`,
`{EVENT}`, `{EVENT}_AFTER`.

| Event | Phase | Description |
|-------|-------|-------------|
| `DATA` | Setup | Load datasets and loss function |
| `MODEL` | Setup | Build and place model on device |
| `OPTIMIZER` | Setup | Create optimizer(s) |
| `FIT` | Training | Start/end of full training run |
| `TRAINING_INTERRUPTED` | Training | Fired on graceful SIGINT/SIGTERM |
| `EPOCH` | Epoch | One training epoch |
| `BATCH` | Batch | One training batch |
| `FORWARD` | Batch | Model forward pass |
| `LOSS` | Batch | Loss computation |
| `BACKWARD` | Batch | `loss.backward()` |
| `EVAL` | Eval | Full validation pass |
| `EVAL_BATCH` | Eval | One validation batch |
| `EVAL_FORWARD` | Eval | Forward pass during eval |
| `EVAL_LOSS` | Eval | Loss computation during eval |

The default `train` command expects user systems for: `DATA_BEFORE` (dataloaders +
`loss_fn`), `MODEL_BEFORE` (`model`, `compute_device`), `OPTIMIZER_BEFORE`
(`optimizer`), `FORWARD_BEFORE`/`FORWARD` (batch prep, forward), and `BACKWARD_AFTER`
(optimizer step).

Eval runs when `eval_dataloader` is set, every `eval_interval` epochs (default 1).
Mean eval loss lands in context as `eval_loss`.

## Common context keys

`model`, `optimizer`, `train_dataloader`, `eval_dataloader`, `batch`, `batch_idx`,
`inputs`, `labels`, `outputs`, `loss`, `epoch`, `global_step`, `eval_loss`,
`stop_training` (set `True` to stop after the current batch), `interrupted`.

To read state that other systems may have updated mid-flight (common in commands), use
the injected `get_state('key')` / `set_state({...})` helpers instead of parameter
injection.

## Optional plugins

```python
from coffeetrain import Trainer, ema_plugin, save_best_model_plugin, wandb_plugin

trainer = Trainer()
trainer.register_plugin([ema_plugin, save_best_model_plugin, wandb_plugin])
```

Available: `wandb_plugin`, `comet_plugin` (experiment tracking), `ema_plugin`, `swa_plugin`, `hf_accelerator` (HuggingFace Accelerate)
(weight averaging), `save_best_model_plugin` (checkpoint on metric improvement),
`lr_monitor_plugin`, `parameter_counter_plugin`, `batch_size_scheduler` (batch size
warmup), `gradient_accumulator`, `text_progress` (plain-text alternative to tqdm).

Plugin parameter defaults become CLI flags, e.g. `--ema_decay 0.999`.

## Writing a custom plugin

```python
from coffeetrain import Plugin, Trainer

my_plugin = Plugin(name="my_plugin", description="Example plugin")

@my_plugin.system('FIT_BEFORE')
def on_fit_start(log):
    log.info("Starting training")

@my_plugin.system('BATCH_AFTER')
def track_step(global_step):
    return {'my_step': global_step}

trainer = Trainer()
trainer.register_plugin(my_plugin)
```

Key rules:

- System/command names must be unique across plugins — collisions raise
  `PluginMergeConflict` at registration.
- Hooking an event nobody registered logs a warning and the system is silently dropped
  — reuse the standard event names above when extending the default loop.
- Systems can be plain functions or `async def`. Plain functions run in an executor.
- Commands: `@my_plugin.command()` async functions become CLI subcommands.
- To replace the body of another plugin's block, use `@plugin.override_block('EVENT')`
  (one override per event allowed). Custom events: `@plugin.event()` on a `str, Enum`
  class auto-creates `_BEFORE`/`_AFTER` variants.
- Inside commands, `async with execution_block(EVENT, {'key': value})` fires the
  before/after hooks around custom logic.

## Gotchas

- Interrupts (Ctrl-C) are handled gracefully by default; a second interrupt force-exits.
- `execution_block` is async-only: `async with` it (a sync `with` raises).
- Hyperparameters are collected from **all** registered system/command signatures;
  name clashes overwrite each other silently, so prefix plugin-specific params
  (e.g. `ema_decay`).

## Reference

- Docs: `docs/getting-started.md`, `docs/core-concepts.md` in the coffeetrain repo
- Examples: `examples/mnist` (simple), `examples/bert-hgnc` (masked LM, HF models)
- Core API: `src/coffeetrain/trainer.py`, `src/coffeetrain/plugin.py`,
  `src/coffeetrain/runtime.py`; default loop: `src/coffeetrain/plugins/interruptable_train.py`
