# Core Concepts

## Plugin

A plugin is a named bundle of events, systems, commands, and optional block overrides. Plugins merge into a `Trainer` via `register_plugin()`. Name collisions across plugins raise `PluginMergeConflict`.

Use the `Plugin` class for reusable extensions, or decorate directly on the trainer instance for one-off training scripts.

## System

Systems are functions tied to events. When an event fires, each registered system runs in registration order. Parameters are resolved from:

1. Injected runtime helpers (`log`, `run_event`, `get_state`, `set_state`, `execution_block`)
2. Hyperparameter defaults from the function signature
3. Keys already present in shared context

Return `None` or a `dict` of state updates.

## Command

Commands are async-capable entry points invoked by the trainer's CLI dispatcher. The built-in `train` command (from `train_plugin`) runs the default block-oriented loop.

```bash
python train.py train --max_epochs 5
```

## Event

Events are string identifiers grouped into execution blocks. Registering an event enum via `@plugin.event()` also creates `{NAME}_BEFORE` and `{NAME}_AFTER` companion events.

Custom plugins should reuse the default event names when extending the standard loop so optional plugins remain compatible.

## State / context

All runtime values live in `trainer.context`, a flat dictionary updated by system return values. Prefer returning explicit keys from systems rather than mutating global objects when possible.

Common keys are documented in the [README](../README.md#state).

## Hyperparameters

Any system or command parameter with a default value is recorded as a hyperparameter. These defaults document the public configuration surface and are intended for CLI overrides.

## Execution blocks

`execution_block(event)` is an async context manager that fires `{EVENT}_BEFORE`, `{EVENT}`, and `{EVENT}_AFTER` around a block of logic. The `train` command uses this to structure epochs, batches, and eval passes.

`override_block(event)` lets a plugin replace the body of a block — used by `gradient_accumulator` to customize the backward/optimizer-step sequence.

## Default vs optional plugins

**Default plugins** ship with every `Trainer()` and define the standard training loop.

**Optional plugins** add logging, checkpointing, EMA/SWA, batch-size scheduling, and similar concerns. Register only what you need to keep startup and conflicts minimal.

## Advanced

### Defining your own events

```python
from enum import Enum

@my_plugin.event()
class MyEvents(str, Enum):
    CUSTOM = 'CUSTOM'
```

### Working with blocks

Inside a command, use `async with execution_block(TrainEvents.EPOCH, {'epoch': epoch}):` to nest custom logic inside the standard before/after hooks.
