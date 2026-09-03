# Getting Started

This walkthrough mirrors [`examples/mnist/main.py`](../examples/mnist/main.py).

## 1. Create a trainer

```python
from coffeetrain import Trainer

trainer = Trainer()
```

`Trainer()` registers the default plugin set: training loop, tqdm progress, CUDA helpers, and `torch.compile` support.

## 2. Wire systems to events

Systems are plain functions decorated with `@trainer.system('EVENT')`. Return a dictionary to update shared training state.

```python
@trainer.system('DATA_BEFORE')
def load_data():
    # build train_dataloader, eval_dataloader, loss_fn
    return {'train_dataloader': ..., 'eval_dataloader': ..., 'loss_fn': ...}

@trainer.system('MODEL_BEFORE')
def create_model():
    return {'model': model, 'compute_device': device}

@trainer.system('FORWARD')
def forward_pass(model, inputs):
    return {'outputs': model(inputs)}
```

The default `train` command expects at minimum:

- `DATA_BEFORE` — dataloaders and `loss_fn`
- `MODEL_BEFORE` — `model` and device info
- `OPTIMIZER_BEFORE` — `optimizer`
- `FORWARD_BEFORE` / `FORWARD` — batch prep and forward pass
- `BACKWARD_AFTER` — optimizer step (the core loop calls `loss.backward()` for you)

## 3. Run from the CLI

```python
if __name__ == '__main__':
    trainer()
```

```bash
cd examples/mnist
uv run python main.py train --max_epochs 10 --lr 5e-4
```

The `train` command accepts hyperparameters from system/command signatures, including `max_epochs` and `lr`.

## 4. Add optional plugins

```python
from coffeetrain import Trainer, ema_plugin, save_best_model_plugin, hf_accelerator

trainer = Trainer()
trainer.register_plugin([ema_plugin, save_best_model_plugin])
# Add `hf_accelerator` when using `pip install coffeetrain[accelerate]`.
```

Plugin parameter defaults become CLI flags, e.g. `--ema_decay 0.999` or `--save_best_model_dir ./checkpoints`.

## Next steps

See [Core Concepts](core-concepts.md) for how plugins, events, and state fit together.
