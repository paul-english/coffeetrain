# Coffee Train

Make your training loop while your coffee is still hot. An event-based PyTorch training runtime that can be used with or without batteries.

Coffee Train provides a bare-minimum event-driven training loop with a set of common plugins for composing any training workflow you need. It ships with a default training loop meant as a quick starting point for experiments and projects.

## Install

```bash
pip install coffeetrain
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add coffeetrain
```

## Minimal example

```python
from coffeetrain import Trainer

trainer = Trainer()

@trainer.system('DATA_BEFORE')
def load_data():
    ...

@trainer.system('MODEL_BEFORE')
def create_model():
    ...

if __name__ == '__main__':
    trainer()
```

```bash
python train.py train --max_epochs 10
```

## Documentation

- [Getting Started](getting-started.md)
- [Core Concepts](core-concepts.md)
