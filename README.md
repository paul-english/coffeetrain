# coffeetrain

Lightweight event-driven PyTorch trainer with composable callbacks. Inspired by MosaicML Composer but implemented fewer external dependencies for maximum compatibility.

## Features

- **Event-driven lifecycle**: `fit_start`, `epoch_start`, `batch_start`, `before_forward`, `after_forward`, `before_loss`, `after_loss`, `before_backward`, `after_backward`, `batch_end`, `eval_*`, etc.
- **Composable callbacks**: EMA, SWA, checkpointing, W&B, Comet, early stopping, batch size scheduling, and more.
- **TrainerModel protocol**: Simple interface (`forward`, `loss`) for model integration.
- **Accelerate support**: Distributed training via HuggingFace Accelerator.

## Installation

```bash
pip install coffeetrain
```

Optional extras:
```bash
pip install coffeetrain[wandb,comet,optimi]
```

## Quick Start

```python
from coffeetrain import Trainer, CosineWarmupScheduler, HistoryCallback, BestModelCheckpointer
from coffeetrain import create_optimizer
from coffeetrain.optimizers import OptimizerConfig

model = MyModel()
optimizer = create_optimizer(model.parameters(), OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01))
scheduler = CosineWarmupScheduler(optimizer, warmup_steps=100, total_steps=1000)

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    optimizers=optimizer,
    schedulers=scheduler,
    max_epochs=10,
    callbacks=[
        HistoryCallback(save_dir="output"),
        BestModelCheckpointer(save_dir="output", metric_name="loss", mode="min"),
    ],
)
trainer.fit()
```

## Callbacks

| Callback | Description |
|----------|-------------|
| `BestModelCheckpointer` | Save best model by metric |
| `HistoryCallback` | Track and save training history to JSON |
| `EMACallback` | Exponential moving average of weights |
| `SWACallback` | Stochastic weight averaging |
| `EarlyStoppingCallback` | Stop when metric stops improving |
| `WandbCallback` | Log to Weights & Biases |
| `CometCallback` | Log to Comet.ml |
| `BatchSizeSchedulerCallback` | Batch size warmup |
| `ScheduleLoggerCallback` | Log LR schedule phase transitions |
| `ParameterCounter` | Print parameter counts at start |
| `SpeedMonitor` | Track samples/sec |
| `ProgressCallback` | Print epoch summaries |
| `LRMonitor` | Log learning rates |
| `TorchMetricsCallback` | Integrate torchmetrics |

## License

Apache-2.0
