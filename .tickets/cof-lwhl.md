---
id: cof-lwhl
status: closed
deps: []
links: []
created: 2026-04-23T17:24:05Z
type: task
priority: 2
assignee: Paul English
completed: 2026-04-23T18:00:00Z
---
# V1 source: src/coffeetrain/callbacks/progress.py. V2 partial: src/coffeetrain/plugins/tqdm_progress.py. Port text-log style as alternative or complete tqdm plugin with eval bar and is_main_process gating

## Summary

Completed both tqdm progress plugin enhancement and created text-log style alternative:

### 1. Enhanced tqdm_progress plugin (`src/coffeetrain/plugins/tqdm_progress.py`)
- Added **is_main_process gating** to prevent duplicate progress bars in distributed training
- Added **eval bar support** with EVAL_BEFORE, EVAL_BATCH_AFTER, EVAL_AFTER handlers
- Made metrics configurable via `tqdm_metrics` parameter (default: "loss,lr,step")
- Proper cleanup of progress bars at epoch and eval end
- Wraps train/eval dataloaders with tqdm for automatic progress updates

### 2. Created text_progress plugin (`src/coffeetrain/plugins/text_progress.py`)
- Ports the V1 text-log style output as a clean alternative to progress bars
- Includes **is_main_process gating** for distributed training
- Handlers for all training phases: FIT_BEFORE, EPOCH_BEFORE, BATCH_AFTER, EPOCH_AFTER, EVAL_AFTER, FIT_AFTER
- Configurable batch loss printing via `text_print_batch_loss` and `text_batch_log_interval` parameters
- Matches the formatting style of the original V1 ProgressCallback

### 3. Updated plugin exports
- Added text_progress to `src/coffeetrain/plugins/__init__.py` for public API

Users can now choose:
- `tqdm_progress` for visual progress bars with metrics
- `text_progress` for clean, readable log output
- Both with proper distributed training support

