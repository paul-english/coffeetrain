# Changelog

## v0.2.0

Breaking release promoting the v2 trainer architecture to the primary API.

### Breaking changes

- `TrainerV2` is now `Trainer`, imported from `coffeetrain`
- The legacy `coffeetrain.callbacks` module has been removed — all functionality
  lives in plugins now

### Added

- Plugin-based architecture: bundle events, systems, and commands into reusable
  `Plugin` objects registered with `trainer.register_plugin(...)`
- Optional plugins: W&B, Comet, EMA, SWA, save-best-model checkpointing, LR
  monitor, parameter counter, batch size scheduler, gradient accumulator, and a
  plain-text progress alternative to tqdm
- CLI hyperparameters: system/command parameter defaults become overridable flags
  (`python train.py train --max_epochs 10 --lr 5e-4`)
- Graceful SIGINT/SIGTERM handling in the default training loop
- Sphinx documentation with Read the Docs hosting
- End-user agent skill in `skill/coffeetrain/`

### Changed

- README and examples (`mnist`, `bert-hgnc`) rewritten around the plugin API
