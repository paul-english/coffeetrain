#!/usr/bin/env python3
"""Parameter counting plugin for TrainerV2.

This plugin displays model architecture and parameter counts at training start.

Features:
  - Print full model architecture representation
  - Display total, trainable, and frozen parameter counts
  - Format large numbers with K/M/B suffixes for readability

Usage:
  from coffeetrain.plugins.parameter_counter import parameter_counter_plugin
  from coffeetrain import TrainerV2

  trainer = TrainerV2()
  trainer.register_plugin(parameter_counter_plugin)
"""

from coffeetrain.plugin import Plugin

parameter_counter_plugin = Plugin(
    name='parameter_counter',
    description='Display model architecture and parameter counts at training start',
)


def _format_size(num_params: int) -> str:
    """Format parameter count with appropriate suffix (K, M, B).

    Args:
        num_params: Number of parameters

    Returns:
        Formatted string with K/M/B suffix
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


@parameter_counter_plugin.system('FIT_BEFORE')
def print_model_summary(model):
    """Print model architecture and parameter counts.

    Args:
        model: PyTorch model instance
    """
    # Print model architecture (PyTorch's built-in repr)
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)

    # Calculate totals
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Print summary
    print("=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"{'Total':<20} {_format_size(total):>10}")
    print(f"{'Trainable':<20} {_format_size(trainable):>10}")
    print(f"{'Frozen':<20} {_format_size(frozen):>10}")
    print("=" * 60)
