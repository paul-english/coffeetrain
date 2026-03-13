"""Parameter counting callback."""

from coffeetrain.callback import Callback
from coffeetrain.state import State


def _format_size(num_params: int) -> str:
    """Format parameter count with appropriate suffix (K, M, B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


class ParameterCounter(Callback):
    """Print parameter counts at training start."""

    def fit_start(self, state: State) -> None:
        """Print model architecture and parameter counts."""
        model = state.model

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
