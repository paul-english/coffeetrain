"""Optimizer factory for flexible optimizer configuration.

Supports multiple optimizer backends via config, including:
- PyTorch's AdamW (default)
- optimi's StableAdamW (recommended for stability)
- PyTorch's Muon optimizer for potential 2x speedup (auto-splits 2D/non-2D params)

Uses duck-typed config: any object with required attributes works.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW as TorchAdamW


@dataclass
class OptimizerConfig:
    """Optimizer configuration for create_optimizer.

    Use this dataclass or any duck-typed object with these attributes.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6

    # StableAdamW-specific options
    max_lr: Optional[float] = None
    decouple_lr: bool = False
    kahan_sum: Optional[bool] = None

    # Muon-specific options (2D param optimizer)
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: Optional[Literal["original", "match_rms_adamw"]] = None
    muon_fallback: Literal["adamw", "stable_adamw"] = "adamw"


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    """Get attribute from config (duck-typed)."""
    return getattr(config, name, default)


def create_optimizer(
    params: Union[Iterable[Tensor], List[Dict[str, Any]]],
    config: Any,
) -> Union[Optimizer, List[Optimizer]]:
    """Create an optimizer from configuration.

    Args:
        params: Model parameters or parameter groups
        config: Config object with required attrs:
            - name: Optimizer name ("adamw", "stable_adamw", "muon")
            - lr: Learning rate
            - weight_decay: Weight decay coefficient
            - betas: (beta1, beta2) for Adam variants
            - eps: Epsilon for numerical stability
            - Additional optimizer-specific kwargs

    Returns:
        Configured optimizer instance, or list of optimizers for Muon
        (Muon returns [muon_optimizer, fallback_optimizer] for 2D/non-2D params)

    Example:
        >>> config = OptimizerConfig(name="stable_adamw", lr=5e-4, weight_decay=0.01)
        >>> optimizer = create_optimizer(model.parameters(), config)
    """
    name = _get_attr(config, "name", "adamw")
    lr = _get_attr(config, "lr", 1e-4)
    weight_decay = _get_attr(config, "weight_decay", 0.01)
    betas = _get_attr(config, "betas", (0.9, 0.98))
    eps = _get_attr(config, "eps", 1e-6)
    max_lr = _get_attr(config, "max_lr")
    decouple_lr = _get_attr(config, "decouple_lr", False)
    kahan_sum = _get_attr(config, "kahan_sum")

    if name == "adamw":
        return TorchAdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    elif name == "stable_adamw":
        try:
            from optimi import StableAdamW
        except ImportError:
            raise ImportError(
                "optimi is required for StableAdamW. Install with: pip install optimi"
            )

        return StableAdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
        )

    elif name == "muon":
        from torch.optim import Muon

        # Muon only supports 2D params, so we split and use fallback for non-2D
        params_list = list(params)
        params_2d = [p for p in params_list if p.ndim == 2]
        params_other = [p for p in params_list if p.ndim != 2]

        optimizers = []

        if params_2d:
            optimizers.append(
                Muon(
                    params_2d,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=_get_attr(config, "momentum", 0.95),
                    nesterov=_get_attr(config, "nesterov", True),
                    ns_steps=_get_attr(config, "ns_steps", 5),
                    adjust_lr_fn=_get_attr(config, "adjust_lr_fn"),
                )
            )

        if params_other:
            # Create fallback optimizer for non-2D params
            fallback = _get_attr(config, "muon_fallback", "adamw")
            if fallback == "adamw":
                optimizers.append(
                    TorchAdamW(
                        params_other,
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=betas,
                        eps=eps,
                    )
                )
            elif fallback == "stable_adamw":
                try:
                    from optimi import StableAdamW
                except ImportError:
                    raise ImportError(
                        "optimi is required for StableAdamW fallback. "
                        "Install with: pip install optimi"
                    )
                optimizers.append(
                    StableAdamW(
                        params_other,
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=betas,
                        eps=eps,
                        decouple_lr=decouple_lr,
                        max_lr=max_lr,
                        kahan_sum=kahan_sum,
                    )
                )

        if not optimizers:
            raise ValueError("No parameters provided for Muon optimizer")

        return optimizers

    else:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Supported: adamw, stable_adamw, muon"
        )


def create_optimizer_with_param_groups(
    model: torch.nn.Module,
    config: Any,
    encoder_lr: Optional[float] = None,
    task_head_lr: Optional[float] = None,
) -> Union[Optimizer, List[Optimizer]]:
    """Create optimizer with differential learning rates and decoupled weight decay.

    Args:
        model: Model with named parameters
        config: Config with decoupled_wd option (duck-typed)
        encoder_lr: Learning rate for encoder parameters (None uses config.lr)
        task_head_lr: Learning rate for non-encoder parameters (None uses config.lr)

    Returns:
        Configured optimizer instance with parameter groups

    Notes:
        When config.decoupled_wd=True (default), weight decay is NOT applied to
        normalization layer parameters (LayerNorm, RMSNorm), following ModernBERT.
    """
    base_lr = _get_attr(config, "lr", 1e-4)
    encoder_lr = float(encoder_lr) if encoder_lr is not None else base_lr
    task_head_lr = float(task_head_lr) if task_head_lr is not None else base_lr
    weight_decay = _get_attr(config, "weight_decay", 0.01)
    decoupled_wd = _get_attr(config, "decoupled_wd", True)

    # Group parameters by: (is_encoder, is_norm)
    groups = {
        ("encoder", True): [],
        ("encoder", False): [],
        ("task_heads", True): [],
        ("task_heads", False): [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_encoder = "encoder" in name
        is_norm = "norm" in name.lower() if decoupled_wd else False
        group_key = ("encoder" if is_encoder else "task_heads", is_norm)
        groups[group_key].append(param)

    # Build param_groups list
    param_groups = []

    if groups[("encoder", False)]:
        param_groups.append({
            "params": groups[("encoder", False)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder",
        })

    if groups[("encoder", True)]:
        param_groups.append({
            "params": groups[("encoder", True)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
            "name": "encoder_norm",
        })

    if groups[("task_heads", False)]:
        param_groups.append({
            "params": groups[("task_heads", False)],
            "lr": task_head_lr,
            "weight_decay": weight_decay,
            "name": "task_heads",
        })

    if groups[("task_heads", True)]:
        param_groups.append({
            "params": groups[("task_heads", True)],
            "lr": task_head_lr,
            "weight_decay": 0.0,
            "name": "task_heads_norm",
        })

    return create_optimizer(param_groups, config)
