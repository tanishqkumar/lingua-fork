# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from functools import partial
import math
import logging
from typing import Callable, Dict, Type

from torch import nn
from torch.optim import AdamW, SGD, lr_scheduler
from torch.optim.optimizer import Optimizer

logger = logging.getLogger()


@dataclass
class OptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0

    # Optimizer selection
    optimizer: str = "adamw"

    # Scheduler selection
    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    annealing_step: int = 1000
    decay_fraction: float = 0.1

    exp_factor: float = 0.5


# =============================================================================
# SCHEDULER REGISTRY
# =============================================================================

def lr_constant(step: int) -> float:
    return 1.0


def lr_linear(step: int, warmup: int, n_steps: int, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = s * min_ratio + (1 - s)
    else:
        lr = min_ratio
    return lr


def lr_inv_sqrt(step: int, warmup: int, exp_factor: float, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    else:
        lr = max((warmup**exp_factor) / (step**exp_factor), min_ratio)
    return lr


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    cycle_length: float,
    theta: float,
    min_ratio: float,
) -> float:
    sign = ((step // (n_steps*cycle_length)) % 2) * -2 + 1
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            sign * math.cos(math.pi * s**theta / cycle_length) + 1
        )
    else:
        lr = min_ratio
    return lr


def lr_wsd(
    step: int,
    warmup: int,
    n_steps: int,
    decay_fraction: float,
    cycle_length: float,
    min_ratio: float,
) -> float:
    """
    UNDERSTANDING WARMUP-STABLE-DECAY LEARNING RATES: A RIVER VALLEY LOSS LANDSCAPE PERSPECTIVE
    https://arxiv.org/pdf/2410.05192
    """
    cycle_num = step // int(n_steps * cycle_length) + 1
    curr_n_steps = int(n_steps * cycle_length) * cycle_num
    decay_length = int(curr_n_steps * decay_fraction)

    if step < warmup:
        lr = float(step) / warmup
    elif step <= curr_n_steps - decay_length:
        lr = 1.0
    elif step > curr_n_steps - decay_length and step <= curr_n_steps:
        step = step - (curr_n_steps - decay_length)
        lr = 1/((step/curr_n_steps)*(1/min_ratio) + (1 - step/curr_n_steps))
    else:
        lr = min_ratio

    return lr


# Registry mapping scheduler names to builder functions
SCHEDULER_REGISTRY: Dict[str, Callable] = {
    "constant": lambda args, n_steps: lr_constant,
    "linear": lambda args, n_steps: partial(
        lr_linear, warmup=args.warmup, n_steps=n_steps, min_ratio=args.lr_min_ratio
    ),
    "inv_sqrt": lambda args, n_steps: partial(
        lr_inv_sqrt,
        warmup=args.warmup,
        exp_factor=args.exp_factor,
        min_ratio=args.lr_min_ratio,
    ),
    "cosine": lambda args, n_steps: partial(
        lr_cosine,
        warmup=args.warmup,
        n_steps=n_steps,
        cycle_length=args.cycle_length,
        theta=args.cosine_theta,
        min_ratio=args.lr_min_ratio,
    ),
    "wsd": lambda args, n_steps: partial(
        lr_wsd,
        warmup=args.warmup,
        n_steps=n_steps,
        decay_fraction=args.decay_fraction,
        cycle_length=args.cycle_length,
        min_ratio=args.lr_min_ratio,
    ),
}


def register_scheduler(name: str, builder: Callable):
    """Register a new scheduler builder function."""
    SCHEDULER_REGISTRY[name] = builder


# =============================================================================
# OPTIMIZER REGISTRY
# =============================================================================

def build_adamw(model: nn.Module, args: OptimArgs) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,
    )


def build_sgd(model: nn.Module, args: OptimArgs) -> Optimizer:
    return SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.beta1,
    )


# Registry mapping optimizer names to builder functions
OPTIMIZER_REGISTRY: Dict[str, Callable[[nn.Module, OptimArgs], Optimizer]] = {
    "adamw": build_adamw,
    "sgd": build_sgd,
}


def register_optimizer(name: str, builder: Callable[[nn.Module, OptimArgs], Optimizer]):
    """Register a new optimizer builder function."""
    OPTIMIZER_REGISTRY[name] = builder


# =============================================================================
# BUILD FUNCTIONS
# =============================================================================

def build_lr_fn(args: OptimArgs, n_steps: int):
    if args.scheduler not in SCHEDULER_REGISTRY:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}. Available: {list(SCHEDULER_REGISTRY.keys())}")

    if args.scheduler == "wsd":
        assert args.decay_fraction < args.cycle_length

    return SCHEDULER_REGISTRY[args.scheduler](args, n_steps)


def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of optimizer...")

    if args.optimizer not in OPTIMIZER_REGISTRY:
        raise NotImplementedError(f"Unknown optimizer: {args.optimizer}. Available: {list(OPTIMIZER_REGISTRY.keys())}")

    optimizer = OPTIMIZER_REGISTRY[args.optimizer](model, args)

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
