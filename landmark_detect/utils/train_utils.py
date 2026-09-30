"""Helpers for the training script: optimizer parameter groups, learning-rate
schedule and multi-GPU setup."""

import builtins
import math
import os

import torch
import torch.distributed as dist


def weight_decay_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split trainable parameters into two optimizer groups.

    Biases and one-dimensional parameters (normalization scales and shifts) are
    not decayed; every other parameter uses ``weight_decay``.
    """
    decayed, not_decayed = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias_or_norm = param.ndim <= 1 or name.endswith('.bias')
        (not_decayed if is_bias_or_norm else decayed).append(param)
    return [
        {'params': decayed, 'weight_decay': weight_decay},
        {'params': not_decayed, 'weight_decay': 0.0},
    ]


def set_epoch_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> float:
    """Set the learning rate of ``epoch``: linear warm-up, then cosine decay.

    The rate is constant within an epoch. It rises linearly from 0 (epoch 0) to
    ``base_lr`` (epoch ``warmup_epochs``) and then follows half a cosine period
    down to ``min_lr`` at ``total_epochs``.
    """
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        cosine = math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + cosine)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def init_distributed(backend: str = 'nccl') -> tuple[bool, int]:
    """Join the process group when launched with ``torchrun``.

    ``torchrun`` exports RANK, WORLD_SIZE and LOCAL_RANK for each process. Without
    them the script runs as a single process.

    Returns:
        (distributed, gpu): whether a process group was initialised, and the
        index of the GPU assigned to this process.
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend=backend, init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()

    if rank != 0:
        # Only the first process writes to the console.
        builtins.print = lambda *args, **kwargs: None
    return True, gpu


def is_main_process() -> bool:
    """True for a single-process run and for rank 0 of a distributed run."""
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0
