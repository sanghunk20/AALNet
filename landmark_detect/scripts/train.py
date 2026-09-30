"""Training script for AALNet.

- single GPU or multi-GPU (torchrun, DistributedDataParallel)
- bfloat16 mixed precision, gradient accumulation
- patient-level cross-validation folds (fold_splits.json)
- YAML config + command-line overrides
- best checkpoint by validation MRE, early stopping

Usage (run from the repository root):
    # one GPU: effective batch 64 = batch_size 8 x accum_iter 8
    python -m landmark_detect.scripts.train \
        --config landmark_detect/configs/lambda02.yaml \
        --data_root /path/to/landmark_detect_800 \
        --fold 0 --output_dir outputs/lambda02/fold0

    # two GPUs: effective batch 64 = 2 x batch_size 8 x accum_iter 4
    torchrun --nproc_per_node=2 -m landmark_detect.scripts.train \
        --config landmark_detect/configs/lambda02.yaml \
        --data_root /path/to/landmark_detect_800 \
        --fold 0 --accum_iter 4 --output_dir outputs/lambda02/fold0
"""

import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..datasets.landmark_dataset import LandmarkDataset
from ..losses.domain_loss import DomainLoss
from ..losses.dsnt_loss import DSNTLoss
from ..models.aalnet import AALNet
from ..trainers.landmark_engine import evaluate, train_one_epoch
from ..utils.train_utils import (
    init_distributed,
    is_main_process,
    set_epoch_learning_rate,
    weight_decay_groups,
)


def get_args_parser():
    parser = argparse.ArgumentParser('AALNet training')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Data
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to the preprocessed dataset '
                             '(images/, coords/, fold_splits.json)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold index')
    parser.add_argument('--num_workers', type=int, default=8)

    # Model
    parser.add_argument('--pretrained', type=str, default='imagenet',
                        choices=['imagenet', 'none'],
                        help="Backbone initialisation: ImageNet weights (timm) or random")
    parser.add_argument('--mlp_fpn_drop_path', type=float, default=0.275,
                        help='Stochastic-depth rate of the decoder blocks')

    # Base loss
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Sigma (px) of the target Gaussian of the JS regulariser')

    # Optimisation
    parser.add_argument('--epochs', type=int, default=400,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=60,
                        help='Early stopping patience (epochs without validation MRE '
                             'improvement). 0 disables early stopping.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', type=int, default=4,
                        help='Gradient accumulation steps '
                             '(effective batch = GPUs x batch_size x accum_iter)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (0 = disabled)')
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')

    # Multi-GPU
    parser.add_argument('--dist_backend', type=str, default='nccl')

    return parser


def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _explicit_cli_keys(argv=None) -> set:
    """Names of the arguments that were given on the command line."""
    parser = get_args_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    namespace, _ = parser.parse_known_args(argv)
    return set(vars(namespace))


DOMAIN_LOSS_KEYS = (
    'lambda_max',
    'alpha_midline_std', 'alpha_midline_old',
    'alpha_midface', 'alpha_lowerface', 'alpha_dental', 'alpha_canting',
)


def merge_config_and_args(config: dict | None, args, argv=None):
    """Fill args from the config; arguments given on the command line take priority.

    Config values are converted and checked like command-line values (type and
    choices); unknown keys are rejected. The nested ``domain_loss`` block
    (asymmetry loss) is stored as ``args.domain_loss``.
    """
    config = config or {}
    actions = {a.dest: a for a in get_args_parser()._actions}
    explicit = _explicit_cli_keys(argv)
    for key, value in config.items():
        if key == 'domain_loss':
            continue
        if key not in actions:
            raise ValueError(f"Unknown key in config file: '{key}'")
        action = actions[key]
        if value is not None and action.type is not None:
            value = action.type(value)
        if action.choices is not None and value not in action.choices:
            raise ValueError(
                f"Invalid value for '{key}' in config file: {value!r} "
                f"(choose from {list(action.choices)})"
            )
        if key not in explicit:
            setattr(args, key, value)

    domain_cfg = config.get('domain_loss', None)
    if domain_cfg is not None:
        unknown = sorted(set(domain_cfg) - set(DOMAIN_LOSS_KEYS))
        if unknown:
            raise ValueError(f"Unknown key(s) in the domain_loss block: {unknown}")
        if 'lambda_max' not in domain_cfg:
            raise ValueError("The domain_loss block requires 'lambda_max'")
    args.domain_loss = domain_cfg
    return args


def main(args):
    # Multi-GPU setup
    distributed, gpu = init_distributed(args.dist_backend)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    if is_main_process():
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # Load fold splits
    splits_path = Path(args.data_root) / 'fold_splits.json'
    with open(splits_path) as f:
        fold_splits = json.load(f)

    train_files = fold_splits['folds'][args.fold]['train_files']
    val_files = fold_splits['folds'][args.fold]['val_files']

    print(f"Fold {args.fold}/{fold_splits['num_folds']}: "
          f"train={len(train_files)}, val={len(val_files)}, "
          f"test={len(fold_splits['test_files'])}")

    # Datasets
    train_dataset = LandmarkDataset(args.data_root, train_files, augment=True)
    val_dataset = LandmarkDataset(args.data_root, val_files, augment=False)

    # The training set is split across processes; the validation fold is
    # evaluated as a whole by the main process.
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    def worker_init_fn(worker_id):
        cv2.setNumThreads(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # Model
    model = AALNet(
        pretrained=(args.pretrained == 'imagenet'),
        decoder_drop_path=args.mlp_fpn_drop_path,
    )
    model.to(device)

    model_without_ddp = model
    if distributed:
        model = DDP(model, device_ids=[gpu])
        model_without_ddp = model.module

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Optimizer
    param_groups = weight_decay_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # Base loss
    criterion = DSNTLoss(temperature=1.0, lambda_js=1.0, sigma=args.sigma)

    # Asymmetry loss (optional, configured by the `domain_loss` block of the YAML)
    domain_criterion = None
    domain_lambda_max = 0.0
    domain_cfg = getattr(args, 'domain_loss', None)
    if domain_cfg and domain_cfg.get('lambda_max', 0.0) > 0:
        domain_lambda_max = float(domain_cfg['lambda_max'])
        domain_criterion = DomainLoss(
            alpha_midline_std=domain_cfg.get('alpha_midline_std', 0.6),
            alpha_midline_old=domain_cfg.get('alpha_midline_old', 0.4),
            alpha_midface=domain_cfg.get('alpha_midface', 1.0),
            alpha_lowerface=domain_cfg.get('alpha_lowerface', 1.0),
            alpha_dental=domain_cfg.get('alpha_dental', 1.0),
            alpha_canting=domain_cfg.get('alpha_canting', 1.0),
        ).to(device)
        print(f"Asymmetry loss enabled: lambda={domain_lambda_max}")

    # AMP gradient scaler
    loss_scaler = torch.cuda.amp.GradScaler()

    # Resume
    start_epoch = 0
    best_mre = float('inf')
    patience_counter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'scaler' in ckpt:
            loss_scaler.load_state_dict(ckpt['scaler'])
        if 'best_mre' in ckpt:
            best_mre = ckpt['best_mre']
        print(f"Resumed from epoch {start_epoch}, best_mre={best_mre:.2f}")

    # Training loop
    print(f"Start training from epoch {start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        set_epoch_learning_rate(
            optimizer, epoch, args.lr, args.min_lr, args.warmup_epochs, args.epochs,
        )

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.max_norm,
            accum_iter=args.accum_iter,
            print_freq=args.print_freq,
            domain_criterion=domain_criterion,
            domain_lambda_max=domain_lambda_max,
        )

        # Validation, checkpointing and logging: main process only, on the whole fold
        if is_main_process():
            val_metrics = evaluate(
                model=model_without_ddp,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                domain_criterion=domain_criterion,
                domain_lambda=train_metrics.get('domain_lambda', 0.0),
            )

            current_mre = val_metrics['mre_mean']
            if current_mre < best_mre:
                best_mre = current_mre
                patience_counter = 0
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'best_mre': best_mre,
                    'args': vars(args),
                }, output_dir / 'checkpoint_best.pth')
                print(f'*** New best MRE: {best_mre:.2f}px at epoch {epoch}')

                # Per-landmark MRE of the best model
                csv_path = output_dir / 'best_mre_per_landmark.csv'
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['landmark', 'mre_px'])
                    for name, mre in zip(train_dataset.landmark_names,
                                         val_metrics['mre_per_landmark']):
                        writer.writerow([name, f'{mre:.4f}'])
                    writer.writerow(['MEAN', f'{best_mre:.4f}'])
                    writer.writerow(['STD', f'{val_metrics["mre_std"]:.4f}'])
            else:
                patience_counter += 1

            log_entry = {
                'epoch': epoch,
                **train_metrics,
                **{k: v for k, v in val_metrics.items()
                   if not isinstance(v, (np.ndarray, list))},
            }
            with open(output_dir / 'log.jsonl', 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')

        # Early stopping on validation MRE. The main process decides and
        # broadcasts, so that all processes leave the loop together.
        if args.patience and args.patience > 0:
            should_stop = torch.zeros(1, dtype=torch.int, device=device)
            if is_main_process() and patience_counter >= args.patience:
                should_stop.fill_(1)
            if distributed:
                dist.broadcast(should_stop, src=0)
            if should_stop.item() == 1:
                print(f'*** Early stopping at epoch {epoch} '
                      f'(no validation MRE improvement for {args.patience} epochs). '
                      f'Best MRE: {best_mre:.2f}px')
                break

    total_time = time.time() - start_time
    print(f'Training complete in {str(datetime.timedelta(seconds=int(total_time)))}')
    print(f'Best MRE: {best_mre:.2f}px')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.config:
        args = merge_config_and_args(load_config(args.config), args)
    else:
        args.domain_loss = None
    if args.data_root is None:
        parser.error('--data_root is required')

    try:
        main(args)
    finally:
        # Explicit teardown of the process group, so that a following run on the
        # same machine does not inherit a stale one.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
