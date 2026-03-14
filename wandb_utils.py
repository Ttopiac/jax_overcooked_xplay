"""W&B logging utilities for training scripts.

Supports two modes:
  1. Post-training: log all metrics at once after training completes.
     Complete training curves, no performance impact.
  2. Real-time (future): periodic logging during training via callbacks.

Usage:
    # In main():
    logger = WandbLogger(
        project="overcooked-xplay",
        name="sp_seed0",
        config=config,
        enabled=args.wandb,
    )

    # After training:
    logger.log_training_curves(metrics, prefix="train")
    logger.finish()
"""

import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbLogger:
    """W&B logger with post-training batch upload.

    Args:
        project: W&B project name.
        name: Run name (e.g. "sp_seed0").
        config: Dict of hyperparameters to log.
        enabled: Whether to actually log (False = no-op).
        group: Optional group name for organizing runs.
        tags: Optional list of tags.
    """

    def __init__(self, project="overcooked-xplay", name=None, config=None,
                 enabled=True, group=None, tags=None):
        self.enabled = enabled and HAS_WANDB
        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                group=group,
                tags=tags,
                reinit=True,
            )
        else:
            self.run = None

    def log_training_curves(self, metrics, prefix="train", seed_idx=None):
        """Log all per-update metrics as training curves.

        Args:
            metrics: dict of arrays, each shape (num_updates,) or (num_seeds, num_updates).
            prefix: prefix for metric names.
            seed_idx: if metrics have seed dimension, which seed to log.
        """
        if not self.enabled:
            return

        for key, values in metrics.items():
            values = np.asarray(values)
            if seed_idx is not None and values.ndim > 1:
                values = values[seed_idx]
            for step, val in enumerate(values):
                wandb.log({f"{prefix}/{key}": float(val), "update": step})

    def log_metrics(self, metrics_dict, step=None):
        """Log a single set of metrics."""
        if not self.enabled:
            return
        wandb.log(metrics_dict, step=step)

    def log_summary(self, summary_dict):
        """Log summary metrics (final values)."""
        if not self.enabled:
            return
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def finish(self):
        """End the W&B run."""
        if self.enabled and self.run:
            self.run.finish()


def setup_wandb_args(parser):
    """Add W&B-related arguments to an argparse parser."""
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="overcooked-xplay")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group name for organizing runs")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated W&B tags")
