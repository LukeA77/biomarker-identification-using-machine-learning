"""Training and inference harness for the LoRA / full-FT arms.

Design points that matter for correctness:
* **bf16 autocast** on the forward/loss only; params stay fp32 (no GradScaler
  needed for bf16). SGConv's FFT ops run in fp32 inside autocast automatically.
* **batch >= 2 guaranteed** by the loader's sampler (SSSM squeezes a size-1 batch).
* **early stopping on inner-validation patient-level balanced accuracy** — the
  metric we ultimately care about, not window loss.
* best weights (by monitored metric) are restored before returning.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livinglab_codebrain.loader import NoSingletonBatchSampler
from .ft_config import FinetuneConfig
from .metrics import aggregate_session_level, classification_metrics

log = logging.getLogger(__name__)

__all__ = ["predict_windows", "evaluate_session_level", "train_model", "TrainResult"]

_AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _autocast(precision: str):
    dtype = _AUTOCAST_DTYPE[precision]
    enabled = precision != "fp32"
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str = "cuda",
    precision: str = "bf16",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference, return ``(probs, y_true, patient_ids, conditions)`` per window."""
    model.eval()
    probs_all, y_all, pid_all, cond_all = [], [], [], []
    for x, y, info in loader:
        x = x.to(device, non_blocking=True)
        with _autocast(precision):
            logits = model(x)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        probs_all.append(p)
        y_all.append(y.numpy())
        pid_all.extend(info["patient_id"])
        cond_all.extend(info["condition"])
    return (
        np.concatenate(probs_all),
        np.concatenate(y_all),
        np.asarray(pid_all),
        np.asarray(cond_all),
    )


def evaluate_session_level(
    model: nn.Module,
    loader: DataLoader,
    ft_cfg: FinetuneConfig,
    *,
    device: str = "cuda",
) -> dict:
    """Window-level + session-level (patient x condition) metrics for a loader."""
    probs, y_true, pids, conds = predict_windows(
        model, loader, device=device, precision=ft_cfg.train.precision
    )
    thr = ft_cfg.eval.threshold
    window_metrics = classification_metrics(y_true, probs, threshold=thr)
    sess = aggregate_session_level(pids, conds, probs, y_true, agg=ft_cfg.eval.patient_agg)
    session_metrics = classification_metrics(sess.y_true, sess.prob, threshold=thr)
    return {
        "window": window_metrics.as_dict(),
        "session": session_metrics.as_dict(),
        "session_scores": {
            "sessions": sess.sessions.tolist(),
            "patients": sess.patients.tolist(),
            "prob": sess.prob.tolist(),
            "y_true": sess.y_true.tolist(),
            "n_windows": sess.n_windows.tolist(),
        },
    }


@dataclass
class TrainResult:
    best_epoch: int
    best_monitor: float
    history: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    trainable_params: int = 0


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _monitored_value(val_metrics: dict, monitor: str) -> float:
    """Resolve an early-stopping monitor name against an evaluation result.

    ``inner_val_<metric>`` reads the SESSION-level metric (the LOPO default and
    the historical behaviour — unchanged). ``inner_val_window_<metric>`` reads the
    WINDOW-level metric instead, which is what the within-patient task-split
    experiment needs: there a validation set spans only that one patient's two
    sessions, so session-level balanced accuracy can take just three values
    (0 / 0.5 / 1) and is far too coarse to early-stop on.
    """
    key = monitor.replace("inner_val_", "")
    level = "session"
    if key.startswith("window_"):
        level, key = "window", key[len("window_"):]
    if key not in val_metrics[level]:
        raise KeyError(
            f"early-stopping monitor {monitor!r} resolves to {level}[{key!r}], which is not in "
            f"the evaluation result (available: {sorted(val_metrics[level])})")
    return float(val_metrics[level][key])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ft_cfg: FinetuneConfig,
    *,
    lr: float,
    device: str = "cuda",
    max_epochs: int | None = None,
) -> TrainResult:
    """Train ``model`` with AdamW + cosine-warmup + bf16, early-stopping on inner-val.

    ``val_loader`` should cover the inner-validation patients (non-overlap). The
    monitored quantity is patient-level balanced accuracy (per config). Best
    weights are restored in-place before returning.
    """
    tcfg = ft_cfg.train
    es = tcfg.early_stopping
    max_epochs = max_epochs or tcfg.max_epochs

    optimizer = _build_optimizer(model, lr, tcfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(tcfg.warmup_frac * total_steps)
    scheduler = _cosine_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training: %d trainable params, %d steps/epoch, up to %d epochs (lr=%.1e).",
             trainable, steps_per_epoch, max_epochs, lr)

    result = TrainResult(best_epoch=-1, best_monitor=-np.inf, trainable_params=trainable)
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        # re-shuffle training order each epoch (sampler is stateful).
        # Duck-typed so SessionBalancedBatchSampler re-draws its weighted sample too.
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        running = 0.0
        n_batches = 0
        for x, y, _info in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(tcfg.precision):
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tcfg.grad_clip
                )
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            n_batches += 1

        val = evaluate_session_level(model, val_loader, ft_cfg, device=device)
        monitor = _monitored_value(val, es.monitor)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, n_batches),
            "inner_val_balanced_accuracy": val["session"]["balanced_accuracy"],
            "inner_val_auc": val["session"]["auc"],
            "lr": scheduler.get_last_lr()[0],
        }
        result.history.append(row)
        log.info("epoch %d | loss %.4f | inner-val bal-acc %.3f | auc %s",
                 epoch, row["train_loss"], row["inner_val_balanced_accuracy"],
                 f"{row['inner_val_auc']:.3f}" if row["inner_val_auc"] is not None else "n/a")

        improved = monitor > result.best_monitor + es.min_delta
        if improved:
            result.best_monitor = monitor
            result.best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es.patience:
                log.info("Early stopping at epoch %d (best epoch %d, %s=%.3f).",
                         epoch, result.best_epoch, es.monitor, result.best_monitor)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.train_seconds = time.time() - t0
    return result
