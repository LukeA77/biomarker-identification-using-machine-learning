"""Stage 4 — frozen linear-probe baseline (the honest floor LoRA must beat).

Freeze the whole CodeBrain backbone, extract per-window features, pool to a
fixed vector, fit a linear classifier (logistic regression / linear SVM) on the
training patients, and evaluate the held-out patient at patient level. No
gradients through the backbone — features are precomputed once per fold, so this
is cheap and validates the whole train/eval/metrics harness before any
gradient-based fine-tuning.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from livinglab_codebrain.config import DataConfig
from .ft_config import FinetuneConfig
from .metrics import aggregate_session_level, classification_metrics

log = logging.getLogger(__name__)

__all__ = ["extract_features", "run_probe_fold"]


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    ft_cfg: FinetuneConfig,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, y, patient_ids, conditions)`` from the frozen backbone.

    ``model`` is a :class:`CodeBrainClassifier`; only its ``backbone_features`` is
    used. Pooling per ``ft_cfg.probe.pooling``: ``mean`` over channels x patches
    -> (200,) ; ``flatten`` -> (18*10*200,).
    """
    model.eval()
    pooling = ft_cfg.probe.pooling
    feats, ys, pids, conds = [], [], [], []
    for x, y, info in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = model.backbone_features(x)          # (B, C, P, S)
        f = f.float()
        if pooling == "mean":
            f = f.mean(dim=(1, 2))                   # (B, S)
        else:  # flatten
            f = f.contiguous().view(f.shape[0], -1)  # (B, C*P*S)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        pids.extend(info["patient_id"])
        conds.extend(info["condition"])
    return np.concatenate(feats), np.concatenate(ys), np.asarray(pids), np.asarray(conds)


def _build_classifier(ft_cfg: FinetuneConfig):
    p = ft_cfg.probe
    if p.classifier == "logreg":
        return LogisticRegression(C=p.C, max_iter=p.max_iter, class_weight="balanced")
    if p.classifier == "linear_svm":
        return LinearSVC(C=p.C, max_iter=p.max_iter, class_weight="balanced")
    raise ValueError(f"probe.classifier must be logreg/linear_svm, got {p.classifier!r}")


def run_probe_fold(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    data_cfg: DataConfig,
    ft_cfg: FinetuneConfig,
    *,
    device: str = "cuda",
) -> dict:
    """Fit the probe on train features, evaluate the held-out patient (session-level)."""
    Xtr, ytr, _, _ = extract_features(model, train_loader, ft_cfg, device=device)
    Xte, yte, pids_te, conds_te = extract_features(model, test_loader, ft_cfg, device=device)
    log.info("Probe features: train %s, test %s (pooling=%s)", Xtr.shape, Xte.shape, ft_cfg.probe.pooling)

    scaler = None
    if ft_cfg.probe.standardize:
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = _build_classifier(ft_cfg)
    clf.fit(Xtr, ytr)

    # probability of OFF (class 1)
    if hasattr(clf, "predict_proba"):
        prob_te = clf.predict_proba(Xte)[:, 1]
    else:  # LinearSVC: map decision function to (0,1) via logistic squashing
        prob_te = 1.0 / (1.0 + np.exp(-clf.decision_function(Xte)))

    thr = ft_cfg.eval.threshold
    window_metrics = classification_metrics(yte, prob_te, threshold=thr)
    sess = aggregate_session_level(pids_te, conds_te, prob_te, yte, agg=ft_cfg.eval.patient_agg)
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
