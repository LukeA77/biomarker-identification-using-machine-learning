"""Session-level metrics, aggregation, bootstrap CI, and the permutation test.

**Evaluation unit = the session/recording (patient x condition), not the
patient.** This dataset is *within-subject*: every patient contributes both an
ON (A) and an OFF (C) session, so a patient does not have a single ON/OFF label.
The clinically meaningful question is "is this recording ON or OFF?", so each
held-out patient's non-overlap windows are aggregated **per condition** to one
score per session (mean predicted probability, then thresholded). Over the full
cohort this yields 70 balanced points (35 ON / 35 OFF). Window-level metrics are
reported too, but session-level is the headline.

(FINE_TUNING_PLAN §7 calls this "patient-level" and imagines 38 one-label
points; that framing predates the confirmation that the design is within-subject.
Session-level is the faithful realisation — see the Stage-4/5 log.)

Label map: 0 = ON (A), 1 = OFF (C). "positive" class is OFF (=1); so
sensitivity = OFF-detection recall, specificity = ON recall. Balanced accuracy
(mean of the two) is the primary, skew-robust metric.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

__all__ = [
    "SessionScores", "aggregate_session_level", "classification_metrics",
    "bootstrap_session_ci", "permutation_test_balanced_accuracy",
]


@dataclass
class SessionScores:
    """Per-session aggregated predictions, aligned by ``sessions``."""

    sessions: np.ndarray      # (S,) session ids "patient|condition"
    patients: np.ndarray      # (S,) patient id per session (for grouped resampling)
    prob: np.ndarray          # (S,) mean predicted P(OFF)
    y_true: np.ndarray        # (S,) true session label (0/1)
    n_windows: np.ndarray     # (S,) windows aggregated per session


def aggregate_session_level(
    patient_ids: np.ndarray,
    conditions: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    agg: str = "mean_prob",
) -> SessionScores:
    """Collapse window predictions to one score per session (patient x condition).

    A session's true label must be constant across its windows (asserted, not
    assumed).
    """
    if agg != "mean_prob":
        raise ValueError(f"unsupported patient_agg {agg!r}")
    patient_ids = np.asarray(patient_ids)
    conditions = np.asarray(conditions)
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)

    session_ids = np.array([f"{p}|{c}" for p, c in zip(patient_ids, conditions)])
    order = np.unique(session_ids)  # sorted, deterministic
    s_prob, s_true, s_n, s_pat = [], [], [], []
    for sid in order:
        m = session_ids == sid
        labels = np.unique(y_true[m])
        if labels.size != 1:
            raise ValueError(f"session {sid} has mixed labels {labels.tolist()} — aggregation invalid")
        s_prob.append(float(probs[m].mean()))
        s_true.append(int(labels[0]))
        s_n.append(int(m.sum()))
        s_pat.append(sid.split("|")[0])
    return SessionScores(
        sessions=order,
        patients=np.asarray(s_pat),
        prob=np.asarray(s_prob),
        y_true=np.asarray(s_true),
        n_windows=np.asarray(s_n),
    )


@dataclass
class Metrics:
    balanced_accuracy: float
    accuracy: float
    auc: float | None
    sensitivity: float | None   # recall of OFF (positive=1)
    specificity: float | None   # recall of ON  (negative=0)
    macro_f1: float
    n: int
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict:
        return asdict(self)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> Metrics:
    """Compute the standard metric set from probabilities + a threshold.

    AUC/sensitivity/specificity return ``None`` when a class is absent (e.g. a
    single-patient window-level slice), rather than raising — callers decide.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    y_pred = (prob >= threshold).astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    both_classes = n_pos > 0 and n_neg > 0

    bal_acc = balanced_accuracy_score(y_true, y_pred) if both_classes else float("nan")
    acc = float((y_pred == y_true).mean())
    auc = float(roc_auc_score(y_true, prob)) if both_classes else None
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    sens = spec = None
    if both_classes:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) else None   # OFF recall
        spec = float(tn / (tn + fp)) if (tn + fp) else None   # ON recall

    return Metrics(
        balanced_accuracy=float(bal_acc), accuracy=acc, auc=auc,
        sensitivity=sens, specificity=spec, macro_f1=macro_f1,
        n=len(y_true), n_pos=n_pos, n_neg=n_neg,
    )


def bootstrap_session_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    patients: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    threshold: float = 0.5,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for a metric, resampling **patients** (clusters) with replacement.

    Cluster (patient) resampling respects the within-subject structure — a
    patient's ON and OFF sessions are kept together — rather than treating the 70
    sessions as independent.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    patients = np.asarray(patients)
    uniq_patients = np.unique(patients)
    # index lists per patient, for fast cluster gathering
    per_patient_idx = {p: np.where(patients == p)[0] for p in uniq_patients}
    rng = np.random.default_rng(seed)
    n_pat = len(uniq_patients)
    vals = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, n_pat, size=n_pat)
        idx = np.concatenate([per_patient_idx[uniq_patients[c]] for c in chosen])
        yt, pr = y_true[idx], prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):  # degenerate resample; skip
            continue
        m = classification_metrics(yt, pr, threshold=threshold)
        v = getattr(m, metric)
        if v is not None and not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    point = getattr(classification_metrics(y_true, prob, threshold=threshold), metric)
    return {
        "metric": metric,
        "point": float(point) if point is not None else None,
        "ci_low": float(np.percentile(vals, lo)) if vals.size else None,
        "ci_high": float(np.percentile(vals, hi)) if vals.size else None,
        "ci": ci,
        "n_valid_resamples": int(vals.size),
    }


def permutation_test_balanced_accuracy(
    y_true: np.ndarray,
    prob: np.ndarray,
    patients: np.ndarray,
    *,
    threshold: float = 0.5,
    n_permutations: int = 200,
    seed_base: int = 10001,
) -> dict:
    """Global permutation test on session-level balanced accuracy (within-subject).

    Permutes labels **within each patient** (the sessions of one patient exchange
    their ON/OFF labels), i.e. a per-patient sign flip. This is the within-subject
    realisation of "permute the patient-level ON/OFF labels" (FINE_TUNING_PLAN §4):
    it respects the paired structure and never permutes window labels (which would
    leak). Returns the observed statistic and an empirical p-value.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    per_patient_idx = [np.where(patients == p)[0] for p in uniq]
    observed = classification_metrics(y_true, prob, threshold=threshold).balanced_accuracy

    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        rng = np.random.default_rng(seed_base + i)
        y_perm = y_true.copy()
        for idx in per_patient_idx:  # shuffle labels among a patient's own sessions
            y_perm[idx] = rng.permutation(y_true[idx])
        null[i] = classification_metrics(y_perm, prob, threshold=threshold).balanced_accuracy

    # one-sided p: fraction of null >= observed (+1 smoothing)
    p_value = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    return {
        "observed_balanced_accuracy": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }
