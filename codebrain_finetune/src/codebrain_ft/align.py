"""Euclidean Alignment (EA) — generic cross-subject distribution alignment.

**Why this is legitimate here, and what it is NOT.** EA does not tell the model
what to look for. It injects no biomarker, no band, no hypothesis about levodopa.
It is a *nuisance-removal* transform in the same family as the per-window amplitude
normalisation already used: it whitens each subject's data so that every subject's
mean spatial covariance becomes the identity, removing the between-subject offset
caused by skull thickness, cortical folding, electrode placement and impedance.
What the model then learns from the aligned signal is entirely its own.

**Why it targets THIS failure.** The LoRA predictions showed between-subject spread
in P(OFF) of 0.199 versus a within-subject ON/OFF effect of 0.096 — subject
identity dominates medication state by 2.1x. EA attacks exactly that ratio, and it
is the standard remedy for cross-subject transfer in the EEG literature
(He & Wu, 2020).

**Method.** For subject s with windows X_i in R^(C x T):

    R_s = (1/n) * sum_i (X_i X_i^T) / T        # mean spatial covariance (C x C)
    X~_i = R_s^(-1/2) X_i                       # aligned window

After alignment the subject's mean covariance is I, so all subjects sit in a
common space.

**Two properties that must be stated in the write-up:**

1. **Unsupervised.** R_s uses no labels whatsoever — only the raw signal.
2. **TRANSDUCTIVE.** R_s for a held-out subject is computed from that subject's own
   (unlabelled) recordings, pooling BOTH sessions. This is deliberate and
   necessary: computing it per-session would normalise each session to identity
   separately and would erase the ON/OFF difference we are trying to detect.
   Pooling both sessions removes the subject offset while preserving the
   within-subject contrast. It does mean the method requires both of a patient's
   recordings before it can score either, so it answers "which of this patient's
   recordings is OFF", not "is this single recording OFF". Declare this.

Whitened windows are rescaled to a target std (cohort-median window std) so the
backbone keeps the input magnitude it was pretrained on.
"""
from __future__ import annotations

import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from livinglab_codebrain.config import DataConfig
from livinglab_codebrain.dataset import CohortManifest, WindowDataset

log = logging.getLogger(__name__)

__all__ = ["compute_ea_whiteners", "EAWindowDataset", "load_or_compute_whiteners"]

_EPS = 1e-10


def _window_cov(args) -> np.ndarray:
    """Spatial covariance (C x C) of one cached window, from the RAW microvolt array."""
    cache_dir, rel_path = args
    with open(Path(cache_dir) / rel_path, "rb") as fh:
        x = np.asarray(pickle.load(fh)["X"], dtype=np.float64)   # (C, T)
    return (x @ x.T) / x.shape[1]


def compute_ea_whiteners(
    manifest: CohortManifest,
    data_cfg: DataConfig,
    *,
    workers: int = 8,
    ridge: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Per-subject EA whitening matrices ``R_s^(-1/2)``.

    Covariances pool BOTH sessions of a subject (see module docstring). No labels
    are used. ``ridge`` regularises the eigenvalues for numerical stability.
    """
    df = manifest.subset(None, nonoverlap_only=False)
    cache = str(data_cfg.paths.cache_dir)
    paths = df["path"].tolist()
    pids = df["patient_id"].to_numpy().astype(str)

    log.info("EA: computing spatial covariances for %d windows (%d workers)...", len(paths), workers)
    covs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(_window_cov, [(cache, p) for p in paths], chunksize=64):
            covs.append(c)
    covs = np.stack(covs)                                        # (N, C, C)

    whiteners: dict[str, np.ndarray] = {}
    for p in np.unique(pids):
        R = covs[pids == p].mean(axis=0)                         # subject mean covariance
        R = R + ridge * np.trace(R) / R.shape[0] * np.eye(R.shape[0])
        evals, evecs = np.linalg.eigh(R)
        evals = np.maximum(evals, _EPS)
        whiteners[p] = (evecs * (evals ** -0.5)) @ evecs.T       # R^(-1/2), symmetric
    log.info("EA: built whiteners for %d subjects (C=%d).", len(whiteners), covs.shape[1])
    return whiteners


def load_or_compute_whiteners(
    manifest: CohortManifest,
    data_cfg: DataConfig,
    cache_path: Path,
    *,
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load cached EA whiteners, computing and saving them on first use."""
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        w = {k: z[k] for k in z.files}
        missing = set(manifest.patients) - set(w)
        if missing:
            raise RuntimeError(
                f"EA cache {cache_path.name} is missing {len(missing)} cohort subjects "
                f"(e.g. {sorted(missing)[:3]}). Delete it and re-run.")
        log.info("EA: reusing cached whiteners from %s", cache_path.name)
        return w
    w = compute_ea_whiteners(manifest, data_cfg, workers=workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **w)
    log.info("EA: cached whiteners -> %s", cache_path)
    return w


class EAWindowDataset(Dataset):
    """Applies each subject's EA whitener to their windows, then rescales.

    A wrapper: the audited loader, the window cache and the manifest are untouched.
    """

    def __init__(
        self,
        base: WindowDataset,
        whiteners: dict[str, np.ndarray],
        target_scale: float,
    ) -> None:
        self.base = base
        self.target = float(target_scale)
        self._w = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                   for k, v in whiteners.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y, info = self.base[i]                    # (C, P, S)
        W = self._w.get(info["patient_id"])
        if W is None:
            raise KeyError(f"no EA whitener for patient {info['patient_id']!r}")
        c, p, s = x.shape
        flat = x.reshape(c, -1)                      # (C, T)
        out = W @ flat                               # whiten across channels
        sd = torch.clamp(out.std(), min=1e-8)
        return (out / sd * self.target).reshape(c, p, s), y, info
