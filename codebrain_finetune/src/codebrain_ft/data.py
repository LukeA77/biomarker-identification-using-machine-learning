"""Fold/split helpers built on top of the audited ``livinglab_codebrain`` loader.

The base loader package provides LOPO folds and per-fold loaders. Fine-tuning
additionally needs:
  * loaders for an *arbitrary* patient list (inner train/val splits), and
  * a grouped inner split of an outer training set for HP selection / early
    stopping (5-fold GroupKFold, grouped on patient — never split within a
    patient).
These reuse ``WindowDataset`` + ``make_dataloader`` so the singleton-safe
batching and window contract are inherited unchanged.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, Sampler

from livinglab_codebrain.config import DataConfig
from livinglab_codebrain.dataset import CohortManifest, WindowDataset
from livinglab_codebrain.loader import make_dataloader, collate_windows

log = logging.getLogger(__name__)

__all__ = [
    "make_patient_loader", "inner_group_split",
    "NormalizedWindowDataset", "INPUT_NORM_SCHEMES", "cohort_median_scale",
    "SessionBalancedBatchSampler", "duration_matched_mask",
]

# Input-amplitude normalisation schemes (Experiment 7).
#   none      -- raw microvolts/100, exactly as the original runs used
#   global_z  -- per-window z over ALL channels: removes overall amplitude but
#                PRESERVES relative channel amplitudes (topography)
#   chan_z    -- per-window per-channel z: also removes each channel's own
#                amplitude, discarding topographic amplitude information
# Motivation is empirical, not stylistic: relative band power decodes ON/OFF
# (p=0.005) while absolute power is at chance, and the LoRA predictions showed
# between-subject spread 2.1x the within-subject effect. See EXPERIMENTS.md.
INPUT_NORM_SCHEMES = ("none", "global_z", "chan_z")


class NormalizedWindowDataset(Dataset):
    """Wraps ``WindowDataset`` and normalises each window's amplitude.

    A wrapper by design: the audited loader package, the window cache and the
    manifest are all left untouched -- only what the model *sees* changes.
    Normalised windows are rescaled to ``target_scale`` (the cohort-median window
    std) so the backbone keeps its pretrained input magnitude; z-scoring to unit
    std alone would inflate inputs ~4x and push the model off its operating point.
    """

    def __init__(self, base: WindowDataset, scheme: str, target_scale: float) -> None:
        if scheme not in INPUT_NORM_SCHEMES:
            raise ValueError(f"input_norm must be one of {INPUT_NORM_SCHEMES}, got {scheme!r}")
        self.base, self.scheme, self.target = base, scheme, float(target_scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y, info = self.base[i]                    # (C, P, S), already /norm_divisor
        if self.scheme == "none":
            return x, y, info
        c = x.shape[0]
        flat = x.reshape(c, -1)                      # (C, P*S)
        if self.scheme == "global_z":
            mu, sd = flat.mean(), flat.std()
        else:                                        # chan_z
            mu = flat.mean(dim=1, keepdim=True)
            sd = flat.std(dim=1, keepdim=True)
        sd = torch.clamp(sd, min=1e-8)
        return ((flat - mu) / sd * self.target).reshape(x.shape), y, info


class SessionBalancedBatchSampler(Sampler):
    """Batches in which every SESSION contributes equally, in expectation.

    Why this matters here (measured, not stylistic): sessions range from 98 to
    1141 windows (11.6x), and ON sessions are systematically LONGER than OFF
    (median 288 vs 239 non-overlap windows, paired Wilcoxon p=0.044). Under
    uniform sampling the gradient is dominated by long recordings, and because
    length covaries with condition, recording duration becomes an exploitable
    shortcut. Weighting each window by 1/(its session's size) removes that: every
    session contributes the same expected mass per epoch regardless of length.

    Sampling is with replacement (standard for weighted sampling), seeded, and
    re-drawn each epoch via :meth:`set_epoch`. The no-singleton guarantee that
    ``NoSingletonBatchSampler`` provides is preserved -- SSSM's ``x.squeeze()``
    would corrupt a size-1 batch.
    """

    def __init__(
        self,
        session_ids: np.ndarray,
        batch_size: int,
        *,
        min_size: int = 2,
        seed: int = 0,
    ) -> None:
        session_ids = np.asarray(session_ids)
        n = len(session_ids)
        if n < min_size:
            raise ValueError(f"{n} items but min_size={min_size}; cannot form a batch.")
        if batch_size < min_size:
            raise ValueError(f"batch_size ({batch_size}) must be >= min_size ({min_size}).")
        uniq, counts = np.unique(session_ids, return_counts=True)
        size_of = dict(zip(uniq, counts))
        w = np.array([1.0 / size_of[s] for s in session_ids], dtype=np.float64)
        self._p = w / w.sum()
        self._n = n
        self._bs = batch_size
        self._min = min_size
        self._seed = seed
        self._epoch = 0
        self._n_sessions = len(uniq)
        log.info("Session-balanced sampling: %d windows across %d sessions "
                 "(sizes %d-%d); each session weighted equally.",
                 n, self._n_sessions, counts.min(), counts.max())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        idx = rng.choice(self._n, size=self._n, replace=True, p=self._p)
        batches = [idx[i:i + self._bs].tolist() for i in range(0, self._n, self._bs)]
        if len(batches) >= 2 and len(batches[-1]) < self._min:
            tail = batches.pop()
            batches[-1].extend(tail)
        assert all(len(b) >= self._min for b in batches), "singleton batch leaked"
        yield from batches

    def __len__(self) -> int:
        n_full, rem = divmod(self._n, self._bs)
        if rem == 0:
            return n_full
        if rem < self._min and n_full >= 1:
            return n_full
        return n_full + 1


def duration_matched_mask(
    frame,
    *,
    k: int,
    where: str = "middle",
    nonoverlap_only: bool = True,
) -> np.ndarray:
    """Boolean mask selecting exactly ``k`` windows from every session.

    Removes the recording-duration confound from evaluation: with an identical
    number of windows per session, a session's aggregated score cannot encode how
    long the recording was. ``where`` picks the contiguous block by time within
    the session -- ``"middle"`` avoids start-of-recording settling artefacts and
    end-of-recording fatigue, and is the pre-registered default.
    """
    if where not in {"first", "middle", "last"}:
        raise ValueError(f"where must be first/middle/last, got {where!r}")
    pids = frame["patient_id"].to_numpy().astype(str)
    conds = frame["condition"].to_numpy().astype(str)
    t = frame["t_cont"].to_numpy().astype(float)
    no = frame["is_nonoverlap_subset"].to_numpy().astype(bool)
    mask = np.zeros(len(frame), dtype=bool)
    for p in np.unique(pids):
        for c in np.unique(conds[pids == p]):
            sel = (pids == p) & (conds == c)
            if nonoverlap_only:
                sel = sel & no
            idx = np.where(sel)[0]
            if len(idx) < k:
                raise ValueError(
                    f"session {p}|{c} has {len(idx)} windows < k={k}; "
                    f"lower k (cohort minimum governs) rather than dropping a patient.")
            idx = idx[np.argsort(t[idx])]
            if where == "first":
                chosen = idx[:k]
            elif where == "last":
                chosen = idx[-k:]
            else:
                start = (len(idx) - k) // 2
                chosen = idx[start:start + k]
            mask[chosen] = True
    return mask


def cohort_median_scale(
    manifest: CohortManifest, data_cfg: DataConfig, *, n: int = 300, seed: int = 1
) -> float:
    """Cohort-median per-window std — the rescale target for normalised inputs."""
    df = manifest.subset(None, nonoverlap_only=False)
    ds = WindowDataset(df, data_cfg.paths.cache_dir, data_cfg.window)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return float(np.median([float(ds[i][0].std()) for i in idx]))


def make_patient_loader(
    manifest: CohortManifest,
    data_cfg: DataConfig,
    patients: list[str],
    *,
    nonoverlap_only: bool,
    shuffle: bool,
    seed: int,
    input_norm: str = "none",
    norm_scale: float | None = None,
    ea_whiteners: dict | None = None,
    session_balanced: bool = False,
    match_k: int | None = None,
    match_where: str = "middle",
) -> DataLoader:
    """Build a loader over exactly ``patients`` (train: full+shuffled; eval: non-overlap).

    Optional, all off by default so existing runs are byte-identical:

    * ``input_norm`` — per-window amplitude normalisation
      (:class:`NormalizedWindowDataset`); needs ``norm_scale``.
    * ``ea_whiteners`` — per-subject Euclidean-Alignment matrices
      (:class:`~.align.EAWindowDataset`); takes precedence over ``input_norm``
      since EA already removes amplitude. Also needs ``norm_scale``.
    * ``session_balanced`` — equalise each session's expected contribution
      (:class:`SessionBalancedBatchSampler`); training loaders only.
    * ``match_k`` / ``match_where`` — duration matching: exactly ``match_k``
      windows per session, removing the recording-length confound.
    """
    df = manifest.subset(patients, nonoverlap_only=nonoverlap_only)
    if match_k is not None:
        keep = duration_matched_mask(df, k=match_k, where=match_where,
                                     nonoverlap_only=nonoverlap_only)
        df = df[keep].reset_index(drop=True)
    ds: Dataset = WindowDataset(df, data_cfg.paths.cache_dir, data_cfg.window)
    if ea_whiteners is not None:
        if norm_scale is None:
            raise ValueError("norm_scale is required when ea_whiteners is given")
        from .align import EAWindowDataset
        ds = EAWindowDataset(ds, ea_whiteners, norm_scale)
    elif input_norm != "none":
        if norm_scale is None:
            raise ValueError("norm_scale is required when input_norm != 'none'")
        ds = NormalizedWindowDataset(ds, input_norm, norm_scale)

    if session_balanced:
        if not shuffle:
            raise ValueError("session_balanced sampling is for training loaders (shuffle=True)")
        sess = (df["patient_id"].astype(str) + "|" + df["condition"].astype(str)).to_numpy()
        sampler = SessionBalancedBatchSampler(
            sess, data_cfg.loader.batch_size,
            min_size=data_cfg.loader.min_batch_size, seed=seed,
        )
        return DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_windows,
            num_workers=data_cfg.loader.num_workers, pin_memory=data_cfg.loader.pin_memory,
        )
    return make_dataloader(ds, data_cfg, shuffle=shuffle, seed=seed)


def inner_group_split(
    train_patients: list[str],
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """Return ``n_folds`` (inner_train, inner_val) patient-list splits.

    GroupKFold groups on patient (each patient wholly in train or val). It is not
    randomised by construction; we apply a seeded patient shuffle first so the
    inner folds vary reproducibly with the seed.
    """
    patients = np.array(sorted(train_patients))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    patients = patients[perm]
    groups = patients  # one group per patient
    gkf = GroupKFold(n_splits=n_folds)
    dummy_x = np.zeros(len(patients))
    splits = []
    for tr_idx, va_idx in gkf.split(dummy_x, groups=groups):
        splits.append((patients[tr_idx].tolist(), patients[va_idx].tolist()))
    return splits
