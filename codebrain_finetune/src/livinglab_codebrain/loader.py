"""DataLoader construction and Leave-One-Patient-Out fold assembly.

Key correctness guarantee: no batch is ever smaller than ``min_batch_size`` (>=2).
CodeBrain's ``SSSM.forward()`` ends in ``x.squeeze()``, which would silently
collapse a size-1 batch dimension and corrupt shapes; :class:`NoSingletonBatchSampler`
makes that impossible for both training and evaluation loaders.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .config import DataConfig
from .dataset import CohortManifest, WindowDataset

log = logging.getLogger(__name__)

__all__ = [
    "collate_windows",
    "NoSingletonBatchSampler",
    "make_dataloader",
    "LopoFold",
    "build_lopo_folds",
    "make_fold_loaders",
]


def collate_windows(batch: list[tuple[torch.Tensor, int, dict]]):
    """Collate ``(x, y, info)`` items into ``(X, y, info)`` batch tensors.

    * ``X``    — ``float32`` tensor ``(B, n_channels, n_patches, samples_per_patch)``
    * ``y``    — ``int64`` tensor ``(B,)``
    * ``info`` — dict of lists (``patient_id``, ``condition``, ``t_cont``,
      ``path``, ``is_nonoverlap_subset``), preserving per-sample metadata.
    """
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    keys = batch[0][2].keys()
    info = {key: [item[2][key] for item in batch] for key in keys}
    return xs, ys, info


class NoSingletonBatchSampler(Sampler[list[int]]):
    """Yield batches of indices, guaranteeing every batch has ``>= min_size`` items.

    Standard batching leaves a remainder batch that can be a singleton when the
    dataset size is not a multiple of ``batch_size``. Dropping it (``drop_last``)
    would discard evaluation windows and bias patient-level metrics; instead, if
    the final batch would be too small, it is merged into the previous batch.

    Shuffling (training) is deterministic given ``seed`` and re-shuffles each epoch.
    """

    def __init__(
        self,
        n_items: int,
        batch_size: int,
        *,
        min_size: int = 2,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if n_items < min_size:
            raise ValueError(
                f"Dataset has {n_items} items but min_size={min_size}; cannot form a valid batch."
            )
        if batch_size < min_size:
            raise ValueError(f"batch_size ({batch_size}) must be >= min_size ({min_size}).")
        self._n = n_items
        self._bs = batch_size
        self._min = min_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._n_batches = self._count_batches(n_items, batch_size, min_size)

    @staticmethod
    def _count_batches(n: int, bs: int, min_size: int) -> int:
        n_full, rem = divmod(n, bs)
        if rem == 0:
            return n_full
        if rem < min_size and n_full >= 1:
            return n_full  # tail merged into the previous batch
        return n_full + 1

    def set_epoch(self, epoch: int) -> None:
        """Advance the shuffle seed so successive epochs see different orderings."""
        self._epoch = epoch

    def __iter__(self):
        order = list(range(self._n))
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            rng.shuffle(order)
            self._epoch += 1

        batches = [order[i : i + self._bs] for i in range(0, self._n, self._bs)]
        if len(batches) >= 2 and len(batches[-1]) < self._min:
            tail = batches.pop()
            batches[-1].extend(tail)  # merge undersized tail into the previous batch

        assert all(len(b) >= self._min for b in batches), "singleton batch leaked"
        yield from batches

    def __len__(self) -> int:
        return self._n_batches


def _seed_worker(worker_id: int) -> None:
    """Make DataLoader worker RNGs deterministic (matters only if num_workers>0)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(
    dataset: WindowDataset,
    config: DataConfig,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` with the singleton-safe sampler and window collate."""
    sampler = NoSingletonBatchSampler(
        len(dataset),
        config.loader.batch_size,
        min_size=config.loader.min_batch_size,
        shuffle=shuffle,
        seed=config.loader.seed if seed is None else seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_windows,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        worker_init_fn=_seed_worker,
    )


@dataclass(frozen=True)
class LopoFold:
    """One Leave-One-Patient-Out fold."""

    index: int
    test_patient: str
    train_patients: tuple[str, ...]


def build_lopo_folds(manifest: CohortManifest) -> list[LopoFold]:
    """Construct all LOPO folds over the current (post-exclusion) cohort.

    Deterministic and cohort-driven: fold *count* and *numbering* follow directly
    from the locked cohort, so this must be regenerated if the exclusion list
    changes. One fold per patient; each holds that patient out for test.
    """
    patients = manifest.patients
    if len(patients) < 3:
        raise ValueError(f"LOPO needs >=3 patients; cohort has {len(patients)}.")
    folds = [
        LopoFold(
            index=i,
            test_patient=p,
            train_patients=tuple(q for q in patients if q != p),
        )
        for i, p in enumerate(patients)
    ]
    log.info("Built %d LOPO folds over %d patients.", len(folds), len(patients))
    return folds


def make_fold_loaders(
    manifest: CohortManifest,
    config: DataConfig,
    fold: LopoFold,
    *,
    seed: int | None = None,
    check_meta: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)`` for one LOPO fold.

    * train — all non-held-out patients, full overlapping set, shuffled;
    * test  — the held-out patient, non-overlap subset only (per config), unshuffled.
    """
    window = config.window
    cache_dir = config.paths.cache_dir
    fold_seed = (config.loader.seed if seed is None else seed) + fold.index

    train_df = manifest.subset(list(fold.train_patients), nonoverlap_only=False)
    test_df = manifest.subset(
        [fold.test_patient],
        nonoverlap_only=config.overlap.eval_nonoverlap_only,
    )

    train_ds = WindowDataset(train_df, cache_dir, window, check_meta=check_meta)
    test_ds = WindowDataset(test_df, cache_dir, window, check_meta=check_meta)

    train_loader = make_dataloader(train_ds, config, shuffle=True, seed=fold_seed)
    test_loader = make_dataloader(test_ds, config, shuffle=False, seed=fold_seed)
    return train_loader, test_loader
