"""Cached frozen-backbone features for the task-split sidecar.

WHY A CACHE IS LEGITIMATE HERE. In the ``frozen_probe`` arm the CodeBrain
backbone is frozen: it never sees a gradient, so the feature vector it produces
for a given window depends only on (window, input-norm scheme) — NOT on which
fold, patient or task is being trained. Extracting features once per scheme and
re-using them across folds is therefore *mathematically* the same computation as
extracting them per fold. Only the probe (a standardiser + logistic regression
fitted on the fold's own training rows) varies by fold.

IT IS NOT BIT-IDENTICAL, AND THE DIFFERENCE IS MEASURED, NOT ASSUMED. Extraction
runs under ``torch.autocast(bfloat16)``, and cuBLAS picks kernels by batch shape,
so a window batched alongside a different set of neighbours accumulates slightly
differently. Measured on this cohort: repeating extraction with the *same* loader
is exactly reproducible (max abs difference 0.0), while extracting the same
windows inside a larger frame moves features by ~0.4% relative. Downstream that
shifts a per-patient balanced accuracy by up to ~0.011 on the smallest held-out
task sets (a couple of windows crossing the 0.5 threshold) and the cohort mean by
~0.002. ``run_pairwise_transfer.py`` checks exactly this against the stored
within-patient numbers and reports the deltas, rather than claiming an equality
the hardware does not provide.

This is what makes two otherwise infeasible analyses cheap:

* the LOPO + task-split driver's ``frozen_probe`` arm — 27 folds x 2 directions
  x 2 normalisations = 108 probe fits, each over ~9,000 training windows;
* the 27 x 27 pairwise transfer matrix with a >=200-permutation max-statistic
  null — 702 ordered pairs re-scored per permutation.

Both become pure CPU linear algebra once the features exist.

The cache is keyed on the exact rows it was built from and refuses to be reused
if they change; the checkpoint SHA and the pooling mode are stored alongside so a
different backbone or a different ``probe.pooling`` cannot silently reuse it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["load_or_compute_backbone_features"]


def load_or_compute_backbone_features(
    frame: pd.DataFrame,
    data_cfg,
    ft_cfg,
    *,
    scheme: str,
    norm_scale: float | None,
    cache_dir: Path,
    device: str = "cuda",
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """``(N, D)`` frozen-backbone features, row-aligned to ``frame``.

    Row alignment is guaranteed, not assumed: the loader is built with
    ``shuffle=False`` and ``NoSingletonBatchSampler`` emits contiguous index
    slices in order (merging only an undersized tail into the previous batch),
    so concatenating the per-batch outputs reproduces the frame's own order. The
    returned patient/condition vectors are checked against the frame and the run
    aborts on any mismatch rather than reporting features attached to the wrong
    windows.
    """
    import hashlib

    from .tracking import file_sha256

    key = np.asarray([f"{p}|{c}" for p, c in
                      zip(frame["path"].astype(str), frame["patient_id"].astype(str))])
    meta = {"scheme": scheme, "pooling": ft_cfg.probe.pooling,
            "norm_scale": str(norm_scale),
            "checkpoint_sha256": file_sha256(data_cfg.paths.checkpoint)}
    # Content-addressed: the rows, the normalisation and the backbone checkpoint
    # all go into the filename, so a smoke run, a different scheme or a different
    # checkpoint can never silently reuse or overwrite each other's features.
    digest = hashlib.sha256(
        ("\n".join(key.tolist()) + "|" + "|".join(f"{k}={v}" for k, v in sorted(meta.items()))
        ).encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{scheme}_{digest}.npz"

    if cache_path.is_file() and not force:
        # Close the handle before returning: an open NpzFile holds a Windows lock
        # that would break a later tmp+replace into the same directory.
        with np.load(cache_path, allow_pickle=False) as z:
            X = z["X"]
        log.info("Backbone feature cache HIT %s (%s)", cache_path.name, X.shape)
        return X, {"cache": "hit", "path": str(cache_path), **meta}

    import torch

    from .model import build_classifier_model
    from .probe import extract_features
    from .tasksplit import make_frame_loader

    model = build_classifier_model(data_cfg, ft_cfg, mode="frozen", device=device)
    try:
        loader = make_frame_loader(frame, data_cfg, shuffle=False, seed=ft_cfg.seed.base,
                                   input_norm=scheme, norm_scale=norm_scale)
        X, _, pids, conds = extract_features(model, loader, ft_cfg, device=device)
    finally:
        del model
        torch.cuda.empty_cache()

    if len(X) != len(frame):
        raise RuntimeError(f"extracted {len(X)} features for {len(frame)} rows")
    bad = (pids != frame["patient_id"].to_numpy().astype(str)) | \
          (conds != frame["condition"].to_numpy().astype(str))
    if bad.any():
        raise RuntimeError(
            f"backbone features are NOT row-aligned to the frame ({int(bad.sum())} mismatched "
            f"rows, first at index {int(np.argmax(bad))}); refusing to cache features that "
            f"would be attached to the wrong windows.")

    tmp = cache_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:                 # savez appends ".npz" to a bare path
        np.savez_compressed(fh, X=X.astype(np.float32), _key=key,
                            **{f"_{k}": np.asarray(v) for k, v in meta.items()})
    tmp.replace(cache_path)
    log.info("Wrote backbone feature cache %s (%s)", cache_path, X.shape)
    return X.astype(np.float32), {"cache": "miss", "path": str(cache_path), **meta}
