"""XAI Tier D — is the explanation SHARED across patients, or patient-specific?

This is the test the XAI chapter's claim rests on. Tiers A-C describe what the
model uses; none of them distinguishes "one shared physiological route" from "27
per-patient fingerprints averaged together". A fingerprint is per-patient by
definition, so any component of the explanation that is consistent across patients
is by construction NOT a fingerprint — and is the only thing that could be handed
to Tier E as a biomarker candidate.

Two measurements, both built on one cheap primitive: a per-patient x per-channel
importance matrix M, where M[p, c] = (patient p's balanced accuracy) - (the same
with channel c occluded). Computing it costs 18 forward passes over the whole
evaluation set, because each pass yields every patient's number at once.

D1  CONSISTENCY. Mean pairwise Spearman correlation between patients' importance
    profiles (rows of M). Null: independently shuffle each patient's channel
    ordering, which destroys any shared structure while preserving each patient's
    marginal distribution of importances. A shared route implies observed >> null.

D2  TRANSFER. For each source patient A take its top-k channels, occlude exactly
    those, and read every patient's drop. Diagonal (A tested on A) vs off-diagonal
    (A's channels tested on B). A shared route implies off-diagonal ~ diagonal; a
    fingerprint implies off-diagonal ~ 0. Costs 27 forward passes.

Both are reported with a patient-level bootstrap. Note that D2's diagonal is
selected on the same data it is evaluated on, so the diagonal is optimistically
biased — the honest comparison is off-diagonal against the NULL, and diagonal
against off-diagonal as a descriptive gap, not a test.

Usage:
    python codebrain_finetune/src/run_xai_discriminative.py \
        --checkpoint checkpoints/xai/xai_seed5__toast_to_tea__full_ft__seed5.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config  # noqa: E402
from codebrain_ft import load_finetune_config, seed_everything  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import make_frame_loader  # noqa: E402
from codebrain_ft.metrics import classification_metrics  # noqa: E402
from codebrain_ft.xai import load_xai_checkpoint  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402
from run_xai_attribution import build_test_frame, CHANNELS, _json_default  # noqa: E402
from run_xai_perturbation import chan_z  # noqa: E402

log = logging.getLogger("xai_discriminative")


def per_patient_ba(model, x_norm, y, pids, patients, *, device, batch_size=64) -> np.ndarray:
    """Balanced accuracy for every patient in one pass over the data."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_norm), batch_size):
            xb = torch.from_numpy(x_norm[i:i + batch_size]).float().to(device)
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    prob = np.concatenate(out)
    return np.array([
        classification_metrics(y[pids == p], prob[pids == p], threshold=0.5).balanced_accuracy
        for p in patients])


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier D: shared vs patient-specific explanation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    import torch
    from scipy.stats import spearmanr
    if not torch.cuda.is_available():
        log.error("CUDA unavailable. Aborting.")
        return 1
    device = "cuda"

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _RUN_ROOT / ckpt

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    seed_everything(ft_cfg.seed.base, deterministic=False)

    model, payload = load_xai_checkpoint(ckpt, data_cfg, ft_cfg, device=device)
    target = float(payload["norm_scale"])
    tag = args.out_tag or ckpt.stem
    out_dir = _RUN_ROOT / "results" / "xai" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    frame = build_test_frame(payload, sidecar, ts_cfg)
    loader = make_frame_loader(frame, data_cfg, shuffle=False, seed=1, input_norm="none")

    xs, ys, pids = [], [], []
    for x, y, info in loader:
        xs.append(x.numpy()); ys.append(y.numpy()); pids.extend(info["patient_id"])
    Xraw = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    pids = np.asarray(pids)
    patients = np.unique(pids)
    Xn = chan_z(Xraw, target)

    base = per_patient_ba(model, Xn, y, pids, patients, device=device, batch_size=args.batch_size)
    log.info("Baseline: patient-mean BA %.4f over %d patients", base.mean(), len(patients))

    # ---- M[patient, channel] : 18 passes, every patient's number per pass -----
    M = np.zeros((len(patients), len(CHANNELS)))
    for ci, name in enumerate(CHANNELS):
        Z = Xn.copy(); Z[:, ci] = 0.0
        M[:, ci] = base - per_patient_ba(model, Z, y, pids, patients,
                                         device=device, batch_size=args.batch_size)
        log.info("  channel %-4s mean dBA %+.4f (sd %.4f across patients)",
                 name, M[:, ci].mean(), M[:, ci].std())

    # ---- D1 consistency -------------------------------------------------------
    # Spearman on 18 values is Pearson on ranks, so rank once and use a vectorised
    # correlation. The permutation shuffles ranks directly — permuting a row then
    # re-ranking is the same thing — which makes 2000 draws seconds instead of
    # minutes.
    from scipy.stats import rankdata

    def mean_pairwise_from_ranks(R):
        C = np.corrcoef(R)
        iu = np.triu_indices(len(R), k=1)
        return float(np.nanmean(C[iu])), C[iu]

    R = np.apply_along_axis(rankdata, 1, M)
    obs, pairvals = mean_pairwise_from_ranks(R)
    # cross-check the vectorised path against scipy on one pair
    _chk = spearmanr(M[0], M[1]).statistic
    if not np.isclose(np.corrcoef(R[0], R[1])[0, 1], _chk, atol=1e-9):
        raise RuntimeError("rank-correlation shortcut disagrees with scipy spearmanr")

    rng = np.random.default_rng(1)
    null = np.empty(args.n_perm)
    for b in range(args.n_perm):
        Rp = np.array([rng.permutation(row) for row in R])
        null[b] = mean_pairwise_from_ranks(Rp)[0]
    p_val = float((np.sum(null >= obs) + 1) / (args.n_perm + 1))
    log.info("D1 consistency: observed mean pairwise Spearman %.4f | null %.4f +/- %.4f | p=%.4f",
             obs, null.mean(), null.std(), p_val)

    # ---- D2 transfer ----------------------------------------------------------
    k = args.top_k
    T = np.zeros((len(patients), len(patients)))
    for ai, a in enumerate(patients):
        top = np.argsort(-M[ai])[:k]
        Z = Xn.copy(); Z[:, top] = 0.0
        T[ai] = base - per_patient_ba(model, Z, y, pids, patients,
                                      device=device, batch_size=args.batch_size)
    diag = np.diag(T)
    off = T[~np.eye(len(patients), dtype=bool)]
    log.info("D2 transfer (top-%d): diagonal %.4f +/- %.4f | off-diagonal %.4f +/- %.4f | ratio %.3f",
             k, diag.mean(), diag.std(), off.mean(), off.std(),
             off.mean() / diag.mean() if diag.mean() else float("nan"))

    # patient-level bootstrap on the off-diagonal mean
    boot = np.array([T[np.ix_(idx := rng.integers(0, len(patients), len(patients)), idx)]
                     [~np.eye(len(patients), dtype=bool)].mean() for _ in range(2000)])

    rec = {
        "checkpoint": str(ckpt.relative_to(_RUN_ROOT)), "seed_base": payload["seed_base"],
        "direction": payload["direction"], "channels": CHANNELS,
        "patients": patients.tolist(), "baseline_per_patient_ba": base.tolist(),
        "baseline_mean_ba": float(base.mean()),
        "importance_matrix": M.tolist(),
        "d1_consistency": {
            "observed_mean_pairwise_spearman": obs,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_p95": float(np.percentile(null, 95)), "p_value": p_val,
            "n_permutations": args.n_perm,
            "pairwise_min": float(np.nanmin(pairvals)), "pairwise_max": float(np.nanmax(pairvals)),
        },
        "d2_transfer": {
            "top_k": k, "diagonal_mean": float(diag.mean()), "diagonal_sd": float(diag.std()),
            "offdiagonal_mean": float(off.mean()), "offdiagonal_sd": float(off.std()),
            "offdiag_over_diag": float(off.mean() / diag.mean()) if diag.mean() else None,
            "offdiag_bootstrap_ci": [float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))],
            "matrix": T.tolist(),
            "note": ("Diagonal is selected on the data it is scored on and is therefore "
                     "optimistically biased; compare off-diagonal against zero and treat "
                     "diagonal-vs-off-diagonal as descriptive."),
        },
        "git_sha": git_sha(_PROJECT), "env": env_versions(),
        "scope_caveat": ("Model is 0.832 on this design, 0.435 leak-free, recid 0.920. "
                         "5-seed spread 0.031 — replicate before any per-channel claim."),
    }
    (out_dir / "discriminative_summary.json").write_text(
        json.dumps(rec, indent=2, default=_json_default), encoding="utf-8")
    np.savez_compressed(out_dir / "discriminative_matrices.npz",
                        M=M, T=T, patients=patients, channels=np.array(CHANNELS), base=base)
    log.info("Wrote %s", out_dir / "discriminative_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
