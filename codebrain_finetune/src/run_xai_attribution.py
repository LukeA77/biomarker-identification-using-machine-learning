"""XAI Tiers B & C on a saved pooled-task-split checkpoint (XAI_PLAN.md 4).

Explains the model that produced the headline number: pooled task split, chan_z,
duration-matched, arm full_ft, direction toast_to_tea (test task = tea).

**What this can and cannot support.** The model scores 0.832 here but 0.435 under
the leak-free cross-subject design, and `recid` is 0.920 on the same windows. Every
map below therefore describes a decision function known to be substantially a
recording fingerprint. These outputs are CANDIDATE GENERATORS for Tier D/E, not
biomarkers. Any frequency axis is a post-hoc lens we impose — the model never saw
band definitions (XAI_PLAN.md 9.1) — and that must be stated in figure captions.

Tier A1 (head filter-bank readout) is deliberately not run here: it was measured
and found indistinguishable from random initialisation, because the pool_mlp head
barely moves at lr=1e-4 while the backbone carries all 15.18M params of adaptation
(`results/xai/tierA_head_filter_response.json`). Input-space methods are the only
route.

Usage:
    python codebrain_finetune/src/run_xai_attribution.py \
        --checkpoint checkpoints/xai/xai_seed1__toast_to_tea__full_ft__seed1.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest  # noqa: E402
from codebrain_ft import (  # noqa: E402
    load_finetune_config, seed_everything, cohort_median_scale,
)
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import duration_match, make_frame_loader  # noqa: E402
from codebrain_ft.xai import integrated_gradients, load_xai_checkpoint  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402

log = logging.getLogger("xai_attribution")

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
            "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_test_frame(payload: dict, sidecar: pd.DataFrame, ts_cfg) -> pd.DataFrame:
    """Rebuild EXACTLY the evaluation frame the checkpoint was scored on.

    Reads the task, matching and eligibility settings out of the checkpoint rather
    than re-deriving them, so an attribution run cannot silently explain a
    different window set from the one behind the reported metric. The window count
    is asserted against the checkpoint's own record.
    """
    _, test_task = payload["direction"].split("_to_", 1)
    eligible = list(payload["eligible_patients"])
    df = sidecar[(sidecar["patient_id"].isin(eligible)) & (sidecar["task_group"] == test_task)]
    if bool(ts_cfg.protocol["eval_nonoverlap_only"]):
        df = df[df["is_nonoverlap_subset"]]
    df = df.reset_index(drop=True)
    if payload.get("match_level", "none") != "none" or payload.get("match_k") is not None:
        df = duration_match(df, level=payload["match_level"], k=payload.get("match_k"),
                            where=payload["match_where"])
    if len(df) != payload["n_test_windows"]:
        raise RuntimeError(
            f"rebuilt test frame has {len(df)} windows but the checkpoint was scored on "
            f"{payload['n_test_windows']} — refusing to attribute a different window set")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier B/C attribution on a saved checkpoint")
    ap.add_argument("--checkpoint", required=True, help="path relative to codebrain_finetune/ or absolute")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--baselines", nargs="+", default=["zero", "trainmean", "phaserand"],
                    choices=["zero", "trainmean", "phaserand"],
                    help="IG baselines. Run all three: baseline choice is the dominant "
                         "source of IG disagreement and is treated as a sensitivity "
                         "analysis, not a default (XAI_PLAN.md B1).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — the SSSM backbone requires GPU. Aborting.")
        return 1
    device = "cuda"

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _RUN_ROOT / ckpt
    if not ckpt.is_file():
        log.error("Checkpoint not found: %s", ckpt)
        return 1

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    seed_everything(ft_cfg.seed.base, deterministic=False)

    model, payload = load_xai_checkpoint(ckpt, data_cfg, ft_cfg, device=device)
    tag = args.out_tag or ckpt.stem
    out_dir = _RUN_ROOT / "results" / "xai" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    frame = build_test_frame(payload, sidecar, ts_cfg)
    log.info("Attribution set: %d windows / %d patients (task=%s)",
             len(frame), frame.patient_id.nunique(), payload["direction"].split("_to_")[1])

    manifest = CohortManifest.from_config(data_cfg)
    norm_scale = payload.get("norm_scale") or cohort_median_scale(
        manifest, data_cfg, seed=ft_cfg.seed.base)
    loader = make_frame_loader(frame, data_cfg, shuffle=False, seed=1,
                              input_norm=payload["input_norm"], norm_scale=norm_scale)

    t0 = time.time()
    record = {
        "checkpoint": str(ckpt.relative_to(_RUN_ROOT)),
        "direction": payload["direction"], "arm": payload["arm"],
        "seed_base": payload["seed_base"], "input_norm": payload["input_norm"],
        "match_level": payload.get("match_level"), "match_where": payload.get("match_where"),
        "n_windows": int(len(frame)), "n_patients": int(frame.patient_id.nunique()),
        "channels": CHANNELS, "ig_steps": args.ig_steps,
        "git_sha": git_sha(_PROJECT), "env": env_versions(),
        "caveat": ("Model scores 0.832 on this design but 0.435 leak-free (Exp 13c); "
                   "recid=0.920 on the same windows. Maps are candidate generators for "
                   "Tier D/E, not biomarkers."),
    }

    # ---- collect the raw windows once (they fit: 1402 x 18 x 10 x 200 fp32 ~ 200 MB)
    # Loader contract is (x, y, info) where info is a dict of collated lists —
    # same unpacking as train.predict_windows, not positional tensors.
    xs, ys, pids, conds = [], [], [], []
    for x, y, info in loader:
        xs.append(x.cpu())
        ys.append(y.numpy())
        pids.extend(info["patient_id"])
        conds.extend(info["condition"])
    X = torch.cat(xs)
    y_true = np.concatenate(ys)
    pids = np.asarray(pids)
    conds = np.asarray(conds)
    if len(X) != len(frame):
        raise RuntimeError(f"loader yielded {len(X)} windows, frame has {len(frame)}")
    log.info("Loaded windows: %s", tuple(X.shape))

    # ---- Tier B: Integrated Gradients under each baseline -------------------
    attrs = {}
    for bl in args.baselines:
        if bl == "zero":
            base = torch.zeros(1, *X.shape[1:])
        elif bl == "trainmean":
            base = X.mean(dim=0, keepdim=True)
        else:  # phaserand — preserve amplitude spectrum, destroy phase
            spec = np.fft.rfft(X.numpy(), axis=-1)
            ph = np.exp(1j * np.random.default_rng(1).uniform(0, 2 * np.pi, spec.shape))
            base = torch.from_numpy(
                np.fft.irfft(np.abs(spec) * ph, n=X.shape[-1], axis=-1)).float().mean(0, keepdim=True)
        chunks = []
        for i in range(0, len(X), args.batch_size):
            chunks.append(integrated_gradients(
                model, X[i:i + args.batch_size], baseline=base,
                steps=args.ig_steps, device=device).cpu())
        A = torch.cat(chunks).numpy()
        attrs[bl] = A
        log.info("IG[%s] done: %s (%.1f s elapsed)", bl, A.shape, time.time() - t0)
        np.save(out_dir / f"ig_{bl}.npy", A.astype(np.float32))

    # ---- aggregate: per-channel, per-patch, signed and absolute -------------
    agg = {}
    for bl, A in attrs.items():
        agg[bl] = {
            "per_channel_abs": np.abs(A).sum(axis=(2, 3)).mean(axis=0),
            "per_channel_signed": A.sum(axis=(2, 3)).mean(axis=0),
            "per_patch_abs": np.abs(A).sum(axis=(1, 3)).mean(axis=0),
        }
        for p in np.unique(pids):
            m = pids == p
            agg[bl].setdefault("per_patient_channel_abs", {})[str(p)] = \
                np.abs(A[m]).sum(axis=(2, 3)).mean(axis=0)

    # baseline agreement — if these disagree, IG is unstable here and Tier C rules
    keys = list(agg)
    if len(keys) > 1:
        from scipy.stats import spearmanr
        record["baseline_agreement"] = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                r = float(spearmanr(agg[keys[i]]["per_channel_abs"],
                                    agg[keys[j]]["per_channel_abs"]).statistic)
                record["baseline_agreement"][f"{keys[i]}_vs_{keys[j]}"] = r
                log.info("baseline agreement %s vs %s: Spearman r=%.3f", keys[i], keys[j], r)

    record["aggregates"] = {bl: {k: (v if not isinstance(v, dict) else
                                     {kk: vv for kk, vv in v.items()})
                                 for k, v in d.items()} for bl, d in agg.items()}
    np.savez_compressed(out_dir / "aggregates.npz",
                        pids=pids, conds=conds, y_true=y_true,
                        **{f"{bl}__{k}": np.asarray(v) for bl, d in agg.items()
                           for k, v in d.items() if not isinstance(v, dict)})
    (out_dir / "attribution_summary.json").write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8")

    log.info("Wrote %s (%.1f s total)", out_dir, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
