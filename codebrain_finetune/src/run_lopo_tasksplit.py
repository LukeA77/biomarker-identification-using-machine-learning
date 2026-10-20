"""LOPO + TASK-SPLIT — the design the supervisor actually described.

WHY THIS EXISTS
---------------
Every task-split arm run so far — per-patient (``run_within_patient_tasksplit.py``)
*and* pooled (``run_pooled_tasksplit.py``) — has the held-out patient's own two
recordings in the TRAINING set. Within a patient the ON/OFF label is perfectly
collinear with recording identity, so in both of those designs a model can score
by answering "which of this patient's two recordings is this window from?"
instead of anything about levodopa. Experiment 12c measured that directly: an
18-number electrode-contact amplitude fingerprint reaches 0.934 on the
per-patient split, and Experiment 13 shows the leak survives ``chan_z``
(``recid`` = 0.924).

This driver closes that route:

    Train on **other** patients' tea/kettle windows -> test on the **held-out**
    patient's toast windows. Leave-one-patient-out over all eligible patients.
    Then the reverse direction (train toast -> test tea).

The held-out patient's recordings are never seen in training, so a per-recording
fingerprint is worthless — exactly as in the cross-subject LOPO arms — while the
task split still holds (train and test are different activities). This is the
only version in which the recording-fingerprint route is genuinely closed, and it
is the honest answer to "is the session confound less prominent when pooled?".

EXPECT THIS NEAR CHANCE. The cross-subject LOPO reference on this cohort is
0.443 (frozen probe) / 0.557 (LoRA), both consistent with chance. A null here is
a legitimate, reportable outcome and is NOT to be tuned away: no hyper-parameter
is re-searched, the LoRA HPs are reused from the pooled LOPO search exactly as in
the other task-split drivers, and every fold is written out whatever it says.

STATISTICS match the LOPO arms, not the per-patient task-split arms: session-level
aggregation (patient x condition -> one mean probability), patient-cluster
bootstrap CI, and a within-subject permutation test. That makes the numbers
directly comparable to the 0.443 / 0.557 cross-subject reference.

FROZEN-PROBE FEATURE CACHE. The backbone is frozen in that arm, so its features
depend only on (window, input-norm) and not on the fold. They are extracted once
per normalisation and reused across all 27 folds — the same computation as
per-fold extraction, though not bit-identical: bfloat16 autocast makes cuBLAS
kernel choice batch-shape dependent, worth ~0.4% relative on the features and
~0.002 on a cohort-mean balanced accuracy. Measured, not assumed — see
``codebrain_ft.backbone_cache``.

CONTROLS under the identical protocol: ``chanamp`` (electrode-contact amplitude),
``relbp`` and ``aperiodic`` (the shape controls that survive ``chan_z``).
``recid`` has NO form here and is deliberately absent rather than faked: its
target is "which of THIS patient's two recordings", which is undefined once the
test patient is unseen. Under LOPO, asking "which recording" and asking "ON or
OFF" stop being the same question — that is precisely what this design fixes.

Usage:
    # smoke test first:
    python codebrain_finetune/src/run_lopo_tasksplit.py \
        --arms frozen_probe --max-patients 3 --max-epochs 3 --run-tag smoke
    # full run, both normalisations:
    python codebrain_finetune/src/run_lopo_tasksplit.py --input-norm none
    python codebrain_finetune/src/run_lopo_tasksplit.py --input-norm chan_z
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent                 # codebrain_finetune/
_PROJECT = _RUN_ROOT.parent                     # Luke_Dissertation/
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest  # noqa: E402
from codebrain_ft import (  # noqa: E402
    load_finetune_config, build_classifier_model, train_model, predict_windows,
    seed_everything, fold_seed, INPUT_NORM_SCHEMES, cohort_median_scale,
    load_controls_config, load_or_compute_features, control_matrix,
)
from codebrain_ft.backbone_cache import load_or_compute_backbone_features  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import (  # noqa: E402
    select_task_frame, train_val_split_by_time, make_frame_loader, direction_pairs,
)
from codebrain_ft.metrics import (  # noqa: E402
    classification_metrics, aggregate_session_level, bootstrap_session_ci,
    permutation_test_balanced_accuracy,
)
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402

log = logging.getLogger("lopo_tasksplit")

EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174

MODEL_ARMS = ("frozen_probe", "lora", "full_ft")
CONTROL_ARMS = ("chanamp", "relbp", "aperiodic")

# Cross-subject LOPO reference (no task split, pooled over 70 sessions) — the
# comparison this experiment exists to be read against.
# Source: EXPERIMENTS.md / results/runs_master.csv.
LOPO_REFERENCE = {
    "frozen_probe": {"balanced_accuracy": 0.443, "ci": [0.357, 0.529], "auc": 0.482, "perm_p": 0.95},
    "lora":         {"balanced_accuracy": 0.557, "ci": [0.457, 0.643], "auc": 0.541, "perm_p": 0.194},
    "full_ft":      {"balanced_accuracy": 0.443, "ci": [0.357, 0.529], "auc": 0.423, "perm_p": 0.945},
}

# Session-level schema, like the LOPO arms — NOT the patient-level schema of
# runs_master_tasksplit.csv, so this experiment gets its own master CSV rather
# than writing values under columns that mean something else.
_MASTER_CSV_FIELDS = [
    "run_tag", "arm", "input_norm", "direction", "git_sha", "n_folds", "n_sessions",
    "session_balanced_accuracy", "ci_low", "ci_high", "session_auc",
    "permutation_p", "null_mean",
]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _with_hp(ft_cfg, *, r: int, dropout: float):
    lora = dataclasses.replace(ft_cfg.lora, r=int(r), dropout=float(dropout))
    return dataclasses.replace(ft_cfg, lora=lora)


def _with_monitor(ft_cfg, monitor: str):
    es = dataclasses.replace(ft_cfg.train.early_stopping, monitor=monitor)
    train = dataclasses.replace(ft_cfg.train, early_stopping=es)
    return dataclasses.replace(ft_cfg, train=train)


def _resolve_lora_hp(ts_cfg, ft_cfg) -> dict:
    """Reuse the LOPO pooled HP search — NOT re-searched.

    Same reasoning as the existing task-split drivers: re-searching would make
    these arms incomparable to both the LOPO numbers and the other task-split
    numbers, and this experiment exists precisely to be compared to them.
    """
    spec = ts_cfg.arms.get("lora", {})
    rel = spec.get("reuse_hp_from")
    if rel:
        p = _RUN_ROOT / rel
        if p.is_file():
            cfg = _read_json(p)["best_config"]
            log.info("LoRA HPs reused from %s: %s (NOT re-searched).", rel, cfg)
            return cfg
        log.warning("LoRA HP source %s not found; falling back to tasksplit.yaml literals.", p)
    fb = spec.get("fallback") or {"r": ft_cfg.lora.r, "lr": ft_cfg.train.lr,
                                  "dropout": ft_cfg.lora.dropout}
    log.warning("LoRA HPs = tasksplit.yaml fallback %s", fb)
    return dict(fb)


# ---------------------------------------------------------------------------
# frames for one fold
# ---------------------------------------------------------------------------
def _fold_frames(sidecar: pd.DataFrame, held_out: str, train_patients: list[str],
                 train_task: str, test_task: str, ts_cfg):
    """(train, test) frames: OTHER patients' train task -> held-out patient's test task."""
    proto = ts_cfg.protocol
    tr = sidecar[(sidecar["patient_id"].isin(train_patients)) &
                 (sidecar["task_group"] == train_task)]
    if bool(proto["train_nonoverlap_only"]):
        tr = tr[tr["is_nonoverlap_subset"]]
    tr = tr.reset_index(drop=True)
    # The held-out patient is validated with the same fail-loud checks the
    # per-patient design uses (both conditions present, no duplicated paths).
    te = select_task_frame(sidecar, held_out, test_task,
                           nonoverlap_only=bool(proto["eval_nonoverlap_only"]))
    if held_out in set(tr["patient_id"]):
        raise RuntimeError(f"fold {held_out}: held-out patient leaked into the training frame")
    overlap = set(tr["path"]) & set(te["path"])
    if overlap:
        raise RuntimeError(f"fold {held_out}: {len(overlap)} windows in both train and test")
    if len(tr) == 0:
        raise ValueError(f"fold {held_out}: no training windows for task {train_task}")
    return tr, te


# ---------------------------------------------------------------------------
# one (fold, direction, arm) run
# ---------------------------------------------------------------------------
def run_one(*, arm, held_out, train_patients, train_task, test_task, sidecar, data_cfg,
            ft_cfg, ts_cfg, cc, lora_hp, out_path: Path, device, max_epochs, common,
            input_norm, norm_scale, backbone_X, control_blocks) -> dict:
    if out_path.exists():
        rec = _read_json(out_path)
        log.info("%-12s %-9s %s->%-6s | RESUME (exists) | session bal-acc %s",
                 arm, held_out, train_task, test_task,
                 f"{rec['session']['balanced_accuracy']:.3f}")
        return rec

    tr_frame, te_frame = _fold_frames(sidecar, held_out, train_patients,
                                      train_task, test_task, ts_cfg)
    run_key = f"{held_out}|{train_task}|{test_task}|{arm}|{input_norm}".encode()
    seed = fold_seed(ft_cfg.seed.base, zlib.crc32(run_key) % 10_000)
    thr = float(ts_cfg.stats["tests"]["chance_level"])
    t0 = time.time()
    extra: dict = {}
    peak = 0

    if arm in CONTROL_ARMS or arm == "frozen_probe":
        # Linear head on precomputed features (backbone frozen, or a control
        # feature block). Standardiser + logistic regression are fitted on THIS
        # fold's training rows only.
        if arm == "frozen_probe":
            X = backbone_X
            clf = LogisticRegression(C=ft_cfg.probe.C, max_iter=ft_cfg.probe.max_iter,
                                     class_weight="balanced")
            standardize = bool(ft_cfg.probe.standardize)
        else:
            X = control_matrix(control_blocks, arm, cc)
            clf = LogisticRegression(max_iter=int(cc.classifier["max_iter"]),
                                     class_weight=cc.classifier.get("class_weight", "balanced"))
            standardize = bool(cc.classifier.get("standardize", True))
        itr, ite = tr_frame["_row"].to_numpy(), te_frame["_row"].to_numpy()
        Xtr, Xte = X[itr], X[ite]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ytr = tr_frame["y_cond"].to_numpy().astype(int)
        yte = te_frame["y_cond"].to_numpy().astype(int)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        n_train_used, n_val_used = len(tr_frame), 0
        extra = {"n_features": int(X.shape[1])}

    elif arm in {"lora", "full_ft"}:
        import torch
        proto = ts_cfg.protocol
        ft_eff = _with_hp(ft_cfg, r=lora_hp["r"], dropout=lora_hp["dropout"])
        ft_eff = _with_monitor(ft_eff, proto["early_stopping_monitor"])
        lr = lora_hp["lr"] if arm == "lora" else ft_cfg.train.lr_full_ft

        tr_sub, va_sub = train_val_split_by_time(
            tr_frame, val_frac=float(proto["val_frac"]),
            stratify_by_session=bool(proto["val_stratify_by_session"]))
        n_train_used, n_val_used = len(tr_sub), len(va_sub)

        seed_everything(seed, deterministic=ft_eff.seed.deterministic)
        torch.cuda.reset_peak_memory_stats()
        model = build_classifier_model(data_cfg, ft_eff, mode=arm, device=device)
        try:
            tr_loader = make_frame_loader(tr_sub, data_cfg, shuffle=True, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            va_loader = make_frame_loader(va_sub, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            te_loader = make_frame_loader(te_frame, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            train_res = train_model(model, tr_loader, va_loader, ft_eff, lr=lr,
                                    device=device, max_epochs=max_epochs)
            prob, yte, _, _ = predict_windows(model, te_loader, device=device,
                                              precision=ft_eff.train.precision)
            peak = int(torch.cuda.max_memory_allocated())
        finally:
            del model
            torch.cuda.empty_cache()
        extra = {"best_epoch": train_res.best_epoch, "best_val_monitor": train_res.best_monitor,
                 "monitor": proto["early_stopping_monitor"],
                 "trainable_params": train_res.trainable_params, "lr": lr,
                 "train_seconds": train_res.train_seconds}
    else:
        raise ValueError(f"unknown arm {arm!r}")

    window = classification_metrics(yte, prob, threshold=thr)
    sess = aggregate_session_level(te_frame["patient_id"].to_numpy(),
                                   te_frame["condition"].to_numpy(), prob, yte, agg="mean_prob")
    session = classification_metrics(sess.y_true, sess.prob, threshold=thr)
    record = {
        **common, "arm": arm, "test_patient": held_out, "n_train_patients": len(train_patients),
        "train_task": train_task, "test_task": test_task,
        "direction": f"{train_task}_to_{test_task}", "seed": seed,
        "n_train_windows": n_train_used, "n_val_windows": n_val_used,
        "n_test_windows": len(te_frame),
        "window": window.as_dict(), "session": session.as_dict(),
        "session_scores": {"sessions": sess.sessions.tolist(), "patients": sess.patients.tolist(),
                           "prob": sess.prob.tolist(), "y_true": sess.y_true.tolist(),
                           "n_windows": sess.n_windows.tolist()},
        "input_norm": input_norm, "norm_scale": norm_scale,
        "wall_seconds": time.time() - t0, "peak_vram_bytes": peak, **extra,
    }
    _write_json(out_path, record)
    log.info("%-12s %-9s %s->%-6s | window bal-acc %.3f | session bal-acc %.3f | "
             "train n=%d (%d patients) test n=%d | %.0fs",
             arm, held_out, train_task, test_task, window.balanced_accuracy,
             session.balanced_accuracy, n_train_used, len(train_patients),
             len(te_frame), record["wall_seconds"])
    return record


# ---------------------------------------------------------------------------
# aggregation — session level, exactly like the LOPO arms
# ---------------------------------------------------------------------------
def aggregate_arm(records: list[dict], ft_cfg, *, arm: str, direction: str | None) -> dict:
    sel = [r for r in records if direction is None or r["direction"] == direction]
    prob, true, pat = [], [], []
    for r in sorted(sel, key=lambda r: (r["direction"], r["test_patient"])):
        sc = r["session_scores"]
        prob.extend(sc["prob"]); true.extend(sc["y_true"]); pat.extend(sc["patients"])
    prob, true, pat = np.asarray(prob), np.asarray(true), np.asarray(pat)
    agg = classification_metrics(true, prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(true, prob, pat, n_resamples=ft_cfg.eval.bootstrap.n_resamples,
                                ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base)
    perm = permutation_test_balanced_accuracy(
        true, prob, pat, n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset)
    return {
        "arm": arm, "direction": direction or "both_directions_pooled",
        "n_folds": len(sel), "n_sessions": int(len(true)),
        "aggregate": agg.as_dict(), "bootstrap": boot, "permutation": perm,
        "per_fold": [{"test_patient": r["test_patient"], "direction": r["direction"],
                      "session_bal_acc": r["session"]["balanced_accuracy"],
                      "window_bal_acc": r["window"]["balanced_accuracy"],
                      "prob": [round(p, 3) for p in r["session_scores"]["prob"]],
                      "y_true": r["session_scores"]["y_true"]} for r in sel],
        "lopo_reference": LOPO_REFERENCE.get(arm),
    }


def _write_report(path: Path, summaries: dict, run_tag: str, input_norm: str) -> None:
    L = [f"# LOPO + task split — results ({run_tag}, input_norm={input_norm})\n\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
         "Train on OTHER patients' windows of one ADL task; test on the HELD-OUT patient's\n"
         "windows of the other task. The held-out patient's recordings are never in training,\n"
         "so a per-recording fingerprint cannot help — this is the only task-split variant in\n"
         "which that route is closed. Session-level statistics, patient-cluster bootstrap,\n"
         "within-subject permutation — the same as the cross-subject LOPO arms, so these\n"
         "numbers are directly comparable to the 0.443 / 0.557 reference.\n\n"]
    L.append("| arm | direction | folds | sessions | session bal-acc | 95% CI | AUC | perm p | "
             "*cross-subject LOPO* |\n|---|---|---|---|---|---|---|---|---|\n")
    for key, s in summaries.items():
        a, b, p = s["aggregate"], s["bootstrap"], s["permutation"]
        ref = (s.get("lopo_reference") or {}).get("balanced_accuracy")
        L.append(f"| {s['arm']} | {s['direction']} | {s['n_folds']} | {s['n_sessions']} | "
                 f"**{a['balanced_accuracy']:.3f}** | "
                 f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}] | "
                 f"{a['auc']:.3f} | {p['p_value']:.3f} | "
                 f"{f'*{ref:.3f}*' if ref else '—'} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L), encoding="utf-8")
    log.info("Wrote markdown report -> %s", path)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="LOPO + task-split experiment")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=list(MODEL_ARMS) + list(CONTROL_ARMS))
    ap.add_argument("--input-norm", default="none", choices=list(INPUT_NORM_SCHEMES))
    ap.add_argument("--max-patients", type=int, default=None, help="smoke test: first N patients")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    run_tag = args.run_tag or f"lopo_tasksplit_{args.input_norm}"
    log_txt = _RUN_ROOT / "logs" / f"{run_tag}_{time.strftime('%Y%m%d')}.log"
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_txt, encoding="utf-8")],
    )

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — SSSM requires GPU. Aborting.")
        return 1
    device = "cuda"

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    cc = load_controls_config()
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d), expected (%d, %d). STOP.",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    results_dir = _RUN_ROOT / "results" / run_tag
    pf_path = _RUN_ROOT / "results" / ts_cfg.output["results_subdir"] / "preflight_summary.json"
    if not pf_path.is_file():
        log.error("PRE-FLIGHT MISSING: %s. Run derive_task_windows.py first.", pf_path)
        return 1
    pf = _read_json(pf_path)
    eligible = list(pf["eligible_patients"])
    if args.max_patients:
        eligible = eligible[: args.max_patients]
        log.warning("SMOKE TEST: restricted to %d patient(s): %s", len(eligible), eligible)
    if len(eligible) < 3:
        log.error("LOPO needs >=3 patients; got %d.", len(eligible))
        return 1

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)
    sidecar["_row"] = np.arange(len(sidecar))
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])
    arms = args.arms or list(MODEL_ARMS) + list(CONTROL_ARMS)
    max_epochs = args.max_epochs or ft_cfg.train.max_epochs
    lora_hp = _resolve_lora_hp(ts_cfg, ft_cfg)

    norm_scale = None
    if args.input_norm != "none":
        norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        log.info("Input normalisation: %s (rescaled to cohort-median window std %.6f)",
                 args.input_norm, norm_scale)

    # Precomputed feature matrices — one pass each, reused by every fold.
    backbone_X = None
    if "frozen_probe" in arms:
        backbone_X, bmeta = load_or_compute_backbone_features(
            sidecar, data_cfg, ft_cfg, scheme=args.input_norm, norm_scale=norm_scale,
            cache_dir=(_RUN_ROOT / "results" / ts_cfg.output["results_subdir"] /
                       "backbone_features"),
            device=device)
        log.info("Frozen-backbone features: %s (%s)", backbone_X.shape, bmeta["cache"])
    control_blocks = None
    if set(arms) & set(CONTROL_ARMS):
        control_blocks, _ = load_or_compute_features(
            sidecar, data_cfg, cc, scheme=args.input_norm,
            norm_scale=norm_scale, run_root=_RUN_ROOT)

    log.info("Run '%s' | arms=%s | %d folds x %d directions | input_norm=%s | max_epochs=%d",
             run_tag, arms, len(eligible), len(pairs), args.input_norm, max_epochs)

    common = dict(
        run_tag=run_tag, git_sha=git_sha(_PROJECT),
        checkpoint_sha256=file_sha256(data_cfg.paths.checkpoint), env=env_versions(),
        cohort={"patients": n_pat, "windows": n_win},
        experiment="lopo_tasksplit", lora_target=ft_cfg.lora.target, lora_hp=lora_hp,
        preflight={"n_eligible": len(pf["eligible_patients"]), "n_cohort": pf["n_cohort_patients"]},
    )

    summaries: dict[str, dict] = {}
    t_start = time.time()
    for arm in arms:
        arm_dir = results_dir / arm
        records = []
        for held_out in eligible:
            train_patients = [q for q in eligible if q != held_out]
            for train_task, test_task in pairs:
                out_path = arm_dir / f"{held_out}_{train_task}_to_{test_task}.json"
                records.append(run_one(
                    arm=arm, held_out=held_out, train_patients=train_patients,
                    train_task=train_task, test_task=test_task, sidecar=sidecar,
                    data_cfg=data_cfg, ft_cfg=ft_cfg, ts_cfg=ts_cfg, cc=cc, lora_hp=lora_hp,
                    out_path=out_path, device=device, max_epochs=max_epochs, common=common,
                    input_norm=args.input_norm, norm_scale=norm_scale,
                    backbone_X=backbone_X, control_blocks=control_blocks))

        for direction in [f"{a}_to_{b}" for a, b in pairs] + [None]:
            s = aggregate_arm(records, ft_cfg, arm=arm, direction=direction)
            summaries[f"{arm}|{s['direction']}"] = s
            a, b, p = s["aggregate"], s["bootstrap"], s["permutation"]
            log.info("ARM %-12s %-24s | %d sessions | bal-acc %.3f [%.3f, %.3f] | AUC %.3f | "
                     "perm p=%.3f (null %.3f)", arm, s["direction"], s["n_sessions"],
                     a["balanced_accuracy"], b["ci_low"], b["ci_high"], a["auc"],
                     p["p_value"], p["null_mean"])
            save_run({
                **common, "arm": arm, "input_norm": args.input_norm,
                "direction": s["direction"], "n_folds": s["n_folds"],
                "n_sessions": s["n_sessions"],
                "session_balanced_accuracy": a["balanced_accuracy"],
                "ci_low": b["ci_low"], "ci_high": b["ci_high"], "session_auc": a["auc"],
                "permutation_p": p["p_value"], "null_mean": p["null_mean"], **s,
            }, arm_dir, _RUN_ROOT / "results" / "runs_master_lopo_tasksplit.csv",
                csv_fields=_MASTER_CSV_FIELDS)

        _write_json(arm_dir / "summary.json",
                    {k: v for k, v in summaries.items() if k.startswith(f"{arm}|")})

    _write_json(results_dir / f"{run_tag}_summary.json",
                {"run_tag": run_tag, "input_norm": args.input_norm, "norm_scale": norm_scale,
                 "arms": summaries, "eligible_patients": eligible,
                 "lopo_reference": LOPO_REFERENCE,
                 "git_sha": git_sha(_PROJECT), "env": env_versions()})
    _write_report(_RUN_ROOT / "logs" / f"{run_tag}_report.md", summaries, run_tag, args.input_norm)
    log.info("Done in %.1f min. Results under %s", (time.time() - t_start) / 60, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
