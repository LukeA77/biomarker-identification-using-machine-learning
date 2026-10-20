"""POOLED within-patient task-split — one shared model across all patients.

A second way of running the same task-split idea as ``run_within_patient_tasksplit.py``,
requested alongside it for comparison (2026-07-30): instead of N separate
per-patient models whose scores get averaged, pool every eligible patient's
train-task windows into ONE training set, fit ONE model, and evaluate it on
every eligible patient's test-task windows pooled the same way. Both directions
(tea->toast, toast->tea) are still run and reported SEPARATELY (not merged into
one combined train/test set), so the two designs stay comparable action-for-
action: same task groups, same eval-window convention, same arms.

WHAT THIS DOES AND DOES NOT ADDRESS
------------------------------------
Pooling does NOT remove the fundamental confound documented in
WITHIN_PATIENT_TASKSPLIT.md and EXPERIMENTS.md Experiment 12: each patient's own
two recordings still appear on both the train and test side (just merged in with
everyone else's), so the ON/OFF label is still collinear with recording identity
for every individual patient inside the pool. What pooling changes is whether the
model can calibrate to each patient's OWN fingerprint individually, or has to
find one rule that works for all patients simultaneously. There is one concrete
data point suggesting this matters for simple/linear features: running the same
18-number amplitude-only features under true cross-patient LOPO (never seeing the
test patient's recordings at all) collapsed from 0.919 to 0.460 (chance). Pooled-
by-task is not the same setup as that (nobody is fully held out here), so do NOT
read an elevated pooled number here as "the confound is fixed" -- CodeBrain's
representations are already known (Experiments 4b/11) to be dominated by subject
identity even without any task split, so a flexible enough model may still find
an indirect route to the same shortcut. This run exists to have a number to
compare and discuss, not to resolve the confound.

ARMS: frozen_probe, lora, full_ft (same LoRA HPs reused from the LOPO search, not
re-searched -- same reasoning as the per-patient driver). PLUS the ``chanamp``
amplitude-fingerprint control, run under the identical pooled protocol, since
without it there is nothing to compare the pooled CodeBrain numbers against.

EARLY STOPPING: the validation carve-out (``train_val_split_by_time``) now groups
by the real SESSION key (patient x condition), not by condition alone, so pooling
multiple patients' recordings doesn't interleave their independent elapsed-time
axes -- see that function's docstring for why this matters once ``frame`` spans
more than one patient.

INPUT NORMALISATION (added 2026-08-01, supervisor request): ``--input-norm``
matches ``run_within_patient_tasksplit.py`` exactly -- same flag name, same
choices, same ``norm_scale`` (the cohort-median window std from
``cohort_median_scale``), same recording of ``input_norm``/``norm_scale`` in the
output JSON. ``chan_z`` z-scores each channel of each window, which is the
supervisor's proposed fix for the amplitude fingerprint. It applies to the
``chanamp`` control too, via the SAME normalised windows the model arms see.
Results are written under ``results/<run-tag>/`` so a normalised run cannot
resume from -- or overwrite -- the raw-input run's JSONs.

Usage:
    python codebrain_finetune/src/run_pooled_tasksplit.py
    python codebrain_finetune/src/run_pooled_tasksplit.py --arms frozen_probe --max-epochs 3
    python codebrain_finetune/src/run_pooled_tasksplit.py --input-norm chan_z \
        --run-tag pooled_tasksplit_chanz
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
    load_finetune_config, build_classifier_model,
    train_model, evaluate_session_level, predict_windows,
    extract_features, seed_everything, fold_seed,
    INPUT_NORM_SCHEMES, cohort_median_scale, load_controls_config,
)
from codebrain_ft.control_features import chanamp_features  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import (  # noqa: E402
    train_val_split_by_time, make_frame_loader, direction_pairs,
    aggregate_across_patients, ordering_stats,
    duration_match, match_report, MATCH_LEVELS, MATCH_WHERE,
)
from codebrain_ft.metrics import aggregate_session_level, classification_metrics  # noqa: E402
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("pooled_tasksplit")

EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174

# Per-patient-design reference (same 27-patient cohort, per-patient-averaged
# protocol) for the side-by-side comparison table.
PER_PATIENT_REFERENCE_PATH = _RUN_ROOT / "results" / "within_patient_tasksplit"

# Unchanged on purpose: matched and unmatched runs must land in the SAME master
# table to be comparable, and `run_tag` already distinguishes them (the guard in
# tracking.save_run rejects a schema change on an existing CSV). Full matching
# detail — level, where, per-cell K, retention — lives in each run's JSON under
# "match".
_MASTER_CSV_FIELDS = [
    "run_tag", "arm", "direction", "git_sha", "n_patients_test",
    "pooled_window_balanced_accuracy", "pooled_session_balanced_accuracy",
    "patient_mean_balanced_accuracy", "ci_low", "ci_high", "ordering_frac",
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


def _save_checkpoint(model, *, ckpt_dir, run_tag, direction, arm, seed, seed_base,
                     common, train_res, ft_eff, n_train, n_test):
    """Persist trained weights + everything needed to rebuild and trust them.

    ``train_model`` reloads the best-epoch state before returning, so ``model``
    here IS the early-stopped model whose metrics are reported — not the last
    epoch. The provenance block matters as much as the tensors: without the
    checkpoint sha, git sha and the exact match/norm settings, a later XAI run
    cannot prove it is explaining the model that produced the published number.
    """
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{run_tag}__{direction}__{arm}__seed{seed_base}.pt"
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "run_tag": run_tag, "direction": direction, "arm": arm,
        "seed": int(seed), "seed_base": int(seed_base),
        "deterministic": bool(ft_eff.seed.deterministic),
        "input_norm": common["input_norm"], "norm_scale": common["norm_scale"],
        "match_level": common["match_level"], "match_where": common["match_where"],
        "match_k": common["match_k"],
        "lora_hp": common["lora_hp"], "head_type": ft_eff.model.head.type,
        "eligible_patients": common["eligible_patients"],
        "n_train_windows": int(n_train), "n_test_windows": int(n_test),
        "best_epoch": train_res.best_epoch, "best_monitor": train_res.best_monitor,
        "trainable_params": train_res.trainable_params,
        "git_sha": common["git_sha"], "checkpoint_sha256": common["checkpoint_sha256"],
        "env": common["env"],
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("  saved checkpoint -> %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def _with_hp(ft_cfg, *, r: int, dropout: float):
    lora = dataclasses.replace(ft_cfg.lora, r=int(r), dropout=float(dropout))
    return dataclasses.replace(ft_cfg, lora=lora)


def _with_monitor(ft_cfg, monitor: str):
    es = dataclasses.replace(ft_cfg.train.early_stopping, monitor=monitor)
    train = dataclasses.replace(ft_cfg.train, early_stopping=es)
    return dataclasses.replace(ft_cfg, train=train)


def _resolve_lora_hp(ts_cfg, ft_cfg) -> dict:
    spec = ts_cfg.arms.get("lora", {})
    rel = spec.get("reuse_hp_from")
    if rel:
        p = _RUN_ROOT / rel
        if p.is_file():
            cfg = _read_json(p)["best_config"]
            log.info("LoRA HPs reused from %s: %s", rel, cfg)
            return cfg
    fb = spec.get("fallback") or {"r": ft_cfg.lora.r, "lr": ft_cfg.train.lr, "dropout": ft_cfg.lora.dropout}
    log.warning("LoRA HPs = tasksplit.yaml fallback %s", fb)
    return dict(fb)


def _select_pooled_frame(sidecar: pd.DataFrame, patients: list[str], task_group: str,
                         *, nonoverlap_only: bool) -> pd.DataFrame:
    """All eligible patients' windows for one task group, pooled.

    Same correctness checks as the per-patient ``select_task_frame`` (both
    conditions present, no duplicated window paths), just over the whole pool.
    """
    df = sidecar[(sidecar["patient_id"].isin(patients)) & (sidecar["task_group"] == task_group)]
    if nonoverlap_only:
        df = df[df["is_nonoverlap_subset"]]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"pooled/{task_group}: no windows (nonoverlap_only={nonoverlap_only})")
    if df["path"].duplicated().any():
        raise ValueError(f"pooled/{task_group}: duplicated window paths across the pool")
    missing_cond = [p for p in patients
                    if set(df.loc[df.patient_id == p, "condition"].unique()) != {"A", "C"}]
    if missing_cond:
        raise ValueError(f"pooled/{task_group}: patients missing a condition: {missing_cond}")
    return df


# ---------------------------------------------------------------------------
# per-window predictions -> the same statistics as the per-patient design
# ---------------------------------------------------------------------------
def _score(prob: np.ndarray, y_true: np.ndarray, pids: np.ndarray, conds: np.ndarray,
          *, threshold: float, patient_agg: str) -> dict:
    """Turn raw per-window (prob, y_true, patient_id, condition) into the full
    report: pooled window/session metrics, per-patient breakdown + patient-level
    statistics (directly comparable to the per-patient-design headline table),
    and the ordering test.
    """
    window_metrics = classification_metrics(y_true, prob, threshold=threshold)
    sess = aggregate_session_level(pids, conds, prob, y_true, agg=patient_agg)
    session_metrics = classification_metrics(sess.y_true, sess.prob, threshold=threshold)

    # per-patient balanced accuracy on JUST that patient's slice of the pooled
    # test set -- comparable to the per-patient design's headline statistic.
    per_patient_bal_acc: dict[str, float] = {}
    for p in np.unique(pids):
        m = pids == p
        per_patient_bal_acc[str(p)] = classification_metrics(
            y_true[m], prob[m], threshold=threshold).balanced_accuracy

    # ordering: does the OFF session score higher than the ON session, per patient
    records = []
    for p in np.unique(sess.patients):
        m = sess.patients == p
        probs_by_cond = {int(y): pr for y, pr in zip(sess.y_true[m], sess.prob[m])}
        oc = bool(probs_by_cond[1] > probs_by_cond[0]) if set(probs_by_cond) == {0, 1} else None
        records.append({"patient": p, "ordering_correct": oc})

    return {
        "window": window_metrics.as_dict(),
        "session": session_metrics.as_dict(),
        "session_scores": {
            "sessions": sess.sessions.tolist(), "patients": sess.patients.tolist(),
            "prob": sess.prob.tolist(), "y_true": sess.y_true.tolist(),
            "n_windows": sess.n_windows.tolist(),
        },
        "per_patient_balanced_accuracy": per_patient_bal_acc,
        "ordering_records": records,
    }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def run_frozen_probe(data_cfg, ft_cfg, train_frame, test_frame, *, device, threshold, patient_agg,
                     input_norm="none", norm_scale=None):
    import torch
    model = build_classifier_model(data_cfg, ft_cfg, mode="frozen", device=device)
    try:
        tr_loader = make_frame_loader(train_frame, data_cfg, shuffle=False, seed=1,
                                      input_norm=input_norm, norm_scale=norm_scale)
        te_loader = make_frame_loader(test_frame, data_cfg, shuffle=False, seed=1,
                                      input_norm=input_norm, norm_scale=norm_scale)
        Xtr, ytr, _, _ = extract_features(model, tr_loader, ft_cfg, device=device)
        Xte, yte, pids_te, conds_te = extract_features(model, te_loader, ft_cfg, device=device)
        log.info("Pooled probe features: train %s, test %s", Xtr.shape, Xte.shape)
        scaler = StandardScaler().fit(Xtr) if ft_cfg.probe.standardize else None
        if scaler is not None:
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        clf = LogisticRegression(C=ft_cfg.probe.C, max_iter=ft_cfg.probe.max_iter, class_weight="balanced")
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        peak = int(torch.cuda.max_memory_allocated())
    finally:
        del model
        torch.cuda.empty_cache()
    return _score(prob, yte, pids_te, conds_te, threshold=threshold, patient_agg=patient_agg), {"peak_vram_bytes": peak}


def run_trained_arm(data_cfg, ft_cfg, ts_cfg, mode, train_frame, lora_hp, *, max_epochs, device,
                    threshold, patient_agg, seed, input_norm="none", norm_scale=None):
    import torch
    proto = ts_cfg.protocol
    ft_eff = _with_hp(ft_cfg, r=lora_hp["r"], dropout=lora_hp["dropout"])
    ft_eff = _with_monitor(ft_eff, proto["early_stopping_monitor"])
    lr = lora_hp["lr"] if mode == "lora" else ft_cfg.train.lr_full_ft

    tr_frame, va_frame = train_val_split_by_time(
        train_frame, val_frac=float(proto["val_frac"]),
        stratify_by_session=bool(proto["val_stratify_by_session"]))
    log.info("Pooled %s: train %d windows / val %d windows (across %d patients)",
             mode, len(tr_frame), len(va_frame), train_frame.patient_id.nunique())

    seed_everything(seed, deterministic=ft_eff.seed.deterministic)
    torch.cuda.reset_peak_memory_stats()
    model = build_classifier_model(data_cfg, ft_eff, mode=mode, device=device)
    tr_loader = make_frame_loader(tr_frame, data_cfg, shuffle=True, seed=seed,
                                  input_norm=input_norm, norm_scale=norm_scale)
    va_loader = make_frame_loader(va_frame, data_cfg, shuffle=False, seed=seed,
                                  input_norm=input_norm, norm_scale=norm_scale)
    train_res = train_model(model, tr_loader, va_loader, ft_eff, lr=lr, device=device,
                            max_epochs=max_epochs)
    # `model` is returned alive on purpose: the caller still needs it for test-set
    # inference (predict_windows) and frees it there (fresh model + empty_cache
    # per training, matching the LOPO/per-patient drivers' leak-safety convention).
    return model, train_res, ft_eff


def run_chanamp_control(train_frame, test_frame, data_cfg, cc, *, threshold, patient_agg,
                        input_norm="none", norm_scale=None):
    """The amplitude-fingerprint control, under the identical pooled protocol.

    Features come through the same normalised windows the model arms see
    (``chanamp_features`` -> ``NormalizedWindowDataset``), so ``--input-norm``
    applies to the control as well. Under ``chan_z`` this control no longer
    measures an electrode-contact AMPLITUDE fingerprint at all — z-scoring
    removes exactly that — so its value there is a floor for waveform-shape
    leakage, not the amplitude leak. The controls that still speak to the leak
    under normalisation are `relbp`/`aperiodic`/`recid` in
    ``run_tasksplit_controls.py``.
    """
    Xtr = chanamp_features(train_frame, data_cfg, cc, scheme=input_norm, norm_scale=norm_scale)
    Xte = chanamp_features(test_frame, data_cfg, cc, scheme=input_norm, norm_scale=norm_scale)
    ytr = train_frame["y_cond"].to_numpy().astype(int)
    yte = test_frame["y_cond"].to_numpy().astype(int)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return _score(prob, yte, test_frame["patient_id"].to_numpy(), test_frame["condition"].to_numpy(),
                 threshold=threshold, patient_agg=patient_agg)


# ---------------------------------------------------------------------------
def _aggregate_and_log(tag: str, result: dict, ts_cfg, *, seed: int) -> dict:
    scfg = ts_cfg.stats
    pooled = aggregate_across_patients(
        result["per_patient_balanced_accuracy"],
        n_resamples=int(scfg["bootstrap"]["n_resamples"]), ci=float(scfg["bootstrap"]["ci"]),
        chance=float(scfg["tests"]["chance_level"]), seed=seed,
        wilcoxon=bool(scfg["tests"]["wilcoxon_vs_chance"]), sign_test=bool(scfg["tests"]["sign_test_vs_chance"]),
    )
    order = ordering_stats(result["ordering_records"])
    log.info("  %-14s | pooled window bal-acc %.3f | pooled session bal-acc %.3f | "
             "patient-mean %.3f [%.3f, %.3f] (%d patients) | ordering %d/%d",
             tag, result["window"]["balanced_accuracy"], result["session"]["balanced_accuracy"],
             pooled["mean"], pooled["ci_low"], pooled["ci_high"], pooled["n_patients"],
             order["n_correct_direction_level"] if "n_correct_direction_level" in order else -1,
             order["n_direction_level"] if "n_direction_level" in order else -1)
    return {**result, "pooled_patient_level": pooled, "ordering": order}


def main() -> int:
    ap = argparse.ArgumentParser(description="Pooled (all-patients-together) within-patient task split")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=["frozen_probe", "lora", "full_ft", "chanamp"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--input-norm", default="none", choices=list(INPUT_NORM_SCHEMES),
                    help="per-window amplitude normalisation, identical to "
                         "run_within_patient_tasksplit.py. chan_z z-scores each channel of "
                         "each window (the supervisor's proposed fix for the amplitude "
                         "fingerprint); global_z removes overall amplitude but keeps "
                         "topography. Applies to the chanamp control too.")
    ap.add_argument("--match-level", default="none", choices=("none",) + MATCH_LEVELS,
                    help="force an equal window count per session so duration cannot encode "
                         "ON/OFF (supervisor 2026-08-06). 'patient_task' = K per patient x task "
                         "= min(ON, OFF), the recommended level: it equalises exactly the "
                         "confounded pair, keeps ~90%% of windows and drops no patients. "
                         "'task'/'global' additionally equalise across patients at 33-53%% "
                         "retention. Applies to the TRAIN frame as well as the test frame — "
                         "the training set is where a duration cue would actually be learned.")
    ap.add_argument("--match-where", default="uniform", choices=MATCH_WHERE,
                    help="which windows survive matching. 'uniform' (default) spreads them "
                         "evenly over each session's task span so both conditions are sampled "
                         "at matched relative phase; the contiguous options would leave the "
                         "shorter session holding the task transitions and the longer one only "
                         "its centre, replacing the count cue with a task-phase cue.")
    ap.add_argument("--match-k", type=int, default=None,
                    help="explicit K for every session, overriding --match-level")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="persist the trained weights for lora/full_ft arms. train_model() "
                         "already reloads the best-epoch state before returning, so what is "
                         "written is the early-stopped model that produced the reported "
                         "metrics. Required for any XAI follow-up: nothing else in this "
                         "project writes fine-tuned weights to disk (XAI_PLAN.md 2.1).")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="override seed.base from finetune.yaml. Per-arm seeds stay derived "
                         "(fold_seed(base, crc32(direction|arm))), so this shifts the whole "
                         "run coherently — used to measure seed stability (XAI_PLAN.md 2.2).")
    ap.add_argument("--max-patients", type=int, default=None, help="smoke test: first N patients")
    ap.add_argument("--run-tag", default="pooled_tasksplit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    log_txt = _RUN_ROOT / "logs" / f"{args.run_tag}_{time.strftime('%Y%m%d')}.log"
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
    if args.seed_base is not None:
        ft_cfg = dataclasses.replace(ft_cfg, seed=dataclasses.replace(ft_cfg.seed, base=args.seed_base))
        log.info("Seed base overridden: %d", args.seed_base)
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d). STOP.", n_pat, n_win)
        return 1

    # Keyed on the run tag, so a --input-norm run writes its own directory instead
    # of resuming from (or overwriting) the raw-input run's per-direction JSONs.
    results_dir = _RUN_ROOT / "results" / args.run_tag
    pf_path = _RUN_ROOT / "results" / "within_patient_tasksplit" / "preflight_summary.json"
    if not pf_path.is_file():
        log.error("Pre-flight missing (%s). Run derive_task_windows.py first.", pf_path)
        return 1
    pf = _read_json(pf_path)
    eligible = list(pf["eligible_patients"])
    if args.max_patients:
        eligible = eligible[: args.max_patients]
        log.warning("SMOKE TEST: restricted to %d patient(s): %s", len(eligible), eligible)
    log.info("Pooled design over %d/%d eligible patients: %s", len(eligible), pf["n_cohort_patients"], eligible)

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)

    arms = args.arms or ["frozen_probe", "lora", "full_ft", "chanamp"]
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])
    max_epochs = args.max_epochs or ft_cfg.train.max_epochs
    threshold = float(ts_cfg.stats["tests"]["chance_level"])
    patient_agg = ft_cfg.eval.patient_agg
    lora_hp = _resolve_lora_hp(ts_cfg, ft_cfg)
    cc = load_controls_config()

    # Same norm_scale handling as run_within_patient_tasksplit.py: the rescale
    # target is the cohort-median window std, so a normalised window keeps the
    # backbone's pretrained input magnitude.
    norm_scale = None
    if args.input_norm != "none":
        norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        log.info("Input normalisation: %s (rescaled to cohort-median window std %.6f)",
                 args.input_norm, norm_scale)

    do_match = args.match_level != "none" or args.match_k is not None
    common = dict(run_tag=args.run_tag, git_sha=git_sha(_PROJECT),
                 checkpoint_sha256=file_sha256(data_cfg.paths.checkpoint), env=env_versions(),
                 n_patients=len(eligible), eligible_patients=eligible, lora_hp=lora_hp,
                 input_norm=args.input_norm, norm_scale=norm_scale,
                 match_level=(args.match_level if do_match else "none"),
                 match_where=(args.match_where if do_match else None),
                 match_k=args.match_k)
    if do_match:
        log.info("Duration matching ON: level=%s k=%s where=%s (applied to BOTH train and test)",
                 args.match_level, args.match_k, args.match_where)

    all_summaries: dict[str, dict] = {}
    for train_task, test_task in pairs:
        direction = f"{train_task}_to_{test_task}"
        log.info("=" * 74)
        log.info("DIRECTION %s (pooled over %d patients)", direction, len(eligible))
        train_frame = _select_pooled_frame(sidecar, eligible, train_task,
                                           nonoverlap_only=bool(ts_cfg.protocol["train_nonoverlap_only"]))
        test_frame = _select_pooled_frame(sidecar, eligible, test_task,
                                          nonoverlap_only=bool(ts_cfg.protocol["eval_nonoverlap_only"]))
        overlap = set(train_frame["path"]) & set(test_frame["path"])
        if overlap:
            raise RuntimeError(f"{direction}: {len(overlap)} windows appear in both pooled train and test")

        # Matching is applied to BOTH sides, each with its own K: the two frames use
        # different window conventions (train = overlapping 5 s stride, test =
        # non-overlap 10 s), so a K derived from one does not transfer to the other.
        match_info = None
        if do_match:
            tr_before, te_before = train_frame, test_frame
            train_frame = duration_match(train_frame, level=args.match_level,
                                         k=args.match_k, where=args.match_where)
            test_frame = duration_match(test_frame, level=args.match_level,
                                        k=args.match_k, where=args.match_where)
            match_info = {"train": match_report(tr_before, train_frame),
                          "test": match_report(te_before, test_frame)}
            log.info("  matched train %d->%d (%.1f%%), test %d->%d (%.1f%%)",
                     len(tr_before), len(train_frame),
                     100.0 * len(train_frame) / len(tr_before),
                     len(te_before), len(test_frame),
                     100.0 * len(test_frame) / len(te_before))

        log.info("  train: %d windows (%d patients) | test: %d windows (%d patients)",
                 len(train_frame), train_frame.patient_id.nunique(),
                 len(test_frame), test_frame.patient_id.nunique())

        for arm in arms:
            out_path = results_dir / f"{direction}_{arm}.json"
            if out_path.exists():
                rec = _read_json(out_path)
                log.info("  %-12s | RESUME (exists)", arm)
                all_summaries.setdefault(arm, {})[direction] = rec
                continue

            t0 = time.time()
            if arm == "frozen_probe":
                result, extra = run_frozen_probe(data_cfg, ft_cfg, train_frame, test_frame,
                                                  device=device, threshold=threshold, patient_agg=patient_agg,
                                                  input_norm=args.input_norm, norm_scale=norm_scale)
            elif arm == "chanamp":
                result, extra = run_chanamp_control(train_frame, test_frame, data_cfg, cc,
                                                     threshold=threshold, patient_agg=patient_agg,
                                                     input_norm=args.input_norm,
                                                     norm_scale=norm_scale), {}
            elif arm in {"lora", "full_ft"}:
                seed = fold_seed(ft_cfg.seed.base, zlib.crc32(f"{direction}|{arm}".encode()) % 10_000)
                model, train_res, ft_eff = run_trained_arm(
                    data_cfg, ft_cfg, ts_cfg, arm, train_frame, lora_hp,
                    max_epochs=max_epochs, device=device, threshold=threshold,
                    patient_agg=patient_agg, seed=seed,
                    input_norm=args.input_norm, norm_scale=norm_scale)
                try:
                    te_loader = make_frame_loader(test_frame, data_cfg, shuffle=False, seed=seed,
                                                  input_norm=args.input_norm, norm_scale=norm_scale)
                    prob, yte, pids_te, conds_te = predict_windows(model, te_loader, device=device,
                                                                    precision=ft_eff.train.precision)
                    peak = int(torch.cuda.max_memory_allocated())
                    if args.save_checkpoint:
                        ckpt_path = _save_checkpoint(
                            model, ckpt_dir=_RUN_ROOT / "checkpoints" / "xai",
                            run_tag=args.run_tag, direction=direction, arm=arm,
                            seed=seed, seed_base=ft_cfg.seed.base, common=common,
                            train_res=train_res, ft_eff=ft_eff,
                            n_train=len(train_frame), n_test=len(test_frame))
                        extra_ckpt = {"checkpoint": str(ckpt_path.relative_to(_RUN_ROOT))}
                    else:
                        extra_ckpt = {}
                finally:
                    del model
                    torch.cuda.empty_cache()
                result = _score(prob, yte, pids_te, conds_te, threshold=threshold, patient_agg=patient_agg)
                extra = {"best_epoch": train_res.best_epoch, "trainable_params": train_res.trainable_params,
                        "train_seconds": train_res.train_seconds, "peak_vram_bytes": peak,
                        **extra_ckpt}
            else:
                raise ValueError(f"unknown arm {arm!r}")

            summary = _aggregate_and_log(f"{direction}/{arm}", result, ts_cfg, seed=ft_cfg.seed.base)
            record = {**common, "direction": direction, "arm": arm,
                     "n_train_windows": len(train_frame), "n_test_windows": len(test_frame),
                     "match": match_info,
                     "wall_seconds": time.time() - t0, **extra, **summary}
            _write_json(out_path, record)
            all_summaries.setdefault(arm, {})[direction] = record

            p = summary["pooled_patient_level"]
            save_run({
                **common, "arm": arm, "direction": direction,
                "n_patients_test": p["n_patients"],
                "pooled_window_balanced_accuracy": result["window"]["balanced_accuracy"],
                "pooled_session_balanced_accuracy": result["session"]["balanced_accuracy"],
                "patient_mean_balanced_accuracy": p["mean"], "ci_low": p["ci_low"], "ci_high": p["ci_high"],
                "ordering_frac": summary["ordering"].get("frac_correct"),
                **record,
            }, results_dir, _RUN_ROOT / "results" / "runs_master_pooled_tasksplit.csv",
                csv_fields=_MASTER_CSV_FIELDS)

    _write_json(results_dir / "pooled_summary.json",
               {"run_tag": args.run_tag, "arms": all_summaries, "eligible_patients": eligible,
                "input_norm": args.input_norm, "norm_scale": norm_scale,
                "match_level": common["match_level"], "match_where": common["match_where"],
                "match_k": args.match_k,
                "git_sha": git_sha(_PROJECT), "env": env_versions()})
    log.info("Done. Results under %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
