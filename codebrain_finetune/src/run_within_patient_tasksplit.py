"""WITHIN-PATIENT TASK-SPLIT experiment — train on one ADL task, test on the other.

An ADDITIONAL experiment alongside the nested-LOPO arms (which are untouched and
remain the dissertation's cross-subject headline). Supervisor-suggested,
2026-07-30.

PROTOCOL
--------
For each eligible patient, and each direction (tea->toast, toast->tea):

  1. TRAIN on that patient's windows for the train task, taken from BOTH of the
     patient's sessions, labelled by the session's existing ON/OFF condition
     (A=ON=0, C=OFF=1). Full overlapping (5 s stride) set for augmentation.
     A trailing time block of the TRAIN task is held out per session for early
     stopping (never the test task).
  2. TEST on that patient's windows for the other task, non-overlap (10 s stride)
     subset only.
  3. Per-patient balanced accuracy = mean of the two directions' window-level
     balanced accuracies on the held-out task.

Then pool across patients treating PATIENT as the unit of analysis: mean of the
per-patient values, patient-level bootstrap CI, Wilcoxon signed-rank + exact sign
test against 0.5, plus a threshold-free ordering test. Never a naive window pool
(per-patient N is small and unequal).

Three arms, the SAME configs as the LOPO experiments so the numbers are
comparable: frozen linear probe, LoRA (r/lr/dropout reused from the LOPO pooled
HP search — NOT re-searched; within-patient training sets are far too small to
support a search and re-searching would break comparability), and full fine-tune.

PRE-FLIGHT IS MANDATORY. ``derive_task_windows.py`` must have run: it recovers
the task labels (the production cache has ``task_id == "continuous"`` for every
window) and enforces that both tasks occur in BOTH the ON and OFF session of a
patient. Without that gate, task type silently re-encodes session type and the
split reintroduces the temporal/duration confound (EXPERIMENTS.md §10c-10d).
This script refuses to run on a patient the pre-flight did not clear.

Usage:
    # smoke test first (2 patients, 2 epochs, probe only):
    python codebrain_finetune/src/run_within_patient_tasksplit.py \
        --arms frozen_probe --max-patients 2 --max-epochs 2
    # full run:
    python codebrain_finetune/src/run_within_patient_tasksplit.py
    # duration-matched secondary variant:
    python codebrain_finetune/src/run_within_patient_tasksplit.py --match-k 20 --run-tag matched_k20
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
    load_finetune_config, build_classifier_model,
    train_model, evaluate_session_level, run_probe_fold,
    seed_everything, fold_seed, INPUT_NORM_SCHEMES, cohort_median_scale,
)
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import (  # noqa: E402
    select_task_frame, train_val_split_by_time, make_frame_loader, eval_duration_match,
    direction_pairs, aggregate_across_patients, ordering_stats,
)
from codebrain_ft.metrics import classification_metrics  # noqa: E402
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402

log = logging.getLogger("tasksplit")

EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174

# LOPO reference numbers (cross-subject, pooled over 70 sessions) for the
# side-by-side table. Source: EXPERIMENTS.md / results/runs_master.csv.
LOPO_REFERENCE = {
    "frozen_probe": {"balanced_accuracy": 0.443, "ci": [0.357, 0.529], "auc": 0.482, "perm_p": 0.95},
    "lora":         {"balanced_accuracy": 0.557, "ci": [0.457, 0.643], "auc": 0.541, "perm_p": 0.194},
    "full_ft":      {"balanced_accuracy": 0.443, "ci": [0.357, 0.529], "auc": 0.423, "perm_p": 0.945},
}

# This experiment reports patient-level statistics, so its summary schema differs
# from the LOPO arms' session-level one. It therefore gets its OWN master CSV
# (results/runs_master_tasksplit.csv) — appending a different field set to
# runs_master.csv would write values under the wrong column names.
_MASTER_CSV_FIELDS = [
    "run_tag", "arm", "mode", "git_sha", "n_patients", "n_directions",
    "patient_mean_balanced_accuracy", "ci_low", "ci_high",
    "wilcoxon_p", "sign_test_p", "ordering_frac", "ordering_p",
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
    """Override the early-stopping monitor (window-level here — see train.py)."""
    es = dataclasses.replace(ft_cfg.train.early_stopping, monitor=monitor)
    train = dataclasses.replace(ft_cfg.train, early_stopping=es)
    return dataclasses.replace(ft_cfg, train=train)


def _resolve_lora_hp(ts_cfg, ft_cfg) -> dict:
    """Reuse the LOPO pooled HP search; fall back to the config literals, loudly."""
    spec = ts_cfg.arms.get("lora", {})
    rel = spec.get("reuse_hp_from")
    if rel:
        p = _RUN_ROOT / rel
        if p.is_file():
            cfg = _read_json(p)["best_config"]
            log.info("LoRA HPs reused from %s: %s (NOT re-searched — see module docstring).", rel, cfg)
            return cfg
        log.warning("LoRA HP source %s not found; falling back to tasksplit.yaml literals.", p)
    fb = spec.get("fallback") or {"r": ft_cfg.lora.r, "lr": ft_cfg.train.lr, "dropout": ft_cfg.lora.dropout}
    log.warning("LoRA HPs = tasksplit.yaml fallback %s", fb)
    return dict(fb)


# ---------------------------------------------------------------------------
# one (patient, direction, arm) run
# ---------------------------------------------------------------------------
def _ordering(session_scores: dict) -> bool | None:
    """Did the OFF session's mean P(OFF) exceed the ON session's?"""
    probs = {int(y): p for y, p in zip(session_scores["y_true"], session_scores["prob"])}
    if set(probs) != {0, 1}:
        return None
    return bool(probs[1] > probs[0])


def run_one(
    *, arm: str, patient: str, train_task: str, test_task: str,
    sidecar, data_cfg, ft_cfg, ts_cfg, lora_hp, out_path: Path, device: str,
    max_epochs: int, match_k: int | None, match_where: str, common: dict,
    input_norm: str = "none", norm_scale: float | None = None,
) -> dict:
    """Train on ``train_task``, evaluate ``test_task``, both within one patient."""
    if out_path.exists():
        rec = _read_json(out_path)
        log.info("%-12s %-9s %s->%-6s | RESUME (exists) | window bal-acc %.3f",
                 arm, patient, train_task, test_task, rec["window"]["balanced_accuracy"])
        return rec

    proto = ts_cfg.protocol
    # Stable across processes: Python's hash() is randomised per interpreter
    # (PYTHONHASHSEED), so using it here would silently change the seed — and the
    # result — between runs of the same configuration. crc32 is deterministic.
    run_key = f"{patient}|{train_task}|{test_task}".encode()
    seed = fold_seed(ft_cfg.seed.base, zlib.crc32(run_key) % 10_000)

    train_all = select_task_frame(
        sidecar, patient, train_task,
        nonoverlap_only=bool(proto["train_nonoverlap_only"]))
    test_frame = select_task_frame(
        sidecar, patient, test_task,
        nonoverlap_only=bool(proto["eval_nonoverlap_only"]))
    if match_k is not None:
        test_frame = eval_duration_match(test_frame, k=match_k, where=match_where)

    # Sanity: the two sides must share no window (the whole point of the design).
    overlap = set(train_all["path"]) & set(test_frame["path"])
    if overlap:
        raise RuntimeError(
            f"{patient} {train_task}->{test_task}: {len(overlap)} window(s) appear in BOTH the "
            f"train and test task — task spans are not disjoint; aborting rather than reporting "
            f"a leaked number. Example: {sorted(overlap)[:3]}")

    n_tr_on = int((train_all.condition == "A").sum())
    n_tr_off = int((train_all.condition == "C").sum())
    n_te_on = int((test_frame.condition == "A").sum())
    n_te_off = int((test_frame.condition == "C").sum())

    t0 = time.time()
    extra: dict = {}

    if arm == "frozen_probe":
        # No gradients, so no early stopping and no validation carve-out: the
        # probe uses every train-task window.
        import torch
        seed_everything(seed, deterministic=ft_cfg.seed.deterministic)
        model = build_classifier_model(data_cfg, ft_cfg, mode="frozen", device=device)
        try:
            tr_loader = make_frame_loader(train_all, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            te_loader = make_frame_loader(test_frame, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            res = run_probe_fold(model, tr_loader, te_loader, data_cfg, ft_cfg, device=device)
            peak = int(torch.cuda.max_memory_allocated())
        finally:
            del model
            torch.cuda.empty_cache()
        window, session, sscores = res["window"], res["session"], res["session_scores"]
        n_train_used, n_val_used = len(train_all), 0

    elif arm in {"lora", "full_ft"}:
        import torch
        mode = arm
        ft_eff = _with_hp(ft_cfg, r=lora_hp["r"], dropout=lora_hp["dropout"])
        ft_eff = _with_monitor(ft_eff, proto["early_stopping_monitor"])
        lr = lora_hp["lr"] if arm == "lora" else ft_cfg.train.lr_full_ft

        tr_frame, va_frame = train_val_split_by_time(
            train_all, val_frac=float(proto["val_frac"]),
            stratify_by_session=bool(proto["val_stratify_by_session"]))
        n_train_used, n_val_used = len(tr_frame), len(va_frame)

        seed_everything(seed, deterministic=ft_eff.seed.deterministic)
        torch.cuda.reset_peak_memory_stats()
        model = build_classifier_model(data_cfg, ft_eff, mode=mode, device=device)
        try:
            tr_loader = make_frame_loader(tr_frame, data_cfg, shuffle=True, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            va_loader = make_frame_loader(va_frame, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            te_loader = make_frame_loader(test_frame, data_cfg, shuffle=False, seed=seed,
                                          input_norm=input_norm, norm_scale=norm_scale)
            train_res = train_model(model, tr_loader, va_loader, ft_eff,
                                    lr=lr, device=device, max_epochs=max_epochs)
            res = evaluate_session_level(model, te_loader, ft_eff, device=device)
            peak = int(torch.cuda.max_memory_allocated())
        finally:
            del model
            torch.cuda.empty_cache()
        window, session, sscores = res["window"], res["session"], res["session_scores"]
        extra = {"best_epoch": train_res.best_epoch,
                 "best_val_monitor": train_res.best_monitor,
                 "monitor": proto["early_stopping_monitor"],
                 "history": train_res.history,
                 "trainable_params": train_res.trainable_params,
                 "lr": lr, "train_seconds": train_res.train_seconds}
    else:
        raise ValueError(f"unknown arm {arm!r}")

    wall = time.time() - t0
    ordering_correct = _ordering(sscores)
    record = {
        **common, "arm": arm, "patient": patient,
        "train_task": train_task, "test_task": test_task,
        "direction": f"{train_task}_to_{test_task}", "seed": seed,
        "n_train_windows": n_train_used, "n_val_windows": n_val_used,
        "n_test_windows": len(test_frame),
        "train_class_counts": {"ON": n_tr_on, "OFF": n_tr_off},
        "test_class_counts": {"ON": n_te_on, "OFF": n_te_off},
        "match_k": match_k, "match_where": match_where if match_k else None,
        "input_norm": input_norm, "norm_scale": norm_scale,
        "window": window, "session": session, "session_scores": sscores,
        "ordering_correct": ordering_correct,
        "wall_seconds": wall, "peak_vram_bytes": peak, **extra,
    }
    _write_json(out_path, record)
    log.info("%-12s %-9s %s->%-6s | window bal-acc %.3f auc %s | test n=%d (ON %d/OFF %d) | "
             "ordering %s | %.0fs",
             arm, patient, train_task, test_task, window["balanced_accuracy"],
             f"{window['auc']:.3f}" if window["auc"] is not None else "n/a",
             len(test_frame), n_te_on, n_te_off,
             "OK" if ordering_correct else ("wrong" if ordering_correct is not None else "n/a"), wall)
    return record


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate_arm(records: list[dict], ts_cfg, *, arm: str, seed: int) -> dict:
    """Per-patient direction-averaged balanced accuracy -> patient-level statistics."""
    scfg = ts_cfg.stats
    by_patient: dict[str, list[float]] = {}
    for r in records:
        by_patient.setdefault(r["patient"], []).append(float(r["window"]["balanced_accuracy"]))
    per_patient = {p: float(np.mean(v)) for p, v in by_patient.items()}
    n_dirs = {p: len(v) for p, v in by_patient.items()}
    incomplete = {p: n for p, n in n_dirs.items() if n != len(ts_cfg.protocol["directions"])}
    if incomplete:
        log.warning("Patients without all %d directions (averaged over what exists): %s",
                    len(ts_cfg.protocol["directions"]), incomplete)

    pooled = aggregate_across_patients(
        per_patient,
        n_resamples=int(scfg["bootstrap"]["n_resamples"]),
        ci=float(scfg["bootstrap"]["ci"]),
        chance=float(scfg["tests"]["chance_level"]),
        seed=seed,
        wilcoxon=bool(scfg["tests"]["wilcoxon_vs_chance"]),
        sign_test=bool(scfg["tests"]["sign_test_vs_chance"]),
    )
    order = ordering_stats(records) if scfg.get("ordering_test", True) else {}

    # Per-direction means, to show the two directions separately as well as averaged.
    per_direction = {}
    for d in ts_cfg.protocol["directions"]:
        vals = [float(r["window"]["balanced_accuracy"]) for r in records if r["direction"] == d]
        if vals:
            per_direction[d] = {"n": len(vals), "mean_balanced_accuracy": float(np.mean(vals))}

    # Pooled session-level metrics, reported ONLY as a secondary reference: this
    # pools every patient x direction session and so re-introduces the unequal
    # weighting the patient-level statistic above deliberately avoids.
    pooled_windows = None
    if records:
        probs, trues = [], []
        for r in records:
            probs.extend(r["session_scores"]["prob"])
            trues.extend(r["session_scores"]["y_true"])
        m = classification_metrics(np.asarray(trues), np.asarray(probs),
                                   threshold=ts_cfg.stats["tests"]["chance_level"])
        pooled_windows = {"note": "session-level pooled over all patient x direction "
                                  "sessions; SECONDARY reference only",
                          **m.as_dict()}
    return {
        "arm": arm, "n_records": len(records), "per_direction": per_direction,
        "per_patient_direction_averaged": per_patient, "n_directions_per_patient": n_dirs,
        "pooled_patient_level": pooled, "ordering": order,
        "pooled_session_level_secondary": pooled_windows,
        "lopo_reference": LOPO_REFERENCE.get(arm),
    }


def _write_report(path: Path, summaries: dict, ts_cfg, run_tag: str, args) -> None:
    L = []
    L.append(f"# Within-patient task-split — results ({run_tag})\n\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    L.append("Train on one ADL task, test on the other, **inside each patient**, using that\n"
             "patient's existing session ON/OFF labels. Both directions, averaged per patient.\n"
             "Patient is the unit of analysis (paired / random-effect style), never a naive\n"
             "window pool. This is an ADDITIONAL experiment; the nested-LOPO arms are unchanged.\n\n")
    if args.match_k:
        L.append(f"**Duration-matched variant:** exactly {args.match_k} eval windows per session "
                 f"({args.match_where} block).\n\n")

    L.append("## Headline — per-patient balanced accuracy on the held-out task\n\n")
    L.append("| arm | N patients | mean bal-acc | 95% CI (patient bootstrap) | Wilcoxon p | "
             "sign-test p | above chance | LOPO bal-acc (cross-subject) |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for arm, s in summaries.items():
        p = s["pooled_patient_level"]
        w = (s["pooled_patient_level"].get("wilcoxon") or {}).get("p_value")
        st = (s["pooled_patient_level"].get("sign_test") or {}).get("p_value")
        ref = s.get("lopo_reference") or {}
        L.append(f"| {arm} | {p['n_patients']} | **{p['mean']:.3f}** | "
                 f"[{p['ci_low']:.3f}, {p['ci_high']:.3f}] | "
                 f"{f'{w:.3f}' if w is not None else 'n/a'} | "
                 f"{f'{st:.3f}' if st is not None else 'n/a'} | "
                 f"{p['n_above_chance']}/{p['n_patients']} | "
                 f"{ref.get('balanced_accuracy', float('nan')):.3f} |\n")

    L.append("\n## Threshold-free ordering test (OFF session scored more OFF-like than ON)\n\n")
    L.append("| arm | correct / total (patient x direction) | frac | binomial p | "
             "patients both directions | patients neither |\n|---|---|---|---|---|---|\n")
    for arm, s in summaries.items():
        o = s.get("ordering") or {}
        if not o:
            continue
        L.append(f"| {arm} | {o['n_correct_direction_level']}/{o['n_direction_level']} | "
                 f"{o['frac_correct']:.3f} | {o['binomial_p']:.3f} | "
                 f"{o['n_patients_both_directions_correct']}/{o['n_patients']} | "
                 f"{o['n_patients_neither_direction_correct']}/{o['n_patients']} |\n")

    L.append("\n## Per-direction means\n\n| arm | direction | n | mean bal-acc |\n|---|---|---|---|\n")
    for arm, s in summaries.items():
        for d, v in s["per_direction"].items():
            L.append(f"| {arm} | {d} | {v['n']} | {v['mean_balanced_accuracy']:.3f} |\n")

    L.append("\n## Per-patient values (direction-averaged)\n\n")
    arms = list(summaries)
    pats = sorted({p for s in summaries.values() for p in s["per_patient_direction_averaged"]},
                  key=lambda x: int(x.split("-")[1]))
    L.append("| patient | " + " | ".join(arms) + " |\n|" + "---|" * (len(arms) + 1) + "\n")
    for pt in pats:
        cells = []
        for a in arms:
            v = summaries[a]["per_patient_direction_averaged"].get(pt)
            cells.append(f"{v:.3f}" if v is not None else "—")
        L.append(f"| {pt} | " + " | ".join(cells) + " |\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L), encoding="utf-8")
    log.info("Wrote markdown report -> %s", path)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Within-patient task-split experiment")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=["frozen_probe", "lora", "full_ft"],
                    help="arms to run (default: all enabled in tasksplit.yaml)")
    ap.add_argument("--max-patients", type=int, default=None, help="smoke test: first N patients")
    ap.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs")
    ap.add_argument("--match-k", type=int, default=None,
                    help="duration-matched eval: exactly K non-overlap windows per session")
    ap.add_argument("--match-where", default=None, choices=["first", "middle", "last"])
    ap.add_argument("--input-norm", default="none", choices=list(INPUT_NORM_SCHEMES),
                    help="per-window amplitude normalisation. global_z removes the "
                         "per-recording amplitude fingerprint that this design is "
                         "vulnerable to (see run_tasksplit_controls.py) while preserving "
                         "topography — run it to quantify how much of a result is fingerprint.")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    ts_cfg = load_tasksplit_config()
    run_tag = args.run_tag or "within_patient_tasksplit"
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
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows", n_pat, n_win)
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d), expected (%d, %d). STOP.",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    # ---- pre-flight is mandatory ----------------------------------------
    results_dir = _RUN_ROOT / "results" / ts_cfg.output["results_subdir"]
    pf_path = results_dir / "preflight_summary.json"
    if not pf_path.is_file():
        log.error("PRE-FLIGHT MISSING: %s not found. Run "
                  "`python codebrain_finetune/src/derive_task_windows.py` first — it recovers the "
                  "task labels and enforces the task-vs-ON/OFF confound gate. Refusing to run.", pf_path)
        return 1
    pf = _read_json(pf_path)
    eligible = list(pf["eligible_patients"])
    if not eligible:
        log.error("PRE-FLIGHT says 0 eligible patients — task type is confounded with ON/OFF for "
                  "this cohort. The design is invalid here. STOP.")
        return 2
    log.info("Pre-flight: %d/%d patients eligible. Ineligible (with reasons) are recorded in %s.",
             len(eligible), pf["n_cohort_patients"], pf_path.name)
    for p, reasons in (pf.get("ineligible") or {}).items():
        log.info("  SKIP %-9s | %s", p, " ; ".join(reasons))

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    # Never touch a patient the gate did not clear.
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)

    if args.max_patients:
        eligible = eligible[: args.max_patients]
        sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)
        log.warning("SMOKE TEST: restricted to %d patient(s): %s", len(eligible), eligible)

    arms = args.arms or [a for a, spec in ts_cfg.arms.items() if spec.get("enabled", True)]
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])
    max_epochs = args.max_epochs or ft_cfg.train.max_epochs
    match_k = args.match_k if args.match_k is not None else ts_cfg.protocol["eval_match_k"]
    match_where = args.match_where or ts_cfg.protocol["eval_match_where"]
    lora_hp = _resolve_lora_hp(ts_cfg, ft_cfg)

    norm_scale = None
    if args.input_norm != "none":
        norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        log.info("Input normalisation: %s (rescaled to cohort-median window std %.4f)",
                 args.input_norm, norm_scale)

    log.info("Run '%s' | arms=%s | %d patients x %d directions | max_epochs=%d | match_k=%s",
             run_tag, arms, len(eligible), len(pairs), max_epochs, match_k)

    common = dict(
        run_tag=run_tag, git_sha=git_sha(_PROJECT),
        checkpoint_sha256=file_sha256(data_cfg.paths.checkpoint),
        env=env_versions(), cohort={"patients": n_pat, "windows": n_win},
        experiment="within_patient_tasksplit",
        lora_target=ft_cfg.lora.target, lora_hp=lora_hp,
        input_norm=args.input_norm,
        preflight={"n_eligible": len(pf["eligible_patients"]),
                   "n_cohort": pf["n_cohort_patients"]},
    )

    summaries: dict[str, dict] = {}
    t_start = time.time()
    for arm in arms:
        arm_dir = results_dir / (f"{run_tag}_{arm}" if run_tag != "within_patient_tasksplit" else arm)
        records = []
        for patient in eligible:
            for train_task, test_task in pairs:
                out_path = arm_dir / f"{patient}_{train_task}_to_{test_task}.json"
                rec = run_one(
                    arm=arm, patient=patient, train_task=train_task, test_task=test_task,
                    sidecar=sidecar, data_cfg=data_cfg, ft_cfg=ft_cfg, ts_cfg=ts_cfg,
                    lora_hp=lora_hp, out_path=out_path, device=device, max_epochs=max_epochs,
                    match_k=match_k, match_where=match_where, common=common,
                    input_norm=args.input_norm, norm_scale=norm_scale,
                )
                records.append(rec)
        summary = aggregate_arm(records, ts_cfg, arm=arm, seed=ft_cfg.seed.base)
        summaries[arm] = summary
        _write_json(arm_dir / "summary.json", summary)

        p = summary["pooled_patient_level"]
        o = summary["ordering"]
        log.info("=" * 74)
        log.info("ARM %s — %d patients, %d runs", arm, p["n_patients"], summary["n_records"])
        log.info("  per-patient bal-acc mean = %.3f  (patient bootstrap 95%% CI [%.3f, %.3f])",
                 p["mean"], p["ci_low"], p["ci_high"])
        log.info("  above chance: %d/%d | Wilcoxon p=%s | sign p=%s",
                 p["n_above_chance"], p["n_patients"],
                 f"{(p.get('wilcoxon') or {}).get('p_value')}", f"{(p.get('sign_test') or {}).get('p_value')}")
        if o:
            log.info("  ordering: %d/%d correct (%.3f), binomial p=%.4f",
                     o["n_correct_direction_level"], o["n_direction_level"],
                     o["frac_correct"], o["binomial_p"])
        ref = summary.get("lopo_reference")
        if ref:
            log.info("  cross-subject LOPO reference for this arm: bal-acc %.3f (CI [%.3f, %.3f])",
                     ref["balanced_accuracy"], ref["ci"][0], ref["ci"][1])
        log.info("=" * 74)

        save_run({
            **common, "arm": arm, "mode": arm,
            "n_patients": p["n_patients"], "n_directions": len(pairs),
            "patient_mean_balanced_accuracy": p["mean"],
            "ci_low": p["ci_low"], "ci_high": p["ci_high"],
            "wilcoxon_p": (p.get("wilcoxon") or {}).get("p_value"),
            "sign_test_p": (p.get("sign_test") or {}).get("p_value"),
            "ordering_frac": o.get("frac_correct") if o else None,
            "ordering_p": o.get("binomial_p") if o else None,
            **summary,
        }, arm_dir, _RUN_ROOT / "results" / "runs_master_tasksplit.csv",
            csv_fields=_MASTER_CSV_FIELDS)

    _write_json(results_dir / f"{run_tag}_summary.json",
                {"run_tag": run_tag, "arms": summaries, "eligible_patients": eligible,
                 "match_k": match_k, "git_sha": git_sha(_PROJECT), "env": env_versions()})
    _write_report(_RUN_ROOT / "logs" / f"{run_tag}_report.md", summaries, ts_cfg, run_tag, args)
    log.info("Done in %.1f min. Results under %s", (time.time() - t_start) / 60, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
