"""NEGATIVE CONTROLS for the within-patient task-split experiment.

WHY THIS EXISTS — read before interpreting any task-split number
----------------------------------------------------------------
The within-patient task-split design has an intrinsic confound that the
task-vs-ON/OFF pre-flight does NOT cover:

    Within one patient, the ON/OFF label is perfectly collinear with
    RECORDING IDENTITY. The patient has exactly two recordings; one is ON and
    one is OFF. Train and test windows both come from those same two
    recordings — only the task differs. So a classifier does not need to learn
    anything about levodopa: it only needs to learn "which of these two
    recordings is this window from", which any stable per-recording
    characteristic supplies (electrode impedance, cap placement, contact
    quality, drift, ambient mains/EM noise, time of day, patient posture).

Cross-subject LOPO has no such loophole: the held-out patient's recordings are
never seen in training, so a recording fingerprint is useless there. That is
exactly why the LOPO frozen probe sits at chance (0.443) while the same frozen
probe reaches ~0.87 on this split. The gap is not a newly found levodopa signal.

These controls measure the size of that loophole using features that CANNOT
carry a neural levodopa signature.

AMPLITUDE controls (the original set, Experiment 12c):
  * ``crude3``  — three numbers per window: mean|x|, std(x), max|x|. Pure
    amplitude/scale descriptors, no spectral or spatial content.
  * ``amp1``    — a single number per window: mean|x|.
  * ``chanamp`` — per-channel mean|x| (18 numbers): adds the spatial amplitude
    pattern, i.e. the electrode-contact fingerprint, still no spectral content.

SHAPE controls (added 2026-08-01, because of ``--input-norm``):
  * ``relbp``     — per-channel RELATIVE band power (90 numbers): each channel's
    five band powers normalised to sum to 1 within that channel.
  * ``aperiodic`` — per-channel 1/f exponent + offset (36 numbers). This dataset
    has NO impedance data anywhere (qc_signal_quality/qc_summary_report.md §C),
    so the aperiodic exponent is the only available contact-quality proxy.
  * ``recid``     — not a feature set but a different TARGET: "which of this
    patient's two recordings is this window from?", on the union of the blocks
    named in ``controls.yaml``. It measures the leak directly instead of
    inferring it from a proxy.

WHY THE SHAPE CONTROLS EXIST. ``--input-norm chan_z`` (the supervisor's
proposed fix) removes each channel's absolute amplitude — i.e. exactly what the
amplitude controls were built to measure. After ``chan_z`` those controls no
longer report on the electrode-contact fingerprint at all, so they cannot be the
evidence for or against the leak under normalisation. ``relbp`` is scale-free by
construction and ``aperiodic``'s exponent is scale-invariant, so both survive
``chan_z`` and can still answer the question.

READING ``recid``. Within a patient, "which recording" is BY CONSTRUCTION the
same variable as ON/OFF — one recording is the ON one, the other is the OFF one.
That is not a flaw; it is the point. It states in one number that on this design
"decode levodopa state" and "decode which recording" are the same problem, so
every task-split ON/OFF number is at most an alias for recording identity. If
``recid`` stays well above chance under ``chan_z``, normalisation has NOT closed
the design.

Each control runs the IDENTICAL protocol as the real experiment (train on one
task, test on the other's non-overlap windows, both directions, per-patient
balanced accuracy, patient-level statistics), so the numbers are directly
comparable to the CodeBrain arms.

THE TRANSFER TEST IS THE ACTUAL EVIDENCE. For every control and every
normalisation we also re-run the within-patient vs cross-patient LOPO contrast:
same features, same windows, only the split changes. A genuine drug effect must
transfer across patients at least partially; a per-recording fingerprint cannot
transfer at all. The DROP is what distinguishes them — not the absolute value.

Cheap: CPU only, no GPU, no model. Feature blocks are computed once per
normalisation and cached under ``results/within_patient_tasksplit/
control_features/``, so re-runs are seconds.

Usage:
    # smoke test (3 patients, fast aperiodic estimator):
    python codebrain_finetune/src/run_tasksplit_controls.py --max-patients 3 \
        --aperiodic-estimator loglog_robust --run-tag smoke
    # full matrix (both normalisations, all six controls):
    python codebrain_finetune/src/run_tasksplit_controls.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import (  # noqa: E402
    select_task_frame, direction_pairs, aggregate_across_patients, ordering_stats,
    duration_match, MATCH_LEVELS, MATCH_WHERE,
)
from codebrain_ft.control_features import (  # noqa: E402
    load_controls_config, load_or_compute_features, control_matrix, control_target,
    CONTROL_DESCRIPTIONS,
)
from codebrain_ft.data import INPUT_NORM_SCHEMES, cohort_median_scale  # noqa: E402
from codebrain_ft.ft_config import load_finetune_config  # noqa: E402
from codebrain_ft.metrics import classification_metrics, aggregate_session_level  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions, save_run  # noqa: E402

log = logging.getLogger("tasksplit_controls")

CONTROLS = dict(CONTROL_DESCRIPTIONS)

# Published Experiment 12c numbers (raw input, no normalisation), for the
# side-by-side table. Source: EXPERIMENTS.md §12b-12c.
RAW_INPUT_REFERENCE = {
    "amp1": 0.598, "crude3": 0.624, "chanamp": 0.934,
    "frozen_probe_arm": 0.883, "lora_arm": 0.704, "full_ft_arm": 0.805,
    "frozen_probe_lopo_cross_subject": 0.443,
}

# Same schema as the existing within-patient rows, so the control rows land in
# results/runs_master_tasksplit.csv in exactly the published format.
_MASTER_CSV_FIELDS = [
    "run_tag", "arm", "mode", "git_sha", "n_patients", "n_directions",
    "patient_mean_balanced_accuracy", "ci_low", "ci_high",
    "wilcoxon_p", "sign_test_p", "ordering_frac", "ordering_p",
]


DEFAULT_RUN_TAG = "tasksplit_controls_norm"


def _tagged_outputs(results_dir: Path, cc, run_tag: str) -> tuple[Path, Path]:
    """(summary, report) paths for this run tag.

    controls.yaml names both with fixed literals, so two runs with different
    ``--run-tag`` would land on the same two files and the second would overwrite
    the first (it did, once: a --match-level run clobbered the canonical
    controls_norm_summary.json on 2026-08-06; the per-run JSONs under
    controls_norm/ made it recoverable). The canonical tag keeps the published
    filenames; every other tag gets its own suffixed pair.
    """
    summary = results_dir / cc.output["summary"]
    report = _RUN_ROOT / cc.output["report"]
    if run_tag != DEFAULT_RUN_TAG:
        summary = summary.with_name(f"{summary.stem}__{run_tag}{summary.suffix}")
        report = report.with_name(f"{report.stem}__{run_tag}{report.suffix}")
    return summary, report


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


def _fit_eval(X: np.ndarray, y: np.ndarray, tr, te, cc) -> np.ndarray:
    """Standardised, class-balanced logistic regression — one place, every control."""
    ccl = cc.classifier
    Xtr, Xte = X[tr], X[te]
    if ccl.get("standardize", True):
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=int(ccl["max_iter"]),
                             class_weight=ccl.get("class_weight", "balanced"))
    clf.fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------------------
# one (patient, direction, control) run, under one normalisation
# ---------------------------------------------------------------------------
def _run_direction(sidecar, blocks, patient, train_task, test_task, control, ts_cfg, cc,
                   *, match_level: str = "none", match_where: str = "uniform",
                   match_k: int | None = None) -> dict:
    tr_frame = select_task_frame(sidecar, patient, train_task,
                                 nonoverlap_only=bool(ts_cfg.protocol["train_nonoverlap_only"]))
    te_frame = select_task_frame(sidecar, patient, test_task,
                                 nonoverlap_only=bool(ts_cfg.protocol["eval_nonoverlap_only"]))
    overlap = set(tr_frame["path"]) & set(te_frame["path"])
    if overlap:
        raise RuntimeError(f"{patient} {train_task}->{test_task}: {len(overlap)} shared windows")

    # Duration matching, identical rule to the pooled driver, so a matched control
    # number is comparable to a matched model-arm number. Applied to both sides;
    # `_row` (the index into the precomputed feature matrix) survives subsetting.
    if match_level != "none" or match_k is not None:
        tr_frame = duration_match(tr_frame, level=match_level, k=match_k, where=match_where)
        te_frame = duration_match(te_frame, level=match_level, k=match_k, where=match_where)

    X = control_matrix(blocks, control, cc)
    y = control_target(sidecar, control, cc)
    tr, te = tr_frame["_row"].to_numpy(), te_frame["_row"].to_numpy()
    prob = _fit_eval(X, y, tr, te, cc)

    thr = float(ts_cfg.stats["tests"]["chance_level"])
    yte = y[te]
    window = classification_metrics(yte, prob, threshold=thr)
    # Session aggregation always uses the ON/OFF condition, including for recid:
    # the ordering test asks whether the OFF recording scored more OFF-like, and
    # for recid the two labels coincide anyway (asserted in main()).
    sess = aggregate_session_level(
        te_frame["patient_id"].to_numpy(), te_frame["condition"].to_numpy(), prob, yte,
        agg="mean_prob")
    probs = {int(v): p for v, p in zip(sess.y_true, sess.prob)}
    ordering = bool(probs[1] > probs[0]) if set(probs) == {0, 1} else None

    return {
        "patient": patient, "control": control, "train_task": train_task,
        "test_task": test_task, "direction": f"{train_task}_to_{test_task}",
        "n_train_windows": len(tr_frame), "n_test_windows": len(te_frame),
        "n_features": int(X.shape[1]),
        "window": window.as_dict(),
        "session_scores": {"sessions": sess.sessions.tolist(), "prob": sess.prob.tolist(),
                           "y_true": sess.y_true.tolist(), "n_windows": sess.n_windows.tolist()},
        "ordering_correct": ordering,
    }


# ---------------------------------------------------------------------------
# transfer test — fingerprint or neural effect?
# ---------------------------------------------------------------------------
def _transfer_test(sidecar, blocks, eligible, ts_cfg, cc, control: str) -> dict:
    """Is this control's signal a NEURAL levodopa effect, or a recording fingerprint?

    The decisive contrast: identical features, identical windows, identical
    statistics — changing exactly ONE thing, whether train and test share the
    patient.

      * within-patient — train on the patient's train-task windows, test on their
        other-task windows (both from the SAME two recordings);
      * cross-patient  — train on every OTHER patient, test on the held-out one
        (the LOPO protocol; the test recordings were never seen).

    A genuine drug effect on this feature is a property of the drug and must
    transfer across patients at least partially. A per-recording fingerprint
    cannot transfer at all, because a held-out patient's recordings have their
    own fingerprints. This is what distinguishes the two explanations, and no
    within-patient number can do it.

    ``recid`` is skipped: its target is defined only WITHIN a patient (which of
    *this* patient's two recordings), so a cross-patient split is not defined
    for it. Recorded as a null rather than silently omitted.
    """
    if control == "recid":
        return {"control": control, "skipped": True,
                "reason": "recid's target (which of THIS patient's two recordings) is not "
                          "defined across patients, so the LOPO arm of the transfer test "
                          "does not exist for it."}
    X = control_matrix(blocks, control, cc)
    y = control_target(sidecar, control, cc)
    pats = sidecar["patient_id"].to_numpy()
    tasks = sidecar["task_group"].to_numpy()
    conds = sidecar["condition"].to_numpy()
    nonov = sidecar["is_nonoverlap_subset"].to_numpy().astype(bool)
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])

    within, cross = {}, {}
    s_prob, s_true = [], []
    for p in eligible:
        vals = []
        for tr_task, te_task in pairs:
            tr = (pats == p) & (tasks == tr_task)
            te = (pats == p) & (tasks == te_task) & nonov
            pr = _fit_eval(X, y, tr, te, cc)
            vals.append(balanced_accuracy_score(y[te], (pr >= 0.5).astype(int)))
        within[p] = float(np.mean(vals))

        tr, te = (pats != p), (pats == p) & nonov
        pr = _fit_eval(X, y, tr, te, cc)
        cross[p] = float(balanced_accuracy_score(y[te], (pr >= 0.5).astype(int)))
        for c in ("A", "C"):                      # session-level, LOPO convention
            m = conds[te] == c
            if m.any():
                s_prob.append(float(pr[m].mean()))
                s_true.append(0 if c == "A" else 1)

    w = np.array(list(within.values()))
    x = np.array(list(cross.values()))
    sp, st = np.asarray(s_prob), np.asarray(s_true)
    thr = float(cc.transfer_test["fingerprint_threshold"])
    result = {
        "control": control, "n_features": int(X.shape[1]),
        "within_patient_mean": float(w.mean()),
        "within_patient_above_chance": int((w > 0.5).sum()),
        "cross_patient_lopo_mean": float(x.mean()),
        "cross_patient_above_chance": int((x > 0.5).sum()),
        "cross_patient_session_balanced_accuracy": float(
            balanced_accuracy_score(st, (sp >= 0.5).astype(int))),
        "cross_patient_session_auc": float(roc_auc_score(st, sp)),
        "n_patients": len(w),
        "drop": float(w.mean() - x.mean()),
        "per_patient_within": within, "per_patient_cross": cross,
        "verdict": ("per-recording FINGERPRINT: the feature works only when the test "
                    "recording was already seen in training, so it is not a neural "
                    "levodopa effect"
                    if x.mean() < thr else
                    "transfers across patients — may carry a genuine effect; re-examine"),
    }
    log.info("  transfer %-9s | within %.3f (%d/%d) -> cross-LOPO %.3f (%d/%d) | drop %.3f | %s",
             control, result["within_patient_mean"], result["within_patient_above_chance"],
             len(w), result["cross_patient_lopo_mean"], result["cross_patient_above_chance"],
             len(x), result["drop"], result["verdict"].split(":")[0])
    return result


# ---------------------------------------------------------------------------
def _write_report(path: Path, summaries: dict, cc, run_tag: str) -> None:
    L = [f"# Task-split controls under input normalisation ({run_tag})\n\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
         "Identical protocol to the within-patient task-split arms: train on one ADL task,\n"
         "test on the other's non-overlap windows, both directions, per-patient balanced\n"
         "accuracy, patient as the unit of analysis. Only the FEATURES (and, for `recid`, the\n"
         "TARGET) change. Raw-input reference numbers are Experiment 12b-12c.\n\n"]

    L.append("## Control matrix — mean per-patient balanced accuracy\n\n")
    schemes = sorted({s for s in summaries}, key=lambda s: s != "none")
    controls = list(dict.fromkeys(c for s in summaries.values() for c in s))
    L.append("| control | features/window | " + " | ".join(f"`{s}`" for s in schemes) +
             " | raw-input reference (Exp. 12) |\n")
    L.append("|---|---|" + "---|" * (len(schemes) + 1) + "\n")
    for c in controls:
        cells = []
        for s in schemes:
            e = summaries.get(s, {}).get(c)
            if not e:
                cells.append("—")
                continue
            p = e["pooled_patient_level"]
            cells.append(f"{p['mean']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]")
        nf = next((e["n_features"] for s in schemes
                   if (e := summaries.get(s, {}).get(c))), "—")
        ref = RAW_INPUT_REFERENCE.get(c)
        L.append(f"| `{c}` | {nf} | " + " | ".join(cells) + " | " +
                 (f"{ref:.3f}" if ref else "—") + " |\n")

    L.append("\n## Ordering test (OFF session scored more OFF-like than ON)\n\n")
    L.append("| control | norm | correct/total | binomial p |\n|---|---|---|---|\n")
    for s in schemes:
        for c, e in summaries.get(s, {}).items():
            o = e["ordering"]
            L.append(f"| `{c}` | `{s}` | {o['n_correct_direction_level']}/"
                     f"{o['n_direction_level']} | {o['binomial_p']:.3g} |\n")

    L.append("\n## Transfer test — within-patient vs cross-patient LOPO\n\n")
    L.append("Same features, same windows, same statistics; only the split changes.\n"
             "The DROP is the evidence, not the absolute value.\n\n")
    L.append("| control | norm | within-patient | cross-patient LOPO | drop | verdict |\n"
             "|---|---|---|---|---|---|\n")
    for s in schemes:
        for c, e in summaries.get(s, {}).items():
            t = e.get("transfer_test") or {}
            if t.get("skipped"):
                L.append(f"| `{c}` | `{s}` | — | — | — | not defined (see JSON) |\n")
            elif t:
                L.append(f"| `{c}` | `{s}` | {t['within_patient_mean']:.3f} | "
                         f"{t['cross_patient_lopo_mean']:.3f} | {t['drop']:.3f} | "
                         f"{t['verdict'].split(':')[0]} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L), encoding="utf-8")
    log.info("Wrote markdown report -> %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Negative controls for the task-split experiment")
    ap.add_argument("--controls", nargs="+", default=list(CONTROLS), choices=list(CONTROLS))
    ap.add_argument("--input-norm", nargs="+", default=None, choices=list(INPUT_NORM_SCHEMES),
                    help="normalisation scheme(s) to run the control matrix under "
                         "(default: input_norm.schemes in controls.yaml)")
    ap.add_argument("--match-level", default="none", choices=("none",) + MATCH_LEVELS,
                    help="duration-match sessions before fitting, same rule as "
                         "run_pooled_tasksplit.py. Use with recid to ask whether equalising "
                         "window counts closes the recording-identity route.")
    ap.add_argument("--match-where", default="uniform", choices=MATCH_WHERE)
    ap.add_argument("--match-k", type=int, default=None)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--aperiodic-estimator", default=None, choices=["specparam", "loglog_robust"],
                    help="override controls.yaml aperiodic.estimator (smoke tests)")
    ap.add_argument("--force-features", action="store_true",
                    help="ignore the feature cache and recompute")
    ap.add_argument("--run-tag", default="tasksplit_controls_norm")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    log_txt = _RUN_ROOT / "logs" / f"{args.run_tag}_{time.strftime('%Y%m%d')}.log"
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_txt, encoding="utf-8")],
    )

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    cc = load_controls_config()
    if args.aperiodic_estimator:
        cc.aperiodic["estimator"] = args.aperiodic_estimator
        log.warning("Aperiodic estimator OVERRIDDEN to %r (not the configured default).",
                    args.aperiodic_estimator)

    results_dir = _RUN_ROOT / "results" / ts_cfg.output["results_subdir"]
    pf_path = results_dir / "preflight_summary.json"
    if not pf_path.is_file():
        log.error("Pre-flight missing (%s). Run derive_task_windows.py first.", pf_path)
        return 1
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    eligible = list(pf["eligible_patients"])
    if args.max_patients:
        eligible = eligible[: args.max_patients]
        log.warning("SMOKE TEST: restricted to %d patient(s): %s", len(eligible), eligible)

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)
    sidecar["_row"] = np.arange(len(sidecar))
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])
    schemes = args.input_norm or cc.schemes

    # recid's collinearity with ON/OFF is CHECKED, not assumed — it is the single
    # most load-bearing claim in this whole analysis.
    recid_y = control_target(sidecar, "recid", cc)
    cond_y = sidecar["y_cond"].to_numpy().astype(int)
    per_patient_identical = {}
    for p in eligible:
        m = sidecar["patient_id"].to_numpy() == p
        per_patient_identical[p] = bool(np.array_equal(recid_y[m], cond_y[m]) or
                                        np.array_equal(1 - recid_y[m], cond_y[m]))
    n_identical = sum(per_patient_identical.values())
    log.info("recid target vs ON/OFF label: identical (up to a label flip) for %d/%d patients.",
             n_identical, len(eligible))
    if n_identical != len(eligible):
        log.error("recid target is NOT collinear with ON/OFF for %s — this contradicts the "
                  "documented design assumption; investigate before reading any number below.",
                  [p for p, v in per_patient_identical.items() if not v])

    # Normalisation rescale target: the SAME cohort-median window std the model
    # arms use, computed from the full manifest, not reimplemented here.
    norm_scale = None
    if any(s != "none" for s in schemes):
        manifest = CohortManifest.from_config(data_cfg)
        norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        log.info("Input normalisation rescale target (cohort-median window std) = %.6f", norm_scale)

    log.info("Controls %s x norms %s over %d patients x %d directions (CPU, no model).",
             args.controls, schemes, len(eligible), len(pairs))

    scfg = ts_cfg.stats
    summaries: dict[str, dict] = {}
    feature_reports: dict[str, dict] = {}
    t_start = time.time()

    for scheme in schemes:
        log.info("=" * 78)
        log.info("NORMALISATION: %s", scheme)
        blocks, frep = load_or_compute_features(
            sidecar, data_cfg, cc, scheme=scheme,
            norm_scale=(None if scheme == "none" else norm_scale),
            run_root=_RUN_ROOT, force=args.force_features)
        feature_reports[scheme] = frep
        summaries[scheme] = {}

        for control in args.controls:
            records = []
            for patient in eligible:
                for train_task, test_task in pairs:
                    records.append(_run_direction(sidecar, blocks, patient, train_task,
                                                  test_task, control, ts_cfg, cc,
                                                  match_level=args.match_level,
                                                  match_where=args.match_where,
                                                  match_k=args.match_k))
            by_patient: dict[str, list[float]] = {}
            for r in records:
                by_patient.setdefault(r["patient"], []).append(
                    float(r["window"]["balanced_accuracy"]))
            per_patient = {p: float(np.mean(v)) for p, v in by_patient.items()}
            pooled = aggregate_across_patients(
                per_patient, n_resamples=int(scfg["bootstrap"]["n_resamples"]),
                ci=float(scfg["bootstrap"]["ci"]),
                chance=float(scfg["tests"]["chance_level"]), seed=1)
            order = ordering_stats(records)
            aucs = [r["window"]["auc"] for r in records if r["window"]["auc"] is not None]
            entry = {
                "control": control, "description": CONTROLS[control], "input_norm": scheme,
                "n_features": records[0]["n_features"], "n_records": len(records),
                "per_patient": per_patient, "pooled_patient_level": pooled, "ordering": order,
                "auc_max": float(np.max(aucs)) if aucs else None,
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "n_runs_auc_above_0_95": int(sum(a > 0.95 for a in aucs)),
                "records": records,
            }
            if cc.transfer_test.get("enabled", True):
                entry["transfer_test"] = _transfer_test(sidecar, blocks, eligible,
                                                        ts_cfg, cc, control)
            summaries[scheme][control] = entry

            p = pooled
            log.info("CONTROL %-9s norm=%-7s | bal-acc %.3f [%.3f, %.3f] | above chance %d/%d "
                     "| AUC mean %.3f max %.3f | ordering %d/%d (p=%.3g)",
                     control, scheme, p["mean"], p["ci_low"], p["ci_high"],
                     p["n_above_chance"], p["n_patients"], entry["auc_mean"] or float("nan"),
                     entry["auc_max"] or float("nan"), order["n_correct_direction_level"],
                     order["n_direction_level"], order["binomial_p"])

            save_run({
                "run_tag": f"{args.run_tag}_{scheme}", "arm": control, "mode": f"control_{scheme}",
                "git_sha": git_sha(_PROJECT), "n_patients": p["n_patients"],
                "n_directions": len(pairs),
                "patient_mean_balanced_accuracy": p["mean"],
                "ci_low": p["ci_low"], "ci_high": p["ci_high"],
                "wilcoxon_p": (p.get("wilcoxon") or {}).get("p_value"),
                "sign_test_p": (p.get("sign_test") or {}).get("p_value"),
                "ordering_frac": order.get("frac_correct"),
                "ordering_p": order.get("binomial_p"),
                **entry,
            }, results_dir / "controls_norm", _RUN_ROOT / cc.output["master_csv"],
                csv_fields=_MASTER_CSV_FIELDS)

    # Summary and report paths are keyed on the run tag. Without this a variant run
    # (e.g. --match-level) silently OVERWRITES the canonical run's summary and
    # markdown report, since controls.yaml names them with fixed literals — the
    # per-run JSONs under controls_norm/ survive, but the derived files do not.
    # The default tag keeps the published filenames so existing references still
    # resolve; any other tag gets its own pair.
    summary_path, report_path = _tagged_outputs(results_dir, cc, args.run_tag)
    out = summary_path
    _write_json(out, {
        "run_tag": args.run_tag, "schemes": schemes, "controls": args.controls,
        "summaries": summaries, "eligible_patients": eligible,
        "norm_scale": norm_scale, "feature_reports": feature_reports,
        "recid_collinear_with_on_off": {"n_identical": n_identical,
                                        "n_patients": len(eligible),
                                        "per_patient": per_patient_identical},
        "raw_input_reference": RAW_INPUT_REFERENCE,
        "controls_config": cc.raw, "git_sha": git_sha(_PROJECT), "env": env_versions(),
    })
    log.info("Wrote %s", out)
    _write_report(report_path, summaries, cc, args.run_tag)

    # ---- verdict --------------------------------------------------------
    log.info("")
    for scheme in schemes:
        entries = summaries[scheme]
        worst = max(entries.values(), key=lambda s: s["pooled_patient_level"]["mean"])
        rec = entries.get("recid")
        log.info("VERDICT norm=%s: strongest non-neural control is %s at %.3f%s",
                 scheme, worst["control"], worst["pooled_patient_level"]["mean"],
                 (f"; recid (direct 'which recording' readout) = "
                  f"{rec['pooled_patient_level']['mean']:.3f}") if rec else "")
        if rec and rec["pooled_patient_level"]["mean"] > 0.55:
            log.warning(
                "norm=%s: recid is %.3f, WELL ABOVE CHANCE — normalisation has NOT closed the "
                "design. The two recordings remain directly identifiable from features with no "
                "neural levodopa content, and within a patient recording identity IS the ON/OFF "
                "label. Every task-split number under this normalisation is an UPPER BOUND that "
                "includes the leak.", scheme, rec["pooled_patient_level"]["mean"])
    log.info("Done in %.1f min.", (time.time() - t_start) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
