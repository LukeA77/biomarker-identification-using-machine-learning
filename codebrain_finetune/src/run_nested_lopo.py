"""Full nested-LOPO LoRA driver — search once, refit per fold, pool the sessions.

This is the production run behind the dissertation's headline number. Protocol
(FINE_TUNING_PLAN §6; the literal per-outer-fold search is ~330 h — we do NOT do
that):

  1. HP SEARCH (once). Random-sample ``n_configs`` from the LoRA search space
     {r, lr, dropout}. Score each by 5-fold GroupKFold over the *pooled* 35-patient
     cohort (grouped on patient — no within-patient leakage): train on inner-train,
     evaluate session-level balanced accuracy on inner-val, average over the 5
     folds. Pick the best config. (Pooled-once selection:
     2026-07-23. Mild optimism — each eventual LOPO test patient's data influenced
     HP choice — noted here and in the run log; standard for small-N nested CV.)

  2. OUTER LOPO (35 folds). With the chosen HPs fixed, refit a fresh LoRA model on
     each fold's 34 training patients (full overlapping windows; a GroupKFold inner
     split of a few patients held off for early stopping), evaluate the held-out
     patient's non-overlap windows, aggregate to its 2 sessions.

  3. AGGREGATE. Pool all 70 session predictions -> balanced-acc (primary), AUC,
     sens/spec, macro-F1; cluster bootstrap CI (resample patients) + within-subject
     permutation test. Per-fold table + pooled summary + LoRA-vs-frozen-probe.

Resumability & leak-safety over a ~15 h run:
  * Each outer fold writes its own JSON (deterministic filename); on restart, a
    fold whose JSON exists is loaded and skipped. The HP search writes one JSON and
    is likewise skipped if present. Never lose 14 h to a crash at fold 34.
  * ``del model; torch.cuda.empty_cache()`` after EVERY training; peak VRAM is
    reset before and logged after each training to catch creep across ~85 trainings.
  * Cohort invariants (35 / 37174) are checked live at start; abort loudly on drift.
  * Config-driven, seeded, fail-loud; no magic numbers.

Usage:
    # de-risk (this is why we don't just launch):
    python codebrain_finetune/src/run_nested_lopo.py --outer-folds 3 --n-configs 2 --max-epochs 3
    # full run:
    python codebrain_finetune/src/run_nested_lopo.py
    # full-FT comparison arm (reuses the LoRA-selected config, smaller LR):
    python codebrain_finetune/src/run_nested_lopo.py --mode full_ft
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# --- make both packages importable (mirrors the Stage-2/3/45 scripts) ---
_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent                 # codebrain_finetune/
_PROJECT = _RUN_ROOT.parent                     # Luke_Dissertation/
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import (  # noqa: E402
    load_data_config, CohortManifest, build_lopo_folds, make_fold_loaders,
)
from codebrain_ft import (  # noqa: E402
    load_finetune_config, build_classifier_model, trainable_parameter_report,
    make_patient_loader, inner_group_split, train_model, evaluate_session_level,
    seed_everything, fold_seed, INPUT_NORM_SCHEMES, cohort_median_scale,
)
from codebrain_ft.metrics import (  # noqa: E402
    classification_metrics, bootstrap_session_ci, permutation_test_balanced_accuracy,
)
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402

log = logging.getLogger("nested_lopo")

# Locked-cohort invariants (sanity-checked live; NOT re-derived).
EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174

# Stage-4 frozen linear-probe floor (mean pooling), for the head-to-head report.
FROZEN_PROBE = {"balanced_accuracy": 0.443, "ci": [0.357, 0.529], "auc": 0.482, "perm_p": 0.95}

_MASTER_CSV_FIELDS = [
    "run_tag", "arm", "mode", "git_sha", "lora_target", "lora_r", "lora_alpha",
    "lr", "lora_dropout", "n_sessions", "session_balanced_accuracy",
    "session_auc", "ci_low", "ci_high", "permutation_p",
]


# ---------------------------------------------------------------------------
# small JSON helpers (deterministic filenames so resume can skip-if-exists)
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=_json_default))
    tmp.replace(path)  # atomic-ish: never leave a half-written JSON that resume trusts


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# config surgery for the HP search (frozen dataclasses -> replace)
# ---------------------------------------------------------------------------
def _with_hp(ft_cfg, *, r: int, dropout: float):
    """Return a copy of ``ft_cfg`` with LoRA r/dropout overridden (alpha follows r)."""
    lora = dataclasses.replace(ft_cfg.lora, r=int(r), dropout=float(dropout))
    return dataclasses.replace(ft_cfg, lora=lora)


def _sample_configs(space: dict, n_configs: int, seed: int) -> list[dict]:
    """Random-sample up to ``n_configs`` distinct {r, lr, dropout} configs.

    Sampling without replacement from the full grid; if the grid is smaller than
    ``n_configs`` we return the whole grid (deduped). Seeded and deterministic.
    """
    grid = [
        {"r": r, "lr": lr, "dropout": d}
        for r, lr, d in itertools.product(space["r"], space["lr"], space["dropout"])
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(grid))
    n = min(n_configs, len(grid))
    return [grid[i] for i in idx[:n]]


# ---------------------------------------------------------------------------
# one training: build -> train -> eval a loader, with VRAM accounting + cleanup
# ---------------------------------------------------------------------------
def _train_and_eval(
    data_cfg, ft_cfg_eff, mode, train_loader, val_loader, eval_loader, *, lr, seed,
    max_epochs, device,
):
    """Build a fresh model, train (early-stop on val), evaluate ``eval_loader``.

    Returns ``(eval_result, train_result, peak_vram_bytes)``. Always frees the
    model + CUDA cache before returning, so VRAM cannot creep across trainings.
    """
    import torch

    seed_everything(seed, deterministic=ft_cfg_eff.seed.deterministic)
    torch.cuda.reset_peak_memory_stats()

    model = build_classifier_model(data_cfg, ft_cfg_eff, mode=mode, device=device)
    try:
        train_res = train_model(
            model, train_loader, val_loader, ft_cfg_eff,
            lr=lr, device=device, max_epochs=max_epochs,
        )
        eval_res = evaluate_session_level(model, eval_loader, ft_cfg_eff, device=device)
        peak = int(torch.cuda.max_memory_allocated())
    finally:
        del model
        torch.cuda.empty_cache()
    return eval_res, train_res, peak


# ---------------------------------------------------------------------------
# Stage: HP search (once, pooled over all patients)
# ---------------------------------------------------------------------------
def run_hp_search(
    data_cfg, ft_cfg, manifest, search_patients, *, n_configs, inner_folds,
    max_epochs, mode, base_seed, out_path, device, input_norm="none", norm_scale=None,
):
    """Search once and return the winning config dict + full search record.

    Skips (loads from disk) if ``out_path`` already exists.
    """
    if out_path.exists():
        rec = _read_json(out_path)
        log.info("HP search: found existing %s -> reuse best config %s (bal-acc %.3f)",
                 out_path.name, rec["best_config"], rec["best_score"])
        return rec["best_config"], rec

    configs = _sample_configs(ft_cfg.nested_lopo.inner_search.space, n_configs, base_seed)
    splits = inner_group_split(list(search_patients), n_folds=inner_folds, seed=base_seed)
    log.info("HP search: %d configs x %d inner GroupKFold folds over %d patients (pooled-once).",
             len(configs), len(splits), len(search_patients))
    log.info("  NOTE: pooled-once selection — mild optimism (test patients seen during "
             "HP choice); standard for small-N nested CV.")

    # Per-config checkpointing: the search is ~half the total compute (10x5=50
    # trainings ~8 h at full scale), so a mid-search crash must not restart it.
    # Each config's result is written to its own JSON (config index is stable
    # because sampling is seeded/deterministic); completed configs are skipped.
    cfg_dir = out_path.parent / (out_path.stem + "_configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)

    peak_vram = 0
    results = []
    for ci, cfg in enumerate(configs):
        cfg_path = cfg_dir / f"config_{ci:02d}.json"
        if cfg_path.exists():
            cres = _read_json(cfg_path)
            if cres["config"] != cfg:  # deterministic sampling drifted — refuse to trust it
                raise RuntimeError(
                    f"HP search checkpoint mismatch at {cfg_path.name}: stored {cres['config']} "
                    f"!= expected {cfg}. Delete {cfg_dir} and re-run.")
            results.append(cres)
            peak_vram = max(peak_vram, cres.get("peak_vram_bytes", 0))
            log.info("  config %d/%d %s | RESUME (exists) -> mean inner-val bal-acc %.3f",
                     ci + 1, len(configs), cfg, cres["mean_score"])
            continue

        ft_eff = _with_hp(ft_cfg, r=cfg["r"], dropout=cfg["dropout"])
        fold_scores = []
        cfg_peak = 0
        t0 = time.time()
        for si, (inner_train, inner_val) in enumerate(splits):
            seed = base_seed + 1000 * (ci + 1) + si
            train_loader = make_patient_loader(
                manifest, data_cfg, inner_train, nonoverlap_only=False, shuffle=True, seed=seed,
                input_norm=input_norm, norm_scale=norm_scale,
            )
            val_loader = make_patient_loader(
                manifest, data_cfg, inner_val,
                nonoverlap_only=data_cfg.overlap.eval_nonoverlap_only, shuffle=False, seed=seed,
                input_norm=input_norm, norm_scale=norm_scale,
            )
            eval_res, _tr, peak = _train_and_eval(
                data_cfg, ft_eff, mode, train_loader, val_loader, val_loader,
                lr=cfg["lr"], seed=seed, max_epochs=max_epochs, device=device,
            )
            cfg_peak = max(cfg_peak, peak)
            peak_vram = max(peak_vram, peak)
            bal = eval_res["session"]["balanced_accuracy"]
            fold_scores.append(bal)
            log.info("  config %d/%d %s | inner fold %d/%d | val bal-acc %.3f | peak %.1f GB",
                     ci + 1, len(configs), cfg, si + 1, len(splits), bal, peak / 1e9)
        cres = {
            "config": cfg, "fold_scores": fold_scores,
            "mean_score": float(np.nanmean(fold_scores)),
            "seconds": time.time() - t0, "peak_vram_bytes": cfg_peak,
        }
        _write_json(cfg_path, cres)
        results.append(cres)
        log.info("  config %d/%d %s -> mean inner-val bal-acc %.3f",
                 ci + 1, len(configs), cfg, cres["mean_score"])

    results.sort(key=lambda r: r["mean_score"], reverse=True)
    best = results[0]
    log.info("HP search complete. Best config %s (mean inner-val bal-acc %.3f).",
             best["config"], best["mean_score"])

    record = {
        "stage": "hp_search", "mode": mode, "pooled_once": True,
        "n_configs": len(configs), "inner_folds": inner_folds,
        "search_patients": list(search_patients),
        "best_config": best["config"], "best_score": best["mean_score"],
        "all_results": results, "peak_vram_bytes": peak_vram,
        "git_sha": git_sha(_PROJECT), "env": env_versions(),
    }
    _write_json(out_path, record)
    return best["config"], record


# ---------------------------------------------------------------------------
# Stage: one outer LOPO fold (refit + evaluate held-out patient)
# ---------------------------------------------------------------------------
def run_outer_fold(
    data_cfg, ft_cfg, manifest, fold, best_config, *, mode, inner_folds,
    max_epochs, base_seed, out_path, device, common, input_norm="none", norm_scale=None,
):
    """Refit on the fold's 34 train patients and evaluate the held-out patient.

    Skips (loads) if ``out_path`` exists. Returns the fold record dict.
    """
    if out_path.exists():
        rec = _read_json(out_path)
        s = rec["session"]
        log.info("fold %2d %-9s | RESUME (exists) | session bal-acc %.3f",
                 fold.index, fold.test_patient, s["balanced_accuracy"])
        return rec

    seed = fold_seed(base_seed, fold.index)
    ft_eff = _with_hp(ft_cfg, r=best_config["r"], dropout=best_config["dropout"])
    lr = best_config["lr"] if mode == "lora" else ft_cfg.train.lr_full_ft

    # Inner split of the 34 training patients: one GroupKFold split for early stopping.
    inner_splits = inner_group_split(list(fold.train_patients), n_folds=inner_folds, seed=seed)
    inner_train, inner_val = inner_splits[0]
    train_loader = make_patient_loader(
        manifest, data_cfg, inner_train, nonoverlap_only=False, shuffle=True, seed=seed,
        input_norm=input_norm, norm_scale=norm_scale,
    )
    val_loader = make_patient_loader(
        manifest, data_cfg, inner_val,
        nonoverlap_only=data_cfg.overlap.eval_nonoverlap_only, shuffle=False, seed=seed,
        input_norm=input_norm, norm_scale=norm_scale,
    )
    # Held-out patient, non-overlap windows. Built via make_patient_loader (rather
    # than make_fold_loaders) so the SAME input normalisation applies at eval time.
    test_loader = make_patient_loader(
        manifest, data_cfg, [fold.test_patient],
        nonoverlap_only=data_cfg.overlap.eval_nonoverlap_only, shuffle=False,
        seed=seed + fold.index, input_norm=input_norm, norm_scale=norm_scale,
    )

    t0 = time.time()
    eval_res, train_res, peak = _train_and_eval(
        data_cfg, ft_eff, mode, train_loader, val_loader, test_loader,
        lr=lr, seed=seed, max_epochs=max_epochs, device=device,
    )
    wall = time.time() - t0

    s, w = eval_res["session"], eval_res["window"]
    log.info("fold %2d %-9s | session bal-acc %.3f auc %s | best@%d | %.0fs | peak %.1f GB",
             fold.index, fold.test_patient, s["balanced_accuracy"],
             f"{s['auc']:.3f}" if s["auc"] is not None else "n/a",
             train_res.best_epoch, wall, peak / 1e9)
    sc = eval_res["session_scores"]
    for sid, prob, yt, nw in zip(sc["sessions"], sc["prob"], sc["y_true"], sc["n_windows"]):
        log.info("    session %-16s P(OFF)=%.3f true=%d (%d windows)", sid, prob, yt, nw)

    record = {
        **common, "stage": "outer_fold", "fold_index": fold.index,
        "test_patient": fold.test_patient, "mode": mode,
        "config": best_config, "lr": lr, "seed": seed,
        "inner_train_patients": inner_train, "inner_val_patients": inner_val,
        "best_epoch": train_res.best_epoch, "best_inner_val_monitor": train_res.best_monitor,
        "history": train_res.history, "train_seconds": train_res.train_seconds,
        "trainable_params": train_res.trainable_params,
        "wall_seconds": wall, "peak_vram_bytes": peak,
        "window": w, "session": s, "session_scores": sc,
    }
    _write_json(out_path, record)
    return record


# ---------------------------------------------------------------------------
# Aggregate: pool sessions across all completed folds -> headline metrics
# ---------------------------------------------------------------------------
def aggregate_and_report(fold_records, ft_cfg, *, mode, best_config, run_tag, log_md_path):
    prob, true, pat, sess = [], [], [], []
    per_fold = []
    for rec in sorted(fold_records, key=lambda r: r["fold_index"]):
        sc = rec["session_scores"]
        prob.extend(sc["prob"]); true.extend(sc["y_true"])
        pat.extend(sc["patients"]); sess.extend(sc["sessions"])
        per_fold.append({
            "fold": rec["fold_index"], "test_patient": rec["test_patient"],
            "session_bal_acc": rec["session"]["balanced_accuracy"],
            "session_auc": rec["session"]["auc"],
            "sessions": sc["sessions"], "prob": [round(p, 3) for p in sc["prob"]],
            "y_true": sc["y_true"], "best_epoch": rec.get("best_epoch"),
        })
    prob = np.asarray(prob); true = np.asarray(true); pat = np.asarray(pat)

    agg = classification_metrics(true, prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(
        true, prob, pat, n_resamples=ft_cfg.eval.bootstrap.n_resamples,
        ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base,
    )
    perm = permutation_test_balanced_accuracy(
        true, prob, pat, n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset,
    )

    log.info("=" * 74)
    log.info("NESTED LOPO (%s) — pooled over %d folds / %d sessions", mode, len(per_fold), len(true))
    log.info("  best config: %s", best_config)
    log.info("  balanced_accuracy = %.3f  (bootstrap %.0f%% CI [%.3f, %.3f])",
             agg.balanced_accuracy, boot["ci"] * 100,
             boot["ci_low"] if boot["ci_low"] is not None else float("nan"),
             boot["ci_high"] if boot["ci_high"] is not None else float("nan"))
    log.info("  AUC=%s  sens(OFF)=%s  spec(ON)=%s  macro_f1=%.3f  acc=%.3f",
             f"{agg.auc:.3f}" if agg.auc is not None else "n/a",
             f"{agg.sensitivity:.3f}" if agg.sensitivity is not None else "n/a",
             f"{agg.specificity:.3f}" if agg.specificity is not None else "n/a",
             agg.macro_f1, agg.accuracy)
    log.info("  permutation p-value = %.4f (null mean %.3f)", perm["p_value"], perm["null_mean"])
    log.info("  vs FROZEN PROBE: bal-acc %.3f (CI [%.3f, %.3f]), AUC %.3f, perm p %.2f",
             FROZEN_PROBE["balanced_accuracy"], FROZEN_PROBE["ci"][0], FROZEN_PROBE["ci"][1],
             FROZEN_PROBE["auc"], FROZEN_PROBE["perm_p"])
    log.info("=" * 74)

    summary = {
        "mode": mode, "best_config": best_config, "n_folds": len(per_fold),
        "n_sessions": len(true), "aggregate": agg.as_dict(),
        "bootstrap": boot, "permutation": perm,
        "frozen_probe_reference": FROZEN_PROBE, "per_fold": per_fold,
    }
    _write_markdown_report(log_md_path, summary, ft_cfg, mode, best_config, agg, boot, perm)
    return summary, agg, boot, perm


def _write_markdown_report(path, summary, ft_cfg, mode, best_config, agg, boot, perm):
    lines = []
    lines.append(f"# Nested LOPO ({mode}) — pooled session-level result\n")
    lines.append(f"- **Best config:** {best_config}\n")
    lines.append(f"- **Folds pooled:** {summary['n_folds']}  |  **sessions:** {summary['n_sessions']}\n")
    lines.append("\n## Pooled metrics\n\n")
    lines.append("| metric | value |\n|---|---|\n")
    ci = f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]" if boot["ci_low"] is not None else "n/a"
    lines.append(f"| balanced accuracy | **{agg.balanced_accuracy:.3f}** {ci} |\n")
    lines.append(f"| AUC | {agg.auc:.3f} |\n" if agg.auc is not None else "| AUC | n/a |\n")
    lines.append(f"| sensitivity (OFF) | {_f(agg.sensitivity)} |\n")
    lines.append(f"| specificity (ON) | {_f(agg.specificity)} |\n")
    lines.append(f"| macro F1 | {agg.macro_f1:.3f} |\n")
    lines.append(f"| accuracy | {agg.accuracy:.3f} |\n")
    lines.append(f"| permutation p | {perm['p_value']:.4f} (null {perm['null_mean']:.3f}) |\n")
    lines.append("\n## LoRA vs frozen probe\n\n")
    lines.append("| arm | bal-acc | CI | AUC | perm p |\n|---|---|---|---|---|\n")
    lines.append(f"| LoRA ({mode}) | {agg.balanced_accuracy:.3f} | {ci} | "
                 f"{_f(agg.auc)} | {perm['p_value']:.3f} |\n")
    fp = summary["frozen_probe_reference"]
    lines.append(f"| frozen probe | {fp['balanced_accuracy']:.3f} | "
                 f"[{fp['ci'][0]:.3f}, {fp['ci'][1]:.3f}] | {fp['auc']:.3f} | {fp['perm_p']:.2f} |\n")
    lines.append("\n## Per-fold\n\n")
    lines.append("| fold | patient | bal-acc | AUC | sessions (P(OFF)/true) | best epoch |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for pf in summary["per_fold"]:
        pairs = ", ".join(f"{s.split('|')[-1]}:{p}/{t}"
                          for s, p, t in zip(pf["sessions"], pf["prob"], pf["y_true"]))
        lines.append(f"| {pf['fold']} | {pf['test_patient']} | {pf['session_bal_acc']:.3f} | "
                     f"{_f(pf['session_auc'])} | {pairs} | {pf['best_epoch']} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote markdown report -> %s", path)


def _f(x):
    return f"{x:.3f}" if x is not None else "n/a"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Full nested-LOPO LoRA (search once + refit per fold)")
    ap.add_argument("--mode", default="lora", choices=["lora", "full_ft"],
                    help="lora (default) or full_ft comparison arm")
    ap.add_argument("--outer-folds", type=int, default=None,
                    help="limit to the first N outer folds (smoke test)")
    ap.add_argument("--n-configs", type=int, default=None,
                    help="override nested_lopo.inner_search.n_configs (smoke test)")
    ap.add_argument("--inner-folds", type=int, default=None,
                    help="override nested_lopo.inner_folds")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override train.max_epochs (smoke test)")
    ap.add_argument("--input-norm", default="none", choices=list(INPUT_NORM_SCHEMES),
                    help="per-window amplitude normalisation before the backbone "
                         "(global_z preserves topography; see EXPERIMENTS.md Exp. 7)")
    ap.add_argument("--run-tag", default=None, help="override results subdir / run tag")
    ap.add_argument("--reuse-hp", default=None,
                    help="path to an existing hp_search.json to reuse (e.g. full_ft reusing LoRA's)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    run_tag = args.run_tag or f"nested_{args.mode}"
    log_md = _RUN_ROOT / "logs" / f"stage5_lora_nested_{time.strftime('%Y%m%d')}.md"
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

    n_configs = args.n_configs if args.n_configs is not None else ft_cfg.nested_lopo.inner_search.n_configs
    inner_folds = args.inner_folds if args.inner_folds is not None else ft_cfg.nested_lopo.inner_folds
    max_epochs = args.max_epochs if args.max_epochs is not None else ft_cfg.train.max_epochs

    # --- cohort sanity-check (live, not re-derived) ---
    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows, balance=%s", n_pat, n_win, manifest.label_balance())
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d), expected (%d, %d). STOP.",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    folds = build_lopo_folds(manifest)
    if args.outer_folds is not None:
        folds = folds[:args.outer_folds]
    log.info("Run '%s' | mode=%s | %d outer folds | search: %d configs x %d inner | max_epochs=%d",
             run_tag, args.mode, len(folds), n_configs, inner_folds, max_epochs)

    # Input-amplitude normalisation (Experiment 7). The rescale target is the
    # cohort-median window std, so the backbone keeps its pretrained magnitude.
    norm_scale = None
    if args.input_norm != "none":
        norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        log.info("Input normalisation: %s (rescaled to cohort-median window std %.4f)",
                 args.input_norm, norm_scale)

    results_dir = ft_cfg.tracking.results_dir / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    hp_path = results_dir / "hp_search.json"

    common = dict(
        run_tag=run_tag, git_sha=git_sha(_PROJECT),
        checkpoint_sha256=file_sha256(data_cfg.paths.checkpoint),
        env=env_versions(), cohort={"patients": n_pat, "windows": n_win},
        lora_target=ft_cfg.lora.target,
        input_norm=args.input_norm, norm_scale=norm_scale,
    )

    # =====================================================================
    # 1. HP search (once) — or reuse
    # =====================================================================
    if args.reuse_hp:
        rec = _read_json(Path(args.reuse_hp))
        best_config = rec["best_config"]
        log.info("Reusing HP config from %s: %s (bal-acc %.3f)",
                 args.reuse_hp, best_config, rec["best_score"])
    elif args.mode == "full_ft":
        # full_ft reuses the LoRA search if available; else falls back to config defaults.
        lora_hp = ft_cfg.tracking.results_dir / "nested_lora" / "hp_search.json"
        if lora_hp.exists():
            best_config = _read_json(lora_hp)["best_config"]
            log.info("full_ft arm: reusing LoRA-selected config %s (lr overridden to %.1e).",
                     best_config, ft_cfg.train.lr_full_ft)
        else:
            best_config = {"r": ft_cfg.lora.r, "lr": ft_cfg.train.lr_full_ft, "dropout": ft_cfg.lora.dropout}
            log.warning("full_ft arm: no LoRA hp_search.json found, using config defaults %s.", best_config)
    else:
        t_search = time.time()
        best_config, _search_rec = run_hp_search(
            data_cfg, ft_cfg, manifest, manifest.patients,
            n_configs=n_configs, inner_folds=inner_folds, max_epochs=max_epochs,
            mode=args.mode, base_seed=ft_cfg.seed.base, out_path=hp_path, device=device,
            input_norm=args.input_norm, norm_scale=norm_scale,
        )
        log.info("HP search wall-clock: %.1f min", (time.time() - t_search) / 60)

    # =====================================================================
    # 2. Outer LOPO refit (resumable per fold)
    # =====================================================================
    fold_records = []
    t_outer = time.time()
    for fi, fold in enumerate(folds):
        out_path = results_dir / f"fold_{fold.index:02d}_{fold.test_patient}.json"
        rec = run_outer_fold(
            data_cfg, ft_cfg, manifest, fold, best_config, mode=args.mode,
            inner_folds=inner_folds, max_epochs=max_epochs, base_seed=ft_cfg.seed.base,
            out_path=out_path, device=device, common=common,
            input_norm=args.input_norm, norm_scale=norm_scale,
        )
        fold_records.append(rec)
        done = fi + 1
        rate = (time.time() - t_outer) / done
        log.info("  progress: %d/%d folds | ~%.1f min/fold | ETA %.1f min",
                 done, len(folds), rate / 60, rate * (len(folds) - done) / 60)

    # =====================================================================
    # 3. Aggregate + report
    # =====================================================================
    summary, agg, boot, perm = aggregate_and_report(
        fold_records, ft_cfg, mode=args.mode, best_config=best_config,
        run_tag=run_tag, log_md_path=log_md,
    )

    summary_record = {
        **common, "run_tag": f"{run_tag}_summary", "arm": run_tag, "mode": args.mode,
        "lora_r": best_config["r"], "lora_alpha": 2 * best_config["r"],
        "lr": best_config["lr"], "lora_dropout": best_config["dropout"],
        "n_sessions": summary["n_sessions"],
        "session_balanced_accuracy": agg.balanced_accuracy, "session_auc": agg.auc,
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "permutation_p": perm["p_value"],
        **summary,
    }
    save_run(summary_record, results_dir, ft_cfg.tracking.master_csv, csv_fields=_MASTER_CSV_FIELDS)
    log.info("Done. Summary JSON + master CSV row written under %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
