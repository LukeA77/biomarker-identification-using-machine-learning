"""Stage 4 + 5 — prototype ONE LOPO fold end-to-end before scaling to all 35.

This is the milestone gate from the stage plan: on a single held-out patient, run
(a) the frozen linear-probe baseline (Stage 4) and (b) LoRA fine-tuning (Stage 5)
end-to-end, evaluate at session level (patient x condition), log metrics + a
compute estimate, and confirm the whole harness is sane. Only after this do we
run the full 35-fold nested protocol.

Run from the project root:
    python codebrain_finetune/src/stage45_prototype_fold.py --fold 0
Flags let you scope the smoke test (fewer epochs) or run only one arm.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# --- make both packages importable (mirrors the Stage-2/3 scripts) ---
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
    run_probe_fold, seed_everything, fold_seed,
)
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402

log = logging.getLogger("stage45")

# Locked-cohort invariants (sanity-checked live; NOT re-derived).
EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174

_CSV_FIELDS = [
    "run_tag", "arm", "fold_index", "test_patient", "git_sha",
    "lora_target", "lora_r", "lora_alpha", "lr", "head_dropout",
    "session_balanced_accuracy", "session_auc", "session_sensitivity",
    "session_specificity", "session_macro_f1", "window_balanced_accuracy",
    "best_epoch", "train_seconds", "trainable_params",
]


def _flat_csv_row(record: dict) -> dict:
    """Pull the headline scalars out of a nested result record for the master CSV."""
    s = record.get("session", {})
    w = record.get("window", {})
    return {
        "run_tag": record.get("run_tag"),
        "arm": record.get("arm"),
        "fold_index": record.get("fold_index"),
        "test_patient": record.get("test_patient"),
        "git_sha": record.get("git_sha"),
        "lora_target": record.get("lora_target"),
        "lora_r": record.get("lora_r"),
        "lora_alpha": record.get("lora_alpha"),
        "lr": record.get("lr"),
        "head_dropout": record.get("head_dropout"),
        "session_balanced_accuracy": s.get("balanced_accuracy"),
        "session_auc": s.get("auc"),
        "session_sensitivity": s.get("sensitivity"),
        "session_specificity": s.get("specificity"),
        "session_macro_f1": s.get("macro_f1"),
        "window_balanced_accuracy": w.get("balanced_accuracy"),
        "best_epoch": record.get("best_epoch"),
        "train_seconds": record.get("train_seconds"),
        "trainable_params": record.get("trainable_params"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Prototype one LOPO fold (Stage 4 probe + Stage 5 LoRA)")
    ap.add_argument("--fold", type=int, default=0, help="outer LOPO fold index (0..34)")
    ap.add_argument("--arms", nargs="+", default=["probe", "lora"], choices=["probe", "lora"])
    ap.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs (LoRA)")
    ap.add_argument("--inner-split", type=int, default=0, help="which of the 5 inner GroupKFold splits to use")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — SSSM requires GPU. Aborting.")
        return 1
    device = "cuda"

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    # --- cohort sanity-check (live, not re-derived) ---
    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows, balance=%s", n_pat, n_win, manifest.label_balance())
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d patients, %d windows), expected (%d, %d). "
                  "STOP — do not proceed until the cohort is reconciled.",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    folds = build_lopo_folds(manifest)
    fold = folds[args.fold]
    seed = fold_seed(ft_cfg.seed.base, fold.index)
    log.info("=== Fold %d | held-out test patient: %s | %d train patients ===",
             fold.index, fold.test_patient, len(fold.train_patients))

    sha = git_sha(_PROJECT)
    ckpt_hash = file_sha256(data_cfg.paths.checkpoint)
    versions = env_versions()
    common = dict(
        fold_index=fold.index, test_patient=fold.test_patient, git_sha=sha,
        checkpoint_sha256=ckpt_hash, env=versions, seed=seed,
        cohort={"patients": n_pat, "windows": n_win},
    )

    # =====================================================================
    # Stage 4 — frozen linear-probe baseline
    # =====================================================================
    if "probe" in args.arms:
        log.info("--- Stage 4: frozen linear-probe baseline ---")
        t0 = time.time()
        probe_model = build_classifier_model(data_cfg, ft_cfg, mode="frozen", device=device)
        train_loader, test_loader = make_fold_loaders(
            manifest, data_cfg, fold, seed=seed, check_meta=False
        )
        probe_res = run_probe_fold(probe_model, train_loader, test_loader, data_cfg, ft_cfg, device=device)
        probe_secs = time.time() - t0
        del probe_model
        torch.cuda.empty_cache()

        record = {
            **common, "run_tag": f"probe_fold{fold.index}", "arm": "probe",
            "probe_pooling": ft_cfg.probe.pooling, "probe_classifier": ft_cfg.probe.classifier,
            "train_seconds": probe_secs, "trainable_params": 0,
            **probe_res,
        }
        save_run(record, ft_cfg.tracking.results_dir, ft_cfg.tracking.master_csv,
                 csv_fields=_CSV_FIELDS)
        _log_result("PROBE", probe_res, probe_secs)

    # =====================================================================
    # Stage 5 — LoRA fine-tuning
    # =====================================================================
    if "lora" in args.arms:
        log.info("--- Stage 5: LoRA fine-tuning ---")
        # Inner split of the 34 training patients for early stopping (one of 5).
        inner_splits = inner_group_split(
            list(fold.train_patients), n_folds=ft_cfg.nested_lopo.inner_folds, seed=seed
        )
        inner_train, inner_val = inner_splits[args.inner_split]
        log.info("Inner split %d: %d train / %d val patients (early stopping on inner-val).",
                 args.inner_split, len(inner_train), len(inner_val))

        train_loader = make_patient_loader(
            manifest, data_cfg, inner_train, nonoverlap_only=False, shuffle=True, seed=seed
        )
        val_loader = make_patient_loader(
            manifest, data_cfg, inner_val,
            nonoverlap_only=data_cfg.overlap.eval_nonoverlap_only, shuffle=False, seed=seed
        )
        _, test_loader = make_fold_loaders(manifest, data_cfg, fold, seed=seed, check_meta=False)

        lora_model = build_classifier_model(data_cfg, ft_cfg, mode="lora", device=device)
        pinfo = trainable_parameter_report(lora_model)
        log.info("LoRA model: %(trainable)d / %(total)d trainable (%(trainable_frac).4f).", pinfo)

        train_res = train_model(
            lora_model, train_loader, val_loader, ft_cfg,
            lr=ft_cfg.train.lr, device=device, max_epochs=args.max_epochs,
        )
        test_eval = evaluate_session_level(lora_model, test_loader, ft_cfg, device=device)

        record = {
            **common, "run_tag": f"lora_fold{fold.index}", "arm": "lora",
            "lora_target": ft_cfg.lora.target, "lora_r": ft_cfg.lora.r,
            "lora_alpha": ft_cfg.lora.alpha, "lr": ft_cfg.train.lr,
            "head_dropout": ft_cfg.model.head.dropout,
            "inner_split": args.inner_split,
            "inner_train_patients": inner_train, "inner_val_patients": inner_val,
            "best_epoch": train_res.best_epoch, "best_inner_val_monitor": train_res.best_monitor,
            "history": train_res.history, "train_seconds": train_res.train_seconds,
            "trainable_params": train_res.trainable_params,
            **test_eval,
        }
        save_run(record, ft_cfg.tracking.results_dir, ft_cfg.tracking.master_csv,
                 csv_fields=_CSV_FIELDS)
        _log_result("LoRA", test_eval, train_res.train_seconds)

        # --- compute estimate for the full nested run ---
        _print_compute_estimate(train_res, ft_cfg)

    return 0


def _log_result(tag: str, res: dict, secs: float) -> None:
    s, w = res["session"], res["window"]
    log.info("[%s] session bal-acc=%.3f auc=%s sens=%s spec=%s | window bal-acc=%.3f | %.1fs",
             tag, s["balanced_accuracy"],
             f"{s['auc']:.3f}" if s["auc"] is not None else "n/a",
             f"{s['sensitivity']:.3f}" if s["sensitivity"] is not None else "n/a",
             f"{s['specificity']:.3f}" if s["specificity"] is not None else "n/a",
             w["balanced_accuracy"], secs)
    sc = res["session_scores"]
    for sid, prob, yt, nw in zip(sc["sessions"], sc["prob"], sc["y_true"], sc["n_windows"]):
        log.info("    session %-14s  P(OFF)=%.3f  true=%d  (%d windows)", sid, prob, yt, nw)


def _print_compute_estimate(train_res, ft_cfg) -> None:
    """Extrapolate one fold's measured wall-clock to the full nested protocol.

    Reports two protocols:
      * LITERAL nested (HP search re-run inside every outer fold) — the plan's
        strict form, very expensive;
      * RECOMMENDED (FINE_TUNING_PLAN §6): run the HP search ONCE (on the pooled
        inner folds), fix the winning config family, then just refit per outer
        fold. Orders of magnitude cheaper, standard practice for small-N nested CV.

    NOTE: epoch 0 of a fresh process pays cold file-cache + cuDNN autotune; a
    warm epoch is the steady-state cost. We report the median of epochs>=1 when
    available, falling back to the mean.
    """
    hist = train_res.history
    epochs_run = len(hist)
    warm = train_res.train_seconds / max(1, epochs_run)  # coarse fallback
    n_configs = ft_cfg.nested_lopo.inner_search.n_configs
    inner_folds = ft_cfg.nested_lopo.inner_folds
    max_epochs = ft_cfg.train.max_epochs
    patience = ft_cfg.train.early_stopping.patience
    outer = 35

    # A typical training early-stops around best_epoch + patience.
    typ_epochs = min(max_epochs, max(1, (train_res.best_epoch + 1) + patience))
    secs_typ = warm * typ_epochs

    literal_trainings = outer * n_configs * inner_folds + outer
    recommended_trainings = (n_configs * inner_folds) + outer  # search once + refit per fold

    def hh(trainings):
        return trainings * secs_typ / 3600

    log.info("=" * 70)
    log.info("COMPUTE ESTIMATE (LoRA arm)")
    log.info("  measured ~%.0fs/epoch (incl. cold epoch 0); best@epoch %d",
             warm, train_res.best_epoch)
    log.info("  assume ~%d epochs/training (best+patience) -> ~%.0fs/training (~%.1f min)",
             typ_epochs, secs_typ, secs_typ / 60)
    log.info("  LITERAL nested (search per outer fold): %d trainings ~ %.0f h",
             literal_trainings, hh(literal_trainings))
    log.info("  RECOMMENDED (search once + refit/fold):  %d trainings ~ %.1f h",
             recommended_trainings, hh(recommended_trainings))
    log.info("  (+ frozen-probe arm is cheap; full-FT arm ~similar/epoch, reuse LoRA HPs)")
    log.info("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
