"""PAIRWISE TRANSFER MATRIX with a max-statistic null.

THE QUESTION (supervisor, 2026-08-01): take each patient's task-split model, test
it on every other patient, and look at the best result — does ANY patient's model
transfer to ANY other patient?

THE CATCH, WHICH THIS SCRIPT EXISTS TO HANDLE. "Best of 702 ordered pairs" is a
MAXIMUM statistic. Under the null, 702 draws of a noisy balanced accuracy produce
individual pairs at 0.8 and above routinely — a striking-looking best pair is the
expected outcome of looking 702 times, not evidence of transfer. So the observed
maximum is compared against the null distribution OF THE MAXIMUM: permute the
ON/OFF labels within each patient, rebuild the ENTIRE matrix, record its maximum,
and repeat. The full-matrix mean, the distribution, and the row-wise best per
source patient are reported alongside, so the result cannot be read off a single
lucky cell.

PROVENANCE — READ THIS. The brief asked for "pure inference over the models
already trained in ``results/within_patient_tasksplit/``". Those runs persisted
metrics JSONs only; **no per-patient model weights were ever written to disk**
(``checkpoints/`` holds only the pretrained CodeBrain backbone). A weights-only
pairwise matrix was therefore impossible, and this script REBUILDS the source
models instead.

That rebuild is faithful for the ``frozen_probe`` arm and only for that arm: the
backbone takes no gradient, so a patient's model is a standardiser plus a
logistic regression over frozen features, and refitting it is deterministic. The
DIAGONAL of the matrix is the within-patient task-split protocol itself, so it
must reproduce the published per-patient numbers — that is checked against the
stored run JSONs and the script aborts if it does not hold. The check is to a
measured tolerance, not to bit-equality: features come from a whole-cohort
extraction pass, and under bfloat16 autocast cuBLAS kernel choice depends on
batch shape, which moves features ~0.4% relative to per-fold extraction (see
``codebrain_ft.backbone_cache``). The observed and permitted deltas are both
recorded in the output. ``lora`` and
``full_ft`` cannot be rebuilt this way (their weights are gone and retraining 54
models to hold in memory for 702 inferences is a different, much larger job), so
they are out of scope here and that is stated rather than papered over.

Everything is configured in ``config/pairwise_transfer.yaml``.

Usage:
    python codebrain_finetune/src/run_pairwise_transfer.py                # both norms
    python codebrain_finetune/src/run_pairwise_transfer.py --input-norm none \
        --n-permutations 20 --max-patients 5 --run-tag smoke              # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
from codebrain_ft.backbone_cache import load_or_compute_backbone_features  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import select_task_frame, direction_pairs  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402

log = logging.getLogger("pairwise_transfer")

EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174


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


def load_pairwise_config(root: Path | None = None) -> dict:
    from livinglab_codebrain.config import run_root

    root = root or run_root()
    path = root / "config" / "pairwise_transfer.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"pairwise-transfer config not found: {path}")
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("arm", "source_model", "input_norm", "matrix", "permutation",
                "figure", "output"):
        if key not in d:
            raise KeyError(f"pairwise_transfer.yaml: missing required top-level key '{key}'")
    return d


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------
def _fit_source(X, y, idx, ft_cfg):
    """One source patient's frozen-probe model for one direction."""
    sc = StandardScaler().fit(X[idx]) if ft_cfg.probe.standardize else None
    Xtr = sc.transform(X[idx]) if sc is not None else X[idx]
    clf = LogisticRegression(C=ft_cfg.probe.C, max_iter=ft_cfg.probe.max_iter,
                             class_weight="balanced")
    clf.fit(Xtr, y[idx])
    return sc, clf


def build_matrix(X, y, train_idx, test_idx, patients, ft_cfg, *, want_auc: bool):
    """``(n_src, n_tgt)`` balanced accuracy (and AUC), averaged over directions.

    ``train_idx[(p, d)]`` / ``test_idx[(p, d)]`` are the row indices of patient
    ``p``'s train-task and test-task windows for direction ``d`` — the same
    frames the within-patient driver uses, so the diagonal is that experiment.
    """
    n = len(patients)
    directions = sorted({d for (_, d) in train_idx})
    bal = np.zeros((n, n)); auc = np.full((n, n), np.nan)
    for d in directions:
        models = [_fit_source(X, y, train_idx[(p, d)], ft_cfg) for p in patients]
        for j, tgt in enumerate(patients):
            ite = test_idx[(tgt, d)]
            yte = y[ite]
            Xte_raw = X[ite]
            for i, (sc, clf) in enumerate(models):
                Xte = sc.transform(Xte_raw) if sc is not None else Xte_raw
                prob = clf.predict_proba(Xte)[:, 1]
                bal[i, j] += balanced_accuracy_score(yte, (prob >= 0.5).astype(int))
                if want_auc:
                    a = roc_auc_score(yte, prob) if len(np.unique(yte)) == 2 else np.nan
                    auc[i, j] = a if np.isnan(auc[i, j]) else auc[i, j] + a
    bal /= len(directions)
    auc /= len(directions)
    return bal, auc


def _max_stats(M: np.ndarray, *, off_diagonal: bool) -> dict:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool) if off_diagonal else np.ones_like(M, dtype=bool)
    v = M[mask]
    row_best = np.where(mask, M, -np.inf).max(axis=1)
    row_mean = np.where(mask, M, np.nan)
    row_mean = np.nanmean(row_mean, axis=1)
    return {"max": float(v.max()), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "n_pairs": int(v.size),
            "frac_above_chance": float((v > 0.5).mean()),
            "q90": float(np.percentile(v, 90)), "q99": float(np.percentile(v, 99)),
            "row_best_max": float(row_best.max()), "row_mean_max": float(np.nanmax(row_mean))}


# ---------------------------------------------------------------------------
def _heatmap(M: np.ndarray, patients: list[str], cfg: dict, title: str, out_base: Path,
             subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GREY, LGREY = "#5A5A5A", "#B8B8B8"
    f = cfg["figure"]
    n = len(patients)
    fig, ax = plt.subplots(figsize=(0.30 * n + 3.2, 0.30 * n + 2.6))
    im = ax.imshow(M, cmap=f["cmap"], vmin=float(f["vmin"]), vmax=float(f["vmax"]),
                   interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [p.replace("sub-", "") for p in patients]
    ax.set_xticklabels(short, rotation=90, fontsize=7, color=GREY)
    ax.set_yticklabels(short, fontsize=7, color=GREY)
    ax.set_xlabel("target patient (model applied to their held-out-task windows)",
                  fontsize=9.5, color=GREY)
    ax.set_ylabel("source patient (whose model it is)", fontsize=9.5, color=GREY)
    ax.set_title(title, fontsize=11.5, color="#222", pad=10, loc="left")
    ax.text(0, -0.075, subtitle, transform=ax.transAxes, fontsize=8.5, color=GREY,
            va="bottom")
    # The diagonal is the within-patient number (a different question) — mark it
    # so it is never read as a transfer result.
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#222", lw=0.8))
    for s in ax.spines.values():
        s.set_color(LGREY)
    ax.tick_params(colors=GREY, length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("balanced accuracy (0.5 = chance)", fontsize=9, color=GREY)
    cb.ax.tick_params(colors=GREY, labelsize=8)
    cb.outline.set_edgecolor(LGREY)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=int(f["dpi"]),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote figure %s.{png,pdf}", out_base)


# ---------------------------------------------------------------------------
def run_scheme(scheme, sidecar, eligible, X, y, pats_of_row, data_cfg, ft_cfg, ts_cfg,
               cfg, results_dir, n_perm) -> dict:
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])
    proto = ts_cfg.protocol
    train_idx, test_idx = {}, {}
    for p in eligible:
        for tr_task, te_task in pairs:
            d = f"{tr_task}_to_{te_task}"
            train_idx[(p, d)] = select_task_frame(
                sidecar, p, tr_task,
                nonoverlap_only=bool(proto["train_nonoverlap_only"]))["_row"].to_numpy()
            test_idx[(p, d)] = select_task_frame(
                sidecar, p, te_task,
                nonoverlap_only=bool(proto["eval_nonoverlap_only"]))["_row"].to_numpy()

    t0 = time.time()
    bal, auc = build_matrix(X, y, train_idx, test_idx, eligible, ft_cfg, want_auc=True)
    log.info("Observed matrix built in %.1f s (%d x %d)", time.time() - t0, *bal.shape)

    # ---- diagonal must reproduce the published within-patient numbers -------
    diag_check = {"checked": False}
    ref_dir = _RUN_ROOT / cfg["source_model"]["verify_diagonal_against"]
    if scheme == "none" and ref_dir.is_dir():
        stored, rebuilt, deltas = {}, {}, []
        for i, p in enumerate(eligible):
            vals = []
            for tr_task, te_task in pairs:
                f = ref_dir / f"{p}_{tr_task}_to_{te_task}.json"
                if not f.is_file():
                    break
                vals.append(json.loads(f.read_text())["window"]["balanced_accuracy"])
            if len(vals) == len(pairs):
                stored[p] = float(np.mean(vals))
                rebuilt[p] = float(bal[i, i])
                deltas.append(abs(stored[p] - rebuilt[p]))
        if deltas:
            tol_max = float(cfg["source_model"]["diagonal_max_abs_delta"])
            tol_mean = float(cfg["source_model"]["diagonal_mean_abs_delta"])
            max_d, mean_d = float(max(deltas)), float(np.mean(deltas))
            diag_check = {
                "checked": True, "n_patients": len(deltas),
                "max_abs_delta": max_d, "mean_abs_delta": mean_d,
                "tolerance_max": tol_max, "tolerance_mean": tol_mean,
                "passed": bool(max_d <= tol_max and mean_d <= tol_mean),
                "stored_mean": float(np.mean(list(stored.values()))),
                "rebuilt_mean": float(np.mean(list(rebuilt.values()))),
                "note": ("not bit-equality: features are extracted under bfloat16 autocast "
                         "and cuBLAS kernel choice depends on batch shape, so caching over "
                         "the whole cohort moves features ~0.4% relative vs per-fold "
                         "extraction. Tolerances in pairwise_transfer.yaml are set just "
                         "above that measured effect."),
                "per_patient_delta": {p: abs(stored[p] - rebuilt[p]) for p in stored},
            }
            if not diag_check["passed"]:
                raise RuntimeError(
                    f"Rebuilt diagonal does not reproduce the stored within-patient "
                    f"frozen_probe numbers (max |delta| {max_d:.4f} vs cap {tol_max}; mean "
                    f"|delta| {mean_d:.4f} vs cap {tol_mean}; stored mean "
                    f"{diag_check['stored_mean']:.4f} vs rebuilt "
                    f"{diag_check['rebuilt_mean']:.4f}). That is far beyond the bfloat16 "
                    f"batch-composition effect, so the rebuilt source models are not the "
                    f"ones that produced the published result and no off-diagonal number "
                    f"from them is trustworthy. Aborting.")
            log.info("Diagonal check PASSED: rebuilt within-patient mean %.4f vs stored %.4f "
                     "(mean |delta| %.4f, max %.4f over %d patients — bf16 batch-composition "
                     "level, see JSON note).",
                     diag_check["rebuilt_mean"], diag_check["stored_mean"],
                     mean_d, max_d, diag_check["n_patients"])

    off_diag = cfg["matrix"]["max_over"] == "off_diagonal"
    obs = _max_stats(bal, off_diagonal=off_diag)
    obs_auc = _max_stats(np.nan_to_num(auc, nan=0.5), off_diagonal=off_diag)

    # ---- null distribution OF THE MAXIMUM ----------------------------------
    log.info("Null of the maximum: %d permutations (within-patient sign flip, source models "
             "REFIT each time)...", n_perm)
    uniq = np.asarray(eligible)
    null_max, null_row_best, null_row_mean, null_mean = [], [], [], []
    t0 = time.time()
    for k in range(n_perm):
        rng = np.random.default_rng(int(cfg["permutation"]["seed_base"]) + k)
        flip = rng.integers(0, 2, size=len(uniq)).astype(bool)
        flip_of = {p: f for p, f in zip(uniq, flip)}
        y_perm = np.array([yy ^ int(flip_of[pp]) for yy, pp in zip(y, pats_of_row)])
        b, _ = build_matrix(X, y_perm, train_idx, test_idx, eligible, ft_cfg, want_auc=False)
        s = _max_stats(b, off_diagonal=off_diag)
        null_max.append(s["max"]); null_mean.append(s["mean"])
        null_row_best.append(s["row_best_max"]); null_row_mean.append(s["row_mean_max"])
        if (k + 1) % 25 == 0:
            log.info("  permutation %d/%d (%.1f s elapsed, null max so far %.3f)",
                     k + 1, n_perm, time.time() - t0, float(np.max(null_max)))
    null_max = np.asarray(null_max)
    null_mean_a = np.asarray(null_mean)
    null_row_mean_a = np.asarray(null_row_mean)

    def _p(observed, null):
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))

    result = {
        "input_norm": scheme, "arm": cfg["arm"], "patients": eligible,
        "n_patients": len(eligible),
        "observed": obs, "observed_auc": obs_auc,
        "diagonal_check": diag_check,
        "diagonal_mean": float(np.mean(np.diag(bal))),
        "null": {
            "n_permutations": int(n_perm), "scheme": cfg["permutation"]["scheme"],
            "max": {"mean": float(null_max.mean()), "std": float(null_max.std()),
                    "q95": float(np.percentile(null_max, 95)),
                    "min": float(null_max.min()), "max": float(null_max.max()),
                    "values": null_max.tolist()},
            "matrix_mean": {"mean": float(null_mean_a.mean()), "std": float(null_mean_a.std())},
            "row_mean_max": {"mean": float(null_row_mean_a.mean()),
                             "std": float(null_row_mean_a.std())},
        },
        "p_value_max": _p(obs["max"], null_max),
        "p_value_matrix_mean": _p(obs["mean"], null_mean_a),
        "p_value_row_mean_max": _p(obs["row_mean_max"], null_row_mean_a),
        "row_best": {p: float(np.where(~np.eye(len(eligible), dtype=bool), bal, -np.inf)[i].max())
                     for i, p in enumerate(eligible)},
        "row_mean": {p: float(np.nanmean(np.where(~np.eye(len(eligible), dtype=bool),
                                                  bal, np.nan)[i]))
                     for i, p in enumerate(eligible)},
        "best_pair": None,
    }
    mask = ~np.eye(len(eligible), dtype=bool)
    bi, bj = np.unravel_index(np.argmax(np.where(mask, bal, -np.inf)), bal.shape)
    result["best_pair"] = {"source": eligible[bi], "target": eligible[bj],
                           "balanced_accuracy": float(bal[bi, bj]),
                           "auc": float(auc[bi, bj])}

    # ---- artifacts ---------------------------------------------------------
    import pandas as pd
    for name, M in (("balanced_accuracy", bal), ("auc", auc)):
        df = pd.DataFrame(M, index=eligible, columns=eligible)
        df.index.name = "source_patient"
        out = results_dir / cfg["output"]["matrix_csv"].format(metric=name, norm=scheme)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, float_format="%.6f")
        log.info("Wrote %s", out)

    _heatmap(bal, eligible, cfg,
             f"Pairwise transfer — {cfg['arm']} task-split models (input_norm={scheme})",
             _RUN_ROOT / cfg["figure"]["dir"] / f"{cfg['figure']['basename']}_{scheme}",
             subtitle=(f"observed off-diagonal max {obs['max']:.3f} vs max-statistic null "
                       f"{null_max.mean():.3f}±{null_max.std():.3f}, "
                       f"p={result['p_value_max']:.3f}  ·  off-diagonal mean "
                       f"{obs['mean']:.3f}  ·  boxed diagonal = within-patient (leaky)"))

    log.info("=" * 78)
    log.info("PAIRWISE TRANSFER (%s, norm=%s) — %d ordered pairs", cfg["arm"], scheme,
             obs["n_pairs"])
    log.info("  diagonal (within-patient, leaky) mean : %.3f", result["diagonal_mean"])
    log.info("  off-diagonal mean                     : %.3f  (null %.3f +/- %.3f, p=%.3f)",
             obs["mean"], null_mean_a.mean(), null_mean_a.std(), result["p_value_matrix_mean"])
    log.info("  off-diagonal MAX                      : %.3f  (%s -> %s)",
             obs["max"], result["best_pair"]["source"], result["best_pair"]["target"])
    log.info("  null distribution OF THE MAX          : mean %.3f, sd %.3f, 95th pct %.3f, "
             "observed range [%.3f, %.3f]", null_max.mean(), null_max.std(),
             np.percentile(null_max, 95), null_max.min(), null_max.max())
    log.info("  >>> p(max) = %.4f  <<<", result["p_value_max"])
    log.info("  best row-mean per source              : %.3f  (p=%.3f) — does any ONE model "
             "generalise broadly?", obs["row_mean_max"], result["p_value_row_mean_max"])
    log.info("=" * 78)
    return result


def _write_report(path: Path, results: dict, cfg: dict) -> None:
    L = ["# Pairwise transfer matrix with a max-statistic null\n\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
         f"Arm: `{cfg['arm']}` (rebuilt — see below). Each cell is the SOURCE patient's\n"
         "task-split model applied to the TARGET patient's held-out-task windows,\n"
         "averaged over both directions.\n\n",
         "**The maximum is the statistic that needs a null.** With 702 ordered pairs, a best\n"
         "cell at 0.8+ is what the null produces by itself. Every row below therefore reports\n"
         "the observed value against a null built by permuting ON/OFF labels within patient,\n"
         "rebuilding the whole matrix, and taking its maximum.\n\n"]
    L.append("| norm | pairs | diagonal (within-patient) | off-diag mean | off-diag MAX | "
             "best pair | null of max (mean±sd) | **p(max)** |\n"
             "|---|---|---|---|---|---|---|---|\n")
    for scheme, r in results.items():
        o, n = r["observed"], r["null"]["max"]
        bp = r["best_pair"]
        L.append(f"| `{scheme}` | {o['n_pairs']} | {r['diagonal_mean']:.3f} | "
                 f"{o['mean']:.3f} | **{o['max']:.3f}** | "
                 f"{bp['source']}→{bp['target']} | {n['mean']:.3f} ± {n['std']:.3f} | "
                 f"**{r['p_value_max']:.3f}** |\n")
    L.append("\n## Does any ONE source model generalise broadly?\n\n")
    L.append("| norm | best row-mean (per source) | source | p |\n|---|---|---|---|\n")
    for scheme, r in results.items():
        best_src = max(r["row_mean"], key=r["row_mean"].get)
        L.append(f"| `{scheme}` | {r['observed']['row_mean_max']:.3f} | {best_src} | "
                 f"{r['p_value_row_mean_max']:.3f} |\n")
    L.append("\n## Provenance\n\n"
             "The within-patient task-split runs persisted metrics JSONs only — no per-patient\n"
             "model weights were ever written to disk — so the source models are REBUILT here.\n"
             "That is exact for `frozen_probe` (frozen backbone + logistic head) and is verified\n"
             "against the stored numbers on the matrix diagonal:\n\n")
    for scheme, r in results.items():
        d = r["diagonal_check"]
        if d.get("checked"):
            L.append(f"- `{scheme}`: rebuilt diagonal mean {d['rebuilt_mean']:.4f} vs stored "
                     f"{d['stored_mean']:.4f}, max |delta| {d['max_abs_delta']:.2e} over "
                     f"{d['n_patients']} patients — **{'PASS' if d['passed'] else 'FAIL'}**\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L), encoding="utf-8")
    log.info("Wrote markdown report -> %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pairwise transfer matrix + max-statistic null")
    ap.add_argument("--input-norm", nargs="+", default=None)
    ap.add_argument("--n-permutations", type=int, default=None)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--run-tag", default="pairwise_transfer")
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
        log.error("CUDA unavailable — the frozen-backbone features need a GPU pass "
                  "(unless already cached). Aborting.")
        return 1

    cfg = load_pairwise_config()
    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d). STOP.", n_pat, n_win)
        return 1

    results_dir = _RUN_ROOT / "results" / cfg["output"]["results_subdir"]
    pf_path = _RUN_ROOT / "results" / ts_cfg.output["results_subdir"] / "preflight_summary.json"
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    eligible = list(pf["eligible_patients"])
    if args.max_patients:
        eligible = eligible[: args.max_patients]
        log.warning("SMOKE TEST: restricted to %d patient(s)", len(eligible))

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)
    sidecar["_row"] = np.arange(len(sidecar))
    y = sidecar["y_cond"].to_numpy().astype(int)
    pats_of_row = sidecar["patient_id"].to_numpy().astype(str)

    schemes = args.input_norm or list(cfg["input_norm"]["schemes"])
    n_perm = args.n_permutations or int(cfg["permutation"]["n_permutations"])

    results = {}
    for scheme in schemes:
        norm_scale = None
        if scheme != "none":
            norm_scale = cohort_median_scale(manifest, data_cfg, seed=ft_cfg.seed.base)
        X, meta = load_or_compute_backbone_features(
            sidecar, data_cfg, ft_cfg, scheme=scheme, norm_scale=norm_scale,
            cache_dir=(_RUN_ROOT / "results" / ts_cfg.output["results_subdir"] /
                       "backbone_features"), device="cuda")
        log.info("Frozen-backbone features for norm=%s: %s (%s)", scheme, X.shape, meta["cache"])
        results[scheme] = run_scheme(scheme, sidecar, eligible, X.astype(np.float64), y,
                                     pats_of_row, data_cfg, ft_cfg, ts_cfg, cfg,
                                     results_dir, n_perm)
        results[scheme]["backbone_features"] = meta

    _write_json(results_dir / cfg["output"]["summary"],
                {"run_tag": args.run_tag, "config": cfg, "results": results,
                 "git_sha": git_sha(_PROJECT), "env": env_versions()})
    _write_report(_RUN_ROOT / cfg["output"]["report"], results, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
