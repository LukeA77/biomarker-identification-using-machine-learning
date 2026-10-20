"""Input-representation experiment — can CodeBrain be made to see the signal?

Motivated directly by the Tier-1 finding, not by guesswork:

  * RELATIVE band power decodes ON/OFF (bal-acc 0.643-0.657, perm p=0.005).
  * ABSOLUTE log band power is at chance (0.514).

CodeBrain currently receives raw microvolts / 100. So the discriminative structure
(spectral *shape*) is buried under between-subject amplitude variance (electrode
impedance, skull thickness, contact quality) -- exactly the nuisance dimension
relative power cancels. The measured LoRA predictions agreed: between-subject
spread in P(OFF) was 2.1x the within-subject ON/OFF effect.

So: normalise each window's amplitude before the backbone, and see whether the
frozen representation moves off chance. This is the CHEAP test (feature extraction
+ logistic regression, minutes) that decides whether the expensive 13 h LoRA rerun
is justified -- the same prototype-before-scaling discipline used throughout.

Schemes (all rescaled to the cohort-median window std so the backbone keeps its
pretrained input magnitude -- z-scoring to unit std alone would inflate inputs ~4x
and push the model off its operating point):
  none      -- baseline, microvolts/100 exactly as the completed runs used
  global_z  -- per-window z over all channels (removes overall amplitude, KEEPS
               relative channel amplitudes / topography)
  chan_z    -- per-window per-channel z (removes each channel's amplitude too;
               strongest normalisation, discards topographic amplitude)

Also fits HYBRID models (CodeBrain features + classical band power) to test
whether the representations carry complementary information.

Run:  python codebrain_finetune/src/stage5b_input_norm.py
"""
from __future__ import annotations

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

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from livinglab_codebrain import load_data_config, CohortManifest, build_lopo_folds  # noqa: E402
from livinglab_codebrain.dataset import WindowDataset  # noqa: E402
from livinglab_codebrain.loader import make_dataloader  # noqa: E402
from codebrain_ft import load_finetune_config, build_classifier_model, seed_everything  # noqa: E402
from codebrain_ft.probe import extract_features  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics,
    bootstrap_session_ci, permutation_test_balanced_accuracy,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage5b")
EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174
SCHEMES = ["none", "global_z", "chan_z"]


class NormalizedWindowDataset(Dataset):
    """Wraps the audited ``WindowDataset`` and normalises each window's amplitude.

    Deliberately a wrapper, not an edit: the audited loader package, the window
    cache, and the manifest are all untouched. Only what the model *sees* changes.
    """

    def __init__(self, base: WindowDataset, scheme: str, target_scale: float) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}")
        self.base, self.scheme, self.target = base, scheme, target_scale

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, info = self.base[i]                 # (C, P, S) float32, already /100
        if self.scheme != "none":
            c = x.shape[0]
            flat = x.reshape(c, -1)               # (C, P*S) = (18, 2000)
            if self.scheme == "global_z":
                mu, sd = flat.mean(), flat.std()
            else:                                  # chan_z
                mu = flat.mean(dim=1, keepdim=True)
                sd = flat.std(dim=1, keepdim=True)
            sd = torch.clamp(sd, min=1e-8)
            x = ((flat - mu) / sd * self.target).reshape(x.shape)
        return x, y, info


def _median_window_scale(base: WindowDataset, n: int = 300, seed: int = 1) -> float:
    """Cohort-median per-window std, so normalised inputs keep the pretrained magnitude."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=min(n, len(base)), replace=False)
    return float(np.median([float(base[i][0].std()) for i in idx]))


def lopo_feature_space(F, y, pids, conds, nonoverlap, folds, fc, tag):
    prob, true, pat = [], [], []
    for fold in folds:
        tr = pids != fold.test_patient
        te = (pids == fold.test_patient) & nonoverlap
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=fc.probe.C, max_iter=fc.probe.max_iter,
                                 class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        p = clf.predict_proba(sc.transform(F[te]))[:, 1]
        s = aggregate_session_level(pids[te], conds[te], p, y[te])
        prob += s.prob.tolist(); true += s.y_true.tolist(); pat += s.patients.tolist()
    prob, true, pat = np.array(prob), np.array(true), np.array(pat)
    agg = classification_metrics(true, prob, threshold=fc.eval.threshold)
    boot = bootstrap_session_ci(true, prob, pat, n_resamples=fc.eval.bootstrap.n_resamples,
                                ci=fc.eval.bootstrap.ci, seed=fc.seed.base)
    perm = permutation_test_balanced_accuracy(
        true, prob, pat, n_permutations=fc.eval.permutation.n_permutations,
        seed_base=fc.seed.base + fc.seed.permutation_offset)
    n = sum(prob[(pat == p) & (true == 1)][0] > prob[(pat == p) & (true == 0)][0]
            for p in np.unique(pat))
    log.info("  %-34s bal-acc %.3f  CI [%.3f, %.3f]  AUC %-6s p %-7.4f ord %2d/35", tag,
             agg.balanced_accuracy, boot["ci_low"] or np.nan, boot["ci_high"] or np.nan,
             f"{agg.auc:.3f}" if agg.auc is not None else "n/a", perm["p_value"], n)
    return {"tag": tag, "aggregate": agg.as_dict(), "bootstrap": boot,
            "permutation": perm, "ordering_correct": int(n)}


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — aborting."); return 1
    device = "cuda"
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)

    mf = CohortManifest.from_config(dc)
    if len(mf.patients) != EXPECTED_PATIENTS or mf.n_windows != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch — STOP."); return 1
    folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()
    base_ds = WindowDataset(df, dc.paths.cache_dir, dc.window)
    target = _median_window_scale(base_ds)
    log.info("Cohort-median window std = %.4f (normalised inputs rescaled to this)", target)

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    nonoverlap = df["is_nonoverlap_subset"].to_numpy().astype(bool)

    results = []
    feats = {}
    log.info("=" * 92)
    log.info("FROZEN CodeBrain probe under each input normalisation (mean pooling)")
    for scheme in SCHEMES:
        ds = NormalizedWindowDataset(base_ds, scheme, target)
        loader = make_dataloader(ds, dc, shuffle=False, seed=fc.seed.base)
        model = build_classifier_model(dc, fc, mode="frozen", device=device)
        t0 = time.time()
        X, yy, pp, cc = extract_features(model, loader, fc, device=device)
        log.info("  [%s] features %s in %.0fs", scheme, X.shape, time.time() - t0)
        del model; torch.cuda.empty_cache()
        assert np.array_equal(yy, y) and np.array_equal(pp, pids), "loader order drifted"
        feats[scheme] = X
        results.append(lopo_feature_space(X, y, pids, conds, nonoverlap, folds, fc,
                                          f"codebrain frozen [{scheme}]"))

    # ---- classical reference + hybrids ------------------------------------
    log.info("=" * 92)
    log.info("CLASSICAL reference and HYBRID (CodeBrain + band power)")
    B = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]     # (N,18,5,2)
    rel_nogamma = B[:, :, :4, 1].reshape(len(B), -1)                    # 72 feats, best classical
    results.append(lopo_feature_space(rel_nogamma, y, pids, conds, nonoverlap, folds, fc,
                                      "band power rel (no gamma)"))
    for scheme in SCHEMES:
        H = np.concatenate([StandardScaler().fit_transform(feats[scheme]),
                            StandardScaler().fit_transform(rel_nogamma)], axis=1)
        results.append(lopo_feature_space(H, y, pids, conds, nonoverlap, folds, fc,
                                          f"HYBRID bandpower + cb[{scheme}]"))

    out = {"run_tag": "stage5b_input_norm", "target_scale": target,
           "schemes": SCHEMES, "results": results,
           "reference": {"lora_raw": 0.557, "frozen_raw": 0.443, "bandpower_rel_nogamma": 0.657}}
    (_RUN_ROOT / "results" / "stage5b_input_norm.json").write_text(json.dumps(out, indent=2, default=str))

    log.info("=" * 92)
    log.info("SUMMARY (all pooled over 35 folds / 70 sessions)")
    for r in sorted(results, key=lambda r: -r["aggregate"]["balanced_accuracy"]):
        a, b, p = r["aggregate"], r["bootstrap"], r["permutation"]
        log.info("  %-34s %.3f  [%.3f, %.3f]  AUC %-6s p %.4f", r["tag"],
                 a["balanced_accuracy"], b["ci_low"] or np.nan, b["ci_high"] or np.nan,
                 f"{a['auc']:.3f}" if a["auc"] is not None else "n/a", p["p_value"])
    log.info("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
