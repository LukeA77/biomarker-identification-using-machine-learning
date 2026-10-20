"""POSITIVE CONTROL + layer sweep + channel-order sensitivity — the interpretability gate.

Everything concluded so far rests on an untested assumption: that information from
our data actually reaches CodeBrain's representation. A negative control (100
refits, null centred on 0.500) proves nothing leaks IN. It does not prove anything
gets THROUGH. Without a positive control we cannot distinguish

    "CodeBrain does not encode levodopa state"          (a scientific finding)
    "CodeBrain encodes nothing from our data at all"    (a silent pipeline failure)

and those license completely different conclusions.

Three questions, one extraction pass each:

1. POSITIVE CONTROL — is the representation alive?
   Decode SUBJECT IDENTITY (35-way) from the embeddings. Subject identity is known
   to be strongly present: we measured between-subject spread in P(OFF) at 2.1x the
   within-subject effect. If subject ID decodes, the representation carries our
   data. Scored as a RATIO against the same task run on raw band-power features,
   which is the ceiling the input data supports -- an absolute threshold would be
   arbitrary, but "CodeBrain 15% vs raw features 90%" is unambiguous.
   * Split: train on Condition A windows, test on Condition C windows. LOPO is
     impossible here (the held-out subject's class never appears in training), and
     A/C are different recording days, so this also partly controls for decoding a
     same-session impedance signature rather than neural content.
   * Caveat recorded: subject ID may still ride on amplitude/impedance rather than
     neural content. Necessary, not sufficient.

2. LAYER SWEEP — are we probing the wrong place?
   All prior probes used the FINAL backbone output, which is specialised to the
   masked-token pretraining objective. Middle layers are frequently more
   transferable. Hooks tap all 8 residual blocks plus the final output.

3. CHANNEL-ORDER SENSITIVITY — is the montage mapping wrong?
   VERIFIED ARCHITECTURALLY: PatchEmbedding.positional_encoding is a depthwise
   Conv2d with kernel_size=(19, 7) applied over the (channel, patch) grid -- a
   19-tap kernel spanning the CHANNEL axis -- and SSSM then flattens to
   'b p (c s)', so channel order also sets sequence adjacency. The model is
   therefore NOT permutation-invariant over channels.
   Context: CodeBrain's own TUAB downstream reshapes to (16, 10, 200), i.e. the
   authors feed a different channel count/order downstream than pretraining, so
   there is no canonical order to match. This is an empirical sensitivity test, not
   a bug hunt: re-extract under (a) our montage order, (b) the reported CodeBrain
   order, (c) a random permutation. If all three perform alike, spatial structure
   is unused and channel order is a non-issue.

Run:  python codebrain_finetune/src/stage7_positive_control.py
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
from codebrain_ft.metrics import aggregate_session_level, classification_metrics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score, top_k_accuracy_score, balanced_accuracy_score  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

log = logging.getLogger("stage7")
EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174

# Our cache's montage order (dataloader.yaml channels.keep), Pz dropped (recording ref).
OUR_ORDER = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
             "C3", "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
# CodeBrain's reported pretraining order, minus Pz which we do not record.
CODEBRAIN_ORDER = ["C3", "C4", "Cz", "F3", "F4", "Fp1", "Fp2", "F7", "F8", "Fz",
                   "O1", "O2", "P3", "P4", "T3", "T4", "T5", "T6"]

# Subject-ID PASS thresholds, fixed BEFORE running (chance = 1/35 = 2.9%).
PASS_TOP1, AMBIGUOUS_TOP1 = 0.40, 0.10
SUBJID_TRAIN_CAP = 12000     # cap training windows for the 35-way fit (speed)


class ChannelPermutedDataset(Dataset):
    """Reorders each window's channels. Wrapper only; cache/manifest untouched."""

    def __init__(self, base, perm: np.ndarray) -> None:
        self.base = base
        self.perm = torch.as_tensor(np.asarray(perm), dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, info = self.base[i]
        return x[self.perm], y, info


def extract_layerwise(model, loader, device="cuda"):
    """Mean-pooled features from each residual block + the final output.

    Returns ``{layer_name: (N, D)}`` plus aligned y/pid/cond arrays.
    """
    backbone = model.backbone
    blocks = backbone.residual_layer.residual_blocks
    taps: dict[str, list] = {f"block{i}": [] for i in range(len(blocks))}
    taps["final"] = []
    handles, buf = [], {}

    def mk_hook(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            # residual blocks emit (B, D, L); pool the sequence axis -> (B, D)
            buf[name] = o.detach().float().mean(dim=-1).cpu()
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(mk_hook(f"block{i}")))

    ys, pids, conds = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, info in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = model.backbone_features(x)          # (B, C, P, S)
            taps["final"].append(feats.float().mean(dim=(1, 2)).cpu())
            for k in list(buf):
                taps[k].append(buf.pop(k))
            ys.append(y.numpy()); pids.extend(info["patient_id"]); conds.extend(info["condition"])
    for h in handles:
        h.remove()
    out = {k: torch.cat(v).numpy() for k, v in taps.items() if v}
    return out, np.concatenate(ys), np.asarray(pids), np.asarray(conds)


def subject_id_decode(F, pids, conds, seed=1):
    """35-way subject ID: train on Condition A windows, test on Condition C."""
    tr, te = conds == "A", conds == "C"
    classes = np.unique(pids)
    idx = np.where(tr)[0]
    if len(idx) > SUBJID_TRAIN_CAP:                       # stratified cap for speed
        rng = np.random.default_rng(seed)
        keep = []
        per = max(1, SUBJID_TRAIN_CAP // len(classes))
        for c in classes:
            ci = idx[pids[idx] == c]
            keep.append(rng.choice(ci, size=min(per, len(ci)), replace=False))
        idx = np.concatenate(keep)
    sc = StandardScaler().fit(F[idx])
    clf = LogisticRegression(max_iter=400, C=1.0)   # multinomial is the lbfgs default
    clf.fit(sc.transform(F[idx]), pids[idx])
    proba = clf.predict_proba(sc.transform(F[te]))
    yhat = clf.classes_[proba.argmax(1)]
    top1 = balanced_accuracy_score(pids[te], yhat)
    top5 = top_k_accuracy_score(pids[te], proba, k=5, labels=clf.classes_)
    return float(top1), float(top5)


def onoff_per_subject_auc(F, y, pids, conds, folds, fc):
    """Per-subject WINDOW-level AUC (well-powered) + pooled session-level bal-acc."""
    per_subj, prob_all, true_all, pat_all = [], [], [], []
    for fold in folds:
        tr = pids != fold.test_patient
        te = pids == fold.test_patient
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=fc.probe.C, max_iter=fc.probe.max_iter,
                                 class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        p = clf.predict_proba(sc.transform(F[te]))[:, 1]
        yt = y[te]
        if len(np.unique(yt)) == 2:
            per_subj.append(roc_auc_score(yt, p))          # within-subject ranking only
        s = aggregate_session_level(pids[te], conds[te], p, yt)
        prob_all += s.prob.tolist(); true_all += s.y_true.tolist(); pat_all += s.patients.tolist()
    per_subj = np.array(per_subj)
    w = wilcoxon(per_subj - 0.5, alternative="greater") if len(per_subj) > 5 else None
    m = classification_metrics(np.array(true_all), np.array(prob_all), threshold=fc.eval.threshold)
    return {"per_subject_auc_median": float(np.median(per_subj)),
            "per_subject_auc_mean": float(per_subj.mean()),
            "n_above_half": int((per_subj > 0.5).sum()), "n_subjects": len(per_subj),
            "wilcoxon_p": float(w.pvalue) if w else None,
            "session_bal_acc": float(m.balanced_accuracy)}


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

    perms = {
        "our_order": np.arange(18),
        "codebrain_order": np.array([OUR_ORDER.index(c) for c in CODEBRAIN_ORDER]),
        "random_perm": np.random.default_rng(fc.seed.base).permutation(18),
    }
    log.info("Channel permutations: %s", {k: v.tolist() for k, v in perms.items()})

    results = {"pass_thresholds": {"top1_pass": PASS_TOP1, "ambiguous_top1": AMBIGUOUS_TOP1,
                                   "chance": 1 / 35}, "arms": {}}

    # ---- raw band-power reference ceiling (what the INPUT data supports) ----
    B = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]
    raw = B[:, :, :, 1].reshape(len(B), -1)
    y = df["y_cond"].to_numpy().astype(int)
    pids0 = df["patient_id"].to_numpy().astype(str)
    conds0 = df["condition"].to_numpy().astype(str)
    t0 = time.time()
    r1, r5 = subject_id_decode(raw, pids0, conds0)
    log.info("=" * 96)
    log.info("REFERENCE CEILING — raw band-power features (not CodeBrain)")
    log.info("  subject-ID  top1 %.3f   top5 %.3f   (chance %.3f)  [%.0fs]", r1, r5, 1 / 35, time.time() - t0)
    results["raw_bandpower_subject_id"] = {"top1": r1, "top5": r5}

    # ---- CodeBrain, per channel ordering ----
    for pname, perm in perms.items():
        ds = base_ds if pname == "our_order" else ChannelPermutedDataset(base_ds, perm)
        loader = make_dataloader(ds, dc, shuffle=False, seed=fc.seed.base)
        model = build_classifier_model(dc, fc, mode="frozen", device=device)
        t0 = time.time()
        feats, yy, pp, cc = extract_layerwise(model, loader, device=device)
        del model; torch.cuda.empty_cache()
        assert np.array_equal(yy, y) and np.array_equal(pp, pids0), "loader order drifted"
        log.info("=" * 96)
        log.info("CHANNEL ORDER = %s   (%d taps, %.0fs)", pname, len(feats), time.time() - t0)
        log.info("  %-8s %-22s %-22s %s", "layer", "subjID top1/top5", "ON/OFF per-subj AUC", "session bal-acc")
        arm = {}
        for lname in sorted(feats, key=lambda s: (s != "final", s)):
            F = feats[lname]
            s1, s5 = subject_id_decode(F, pp, cc)
            oo = onoff_per_subject_auc(F, y, pp, cc, folds, fc)
            arm[lname] = {"subject_id_top1": s1, "subject_id_top5": s5, **oo}
            log.info("  %-8s %.3f / %.3f          %.3f (%2d/%d >0.5, p=%s)   %.3f",
                     lname, s1, s5, oo["per_subject_auc_median"], oo["n_above_half"],
                     oo["n_subjects"],
                     f"{oo['wilcoxon_p']:.3f}" if oo["wilcoxon_p"] is not None else "n/a",
                     oo["session_bal_acc"])
        results["arms"][pname] = arm

    # ---- verdict ----
    best = max(results["arms"]["our_order"].items(), key=lambda kv: kv[1]["subject_id_top1"])
    top1 = best[1]["subject_id_top1"]
    verdict = ("PASS — representation is alive" if top1 > PASS_TOP1 else
               "AMBIGUOUS — triage needed" if top1 > AMBIGUOUS_TOP1 else
               "FAIL — probable pipeline failure")
    log.info("=" * 96)
    log.info("POSITIVE-CONTROL VERDICT: best layer %s, subject-ID top1 %.3f -> %s",
             best[0], top1, verdict)
    log.info("  raw band-power ceiling was %.3f; ratio CodeBrain/raw = %.2f",
             results["raw_bandpower_subject_id"]["top1"],
             top1 / max(results["raw_bandpower_subject_id"]["top1"], 1e-9))
    results["verdict"] = {"best_layer": best[0], "top1": top1, "verdict": verdict}

    (_RUN_ROOT / "results" / "stage7_positive_control.json").write_text(
        json.dumps(results, indent=2, default=str))
    log.info("Saved -> results/stage7_positive_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
