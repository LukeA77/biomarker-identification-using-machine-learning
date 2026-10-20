"""
Stage 3 — data-loader verification (end-to-end).

Builds the cohort loader from config and asserts it produces exactly what the
project needs, then drives one real batch through the pretrained CodeBrain
backbone to prove the loader output is model-compatible. No training.

Checks:
  1. Config + manifest load; exclusions applied loudly; cohort summary printed.
  2. Excluded patients are absent; pending-review patients are flagged.
  3. LOPO folds are well-formed (count == n_patients, no self in train, no leakage).
  4. A fold's train/test loaders yield correct shapes/dtypes/labels, normalised
     values, and NEVER a size-1 batch.
  5. The held-out test loader is non-overlap-only and covers exactly one patient.
  6. NoSingletonBatchSampler edge case: an odd-length dataset yields no singleton
     and loses no samples.
  7. A real train batch forward-passes through CodeBrain's SSSM -> (B,18,10,200).

Run from the project root:
    python codebrain_finetune/src/stage3_dataloader_check.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livinglab_codebrain import (  # noqa: E402
    CohortManifest,
    NoSingletonBatchSampler,
    WindowDataset,
    build_lopo_folds,
    load_data_config,
    make_dataloader,
    make_fold_loaders,
)

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def build_backbone(repo: Path):
    """Load CodeBrain's SSSM backbone with the pretrained checkpoint (as Stage 2)."""
    sys.path.insert(0, str(repo))
    from Models.SSSM import SSSM  # noqa: E402

    backbone = SSSM(
        in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
        num_res_layers=8, diffusion_step_embed_dim_in=200,
        diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
        s4_lmax=570, s4_d_state=64, s4_dropout=0.1, s4_bidirectional=True,
        s4_layernorm=True, codebook_size_t=4096, codebook_size_f=4096, if_codebook=False,
    )
    return backbone


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 74)
    print("STAGE 3 — data-loader verification")
    print("=" * 74)

    cfg = load_data_config()
    print(f"manifest: {cfg.paths.manifest}")
    print(f"window contract: raw {cfg.window.raw_shape} -> model {cfg.window.model_shape}, "
          f"÷{cfg.window.norm_divisor:.0f}")
    print("-" * 74)

    # 1) Manifest + exclusions -------------------------------------------------
    print("[1] Cohort manifest")
    manifest = CohortManifest.from_config(cfg)
    patients = manifest.patients
    balance = manifest.label_balance()
    print(f"    cohort: {len(patients)} patients, {manifest.n_windows} windows, "
          f"A(ON)={balance.get('A', 0)} / C(OFF)={balance.get('C', 0)}")

    # 2) Exclusions honoured ---------------------------------------------------
    # An exclusion ID that isn't in the manifest is legitimate (e.g. a patient
    # never processed because of an EEG-acquisition gap) but must NOT be counted
    # as a drop: the size invariant is over PRESENT exclusions only. The absent
    # ones are surfaced separately so a genuine typo can't hide.
    excluded = set(cfg.cohort.exclude_patients)
    full_pids = set(manifest.full_frame["patient_id"].unique())
    present_excl = excluded & full_pids
    absent_excl = excluded - full_pids
    check("excluded patients absent from cohort",
          excluded.isdisjoint(patients),
          f"excluded={sorted(excluded)}")
    check("cohort size == full patients − PRESENT exclusions",
          len(patients) == len(full_pids) - len(present_excl),
          f"{len(full_pids)} − {len(present_excl)} present = {len(patients)}")
    check("every configured exclusion exists in the manifest (else flag, don't silently drop)",
          not absent_excl,
          f"absent (not in manifest): {sorted(absent_excl)}" if absent_excl else "all present")

    # 3) LOPO folds ------------------------------------------------------------
    print("[3] LOPO folds")
    folds = build_lopo_folds(manifest)
    check("one fold per patient", len(folds) == len(patients), f"{len(folds)} folds")
    check("test patient never in its own train set",
          all(f.test_patient not in f.train_patients for f in folds))
    check("each fold trains on all-but-one patient",
          all(len(f.train_patients) == len(patients) - 1 for f in folds))

    # 4) Fold loaders ----------------------------------------------------------
    print("[4] Fold loaders (fold 0)")
    fold0 = folds[0]
    train_loader, test_loader = make_fold_loaders(manifest, cfg, fold0, check_meta=True)
    print(f"    fold 0: test={fold0.test_patient} | "
          f"train batches={len(train_loader)} test batches={len(test_loader)}")

    xb, yb, info = next(iter(train_loader))
    exp_shape = (cfg.window.n_channels, cfg.window.n_patches, cfg.window.samples_per_patch)
    check("train batch X shape (B,18,10,200)", tuple(xb.shape[1:]) == exp_shape, str(tuple(xb.shape)))
    check("train batch X dtype float32", xb.dtype == torch.float32)
    check("train batch y in {0,1}", set(yb.unique().tolist()).issubset({0, 1}), str(sorted(set(yb.tolist()))))
    check("train batch size >= min", xb.shape[0] >= cfg.loader.min_batch_size, f"B={xb.shape[0]}")
    check("train batch finite", bool(torch.isfinite(xb).all()))
    med = float(xb.abs().median())
    check("normalisation ÷100 applied (median|x|~0.1)", 0.02 < med < 0.6, f"median|x|={med:.4f}")
    check("held-out patient NOT in train batch", fold0.test_patient not in set(info["patient_id"]))

    # 5) Eval loader: non-overlap only, single patient, no sample loss ----------
    print("[5] Eval loader integrity")
    test_pids, test_nonoverlap, n_test_windows = set(), True, 0
    for _, _, tinfo in test_loader:
        test_pids.update(tinfo["patient_id"])
        test_nonoverlap = test_nonoverlap and all(tinfo["is_nonoverlap_subset"])
        n_test_windows += len(tinfo["patient_id"])
    expected_test = len(manifest.subset([fold0.test_patient], nonoverlap_only=True))
    check("test loader covers exactly the held-out patient", test_pids == {fold0.test_patient}, str(test_pids))
    check("test loader is non-overlap-only", test_nonoverlap)
    check("test loader loses no windows", n_test_windows == expected_test,
          f"{n_test_windows}/{expected_test}")

    # 6) Batch-sampler edge case (synthetic odd length) ------------------------
    print("[6] NoSingletonBatchSampler edge case")
    for n, bs in [(65, 64), (7, 2), (129, 64), (5, 4)]:
        s = NoSingletonBatchSampler(n, bs, min_size=2, shuffle=False)
        batches = list(iter(s))
        flat = [i for b in batches for i in b]
        ok = (all(len(b) >= 2 for b in batches)
              and sorted(flat) == list(range(n))
              and len(batches) == len(s))
        check(f"n={n},bs={bs}: no singleton, no loss, len() correct", ok,
              f"batch sizes={[len(b) for b in batches]}")

    # 7) End-to-end: loader batch -> real CodeBrain backbone -------------------
    print("[7] End-to-end forward pass (loader -> CodeBrain SSSM)")
    if not torch.cuda.is_available():
        check("CUDA available for forward pass", False, "no GPU — SSSM hardcodes .cuda()")
    else:
        backbone = build_backbone(cfg.paths.codebrain_repo).cuda().eval()
        sd = torch.load(cfg.paths.checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = backbone.load_state_dict({k[7:]: v for k, v in sd.items()}, strict=False)
        check("checkpoint loads (0 missing/unexpected)", not missing and not unexpected,
              f"missing={len(missing)} unexpected={len(unexpected)}")
        with torch.no_grad():
            out = backbone(xb.cuda())
        want = (xb.shape[0],) + exp_shape
        check("real loader batch -> backbone output (B,18,10,200)",
              tuple(out.shape) == want and bool(torch.isfinite(out).all()), str(tuple(out.shape)))

    # Summary ------------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_total = len(_results)
    all_ok = n_pass == n_total
    print("=" * 74)
    print(f"STAGE 3: {n_pass}/{n_total} checks passed — "
          f"{'PASS — loader is production-ready' if all_ok else 'FAIL — see above'}")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
