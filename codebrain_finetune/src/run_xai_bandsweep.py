"""XAI Tier C-2 — MULTI-WIDTH band-stop sweep (supersedes the 2 Hz sweep).

Why this exists
---------------
`run_xai_perturbation.py --perturbations bandstop` notches **2 Hz** bands across
1-45 Hz and concluded "no band is necessary" (mean |dBA| 0.002, largest 0.018 at
5-7 Hz). That conclusion is width-limited, not general: a 2 Hz notch removes ~4%
of the 1-45 Hz power, so a model reading a BROAD spectral feature — which is
exactly what the classical result is (the low-frequency slowing candidate:
a ~2-6 Hz *relative* power shift, a smooth profile with no peak) — routes around
it trivially. A negative result at 2 Hz says nothing about 12 Hz.

This script sweeps band **width from 5 to 18 Hz** and band **position across
1-45 Hz**, on the SAME checkpoints, the SAME 1402-window test frame and the SAME
filter construction, so the two sweeps are directly comparable and the 2 Hz result
becomes the narrow end of one continuum rather than a separate claim.

The width confound, and how it is handled
-----------------------------------------
Wider bands remove more signal, so raw dBA MUST rise with width for reasons that
have nothing to do with what the model learned. Three things keep that from being
mistaken for a finding, all computed here and consumed by `summarize_bandsweep.py`:

1. **Within-width reference.** Every band is scored against the spread of dBA
   across all other positions AT ITS OWN WIDTH. "Is 4-16 Hz special?" only ever
   means "special compared with other 12 Hz bands", never "compared with 2 Hz
   bands".
2. **Power-removal covariate.** `power_removed_frac` is the fraction of 1-45 Hz
   power the notch actually removes, measured per band from the cohort PSD. A
   low-frequency band removes far more power than a high-frequency band of equal
   Hz-width, so width alone does not control for it. The summariser regresses dBA
   on this and reports residuals — a band that hurts MORE than its power removal
   predicts is the only kind of candidate worth reporting.
3. **Per-patient BAs are persisted** (`per_patient_ba` in the npz) so paired
   patient-level tests and CIs can be run post hoc without re-running the sweep.

Cost trick
----------
All five XAI checkpoints were verified to share a byte-identical test frame (same
27 eligible patients, same 1402 windows; `duration_match(where="uniform")` is
deterministic). Only `norm_scale` differs per seed. So each band is filtered ONCE
on the CPU (the expensive step, ~3 s) and scored against all five models, which is
a 5x saving over running the sweep per checkpoint.

FREQUENCY CAVEAT (XAI_PLAN.md 9.1): band edges are a post-hoc interpretive lens we
impose. The model never saw a band definition during training or inference. Say so
in every caption that carries a frequency axis.

SCOPE CAVEAT: this model scores 0.832 here but 0.435 leak-free, with recid=0.920 on
the same windows. A band that matters here is a Tier E candidate, not a biomarker.

BAND-FISHING CAVEAT: with 469 bands, the largest dBA is a maximum over 469 draws.
Never quote the argmax band edges as "the band". Report the profile.

Usage:
    python codebrain_finetune/src/run_xai_bandsweep.py --preflight   # validate filters only
    python codebrain_finetune/src/run_xai_bandsweep.py               # full sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# 469 bands x 5 seeds x 11 batches emits this once per forward pass and buries the
# sweep log. It is a torch deprecation notice about dropout2d on a 3D input, which
# is inert here: every model is in eval() and the sweep never trains.
warnings.filterwarnings("ignore", message=".*dropout2d: Received a 3D input.*")

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config  # noqa: E402
from codebrain_ft import load_finetune_config, seed_everything  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import make_frame_loader  # noqa: E402
from codebrain_ft.metrics import classification_metrics  # noqa: E402
from codebrain_ft.xai import load_xai_checkpoint  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402
from run_xai_attribution import build_test_frame, CHANNELS, _json_default  # noqa: E402

log = logging.getLogger("xai_bandsweep")
SFREQ = 200.0

# Filter construction — IDENTICAL to run_xai_perturbation.py's bandstop, so the
# 2 Hz sweep and this one differ only in band width. The explicit transition
# widths and filter_length are REQUIRED: MNE's defaults ask for a transition band
# the default length cannot realise and raise.
TRANS_BW = 1.0
FILTER_LEN = "4s"


def bandstop(flat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Notch [lo, hi] out of (n_rows, n_samples). MNE reads l_freq > h_freq as a stop band."""
    from mne.filter import filter_data
    return filter_data(flat, SFREQ, l_freq=hi, h_freq=lo, method="fir", phase="zero",
                       l_trans_bandwidth=TRANS_BW, h_trans_bandwidth=TRANS_BW,
                       filter_length=FILTER_LEN, verbose="error")


def chan_z_unit(x: np.ndarray) -> np.ndarray:
    """NormalizedWindowDataset's chan_z at unit scale, on (N, C, P, S).

    Returned at target=1 so each seed's own `norm_scale` can be applied as a
    scalar at scoring time instead of materialising five copies of a 200 MB array.
    """
    n, c = x.shape[0], x.shape[1]
    flat = x.reshape(n, c, -1)
    mu = flat.mean(axis=2, keepdims=True)
    sd = np.clip(flat.std(axis=2, keepdims=True), 1e-8, None)
    return ((flat - mu) / sd).reshape(x.shape)


def per_patient_ba(prob, y, pids, patients, threshold=0.5) -> np.ndarray:
    """Balanced accuracy per patient, in a FIXED patient order so bands are comparable."""
    return np.array([
        classification_metrics(y[pids == p], prob[pids == p], threshold=threshold).balanced_accuracy
        for p in patients], dtype=np.float64)


def score(model, z_unit, scale, y, pids, patients, *, device, batch_size=128) -> np.ndarray:
    """Per-patient BA of one model on unit-scaled input rescaled by that seed's norm_scale."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(z_unit), batch_size):
            xb = torch.from_numpy(z_unit[i:i + batch_size]).float().to(device) * scale
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return per_patient_ba(np.concatenate(out), y, pids, patients)


def build_grid(widths, fmin, fmax, stride) -> list[tuple[float, float]]:
    """(lo, hi) pairs: every width, every position that fits inside [fmin, fmax]."""
    grid = []
    for w in widths:
        lo = fmin
        while lo + w <= fmax + 1e-9:
            grid.append((round(float(lo), 3), round(float(lo + w), 3)))
            lo += stride
    return grid


def preflight(grid) -> int:
    """Construct every filter on a synthetic and assert stopband kill / passband keep.

    Cheap insurance: 469 MNE filter calls is 469 chances for a length/transition
    combination that raises or silently under-attenuates. A sweep that runs for an
    hour and then fails on band 400 is a wasted hour, and worse, a sweep whose
    band 400 quietly did nothing is a wrong result.
    """
    t = np.arange(0, 10.0, 1.0 / SFREQ)
    bad = []
    for lo, hi in grid:
        f_in = (lo + hi) / 2.0                      # dead centre of the stop band
        f_out = 47.0 if hi < 45.5 else 0.2          # well outside it
        sig = np.stack([np.sin(2 * np.pi * f_in * t), np.sin(2 * np.pi * f_out * t)])
        try:
            filt = bandstop(sig.astype(np.float64), lo, hi)
        except Exception as exc:                     # noqa: BLE001 — report, don't abort
            bad.append((lo, hi, f"raised {type(exc).__name__}: {exc}"))
            continue
        # Ignore filter-length edge transients when measuring attenuation.
        keep = slice(500, -500)
        att = filt[0, keep].std() / max(sig[0, keep].std(), 1e-12)
        thru = filt[1, keep].std() / max(sig[1, keep].std(), 1e-12)
        if att > 0.10:
            bad.append((lo, hi, f"stopband only {20 * np.log10(max(att, 1e-12)):.1f} dB"))
        elif thru < 0.70:
            bad.append((lo, hi, f"passband lost {100 * (1 - thru):.0f}% at {f_out} Hz"))
    print(f"preflight: {len(grid)} bands, {len(bad)} problems")
    for lo, hi, why in bad[:40]:
        print(f"  {lo:5.1f}-{hi:<5.1f} {why}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-width band-stop sweep (Tier C-2)")
    ap.add_argument("--checkpoint-glob",
                    default="checkpoints/xai/xai_seed*__toast_to_tea__full_ft__seed*.pt",
                    help="ALL matched checkpoints are scored per band; they share a test frame")
    ap.add_argument("--width-min", type=float, default=5.0)
    ap.add_argument("--width-max", type=float, default=18.0)
    ap.add_argument("--width-step", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=1.0, help="band-position step in Hz")
    ap.add_argument("--fmin", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=45.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--preflight", action="store_true", help="validate filters and exit")
    ap.add_argument("--out-tag", default="bandsweep_multiwidth")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    widths = np.arange(args.width_min, args.width_max + 1e-9, args.width_step)
    grid = build_grid(widths, args.fmin, args.fmax, args.stride)
    log.info("Grid: %d bands | widths %g-%g Hz step %g | positions %g-%g Hz stride %g",
             len(grid), args.width_min, args.width_max, args.width_step,
             args.fmin, args.fmax, args.stride)
    if args.preflight:
        return preflight(grid)

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — SSSM requires GPU. Aborting.")
        return 1
    device = "cuda"

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    seed_everything(ft_cfg.seed.base, deterministic=False)

    ckpts = sorted((_RUN_ROOT).glob(args.checkpoint_glob))
    if not ckpts:
        log.error("No checkpoints matched %s", args.checkpoint_glob)
        return 1

    # ---- load every model; assert they really do share one test frame ----------
    models, seeds, scales, frame, payload0 = [], [], [], None, None
    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    for ck in ckpts:
        model, payload = load_xai_checkpoint(ck, data_cfg, ft_cfg, device=device)
        f = build_test_frame(payload, sidecar, ts_cfg)
        if frame is None:
            frame, payload0 = f, payload
        elif not frame["path"].equals(f["path"]):
            # The whole filter-once-score-five design rests on this. If it ever
            # stops holding, fail loudly rather than compare bands across sets.
            log.error("%s has a DIFFERENT test frame — cannot share filtered data", ck.name)
            return 1
        models.append(model)
        seeds.append(int(payload["seed_base"]))
        scales.append(float(payload["norm_scale"]))
    log.info("Loaded %d checkpoints (seeds %s), shared test frame of %d windows",
             len(models), seeds, len(frame))

    # ---- raw windows (input_norm='none'): band-stop must precede normalisation --
    loader = make_frame_loader(frame, data_cfg, shuffle=False, seed=1, input_norm="none")
    xs, ys, pids = [], [], []
    for x, y, info in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
        pids.extend(info["patient_id"])
    Xraw = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    pids = np.asarray(pids)
    patients = np.unique(pids)                      # fixed order for every band
    n, c, p, s = Xraw.shape
    log.info("Windows %s | %d patients", Xraw.shape, len(patients))

    flat = Xraw.reshape(n * c, p * s).astype(np.float64)   # contiguous 10 s per row

    # ---- power-removal covariate, from the cohort PSD of the UNFILTERED data ----
    # Analytic and exact, so it costs one Welch instead of one extra filter per band.
    from scipy.signal import welch
    fx, px = welch(flat, fs=SFREQ, nperseg=int(2 * SFREQ), axis=-1)
    psd = px.mean(axis=0)
    inband = (fx >= args.fmin) & (fx <= args.fmax)
    total_power = float(psd[inband].sum())

    # ---- baselines -------------------------------------------------------------
    Z0 = chan_z_unit(Xraw).astype(np.float32)
    base_pp = np.stack([score(m, Z0, sc, y, pids, patients, device=device,
                              batch_size=args.batch_size)
                        for m, sc in zip(models, scales)])          # (n_seed, n_pat)
    base_ba = base_pp.mean(axis=1)
    for sd, b in zip(seeds, base_ba):
        log.info("BASELINE seed %d patient-mean BA = %.4f", sd, b)
    del Z0

    # ---- sweep -----------------------------------------------------------------
    pp = np.full((len(grid), len(models), len(patients)), np.nan)   # per-patient BA
    removed = np.zeros(len(grid))
    var_ratio = np.zeros(len(grid))
    t0 = time.time()
    for bi, (lo, hi) in enumerate(grid):
        filt = bandstop(flat, lo, hi)
        Z = chan_z_unit(filt.reshape(n, c, p, s).astype(np.float32)).astype(np.float32)
        for si, (m, sc) in enumerate(zip(models, scales)):
            pp[bi, si] = score(m, Z, sc, y, pids, patients, device=device,
                               batch_size=args.batch_size)
        band = (fx >= lo) & (fx < hi)
        removed[bi] = float(psd[band].sum()) / total_power
        var_ratio[bi] = 1.0 - float(filt.var()) / float(flat.var())
        d = base_ba - pp[bi].mean(axis=1)
        eta = (time.time() - t0) / (bi + 1) * (len(grid) - bi - 1)
        log.info("[%3d/%3d] %5.1f-%-5.1f Hz (w=%2.0f) removed %4.1f%% | dBA %+.4f "
                 "(seeds %s) | eta %.0f min", bi + 1, len(grid), lo, hi, hi - lo,
                 100 * removed[bi], d.mean(),
                 " ".join(f"{v:+.3f}" for v in d), eta / 60)
        del filt, Z

    lo_arr = np.array([g[0] for g in grid])
    hi_arr = np.array([g[1] for g in grid])
    out_dir = _RUN_ROOT / "results" / "xai" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "bandsweep.npz",
        lo=lo_arr, hi=hi_arr, width=hi_arr - lo_arr,
        per_patient_ba=pp, baseline_per_patient_ba=base_pp,
        seeds=np.array(seeds), patients=patients.astype(str),
        power_removed_frac=removed, var_removed_frac=var_ratio,
        psd_freqs=fx, psd=psd,
    )
    meta = {
        "checkpoints": [str(c.relative_to(_RUN_ROOT)) for c in ckpts],
        "experiment": "pooled_tasksplit_chanz_matched",
        "direction": payload0["direction"], "seeds": seeds, "norm_scales": scales,
        "n_bands": len(grid), "n_windows": int(n), "n_patients": int(len(patients)),
        "widths": [float(w) for w in widths], "stride": args.stride,
        "fmin": args.fmin, "fmax": args.fmax,
        "filter": {"method": "fir", "phase": "zero", "trans_bw": TRANS_BW,
                   "filter_length": FILTER_LEN, "note": "identical to the 2 Hz sweep"},
        "baseline_ba_per_seed": {str(sd): float(b) for sd, b in zip(seeds, base_ba)},
        "channels": CHANNELS, "git_sha": git_sha(_PROJECT), "env": env_versions(),
        "runtime_s": time.time() - t0,
        "frequency_caveat": ("Band edges are a post-hoc interpretive lens; the model "
                             "never saw a band definition (XAI_PLAN.md 9.1)."),
        "width_caveat": ("dBA rises with width for trivial reasons. Compare bands only "
                         "WITHIN a width, and against power_removed_frac."),
        "fishing_caveat": (f"{len(grid)} bands were tested; the largest dBA is a maximum "
                           "over that many draws. Report the profile, not the argmax."),
        "scope_caveat": ("0.832 here vs 0.435 leak-free; recid=0.920. Findings are "
                         "Tier E candidates, not biomarkers."),
    }
    (out_dir / "bandsweep_meta.json").write_text(
        json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    log.info("Wrote %s (%.1f min)", out_dir, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
