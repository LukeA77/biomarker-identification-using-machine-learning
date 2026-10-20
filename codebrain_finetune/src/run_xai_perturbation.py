"""XAI Tier C — causal perturbation of a saved checkpoint (XAI_PLAN.md 4, Tier C).

Measures what the trained model actually DEPENDS ON by breaking the input and
re-scoring, rather than by trusting a gradient. This is the strongest evidence in
the plan and should carry any claim we make; gradient saliency (Tier B) is
corroboration, not the headline.

Perturbations
  channel   — zero, or substitute that channel from a different window of the SAME
              session (preserves the channel's marginal statistics and destroys
              only its pairing with the rest of the window; a stricter control
              than zeroing, which also removes signal power).
  patch     — zero each of the 10 one-second patches in turn. Mostly a control:
              near-uniform is expected, a strong position effect would mean the
              model reads within-window position.
  bandstop  — notch 2 Hz bands out of the raw signal, then re-apply chan_z exactly
              as the loader does. Filtering must happen BEFORE normalisation,
              because removing a band changes each channel's std.
  phaserand — randomise Fourier phase, preserve the amplitude spectrum. If accuracy
              survives, the model uses only the spectrum; if it collapses, it uses
              waveform shape / phase structure.

FREQUENCY CAVEAT (XAI_PLAN.md 9.1): band edges are a post-hoc interpretive lens we
impose. The model never saw a band definition during training or inference. Say so
in every caption that carries a frequency axis.

SCOPE CAVEAT: this model scores 0.832 here but 0.435 leak-free, with recid=0.920 on
the same windows. A channel or band that matters here is a candidate for Tier E,
not a biomarker.

Usage:
    python codebrain_finetune/src/run_xai_perturbation.py \
        --checkpoint checkpoints/xai/xai_seed1__toast_to_tea__full_ft__seed1.pt
"""
from __future__ import annotations

import argparse
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

from livinglab_codebrain import load_data_config  # noqa: E402
from codebrain_ft import load_finetune_config, seed_everything  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import make_frame_loader  # noqa: E402
from codebrain_ft.metrics import classification_metrics  # noqa: E402
from codebrain_ft.xai import load_xai_checkpoint  # noqa: E402
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402
from run_xai_attribution import build_test_frame, CHANNELS, _json_default  # noqa: E402

log = logging.getLogger("xai_perturbation")
SFREQ = 200.0


def chan_z(x: np.ndarray, target: float) -> np.ndarray:
    """Replicate NormalizedWindowDataset's chan_z exactly, on (N, C, P, S)."""
    n, c = x.shape[0], x.shape[1]
    flat = x.reshape(n, c, -1)
    mu = flat.mean(axis=2, keepdims=True)
    sd = np.clip(flat.std(axis=2, keepdims=True), 1e-8, None)
    return ((flat - mu) / sd * target).reshape(x.shape)


def patient_mean_ba(prob, y, pids, threshold=0.5) -> float:
    """Patient-mean balanced accuracy — the headline metric, so perturbations move
    the same number the write-up quotes."""
    vals = [classification_metrics(y[pids == p], prob[pids == p],
                                   threshold=threshold).balanced_accuracy
            for p in np.unique(pids)]
    return float(np.mean(vals))


def score(model, x_norm, y, pids, *, device, batch_size=64) -> float:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_norm), batch_size):
            xb = torch.from_numpy(x_norm[i:i + batch_size]).float().to(device)
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return patient_mean_ba(np.concatenate(out), y, pids)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier C causal perturbation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--perturbations", nargs="+",
                    default=["channel", "patch", "bandstop", "phaserand"],
                    choices=["channel", "patch", "bandstop", "phaserand"])
    ap.add_argument("--band-width", type=float, default=2.0)
    ap.add_argument("--band-max", type=float, default=45.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — SSSM requires GPU. Aborting.")
        return 1
    device = "cuda"

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _RUN_ROOT / ckpt

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    ts_cfg = load_tasksplit_config()
    seed_everything(ft_cfg.seed.base, deterministic=False)

    model, payload = load_xai_checkpoint(ckpt, data_cfg, ft_cfg, device=device)
    target = float(payload["norm_scale"])
    tag = args.out_tag or ckpt.stem
    out_dir = _RUN_ROOT / "results" / "xai" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    frame = build_test_frame(payload, sidecar, ts_cfg)

    # RAW windows (input_norm='none'); we normalise ourselves so band-stop can be
    # applied before normalisation.
    loader = make_frame_loader(frame, data_cfg, shuffle=False, seed=1, input_norm="none")
    xs, ys, pids, sess = [], [], [], []
    for x, y, info in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
        pids.extend(info["patient_id"])
        sess.extend(f"{p}-{c}" for p, c in zip(info["patient_id"], info["condition"]))
    Xraw = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    pids = np.asarray(pids)
    sess = np.asarray(sess)
    log.info("Windows %s | %d patients", Xraw.shape, len(np.unique(pids)))

    Xn = chan_z(Xraw, target)
    base_ba = score(model, Xn, y, pids, device=device, batch_size=args.batch_size)
    log.info("BASELINE patient-mean BA = %.4f (checkpoint reported on this set)", base_ba)

    rec = {
        "checkpoint": str(ckpt.relative_to(_RUN_ROOT)), "direction": payload["direction"],
        "seed_base": payload["seed_base"], "baseline_ba": base_ba,
        "n_windows": int(len(Xn)), "n_patients": int(len(np.unique(pids))),
        "channels": CHANNELS, "git_sha": git_sha(_PROJECT), "env": env_versions(),
        "frequency_caveat": ("Band edges are a post-hoc interpretive lens; the model "
                             "never saw a band definition (XAI_PLAN.md 9.1)."),
        "scope_caveat": ("0.832 here vs 0.435 leak-free; recid=0.920. Findings are "
                         "Tier E candidates, not biomarkers."),
    }
    t0 = time.time()
    rng = np.random.default_rng(1)

    if "channel" in args.perturbations:
        rec["channel_zero"], rec["channel_shuffle"] = {}, {}
        # within-session donor index, so a substituted channel keeps its session's stats
        donor = np.arange(len(Xn))
        for s in np.unique(sess):
            idx = np.where(sess == s)[0]
            donor[idx] = rng.permutation(idx)
        for ci, name in enumerate(CHANNELS):
            Z = Xn.copy(); Z[:, ci] = 0.0
            rec["channel_zero"][name] = base_ba - score(model, Z, y, pids, device=device,
                                                        batch_size=args.batch_size)
            S = Xn.copy(); S[:, ci] = Xn[donor][:, ci]
            rec["channel_shuffle"][name] = base_ba - score(model, S, y, pids, device=device,
                                                           batch_size=args.batch_size)
            log.info("  channel %-4s dBA zero %+.4f | shuffle %+.4f",
                     name, rec["channel_zero"][name], rec["channel_shuffle"][name])

    if "patch" in args.perturbations:
        rec["patch_zero"] = {}
        for p in range(Xn.shape[2]):
            Z = Xn.copy(); Z[:, :, p] = 0.0
            rec["patch_zero"][str(p)] = base_ba - score(model, Z, y, pids, device=device,
                                                        batch_size=args.batch_size)
        log.info("  patch dBA: %s", {k: round(v, 4) for k, v in rec["patch_zero"].items()})

    if "bandstop" in args.perturbations:
        from mne.filter import filter_data
        rec["bandstop"] = {}
        n, c, p, s = Xraw.shape
        flat = Xraw.reshape(n * c, p * s).astype(np.float64)  # contiguous 10 s per row
        edges = np.arange(1.0, args.band_max, args.band_width)
        for lo in edges:
            hi = lo + args.band_width
            # band-STOP: MNE treats l_freq > h_freq as a stop band. The explicit
            # transition widths and filter_length are REQUIRED — the defaults ask
            # for a transition band the default length cannot realise and raise.
            # Verified on a 5 Hz + 20 Hz synthetic: stopband -40 dB, passband
            # unchanged, and all 22 bands of the 1-45 Hz sweep construct cleanly.
            filt = filter_data(flat, SFREQ, l_freq=hi, h_freq=lo, method="fir",
                               phase="zero", l_trans_bandwidth=1.0,
                               h_trans_bandwidth=1.0, filter_length="4s",
                               verbose="error")
            Z = chan_z(filt.reshape(n, c, p, s).astype(np.float32), target)
            d = base_ba - score(model, Z, y, pids, device=device, batch_size=args.batch_size)
            rec["bandstop"][f"{lo:.0f}-{hi:.0f}"] = d
            log.info("  bandstop %4.0f-%-4.0f Hz  dBA %+.4f", lo, hi, d)

    if "phaserand" in args.perturbations:
        spec = np.fft.rfft(Xraw.reshape(len(Xraw), 18, -1), axis=-1)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, spec.shape))
        pr = np.fft.irfft(np.abs(spec) * ph, n=Xraw.shape[2] * Xraw.shape[3], axis=-1)
        Z = chan_z(pr.reshape(Xraw.shape).astype(np.float32), target)
        rec["phase_randomised_ba"] = score(model, Z, y, pids, device=device,
                                           batch_size=args.batch_size)
        rec["phase_randomised_dBA"] = base_ba - rec["phase_randomised_ba"]
        log.info("  phase-randomised BA %.4f (dBA %+.4f)",
                 rec["phase_randomised_ba"], rec["phase_randomised_dBA"])

    (out_dir / "perturbation_summary.json").write_text(
        json.dumps(rec, indent=2, default=_json_default), encoding="utf-8")
    log.info("Wrote %s (%.1f s)", out_dir / "perturbation_summary.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
