"""XAI primitives for the pooled `chan_z` full fine-tune (XAI_PLAN.md).

Scope note. The model being explained scores 0.832 on the pooled task split but
0.435 under the leak-free cross-subject design, and `recid` ("which recording is
this?") sits at 0.920 on the same windows. So every map produced here describes a
decision function KNOWN to be substantially a recording fingerprint. Nothing in
this module, on its own, licenses a physiological claim — that requires Tier E
(independent leak-free re-test of an isolated candidate). Keep that in the
docstrings and the figure captions, not just in someone's memory.

Architecture facts that drive the method choices (verified against model.py):

* ``CodeBrainClassifier`` head is ``pool_mlp``: backbone -> (B, 18, 10, 200) ->
  **mean over channels AND patches** -> (B, 200) -> Linear(200,64) -> ELU ->
  Dropout -> Linear(64,1). The head is therefore channel- and time-blind: no
  spatial selectivity can be read off it, and all of it must come from input-space
  attribution or occlusion.
* The surviving 200-d axis is ``samples_per_patch`` at 200 Hz = **one second**.
  So ``head[0].weight`` is literally a bank of 64 FIR filters of length 200 and
  its frequency response is exactly computable — no attribution approximation.
  That is :func:`head_filter_response` (Tier A1).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = [
    "head_filter_response", "effective_input_filter",
    "integrated_gradients", "occlude_channels",
    "load_xai_checkpoint",
]

SFREQ = 200.0


# ---------------------------------------------------------------------------
# Tier A — exact readouts from weights
# ---------------------------------------------------------------------------
def head_filter_response(model: nn.Module, *, n_fft: int = 512) -> dict:
    """Frequency response of the head's first layer (Tier A1). Exact, no gradients.

    ``head[0]`` maps the 200-sample (1 s @ 200 Hz) pooled axis to 64 hidden units,
    so each of its 64 rows is an FIR filter over time. Returns the magnitude
    response of every row plus their mean, on a real-FFT frequency grid.

    Interpreting the frequency axis is a POST-HOC lens we impose; the model never
    saw band definitions (XAI_PLAN.md 9.1). Say so wherever this is plotted.
    """
    w = model.head[0].weight.detach().cpu().float().numpy()   # (64, 200)
    if w.shape[1] != 200:
        raise ValueError(
            f"expected a 200-sample head input axis, got {w.shape[1]}; this readout "
            f"is only valid for the pool_mlp head (see module docstring)")
    spec = np.abs(np.fft.rfft(w, n=n_fft, axis=1))            # (64, n_fft//2+1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SFREQ)
    return {
        "freqs": freqs,
        "per_unit": spec,
        "mean": spec.mean(axis=0),
        "sd": spec.std(axis=0),
        "weights": w,
    }


def effective_input_filter(model: nn.Module, loader, *, device: str = "cuda",
                           max_batches: int | None = None) -> dict:
    """First-order effective filter of the WHOLE model on the pooled 1 s axis (A2).

    Mean input-gradient of the logit w.r.t. the pooled 200-d representation over
    the evaluation set. Because ELU is smooth this is a first-order linear
    approximation of the end-to-end response; comparing its spectrum against
    :func:`head_filter_response` localises how much the backbone reshapes things.
    """
    model.eval()
    acc, n = None, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch[0].to(device).float().requires_grad_(True)
        feats = model.backbone_features(x).mean(dim=(1, 2))    # (B, 200)
        feats.retain_grad()
        logit = model.head(feats).sum()
        model.zero_grad(set_to_none=True)
        logit.backward()
        g = feats.grad.detach().cpu().float().numpy()
        acc = g.sum(axis=0) if acc is None else acc + g.sum(axis=0)
        n += g.shape[0]
    mean_grad = acc / max(n, 1)
    spec = np.abs(np.fft.rfft(mean_grad, n=512))
    return {"mean_grad": mean_grad, "spectrum": spec,
            "freqs": np.fft.rfftfreq(512, d=1.0 / SFREQ), "n_windows": n}


# ---------------------------------------------------------------------------
# Tier B — input attribution
# ---------------------------------------------------------------------------
def integrated_gradients(model: nn.Module, x: torch.Tensor, *, baseline: torch.Tensor,
                         steps: int = 64, device: str = "cuda") -> torch.Tensor:
    """Integrated Gradients of the OFF logit w.r.t. the (B, 18, 10, 200) input.

    Straight-line path from ``baseline`` to ``x``, Riemann midpoint rule. Returns
    an attribution tensor shaped like ``x``.

    Baseline choice is the dominant source of IG disagreement and is treated as a
    sensitivity analysis, not a default — see XAI_PLAN.md B1. Under ``chan_z`` a
    zero baseline is the per-channel mean, which is the most defensible starting
    point, but run the alternatives before trusting any map.
    """
    model.eval()
    x = x.to(device).float()
    baseline = baseline.to(device).float().expand_as(x)
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(x)
    for a in alphas:
        pt = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        logit = model(pt).sum()
        model.zero_grad(set_to_none=True)
        (grad,) = torch.autograd.grad(logit, pt)
        total += grad.detach()
    return ((x - baseline) * total / steps).detach()


# ---------------------------------------------------------------------------
# Tier C — causal perturbation
# ---------------------------------------------------------------------------
@torch.no_grad()
def occlude_channels(model: nn.Module, x: torch.Tensor, channels, *,
                     mode: str = "zero", replacement: torch.Tensor | None = None,
                     device: str = "cuda") -> torch.Tensor:
    """Logits after occluding ``channels``. ``mode``: 'zero' or 'shuffle'.

    'shuffle' substitutes the same channel from ``replacement`` (a different window
    of the SAME session), which preserves that channel's marginal statistics and
    destroys only its pairing with the rest of the window — a stricter control than
    zeroing, which also removes signal power.
    """
    model.eval()
    xo = x.clone().to(device).float()
    idx = list(channels)
    if mode == "zero":
        xo[:, idx] = 0.0
    elif mode == "shuffle":
        if replacement is None:
            raise ValueError("mode='shuffle' needs a replacement batch")
        xo[:, idx] = replacement.to(device).float()[:, idx]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return model(xo).detach().cpu()


# ---------------------------------------------------------------------------
def load_xai_checkpoint(path, data_cfg, ft_cfg, *, device: str = "cuda"):
    """Rebuild the classifier and load a checkpoint written by ``--save-checkpoint``.

    Returns ``(model, payload)``. Fails loudly if the head type recorded in the
    checkpoint disagrees with the current config — that would silently attribute
    the wrong architecture.
    """
    from .model import build_classifier_model
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("head_type") != ft_cfg.model.head.type:
        raise ValueError(
            f"checkpoint head_type={payload.get('head_type')!r} but config says "
            f"{ft_cfg.model.head.type!r} — refusing to load a mismatched architecture")
    model = build_classifier_model(data_cfg, ft_cfg, mode="full_ft", device=device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True), None
    model.eval()
    log.info("Loaded %s (seed_base=%s, best_epoch=%s, %s)", getattr(path, "name", path),
             payload.get("seed_base"), payload.get("best_epoch"), payload.get("direction"))
    return model, payload
