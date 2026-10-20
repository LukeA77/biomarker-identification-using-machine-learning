"""CodeBrain model assembly for fine-tuning: backbone + head, LoRA, features.

Everything model-specific here follows the verified downstream contract
(the CodeBrain downstream contract and the Stage-2 forward-pass gate):

* backbone = ``Models/SSSM.py::SSSM`` built with CodeBrain's TUAB args, checkpoint
  loaded by stripping the 7-char ``module.`` prefix (0 missing / 0 unexpected);
* ``if_codebook=False`` — tokenizer/codebooks are not on the downstream path;
* backbone output is per-patch features ``(B, 18, 10, 200)`` (SSSM ends in
  ``x.squeeze()``, so **batch >= 2** and **GPU** are mandatory);
* head flattens ``18*10*200 -> 2000 -> 200 -> 1`` (binary logit).

LoRA target modules were confirmed live against ``named_modules()``: the 8 SWA
blocks are ``nn.MultiheadAttention`` (q/k/v fused in ``in_proj_weight``, only
``out_proj`` a standard ``nn.Linear``). PEFT 0.19 adapts either the whole
attention module (q/k/v+out) or just ``out_proj``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from livinglab_codebrain.config import DataConfig
from .ft_config import FinetuneConfig

log = logging.getLogger(__name__)

__all__ = [
    "build_backbone", "load_backbone_checkpoint", "CodeBrainClassifier",
    "apply_lora", "trainable_parameter_report", "build_classifier_model",
]

# Modules that must never be trained/adapted: the tokenizer/codebook path.
# spectral_proj IS used in the forward (patch embedding) but is frozen pretrained;
# lm_head_t/lm_head_f are unused when if_codebook=False.
_TOKENIZER_PATH_KEYWORDS = ("lm_head_t", "lm_head_f", "spectral_proj")


def build_backbone(cfg: FinetuneConfig, repo: Path) -> nn.Module:
    """Instantiate CodeBrain's SSSM backbone from config (no weights loaded yet)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from Models.SSSM import SSSM  # noqa: E402 (path set above)

    return SSSM(**cfg.model.backbone)


def load_backbone_checkpoint(backbone: nn.Module, ckpt_path: Path) -> None:
    """Load ``CodeBrain.pth`` (strip 7-char ``module.`` prefix); require 0/0 keys."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    stripped = {k[7:]: v for k, v in sd.items()}
    result = backbone.load_state_dict(stripped, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"CodeBrain checkpoint load not clean: "
            f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected. "
            f"missing[:5]={result.missing_keys[:5]} unexpected[:5]={result.unexpected_keys[:5]}"
        )
    log.info("CodeBrain backbone checkpoint loaded (0 missing / 0 unexpected).")


class CodeBrainClassifier(nn.Module):
    """SSSM backbone + a 3-layer MLP head producing a single binary logit.

    ``backbone`` may be plain (frozen probe / full-FT) or PEFT-wrapped (LoRA);
    the head is always a fresh, fully-trainable module.
    """

    def __init__(self, backbone: nn.Module, data_cfg: DataConfig, ft_cfg: FinetuneConfig) -> None:
        super().__init__()
        self.backbone = backbone
        w = data_cfg.window
        self._samples = w.samples_per_patch                                  # 200
        self._flat_dim = w.n_channels * w.n_patches * w.samples_per_patch     # 36000
        h = ft_cfg.model.head
        self._head_type = h.type
        if h.type == "flatten_mlp":
            # CodeBrain's TUAB head: flatten everything (72M params in layer 1).
            self.head = nn.Sequential(
                nn.Linear(self._flat_dim, h.hidden1), nn.ELU(), nn.Dropout(h.dropout),
                nn.Linear(h.hidden1, h.hidden2),      nn.ELU(), nn.Dropout(h.dropout),
                nn.Linear(h.hidden2, 1),
            )
        else:  # pool_mlp: mean-pool channels x patches -> samples-d, small MLP (~13k)
            self.head = nn.Sequential(
                nn.Linear(self._samples, h.pool_hidden), nn.ELU(), nn.Dropout(h.dropout),
                nn.Linear(h.pool_hidden, 1),
            )

    def backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch backbone features ``(B, n_channels, n_patches, samples)``."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feats = self.backbone_features(x)                    # (B, C, P, S)
        if self._head_type == "flatten_mlp":
            feats = feats.contiguous().view(b, self._flat_dim)
        else:  # pool over channels and patches
            feats = feats.mean(dim=(1, 2))                   # (B, S)
        return self.head(feats).view(b)  # (B,) logit


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def apply_lora(backbone: nn.Module, cfg: FinetuneConfig) -> nn.Module:
    """Wrap the backbone with a PEFT LoRA adapter per ``cfg.lora``.

    ``get_peft_model`` freezes every non-adapter base parameter automatically, so
    SGConv kernels and the tokenizer path are frozen with no extra bookkeeping.
    Returns the PEFT-wrapped backbone (call ``.forward`` exactly as before).
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=[cfg.lora.target],  # "attention" (q/k/v+out) or "out_proj"
    )
    wrapped = get_peft_model(backbone, lora_cfg)
    log.info("LoRA applied: target=%s r=%d alpha=%d dropout=%s",
             cfg.lora.target, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout)
    return wrapped


def build_classifier_model(
    data_cfg: DataConfig,
    ft_cfg: FinetuneConfig,
    *,
    mode: str,
    device: str = "cuda",
) -> CodeBrainClassifier:
    """Assemble a ready-to-use classifier for one of three training modes.

    * ``mode="lora"``     — backbone frozen except LoRA adapters; head trainable.
    * ``mode="full_ft"``  — entire backbone + head trainable (comparison arm).
    * ``mode="frozen"``   — backbone fully frozen; head trainable (used as the
      backbone for the linear probe's feature extraction, or a frozen-backbone
      head-only baseline).
    """
    if mode not in {"lora", "full_ft", "frozen"}:
        raise ValueError(f"mode must be lora/full_ft/frozen, got {mode!r}")

    backbone = build_backbone(ft_cfg, data_cfg.paths.codebrain_repo)
    load_backbone_checkpoint(backbone, data_cfg.paths.checkpoint)

    if mode == "lora":
        backbone = apply_lora(backbone, ft_cfg)
    elif mode == "frozen":
        _freeze_all(backbone)
    # full_ft: leave all backbone params trainable.

    model = CodeBrainClassifier(backbone, data_cfg, ft_cfg).to(device)
    return model


def trainable_parameter_report(model: nn.Module) -> dict:
    """Trainable vs total parameter counts (for logging/provenance)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_frac": trainable / total if total else 0.0,
    }
