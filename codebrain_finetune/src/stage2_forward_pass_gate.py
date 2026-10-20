"""
Stage 2 — CodeBrain forward-pass gate.

Purpose (and ONLY purpose): prove that the pretrained CodeBrain EEGSSM backbone
(a) instantiates, (b) loads the downloaded checkpoint cleanly, and (c) runs one
real no-grad forward pass over a genuine cached window on the RTX 5090, returning
the expected shape. No training, no data loader, no fine-tuning.

Everything model-specific here is copied from CodeBrain's own downstream recipe so
the test is faithful to how they use the model:
  - backbone build args        -> Models/model_for_tuab.py  (TUAB = binary, like our ON/OFF)
  - checkpoint key handling     -> model_for_tuab.py (strip 7-char "module." prefix)
  - if_codebook=False           -> codebooks/tokenizer NOT used downstream
  - input normalisation ÷100    -> Datasets/tuab_dataset.py  (resolves plan OQ-1b)
  - input layout (B, C, P, S)   -> SSSM.forward: (batch, channels, patches, samples)

Paths come from codebrain_finetune/config/paths.yaml (single source of truth).
Run from the project root:  python codebrain_finetune/src/stage2_forward_pass_gate.py
"""
from __future__ import annotations
import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


# ---- CodeBrain's downstream backbone config (verbatim from model_for_tuab.py) ----
# Only n_layer/dropout are exposed by their CLI; the rest are fixed in the wrapper.
BACKBONE_KWARGS = dict(
    in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
    num_res_layers=8,
    diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200,
    diffusion_step_embed_dim_out=200,
    s4_lmax=570, s4_d_state=64, s4_dropout=0.1,
    s4_bidirectional=True, s4_layernorm=True,
    codebook_size_t=4096, codebook_size_f=4096,
    if_codebook=False,          # downstream: no tokenizer/codebooks
)
NORM_DIVISOR = 100.0            # ÷100, per tuab_dataset.py (plan OQ-1b resolved)
N_CHANNELS_EXPECTED = 18
N_PATCHES_EXPECTED = 10
SAMPLES_PER_PATCH = 200


def load_paths() -> dict:
    """Load config/paths.yaml relative to this file, return resolved absolute paths."""
    here = Path(__file__).resolve()
    run_root = here.parent.parent                      # codebrain_finetune/
    project_root = run_root.parent                     # Luke_Dissertation/
    cfg = yaml.safe_load((run_root / "config" / "paths.yaml").read_text())
    return {
        "project_root": project_root,
        "codebrain_repo": project_root / cfg["run"]["codebrain"],
        "checkpoint": run_root / "checkpoints" / "CodeBrain.pth",
        "manifest": project_root / cfg["stage0"]["manifest"],
        "cache_dir": project_root / cfg["stage0"]["cache_dir"],
    }


def build_backbone(repo: Path):
    """Import CodeBrain's SSSM and build the downstream backbone exactly as they do."""
    sys.path.insert(0, str(repo))
    from Models.SSSM import SSSM          # noqa: E402  (path set above)
    return SSSM(**BACKBONE_KWARGS)


def load_checkpoint(backbone: torch.nn.Module, ckpt_path: Path) -> tuple[list, list]:
    """Load CodeBrain.pth, stripping the 7-char 'module.' DataParallel prefix.

    Returns (missing_keys, unexpected_keys) from a non-strict load so we can report
    exactly what did/didn't line up rather than failing opaquely.
    """
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - report, then retry loudly
        print(f"[warn] weights_only=True load failed ({exc}); retrying weights_only=False")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # CodeBrain saved with DataParallel -> keys look like 'module.<name>'. Strip 7 chars.
    stripped = {k[7:]: v for k, v in state_dict.items()}
    prefix_ok = all(k.startswith("module.") for k in state_dict)
    print(f"[ckpt] {len(state_dict)} tensors | all keys carry 'module.' prefix: {prefix_ok}")
    print(f"[ckpt] example original key: {next(iter(state_dict))!r}")

    result = backbone.load_state_dict(stripped, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_one_window(manifest: Path, cache_dir: Path):
    """Pull one real cached window, reshape to CodeBrain's (C, P, S), normalise ÷100."""
    df = pd.read_parquet(manifest)
    row = df.iloc[0]
    with open(cache_dir / row["path"], "rb") as fh:
        obj = pickle.load(fh)
    x = obj["X"]  # (18, 2000) float32 µV
    assert x.shape == (N_CHANNELS_EXPECTED, N_PATCHES_EXPECTED * SAMPLES_PER_PATCH), x.shape
    x = x.reshape(N_CHANNELS_EXPECTED, N_PATCHES_EXPECTED, SAMPLES_PER_PATCH)
    x = x.astype(np.float32) / NORM_DIVISOR
    return x, row, obj["meta"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 CodeBrain forward-pass gate")
    ap.add_argument("--batch", type=int, default=2,
                    help="batch size for the test (>=2 avoids SSSM's squeeze() collapsing the batch dim)")
    args = ap.parse_args()

    # Windows consoles default to cp1252, which can't encode the /— glyphs below.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - non-fatal; fall back to default stream
        pass

    print("=" * 72)
    print("STAGE 2 — CodeBrain forward-pass gate")
    print("=" * 72)
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available. SSSM hardcodes .cuda() on the SWA mask; "
              "the forward pass cannot run on CPU. Aborting.")
        return 1
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")

    paths = load_paths()
    print(f"repo:       {paths['codebrain_repo']}")
    print(f"checkpoint: {paths['checkpoint']}")

    # 1) Build + load
    backbone = build_backbone(paths["codebrain_repo"])
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[build] SSSM instantiated | {n_params:,} parameters")

    missing, unexpected = load_checkpoint(backbone, paths["checkpoint"])
    print(f"[load] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print(f"        missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"        unexpected (first 10): {unexpected[:10]}")

    backbone = backbone.to(device).eval()

    # 2) One real window
    x_np, row, meta = load_one_window(paths["manifest"], paths["cache_dir"])
    print(f"[data] window {row['path']} | patient {meta['patient_id']} "
          f"cond {meta['condition']} y={row['y_cond']} | reshaped {x_np.shape} "
          f"| after ÷{NORM_DIVISOR:.0f} median|x|={np.median(np.abs(x_np)):.4f}")
    batch = np.repeat(x_np[None, ...], args.batch, axis=0)  # (B, 18, 10, 200)
    x = torch.from_numpy(batch).to(device)
    print(f"[data] batched input tensor: {tuple(x.shape)} {x.dtype} on {x.device}")

    # 3) Forward pass
    expected = (args.batch, N_CHANNELS_EXPECTED, N_PATCHES_EXPECTED, SAMPLES_PER_PATCH)
    with torch.no_grad():
        out = backbone(x)
    print(f"[forward] backbone output: {tuple(out.shape)} {out.dtype}")
    print(f"[forward] output finite: {torch.isfinite(out).all().item()} "
          f"| mean {out.mean().item():.5f} std {out.std().item():.5f}")

    ok_shape = tuple(out.shape) == expected
    ok_finite = bool(torch.isfinite(out).all().item())
    print("-" * 72)
    print(f"expected backbone output shape: {expected}  ->  {'PASS' if ok_shape else 'FAIL'}")
    print(f"output all finite (no NaN/Inf):        ->  {'PASS' if ok_finite else 'FAIL'}")

    # 4) Bonus: prove the full classification path with an 18-channel head
    #    (TUAB head is hardcoded to 16 channels; ours is 18 -> 18*10*200).
    head = torch.nn.Sequential(
        torch.nn.Linear(N_CHANNELS_EXPECTED * N_PATCHES_EXPECTED * SAMPLES_PER_PATCH, N_PATCHES_EXPECTED * SAMPLES_PER_PATCH),
        torch.nn.ELU(),
        torch.nn.Linear(N_PATCHES_EXPECTED * SAMPLES_PER_PATCH, SAMPLES_PER_PATCH),
        torch.nn.ELU(),
        torch.nn.Linear(SAMPLES_PER_PATCH, 1),
    ).to(device).eval()
    with torch.no_grad():
        feats = out.contiguous().view(args.batch, -1)
        logits = head(feats).view(args.batch)
    ok_head = tuple(logits.shape) == (args.batch,)
    print(f"[head] fresh 18-ch classifier logits: {tuple(logits.shape)}  ->  {'PASS' if ok_head else 'FAIL'}")

    passed = ok_shape and ok_finite and ok_head
    print("=" * 72)
    print(f"STAGE 2 GATE: {'PASS — cleared for Stage 3 (loader/probe)' if passed else 'FAIL — see above'}")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
