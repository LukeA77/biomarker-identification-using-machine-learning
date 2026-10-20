"""XAI Phase 0 gate: is the model stable enough across seeds to be worth explaining?

The published 0.832 model was never saved and the run is not deterministic
(XAI_PLAN.md 2.1-2.2), so the XAI subject is a retrained model. This scores the
gate that was fixed BEFORE the numbers were seen (XAI_PLAN.md 9.5):

    spread (max - min test balanced accuracy over seeds)
      <= 0.03  -> proceed normally
      <= 0.05  -> proceed, but report the spread prominently on every figure
       > 0.05  -> STOP. No attribution claim is published; the instability is
                  itself the finding.

Also checks reproducibility: seed base 1 is the finetune.yaml default, the same
base as the original matched run, so its result should land near the published
0.832 if non-determinism is the only difference.

Usage:
    python codebrain_finetune/src/check_seed_stability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
DIRECTION = "toast_to_tea"          # the headline direction
ARM = "full_ft"
PUBLISHED = 0.832                   # pooled_tasksplit_chanz_matched
FLAG_AT, STOP_AT = 0.03, 0.05


def main() -> int:
    rows = []
    for seed in range(1, 6):
        p = _RUN_ROOT / "results" / f"xai_seed{seed}" / f"{DIRECTION}_{ARM}.json"
        if not p.is_file():
            print(f"seed {seed}: MISSING ({p.name}) — run not finished?")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        pp = d["pooled_patient_level"]
        rows.append({
            "seed": seed, "ba": pp["mean"], "lo": pp["ci_low"], "hi": pp["ci_high"],
            "window_ba": d["window"]["balanced_accuracy"],
            "session_ba": d["session"]["balanced_accuracy"],
            "best_epoch": d.get("best_epoch"),
            "ckpt": d.get("checkpoint"),
        })
    if not rows:
        print("No seed results found.")
        return 1

    ba = np.array([r["ba"] for r in rows])
    spread = float(ba.max() - ba.min())

    print(f"XAI Phase 0 — seed stability, {DIRECTION} / {ARM}")
    print(f"published (never saved): {PUBLISHED:.3f}\n")
    print(f"{'seed':>5}{'patient-mean BA':>18}{'95% CI':>20}{'window':>9}{'session':>9}{'ep':>4}")
    for r in rows:
        print(f"{r['seed']:>5}{r['ba']:>18.3f}   [{r['lo']:.3f}, {r['hi']:.3f}]"
              f"{r['window_ba']:>9.3f}{r['session_ba']:>9.3f}{r['best_epoch']:>4}")

    print(f"\nmean {ba.mean():.3f} | sd {ba.std(ddof=1):.4f} | min {ba.min():.3f} | "
          f"max {ba.max():.3f} | SPREAD {spread:.3f}")

    seed1 = next((r for r in rows if r["seed"] == 1), None)
    if seed1:
        print(f"reproducibility (seed base 1 vs published): {seed1['ba']:.3f} vs "
              f"{PUBLISHED:.3f}  (delta {seed1['ba'] - PUBLISHED:+.3f})")

    print()
    if spread > STOP_AT:
        print(f"GATE: **STOP** — spread {spread:.3f} > {STOP_AT}. Per XAI_PLAN 9.5 no "
              f"attribution claim is published; report the instability instead.")
        return 2
    if spread > FLAG_AT:
        print(f"GATE: **PROCEED WITH FLAG** — spread {spread:.3f} > {FLAG_AT}. Report it "
              f"prominently on every attribution figure.")
    else:
        print(f"GATE: **PROCEED** — spread {spread:.3f} <= {FLAG_AT}.")

    # Median seed is the primary XAI subject in BOTH proceed branches — picking it
    # only on the clean branch would leave the flagged path with no named subject.
    med = sorted(rows, key=lambda r: r["ba"])[len(rows) // 2]
    print(f"\nPrimary XAI subject = median seed {med['seed']} (BA {med['ba']:.3f})")
    print(f"  checkpoint: {med['ckpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
