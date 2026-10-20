#!/usr/bin/env bash
# XAI confirmation package: Tier C across all 5 seeds, then the multi-width sweep.
#
# Tier D (run_xai_discriminative.py) is NO LONGER RUN as part of the package.
# Withdrawn 2026-08-15: every Tier D statistic compares channel identity across
# patients, and the cap moves during a session, so an electrode label does not
# index a stable scalp location. The script is kept and still works if the
# question is ever revisited under a design that tracks electrode position.
set -u
cd "$(dirname "$0")/../.." || exit 1
for S in 1 2 3 4 5; do
  CK="checkpoints/xai/xai_seed${S}__toast_to_tea__full_ft__seed${S}.pt"
  echo "=== Tier C seed ${S} ==="
  python codebrain_finetune/src/run_xai_perturbation.py --checkpoint "$CK" \
    || echo "TIERC SEED ${S} FAILED ($?)"
done

# Tier C-2: the multi-width band-stop sweep that supersedes the 2 Hz result. Runs
# once over ALL seeds rather than per checkpoint — the five share a test frame, so
# each band is filtered once and scored five times (~65 min total for 469 bands).
echo "=== Tier C-2 multi-width band sweep (all seeds) ==="
python codebrain_finetune/src/run_xai_bandsweep.py --preflight \
  && python codebrain_finetune/src/run_xai_bandsweep.py \
  && python codebrain_finetune/src/summarize_bandsweep.py --json \
  && python codebrain_finetune/src/plot_bandsweep.py \
  || echo "TIER C-2 FAILED ($?)"
echo "=== xai full package done ==="
