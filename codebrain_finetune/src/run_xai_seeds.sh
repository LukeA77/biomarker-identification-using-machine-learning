#!/usr/bin/env bash
# XAI Phase 0: retrain the headline arm under 5 seeds and persist every checkpoint.
#
# The published 0.832 model was never saved (XAI_PLAN.md 2.1) and the run is not
# deterministic (2.2), so the XAI subject has to be rebuilt. Seed base 1 is the
# finetune.yaml default, i.e. the same base as the original matched run — its
# result is therefore the reproducibility check, and seeds 2-5 give the spread
# that the 2.2 gate is scored against (flag at +/-0.03, stop at +/-0.05).
#
# Only the full_ft arm is retrained: it is the arm being explained. Both
# directions run because the driver pairs them, and toast_to_tea is the headline.
set -u
cd "$(dirname "$0")/../.." || exit 1

for SEED in 1 2 3 4 5; do
  echo "=== seed ${SEED} ==="
  python codebrain_finetune/src/run_pooled_tasksplit.py \
    --input-norm chan_z \
    --match-level patient_task \
    --match-where uniform \
    --arms full_ft \
    --save-checkpoint \
    --seed-base "${SEED}" \
    --run-tag "xai_seed${SEED}" || echo "SEED ${SEED} FAILED (exit $?)"
done
echo "=== all seeds done ==="
