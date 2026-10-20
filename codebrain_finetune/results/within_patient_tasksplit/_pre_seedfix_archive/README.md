# Archived: produced BEFORE the seeding fix

These lora / full_ft runs were produced with a seed derived from Python's builtin
`hash()`, which is randomised per interpreter (PYTHONHASHSEED). Their exact numbers
are therefore not reproducible. Kept only as an audit trail.

The reported lora / full_ft results are the re-run with `zlib.crc32`-derived seeds
(deterministic across processes) in the sibling `lora/` and `full_ft/` directories.

The `frozen_probe/` arm was NOT re-run and was NOT archived: it does no training
(frozen backbone in eval mode + logistic regression), so it is seed-invariant.

Pre-fix values, for comparison with the re-run:
  lora    mean per-patient bal-acc 0.661  CI [0.591, 0.731]  17/20 above chance  ordering 30/40
  full_ft mean per-patient bal-acc 0.779  CI [0.703, 0.850]  19/20 above chance  ordering 34/40
