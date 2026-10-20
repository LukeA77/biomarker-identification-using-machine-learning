# Pairwise transfer matrix with a max-statistic null

Generated 2026-08-01 22:32:01

Arm: `frozen_probe` (rebuilt — see below). Each cell is the SOURCE patient's
task-split model applied to the TARGET patient's held-out-task windows,
averaged over both directions.

**The maximum is the statistic that needs a null.** With 702 ordered pairs, a best
cell at 0.8+ is what the null produces by itself. Every row below therefore reports
the observed value against a null built by permuting ON/OFF labels within patient,
rebuilding the whole matrix, and taking its maximum.

| norm | pairs | diagonal (within-patient) | off-diag mean | off-diag MAX | best pair | null of max (mean±sd) | **p(max)** |
|---|---|---|---|---|---|---|---|
| `none` | 702 | 0.883 | 0.496 | **0.984** | sub-103→sub-106 | 0.980 ± 0.010 | **0.736** |
| `chan_z` | 702 | 0.811 | 0.500 | **0.856** | sub-72→sub-77 | 0.902 ± 0.022 | **0.965** |

## Does any ONE source model generalise broadly?

| norm | best row-mean (per source) | source | p |
|---|---|---|---|
| `none` | 0.520 | sub-72 | 0.970 |
| `chan_z` | 0.555 | sub-72 | 0.075 |

## Provenance

The within-patient task-split runs persisted metrics JSONs only — no per-patient
model weights were ever written to disk — so the source models are REBUILT here.
That is exact for `frozen_probe` (frozen backbone + logistic head) and is verified
against the stored numbers on the matrix diagonal:

- `none`: rebuilt diagonal mean 0.8831 vs stored 0.8834, max |delta| 7.74e-03 over 27 patients — **PASS**
