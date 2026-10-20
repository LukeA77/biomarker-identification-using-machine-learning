# Task-split controls under input normalisation (tasksplit_controls_norm)

Generated 2026-08-06 11:13:18

Identical protocol to the within-patient task-split arms: train on one ADL task,
test on the other's non-overlap windows, both directions, per-patient balanced
accuracy, patient as the unit of analysis. Only the FEATURES (and, for `recid`, the
TARGET) change. Raw-input reference numbers are Experiment 12b-12c.

## Control matrix — mean per-patient balanced accuracy

| control | features/window | `none` | `chan_z` | raw-input reference (Exp. 12) |
|---|---|---|---|---|
| `amp1` | 1 | 0.598 [0.547, 0.651] | 0.518 [0.492, 0.547] | 0.598 |
| `crude3` | 3 | 0.624 [0.569, 0.680] | 0.522 [0.501, 0.544] | 0.624 |
| `chanamp` | 18 | 0.934 [0.901, 0.961] | 0.609 [0.578, 0.641] | 0.934 |
| `relbp` | 90 | 0.910 [0.875, 0.941] | 0.910 [0.875, 0.941] | — |
| `aperiodic` | 36 | 0.970 [0.955, 0.982] | 0.886 [0.846, 0.920] | — |
| `recid` | 144 | 0.963 [0.946, 0.978] | 0.924 [0.893, 0.952] | — |

## Ordering test (OFF session scored more OFF-like than ON)

| control | norm | correct/total | binomial p |
|---|---|---|---|
| `amp1` | `none` | 40/54 | 0.000268 |
| `crude3` | `none` | 42/54 | 2.6e-05 |
| `chanamp` | `none` | 54/54 | 5.55e-17 |
| `relbp` | `none` | 54/54 | 5.55e-17 |
| `aperiodic` | `none` | 54/54 | 5.55e-17 |
| `recid` | `none` | 54/54 | 5.55e-17 |
| `amp1` | `chan_z` | 28/54 | 0.446 |
| `crude3` | `chan_z` | 36/54 | 0.00992 |
| `chanamp` | `chan_z` | 46/54 | 6.92e-08 |
| `relbp` | `chan_z` | 54/54 | 5.55e-17 |
| `aperiodic` | `chan_z` | 54/54 | 5.55e-17 |
| `recid` | `chan_z` | 54/54 | 5.55e-17 |

## Transfer test — within-patient vs cross-patient LOPO

Same features, same windows, same statistics; only the split changes.
The DROP is the evidence, not the absolute value.

| control | norm | within-patient | cross-patient LOPO | drop | verdict |
|---|---|---|---|---|---|
| `amp1` | `none` | 0.598 | 0.476 | 0.122 | per-recording FINGERPRINT |
| `crude3` | `none` | 0.624 | 0.499 | 0.125 | per-recording FINGERPRINT |
| `chanamp` | `none` | 0.934 | 0.495 | 0.440 | per-recording FINGERPRINT |
| `relbp` | `none` | 0.910 | 0.558 | 0.352 | transfers across patients — may carry a genuine effect; re-examine |
| `aperiodic` | `none` | 0.970 | 0.518 | 0.452 | per-recording FINGERPRINT |
| `recid` | `none` | — | — | — | not defined (see JSON) |
| `amp1` | `chan_z` | 0.518 | 0.509 | 0.009 | per-recording FINGERPRINT |
| `crude3` | `chan_z` | 0.522 | 0.495 | 0.027 | per-recording FINGERPRINT |
| `chanamp` | `chan_z` | 0.609 | 0.521 | 0.088 | per-recording FINGERPRINT |
| `relbp` | `chan_z` | 0.910 | 0.558 | 0.352 | transfers across patients — may carry a genuine effect; re-examine |
| `aperiodic` | `chan_z` | 0.886 | 0.533 | 0.352 | per-recording FINGERPRINT |
| `recid` | `chan_z` | — | — | — | not defined (see JSON) |
