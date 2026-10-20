# Task-split controls under input normalisation (tasksplit_controls_matched)

Generated 2026-08-06 11:16:49

Identical protocol to the within-patient task-split arms: train on one ADL task,
test on the other's non-overlap windows, both directions, per-patient balanced
accuracy, patient as the unit of analysis. Only the FEATURES (and, for `recid`, the
TARGET) change. Raw-input reference numbers are Experiment 12b-12c.

## Control matrix — mean per-patient balanced accuracy

| control | features/window | `chan_z` | raw-input reference (Exp. 12) |
|---|---|---|---|
| `chanamp` | 18 | 0.599 [0.566, 0.632] | 0.934 |
| `relbp` | 90 | 0.907 [0.871, 0.939] | — |
| `aperiodic` | 36 | 0.882 [0.841, 0.917] | — |
| `recid` | 144 | 0.920 [0.887, 0.949] | — |

## Ordering test (OFF session scored more OFF-like than ON)

| control | norm | correct/total | binomial p |
|---|---|---|---|
| `chanamp` | `chan_z` | 46/54 | 6.92e-08 |
| `relbp` | `chan_z` | 54/54 | 5.55e-17 |
| `aperiodic` | `chan_z` | 54/54 | 5.55e-17 |
| `recid` | `chan_z` | 54/54 | 5.55e-17 |

## Transfer test — within-patient vs cross-patient LOPO

Same features, same windows, same statistics; only the split changes.
The DROP is the evidence, not the absolute value.

| control | norm | within-patient | cross-patient LOPO | drop | verdict |
|---|---|---|---|---|---|
| `chanamp` | `chan_z` | 0.609 | 0.521 | 0.088 | per-recording FINGERPRINT |
| `relbp` | `chan_z` | 0.910 | 0.558 | 0.352 | transfers across patients — may carry a genuine effect; re-examine |
| `aperiodic` | `chan_z` | 0.886 | 0.533 | 0.352 | per-recording FINGERPRINT |
| `recid` | `chan_z` | — | — | — | not defined (see JSON) |
