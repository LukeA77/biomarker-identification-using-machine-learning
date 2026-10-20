# LOPO + task split — results (smoke_lopo_ts, input_norm=none)

Generated 2026-08-01 21:37:47

Train on OTHER patients' windows of one ADL task; test on the HELD-OUT patient's
windows of the other task. The held-out patient's recordings are never in training,
so a per-recording fingerprint cannot help — this is the only task-split variant in
which that route is closed. Session-level statistics, patient-cluster bootstrap,
within-subject permutation — the same as the cross-subject LOPO arms, so these
numbers are directly comparable to the 0.443 / 0.557 reference.

| arm | direction | folds | sessions | session bal-acc | 95% CI | AUC | perm p | *cross-subject LOPO* |
|---|---|---|---|---|---|---|---|---|
| frozen_probe | tea_to_toast | 3 | 6 | **0.500** | [0.500, 0.500] | 0.333 | 1.000 | *0.443* |
| frozen_probe | toast_to_tea | 3 | 6 | **0.667** | [0.500, 1.000] | 0.333 | 0.517 | *0.443* |
| frozen_probe | both_directions_pooled | 6 | 12 | **0.583** | [0.500, 0.750] | 0.306 | 0.483 | *0.443* |
| lora | tea_to_toast | 3 | 6 | **0.500** | [0.500, 0.500] | 0.444 | 1.000 | *0.557* |
| lora | toast_to_tea | 3 | 6 | **0.667** | [0.500, 1.000] | 0.778 | 0.517 | *0.557* |
| lora | both_directions_pooled | 6 | 12 | **0.583** | [0.500, 0.750] | 0.583 | 0.483 | *0.557* |
| full_ft | tea_to_toast | 3 | 6 | **0.500** | [0.500, 0.500] | 0.556 | 1.000 | *0.443* |
| full_ft | toast_to_tea | 3 | 6 | **0.833** | [0.500, 1.000] | 0.778 | 0.274 | *0.443* |
| full_ft | both_directions_pooled | 6 | 12 | **0.667** | [0.500, 0.750] | 0.583 | 0.224 | *0.443* |
| chanamp | tea_to_toast | 3 | 6 | **0.500** | [0.500, 0.500] | 0.111 | 1.000 | — |
| chanamp | toast_to_tea | 3 | 6 | **0.333** | [0.000, 1.000] | 0.444 | 0.881 | — |
| chanamp | both_directions_pooled | 6 | 12 | **0.417** | [0.250, 0.750] | 0.306 | 0.866 | — |
| relbp | tea_to_toast | 3 | 6 | **0.667** | [0.500, 1.000] | 0.556 | 0.517 | — |
| relbp | toast_to_tea | 3 | 6 | **0.333** | [0.000, 0.500] | 0.222 | 1.000 | — |
| relbp | both_directions_pooled | 6 | 12 | **0.500** | [0.250, 0.750] | 0.417 | 0.726 | — |
| aperiodic | tea_to_toast | 3 | 6 | **0.500** | [0.500, 0.500] | 0.000 | 1.000 | — |
| aperiodic | toast_to_tea | 3 | 6 | **0.333** | [0.000, 0.500] | 0.222 | 1.000 | — |
| aperiodic | both_directions_pooled | 6 | 12 | **0.417** | [0.250, 0.500] | 0.139 | 1.000 | — |
