# LOPO + task split — results (lopo_tasksplit_none, input_norm=none)

Generated 2026-08-02 14:24:03

Train on OTHER patients' windows of one ADL task; test on the HELD-OUT patient's
windows of the other task. The held-out patient's recordings are never in training,
so a per-recording fingerprint cannot help — this is the only task-split variant in
which that route is closed. Session-level statistics, patient-cluster bootstrap,
within-subject permutation — the same as the cross-subject LOPO arms, so these
numbers are directly comparable to the 0.443 / 0.557 reference.

| arm | direction | folds | sessions | session bal-acc | 95% CI | AUC | perm p | *cross-subject LOPO* |
|---|---|---|---|---|---|---|---|---|
| frozen_probe | tea_to_toast | 27 | 54 | **0.556** | [0.444, 0.667] | 0.484 | 0.244 | *0.443* |
| frozen_probe | toast_to_tea | 27 | 54 | **0.481** | [0.370, 0.593] | 0.479 | 0.776 | *0.443* |
| frozen_probe | both_directions_pooled | 54 | 108 | **0.519** | [0.444, 0.602] | 0.484 | 0.363 | *0.443* |
| lora | tea_to_toast | 27 | 54 | **0.463** | [0.352, 0.574] | 0.440 | 0.846 | *0.557* |
| lora | toast_to_tea | 27 | 54 | **0.389** | [0.278, 0.500] | 0.377 | 0.965 | *0.557* |
| lora | both_directions_pooled | 54 | 108 | **0.426** | [0.333, 0.519] | 0.401 | 0.955 | *0.557* |
| full_ft | tea_to_toast | 27 | 54 | **0.481** | [0.370, 0.593] | 0.458 | 0.741 | *0.443* |
| full_ft | toast_to_tea | 27 | 54 | **0.389** | [0.259, 0.519] | 0.376 | 0.965 | *0.443* |
| full_ft | both_directions_pooled | 54 | 108 | **0.435** | [0.343, 0.528] | 0.413 | 0.945 | *0.443* |
| chanamp | tea_to_toast | 27 | 54 | **0.426** | [0.333, 0.519] | 0.420 | 0.950 | — |
| chanamp | toast_to_tea | 27 | 54 | **0.519** | [0.444, 0.593] | 0.534 | 0.493 | — |
| chanamp | both_directions_pooled | 54 | 108 | **0.472** | [0.407, 0.537] | 0.480 | 0.856 | — |
| relbp | tea_to_toast | 27 | 54 | **0.556** | [0.407, 0.704] | 0.604 | 0.308 | — |
| relbp | toast_to_tea | 27 | 54 | **0.685** | [0.574, 0.778] | 0.701 | 0.010 | — |
| relbp | both_directions_pooled | 54 | 108 | **0.620** | [0.519, 0.713] | 0.645 | 0.005 | — |
| aperiodic | tea_to_toast | 27 | 54 | **0.519** | [0.389, 0.630] | 0.495 | 0.532 | — |
| aperiodic | toast_to_tea | 27 | 54 | **0.519** | [0.407, 0.630] | 0.519 | 0.542 | — |
| aperiodic | both_directions_pooled | 54 | 108 | **0.519** | [0.407, 0.620] | 0.505 | 0.458 | — |
