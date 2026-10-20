# LOPO + task split — results (lopo_tasksplit_chan_z, input_norm=chan_z)

Generated 2026-08-02 14:26:41

Train on OTHER patients' windows of one ADL task; test on the HELD-OUT patient's
windows of the other task. The held-out patient's recordings are never in training,
so a per-recording fingerprint cannot help — this is the only task-split variant in
which that route is closed. Session-level statistics, patient-cluster bootstrap,
within-subject permutation — the same as the cross-subject LOPO arms, so these
numbers are directly comparable to the 0.443 / 0.557 reference.

| arm | direction | folds | sessions | session bal-acc | 95% CI | AUC | perm p | *cross-subject LOPO* |
|---|---|---|---|---|---|---|---|---|
| frozen_probe | tea_to_toast | 27 | 54 | **0.444** | [0.333, 0.574] | 0.487 | 0.871 | *0.443* |
| frozen_probe | toast_to_tea | 27 | 54 | **0.519** | [0.407, 0.630] | 0.495 | 0.547 | *0.443* |
| frozen_probe | both_directions_pooled | 54 | 108 | **0.481** | [0.389, 0.583] | 0.487 | 0.761 | *0.443* |
| lora | tea_to_toast | 27 | 54 | **0.574** | [0.444, 0.704] | 0.529 | 0.214 | *0.557* |
| lora | toast_to_tea | 27 | 54 | **0.537** | [0.426, 0.648] | 0.542 | 0.408 | *0.557* |
| lora | both_directions_pooled | 54 | 108 | **0.556** | [0.472, 0.648] | 0.534 | 0.139 | *0.557* |
| full_ft | tea_to_toast | 27 | 54 | **0.537** | [0.444, 0.630] | 0.561 | 0.358 | *0.443* |
| full_ft | toast_to_tea | 27 | 54 | **0.537** | [0.426, 0.648] | 0.521 | 0.398 | *0.443* |
| full_ft | both_directions_pooled | 54 | 108 | **0.537** | [0.472, 0.611] | 0.535 | 0.199 | *0.443* |
| chanamp | tea_to_toast | 27 | 54 | **0.556** | [0.444, 0.685] | 0.558 | 0.239 | — |
| chanamp | toast_to_tea | 27 | 54 | **0.611** | [0.537, 0.704] | 0.609 | 0.030 | — |
| chanamp | both_directions_pooled | 54 | 108 | **0.583** | [0.509, 0.657] | 0.582 | 0.020 | — |
| relbp | tea_to_toast | 27 | 54 | **0.556** | [0.407, 0.704] | 0.604 | 0.308 | — |
| relbp | toast_to_tea | 27 | 54 | **0.685** | [0.574, 0.778] | 0.701 | 0.010 | — |
| relbp | both_directions_pooled | 54 | 108 | **0.620** | [0.519, 0.713] | 0.645 | 0.005 | — |
| aperiodic | tea_to_toast | 27 | 54 | **0.537** | [0.407, 0.667] | 0.512 | 0.413 | — |
| aperiodic | toast_to_tea | 27 | 54 | **0.574** | [0.481, 0.667] | 0.547 | 0.149 | — |
| aperiodic | both_directions_pooled | 54 | 108 | **0.556** | [0.463, 0.648] | 0.539 | 0.144 | — |
