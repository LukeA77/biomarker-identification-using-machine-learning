# Within-patient task-split — results (within_patient_tasksplit)

Generated 2026-07-30 13:54:48

Train on one ADL task, test on the other, **inside each patient**, using that
patient's existing session ON/OFF labels. Both directions, averaged per patient.
Patient is the unit of analysis (paired / random-effect style), never a naive
window pool. This is an ADDITIONAL experiment; the nested-LOPO arms are unchanged.

## Headline — per-patient balanced accuracy on the held-out task

| arm | N patients | mean bal-acc | 95% CI (patient bootstrap) | Wilcoxon p | sign-test p | above chance | LOPO bal-acc (cross-subject) |
|---|---|---|---|---|---|---|---|
| frozen_probe | 27 | **0.883** | [0.842, 0.919] | 0.000 | 0.000 | 27/27 | 0.443 |
| lora | 27 | **0.704** | [0.648, 0.758] | 0.000 | 0.000 | 23/27 | 0.557 |
| full_ft | 27 | **0.805** | [0.741, 0.861] | 0.000 | 0.000 | 24/27 | 0.443 |

## Threshold-free ordering test (OFF session scored more OFF-like than ON)

| arm | correct / total (patient x direction) | frac | binomial p | patients both directions | patients neither |
|---|---|---|---|---|---|
| frozen_probe | 54/54 | 1.000 | 0.000 | 27/27 | 0/27 |
| lora | 47/54 | 0.870 | 0.000 | 22/27 | 2/27 |
| full_ft | 48/54 | 0.889 | 0.000 | 23/27 | 2/27 |

## Per-direction means

| arm | direction | n | mean bal-acc |
|---|---|---|---|
| frozen_probe | tea_to_toast | 27 | 0.854 |
| frozen_probe | toast_to_tea | 27 | 0.913 |
| lora | tea_to_toast | 27 | 0.663 |
| lora | toast_to_tea | 27 | 0.745 |
| full_ft | tea_to_toast | 27 | 0.776 |
| full_ft | toast_to_tea | 27 | 0.834 |

## Per-patient values (direction-averaged)

| patient | frozen_probe | lora | full_ft |
|---|---|---|---|
| sub-51 | 0.591 | 0.434 | 0.540 |
| sub-62 | 0.872 | 0.505 | 0.871 |
| sub-70 | 0.749 | 0.719 | 0.811 |
| sub-71 | 0.887 | 0.870 | 0.888 |
| sub-72 | 0.930 | 0.667 | 0.714 |
| sub-76 | 0.989 | 0.796 | 0.891 |
| sub-77 | 0.964 | 0.849 | 0.953 |
| sub-81 | 0.948 | 0.730 | 0.910 |
| sub-84 | 0.981 | 0.861 | 0.938 |
| sub-90 | 0.946 | 0.748 | 0.912 |
| sub-92 | 0.766 | 0.688 | 0.818 |
| sub-94 | 0.715 | 0.500 | 0.486 |
| sub-96 | 1.000 | 0.922 | 0.992 |
| sub-103 | 0.974 | 0.794 | 0.961 |
| sub-106 | 0.982 | 0.963 | 0.970 |
| sub-107 | 0.828 | 0.500 | 0.500 |
| sub-114 | 0.946 | 0.792 | 0.963 |
| sub-115 | 0.888 | 0.454 | 0.496 |
| sub-116 | 0.904 | 0.842 | 0.937 |
| sub-119 | 0.891 | 0.616 | 0.833 |
| sub-121 | 0.873 | 0.535 | 0.811 |
| sub-123 | 0.876 | 0.532 | 0.642 |
| sub-125 | 0.831 | 0.761 | 0.769 |
| sub-129 | 0.961 | 0.589 | 0.781 |
| sub-131 | 0.941 | 0.821 | 0.906 |
| sub-134 | 0.672 | 0.669 | 0.552 |
| sub-135 | 0.949 | 0.855 | 0.892 |
