# Multi-width band-stop sweep — pooled_tasksplit_chanz_matched

Experiment: **pooled_tasksplit_chanz_matched**, direction `toast_to_tea`, arm `full_ft`. 469 bands, widths 5-18 Hz (step 1), positions 1-45 Hz at 1 Hz stride, scored on 1402 windows / 27 patients across 5 seeds.

Baseline patient-mean BA per seed: 1=0.8402, 2=0.8073, 3=0.8362, 4=0.8112, 5=0.8353 (mean 0.8260).

## Continuity with the 2 Hz sweep

The superseded 2 Hz sweep (seed 1, same filters, same frame) had mean |dBA| 0.0052 and a maximum of 0.0176. This sweep's widths reach the values below, which is the point: the 2 Hz null was a statement about 2 Hz, not about the model.

## Best band at each width (within-width comparison — the only valid one)

| width | best band | dBA | seed sd | seeds agreeing | power removed | within-width z |
|---|---|---|---|---|---|---|
| 5 Hz | 4-9 Hz | **+0.0749** | 0.0111 | 5/5 | 10.2% | +3.40 |
| 6 Hz | 4-10 Hz | **+0.0946** | 0.0069 | 5/5 | 11.1% | +3.07 |
| 7 Hz | 3-10 Hz | **+0.1086** | 0.0067 | 5/5 | 18.3% | +2.73 |
| 8 Hz | 1-9 Hz | **+0.1255** | 0.0047 | 5/5 | 94.7% | +2.80 |
| 9 Hz | 1-10 Hz | **+0.1374** | 0.0030 | 5/5 | 95.5% | +2.92 |
| 10 Hz | 1-11 Hz | **+0.1391** | 0.0034 | 5/5 | 96.2% | +2.82 |
| 11 Hz | 1-12 Hz | **+0.1416** | 0.0055 | 5/5 | 96.8% | +2.67 |
| 12 Hz | 1-13 Hz | **+0.1456** | 0.0048 | 5/5 | 97.2% | +2.55 |
| 13 Hz | 1-14 Hz | **+0.1499** | 0.0052 | 5/5 | 97.6% | +2.46 |
| 14 Hz | 1-15 Hz | **+0.1578** | 0.0060 | 5/5 | 97.8% | +2.51 |
| 15 Hz | 1-16 Hz | **+0.1598** | 0.0076 | 5/5 | 98.1% | +2.39 |
| 16 Hz | 1-17 Hz | **+0.1627** | 0.0081 | 5/5 | 98.3% | +2.34 |
| 17 Hz | 1-18 Hz | **+0.1683** | 0.0066 | 5/5 | 98.4% | +2.39 |
| 18 Hz | 1-19 Hz | **+0.1707** | 0.0077 | 5/5 | 98.6% | +2.35 |

Width is NOT comparable down this column — a wider notch removes more signal by construction. The column shows where within each width the model is most vulnerable, and whether that location is stable as the width grows.

> **Edge-truncation warning.** At widths 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 Hz the best band is pinned to the sweep boundary (1 or 45 Hz). The optimum wants to extend outside the searched range, so those band edges are an artefact of where the search stopped and must not be quoted as a result. Only widths whose optimum sits strictly inside the range are interpretable.

## Controlling for how much power the notch removes

EEG is 1/f, so a low band and a high band of equal Hz-width are not equal interventions. `expected` is an isotonic (monotone) fit of dBA on the fraction of 1-45 Hz power removed, fitted **separately within each width**: what a notch OF THAT WIDTH destroying that much power costs, regardless of where it sits. The residual is what the position itself buys. Fitting one curve across all widths instead makes the residual a restatement of the width confound — see the code comment.

The CI column is the patient-paired 95% CI of the RAW dBA, not of the residual — it says the drop is real, not that the residual is.

| band | width | dBA | expected(power) | residual | power removed | seeds | patient-paired 95% CI on dBA |
|---|---|---|---|---|---|---|---|
| 4-9 Hz | 5 | +0.0749 | +0.0518 | **+0.0230** | 10.2% | 5/5 | [+0.0380, +0.1167] |
| 4-10 Hz | 6 | +0.0946 | +0.0757 | **+0.0188** | 11.1% | 5/5 | [+0.0506, +0.1449] |
| 3-10 Hz | 7 | +0.1086 | +0.1014 | **+0.0072** | 18.3% | 5/5 | [+0.0600, +0.1650] |
| 31-39 Hz | 8 | +0.0158 | +0.0109 | **+0.0049** | 0.3% | 5/5 | [+0.0016, +0.0313] |
| 31-40 Hz | 9 | +0.0209 | +0.0137 | **+0.0072** | 0.3% | 5/5 | [+0.0050, +0.0385] |
| 31-41 Hz | 10 | +0.0244 | +0.0180 | **+0.0063** | 0.3% | 5/5 | [+0.0065, +0.0444] |
| 19-30 Hz | 11 | +0.0608 | +0.0551 | **+0.0056** | 0.8% | 5/5 | [+0.0307, +0.0950] |
| 17-29 Hz | 12 | +0.0705 | +0.0652 | **+0.0053** | 1.0% | 5/5 | [+0.0365, +0.1084] |
| 18-31 Hz | 13 | +0.0793 | +0.0753 | **+0.0040** | 1.0% | 5/5 | [+0.0440, +0.1176] |
| 18-32 Hz | 14 | +0.0895 | +0.0842 | **+0.0052** | 1.1% | 5/5 | [+0.0521, +0.1300] |
| 18-33 Hz | 15 | +0.0979 | +0.0934 | **+0.0046** | 1.1% | 5/5 | [+0.0581, +0.1409] |
| 11-27 Hz | 16 | +0.1054 | +0.1018 | **+0.0035** | 3.0% | 5/5 | [+0.0629, +0.1497] |
| 11-28 Hz | 17 | +0.1118 | +0.1091 | **+0.0028** | 3.0% | 5/5 | [+0.0682, +0.1571] |
| 10-28 Hz | 18 | +0.1199 | +0.1164 | **+0.0036** | 3.7% | 5/5 | [+0.0735, +0.1676] |

## Cost per unit of power removed

dBA divided by the fraction of power removed. Large values mark bands the model depends on out of all proportion to their energy — the signature of a feature that is informative but low-power, which in this cohort is more consistent with a recording/contact fingerprint than with a rhythm.

Reported as the best band WITHIN each width, because the ratio also rises with width and a pooled top-10 would simply list the widest bands.

| width | most efficient band | dBA | power removed | dBA per unit power |
|---|---|---|---|---|
| 5 Hz | 23-28 Hz | +0.0109 | 0.31% | 3.52 |
| 6 Hz | 24-30 Hz | +0.0161 | 0.36% | 4.48 |
| 7 Hz | 23-30 Hz | +0.0245 | 0.43% | 5.74 |
| 8 Hz | 22-30 Hz | +0.0334 | 0.50% | 6.67 |
| 9 Hz | 24-33 Hz | +0.0393 | 0.53% | 7.43 |
| 10 Hz | 26-36 Hz | +0.0406 | 0.50% | 8.05 |
| 11 Hz | 25-36 Hz | +0.0491 | 0.57% | 8.66 |
| 12 Hz | 24-36 Hz | +0.0591 | 0.63% | 9.31 |
| 13 Hz | 23-36 Hz | +0.0690 | 0.70% | 9.83 |
| 14 Hz | 24-38 Hz | +0.0730 | 0.69% | 10.51 |
| 15 Hz | 25-40 Hz | +0.0761 | 0.69% | 11.09 |
| 16 Hz | 23-39 Hz | +0.0895 | 0.79% | 11.29 |
| 17 Hz | 24-41 Hz | +0.0913 | 0.79% | 11.63 |
| 18 Hz | 23-41 Hz | +0.1006 | 0.85% | 11.80 |

## Frequency profile, marginalised over width

Mean dBA of every band COVERING each 1 Hz bin, averaged within width first so wide bands do not dominate. This is the profile to report; it has no band edges in it and so cannot be band-fished.

| Hz | mean dBA of covering bands |
|---|---|
| 1-2 | +0.1303 ######################################## |
| 2-3 | +0.1293 ######################################## |
| 3-4 | +0.1286 ####################################### |
| 4-5 | +0.1269 ####################################### |
| 5-6 | +0.1228 ###################################### |
| 6-7 | +0.1173 #################################### |
| 7-8 | +0.1102 ################################## |
| 8-9 | +0.1031 ################################ |
| 9-10 | +0.0961 ############################# |
| 10-11 | +0.0896 ############################ |
| 11-12 | +0.0839 ########################## |
| 12-13 | +0.0789 ######################## |
| 13-14 | +0.0748 ####################### |
| 14-15 | +0.0714 ###################### |
| 15-16 | +0.0686 ##################### |
| 16-17 | +0.0665 #################### |
| 17-18 | +0.0650 #################### |
| 18-19 | +0.0639 #################### |
| 19-20 | +0.0631 ################### |
| 20-21 | +0.0627 ################### |
| 21-22 | +0.0625 ################### |
| 22-23 | +0.0624 ################### |
| 23-24 | +0.0622 ################### |
| 24-25 | +0.0619 ################### |
| 25-26 | +0.0612 ################### |
| 26-27 | +0.0602 ################## |
| 27-28 | +0.0587 ################## |
| 28-29 | +0.0571 ################## |
| 29-30 | +0.0554 ################# |
| 30-31 | +0.0535 ################ |
| 31-32 | +0.0515 ################ |
| 32-33 | +0.0494 ############### |
| 33-34 | +0.0473 ############### |
| 34-35 | +0.0454 ############## |
| 35-36 | +0.0435 ############# |
| 36-37 | +0.0417 ############# |
| 37-38 | +0.0401 ############ |
| 38-39 | +0.0385 ############ |
| 39-40 | +0.0368 ########### |
| 40-41 | +0.0352 ########### |
| 41-42 | +0.0337 ########## |
| 42-43 | +0.0323 ########## |
| 43-44 | +0.0311 ########## |
| 44-45 | +0.0298 ######### |

## Caveats carried by every number above

* Band edges are a post-hoc interpretive lens; the model never saw a band definition (XAI_PLAN.md 9.1).
* dBA rises with width for trivial reasons. Compare bands only WITHIN a width, and against power_removed_frac.
* 469 bands were tested; the largest dBA is a maximum over that many draws. Report the profile, not the argmax.
* 0.832 here vs 0.435 leak-free; recid=0.920. Findings are Tier E candidates, not biomarkers.

* Seed spread of the baseline is 0.033 (0.807-0.840); any dBA of comparable size is not separable from which seed was trained.