# Multi-width band-stop sweep — pooled_tasksplit_chanz_matched

Experiment: **pooled_tasksplit_chanz_matched**, direction `tea_to_toast`, arm `full_ft`. 469 bands, widths 5-18 Hz (step 1), positions 1-45 Hz at 1 Hz stride, scored on 3058 windows / 27 patients across 5 seeds.

Baseline patient-mean BA per seed: 1=0.7031, 2=0.6976, 3=0.7101, 4=0.7205, 5=0.7177 (mean 0.7098).

## Continuity with the 2 Hz sweep

The superseded 2 Hz sweep (seed 1, same filters, same frame) had mean |dBA| 0.0052 and a maximum of 0.0176. This sweep's widths reach the values below, which is the point: the 2 Hz null was a statement about 2 Hz, not about the model.

## Best band at each width (within-width comparison — the only valid one)

| width | best band | dBA | seed sd | seeds agreeing | power removed | within-width z |
|---|---|---|---|---|---|---|
| 5 Hz | 23-28 Hz | **+0.0056** | 0.0035 | 5/5 | 0.4% | +1.36 |
| 6 Hz | 26-32 Hz | **+0.0085** | 0.0037 | 5/5 | 0.5% | +1.54 |
| 7 Hz | 21-28 Hz | **+0.0115** | 0.0034 | 5/5 | 0.6% | +1.31 |
| 8 Hz | 1-9 Hz | **+0.0201** | 0.0077 | 5/5 | 93.6% | +2.28 |
| 9 Hz | 1-10 Hz | **+0.0224** | 0.0089 | 5/5 | 94.6% | +1.98 |
| 10 Hz | 1-11 Hz | **+0.0260** | 0.0080 | 5/5 | 95.4% | +1.70 |
| 11 Hz | 1-12 Hz | **+0.0296** | 0.0090 | 5/5 | 96.1% | +1.49 |
| 12 Hz | 21-33 Hz | **+0.0338** | 0.0083 | 5/5 | 0.9% | +1.46 |
| 13 Hz | 20-33 Hz | **+0.0397** | 0.0066 | 5/5 | 1.1% | +1.58 |
| 14 Hz | 19-33 Hz | **+0.0449** | 0.0067 | 5/5 | 1.2% | +1.53 |
| 15 Hz | 18-33 Hz | **+0.0500** | 0.0080 | 5/5 | 1.4% | +1.48 |
| 16 Hz | 20-36 Hz | **+0.0542** | 0.0071 | 5/5 | 1.2% | +1.39 |
| 17 Hz | 21-38 Hz | **+0.0588** | 0.0088 | 5/5 | 1.1% | +1.37 |
| 18 Hz | 21-39 Hz | **+0.0631** | 0.0087 | 5/5 | 1.2% | +1.33 |

Width is NOT comparable down this column — a wider notch removes more signal by construction. The column shows where within each width the model is most vulnerable, and whether that location is stable as the width grows.

> **Edge-truncation warning.** At widths 8, 9, 10, 11 Hz the best band is pinned to the sweep boundary (1 or 45 Hz). The optimum wants to extend outside the searched range, so those band edges are an artefact of where the search stopped and must not be quoted as a result. Only widths whose optimum sits strictly inside the range are interpretable.

## Controlling for how much power the notch removes

EEG is 1/f, so a low band and a high band of equal Hz-width are not equal interventions. `expected` is an isotonic (monotone) fit of dBA on the fraction of 1-45 Hz power removed, fitted **separately within each width**: what a notch OF THAT WIDTH destroying that much power costs, regardless of where it sits. The residual is what the position itself buys. Fitting one curve across all widths instead makes the residual a restatement of the width confound — see the code comment.

The CI column is the patient-paired 95% CI of the RAW dBA, not of the residual — it says the drop is real, not that the residual is.

| band | width | dBA | expected(power) | residual | power removed | seeds | patient-paired 95% CI on dBA |
|---|---|---|---|---|---|---|---|
| 23-28 Hz | 5 | +0.0056 | +0.0021 | **+0.0036** | 0.4% | 5/5 | [+0.0005, +0.0108] |
| 26-32 Hz | 6 | +0.0085 | +0.0038 | **+0.0047** | 0.5% | 5/5 | [+0.0020, +0.0154] |
| 21-28 Hz | 7 | +0.0115 | +0.0062 | **+0.0053** | 0.6% | 5/5 | [+0.0034, +0.0199] |
| 26-34 Hz | 8 | +0.0142 | +0.0089 | **+0.0053** | 0.5% | 5/5 | [+0.0044, +0.0247] |
| 21-30 Hz | 9 | +0.0184 | +0.0122 | **+0.0062** | 0.7% | 5/5 | [+0.0064, +0.0318] |
| 20-30 Hz | 10 | +0.0237 | +0.0159 | **+0.0077** | 0.9% | 5/5 | [+0.0102, +0.0395] |
| 19-30 Hz | 11 | +0.0292 | +0.0201 | **+0.0091** | 1.0% | 5/5 | [+0.0144, +0.0475] |
| 21-33 Hz | 12 | +0.0338 | +0.0243 | **+0.0095** | 0.9% | 5/5 | [+0.0154, +0.0561] |
| 20-33 Hz | 13 | +0.0397 | +0.0286 | **+0.0111** | 1.1% | 5/5 | [+0.0197, +0.0644] |
| 19-33 Hz | 14 | +0.0449 | +0.0334 | **+0.0114** | 1.2% | 5/5 | [+0.0228, +0.0718] |
| 18-33 Hz | 15 | +0.0500 | +0.0382 | **+0.0118** | 1.4% | 5/5 | [+0.0259, +0.0786] |
| 20-36 Hz | 16 | +0.0542 | +0.0427 | **+0.0116** | 1.2% | 5/5 | [+0.0292, +0.0840] |
| 21-38 Hz | 17 | +0.0588 | +0.0469 | **+0.0119** | 1.1% | 5/5 | [+0.0331, +0.0898] |
| 21-39 Hz | 18 | +0.0631 | +0.0507 | **+0.0124** | 1.2% | 5/5 | [+0.0357, +0.0954] |

## Cost per unit of power removed

dBA divided by the fraction of power removed. Large values mark bands the model depends on out of all proportion to their energy — the signature of a feature that is informative but low-power, which in this cohort is more consistent with a recording/contact fingerprint than with a rhythm.

Reported as the best band WITHIN each width, because the ratio also rises with width and a pooled top-10 would simply list the widest bands.

| width | most efficient band | dBA | power removed | dBA per unit power |
|---|---|---|---|---|
| 5 Hz | 31-36 Hz | +0.0048 | 0.21% | 2.27 |
| 6 Hz | 31-37 Hz | +0.0060 | 0.25% | 2.38 |
| 7 Hz | 32-39 Hz | +0.0083 | 0.27% | 3.06 |
| 8 Hz | 31-39 Hz | +0.0106 | 0.32% | 3.32 |
| 9 Hz | 32-41 Hz | +0.0119 | 0.34% | 3.45 |
| 10 Hz | 31-41 Hz | +0.0142 | 0.39% | 3.61 |
| 11 Hz | 32-43 Hz | +0.0154 | 0.42% | 3.69 |
| 12 Hz | 31-43 Hz | +0.0194 | 0.47% | 4.15 |
| 13 Hz | 31-44 Hz | +0.0222 | 0.51% | 4.37 |
| 14 Hz | 24-38 Hz | +0.0403 | 0.84% | 4.78 |
| 15 Hz | 24-39 Hz | +0.0442 | 0.88% | 5.03 |
| 16 Hz | 24-40 Hz | +0.0476 | 0.92% | 5.20 |
| 17 Hz | 23-40 Hz | +0.0554 | 1.01% | 5.51 |
| 18 Hz | 23-41 Hz | +0.0590 | 1.04% | 5.66 |

## Frequency profile, marginalised over width

Mean dBA of every band COVERING each 1 Hz bin, averaged within width first so wide bands do not dominate. This is the profile to report; it has no band edges in it and so cannot be band-fished.

| Hz | mean dBA of covering bands |
|---|---|
| 1-2 | +0.0275 ##################################### |
| 2-3 | +0.0238 ################################ |
| 3-4 | +0.0211 ############################# |
| 4-5 | +0.0199 ########################### |
| 5-6 | +0.0187 ######################### |
| 6-7 | +0.0175 ######################## |
| 7-8 | +0.0171 ####################### |
| 8-9 | +0.0169 ####################### |
| 9-10 | +0.0168 ####################### |
| 10-11 | +0.0167 ####################### |
| 11-12 | +0.0167 ####################### |
| 12-13 | +0.0170 ####################### |
| 13-14 | +0.0174 ######################## |
| 14-15 | +0.0179 ######################## |
| 15-16 | +0.0187 ######################### |
| 16-17 | +0.0197 ########################### |
| 17-18 | +0.0210 ############################# |
| 18-19 | +0.0224 ############################### |
| 19-20 | +0.0239 ################################# |
| 20-21 | +0.0253 ################################## |
| 21-22 | +0.0266 #################################### |
| 22-23 | +0.0277 ###################################### |
| 23-24 | +0.0286 ####################################### |
| 24-25 | +0.0291 ######################################## |
| 25-26 | +0.0293 ######################################## |
| 26-27 | +0.0294 ######################################## |
| 27-28 | +0.0292 ######################################## |
| 28-29 | +0.0289 ####################################### |
| 29-30 | +0.0285 ####################################### |
| 30-31 | +0.0280 ###################################### |
| 31-32 | +0.0275 ###################################### |
| 32-33 | +0.0270 ##################################### |
| 33-34 | +0.0263 #################################### |
| 34-35 | +0.0256 ################################### |
| 35-36 | +0.0249 ################################## |
| 36-37 | +0.0241 ################################# |
| 37-38 | +0.0233 ################################ |
| 38-39 | +0.0225 ############################### |
| 39-40 | +0.0217 ############################## |
| 40-41 | +0.0209 ############################ |
| 41-42 | +0.0201 ########################### |
| 42-43 | +0.0194 ########################## |
| 43-44 | +0.0187 ######################### |
| 44-45 | +0.0177 ######################## |

## Caveats carried by every number above

* Band edges are a post-hoc interpretive lens; the model never saw a band definition (XAI_PLAN.md 9.1).
* dBA rises with width for trivial reasons. Compare bands only WITHIN a width, and against power_removed_frac.
* 469 bands were tested; the largest dBA is a maximum over that many draws. Report the profile, not the argmax.
* 0.832 here vs 0.435 leak-free; recid=0.920. Findings are Tier E candidates, not biomarkers.

* Seed spread of the baseline is 0.023 (0.698-0.720); any dBA of comparable size is not separable from which seed was trained.