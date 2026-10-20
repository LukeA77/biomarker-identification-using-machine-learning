# Within-patient task-split — PRE-FLIGHT report

Generated 2026-07-30 13:35:21 | drive `C:\Data\LivingLab_PD`

Task groups: **tea** = ['Kettel Task'], **toast** = ['Toast Task', 'Toast Task1', 'Toast Task2']

## Verdict: **27 / 35** cohort patients are eligible

### The confound question the supervisor asked

The design is only valid if each task type occurs in BOTH the ON and the OFF
session — otherwise task type is confounded with medication state and the split
just relabels the temporal/duration confound. Two levels were checked:

* **Protocol level** (was the task administered at all — the union of fitting and
  non-fitting rows in the session's annotation CSV);
* **Recoverable level** (does the task actually land inside the recording once the
  two clocks are reconciled, and does it contain eval windows).

### Per-session derivation

| session | shift (s) | tasks fit | tea win/eval | toast win/eval | status |
|---|---|---|---|---|---|
| sub-34-a | -6.00 | 5/7 | 115/58 | 189/95 | OK |
| sub-34-c | -3574.00 | 1/6 | 0/0 | 0/0 | no recoverable tea/toast windows |
| sub-42-a | 0.00 | 5/7 | 36/18 | 97/48 | OK |
| sub-42-c | -1584.00 | 4/7 | 47/23 | 0/0 | no recoverable toast windows |
| sub-48-a | 730.00 | 4/7 | 74/37 | 0/0 | no recoverable toast windows |
| sub-48-c | 1916.00 | 4/5 | 80/40 | 0/0 | no recoverable toast windows |
| sub-51-a | -912.00 | 6/6 | 55/27 | 116/58 | OK |
| sub-51-c | 0.00 | 5/5 | 53/27 | 85/42 | OK |
| sub-53-a | 0.00 | 5/7 | 62/31 | 91/45 | OK |
| sub-53-c | 0.00 | 4/4 | 69/35 | 69/35 | TASK SPANS OVERLAP by 354.7s (tea vs toast) — the annotation CSV logs these as the same/overlapping interval, so a 'task split' here would train and test on the same windows |
| sub-62-a | -3600.00 | 5/7 | 67/34 | 98/49 | OK |
| sub-62-c | -3600.00 | 5/5 | 43/21 | 71/35 | OK |
| sub-69-a | -4360.00 | 6/7 | 104/52 | 0/0 | no recoverable toast windows |
| sub-69-c | -3600.00 | 5/5 | 56/28 | 134/67 | OK |
| sub-70-a | -5706.00 | 7/7 | 61/31 | 84/42 | OK |
| sub-70-c | -3730.00 | 5/7 | 60/30 | 79/40 | OK |
| sub-71-a | -3600.00 | 5/7 | 60/30 | 127/63 | OK |
| sub-71-c | -3600.00 | 4/4 | 60/30 | 116/58 | OK |
| sub-72-a | -3600.00 | 5/7 | 55/27 | 98/49 | OK |
| sub-72-c | -3600.00 | 5/5 | 39/20 | 97/49 | OK |
| sub-73-a | -3600.00 | 5/7 | 61/30 | 128/64 | OK |
| sub-73-c | -4576.00 | 5/6 | 53/27 | 0/0 | no recoverable toast windows |
| sub-76-a | -3600.00 | 5/7 | 82/41 | 143/72 | OK |
| sub-76-c | -3600.00 | 5/5 | 59/30 | 139/70 | OK |
| sub-77-a | -3600.00 | 5/7 | 63/31 | 143/72 | OK |
| sub-77-c | -3600.00 | 5/5 | 51/26 | 137/68 | OK |
| sub-81-a | 0.00 | 7/9 | 71/35 | 146/73 | OK |
| sub-81-c | 0.00 | 7/7 | 52/26 | 133/67 | OK |
| sub-84-a | 0.00 | 7/9 | 50/25 | 141/71 | OK |
| sub-84-c | 0.00 | 7/7 | 56/28 | 128/64 | OK |
| sub-90-a | 928.00 | 6/9 | 50/25 | 51/26 | OK |
| sub-90-c | 0.00 | 7/7 | 43/21 | 92/46 | OK |
| sub-92-a | 0.00 | 7/9 | 45/23 | 94/48 | OK |
| sub-92-c | 0.00 | 7/7 | 48/24 | 77/38 | OK |
| sub-93-a | 0.00 | 7/9 | 18/9 | 85/43 | OK |
| sub-93-c | 0.00 | 6/6 | 0/0 | 0/0 | no recoverable tea/toast windows |
| sub-94-a | 0.00 | 7/9 | 42/21 | 119/60 | OK |
| sub-94-c | 0.00 | 7/7 | 36/18 | 75/37 | OK |
| sub-96-a | 0.00 | 7/9 | 53/26 | 60/30 | OK |
| sub-96-c | 0.00 | 7/7 | 58/29 | 59/30 | OK |
| sub-103-a | -3600.00 | 6/8 | 119/60 | 298/149 | OK |
| sub-103-c | -3600.00 | 7/7 | 117/59 | 300/150 | OK |
| sub-106-a | -3600.00 | 7/9 | 54/27 | 119/59 | OK |
| sub-106-c | -3600.00 | 7/7 | 43/21 | 83/41 | OK |
| sub-107-a | -3600.00 | 7/9 | 54/27 | 126/63 | OK |
| sub-107-c | -3600.00 | 6/7 | 44/22 | 66/33 | OK |
| sub-114-a | 0.00 | 7/7 | 60/30 | 147/73 | OK |
| sub-114-c | -2056.00 | 9/9 | 54/27 | 153/76 | OK |
| sub-115-a | 0.00 | 6/9 | 47/23 | 87/44 | OK |
| sub-115-c | 0.00 | 7/7 | 43/22 | 141/70 | OK |
| sub-116-a | 0.00 | 7/7 | 68/34 | 144/72 | OK |
| sub-116-c | 0.00 | 6/6 | 53/26 | 94/47 | OK |
| sub-117-a | 0.00 | 7/9 | 60/30 | 153/77 | OK |
| sub-117-c | -198.00 | 3/3 | 0/0 | 0/0 | no recoverable tea/toast windows |
| sub-119-a | 0.00 | 7/7 | 52/26 | 153/76 | OK |
| sub-119-c | 0.00 | 7/7 | 98/49 | 159/80 | OK |
| sub-121-a | 724.00 | 7/9 | 55/28 | 60/30 | OK |
| sub-121-c | 0.00 | 7/7 | 103/52 | 124/63 | OK |
| sub-123-a | 0.00 | 7/7 | 73/36 | 162/81 | OK |
| sub-123-c | 0.00 | 7/7 | 54/27 | 86/43 | OK |
| sub-125-a | -3600.00 | 7/9 | 47/23 | 160/80 | OK |
| sub-125-c | -3600.00 | 7/7 | 50/25 | 119/59 | OK |
| sub-129-a | -3600.00 | 7/7 | 57/28 | 157/78 | OK |
| sub-129-c | -3600.00 | 7/7 | 73/36 | 181/90 | OK |
| sub-131-a | -3600.00 | 8/10 | 42/21 | 161/80 | OK |
| sub-131-c | -3600.00 | 8/8 | 31/15 | 172/86 | OK |
| sub-134-a | 0.00 | 7/10 | 59/29 | 131/66 | OK |
| sub-134-c | 0.00 | 7/7 | 48/24 | 133/67 | OK |
| sub-135-a | -3600.00 | 8/8 | 60/30 | 174/86 | OK |
| sub-135-c | -3600.00 | 8/10 | 87/44 | 208/104 | OK |

### Eligible patients

`sub-103`, `sub-106`, `sub-107`, `sub-114`, `sub-115`, `sub-116`, `sub-119`, `sub-121`, `sub-123`, `sub-125`, `sub-129`, `sub-131`, `sub-134`, `sub-135`, `sub-51`, `sub-62`, `sub-70`, `sub-71`, `sub-72`, `sub-76`, `sub-77`, `sub-81`, `sub-84`, `sub-90`, `sub-92`, `sub-94`, `sub-96`

### Ineligible patients (every one listed with its reason — no silent drops)

| patient | reason(s) |
|---|---|
| sub-117 | task group 'tea' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF<br>task group 'toast' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF |
| sub-34 | task group 'tea' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF<br>task group 'toast' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF |
| sub-42 | task group 'toast' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF |
| sub-48 | task group 'toast' has NO recoverable eval windows in session(s) A/C — task type would be confounded with ON/OFF |
| sub-53 | C: TASK SPANS OVERLAP by 354.7s (tea vs toast) — the annotation CSV logs these as the same/overlapping interval, so a 'task split' here would train and test on the same windows |
| sub-69 | task group 'toast' has NO recoverable eval windows in session(s) A — task type would be confounded with ON/OFF |
| sub-73 | task group 'toast' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF |
| sub-93 | task group 'tea' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF<br>task group 'toast' has NO recoverable eval windows in session(s) C — task type would be confounded with ON/OFF |

### Labelled-window totals

| task group | windows | eval (non-overlap) windows | ON | OFF |
|---|---|---|---|---|
| tea | 3992 | 1995 | 2131 | 1861 |
| toast | 7752 | 3877 | 4242 | 3510 |
