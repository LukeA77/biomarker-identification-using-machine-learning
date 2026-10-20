"""Splits, loaders and statistics for the WITHIN-PATIENT TASK-SPLIT experiment.

The design (supervisor-suggested, 2026-07-30): inside ONE patient, train on the
windows of one ADL task and test on the windows of the other, using that
patient's existing session-level ON/OFF labels. Both directions, then averaged.

How this differs structurally from the LOPO arms, and why it needs its own code:

* **The split is on TASK, not on patient.** ``make_patient_loader`` filters the
  manifest by ``patient_id``; here train and test come from the SAME patient and
  are separated by ``task_group`` (from the sidecar derived by
  ``task_windows.py``). So loaders are built from an explicit frame slice.
* **There are no held-out patients to validate on.** The LOPO inner
  GroupKFold-on-patient scheme has nothing to group. Early stopping instead uses
  a contiguous, end-of-session block of the TRAIN task (never the test task) —
  see :func:`train_val_split_by_time`.
* **Evaluation is window-level within a patient.** A patient contributes only two
  sessions, so a session-level balanced accuracy could only be 0, 0.5 or 1. The
  per-patient primary metric is therefore balanced accuracy over the held-out
  task's non-overlap windows; the session-level scores are kept for the
  threshold-free ordering test.
* **Patient is the unit of analysis.** :func:`aggregate_across_patients` treats
  each patient as one observation (paired/random-effect style) rather than
  pooling windows, because per-patient N is small and unequal and window pooling
  would both over-weight long recordings and understate between-patient variance.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from livinglab_codebrain.config import DataConfig
from livinglab_codebrain.dataset import WindowDataset
from livinglab_codebrain.loader import make_dataloader

log = logging.getLogger(__name__)

__all__ = [
    "select_task_frame", "train_val_split_by_time", "make_frame_loader",
    "eval_duration_match", "duration_match", "match_report", "direction_pairs",
    "MATCH_LEVELS", "MATCH_WHERE",
    "aggregate_across_patients", "ordering_stats",
]

MATCH_LEVELS = ("patient_task", "task", "global")
MATCH_WHERE = ("first", "middle", "last", "uniform")


def direction_pairs(groups: tuple[str, ...], directions: list[str]) -> list[tuple[str, str]]:
    """Turn config direction names (``"tea_to_toast"``) into ``(train, test)`` pairs."""
    out = []
    for d in directions:
        if "_to_" not in d:
            raise ValueError(f"direction {d!r} must be '<train>_to_<test>'")
        a, b = d.split("_to_", 1)
        if a not in groups or b not in groups:
            raise ValueError(f"direction {d!r} names unknown task group(s); known: {groups}")
        if a == b:
            raise ValueError(f"direction {d!r} trains and tests on the same task group")
        out.append((a, b))
    return out


def select_task_frame(
    sidecar: pd.DataFrame,
    patient: str,
    task_group: str,
    *,
    nonoverlap_only: bool,
) -> pd.DataFrame:
    """One patient's windows for one task group (both of that patient's sessions).

    Fails loudly rather than returning a degenerate slice: a missing condition
    would make balanced accuracy undefined, and a duplicated window path would
    mean the task spans overlap (which the pre-flight is supposed to have caught).
    """
    df = sidecar[(sidecar["patient_id"] == patient) & (sidecar["task_group"] == task_group)]
    if nonoverlap_only:
        df = df[df["is_nonoverlap_subset"]]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{patient}/{task_group}: no windows (nonoverlap_only={nonoverlap_only})")
    conds = set(df["condition"].unique())
    if conds != {"A", "C"}:
        raise ValueError(
            f"{patient}/{task_group}: conditions present = {sorted(conds)}, need both A and C "
            f"(balanced accuracy is undefined otherwise) — the pre-flight gate should have "
            f"excluded this patient")
    if df["path"].duplicated().any():
        raise ValueError(
            f"{patient}/{task_group}: duplicated window paths — task spans overlap; "
            f"the pre-flight disjointness gate should have excluded this patient")
    return df


def train_val_split_by_time(
    frame: pd.DataFrame,
    *,
    val_frac: float,
    stratify_by_session: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the TRAIN-task windows into (train, val) by a contiguous time block.

    The validation block is the LAST ``val_frac`` of the task, taken per SESSION
    (``patient_id`` x ``condition``) when ``stratify_by_session`` so both ON and
    OFF are represented in val. Grouping by the real session rather than by
    ``condition`` alone matters once ``frame`` can span multiple patients (the
    pooled-training design): ``t_cont`` is elapsed time WITHIN a recording and
    resets per session, so sorting condition-A windows from many different
    patients together by raw ``t_cont`` would interleave unrelated sessions'
    timelines. Grouping by session first keeps every patient's own chronology
    intact; for a single-patient frame this reduces to exactly the same
    per-condition split as before (a patient contributes only one A and one C
    session), so existing per-patient callers are unaffected.

    Contiguous-by-time rather than random: windows overlap (5 s stride on 10 s
    windows), so a random holdout would put near-duplicate neighbours on both
    sides of the split and make the early-stopping signal optimistic. A trailing
    block keeps val temporally separated from most of train.

    Note this is still same-task, same-session data and is used ONLY to pick the
    stopping epoch — never to report a number. The reported test task is wholly
    disjoint from both train and val.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    # Group per real session when stratifying, else treat the whole frame as one block.
    grouper = "session" if stratify_by_session else np.zeros(len(frame), dtype=int)
    tr_parts, va_parts = [], []
    for _, grp in frame.groupby(grouper):
        grp = grp.sort_values("t_cont")
        n_val = max(1, int(round(len(grp) * val_frac)))
        if len(grp) - n_val < 1:                     # keep at least one train window
            n_val = len(grp) - 1
        if n_val < 1:
            raise ValueError(
                f"cannot carve a validation block from {len(grp)} window(s) in one session")
        tr_parts.append(grp.iloc[: len(grp) - n_val])
        va_parts.append(grp.iloc[len(grp) - n_val:])
    train = pd.concat(tr_parts).reset_index(drop=True)
    val = pd.concat(va_parts).reset_index(drop=True)
    return train, val


def make_frame_loader(
    frame: pd.DataFrame,
    data_cfg: DataConfig,
    *,
    shuffle: bool,
    seed: int,
    input_norm: str = "none",
    norm_scale: float | None = None,
) -> DataLoader:
    """Loader over an explicit manifest/sidecar slice.

    Reuses the audited ``WindowDataset`` (reshape + /100) and ``make_dataloader``
    (singleton-safe batching — SSSM's trailing ``squeeze()`` would corrupt a
    size-1 batch), so the window contract is inherited unchanged.

    ``input_norm`` matters more here than anywhere else in the project. Within a
    patient the ON/OFF label is perfectly collinear with recording identity, and
    the two recordings carry a strong per-session amplitude fingerprint —
    measured, not hypothesised: three crude non-neural features (mean|x|, std,
    max|x|) reach 0.599 mean per-patient balanced accuracy on this very split,
    amplitude alone 0.573, with AUC 1.000 for several patients. ``global_z``
    removes per-window amplitude while preserving topography, so a
    ``global_z`` re-run quantifies how much of any result is that fingerprint.
    """
    if len(frame) < data_cfg.loader.min_batch_size:
        raise ValueError(
            f"frame has {len(frame)} windows but min_batch_size is "
            f"{data_cfg.loader.min_batch_size}; cannot form a valid batch")
    ds = WindowDataset(frame, data_cfg.paths.cache_dir, data_cfg.window)
    if input_norm != "none":
        if norm_scale is None:
            raise ValueError("norm_scale is required when input_norm != 'none'")
        from .data import NormalizedWindowDataset
        ds = NormalizedWindowDataset(ds, input_norm, norm_scale)
    return make_dataloader(ds, data_cfg, shuffle=shuffle, seed=seed)


def _take_k(grp: pd.DataFrame, k: int, where: str, label: str) -> pd.DataFrame:
    """Pick ``k`` of a session's windows, ordered by elapsed time within its recording."""
    grp = grp.sort_values("t_cont")
    n = len(grp)
    if n < k:
        raise ValueError(
            f"{label} has {n} windows < K={k}; lower K (the binding cell governs) "
            f"rather than dropping the session")
    if where == "first":
        return grp.iloc[:k]
    if where == "last":
        return grp.iloc[-k:]
    if where == "middle":
        s = (n - k) // 2
        return grp.iloc[s: s + k]
    # uniform: evenly spaced over the WHOLE task span. Spacing is (n-1)/(k-1) >= 1
    # whenever k <= n, so the rounded indices are strictly increasing (no collisions).
    idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    return grp.iloc[idx]


def duration_match(
    frame: pd.DataFrame,
    *,
    level: str = "patient_task",
    k: int | None = None,
    where: str = "uniform",
) -> pd.DataFrame:
    """Force an equal number of windows per SESSION, so window count cannot encode ON/OFF.

    The confound this closes (supervisor, 2026-08-06): a patient's ON and OFF
    recordings contain different numbers of windows for the same task, because the
    task took a different amount of time. Measured over the 27 eligible patients
    (``logs/window_count_distribution_20260806.md``) the median gap is 17% of the
    longer session for tea and 13% for toast, and only 4/54 patient x task pairs
    are already equal. The gap has NO consistent direction — ON is the longer
    session more often (17/27 tea, 15/27 toast), both non-significant — so it is
    an idiosyncratic per-patient cue rather than a cohort-level one, exploitable
    only by a model that can condition on patient identity. CodeBrain's
    representations can (EXPERIMENTS.md 4b/11), hence this.

    ``level`` sets what K is shared across:
      * ``patient_task`` — K = min(ON, OFF) for that patient and task. The minimal
        intervention: it equalises exactly the ON/OFF pair carrying the confound
        and nothing else, keeping ~90% of windows and all 27 patients.
      * ``task``   — one K per task group, the cohort floor (tea 15, toast 26 eval
        windows). Fully balanced across patients too, at ~41-53% retention.
      * ``global`` — one K everywhere (15). ~33% retention.
    An explicit ``k`` overrides ``level`` and is applied to every session.

    ``where`` sets WHICH windows survive. **Use ``uniform``.** The contiguous
    options are kept for backwards compatibility but interact badly with
    ``patient_task``: under K = min(ON, OFF) the shorter session keeps everything
    including task onset and offset, while the longer session keeps only its
    centre, so "does this window contain a task transition" becomes a proxy for
    "is this the shorter session" — which is a proxy for the label in that
    patient. That trades a count cue, invisible to a window-level classifier, for
    a task-phase cue that is plainly visible inside a single window. ``uniform``
    spreads the K survivors evenly across each session's own task span, so both
    conditions are sampled at matched RELATIVE phase and no such proxy appears.
    """
    if where not in MATCH_WHERE:
        raise ValueError(f"where must be one of {MATCH_WHERE}, got {where!r}")
    if k is None and level not in MATCH_LEVELS:
        raise ValueError(f"level must be one of {MATCH_LEVELS}, got {level!r}")
    for col in ("patient_id", "task_group", "condition", "t_cont"):
        if col not in frame.columns:
            raise ValueError(f"duration_match needs column {col!r}")

    # Session = patient x condition, ALWAYS grouped with task_group. Grouping by
    # `condition` alone (the old eval_duration_match behaviour) is only correct on
    # a single-patient, single-task frame; on the pooled frames it would take the
    # middle K of the entire 27-patient pool per condition. Same trap documented
    # in train_val_split_by_time.
    counts = frame.groupby(["patient_id", "task_group", "condition"]).size()
    cells = counts.index.droplevel("condition").unique()

    if k is not None:
        kmap = {c: int(k) for c in cells}
    elif level == "patient_task":
        kmap = {c: int(v) for c, v in counts.groupby(level=["patient_id", "task_group"]).min().items()}
    elif level == "task":
        per_task = counts.groupby(level="task_group").min()
        kmap = {c: int(per_task[c[1]]) for c in cells}
    else:
        kmap = {c: int(counts.min()) for c in cells}

    parts = []
    for (pid, task, cond), grp in frame.groupby(["patient_id", "task_group", "condition"], sort=True):
        parts.append(_take_k(grp, kmap[(pid, task)], where, f"{pid}/{task}/{cond}"))
    out = pd.concat(parts).reset_index(drop=True)

    # Post-conditions: the whole point is that count no longer differs by condition.
    after = out.groupby(["patient_id", "task_group", "condition"]).size()
    spread = after.groupby(level=["patient_id", "task_group"]).agg(lambda s: s.max() - s.min())
    if (spread != 0).any():
        bad = spread[spread != 0].to_dict()
        raise RuntimeError(f"duration_match failed to equalise sessions: {bad}")
    if out["path"].duplicated().any():
        raise RuntimeError("duration_match produced duplicated window paths")

    log.info("duration_match(level=%s, k=%s, where=%s): %d -> %d windows (%.1f%%), "
             "K in [%d, %d] over %d patient x task cells",
             level if k is None else "explicit", k, where, len(frame), len(out),
             100.0 * len(out) / max(len(frame), 1),
             min(kmap.values()), max(kmap.values()), len(cells))
    return out


def match_report(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Machine-readable record of what a duration_match call did, for the run JSON."""
    kmap = after.groupby(["patient_id", "task_group", "condition"]).size()
    per_cell = kmap.groupby(level=["patient_id", "task_group"]).min()
    return {
        "n_windows_before": int(len(before)),
        "n_windows_after": int(len(after)),
        "retained_frac": float(len(after) / max(len(before), 1)),
        "k_min": int(per_cell.min()), "k_max": int(per_cell.max()),
        "k_median": float(per_cell.median()),
        "k_per_cell": {f"{p}|{t}": int(v) for (p, t), v in per_cell.items()},
        "n_patients": int(after["patient_id"].nunique()),
    }


def eval_duration_match(frame: pd.DataFrame, *, k: int, where: str = "middle") -> pd.DataFrame:
    """Back-compatible wrapper: exactly ``k`` windows per session, contiguous by default.

    Retained so ``run_within_patient_tasksplit.py``'s ``--match-k`` flag and the
    stage4g numbers stay reproducible. New code should call :func:`duration_match`
    with ``level="patient_task", where="uniform"`` — see that docstring for why the
    contiguous variants are a poor fit for a min(ON, OFF) rule.
    """
    if where not in {"first", "middle", "last"}:
        raise ValueError(f"where must be first/middle/last, got {where!r}")
    return duration_match(frame, k=int(k), where=where)


# ---------------------------------------------------------------------------
# statistics — patient as the unit of analysis
# ---------------------------------------------------------------------------
def aggregate_across_patients(
    per_patient: dict[str, float],
    *,
    n_resamples: int,
    ci: float,
    chance: float,
    seed: int,
    wilcoxon: bool = True,
    sign_test: bool = True,
) -> dict:
    """Pool per-patient balanced accuracies treating PATIENT as the observation.

    Deliberately not a naive window pool. Returns the mean, a patient-level
    bootstrap CI, and paired tests of the per-patient values against ``chance``:

    * **Wilcoxon signed-rank** — paired, distribution-free, the standard choice
      at N~20; assumes a symmetric difference distribution.
    * **Exact sign test** — makes no symmetry assumption, so it is reported
      alongside rather than instead. Ties (exactly at chance) are dropped, which
      is the conventional handling and is recorded in ``n_ties``.
    """
    patients = sorted(per_patient)
    vals = np.array([per_patient[p] for p in patients], dtype=float)
    finite = np.isfinite(vals)
    if not finite.all():
        log.warning("Dropping %d patient(s) with non-finite balanced accuracy from the pooled "
                    "statistics: %s", int((~finite).sum()),
                    [p for p, f in zip(patients, finite) if not f])
    vals = vals[finite]
    patients = [p for p, f in zip(patients, finite) if f]
    n = len(vals)
    if n == 0:
        raise ValueError("no patients with a finite balanced accuracy — nothing to pool")

    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, size=n, replace=True).mean() for _ in range(n_resamples)])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    out = {
        "n_patients": n,
        "patients": patients,
        "per_patient": {p: float(v) for p, v in zip(patients, vals)},
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if n > 1 else None,
        "ci_low": float(np.percentile(boot, lo)),
        "ci_high": float(np.percentile(boot, hi)),
        "ci": ci,
        "n_resamples": n_resamples,
        "chance": chance,
        "n_above_chance": int((vals > chance).sum()),
        "n_below_chance": int((vals < chance).sum()),
        "n_ties": int((vals == chance).sum()),
    }

    if wilcoxon:
        from scipy.stats import wilcoxon as _w
        d = vals - chance
        nz = d[d != 0]
        if len(nz) >= 1:
            stat, p = _w(nz, alternative="greater")
            out["wilcoxon"] = {"statistic": float(stat), "p_value": float(p),
                              "alternative": "greater", "n_nonzero": int(len(nz))}
        else:
            out["wilcoxon"] = {"statistic": None, "p_value": None,
                               "note": "every patient exactly at chance"}
    if sign_test:
        from scipy.stats import binomtest
        k = int((vals > chance).sum())
        m = int((vals != chance).sum())
        out["sign_test"] = (
            {"n_above": k, "n_effective": m,
             "p_value": float(binomtest(k, m, 0.5, alternative="greater").pvalue)}
            if m > 0 else {"n_above": k, "n_effective": 0, "p_value": None})
    return out


def ordering_stats(records: list[dict], *, seed: int = 0) -> dict:
    """Threshold-free readout: did the OFF session score more OFF-like than ON?

    One binary outcome per (patient, direction): ``mean P(OFF) | OFF session >
    mean P(OFF) | ON session``. Immune to the calibration failure that dominated
    the LOPO result (EXPERIMENTS.md §4a: many patients had BOTH sessions on the
    same side of the 0.5 threshold, so ordering was informative where thresholded
    accuracy was not).

    Reported at two levels: per (patient, direction) via an exact binomial, and
    per patient (both directions correct / split / neither).
    """
    from scipy.stats import binomtest

    flags = [bool(r["ordering_correct"]) for r in records if r.get("ordering_correct") is not None]
    n, k = len(flags), int(sum(flags))
    by_patient: dict[str, list[bool]] = {}
    for r in records:
        if r.get("ordering_correct") is not None:
            by_patient.setdefault(r["patient"], []).append(bool(r["ordering_correct"]))
    per_patient_score = {p: float(np.mean(v)) for p, v in by_patient.items()}
    both = sum(1 for v in by_patient.values() if all(v))
    none = sum(1 for v in by_patient.values() if not any(v))
    return {
        "n_direction_level": n,
        "n_correct_direction_level": k,
        "frac_correct": (k / n) if n else None,
        "binomial_p": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
        "per_patient_frac": per_patient_score,
        "n_patients_both_directions_correct": both,
        "n_patients_neither_direction_correct": none,
        "n_patients": len(by_patient),
    }
