"""Read-only audit: raw-signal CSV vs EDF equivalence, and native Trigger/Event
usability, across the FULL LivingLab_PD cohort at C:\\Data\\LivingLab_PD.

This script does NOT modify, convert, or write anything inside the dataset.
It reads files under C:\\Data\\LivingLab_PD only (open(..., 'r')/pyedflib
EdfReader, never 'w'/'a' mode against that tree) and writes its own outputs
(log + JSON/CSV results) only under the repo root.

Investigation questions answered (see RAW_CSV_TRIGGER_AUDIT_REPORT.md):
  a. Per-session inventory of EDF / raw-signal CSV / task-timestamp CSV.
  b. EDF vs raw-signal-CSV equivalence (channels, sample rate, sample count,
     spot-check voltage values).
  c. Native Trigger/Event/Time_Offset/ADC_Sequence usability as task markers
     (full-file scan, not head/tail sampling).
  d. Clock alignment: raw-signal header start-time vs task-CSV Start/End,
     fixed-offset vs drift vs unresolvable, for every subject.
  e. Coverage: recorded duration vs task span.

Column layout of the raw-signal CSV, confirmed by header/value zip on
sub-131-a (see audit log for the transcript):
  Time, <24 electrode channels re: Pz>, Trigger, Time_Offset, ADC_Status,
  ADC_Sequence, Event, Comments
"Pz" itself is always 0 (it is the reference electrode).

Run: python raw_csv_edf_trigger_audit.py [--limit N] [--subjects sub-131-a,...]
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyedflib
except ImportError:
    pyedflib = None

DATA_ROOT = Path(r"C:\Data\LivingLab_PD")
REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/audits/ -> scripts/ -> repo root
LOG_PATH = REPO_ROOT / "logs" / "raw_csv_edf_trigger_audit.log"
RESULTS_JSON = REPO_ROOT / "outputs" / "audits" / "raw_csv_edf_trigger_audit_results.json"
RESULTS_CSV = REPO_ROOT / "outputs" / "audits" / "raw_csv_edf_trigger_audit_sessions.csv"

TASK_SUBFOLDER_CANDIDATES = ["living-lab-tasks", "livinglab"]
KNOWN_EMPTY_SESSIONS = {"sub-100-a", "sub-82-a", "sub-130-c"}
META_COLS = {"Time", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event", "Comments"}
SPOT_CHECK_STRIDE = 200_000  # capture a data row every N rows for CSV<->EDF spot check
BST_TOLERANCE_S = 180.0      # residual allowed when calling an offset "whole-hour"
FIXED_OFFSET_TOLERANCE_S = 5.0  # |offset_last - offset_first| below this => "fixed", not drift

_LOG_FH = None


def log(msg: str, level: str = "INFO") -> None:
    global _LOG_FH
    line = f"[{datetime.now().isoformat(timespec='seconds')}] [{level}] {msg}"
    print(line)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def open_log():
    global _LOG_FH
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(LOG_PATH, "a", encoding="utf-8")
    log("=" * 100)
    log(f"Audit run started. DATA_ROOT={DATA_ROOT}")


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

SESSION_RE = re.compile(r"^sub-(\d+)-([ac])$", re.IGNORECASE)


def discover_sessions(limit=None, subjects_filter=None):
    sessions = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue
        m = SESSION_RE.match(p.name)
        if not m:
            log(f"Skipping top-level entry that doesn't match sub-<id>-<a|c>: {p.name}", "WARN")
            continue
        if subjects_filter and p.name not in subjects_filter:
            continue
        sessions.append(p)
    if limit:
        sessions = sessions[:limit]
    return sessions


def find_task_folder(session_dir: Path):
    eeg = session_dir / "eeg"
    if not eeg.is_dir():
        return None, []
    present = [c.name for c in eeg.iterdir() if c.is_dir()]
    for cand in TASK_SUBFOLDER_CANDIDATES:
        for name in present:
            if name.lower() == cand:
                return eeg / name, present
    return None, present


def sniff_first_line(path: Path) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.readline().strip()
    except Exception as e:
        log(f"Could not sniff first line of {path}: {e}", "WARN")
        return ""


def classify_task_folder_files(task_folder: Path):
    """Return dict of dsi / raw_csv / raw_edf / accel_csv / other file lists."""
    dsi, raw_csv, raw_edf, accel_csv, other = [], [], [], [], []
    for f in sorted(task_folder.iterdir()):
        if f.is_dir():
            other.append(f)
            continue
        low = f.name.lower()
        if low.endswith(".dsi"):
            dsi.append(f)
        elif low.endswith(".edf"):
            raw_edf.append(f)
            if "_raw" not in low:
                log(f"EDF file without '_raw' in name: {f}", "WARN")
        elif low.endswith(".csv"):
            if "accel" in low:
                accel_csv.append(f)
            else:
                first = sniff_first_line(f)
                if first.startswith("#") and "mains_frequency" in first.lower():
                    raw_csv.append(f)
                elif first.lower().startswith("tasks,start,end"):
                    log(f"Task-timestamp-shaped CSV found INSIDE eeg task folder "
                        f"(unexpected location): {f}", "WARN")
                    other.append(f)
                else:
                    log(f"Unclassified CSV in task folder (first line: {first!r}): {f}", "WARN")
                    other.append(f)
        else:
            other.append(f)
    return {"dsi": dsi, "raw_csv": raw_csv, "raw_edf": raw_edf, "accel_csv": accel_csv, "other": other}


def find_annotation_csvs(session_dir: Path):
    ann_dir = session_dir / "annotation"
    if not ann_dir.is_dir():
        return [], []
    task_csvs, rejected = [], []
    for f in sorted(ann_dir.glob("*.csv")):
        first = sniff_first_line(f)
        norm = first.lower().replace(" ", "")
        if norm.startswith("tasks,start,end"):
            task_csvs.append(f)
        else:
            rejected.append((f, first))
            log(f"Rejected non-task-shaped CSV in annotation/ (first line: {first!r}): {f}", "WARN")
    return task_csvs, rejected


# --------------------------------------------------------------------------
# Raw-signal CSV parsing
# --------------------------------------------------------------------------

HEADER_KV_RE = re.compile(r"^#?\s*([^,=]+?)\s*=\s*,\s*([^,]*)")


def parse_raw_csv_header(path: Path, max_header_lines: int = 40):
    """Read the header block up to the 'Time,...' column row. Returns
    (meta_dict, header_cols, header_row_idx) or raises ValueError."""
    meta = {}
    with open(path, "r", errors="replace") as f:
        for i in range(max_header_lines):
            line = f.readline()
            if not line:
                raise ValueError(f"EOF before finding 'Time,' header row in {path}")
            if line.startswith("Time,"):
                cols = line.rstrip("\n").rstrip(",").split(",")
                # keep trailing empty (Comments) if present by re-splitting w/o rstrip of comma
                cols = line.rstrip("\n").split(",")
                return meta, cols, i
            m = HEADER_KV_RE.match(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                meta[key] = val
    raise ValueError(f"Header block exceeded {max_header_lines} lines without 'Time,' row: {path}")


def parse_header_datetime(meta: dict):
    date_s = meta.get("Date", "").strip()
    time_s = meta.get("Time", "").strip()
    if not date_s or not time_s:
        raise ValueError(f"Missing Date/Time in header meta: {meta}")
    d, m, y = date_s.split("/")
    # confirmed DD/MM/YYYY via cross-check against Record_ID '07-AUG-2025' style field
    day, month, year = int(d), int(m), int(y)
    if "." in time_s:
        hms, frac = time_s.split(".")
        micro = int((frac + "000000")[:6])
    else:
        hms, micro = time_s, 0
    hh, mm, ss = [int(x) for x in hms.split(":")]
    return datetime(year, month, day, hh, mm, ss, micro), date_s, time_s


def scan_raw_csv_body(path: Path, header_cols, header_row_idx: int):
    """Single pass over the whole data section. Returns a stats dict.
    Defensive: malformed rows are counted/logged (a sample), not fatal."""
    idx = {name: i for i, name in enumerate(header_cols)}
    required = ["Time", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event"]
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"Raw CSV {path} missing expected columns: {missing}; got {header_cols}")

    n_rows = 0
    malformed = 0
    malformed_samples = []
    first_time = None
    last_time = None
    trig_nonzero = 0
    trig_vals = set()
    event_nonzero = 0
    event_vals = set()
    adc_status_nonzero = 0
    adc_status_vals = set()
    adc_seq_min, adc_seq_max = None, None
    time_offset_sum = 0.0
    time_offset_min, time_offset_max = None, None
    spot_rows = {}
    spot_channels = [c for c in ("P3", "Cz", "Fp1", "Pz") if c in idx]

    ti, tri, toi, asi, aqi, evi = (idx["Time"], idx["Trigger"], idx["Time_Offset"],
                                    idx["ADC_Status"], idx["ADC_Sequence"], idx["Event"])
    ncols_expected = len(header_cols)

    with open(path, "r", errors="replace") as f:
        for _ in range(header_row_idx + 1):
            f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < ncols_expected - 1:  # allow missing trailing empty Comments field
                malformed += 1
                if len(malformed_samples) < 5:
                    malformed_samples.append(line[:200])
                continue
            try:
                tval = float(parts[ti])
                trig = float(parts[tri]) if parts[tri] != "" else 0.0
                toff = float(parts[toi]) if parts[toi] != "" else 0.0
                astat = float(parts[asi]) if parts[asi] != "" else 0.0
                aseq = float(parts[aqi]) if parts[aqi] != "" else 0.0
                ev = float(parts[evi]) if parts[evi] != "" else 0.0
            except (ValueError, IndexError):
                malformed += 1
                if len(malformed_samples) < 5:
                    malformed_samples.append(line[:200])
                continue

            if first_time is None:
                first_time = tval
            last_time = tval

            if trig != 0:
                trig_nonzero += 1
                if len(trig_vals) < 50:
                    trig_vals.add(trig)
            if ev != 0:
                event_nonzero += 1
            if len(event_vals) < 50:
                event_vals.add(ev)
            if astat != 0:
                adc_status_nonzero += 1
                if len(adc_status_vals) < 50:
                    adc_status_vals.add(astat)
            if adc_seq_min is None or aseq < adc_seq_min:
                adc_seq_min = aseq
            if adc_seq_max is None or aseq > adc_seq_max:
                adc_seq_max = aseq
            time_offset_sum += toff
            if time_offset_min is None or toff < time_offset_min:
                time_offset_min = toff
            if time_offset_max is None or toff > time_offset_max:
                time_offset_max = toff

            if n_rows % SPOT_CHECK_STRIDE == 0:
                try:
                    spot_rows[n_rows] = {c: float(parts[idx[c]]) for c in spot_channels}
                except (ValueError, IndexError):
                    pass

            n_rows += 1

    return {
        "n_rows": n_rows,
        "malformed_rows": malformed,
        "malformed_samples": malformed_samples,
        "first_time_s": first_time,
        "last_time_s": last_time,
        "duration_s": (last_time - first_time) if (first_time is not None and last_time is not None) else None,
        "trigger_nonzero_rows": trig_nonzero,
        "trigger_distinct_nonzero_values": sorted(trig_vals),
        "event_nonzero_rows": event_nonzero,
        "event_distinct_values_sample": sorted(event_vals),
        "adc_status_nonzero_rows": adc_status_nonzero,
        "adc_status_distinct_nonzero_values_sample": sorted(adc_status_vals),
        "adc_sequence_min": adc_seq_min,
        "adc_sequence_max": adc_seq_max,
        "time_offset_min": time_offset_min,
        "time_offset_max": time_offset_max,
        "time_offset_mean": (time_offset_sum / n_rows) if n_rows else None,
        "spot_rows": spot_rows,
    }


# --------------------------------------------------------------------------
# EDF inspection
# --------------------------------------------------------------------------

def inspect_edf(path: Path):
    if pyedflib is None:
        raise RuntimeError("pyedflib not available")
    f = pyedflib.EdfReader(str(path))
    try:
        labels = f.getSignalLabels()
        freqs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
        nsamp = [int(f.getNSamples()[i]) for i in range(f.signals_in_file)]
        start = f.getStartdatetime()
        dur = f.getFileDuration()
        pmins = [f.getPhysicalMinimum(i) for i in range(f.signals_in_file)]
        pmaxs = [f.getPhysicalMaximum(i) for i in range(f.signals_in_file)]
        dmins = [f.getDigitalMinimum(i) for i in range(f.signals_in_file)]
        dmaxs = [f.getDigitalMaximum(i) for i in range(f.signals_in_file)]

        trig_unique, event_unique = None, None
        if "Trigger" in labels:
            trig_unique = sorted(set(np.unique(f.readSignal(labels.index("Trigger"))).tolist()))
        if "Event" in labels:
            event_unique = sorted(set(np.unique(f.readSignal(labels.index("Event"))).tolist()))

        return {
            "labels": labels,
            "sample_freqs": freqs,
            "n_samples": nsamp,
            "start_datetime": start.isoformat(),
            "duration_s": dur,
            "physical_min": pmins,
            "physical_max": pmaxs,
            "digital_min": dmins,
            "digital_max": dmaxs,
            "trigger_unique_values": trig_unique,
            "event_unique_values": event_unique,
        }
    finally:
        f.close()


def edf_base_channel_name(label: str) -> str:
    n = label
    if n.startswith("EEG "):
        n = n[4:]
    if n.endswith("-Pz"):
        n = n[:-3]
    return n


def spot_check_edf_vs_csv(edf_path: Path, spot_rows: dict, csv_channel_to_edf_idx: dict):
    if pyedflib is None or not spot_rows:
        return []
    f = pyedflib.EdfReader(str(edf_path))
    out = []
    try:
        n_edf = int(f.getNSamples()[0])
        for row_idx, chvals in spot_rows.items():
            if row_idx >= n_edf:
                out.append({"row_idx": row_idx, "skipped": "beyond EDF n_samples"})
                continue
            entry = {"row_idx": row_idx, "channels": {}}
            for ch, csv_val in chvals.items():
                if ch not in csv_channel_to_edf_idx:
                    continue
                edf_idx = csv_channel_to_edf_idx[ch]
                edf_val = float(f.readSignal(edf_idx, start=row_idx, n=1)[0])
                entry["channels"][ch] = {
                    "csv_uV": csv_val, "edf_uV": edf_val, "abs_diff_uV": abs(csv_val - edf_val)
                }
            out.append(entry)
    finally:
        f.close()
    return out


# --------------------------------------------------------------------------
# Task-timestamp CSV parsing
# --------------------------------------------------------------------------

def parse_epoch_us(raw: str):
    raw = raw.strip()
    if raw == "":
        return None
    try:
        if "e" in raw.lower() or "." in raw:
            return int(round(float(raw)))
        return int(raw)
    except ValueError:
        return None


PLAUSIBLE_YEAR_RANGE = (2015, 2035)


def parse_task_csv(path: Path):
    rows = []
    rejected = []
    with open(path, "r", errors="replace", newline="") as f:
        reader = csvmod.reader(f)
        header = next(reader, None)
        for r in reader:
            if len(r) < 3:
                rejected.append((r, "too few columns"))
                continue
            task, start_raw, end_raw = r[0], r[1], r[2]
            start_us = parse_epoch_us(start_raw)
            end_us = parse_epoch_us(end_raw)
            if start_us is None or end_us is None:
                rejected.append((r, "unparseable epoch"))
                continue
            try:
                start_dt = datetime(1970, 1, 1) + timedelta(microseconds=start_us)
                end_dt = datetime(1970, 1, 1) + timedelta(microseconds=end_us)
            except (OverflowError, OSError, ValueError):
                rejected.append((r, "epoch out of range"))
                continue
            if not (PLAUSIBLE_YEAR_RANGE[0] <= start_dt.year <= PLAUSIBLE_YEAR_RANGE[1]):
                rejected.append((r, f"implausible year {start_dt.year} (corrupt timestamp)"))
                continue
            rows.append({
                "task": task, "start_us": start_us, "end_us": end_us,
                "start_dt": start_dt, "end_dt": end_dt,
                "duration_s": (end_us - start_us) / 1e6,
            })
    return rows, rejected, header


# --------------------------------------------------------------------------
# Session processing
# --------------------------------------------------------------------------

def pair_csv_edf(raw_csv_files, raw_edf_files):
    """Return list of (csv_path_or_None, edf_path_or_None, pairing_method)."""
    def stem(p, suffix):
        s = p.name.lower()
        if s.endswith(suffix):
            s = s[: -len(suffix)]
        return re.sub(r"[\s\-_;]", "", s)

    csv_stems = {stem(c, "_raw.csv"): c for c in raw_csv_files}
    edf_stems = {stem(e, "_raw.edf"): e for e in raw_edf_files}

    pairs = []
    used_csv, used_edf = set(), set()
    for s, c in csv_stems.items():
        if s in edf_stems:
            pairs.append((c, edf_stems[s], "matched_stem"))
            used_csv.add(c)
            used_edf.add(edf_stems[s])

    remaining_csv = [c for c in raw_csv_files if c not in used_csv]
    remaining_edf = [e for e in raw_edf_files if e not in used_edf]
    if remaining_csv and remaining_edf:
        for c, e in zip(sorted(remaining_csv, key=lambda p: p.name),
                         sorted(remaining_edf, key=lambda p: p.name)):
            pairs.append((c, e, "positional_fallback_stems_differ"))
            used_csv.add(c)
            used_edf.add(e)

    for c in raw_csv_files:
        if c not in used_csv:
            pairs.append((c, None, "csv_unpaired"))
    for e in raw_edf_files:
        if e not in used_edf:
            pairs.append((None, e, "edf_unpaired"))
    return pairs


@dataclass
class SessionResult:
    session: str
    condition: str
    task_folder: Optional[str] = None
    eeg_subfolders_present: list = field(default_factory=list)
    empty_flag: bool = False
    file_inventory: dict = field(default_factory=dict)
    task_csvs: list = field(default_factory=list)
    task_csv_rejected: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def process_session(session_dir: Path) -> dict:
    sname = session_dir.name
    cond = sname.rsplit("-", 1)[-1]
    result = {"session": sname, "condition": cond, "errors": []}

    task_folder, subfolders = find_task_folder(session_dir)
    result["eeg_subfolders_present"] = subfolders
    if task_folder is None:
        if sname in KNOWN_EMPTY_SESSIONS or not subfolders:
            result["status"] = "NO_TASK_SUBFOLDER"
        else:
            result["status"] = "NO_TASK_SUBFOLDER_UNEXPECTED"
            log(f"{sname}: no living-lab-tasks/livinglab subfolder found; present={subfolders}", "WARN")
        result["file_inventory"] = {}
        result["pairs"] = []
    else:
        result["task_folder"] = str(task_folder)
        inv = classify_task_folder_files(task_folder)
        result["file_inventory"] = {k: [str(p) for p in v] for k, v in inv.items()}
        if not any([inv["dsi"], inv["raw_csv"], inv["raw_edf"], inv["accel_csv"], inv["other"]]):
            if sname in KNOWN_EMPTY_SESSIONS:
                result["status"] = "EMPTY_FOLDER_CONFIRMED"
                log(f"{sname}: confirmed empty living-lab task folder (matches known list)", "INFO")
            else:
                result["status"] = "EMPTY_FOLDER_NEW_FINDING"
                log(f"{sname}: EMPTY task folder NOT in the pre-confirmed empty list -- new finding", "WARN")
        else:
            result["status"] = "HAS_FILES"

        pairs = pair_csv_edf(inv["raw_csv"], inv["raw_edf"])
        pair_results = []
        for csv_p, edf_p, method in pairs:
            pr = {"csv": str(csv_p) if csv_p else None, "edf": str(edf_p) if edf_p else None,
                  "pairing_method": method}
            if csv_p is not None:
                try:
                    meta, cols, hdr_idx = parse_raw_csv_header(csv_p)
                    start_dt, date_s, time_s = parse_header_datetime(meta)
                    pr["csv_meta"] = meta
                    pr["csv_columns"] = cols
                    pr["csv_header_row_idx"] = hdr_idx
                    pr["csv_start_datetime"] = start_dt.isoformat()
                    t0 = time.time()
                    body = scan_raw_csv_body(csv_p, cols, hdr_idx)
                    pr["csv_scan_seconds"] = round(time.time() - t0, 2)
                    pr["csv_body"] = {k: v for k, v in body.items() if k != "spot_rows"}
                    pr["_spot_rows"] = body["spot_rows"]
                    pr["csv_size_bytes"] = csv_p.stat().st_size
                    log(f"{sname}: scanned {csv_p.name} ({pr['csv_size_bytes']/1e6:.1f} MB, "
                        f"{body['n_rows']} rows, {pr['csv_scan_seconds']}s) -- "
                        f"trigger_nonzero={body['trigger_nonzero_rows']} "
                        f"event_nonzero={body['event_nonzero_rows']} "
                        f"malformed={body['malformed_rows']}")
                except Exception as e:
                    pr["csv_error"] = f"{type(e).__name__}: {e}"
                    log(f"{sname}: FAILED parsing {csv_p}: {e}", "ERROR")
                    log(traceback.format_exc(), "ERROR")
            if edf_p is not None:
                try:
                    edf_info = inspect_edf(edf_p)
                    pr["edf_info"] = edf_info
                    pr["edf_size_bytes"] = edf_p.stat().st_size
                    log(f"{sname}: read EDF header {edf_p.name} ({pr['edf_size_bytes']/1e6:.1f} MB, "
                        f"{edf_info['n_samples'][0]} samples @ {edf_info['sample_freqs'][0]} Hz, "
                        f"trigger_unique={edf_info['trigger_unique_values']}, "
                        f"event_unique={edf_info['event_unique_values']})")
                except Exception as e:
                    pr["edf_error"] = f"{type(e).__name__}: {e}"
                    log(f"{sname}: FAILED reading EDF {edf_p}: {e}", "ERROR")
                    log(traceback.format_exc(), "ERROR")

            if csv_p is not None and edf_p is not None and "csv_columns" in pr and "edf_info" in pr:
                try:
                    pr["equivalence"] = compare_csv_edf(pr)
                except Exception as e:
                    pr["equivalence_error"] = f"{type(e).__name__}: {e}"
                    log(f"{sname}: FAILED equivalence check {csv_p.name} vs {edf_p.name}: {e}", "ERROR")

            pair_results.append(pr)
        result["pairs"] = pair_results

    task_csvs, rejected = find_annotation_csvs(session_dir)
    result["task_csvs"] = [str(p) for p in task_csvs]
    result["task_csv_rejected"] = [(str(p), first) for p, first in rejected]
    if not task_csvs:
        log(f"{sname}: NO usable task-timestamp CSV found in annotation/", "WARN")

    if task_csvs and result.get("pairs"):
        # Primary raw CSV for alignment/coverage checks: prefer the file whose
        # normalized name actually contains this session's own id (e.g.
        # "sub-70-c_raw.csv" for session sub-70-c), since messy folders can
        # contain a raw CSV from an unrelated recording/attempt (see sub-70-c,
        # which holds both "sub-70-a-living-lab_raw.csv" -- someone else's
        # attempt label, NOT this session -- and "sub-70-c_raw.csv"). Falling
        # back to "largest file" for that case silently picks the wrong one.
        candidates = [pr for pr in result["pairs"] if pr.get("csv") and "csv_body" in pr]

        def _norm(s):
            return re.sub(r"[\s\-_;]", "", s.lower())

        # Other task types (UPDRS, Motor, Blocks, Manipulations, memory,
        # PAL, cats/dogs, resting) sometimes have a stray recording dropped
        # into the living-lab task subfolder (documented: sub-42-c has a
        # UPDRS recording alongside its real living-lab CSV). Prefer a
        # name-matching candidate WITHOUT one of those keywords.
        DECOY_KEYWORDS = ("updrs", "motor", "block", "manipulation", "mem_", "memory",
                          "-pal", "_pal", "catsndog", "cats-and-dog", "resting")
        sess_norm = _norm(sname)
        name_matches = [pr for pr in candidates if sess_norm in _norm(Path(pr["csv"]).stem)]
        if len(name_matches) == 1:
            primary = name_matches[0]
        elif len(name_matches) > 1:
            non_decoy = [pr for pr in name_matches
                         if not any(k in Path(pr["csv"]).name.lower() for k in DECOY_KEYWORDS)]
            pool = non_decoy if non_decoy else name_matches
            primary = max(pool, key=lambda pr: pr.get("csv_size_bytes", 0))
            log(f"{sname}: multiple raw CSVs match session name; picked "
                f"{'largest non-decoy' if non_decoy else 'largest (all looked like decoys)'} "
                f"({Path(primary['csv']).name}) among {[Path(p['csv']).name for p in name_matches]}", "WARN")
        elif candidates:
            primary = max(candidates, key=lambda pr: pr.get("csv_size_bytes", 0))
            log(f"{sname}: NO raw CSV filename contains the session id '{sname}'; "
                f"falling back to largest file ({Path(primary['csv']).name}) as primary -- "
                f"treat alignment result for this session with extra caution", "WARN")
        else:
            primary = None
        if primary is not None:
            try:
                task_rows, task_rejected, task_header = parse_task_csv(Path(task_csvs[0]))
                align = compute_alignment(sname, primary, task_rows, task_rejected)
                result["alignment"] = align
                if len(task_csvs) > 1:
                    log(f"{sname}: multiple task-timestamp CSVs found ({len(task_csvs)}); "
                        f"used {task_csvs[0]} for alignment", "WARN")
            except Exception as e:
                result["alignment_error"] = f"{type(e).__name__}: {e}"
                log(f"{sname}: FAILED alignment computation: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")

    return result


def compare_csv_edf(pr: dict) -> dict:
    csv_cols = pr["csv_columns"]
    csv_channels = [c for c in csv_cols if c not in META_COLS]
    edf_labels = pr["edf_info"]["labels"]
    edf_base_names = {edf_base_channel_name(l): i for i, l in enumerate(edf_labels)}

    csv_set = set(csv_channels)
    edf_set = set(edf_base_names.keys()) - {"Trigger", "Event", "Ax", "Ay", "Az", "CM"}
    common = sorted(csv_set & edf_set)
    csv_only = sorted(csv_set - edf_set)
    edf_only_extra = sorted(set(edf_base_names.keys()) - csv_set)

    csv_freq = pr["csv_meta"].get("Sample_Frequency_(Hz)")
    edf_freqs_eeg = set(f for l, f in zip(edf_labels, pr["edf_info"]["sample_freqs"])
                         if edf_base_channel_name(l) in common)

    csv_n_rows = pr["csv_body"]["n_rows"]
    edf_n_samples = None
    if common:
        first_common_idx = edf_base_names[common[0]]
        edf_n_samples = pr["edf_info"]["n_samples"][first_common_idx]
    sample_count_diff = (csv_n_rows - edf_n_samples) if edf_n_samples is not None else None

    csv_channel_to_edf_idx = {c: edf_base_names[c] for c in common if c in edf_base_names}
    spot_results = spot_check_edf_vs_csv(Path(pr["edf"]), pr.get("_spot_rows", {}), csv_channel_to_edf_idx)
    max_abs_diff = 0.0
    n_compared = 0
    for entry in spot_results:
        for ch, d in entry.get("channels", {}).items():
            max_abs_diff = max(max_abs_diff, d["abs_diff_uV"])
            n_compared += 1

    # Quantization noise for a genuinely-matched CSV/EDF pair should be at
    # most a couple of µV (EDF 16-bit digitization step here is
    # 2*9830/65534 =~ 0.3 uV/bit). A pairing_method of
    # "positional_fallback_stems_differ" combined with either a large sample
    # count gap or a spot-check diff far above quantization noise means the
    # CSV and EDF almost certainly come from different recordings/attempts
    # (see sub-70-c) -- flag it rather than reporting it as a genuine
    # CSV-vs-EDF numerical divergence.
    likely_mismatched_pair = (
        (sample_count_diff is not None and abs(sample_count_diff) > 1500)
        or (n_compared > 0 and max_abs_diff > 50.0)
    )

    return {
        "csv_channel_count": len(csv_channels),
        "edf_signal_count": len(edf_labels),
        "common_eeg_channels": common,
        "common_eeg_channel_count": len(common),
        "csv_only_channels": csv_only,
        "edf_only_extra_channels": edf_only_extra,
        "csv_sample_freq_hz": csv_freq,
        "edf_eeg_sample_freqs_hz": sorted(edf_freqs_eeg),
        "csv_n_rows": csv_n_rows,
        "edf_n_samples_eeg": edf_n_samples,
        "sample_count_diff_csv_minus_edf": sample_count_diff,
        "sample_count_diff_seconds": (sample_count_diff / 300.0) if sample_count_diff is not None else None,
        "spot_check_n_points_compared": n_compared,
        "spot_check_max_abs_diff_uV": max_abs_diff,
        "spot_check_detail": spot_results,
        "likely_mismatched_pair": likely_mismatched_pair,
    }


def _fit_count_for_shift(task_rows, raw_start, raw_end, shift_s, slack_s=5.0):
    """A task 'fits' if, after subtracting shift_s from its Start/End (i.e.
    mapping task-computer time back onto the raw-signal clock), both ends
    land inside [raw_start, raw_end]. Returns (fit_count, fitting_rows)."""
    fitting = []
    for r in task_rows:
        s = r["start_dt"] - timedelta(seconds=shift_s)
        e = r["end_dt"] - timedelta(seconds=shift_s)
        if (raw_start - timedelta(seconds=slack_s)) <= s and e <= (raw_end + timedelta(seconds=slack_s)):
            fitting.append(r)
    return len(fitting), fitting


def _search_best_shift(task_rows, raw_start, raw_end):
    """Dense grid search (2s steps over +/-4h, ~14k candidates x ~10 tasks --
    cheap, no file IO) for the task_dt -> raw_dt shift that maximizes how
    many tasks fit inside the recording window, then a +/-2s refinement in
    0.05s steps around the best candidate for a precise offset figure.

    An earlier version only tried whole-hour candidates (the documented
    BST/GMT failure mode) with a narrow +/-300s local refinement. That missed
    subjects whose true offset is neither near a whole hour nor near zero
    (e.g. sub-42-c needs roughly -500s / -8.3min to make its last task fit) --
    switched to an exhaustive dense scan so no shift shape is assumed."""
    best = (-1, 0.0, [])
    shift = -4 * 3600.0
    end = 4 * 3600.0
    while shift <= end:
        cnt, fit = _fit_count_for_shift(task_rows, raw_start, raw_end, shift)
        if cnt > best[0]:
            best = (cnt, shift, fit)
        shift += 2.0

    base = best[1]
    fine = base - 2.0
    fine_end = base + 2.0
    while fine <= fine_end:
        cnt, fit = _fit_count_for_shift(task_rows, raw_start, raw_end, fine)
        if cnt > best[0]:
            best = (cnt, fine, fit)
        fine += 0.05
    return best  # (fit_count, shift_s, fitting_rows)


def compute_alignment(sname, primary_pair, task_rows, task_rejected):
    if not task_rows:
        return {"status": "NO_VALID_TASK_ROWS", "task_rejected": [str(r) for r in task_rejected]}

    raw_start = datetime.fromisoformat(primary_pair["csv_start_datetime"])
    duration_s = primary_pair["csv_body"]["duration_s"]
    if duration_s is None:
        return {"status": "NO_RAW_DURATION", "n_valid_task_rows": len(task_rows)}
    raw_end = raw_start + timedelta(seconds=duration_s)

    # Literal (no-search) offset using the CSV's own first Start / last End,
    # exactly as named in the brief -- kept for transparency even though it
    # can be thrown off by pre/post-recording setup tasks (e.g. sub-131-a).
    task_rows_sorted = sorted(task_rows, key=lambda r: r["start_us"])
    literal_first = task_rows_sorted[0]
    literal_last = max(task_rows, key=lambda r: r["end_us"])
    literal_offset_first_s = (literal_first["start_dt"] - raw_start).total_seconds()
    literal_offset_last_s = (literal_last["end_dt"] - raw_end).total_seconds()

    fit_count, best_shift, fitting_rows = _search_best_shift(task_rows, raw_start, raw_end)

    if fit_count == 0:
        return {
            "status": "OK",
            "n_valid_task_rows": len(task_rows),
            "n_rejected_task_rows": len(task_rejected),
            "task_rejected_detail": [{"row": r, "reason": reason} for r, reason in task_rejected],
            "raw_start_dt": raw_start.isoformat(),
            "raw_end_dt": raw_end.isoformat(),
            "raw_duration_s": duration_s,
            "literal_offset_first_s": literal_offset_first_s,
            "literal_offset_last_s": literal_offset_last_s,
            "best_fit_task_count": 0,
            "classification": "UNRESOLVABLE",
            "coverage_duration_covers_task_span": None,
        }

    fitting_sorted = sorted(fitting_rows, key=lambda r: r["start_us"])
    fit_first, fit_last = fitting_sorted[0], max(fitting_rows, key=lambda r: r["end_us"])
    offset_first_s = (fit_first["start_dt"] - raw_start).total_seconds()
    offset_last_s = (fit_last["end_dt"] - raw_end).total_seconds()
    # re-anchor offset_last against the fitting task's position, not raw_end,
    # since a task can fit well inside the window without reaching its edge
    raw_at_fit_last_end = raw_start + timedelta(seconds=(fit_last["end_dt"] - raw_start).total_seconds())
    delta_s = offset_last_s - offset_first_s
    task_span_s = (fit_last["end_us"] - fit_first["start_us"]) / 1e6

    def whole_hour_tag(off):
        nearest_hour = round(off / 3600.0) * 3600.0
        residual = off - nearest_hour
        if abs(residual) <= BST_TOLERANCE_S:
            return {"nearest_hour_shift_s": nearest_hour, "residual_s": residual}
        return None

    # How close the anchor tasks actually sit to the recording's own edges.
    # If the first/last USABLE task is itself far from raw_start/raw_end
    # (e.g. sub-134-c, where the genuine first 3 tasks have corrupted
    # zero-epoch timestamps and get filtered out upstream, leaving "Balance"
    # -- ~20 min into the protocol -- as the earliest usable anchor), a large
    # delta_s reflects "no anchor near the edge", NOT necessarily clock
    # drift; flag it so the report doesn't conflate the two.
    ANCHOR_NEAR_EDGE_S = 120.0
    elapsed_from_start_s = (fit_first["start_dt"] - timedelta(seconds=best_shift) - raw_start).total_seconds()
    elapsed_before_end_s = (raw_end - (fit_last["end_dt"] - timedelta(seconds=best_shift))).total_seconds()
    start_anchor_near_edge = abs(elapsed_from_start_s) <= ANCHOR_NEAR_EDGE_S
    end_anchor_near_edge = abs(elapsed_before_end_s) <= ANCHOR_NEAR_EDGE_S

    n_total = len(task_rows)
    if fit_count < n_total:
        classification = "PARTIAL_FIT"
    elif not (start_anchor_near_edge and end_anchor_near_edge):
        classification = "EDGE_ANCHOR_UNAVAILABLE"
    elif abs(delta_s) <= FIXED_OFFSET_TOLERANCE_S:
        classification = "FIXED_OFFSET"
    elif abs(delta_s) <= max(60.0, 0.02 * task_span_s):
        classification = "MINOR_DRIFT"
    else:
        classification = "DRIFT"

    coverage_ok = duration_s >= task_span_s

    return {
        "status": "OK",
        "n_valid_task_rows": n_total,
        "n_rejected_task_rows": len(task_rejected),
        "task_rejected_detail": [{"row": r, "reason": reason} for r, reason in task_rejected],
        "raw_start_dt": raw_start.isoformat(),
        "raw_end_dt": raw_end.isoformat(),
        "raw_duration_s": duration_s,
        "literal_offset_first_s": literal_offset_first_s,
        "literal_offset_last_s": literal_offset_last_s,
        "best_shift_s": best_shift,
        "best_fit_task_count": fit_count,
        "n_total_valid_tasks": n_total,
        "fitting_tasks": [r["task"] for r in fitting_sorted],
        "non_fitting_tasks": [r["task"] for r in task_rows if r not in fitting_rows],
        "first_fitting_task": fit_first["task"],
        "last_fitting_task": fit_last["task"],
        "offset_first_s": offset_first_s,
        "offset_last_s": offset_last_s,
        "delta_offset_s": delta_s,
        "drift_rate_s_per_s": (delta_s / task_span_s) if task_span_s else None,
        "whole_hour_tag_first": whole_hour_tag(offset_first_s),
        "whole_hour_tag_last": whole_hour_tag(offset_last_s),
        "elapsed_from_start_s": elapsed_from_start_s,
        "elapsed_before_end_s": elapsed_before_end_s,
        "start_anchor_near_edge": start_anchor_near_edge,
        "end_anchor_near_edge": end_anchor_near_edge,
        "classification": classification,
        "task_span_s": task_span_s,
        "coverage_duration_covers_task_span": coverage_ok,
        "coverage_shortfall_s": (task_span_s - duration_s) if not coverage_ok else None,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subjects", type=str, default=None, help="comma-separated session names")
    args = ap.parse_args()

    subjects_filter = set(args.subjects.split(",")) if args.subjects else None

    open_log()
    if not DATA_ROOT.is_dir():
        log(f"DATA_ROOT does not exist or is not accessible: {DATA_ROOT}", "ERROR")
        sys.exit(1)

    sessions = discover_sessions(limit=args.limit, subjects_filter=subjects_filter)
    log(f"Discovered {len(sessions)} session folders to process.")

    all_results = []
    t_start = time.time()
    for i, sdir in enumerate(sessions):
        log(f"--- [{i+1}/{len(sessions)}] Processing {sdir.name} ---")
        try:
            res = process_session(sdir)
        except Exception as e:
            log(f"{sdir.name}: UNCAUGHT ERROR: {e}", "ERROR")
            log(traceback.format_exc(), "ERROR")
            res = {"session": sdir.name, "status": "UNCAUGHT_ERROR", "error": str(e)}
        all_results.append(res)

    elapsed = time.time() - t_start
    log(f"Done processing {len(sessions)} sessions in {elapsed:.1f}s.")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Wrote full results to {RESULTS_JSON}")

    write_flat_csv(all_results)
    log(f"Wrote flat summary to {RESULTS_CSV}")


def write_flat_csv(all_results):
    rows = []
    for r in all_results:
        base = {
            "session": r.get("session"),
            "condition": r.get("condition"),
            "status": r.get("status"),
            "n_raw_csv": len(r.get("file_inventory", {}).get("raw_csv", [])),
            "n_raw_edf": len(r.get("file_inventory", {}).get("raw_edf", [])),
            "n_dsi": len(r.get("file_inventory", {}).get("dsi", [])),
            "n_task_csv": len(r.get("task_csvs", [])),
        }
        align = r.get("alignment", {})
        base.update({
            "alignment_status": align.get("status"),
            "alignment_classification": align.get("classification"),
            "best_shift_s": align.get("best_shift_s"),
            "best_fit_task_count": align.get("best_fit_task_count"),
            "n_total_valid_tasks": align.get("n_total_valid_tasks") or align.get("n_valid_task_rows"),
            "offset_first_s": align.get("offset_first_s"),
            "offset_last_s": align.get("offset_last_s"),
            "delta_offset_s": align.get("delta_offset_s"),
            "coverage_ok": align.get("coverage_duration_covers_task_span"),
            "raw_duration_s": align.get("raw_duration_s"),
            "task_span_s": align.get("task_span_s"),
        })
        pairs = r.get("pairs", [])
        trig_any = any(p.get("csv_body", {}).get("trigger_nonzero_rows", 0) for p in pairs)
        event_any = any(p.get("csv_body", {}).get("event_nonzero_rows", 0) for p in pairs)
        base["any_trigger_nonzero"] = trig_any
        base["any_event_nonzero"] = event_any
        eq = next((p["equivalence"] for p in pairs if "equivalence" in p), None)
        if eq:
            base["edf_csv_common_channels"] = eq["common_eeg_channel_count"]
            base["edf_csv_sample_count_diff"] = eq["sample_count_diff_csv_minus_edf"]
            base["edf_csv_spot_max_abs_diff_uV"] = eq["spot_check_max_abs_diff_uV"]
            base["edf_csv_likely_mismatched_pair"] = eq["likely_mismatched_pair"]
        rows.append(base)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
