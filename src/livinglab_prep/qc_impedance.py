"""QC Part C - impedance inventory.

Before assuming impedance values live anywhere in particular, this module
checks, for a handful of sample sessions (`qc.impedance_inventory_sample`),
every source format present in this dataset:

  1. The raw-signal CSV's metadata header block and data-column names
     (`csv_signal.py`'s `_parse_header` -- the same parser the production
     pipeline uses) -- searched for any key/column mentioning "imped".
  2. Any EDF file under the session's `eeg/` tree (header-only read: channel
     names + annotation descriptions) -- searched the same way.
  3. Any DSI-Streamer `.dsi` file under the session's `eeg/` tree -- a
     best-effort ASCII sniff of the raw bytes for "imped"/"impedance". `.dsi`
     is a proprietary binary packet format with no public spec and no vendor
     SDK available in this environment, so this is NOT a full packet-level
     parse; a miss here means "no plain-text impedance marker found in the
     byte stream", not a certainty that no impedance packet exists in some
     structured (non-ASCII) form.

If none of these three sources shows any impedance data across the sample
(as found for this cohort -- see qc_summary_report.md), that is reported as a
clear finding, not a silent skip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne

from .config import Config
from .csv_signal import _parse_header
from .qc_config import QcConfig

_DSI_SCAN_KEYWORDS = (b"imped", b"impedance")
_DSI_SCAN_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass
class ImpedanceSourceCheck:
    """Result of checking one file for impedance data.

    Attributes
    ----------
    patient_id, condition : str
    source_type : str
        One of ``"raw_csv"``, ``"edf"``, ``"dsi"``.
    source_path : str
    impedance_found : bool
    notes : str
    """
    patient_id: str
    condition: str
    source_type: str
    source_path: str
    impedance_found: bool
    notes: str


def _check_raw_csv(csv_path: Path) -> ImpedanceSourceCheck | None:
    try:
        _, columns, meta = _parse_header(csv_path)
    except Exception as e:
        return ImpedanceSourceCheck("", "", "raw_csv", str(csv_path), False,
                                    f"could not parse header: {e}")
    hits = [k for k in meta if "imped" in k.lower()]
    hits += [c for c in columns if "imped" in c.lower()]
    found = len(hits) > 0
    notes = (f"metadata/column keys matching 'imped': {hits}" if found else
            f"no 'imped*' key in {len(meta)} metadata field(s) or {len(columns)} data column(s) "
            f"(checked keys: {list(meta)}; checked columns: {columns})")
    return ImpedanceSourceCheck("", "", "raw_csv", str(csv_path), found, notes)


def _check_edf(edf_path: Path) -> ImpedanceSourceCheck | None:
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    except Exception as e:
        return ImpedanceSourceCheck("", "", "edf", str(edf_path), False,
                                    f"could not read EDF header: {e}")
    ch_hits = [c for c in raw.ch_names if "imped" in c.lower()]
    annot_descs = set(raw.annotations.description) if len(raw.annotations) else set()
    annot_hits = [d for d in annot_descs if "imped" in d.lower()]
    found = bool(ch_hits or annot_hits)
    notes = (f"channel/annotation hits: channels={ch_hits} annotations={annot_hits}" if found else
            f"no 'imped*' channel name or annotation description among "
            f"{len(raw.ch_names)} channels / {len(annot_descs)} distinct annotation "
            f"description(s) (channels checked: {raw.ch_names})")
    return ImpedanceSourceCheck("", "", "edf", str(edf_path), found, notes)


def _check_dsi(dsi_path: Path) -> ImpedanceSourceCheck | None:
    found = False
    hit_keyword = None
    hit_offset = None
    try:
        with open(dsi_path, "rb") as fh:
            offset = 0
            while True:
                chunk = fh.read(_DSI_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                low = chunk.lower()
                for kw in _DSI_SCAN_KEYWORDS:
                    idx = low.find(kw)
                    if idx != -1:
                        found, hit_keyword, hit_offset = True, kw.decode(), offset + idx
                        break
                if found:
                    break
                offset += len(chunk)
    except Exception as e:
        return ImpedanceSourceCheck("", "", "dsi", str(dsi_path), False,
                                    f"could not scan .dsi file: {e}")
    notes = (f"ASCII sniff found {hit_keyword!r} at byte offset {hit_offset} "
            f"(proprietary binary format -- verify manually, this is not a packet parse)"
            if found else
            "ASCII sniff of the full file found no 'imped'/'impedance' byte sequence "
            "(proprietary binary packet format, no vendor SDK available -- a structured, "
            "non-ASCII-tagged impedance packet cannot be ruled out by this check alone)")
    return ImpedanceSourceCheck("", "", "dsi", str(dsi_path), found, notes)


def check_session(cfg: Config, qc: QcConfig, patient_id: str, condition: str) -> list[ImpedanceSourceCheck]:
    """Run all applicable impedance checks for one session.

    Searches `<drive_root>/<patient_id>-<condition>/eeg/**` for every raw-
    signal CSV (content-sniffed the same way `cohort_discovery_csv.py` does),
    EDF, and `.dsi` file, and checks each one found.

    Parameters
    ----------
    cfg : Config
        Pipeline config (used only for `cfg.signal.orig_sfreq`-independent
        parsing; no processed output is touched).
    qc : QcConfig
    patient_id, condition : str

    Returns
    -------
    list of ImpedanceSourceCheck
    """
    from .cohort_discovery_csv import _looks_like_raw_signal_csv

    session_dir = qc.drive_root / f"{patient_id}-{condition}" / "eeg"
    results: list[ImpedanceSourceCheck] = []

    if not session_dir.is_dir():
        return [ImpedanceSourceCheck(patient_id, condition, "raw_csv", str(session_dir),
                                    False, "session eeg/ directory not found")]

    csv_candidates = [p for p in session_dir.rglob("*.csv") if _looks_like_raw_signal_csv(p)]
    edf_candidates = list(session_dir.rglob("*.edf"))
    dsi_candidates = list(session_dir.rglob("*.dsi"))

    if not (csv_candidates or edf_candidates or dsi_candidates):
        return [ImpedanceSourceCheck(patient_id, condition, "raw_csv", str(session_dir),
                                    False, "no raw-signal CSV, EDF, or .dsi file found under this session")]

    for path in csv_candidates:
        r = _check_raw_csv(path)
        if r is not None:
            results.append(ImpedanceSourceCheck(patient_id, condition, r.source_type,
                                               r.source_path, r.impedance_found, r.notes))
    for path in edf_candidates:
        r = _check_edf(path)
        if r is not None:
            results.append(ImpedanceSourceCheck(patient_id, condition, r.source_type,
                                               r.source_path, r.impedance_found, r.notes))
    for path in dsi_candidates:
        r = _check_dsi(path)
        if r is not None:
            results.append(ImpedanceSourceCheck(patient_id, condition, r.source_type,
                                               r.source_path, r.impedance_found, r.notes))
    return results


def run_impedance_inventory(cfg: Config, qc: QcConfig) -> list[ImpedanceSourceCheck]:
    """Run `check_session` over every session in `qc.impedance_inventory_sample`.

    Returns
    -------
    list of ImpedanceSourceCheck
    """
    all_results: list[ImpedanceSourceCheck] = []
    for s in qc.impedance_inventory_sample.sessions:
        print(f"[qc-impedance] checking {s.patient}-{s.condition} ...")
        all_results.extend(check_session(cfg, qc, s.patient, s.condition))
    return all_results
