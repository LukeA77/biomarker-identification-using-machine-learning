"""Run tracking: one JSON per run + an appended master CSV row.

Plain files, no W&B — offline-friendly and enough to regenerate every number in
the dissertation (FINE_TUNING_PLAN §8). Each record captures git SHA, checkpoint
hash, library versions, the full config, and the metrics, so a run is
self-describing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["git_sha", "file_sha256", "env_versions", "save_run"]


def git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("git SHA unavailable: %s", exc)
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict:
    import numpy, torch  # local import keeps module import cheap
    versions = {"torch": torch.__version__, "numpy": numpy.__version__}
    for name in ("peft", "transformers", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[name] = "not-installed"
    return versions


def save_run(record: dict, results_dir: Path, master_csv: Path, *, csv_fields: list[str]) -> Path:
    """Write ``record`` as JSON and append a flattened row to the master CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = record.get("run_tag", "run")
    json_path = results_dir / f"{tag}_{ts}.json"
    json_path.write_text(json.dumps(record, indent=2, default=str))

    master_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not master_csv.exists()
    if not write_header:
        # Appending a DIFFERENT field set to an existing CSV writes values under
        # the wrong column names — silent corruption of every earlier row's
        # meaning. Refuse: an experiment with a new schema gets its own CSV.
        with open(master_csv, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != list(csv_fields):
            raise ValueError(
                f"master CSV {master_csv} already has header {existing}, which differs from the "
                f"requested fields {list(csv_fields)}. Appending would misalign the columns. "
                f"Point this experiment at its own master CSV instead.")
    with open(master_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k) for k in csv_fields})
    log.info("Saved run -> %s (+ master CSV row)", json_path.name)
    return json_path
