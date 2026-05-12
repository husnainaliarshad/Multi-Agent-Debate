"""
Experiment discovery: merge in-memory runs with persisted index + on-disk logs.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from services.experiment_manager import experiment_manager


def experiments_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "experiments"


def index_path() -> Path:
    return experiments_root() / "experiments_index.json"


def load_index_map() -> Dict[str, Dict[str, Any]]:
    p = index_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(e["id"]): dict(e) for e in data.get("experiments", []) if e.get("id")}
    except Exception:
        return {}


def _meta_from_experiment_log(exp_dir: Path) -> Optional[Dict[str, Any]]:
    log = exp_dir / "experiment_log.json"
    if not log.is_file():
        return None
    try:
        uuid.UUID(exp_dir.name)
    except ValueError:
        return None
    try:
        data = json.loads(log.read_text(encoding="utf-8"))
    except Exception:
        return None
    st = log.stat().st_mtime
    return {
        "id": exp_dir.name,
        "name": data.get("experiment_name") or data.get("name") or "Unknown",
        "status": data.get("status") or "unknown",
        "total_runs": int(data.get("total_runs") or 0),
        "completed_runs": int(data.get("completed_runs") or len(data.get("runs") or [])),
        "results_path": data.get("results_file"),
        "config": data.get("config") or {},
        "sort_time": st,
        "source": "disk_log",
    }


def build_experiment_catalog() -> List[Dict[str, Any]]:
    """In-memory experiments overlaid on index + any UUID folders with experiment_log.json."""
    by_id: Dict[str, Dict[str, Any]] = {}

    for e in experiment_manager.list_experiments():
        row = dict(e)
        row["source"] = "memory"
        row.setdefault("name", "Unnamed Experiment")
        row["sort_time"] = float(row.get("start_time") or 0)
        by_id[row["id"]] = row

    root = experiments_root()
    for eid, row in load_index_map().items():
        if eid in by_id:
            merged = {**row, **by_id[eid]}
            merged["source"] = "memory" if by_id[eid].get("source") == "memory" else "index+memory"
            by_id[eid] = merged
        else:
            row = dict(row)
            row.setdefault("name", "Unnamed Experiment")
            row["sort_time"] = float(row.get("sort_time") or 0)
            row["source"] = "index"
            by_id[eid] = row

    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            eid = child.name
            try:
                uuid.UUID(eid)
            except ValueError:
                continue
            if eid in by_id:
                continue
            meta = _meta_from_experiment_log(child)
            if meta:
                by_id[eid] = meta

    for eid, row in list(by_id.items()):
        csv_p = root / eid / "results.csv"
        row["has_results_csv"] = csv_p.is_file()

    rows = list(by_id.values())
    rows.sort(
        key=lambda r: (
            (r.get("name") or "zzz").lower(),
            -(float(r.get("sort_time") or r.get("start_time") or 0)),
        )
    )
    return rows
