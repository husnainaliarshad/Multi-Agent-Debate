from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "run_index",
    "total_runs",
    "status",
    "topic",
    "mode",
    "provider",
    "repeat_index",
    "proposer_model",
    "critic_model",
    "judge_model",
    "use_rag",
    "use_search",
    "session_id",
    "verdict",
    "consensus_score",
]

MODE_EXPECTATIONS = {
    "baseline": {"use_search": False, "use_rag": False},
    "react_only": {"use_search": True, "use_rag": False},
    "naive_rag": {"use_search": False, "use_rag": True},
    "active_rag": {"use_search": False, "use_rag": True},
    "hybrid": {"use_search": True, "use_rag": True},
}


def validate_experiment_results(experiment_id: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    experiment_dir = repo_root / "backend" / "data" / "experiments" / experiment_id
    results_file = experiment_dir / "results.csv"
    log_file = experiment_dir / "experiment_log.json"

    validation: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "results_file": str(results_file),
        "log_file": str(log_file),
        "exists": experiment_dir.exists(),
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    if not experiment_dir.exists():
        validation["errors"].append("Experiment directory does not exist.")
        return validation

    if not results_file.exists():
        validation["errors"].append("results.csv is missing.")
        return validation

    df = pd.read_csv(results_file)
    validation["summary"]["row_count"] = int(len(df))
    validation["summary"]["log_file_exists"] = log_file.exists()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        validation["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")

    if df.empty:
        validation["errors"].append("results.csv is empty.")
        return validation

    completed_df = df[df["status"] == "completed"].copy() if "status" in df.columns else df.copy()
    failed_df = df[df["status"] == "failed"].copy() if "status" in df.columns else df.iloc[0:0].copy()

    validation["summary"]["completed_runs"] = int(len(completed_df))
    validation["summary"]["failed_runs"] = int(len(failed_df))

    for field in ["topic", "mode", "provider", "proposer_model", "critic_model", "judge_model"]:
        if field in completed_df.columns:
            missing_mask = completed_df[field].isna() | (completed_df[field].astype(str).str.strip() == "")
            if missing_mask.any():
                validation["errors"].append(f"Completed rows are missing values in '{field}'.")

    duplicate_subset = [
        column
        for column in ["topic", "mode", "provider", "repeat_index", "proposer_model", "critic_model", "judge_model"]
        if column in df.columns
    ]
    if duplicate_subset:
        duplicates = df[df.duplicated(subset=duplicate_subset, keep=False)]
        if not duplicates.empty:
            validation["warnings"].append(
                f"Potential duplicate rows detected for keys: {', '.join(duplicate_subset)}."
            )
            validation["summary"]["duplicate_rows"] = duplicates[duplicate_subset].to_dict(orient="records")

    numeric_columns = [column for column in ["consensus_score", "avg_info_gain", "format_adherence", "faithfulness"] if column in completed_df.columns]
    if numeric_columns and not completed_df.empty:
        zero_metric_rows = completed_df[(completed_df[numeric_columns].fillna(0) == 0).all(axis=1)]
        if not zero_metric_rows.empty:
            validation["warnings"].append(
                f"{len(zero_metric_rows)} completed run(s) have all tracked metrics equal to zero."
            )

    if {"mode", "use_search", "use_rag"}.issubset(df.columns):
        mismatches: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            expected = MODE_EXPECTATIONS.get(str(row["mode"]))
            if not expected:
                continue
            row_use_search = _to_bool(row["use_search"])
            row_use_rag = _to_bool(row["use_rag"])
            # Batch experiments force use_rag=False while mode may still name a RAG-oriented preset.
            rag_mismatch = row_use_rag != expected["use_rag"] and (
                not expected["use_rag"] or row_use_rag
            )
            if row_use_search != expected["use_search"] or rag_mismatch:
                mismatches.append(
                    {
                        "topic": row.get("topic"),
                        "mode": row.get("mode"),
                        "use_search": row_use_search,
                        "use_rag": row_use_rag,
                    }
                )
        if mismatches:
            validation["warnings"].append("Some rows have mode/flag mismatches.")
            validation["summary"]["mode_flag_mismatches"] = mismatches

    if not completed_df.empty and "mode" in completed_df.columns:
        aggregates = []
        for mode, group in completed_df.groupby("mode"):
            aggregates.append(
                {
                    "mode": mode,
                    "runs": int(len(group)),
                    "avg_consensus_score": _safe_mean(group, "consensus_score"),
                    "avg_info_gain": _safe_mean(group, "avg_info_gain"),
                    "avg_faithfulness": _safe_mean(group, "faithfulness"),
                    "avg_format_adherence": _safe_mean(group, "format_adherence"),
                }
            )
        validation["summary"]["by_mode"] = aggregates

    if "error" in failed_df.columns and not failed_df.empty:
        missing_error_rows = failed_df[failed_df["error"].isna() | (failed_df["error"].astype(str).str.strip() == "")]
        if not missing_error_rows.empty:
            validation["warnings"].append("Some failed rows do not contain an error message.")

    return validation


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean())


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
