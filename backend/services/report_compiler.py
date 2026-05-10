import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ReportCompilerService:
    def __init__(
        self,
        experiments_root: str = "backend/data/experiments",
        benchmarks_root: str = "backend/data/legalbench_benchmarks",
        reports_root: str = "backend/data/report_summaries",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = repo_root
        self.experiments_root = self._resolve_path(repo_root, experiments_root)
        self.benchmarks_root = self._resolve_path(repo_root, benchmarks_root)
        self.reports_root = self._resolve_path(repo_root, reports_root)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def list_available_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        experiments: List[Dict[str, Any]] = []
        if self.experiments_root.exists():
            for experiment_dir in sorted(self.experiments_root.iterdir(), key=lambda item: item.name):
                results_path = experiment_dir / "results.csv"
                if not results_path.exists():
                    continue
                experiments.append(
                    {
                        "id": experiment_dir.name,
                        "results_csv": str(results_path),
                        "experiment_log": str(experiment_dir / "experiment_log.json"),
                    }
                )

        benchmarks: List[Dict[str, Any]] = []
        if self.benchmarks_root.exists():
            for benchmark_dir in sorted(self.benchmarks_root.iterdir(), key=lambda item: item.name):
                summary_path = benchmark_dir / "summary.json"
                if not summary_path.exists():
                    continue
                benchmarks.append(
                    {
                        "id": benchmark_dir.name,
                        "summary_json": str(summary_path),
                        "benchmark_summary_csv": str(benchmark_dir / "benchmark_summary.csv"),
                    }
                )

        reports: List[Dict[str, Any]] = []
        if self.reports_root.exists():
            for report_dir in sorted(self.reports_root.iterdir(), key=lambda item: item.name):
                manifest_path = report_dir / "report_manifest.json"
                if not manifest_path.exists():
                    continue
                reports.append(
                    {
                        "id": report_dir.name,
                        "manifest": str(manifest_path),
                    }
                )

        return {
            "experiments": experiments,
            "benchmark_runs": benchmarks,
            "reports": reports,
        }

    def compile_report(
        self,
        config: Dict[str, Any],
        report_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        report_id = report_id or str(uuid.uuid4())
        report_name = config.get("name", f"report_{report_id[:8]}")
        experiment_ids = config.get("experiment_ids") or self._discover_experiment_ids()
        benchmark_run_ids = config.get("benchmark_run_ids") or self._discover_benchmark_run_ids()

        output_dir = self.reports_root / report_id
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "report_manifest.json"
        debate_by_mode_path = output_dir / "debate_by_mode.csv"
        debate_by_topic_mode_path = output_dir / "debate_by_topic_mode.csv"
        debate_topic_pivot_path = output_dir / "debate_topic_consensus_pivot.csv"
        debate_topic_delta_path = output_dir / "debate_topic_deltas.csv"
        benchmark_by_dataset_path = output_dir / "legalbench_by_dataset.csv"
        notes_path = output_dir / "report_notes.txt"

        debate_payload = self._compile_debate_tables(experiment_ids)
        benchmark_payload = self._compile_benchmark_tables(benchmark_run_ids)
        guardrails = self._build_guardrails(debate_payload, benchmark_payload)

        if debate_payload["by_mode"] is not None:
            debate_payload["by_mode"].to_csv(debate_by_mode_path, index=False)
        if debate_payload["by_topic_mode"] is not None:
            debate_payload["by_topic_mode"].to_csv(debate_by_topic_mode_path, index=False)
        if debate_payload["topic_consensus_pivot"] is not None:
            debate_payload["topic_consensus_pivot"].to_csv(debate_topic_pivot_path, index=False)
        if debate_payload["topic_deltas"] is not None:
            debate_payload["topic_deltas"].to_csv(debate_topic_delta_path, index=False)
        if benchmark_payload["by_dataset"] is not None:
            benchmark_payload["by_dataset"].to_csv(benchmark_by_dataset_path, index=False)

        notes_text = self._render_notes(guardrails)
        notes_path.write_text(notes_text, encoding="utf-8")

        manifest = {
            "report_id": report_id,
            "name": report_name,
            "created_at": self._utc_now(),
            "inputs": {
                "experiment_ids": experiment_ids,
                "benchmark_run_ids": benchmark_run_ids,
            },
            "summary": {
                "debate_runs_included": debate_payload["row_count"],
                "debate_modes": debate_payload["modes"],
                "topics": debate_payload["topics"],
                "benchmark_runs_included": benchmark_payload["run_count"],
                "benchmark_datasets": benchmark_payload["datasets"],
            },
            "guardrails": guardrails,
            "files": {
                "debate_by_mode_csv": str(debate_by_mode_path) if debate_payload["by_mode"] is not None else None,
                "debate_by_topic_mode_csv": str(debate_by_topic_mode_path) if debate_payload["by_topic_mode"] is not None else None,
                "debate_topic_consensus_pivot_csv": str(debate_topic_pivot_path) if debate_payload["topic_consensus_pivot"] is not None else None,
                "debate_topic_deltas_csv": str(debate_topic_delta_path) if debate_payload["topic_deltas"] is not None else None,
                "legalbench_by_dataset_csv": str(benchmark_by_dataset_path) if benchmark_payload["by_dataset"] is not None else None,
                "notes_txt": str(notes_path),
            },
        }

        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        manifest["files"]["manifest_json"] = str(manifest_path)
        return manifest

    def _compile_debate_tables(self, experiment_ids: List[str]) -> Dict[str, Any]:
        frames: List[pd.DataFrame] = []
        for experiment_id in experiment_ids:
            results_path = self.experiments_root / experiment_id / "results.csv"
            if not results_path.exists():
                continue
            frame = pd.read_csv(results_path)
            frame["experiment_id"] = frame.get("experiment_id", experiment_id)
            frames.append(frame)

        if not frames:
            return {
                "row_count": 0,
                "modes": [],
                "topics": [],
                "by_mode": None,
                "by_topic_mode": None,
                "topic_consensus_pivot": None,
                "topic_deltas": None,
            }

        combined = pd.concat(frames, ignore_index=True)
        if "status" in combined.columns:
            combined = combined[combined["status"] == "completed"].copy()

        numeric_columns = [
            "consensus_score",
            "avg_info_gain",
            "faithfulness",
            "format_adherence",
            "search_total",
            "search_empty",
            "duration_seconds",
        ]
        for column in numeric_columns:
            if column in combined.columns:
                combined[column] = pd.to_numeric(combined[column], errors="coerce")

        modes = sorted(combined["mode"].dropna().astype(str).unique().tolist()) if "mode" in combined.columns else []
        topics = sorted(combined["topic"].dropna().astype(str).unique().tolist()) if "topic" in combined.columns else []

        by_mode = combined.groupby("mode", dropna=False).agg(
            runs=("mode", "size"),
            avg_consensus_score=("consensus_score", "mean"),
            avg_info_gain=("avg_info_gain", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_format_adherence=("format_adherence", "mean"),
            avg_search_total=("search_total", "mean"),
            avg_search_empty=("search_empty", "mean"),
            avg_duration_seconds=("duration_seconds", "mean"),
            total_searches=("search_total", "sum"),
            total_empty_searches=("search_empty", "sum"),
        ).reset_index()
        by_mode["empty_search_rate"] = by_mode.apply(
            lambda row: (row["total_empty_searches"] / row["total_searches"]) if row["total_searches"] else 0.0,
            axis=1,
        )

        by_topic_mode = combined.groupby(["topic", "mode"], dropna=False).agg(
            runs=("mode", "size"),
            avg_consensus_score=("consensus_score", "mean"),
            avg_info_gain=("avg_info_gain", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_format_adherence=("format_adherence", "mean"),
            avg_search_total=("search_total", "mean"),
            avg_duration_seconds=("duration_seconds", "mean"),
        ).reset_index()

        topic_consensus_pivot = by_topic_mode.pivot(index="topic", columns="mode", values="avg_consensus_score").reset_index()
        topic_deltas = self._build_topic_delta_table(by_topic_mode)

        return {
            "row_count": int(len(combined)),
            "modes": modes,
            "topics": topics,
            "by_mode": by_mode.round(4),
            "by_topic_mode": by_topic_mode.round(4),
            "topic_consensus_pivot": topic_consensus_pivot.round(4),
            "topic_deltas": topic_deltas.round(4) if topic_deltas is not None else None,
        }

    def _build_topic_delta_table(self, by_topic_mode: pd.DataFrame) -> Optional[pd.DataFrame]:
        if by_topic_mode.empty:
            return None

        metrics = ["avg_consensus_score", "avg_info_gain", "avg_faithfulness"]
        pivot_tables = {
            metric: by_topic_mode.pivot(index="topic", columns="mode", values=metric)
            for metric in metrics
        }

        topics = sorted(by_topic_mode["topic"].astype(str).unique().tolist())
        rows: List[Dict[str, Any]] = []
        for topic in topics:
            row: Dict[str, Any] = {"topic": topic}
            for metric, pivot in pivot_tables.items():
                if topic not in pivot.index:
                    continue
                metric_row = pivot.loc[topic]
                comparisons = [
                    ("active_rag", "react_only"),
                    ("active_rag", "naive_rag"),
                    ("hybrid", "react_only"),
                    ("hybrid", "naive_rag"),
                    ("hybrid", "active_rag"),
                ]
                for left_mode, right_mode in comparisons:
                    if left_mode in metric_row.index and right_mode in metric_row.index:
                        left_value = metric_row[left_mode]
                        right_value = metric_row[right_mode]
                        if pd.notna(left_value) and pd.notna(right_value):
                            row[f"{left_mode}_minus_{right_mode}_{metric}"] = float(left_value - right_value)
            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def _compile_benchmark_tables(self, benchmark_run_ids: List[str]) -> Dict[str, Any]:
        dataset_rows: List[Dict[str, Any]] = []
        datasets: List[str] = []
        for run_id in benchmark_run_ids:
            summary_path = self.benchmarks_root / run_id / "summary.json"
            if not summary_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            for benchmark_row in summary.get("benchmarks", []):
                dataset_row = dict(benchmark_row)
                dataset_row["benchmark_run_id"] = run_id
                dataset_rows.append(dataset_row)
                datasets.append(benchmark_row.get("benchmark"))

        if not dataset_rows:
            return {
                "run_count": 0,
                "datasets": [],
                "by_dataset": None,
            }

        dataset_df = pd.DataFrame(dataset_rows)
        metric_columns = [
            "top1_file_hit",
            "file_hit_at_k",
            "file_precision_at_k",
            "file_recall_at_k",
            "best_span_recall",
            "top1_span_recall",
            "answer_coverage_at_k",
        ]
        for column in metric_columns + ["queries_evaluated", "n_results"]:
            if column in dataset_df.columns:
                dataset_df[column] = pd.to_numeric(dataset_df[column], errors="coerce")

        grouped_rows: List[Dict[str, Any]] = []
        for benchmark, group in dataset_df.groupby("benchmark", dropna=False):
            total_queries = max(group["queries_evaluated"].sum(), 1)
            row = {
                "benchmark": benchmark,
                "benchmark_runs": int(len(group)),
                "queries_evaluated": int(group["queries_evaluated"].sum()),
                "avg_n_results": float(group["n_results"].mean()) if "n_results" in group.columns else None,
            }
            for metric in metric_columns:
                row[metric] = float((group[metric] * group["queries_evaluated"]).sum() / total_queries)
            grouped_rows.append(row)

        return {
            "run_count": len(benchmark_run_ids),
            "datasets": sorted({dataset for dataset in datasets if dataset}),
            "by_dataset": pd.DataFrame(grouped_rows).round(4),
        }

    def _build_guardrails(self, debate_payload: Dict[str, Any], benchmark_payload: Dict[str, Any]) -> Dict[str, Any]:
        modes = set(debate_payload["modes"])
        guardrails = {
            "claimable_modes": sorted(modes),
            "missing_modes": sorted({"baseline", "react_only", "naive_rag", "active_rag", "hybrid"} - modes),
            "has_debate_results": debate_payload["row_count"] > 0,
            "has_legalbench_benchmark_results": benchmark_payload["run_count"] > 0,
            "notes": [],
        }

        if debate_payload["row_count"] == 0:
            guardrails["notes"].append("No completed debate experiment rows were included. Do not make debate-performance claims.")
        if benchmark_payload["run_count"] == 0:
            guardrails["notes"].append("No LegalBench benchmark runs were included. Do not make retrieval-benchmark claims.")
        if "active_rag" not in modes:
            guardrails["notes"].append("Active RAG is absent from the aggregated debate results. Do not claim Active RAG performance.")
        if "hybrid" not in modes:
            guardrails["notes"].append("Hybrid is absent from the aggregated debate results. Do not claim Hybrid performance.")
        if debate_payload["row_count"] > 0 and len(debate_payload["topics"]) < 3:
            guardrails["notes"].append("The aggregated debate set uses fewer than 3 topics. Treat conclusions as limited-sample findings.")
        if benchmark_payload["run_count"] > 0 and len(benchmark_payload["datasets"]) < 4:
            guardrails["notes"].append("Not all LegalBench benchmark datasets were included. Benchmark conclusions apply only to the included datasets.")

        return guardrails

    def _render_notes(self, guardrails: Dict[str, Any]) -> str:
        lines = [
            "Report Claim Guardrails",
            f"Generated: {self._utc_now()}",
            "",
            f"Debate results included: {guardrails['has_debate_results']}",
            f"LegalBench benchmark results included: {guardrails['has_legalbench_benchmark_results']}",
            f"Claimable modes: {', '.join(guardrails['claimable_modes']) if guardrails['claimable_modes'] else 'None'}",
        ]
        if guardrails["missing_modes"]:
            lines.append(f"Missing modes: {', '.join(guardrails['missing_modes'])}")
        lines.append("")
        lines.append("Notes:")
        if guardrails["notes"]:
            lines.extend([f"- {note}" for note in guardrails["notes"]])
        else:
            lines.append("- No claim guardrails were triggered by the selected inputs.")
        return "\n".join(lines) + "\n"

    def _discover_experiment_ids(self) -> List[str]:
        return [
            item.name
            for item in sorted(self.experiments_root.iterdir(), key=lambda path: path.name)
            if item.is_dir() and (item / "results.csv").exists()
        ] if self.experiments_root.exists() else []

    def _discover_benchmark_run_ids(self) -> List[str]:
        return [
            item.name
            for item in sorted(self.benchmarks_root.iterdir(), key=lambda path: path.name)
            if item.is_dir() and (item / "summary.json").exists()
        ] if self.benchmarks_root.exists() else []

    def _resolve_path(self, repo_root: Path, path_str: str) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        return (repo_root / candidate).resolve()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


report_compiler_service = ReportCompilerService()
