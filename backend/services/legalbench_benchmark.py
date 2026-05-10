import csv
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from services.rag_service import RAGService


class LegalBenchBenchmarkService:
    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        benchmark_path: str = "LegalBench-RAG/benchmarks",
        output_root: str = "backend/data/legalbench_benchmarks",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = repo_root
        self.benchmark_path = self._resolve_path(repo_root, benchmark_path)
        self.output_root = self._resolve_path(repo_root, output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rag_service = rag_service or RAGService()

    def list_benchmarks(self) -> List[str]:
        return sorted(path.stem for path in self.benchmark_path.glob("*.json"))

    def run_suite(
        self,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        benchmark_names = config.get("benchmarks") or self.list_benchmarks()
        limit_per_benchmark = config.get("limit_per_benchmark")
        n_results = int(config.get("n_results", 5))
        run_id = run_id or str(uuid.uuid4())
        run_name = config.get("name") or f"legalbench_benchmark_{run_id[:8]}"
        started_at = self._utc_now()

        output_dir = self.output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = output_dir / "summary.json"
        summary_csv_path = output_dir / "benchmark_summary.csv"
        query_results_path = output_dir / "query_results.json"

        benchmark_summaries: List[Dict[str, Any]] = []
        query_results: List[Dict[str, Any]] = []

        total_benchmarks = len(benchmark_names)
        for index, benchmark_name in enumerate(benchmark_names, start=1):
            tests = self._load_tests(benchmark_name)
            if limit_per_benchmark is not None:
                tests = tests[: int(limit_per_benchmark)]

            benchmark_query_results = [
                self._score_query(benchmark_name, query_index, test, n_results)
                for query_index, test in enumerate(tests, start=1)
            ]
            query_results.extend(benchmark_query_results)

            benchmark_summary = self._aggregate_results(
                benchmark_name=benchmark_name,
                query_results=benchmark_query_results,
                n_results=n_results,
            )
            benchmark_summaries.append(benchmark_summary)

            if progress_callback:
                progress_callback(
                    {
                        "completed_benchmarks": index,
                        "total_benchmarks": total_benchmarks,
                        "current_benchmark": benchmark_name,
                        "queries_completed": len(query_results),
                    }
                )

        overall_summary = self._aggregate_overall(benchmark_summaries)
        payload = {
            "run_id": run_id,
            "name": run_name,
            "status": "completed",
            "created_at": started_at,
            "completed_at": self._utc_now(),
            "config": {
                "benchmarks": benchmark_names,
                "limit_per_benchmark": limit_per_benchmark,
                "n_results": n_results,
            },
            "overall": overall_summary,
            "benchmarks": benchmark_summaries,
            "files": {
                "summary_json": str(summary_json_path),
                "summary_csv": str(summary_csv_path),
                "query_results_json": str(query_results_path),
            },
        }

        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with query_results_path.open("w", encoding="utf-8") as handle:
            json.dump(query_results, handle, indent=2)

        self._write_summary_csv(summary_csv_path, benchmark_summaries)
        return payload

    def _load_tests(self, benchmark_name: str) -> List[Dict[str, Any]]:
        benchmark_file = self.benchmark_path / f"{benchmark_name}.json"
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

        with benchmark_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        tests = data.get("tests")
        if not isinstance(tests, list):
            raise ValueError(f"Unexpected benchmark format for {benchmark_name}: missing 'tests' list")
        return tests

    def _score_query(
        self,
        benchmark_name: str,
        query_index: int,
        test: Dict[str, Any],
        n_results: int,
    ) -> Dict[str, Any]:
        query = test["query"]
        gold_snippets = test.get("snippets", [])
        gold_files = sorted(
            {
                self._normalize_path(snippet.get("file_path", ""))
                for snippet in gold_snippets
                if snippet.get("file_path")
            }
        )

        retrieved = self.rag_service.query_structured(query, n_results=n_results)
        retrieved_files = [item["relative_path"] for item in retrieved if item.get("relative_path")]
        matched_retrievals = [item for item in retrieved if item.get("relative_path") in gold_files]
        matched_files = sorted({item["relative_path"] for item in matched_retrievals})

        best_span_recall = 0.0
        top1_span_recall = 0.0
        answer_coverage_at_k = False

        for item in retrieved:
            for snippet in gold_snippets:
                snippet_path = self._normalize_path(snippet.get("file_path", ""))
                if item.get("relative_path") != snippet_path:
                    continue

                overlap = self._span_overlap(
                    item.get("start_char"),
                    item.get("end_char"),
                    snippet.get("span", [None, None])[0],
                    snippet.get("span", [None, None])[1],
                )
                best_span_recall = max(best_span_recall, overlap)
                if item["rank"] == 1:
                    top1_span_recall = max(top1_span_recall, overlap)

                if self._contains_answer(item.get("document", ""), snippet.get("answer", "")):
                    answer_coverage_at_k = True

        retrieved_count = len(retrieved) or 1
        top1_file_hit = bool(retrieved_files and retrieved_files[0] in gold_files)
        file_hit_at_k = bool(matched_files)
        file_precision_at_k = len(matched_retrievals) / retrieved_count
        file_recall_at_k = len(matched_files) / max(len(gold_files), 1)

        return {
            "benchmark": benchmark_name,
            "query_index": query_index,
            "query": query,
            "n_gold_files": len(gold_files),
            "gold_files": gold_files,
            "gold_snippets": gold_snippets,
            "retrieved": [
                {
                    "rank": item["rank"],
                    "relative_path": item.get("relative_path"),
                    "filename": item.get("filename"),
                    "category": item.get("category"),
                    "start_char": item.get("start_char"),
                    "end_char": item.get("end_char"),
                    "distance": item.get("distance"),
                    "preview": item.get("preview"),
                }
                for item in retrieved
            ],
            "metrics": {
                "top1_file_hit": top1_file_hit,
                "file_hit_at_k": file_hit_at_k,
                "file_precision_at_k": file_precision_at_k,
                "file_recall_at_k": file_recall_at_k,
                "best_span_recall": best_span_recall,
                "top1_span_recall": top1_span_recall,
                "answer_coverage_at_k": answer_coverage_at_k,
            },
        }

    def _aggregate_results(
        self,
        benchmark_name: str,
        query_results: List[Dict[str, Any]],
        n_results: int,
    ) -> Dict[str, Any]:
        metric_names = [
            "top1_file_hit",
            "file_hit_at_k",
            "file_precision_at_k",
            "file_recall_at_k",
            "best_span_recall",
            "top1_span_recall",
            "answer_coverage_at_k",
        ]
        summary = {
            "benchmark": benchmark_name,
            "queries_evaluated": len(query_results),
            "n_results": n_results,
        }
        for name in metric_names:
            values = [float(result["metrics"][name]) for result in query_results]
            summary[name] = (sum(values) / len(values)) if values else 0.0
        return summary

    def _aggregate_overall(self, benchmark_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        overall = {
            "benchmarks_evaluated": len(benchmark_summaries),
            "queries_evaluated": sum(summary["queries_evaluated"] for summary in benchmark_summaries),
        }
        metric_names = [
            "top1_file_hit",
            "file_hit_at_k",
            "file_precision_at_k",
            "file_recall_at_k",
            "best_span_recall",
            "top1_span_recall",
            "answer_coverage_at_k",
        ]
        for name in metric_names:
            weighted_total = sum(summary[name] * summary["queries_evaluated"] for summary in benchmark_summaries)
            total_queries = overall["queries_evaluated"] or 1
            overall[name] = weighted_total / total_queries
        return overall

    def _write_summary_csv(self, path: Path, benchmark_summaries: List[Dict[str, Any]]):
        fieldnames = [
            "benchmark",
            "queries_evaluated",
            "n_results",
            "top1_file_hit",
            "file_hit_at_k",
            "file_precision_at_k",
            "file_recall_at_k",
            "best_span_recall",
            "top1_span_recall",
            "answer_coverage_at_k",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in benchmark_summaries:
                writer.writerow(row)

    def _contains_answer(self, document: str, answer: str) -> bool:
        normalized_document = self._normalize_text(document)
        normalized_answer = self._normalize_text(answer)
        return bool(normalized_answer) and normalized_answer in normalized_document

    def _normalize_text(self, value: str) -> str:
        return " ".join((value or "").lower().split())

    def _normalize_path(self, value: str) -> str:
        return value.replace("\\", "/").lstrip("./")

    def _span_overlap(
        self,
        retrieved_start: Optional[int],
        retrieved_end: Optional[int],
        gold_start: Optional[int],
        gold_end: Optional[int],
    ) -> float:
        if None in {retrieved_start, retrieved_end, gold_start, gold_end}:
            return 0.0

        retrieved_start = int(retrieved_start)
        retrieved_end = int(retrieved_end)
        gold_start = int(gold_start)
        gold_end = int(gold_end)

        overlap_start = max(retrieved_start, gold_start)
        overlap_end = min(retrieved_end, gold_end)
        if overlap_end <= overlap_start:
            return 0.0

        gold_length = max(gold_end - gold_start, 1)
        return (overlap_end - overlap_start) / gold_length

    def _resolve_path(self, repo_root: Path, path_str: str) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        return (repo_root / candidate).resolve()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class LegalBenchBenchmarkManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LegalBenchBenchmarkManager, cls).__new__(cls)
            cls._instance.runs = {}
            cls._instance._service = None
        return cls._instance

    def start_run(self, config: Dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        benchmarks = config.get("benchmarks") or self._get_service().list_benchmarks()
        self.runs[run_id] = {
            "id": run_id,
            "name": config.get("name", "LegalBench Benchmark"),
            "status": "starting",
            "progress": 0,
            "completed_benchmarks": 0,
            "total_benchmarks": len(benchmarks),
            "queries_completed": 0,
            "config": config,
            "start_time": time.time(),
            "results": None,
        }

        thread = threading.Thread(target=self._run_thread, args=(run_id, config), daemon=True)
        thread.start()
        return run_id

    def get_status(self, run_id: str) -> Dict[str, Any]:
        return self.runs.get(run_id, {"status": "not_found"})

    def list_runs(self) -> List[Dict[str, Any]]:
        return list(self.runs.values())

    def list_benchmarks(self) -> List[str]:
        return self._get_service().list_benchmarks()

    def _run_thread(self, run_id: str, config: Dict[str, Any]):
        run = self.runs[run_id]
        run["status"] = "running"
        try:
            result = self._get_service().run_suite(
                config=config,
                run_id=run_id,
                progress_callback=lambda update: self._handle_progress(run_id, update),
            )
            run["status"] = "completed"
            run["progress"] = 100
            run["completed_benchmarks"] = run["total_benchmarks"]
            run["results"] = result
            run["end_time"] = time.time()
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = str(exc)
            run["end_time"] = time.time()

    def _handle_progress(self, run_id: str, update: Dict[str, Any]):
        run = self.runs.get(run_id)
        if not run:
            return

        completed_benchmarks = update.get("completed_benchmarks", 0)
        total_benchmarks = max(update.get("total_benchmarks", 0), 1)
        run["completed_benchmarks"] = completed_benchmarks
        run["queries_completed"] = update.get("queries_completed", run.get("queries_completed", 0))
        run["current_benchmark"] = update.get("current_benchmark")
        run["progress"] = int((completed_benchmarks / total_benchmarks) * 100)

    def _get_service(self) -> LegalBenchBenchmarkService:
        if self._service is None:
            self._service = LegalBenchBenchmarkService()
        return self._service


legalbench_benchmark_manager = LegalBenchBenchmarkManager()
