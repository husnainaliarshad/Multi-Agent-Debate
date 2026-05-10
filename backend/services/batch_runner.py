import csv
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.agents import DebateOrchestrator
from core.config import AgentConfig, DebateConfig
from services.rag_service import RAGService


class BatchRunner:
    MODE_DEFAULTS = {
        "baseline": {"use_search": False, "use_rag": False},
        "react_only": {"use_search": True, "use_rag": False},
        "naive_rag": {"use_search": False, "use_rag": True},
        "active_rag": {"use_search": False, "use_rag": True},
        "hybrid": {"use_search": True, "use_rag": True},
    }

    CSV_FIELDS = [
        "experiment_id",
        "experiment_name",
        "run_index",
        "total_runs",
        "status",
        "error",
        "topic",
        "mode",
        "provider",
        "repeat_index",
        "requested_rounds",
        "actual_rounds",
        "num_proposers",
        "proposer_model",
        "critic_model",
        "judge_model",
        "use_rag",
        "use_search",
        "session_id",
        "verdict",
        "consensus_score",
        "avg_info_gain",
        "format_adherence",
        "faithfulness",
        "search_total",
        "search_empty",
        "spb_score",
        "event_count",
        "started_at",
        "completed_at",
        "duration_seconds",
    ]

    def __init__(self, experiment_id: str = None, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.repo_root = Path(__file__).resolve().parents[2]
        self.results_dir = self.repo_root / "backend" / "data" / "experiments" / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "results.csv"
        self.logs_file = self.results_dir / "experiment_log.json"
        self.runs = []
        self.progress_callback = progress_callback
        self._shared_rag_service: Optional[RAGService] = None

    def run_experiment(self, config: Dict[str, Any]):
        """
        Run a batch experiment based on config.
        Config structure:
        {
            "name": str,
            "topics": List[str],
            "model_configs": List[Dict],
            "max_rounds": int,
            "repeats": int,
            "use_rag": bool,
            "use_search": bool
        }
        """
        experiment_name = config.get("name", "Unnamed Experiment")
        topics = config.get("topics", [])
        model_configs = config.get("model_configs", [{}])
        max_rounds = config.get("max_rounds", 1)
        repeats = config.get("repeats", 1)
        total_runs = len(topics) * len(model_configs) * repeats

        print(f"Starting Experiment: {experiment_name} (ID: {self.experiment_id})")
        self._write_log_snapshot(experiment_name, config, status="running", total_runs=total_runs)

        current_run = 0
        for topic in topics:
            for m_cfg in model_configs:
                for repeat_index in range(repeats):
                    current_run += 1
                    metadata = self._build_run_metadata(
                        experiment_name=experiment_name,
                        topic=topic,
                        model_config=m_cfg,
                        repeat_index=repeat_index,
                        run_index=current_run,
                        total_runs=total_runs,
                        requested_rounds=max_rounds,
                        default_use_rag=config.get("use_rag", False),
                        default_use_search=config.get("use_search", True),
                    )
                    print(
                        f"[{current_run}/{total_runs}] Running: "
                        f"Topic='{topic[:30]}...', Mode={metadata['mode']}, Model={metadata['proposer_model']}"
                    )

                    started_at = self._utc_now()
                    started_ts = time.time()
                    try:
                        orchestrator = self._setup_orchestrator(metadata)
                        result = orchestrator.run_debate(topic)
                        record = self._build_success_record(result, metadata, started_at, started_ts)
                    except Exception as exc:
                        print(f"Error in run {current_run}: {exc}")
                        record = self._build_failure_record(exc, metadata, started_at, started_ts)

                    self.runs.append(record)
                    self._save_result_to_csv(record)
                    self._write_log_snapshot(experiment_name, config, status="running", total_runs=total_runs)
                    self._emit_progress(record)

        self._write_log_snapshot(experiment_name, config, status="completed", total_runs=total_runs)
        print(f"Experiment {experiment_name} complete. Results saved to {self.results_dir}")
        return str(self.results_file)

    def _build_run_metadata(
        self,
        experiment_name: str,
        topic: str,
        model_config: Dict[str, Any],
        repeat_index: int,
        run_index: int,
        total_runs: int,
        requested_rounds: int,
        default_use_rag: bool,
        default_use_search: bool,
    ) -> Dict[str, Any]:
        proposer_model = model_config.get("proposer_model", "liquid/lfm2.5-1.2b")
        critic_model = model_config.get("critic_model", proposer_model)
        judge_model = model_config.get("judge_model", proposer_model)
        mode = model_config.get("mode", "hybrid")
        mode_defaults = self.MODE_DEFAULTS.get(mode, {"use_search": default_use_search, "use_rag": default_use_rag})

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "run_index": run_index,
            "total_runs": total_runs,
            "topic": topic,
            "mode": mode,
            "provider": model_config.get("provider", "openai"),
            "repeat_index": repeat_index,
            "requested_rounds": requested_rounds,
            "proposer_model": proposer_model,
            "critic_model": critic_model,
            "judge_model": judge_model,
            "use_rag": bool(model_config.get("use_rag", mode_defaults["use_rag"])),
            "use_search": bool(model_config.get("use_search", mode_defaults["use_search"])),
        }

    def _setup_orchestrator(self, metadata: Dict[str, Any]) -> DebateOrchestrator:
        proposer_config = AgentConfig(
            model=metadata["proposer_model"],
            temperature=0.7,
        )

        config = DebateConfig(
            proposer=proposer_config,
            critic=AgentConfig(
                model=metadata["critic_model"],
                temperature=0.7,
            ),
            judge=AgentConfig(
                model=metadata["judge_model"],
                temperature=0.5,
            ),
            max_rounds=metadata["requested_rounds"],
            model_provider=metadata["provider"],
            base_url=os.getenv("BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("API_KEY", "lm-studio"),
            groq_api_key=os.getenv("GROQ_KEY"),
            use_rag=metadata["use_rag"],
            mode=metadata["mode"],
        )

        return DebateOrchestrator(
            config,
            max_tokens=500,
            proposer_configs=[proposer_config],
            num_rounds=metadata["requested_rounds"],
            use_search=metadata["use_search"],
            use_rag=metadata["use_rag"],
            rag_service=self._get_shared_rag_service() if metadata["use_rag"] else None,
        )

    def _get_shared_rag_service(self) -> RAGService:
        if self._shared_rag_service is None:
            self._shared_rag_service = RAGService()
        return self._shared_rag_service

    def _build_success_record(
        self,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        started_at: str,
        started_ts: float,
    ) -> Dict[str, Any]:
        record = dict(result)
        record.update(metadata)
        record["status"] = "completed"
        record["error"] = ""
        record["started_at"] = started_at
        record["completed_at"] = self._utc_now()
        record["duration_seconds"] = round(time.time() - started_ts, 3)
        record["actual_rounds"] = result.get("num_rounds", metadata["requested_rounds"])
        record["event_count"] = len(result.get("events", []))
        return record

    def _build_failure_record(
        self,
        exc: Exception,
        metadata: Dict[str, Any],
        started_at: str,
        started_ts: float,
    ) -> Dict[str, Any]:
        return {
            **metadata,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started_at,
            "completed_at": self._utc_now(),
            "duration_seconds": round(time.time() - started_ts, 3),
            "actual_rounds": 0,
            "num_proposers": 0,
            "event_count": 0,
        }

    def _save_result_to_csv(self, result: Dict[str, Any]):
        """Flatten one run record and append it to the experiment CSV."""
        metrics = result.get("metrics", {})
        faithfulness_scores = metrics.get("turn_faithfulness", [])
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0

        row = {
            "experiment_id": result.get("experiment_id"),
            "experiment_name": result.get("experiment_name"),
            "run_index": result.get("run_index"),
            "total_runs": result.get("total_runs"),
            "status": result.get("status"),
            "error": result.get("error", ""),
            "topic": result.get("topic"),
            "mode": result.get("mode"),
            "provider": result.get("provider"),
            "repeat_index": result.get("repeat_index"),
            "requested_rounds": result.get("requested_rounds"),
            "actual_rounds": result.get("actual_rounds", result.get("num_rounds")),
            "num_proposers": result.get("num_proposers", 0),
            "proposer_model": result.get("proposer_model"),
            "critic_model": result.get("critic_model"),
            "judge_model": result.get("judge_model"),
            "use_rag": result.get("use_rag"),
            "use_search": result.get("use_search"),
            "session_id": result.get("session_id", ""),
            "verdict": result.get("verdict", ""),
            "consensus_score": result.get("consensus_score", ""),
            "avg_info_gain": metrics.get("average_information_gain", 0),
            "format_adherence": metrics.get("format_adherence_percent", 0),
            "faithfulness": avg_faithfulness,
            "search_total": metrics.get("search_efficiency", {}).get("total_searches", 0),
            "search_empty": metrics.get("search_efficiency", {}).get("empty_searches", 0),
            "spb_score": metrics.get("spb_score", 0),
            "event_count": result.get("event_count", len(result.get("events", []))),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "duration_seconds": result.get("duration_seconds"),
        }

        file_exists = self.results_file.is_file()
        with self.results_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_log_snapshot(self, experiment_name: str, config: Dict[str, Any], status: str, total_runs: int):
        succeeded_runs = sum(1 for run in self.runs if run.get("status") == "completed")
        failed_runs = sum(1 for run in self.runs if run.get("status") == "failed")
        payload = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "status": status,
            "results_file": str(self.results_file),
            "total_runs": total_runs,
            "completed_runs": len(self.runs),
            "succeeded_runs": succeeded_runs,
            "failed_runs": failed_runs,
            "config": config,
            "updated_at": self._utc_now(),
            "runs": self.runs,
        }
        with self.logs_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _emit_progress(self, record: Dict[str, Any]):
        if not self.progress_callback:
            return

        succeeded_runs = sum(1 for run in self.runs if run.get("status") == "completed")
        failed_runs = sum(1 for run in self.runs if run.get("status") == "failed")
        self.progress_callback(
            {
                "run_index": record.get("run_index", 0),
                "total_runs": record.get("total_runs", 0),
                "status": record.get("status"),
                "succeeded_runs": succeeded_runs,
                "failed_runs": failed_runs,
                "results_path": str(self.results_file),
                "last_topic": record.get("topic"),
            }
        )

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    test_config = {
        "name": "Small Test Run",
        "topics": ["Should AI have legal rights?"],
        "model_configs": [
            {"proposer_model": "liquid/lfm2.5-1.2b", "provider": "openai"}
        ],
        "max_rounds": 1,
        "repeats": 1,
        "use_rag": False,
        "use_search": False,
    }
    runner = BatchRunner("test_exp")
    runner.run_experiment(test_config)
