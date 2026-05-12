import threading
import time
import uuid
from typing import Any, Dict, List

from services.batch_runner import BatchRunner


class ExperimentManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExperimentManager, cls).__new__(cls)
            cls._instance.experiments = {}
            cls._instance.locks = {}
        return cls._instance

    def start_experiment(self, config: Dict[str, Any]) -> str:
        experiment_id = str(uuid.uuid4())
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": config.get("name", "Unnamed Experiment"),
            "status": "starting",
            "progress": 0,
            "total_runs": self._calculate_total_runs(config),
            "completed_runs": 0,
            "succeeded_runs": 0,
            "failed_runs": 0,
            "results_path": None,
            "start_time": time.time(),
            "config": config,
        }

        thread = threading.Thread(target=self._run_experiment_thread, args=(experiment_id, config), daemon=True)
        thread.start()
        return experiment_id

    def _calculate_total_runs(self, config: Dict[str, Any]) -> int:
        topics = len(config.get("topics", []))
        model_configs = len(config.get("model_configs", [{}]))
        repeats = config.get("repeats", 1)
        return topics * model_configs * repeats

    def _run_experiment_thread(self, experiment_id: str, config: Dict[str, Any]):
        try:
            exp = self.experiments[experiment_id]
            exp["status"] = "running"

            runner = BatchRunner(
                experiment_id,
                progress_callback=lambda update: self._handle_progress(experiment_id, update),
            )
            results_file = runner.run_experiment(config)

            exp["status"] = "completed"
            exp["progress"] = 100
            exp["results_path"] = results_file
            exp["end_time"] = time.time()
            exp["completed_runs"] = len(runner.runs)
            exp["succeeded_runs"] = sum(1 for run in runner.runs if run.get("status") == "completed")
            exp["failed_runs"] = sum(1 for run in runner.runs if run.get("status") == "failed")
        except Exception as exc:
            print(f"Error in experiment {experiment_id}: {exc}")
            if experiment_id in self.experiments:
                self.experiments[experiment_id]["status"] = "failed"
                self.experiments[experiment_id]["error"] = str(exc)

    def _handle_progress(self, experiment_id: str, update: Dict[str, Any]):
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        total_runs = max(update.get("total_runs", 0), 1)
        completed_runs = update.get("run_index", 0)
        exp["completed_runs"] = completed_runs
        exp["succeeded_runs"] = update.get("succeeded_runs", exp.get("succeeded_runs", 0))
        exp["failed_runs"] = update.get("failed_runs", exp.get("failed_runs", 0))
        exp["progress"] = int((completed_runs / total_runs) * 100)
        exp["results_path"] = update.get("results_path", exp.get("results_path"))
        exp["last_topic"] = update.get("last_topic")

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        return self.experiments.get(experiment_id, {"status": "not_found"})

    def list_experiments(self) -> List[Dict[str, Any]]:
        rows = list(self.experiments.values())
        rows.sort(
            key=lambda e: (
                (e.get("name") or "zzz").lower(),
                -float(e.get("start_time") or 0),
            )
        )
        return rows


experiment_manager = ExperimentManager()
