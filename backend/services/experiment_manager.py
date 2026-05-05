import threading
import uuid
import time
import os
import json
from typing import Dict, Any, List
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
            "results_path": None,
            "start_time": time.time(),
            "config": config
        }
        
        # Run in background thread
        thread = threading.Thread(target=self._run_experiment_thread, args=(experiment_id, config))
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
            
            runner = BatchRunner(experiment_id)
            
            # We wrap the runner.run_experiment logic to update progress
            # For simplicity, we'll let BatchRunner handle the core but maybe add a callback?
            # Let's just update periodically by checking the results.csv or similar
            
            # For now, let's just run it
            results_file = runner.run_experiment(config)
            
            exp["status"] = "completed"
            exp["progress"] = 100
            exp["results_path"] = results_file
            exp["end_time"] = time.time()
            
        except Exception as e:
            print(f"Error in experiment {experiment_id}: {e}")
            if experiment_id in self.experiments:
                self.experiments[experiment_id]["status"] = "failed"
                self.experiments[experiment_id]["error"] = str(e)

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        return self.experiments.get(experiment_id, {"status": "not_found"})

    def list_experiments(self) -> List[Dict[str, Any]]:
        return list(self.experiments.values())

# Global manager instance
experiment_manager = ExperimentManager()
