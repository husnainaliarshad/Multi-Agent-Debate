import json
import csv
import os
import time
import uuid
from typing import List, Dict, Any
from core.agents import DebateOrchestrator
from core.config import AgentConfig, DebateConfig
import pandas as pd

class BatchRunner:
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.results_dir = f"backend/data/experiments/{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = os.path.join(self.results_dir, "results.csv")
        self.logs_file = os.path.join(self.results_dir, "experiment_log.json")
        self.runs = []

    def run_experiment(self, config: Dict[str, Any]):
        """
        Run a batch experiment based on config.
        Config structure:
        {
            "name": str,
            "topics": List[str],
            "model_configs": List[Dict], # List of {proposer_model, critic_model, judge_model, provider}
            "max_rounds": int,
            "repeats": int,
            "use_rag": bool,
            "use_search": bool
        }
        """
        name = config.get("name", "Unnamed Experiment")
        topics = config.get("topics", [])
        model_configs = config.get("model_configs", [{}])
        max_rounds = config.get("max_rounds", 1)
        repeats = config.get("repeats", 1)
        use_rag = config.get("use_rag", False)
        use_search = config.get("use_search", True)

        print(f"Starting Experiment: {name} (ID: {self.experiment_id})")
        
        total_runs = len(topics) * len(model_configs) * repeats
        current_run = 0

        for topic in topics:
            for m_cfg in model_configs:
                for r in range(repeats):
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] Running: Topic='{topic[:30]}...', Model={m_cfg.get('proposer_model')}")
                    
                    try:
                        # Setup individual debate config
                        orchestrator = self._setup_orchestrator(topic, m_cfg, max_rounds, use_rag, use_search)
                        result = orchestrator.run_debate(topic)
                        
                        # Add metadata to result
                        result["experiment_id"] = self.experiment_id
                        result["repeat_index"] = r
                        result["proposer_model"] = m_cfg.get("proposer_model")
                        result["critic_model"] = m_cfg.get("critic_model")
                        result["judge_model"] = m_cfg.get("judge_model")
                        result["use_rag"] = use_rag
                        result["use_search"] = use_search
                        
                        self.runs.append(result)
                        self._save_result_to_csv(result)
                        
                    except Exception as e:
                        print(f"Error in run {current_run}: {e}")
                        self.runs.append({
                            "topic": topic,
                            "error": str(e),
                            "experiment_id": self.experiment_id
                        })

        # Final save of full logs
        with open(self.logs_file, "w") as f:
            json.dump(self.runs, f, indent=2)
            
        print(f"Experiment {name} complete. Results saved to {self.results_dir}")
        return self.results_file

    def _setup_orchestrator(self, topic: str, m_cfg: Dict, max_rounds: int, use_rag: bool, use_search: bool):
        # Create proposer configs
        proposer_config = AgentConfig(
            model=m_cfg.get("proposer_model", "liquid/lfm2.5-1.2b"),
            temperature=0.7
        )
        
        # Create debate configuration
        config = DebateConfig(
            proposer=proposer_config,
            critic=AgentConfig(
                model=m_cfg.get("critic_model", "liquid/lfm2.5-1.2b"),
                temperature=0.7
            ),
            judge=AgentConfig(
                model=m_cfg.get("judge_model", "liquid/lfm2.5-1.2b"),
                temperature=0.5
            ),
            max_rounds=max_rounds,
            model_provider=m_cfg.get("provider", "openai"),
            groq_api_key=os.getenv("GROQ_KEY"),
            use_rag=use_rag,
            mode=m_cfg.get("mode", "hybrid")
        )
        
        orchestrator = DebateOrchestrator(
            config,
            max_tokens=500,
            proposer_configs=[proposer_config],
            num_rounds=max_rounds,
            use_search=use_search,
            use_rag=use_rag
        )
        return orchestrator

    def _save_result_to_csv(self, result: Dict[str, Any]):
        """Flatten result and save to CSV."""
        metrics = result.get("metrics", {})
        
        row = {
            "session_id": result.get("session_id"),
            "topic": result.get("topic"),
            "mode": result.get("mode"),
            "proposer_model": result.get("proposer_model"),
            "critic_model": result.get("critic_model"),
            "judge_model": result.get("judge_model"),
            "use_rag": result.get("use_rag"),
            "use_search": result.get("use_search"),
            "verdict": result.get("verdict"),
            "consensus_score": result.get("consensus_score"),
            "avg_info_gain": metrics.get("average_information_gain", 0),
            "format_adherence": metrics.get("format_adherence_percent", 0),
            "faithfulness": sum(metrics.get("turn_faithfulness", [0])) / max(len(metrics.get("turn_faithfulness", [1])), 1),
            "search_total": metrics.get("search_efficiency", {}).get("total_searches", 0),
            "search_empty": metrics.get("search_efficiency", {}).get("empty_searches", 0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write header if file doesn't exist
        file_exists = os.path.isfile(self.results_file)
        with open(self.results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

if __name__ == "__main__":
    # Example usage
    test_config = {
        "name": "Small Test Run",
        "topics": ["Should AI have legal rights?"],
        "model_configs": [
            {"proposer_model": "liquid/lfm2.5-1.2b", "provider": "openai"}
        ],
        "max_rounds": 1,
        "repeats": 1,
        "use_rag": True
    }
    runner = BatchRunner("test_exp")
    runner.run_experiment(test_config)
