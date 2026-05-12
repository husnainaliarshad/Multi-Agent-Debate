from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import requests
import datetime
import traceback
import uuid
from pathlib import Path

import pandas as pd
from fastapi.responses import FileResponse
from core.config import DebateConfig, AgentConfig, Settings, JUDGE_PROFILES
from core.agents import DebateOrchestrator
from core.database import init_db, get_debate_events, get_recent_debates, delete_debate_session, save_debate_session, get_debate_result_from_db
from dotenv import load_dotenv
import os
import threading
from services.experiment_manager import experiment_manager
from services.experiment_catalog import build_experiment_catalog
from services.rag_service import RAGService
from services.experiment_validator import validate_experiment_results
from services.knk_dataset import build_experiment_payload, preview_rows
from services.legalbench_benchmark import legalbench_benchmark_manager
from services.report_compiler import report_compiler_service

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Debate API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Initialize shared RAG service
rag_service = RAGService()

# In-memory session storage
sessions: Dict[str, DebateOrchestrator] = {}
session_results: Dict[str, Dict[str, Any]] = {}
session_locks: Dict[str, threading.Lock] = {}
session_created_at: Dict[str, datetime.datetime] = {}

SESSION_MAX_AGE_HOURS = 24

def cleanup_old_sessions():
    """Remove in-memory sessions older than SESSION_MAX_AGE_HOURS."""
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=SESSION_MAX_AGE_HOURS)
    to_remove = [sid for sid, created in session_created_at.items() if created < cutoff]
    for sid in to_remove:
        sessions.pop(sid, None)
        session_results.pop(sid, None)
        session_locks.pop(sid, None)
        session_created_at.pop(sid, None)
    if to_remove:
        print(f"[Cleanup] Removed {len(to_remove)} stale session(s)")

# Pydantic schemas
class ProposerConfig(BaseModel):
    model: str = "liquid/lfm2.5-1.2b"
    temperature: float = 0.7
    system_prompt: Optional[str] = None

class DebateInitRequest(BaseModel):
    topic: str
    proposers: list[ProposerConfig] = [ProposerConfig()]
    critic_model: Optional[str] = "liquid/lfm2.5-1.2b"
    judge_model: Optional[str] = "liquid/lfm2.5-1.2b"
    critic_temperature: Optional[float] = 0.7
    judge_temperature: Optional[float] = 0.5
    critic_prompt: Optional[str] = None
    judge_prompt: Optional[str] = None
    judge_profile: Optional[str] = "default"
    use_position_swap: Optional[bool] = True
    use_info_gain: Optional[bool] = True
    use_faithfulness: Optional[bool] = True
    use_summary_relay: Optional[bool] = True
    max_rounds: Optional[int] = 1
    max_tokens: Optional[int] = 500
    use_search: Optional[bool] = True
    use_rag: Optional[bool] = False
    model_provider: Optional[str] = "openai"
    mode: Optional[str] = "hybrid"
    force_different_proposers: Optional[bool] = False
    force_different_rounds: Optional[bool] = False
    critic_repetition_check: Optional[bool] = False
    negative_constraints: Optional[bool] = False
    round_specific_topics: Optional[bool] = False
    temperature_annealing: Optional[bool] = False
    judge_intervention: Optional[bool] = False
    perspective_rotation: Optional[bool] = False
    contradiction_detection: Optional[bool] = False
    early_termination_loop: Optional[bool] = False

class DebateInitResponse(BaseModel):
    session_id: str
    status: str

@app.post("/debate/init", response_model=DebateInitResponse)
def init_debate(request: DebateInitRequest):
    """Initialize a new debate session."""
    try:
        # Create proposer configs
        proposer_configs = [
            AgentConfig(
                model=p.model,
                temperature=p.temperature,
                system_prompt=p.system_prompt or "You are a Proposer in a structured debate. Your role is to generate a well-reasoned legal argument on the given topic."
            )
            for p in request.proposers
        ]
        
        # Get judge system prompt from profile or custom prompt
        judge_system_prompt = request.judge_prompt
        if not judge_system_prompt:
            judge_system_prompt = JUDGE_PROFILES.get(request.judge_profile, JUDGE_PROFILES["default"])
        
        # Create debate configuration
        config = DebateConfig(
            proposer=proposer_configs[0] if proposer_configs else AgentConfig(
                model="liquid/lfm2.5-1.2b",
                temperature=0.7,
                system_prompt="You are a Proposer in a structured debate. Your role is to generate a well-reasoned legal argument on the given topic."
            ),
            critic=AgentConfig(
                model=request.critic_model,
                temperature=request.critic_temperature,
                system_prompt=request.critic_prompt or "You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument."
            ),
            judge=AgentConfig(
                model=request.judge_model,
                temperature=request.judge_temperature,
                system_prompt=judge_system_prompt
            ),
            max_rounds=request.max_rounds,
            model_provider=request.model_provider or os.getenv("MODEL_PROVIDER", "openai"),
            base_url=os.getenv("BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("API_KEY", "lm-studio"),
            groq_api_key=os.getenv("GROQ_KEY"),
            use_rag=request.use_rag or False,
            mode=request.mode or "hybrid"
        )
        
        # Create orchestrator with multiple proposers and rounds
        mode_requires_rag = (request.mode or "hybrid") in {"naive_rag", "active_rag", "hybrid"}
        orchestrator = DebateOrchestrator(
            config,
            max_tokens=request.max_tokens or 500,
            proposer_configs=proposer_configs,
            num_rounds=request.max_rounds or 1,
            use_search=request.use_search or False,
            use_position_swap=request.use_position_swap or True,
            use_info_gain=request.use_info_gain or True,
            use_faithfulness=request.use_faithfulness or True,
            use_summary_relay=request.use_summary_relay or True,
            use_rag=request.use_rag or False,
            rag_service=rag_service if (request.use_rag or mode_requires_rag) else None,
            force_different_proposers=request.force_different_proposers or False,
            force_different_rounds=request.force_different_rounds or False,
            critic_repetition_check=request.critic_repetition_check or False,
            negative_constraints=request.negative_constraints or False,
            round_specific_topics=request.round_specific_topics or False,
            temperature_annealing=request.temperature_annealing or False,
            judge_intervention=request.judge_intervention or False,
            perspective_rotation=request.perspective_rotation or False,
            contradiction_detection=request.contradiction_detection or False,
            early_termination_loop=request.early_termination_loop or False
        )
        session_id = orchestrator.session_id
        
        # Store session
        sessions[session_id] = orchestrator
        session_locks[session_id] = threading.Lock()
        session_created_at[session_id] = datetime.datetime.now()
        
        # Run debate in background thread
        thread = threading.Thread(target=run_debate_background, args=(orchestrator, request.topic))
        thread.start()
        
        return DebateInitResponse(session_id=session_id, status="initialized")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize debate: {str(e)}")

def run_debate_background(orchestrator: DebateOrchestrator, topic: str):
    """Run debate in background, store results, and auto-save to DB."""
    try:
        result = orchestrator.run_debate(topic)
        session_results[orchestrator.session_id] = result
        print(f"[{orchestrator.session_id}] Debate complete")

        # Auto-save to database
        try:
            save_debate_session(
                session_id=orchestrator.session_id,
                topic=topic,
                events=orchestrator.events,
                result=result
            )
            print(f"[{orchestrator.session_id}] Auto-saved to database")
        except Exception as db_err:
            print(f"[{orchestrator.session_id}] Auto-save failed: {db_err}")

    except Exception as e:
        error_result = {
            "error": str(e),
            "session_id": orchestrator.session_id,
            "traceback": traceback.format_exc()
        }
        session_results[orchestrator.session_id] = error_result
        print(f"[{orchestrator.session_id}] Error in background thread: {e}")

        # Auto-save error state to database so user can see what happened
        try:
            save_debate_session(
                session_id=orchestrator.session_id,
                topic=topic,
                events=orchestrator.events,
                result=error_result
            )
            print(f"[{orchestrator.session_id}] Auto-saved error state to database")
        except Exception as db_err:
            print(f"[{orchestrator.session_id}] Auto-save of error state failed: {db_err}")

@app.get("/debate/events/{session_id}")
def get_debate_events_endpoint(session_id: str):
    """Get all events for a debate session."""
    cleanup_old_sessions()
    if session_id in sessions:
        orchestrator = sessions[session_id]
        events = orchestrator.events
        is_complete = session_id in session_results
    else:
        # Try loading from database
        events = get_debate_events(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="Session not found")
        # If we have a saved result in the DB, mark as complete
        db_result = get_debate_result_from_db(session_id)
        is_complete = db_result is not None
    
    return {
        "session_id": session_id,
        "events": events,
        "complete": is_complete
    }

@app.get("/debate/result/{session_id}")
def get_debate_result(session_id: str, wait_seconds: int = 5):
    """Get the final result of a debate with optional waiting."""
    import time
    start_time = time.time()
    
    while session_id not in session_results and (time.time() - start_time) < wait_seconds:
        time.sleep(1)
    
    if session_id in session_results:
        return session_results[session_id]
    
    # Fallback to database for saved debates
    db_result = get_debate_result_from_db(session_id)
    if db_result is not None:
        return db_result
    
    raise HTTPException(status_code=404, detail="Result not available yet")

class SaveDebateRequest(BaseModel):
    session_id: str
    topic: str
    events: list
    result: dict

@app.post("/debate/save")
def save_debate_endpoint(request: SaveDebateRequest):
    """Save a debate session to the database."""
    try:
        save_debate_session(
            request.session_id,
            request.topic,
            request.events,
            request.result
        )
        return {"message": "Debate saved successfully", "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save debate: {str(e)}")

@app.delete("/debate/{session_id}")
def delete_debate_endpoint(session_id: str):
    """Delete a debate session from the database."""
    try:
        delete_debate_session(session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.get("/debates/recent")
def get_recent_debates_endpoint():
    """List recent debate sessions from database."""
    try:
        recent = get_recent_debates(limit=10)
        return {"sessions": recent}
    except Exception as e:
        print(f"Error fetching recent debates: {e}")
        return {"sessions": []}

@app.get("/models")
def get_available_models():
    """Get available models from LM Studio."""
    try:
        base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
        print(f"Fetching models from: {base_url}/models")
        response = requests.get(f"{base_url}/models", timeout=5)
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Parsed data: {data}")
            models_data = data.get("data", [])
            model_names = [model.get("id", "") for model in models_data]
            # Add Groq models if API key is present
            groq_key = os.getenv("GROQ_KEY")
            if groq_key:
                groq_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
                return {"models": model_names + groq_models, "groq_models": groq_models}
                
            return {"models": model_names}
        else:
            # Return default models if LM Studio is not responding
            groq_key = os.getenv("GROQ_KEY")
            defaults = ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"]
            if groq_key:
                groq_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
                return {"models": defaults + groq_models, "groq_models": groq_models, "warning": "Could not connect to LM Studio, but Groq is available"}
            
            return {
                "models": defaults,
                "warning": f"Could not connect to LM Studio (status {response.status_code}), using default models"
            }
    except Exception as e:
        # Return default models on error
        print(f"Error fetching models: {str(e)}")
        groq_key = os.getenv("GROQ_KEY")
        defaults = ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"]
        if groq_key:
            groq_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
            return {"models": defaults + groq_models, "groq_models": groq_models, "warning": "Error fetching local models, but Groq is available"}
            
        return {
            "models": defaults,
            "warning": f"Error fetching models: {str(e)}"
        }

@app.get("/debate/dummy")
def dummy_debate():
    """Return a dummy debate result for testing the frontend."""
    return {
        "session_id": "dummy_session_123",
        "proposer_responses": [["This is a dummy proposer argument about the topic. It incorporates evidence from the search results below."]],
        "critic_responses": [["This is a dummy critic critique of the argument."]],
        "search_results": [["Based on our search, we found that 95% of legal experts agree that dummy topics are useful for testing."]],
        "judge_response": "This is a dummy judge verdict with a consensus score.",
        "consensus_score": 75,
        "verdict": "Partially valid",
        "num_proposers": 1,
        "num_rounds": 1,
        "events": [
            {"event_type": "DEBATE_START", "data": {"topic": "Dummy Topic"}, "timestamp": 1234567890},
            {"event_type": "SEARCH_COMPLETE", "data": {"proposer_id": 1, "results": "Based on our search, we found that 95% of legal experts agree that dummy topics are useful for testing."}, "timestamp": 1234567891},
            {"event_type": "PROPOSER_FINAL", "data": {"response": "Dummy response"}, "timestamp": 1234567892}
        ]
    }

@app.get("/")
def root():
    return {"status": "running", "message": "Multi-Agent Debate Backend API"}

# Experiment Endpoints
class KnKDatasetSpec(BaseModel):
    """Load puzzles from Hugging Face `K-and-K/knights-and-knaves` as experiment topics."""

    config_name: str = "test"
    split: str = "2ppl"
    limit: Optional[int] = None
    offset: int = 0
    shuffle: bool = False
    seed: Optional[int] = None
    add_topic_suffix: bool = True


class ExperimentInitRequest(BaseModel):
    name: str
    topics: List[str] = Field(default_factory=list)
    knk_dataset: Optional[KnKDatasetSpec] = None
    model_configs: List[Dict[str, Any]]
    max_rounds: Optional[int] = 1
    repeats: Optional[int] = 1
    use_rag: Optional[bool] = False
    use_search: Optional[bool] = True
    force_different_proposers: Optional[bool] = False
    force_different_rounds: Optional[bool] = False
    critic_repetition_check: Optional[bool] = False
    negative_constraints: Optional[bool] = False
    round_specific_topics: Optional[bool] = False
    temperature_annealing: Optional[bool] = False
    judge_intervention: Optional[bool] = False
    perspective_rotation: Optional[bool] = False
    contradiction_detection: Optional[bool] = False
    early_termination_loop: Optional[bool] = False

class LegalBenchBenchmarkRunRequest(BaseModel):
    name: Optional[str] = "LegalBench Retrieval Benchmark"
    benchmarks: Optional[List[str]] = None
    limit_per_benchmark: Optional[int] = None
    n_results: Optional[int] = 5

class ReportCompileRequest(BaseModel):
    name: Optional[str] = "Final Report Summary"
    experiment_ids: Optional[List[str]] = None
    benchmark_run_ids: Optional[List[str]] = None

@app.post("/experiments/run")
def run_experiment_endpoint(request: ExperimentInitRequest):
    """Start a batch experiment."""
    try:
        if request.knk_dataset:
            spec = request.knk_dataset
            knk_gold, topics = build_experiment_payload(
                config_name=spec.config_name,
                split=spec.split,
                offset=spec.offset,
                limit=spec.limit,
                shuffle=spec.shuffle,
                seed=spec.seed,
                add_topic_suffix=spec.add_topic_suffix,
            )
            payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            payload["topics"] = topics
            payload["knk_gold"] = knk_gold
            payload.pop("knk_dataset", None)
        else:
            if not request.topics:
                raise HTTPException(
                    status_code=400,
                    detail="Provide a non-empty topics list or knk_dataset to run an experiment.",
                )
            payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()

        experiment_id = experiment_manager.start_experiment(payload)
        return {"experiment_id": experiment_id, "status": "started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")

@app.get("/experiments/list")
def list_experiments_endpoint():
    """List in-process experiments (sorted by name)."""
    return {"experiments": experiment_manager.list_experiments()}


@app.get("/experiments/catalog")
def experiments_catalog_endpoint():
    """All experiments sorted by display name: live runs + disk index + log scan."""
    return {"experiments": build_experiment_catalog()}

@app.get("/experiments/status/{experiment_id}")
def get_experiment_status_endpoint(experiment_id: str):
    """Get the status of an experiment."""
    status = experiment_manager.get_status(experiment_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Experiment not found")
    return status

@app.get("/experiments/validate/{experiment_id}")
def validate_experiment_endpoint(experiment_id: str):
    """Validate experiment outputs and summarize result quality."""
    try:
        return validate_experiment_results(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate experiment: {str(e)}")


def _experiment_results_csv_path(experiment_id: str) -> Path:
    """Resolve results.csv for a UUID experiment folder (no path traversal)."""
    exp_uuid = str(uuid.UUID(experiment_id))
    base = Path(__file__).resolve().parent / "data" / "experiments"
    csv_path = (base / exp_uuid / "results.csv").resolve()
    if not str(csv_path).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return csv_path


@app.get("/experiments/{experiment_id}/results")
def get_experiment_results_json(experiment_id: str):
    """Return experiment `results.csv` as JSON (columns + rows)."""
    try:
        csv_path = _experiment_results_csv_path(experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment id")
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="results.csv not found for this experiment")
    df = pd.read_csv(csv_path)
    # Round-trip through pandas JSON so NaN/Inf become null (stdlib json cannot encode float nan).
    rows = json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
    return {
        "experiment_id": str(uuid.UUID(experiment_id)),
        "columns": [str(c) for c in df.columns],
        "rows": rows,
        "row_count": int(len(df)),
    }


@app.get("/experiments/{experiment_id}/results.csv")
def download_experiment_results_csv(experiment_id: str):
    """Download raw `results.csv` for an experiment."""
    try:
        csv_path = _experiment_results_csv_path(experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment id")
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="results.csv not found for this experiment")
    exp_uuid = str(uuid.UUID(experiment_id))
    return FileResponse(
        path=str(csv_path),
        filename=f"experiment_{exp_uuid}_results.csv",
        media_type="text/csv",
    )

@app.get("/benchmarks/knk/preview")
def knk_preview_endpoint(
    config_name: str = "test",
    split: str = "2ppl",
    limit: int = 5,
    offset: int = 0,
):
    """Preview rows from `K-and-K/knights-and-knaves` (requires `datasets` package and network on first load)."""
    try:
        return preview_rows(config_name=config_name, split=split, limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load K&K dataset: {str(e)}")

@app.get("/benchmarks/legalbench/datasets")
def list_legalbench_datasets_endpoint():
    """List available LegalBench benchmark datasets."""
    try:
        return {"benchmarks": legalbench_benchmark_manager.list_benchmarks()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list benchmarks: {str(e)}")

@app.post("/benchmarks/legalbench/run")
def run_legalbench_benchmark_endpoint(request: LegalBenchBenchmarkRunRequest):
    """Start a LegalBench retrieval benchmark run."""
    try:
        run_id = legalbench_benchmark_manager.start_run(request.dict())
        return {"run_id": run_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start LegalBench benchmark: {str(e)}")

@app.get("/benchmarks/legalbench/status/{run_id}")
def get_legalbench_benchmark_status_endpoint(run_id: str):
    """Get the status of a LegalBench retrieval benchmark run."""
    status = legalbench_benchmark_manager.get_status(run_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return status

@app.get("/benchmarks/legalbench/list")
def list_legalbench_benchmark_runs_endpoint():
    """List LegalBench retrieval benchmark runs started in this process."""
    return {"runs": legalbench_benchmark_manager.list_runs()}

class BenchmarkRunRequest(BaseModel):
    benchmarks: list[str]
    queries_per_benchmark: int = 10

@app.post("/benchmarks/run")
def run_benchmark_endpoint(request: BenchmarkRunRequest):
    """Run LegalBench benchmarks on demand."""
    try:
        from services.rag_service import RAGService
        from services.legalbench_benchmark import LegalBenchBenchmarkService
        
        rag_service = RAGService()
        benchmark_service = LegalBenchBenchmarkService(rag_service=rag_service)
        
        config = {
            "benchmarks": request.benchmarks,
            "limit_per_benchmark": request.queries_per_benchmark,
            "n_results": 5
        }
        
        result = benchmark_service.run_suite(config)
        
        # Simplify result for frontend - map actual metric names
        overall = result.get("overall", {})
        overall_precision = overall.get("file_precision_at_k", 0)
        overall_recall = overall.get("file_recall_at_k", 0)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        simplified = {
            "overall": {
                "avg_precision": overall_precision,
                "avg_recall": overall_recall,
                "overall_f1": overall_f1
            },
            "benchmarks": []
        }
        
        for bench in result.get("benchmarks", []):
            precision = bench.get("file_precision_at_k", 0)
            recall = bench.get("file_recall_at_k", 0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            simplified["benchmarks"].append({
                "name": bench.get("benchmark", "Unknown"),
                "avg_precision": precision,
                "avg_recall": recall,
                "f1_score": f1,
                "num_queries": bench.get("queries_evaluated", 0)
            })
        
        return simplified
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run benchmarks: {str(e)}")

@app.get("/reports/sources")
def list_report_sources_endpoint():
    """List completed experiment and benchmark outputs that can feed a final report."""
    try:
        return report_compiler_service.list_available_sources()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list report sources: {str(e)}")

@app.post("/reports/compile")
def compile_report_endpoint(request: ReportCompileRequest):
    """Compile report-ready debate and benchmark summary tables."""
    try:
        report = report_compiler_service.compile_report(request.dict())
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile report: {str(e)}")

@app.get("/prompts/export")
async def export_system_prompts():
    from utils.export_prompts import export_prompts
    try:
        file_path = export_prompts()
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "filename": "prompts.txt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
