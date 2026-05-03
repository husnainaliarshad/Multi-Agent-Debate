from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import requests
from config import DebateConfig, AgentConfig, Settings, JUDGE_PROFILES
from agents import DebateOrchestrator
from database import init_db, get_debate_events, get_recent_debates, delete_debate_session, save_debate_session
from dotenv import load_dotenv
import os
import threading

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

# In-memory session storage
sessions: Dict[str, DebateOrchestrator] = {}
session_results: Dict[str, Dict[str, Any]] = {}
session_locks: Dict[str, threading.Lock] = {}

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
            model_provider=os.getenv("MODEL_PROVIDER", "openai"),
            base_url=os.getenv("BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("API_KEY", "lm-studio")
        )
        
        # Create orchestrator with multiple proposers and rounds
        orchestrator = DebateOrchestrator(
            config,
            max_tokens=request.max_tokens or 500,
            proposer_configs=proposer_configs,
            num_rounds=request.max_rounds or 1,
            use_search=request.use_search or False,
            use_position_swap=request.use_position_swap or True,
            use_info_gain=request.use_info_gain or True,
            use_faithfulness=request.use_faithfulness or True,
            use_summary_relay=request.use_summary_relay or True
        )
        session_id = orchestrator.session_id
        
        # Store session
        sessions[session_id] = orchestrator
        session_locks[session_id] = threading.Lock()
        
        # Run debate in background thread
        thread = threading.Thread(target=run_debate_background, args=(orchestrator, request.topic))
        thread.start()
        
        return DebateInitResponse(session_id=session_id, status="initialized")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize debate: {str(e)}")

def run_debate_background(orchestrator: DebateOrchestrator, topic: str):
    """Run debate in background and store results."""
    try:
        result = orchestrator.run_debate(topic)
        session_results[orchestrator.session_id] = result
        print(f"[{orchestrator.session_id}] Debate complete")
    except Exception as e:
        session_results[orchestrator.session_id] = {
            "error": str(e),
            "session_id": orchestrator.session_id
        }
        print(f"[{orchestrator.session_id}] Error in background thread: {e}")

@app.get("/debate/events/{session_id}")
def get_debate_events_endpoint(session_id: str):
    """Get all events for a debate session."""
    if session_id in sessions:
        orchestrator = sessions[session_id]
        events = orchestrator.events
    else:
        # Try loading from database
        events = get_debate_events(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "events": events,
        "complete": session_id in session_results
    }

@app.get("/debate/result/{session_id}")
def get_debate_result(session_id: str, wait_seconds: int = 5):
    """Get the final result of a debate with optional waiting."""
    import time
    start_time = time.time()
    
    while session_id not in session_results and (time.time() - start_time) < wait_seconds:
        time.sleep(1)
        
    if session_id not in session_results:
        # Last attempt to load from persistence if it's an old session
        pass # We'll implement results persistence if needed, but for now events are enough
        raise HTTPException(status_code=404, detail="Result not available yet")
    
    return session_results[session_id]

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
            models = data.get("data", [])
            model_names = [model.get("id", "") for model in models]
            print(f"Model names: {model_names}")
            return {"models": model_names}
        else:
            # Return default models if LM Studio is not responding
            return {
                "models": ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"],
                "warning": f"Could not connect to LM Studio (status {response.status_code}), using default models"
            }
    except Exception as e:
        # Return default models on error
        print(f"Error fetching models: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "models": ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"],
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