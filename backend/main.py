from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import requests
from config import DebateConfig, AgentConfig, Settings
from agents import DebateOrchestrator
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
    max_rounds: Optional[int] = 1
    max_tokens: Optional[int] = 500

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
                system_prompt=request.judge_prompt or "You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict."
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
            num_rounds=request.max_rounds or 1
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
    except Exception as e:
        session_results[orchestrator.session_id] = {
            "error": str(e),
            "session_id": orchestrator.session_id
        }

@app.get("/debate/events/{session_id}")
def get_debate_events(session_id: str):
    """Get all events for a debate session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = sessions[session_id]
    return {
        "session_id": session_id,
        "events": orchestrator.events,
        "complete": session_id in session_results
    }

@app.get("/debate/result/{session_id}")
def get_debate_result(session_id: str):
    """Get the final result of a debate."""
    if session_id not in session_results:
        raise HTTPException(status_code=404, detail="Result not available yet")
    
    return session_results[session_id]

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

@app.get("/")
def root():
    return {"status": "running", "message": "Multi-Agent Debate Backend API"}