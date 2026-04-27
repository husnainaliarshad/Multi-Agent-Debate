from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
try:
    model = init_chat_model(
        os.getenv("MODEL_NAME", "liquid/lfm2.5-1.2b"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        base_url=os.getenv("BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("API_KEY", "lm-studio"),
        temperature=float(os.getenv("TEMPERATURE", "0"))
    )
except Exception as e:
    print(f"Warning: Failed to initialize model: {e}")
    model = None

# Pydantic schemas
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message using the local LLM."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="LM Studio is not running or model initialization failed. Please start LM Studio and load the model."
        )
    
    try:
        response = model.invoke(request.message)
        return ChatResponse(response=str(response.content))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to generate response: {str(e)}. Ensure LM Studio is running on http://localhost:1234"
        )

@app.get("/")
async def root():
    return {"status": "running", "message": "Multi-Agent Debate Backend API"}