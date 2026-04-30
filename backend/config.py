from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class AgentConfig(BaseSettings):
    """Configuration for a single agent."""
    model: str = Field(default="liquid/lfm2.5-1.2b", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for sampling")
    system_prompt: str = Field(default="", description="System prompt for the agent")


class DebateConfig(BaseSettings):
    """Configuration for the debate system."""
    proposer: AgentConfig = Field(default_factory=AgentConfig)
    critic: AgentConfig = Field(default_factory=AgentConfig)
    judge: AgentConfig = Field(default_factory=AgentConfig)
    max_rounds: int = Field(default=3, ge=1, le=10, description="Maximum debate rounds")
    
    # Model connection settings
    model_provider: str = Field(default="openai", description="Model provider")
    base_url: str = Field(default="http://localhost:1234/v1", description="Base URL for model API")
    api_key: str = Field(default="lm-studio", description="API key")


class Settings(BaseSettings):
    """Application settings."""
    debate: DebateConfig = Field(default_factory=DebateConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Default system prompts
DEFAULT_PROPOSER_PROMPT = """You are a Proposer in a structured debate. Your role is to generate a well-reasoned legal argument on the given topic.

Your response must include:
1. A clear thesis statement
2. 3-5 supporting arguments with logical reasoning
3. Relevant legal principles or precedents if applicable

Format your response as a structured argument with clear sections."""

DEFAULT_CRITIC_PROMPT = """You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument.

Your response must include:
1. Exactly 3 logical fallacies or counter-points
2. Specific analysis of why each point is problematic
3. Alternative perspectives or evidence

Be critical but constructive in your analysis."""

DEFAULT_JUDGE_PROMPT = """You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict.

Your response must include:
1. A summary of the key arguments from both sides
2. A final Verdict (which side presented the stronger case)
3. A Consensus Score from 0-100 (where 100 indicates strong agreement/consensus)
4. Brief justification for your verdict

Be objective and evidence-based in your judgment."""
