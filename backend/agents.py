from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal, Dict, Any
import time
import json
import uuid
from config import AgentConfig, DebateConfig, DEFAULT_PROPOSER_PROMPT, DEFAULT_CRITIC_PROMPT, DEFAULT_JUDGE_PROMPT


class DebateState(TypedDict):
    """State for the debate workflow."""
    topic: str
    proposer_response: str
    critic_response: str
    judge_response: str
    proposer_thought: str
    critic_thought: str
    judge_thought: str
    proposer_latency: float
    critic_latency: float
    judge_latency: float
    proposer_valid: bool
    critic_valid: bool
    judge_valid: bool
    consensus_score: int
    verdict: str
    round: int
    messages: Annotated[list, operator.add]


class DebateAgent:
    """Base class for debate agents."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, role: str, max_tokens: int = 500):
        self.config = config
        self.debate_config = debate_config
        self.role = role
        self.max_tokens = max_tokens
        self.model = init_chat_model(
            config.model,
            model_provider=debate_config.model_provider,
            base_url=debate_config.base_url,
            api_key=debate_config.api_key,
            temperature=config.temperature,
            model_kwargs={"max_tokens": max_tokens}
        )
    
    def invoke(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Invoke the model and return response with metrics."""
        start_time = time.time()
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.model.invoke(messages)
            latency = time.time() - start_time
            
            # Validate JSON format if response contains JSON
            syntactic_valid = True
            try:
                if "{" in str(response.content) and "}" in str(response.content):
                    json.loads(str(response.content))
            except:
                syntactic_valid = False
            
            return {
                "content": str(response.content),
                "latency": latency,
                "syntactic_valid": syntactic_valid
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "content": f"Error: {str(e)}",
                "latency": latency,
                "syntactic_valid": False
            }


class ProposerAgent(DebateAgent):
    """Proposer agent that generates the initial argument."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500):
        super().__init__(config, debate_config, "proposer", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_PROPOSER_PROMPT
    
    def generate_argument(self, topic: str) -> Dict[str, Any]:
        """Generate the initial argument on the topic."""
        prompt = f"Topic: {topic}\n\nGenerate your argument."
        return self.invoke(prompt, self.system_prompt)


class CriticAgent(DebateAgent):
    """Critic agent that identifies fallacies in the proposer's argument."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500):
        super().__init__(config, debate_config, "critic", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_CRITIC_PROMPT
    
    def critique(self, proposer_argument: str) -> Dict[str, Any]:
        """Critique the proposer's argument."""
        prompt = f"Proposer's Argument:\n{proposer_argument}\n\nProvide your critique."
        return self.invoke(prompt, self.system_prompt)


class JudgeAgent(DebateAgent):
    """Judge agent that synthesizes both sides and provides a verdict."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500):
        super().__init__(config, debate_config, "judge", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_JUDGE_PROMPT
    
    def judge(self, proposer_argument: str, critic_argument: str) -> Dict[str, Any]:
        """Judge the debate and provide a verdict."""
        prompt = f"""Proposer's Argument:\n{proposer_argument}\n\nCritic's Critique:\n{critic_argument}\n\nProvide your verdict and consensus score."""
        return self.invoke(prompt, self.system_prompt)


class DebateOrchestrator:
    """Orchestrates the multi-agent debate using LangGraph."""
    
    def __init__(self, config: DebateConfig, max_tokens: int = 500):
        self.config = config
        self.max_tokens = max_tokens
        self.proposer = ProposerAgent(config.proposer, config, max_tokens)
        self.critic = CriticAgent(config.critic, config, max_tokens)
        self.judge = JudgeAgent(config.judge, config, max_tokens)
        self.session_id = str(uuid.uuid4())
        self.events = []
    
    def run_debate(self, topic: str):
        """Run the complete debate workflow."""
        try:
            # Proposer generates argument
            print(f"[{self.session_id}] Starting debate on topic: {topic}")
            proposer_result = self.proposer.generate_argument(topic)
            self._emit_event("PROPOSER_START", {"topic": topic})
            self._emit_event("PROPOSER_THOUGHT", {"thought": "Analyzing topic and constructing argument..."})
            self._emit_event("PROPOSER_FINAL", {
                "response": proposer_result["content"],
                "latency": proposer_result["latency"],
                "syntactic_valid": proposer_result["syntactic_valid"]
            })
            print(f"[{self.session_id}] Proposer complete")
            
            # Critic critiques the argument
            critic_result = self.critic.critique(proposer_result["content"])
            self._emit_event("CRITIC_START", {})
            self._emit_event("CRITIC_THOUGHT", {"thought": "Analyzing proposer's argument for fallacies..."})
            self._emit_event("CRITIC_FINAL", {
                "response": critic_result["content"],
                "latency": critic_result["latency"],
                "syntactic_valid": critic_result["syntactic_valid"]
            })
            print(f"[{self.session_id}] Critic complete")
            
            # Judge provides verdict
            judge_result = self.judge.judge(proposer_result["content"], critic_result["content"])
            self._emit_event("JUDGE_START", {})
            self._emit_event("JUDGE_THOUGHT", {"thought": "Synthesizing both arguments and forming verdict..."})
            self._emit_event("JUDGE_FINAL", {
                "response": judge_result["content"],
                "latency": judge_result["latency"],
                "syntactic_valid": judge_result["syntactic_valid"]
            })
            print(f"[{self.session_id}] Judge complete")
            
            # Extract consensus score from judge response
            consensus_score = self._extract_consensus_score(judge_result["content"])
            verdict = self._extract_verdict(judge_result["content"])
            
            self._emit_event("DEBATE_COMPLETE", {
                "consensus_score": consensus_score,
                "verdict": verdict
            })
            print(f"[{self.session_id}] Debate complete")
            
            return {
                "session_id": self.session_id,
                "proposer_response": proposer_result["content"],
                "critic_response": critic_result["content"],
                "judge_response": judge_result["content"],
                "consensus_score": consensus_score,
                "verdict": verdict,
                "events": self.events
            }
        except Exception as e:
            print(f"[{self.session_id}] Error in run_debate: {str(e)}")
            import traceback
            traceback.print_exc()
            self._emit_event("ERROR", {"error": str(e)})
            raise
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event for streaming."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self.events.append(event)
    
    def _extract_consensus_score(self, judge_response: str) -> int:
        """Extract consensus score from judge response."""
        import re
        match = re.search(r'consensus.*?(\d+)', judge_response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 10)
        return 5  # Default middle score
    
    def _extract_verdict(self, judge_response: str) -> str:
        """Extract verdict from judge response."""
        if "proposer" in judge_response.lower():
            return "Proposer"
        elif "critic" in judge_response.lower():
            return "Critic"
        return "Inconclusive"
