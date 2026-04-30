from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal, Dict, Any
import time
import json
import uuid
import os
from database import save_debate_session
from tools import search_tool
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
        self.event_callback = None
        
        # Merge max_tokens with any other model_kwargs from environment or config
        model_kwargs = {"max_tokens": max_tokens}
            
        self.model = init_chat_model(
            config.model,
            model_provider=debate_config.model_provider,
            base_url=debate_config.base_url,
            api_key=debate_config.api_key,
            temperature=config.temperature,
            model_kwargs=model_kwargs
        )
    
    def set_event_callback(self, callback):
        """Set a callback for emitting events."""
        self.event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event via the callback."""
        if self.event_callback:
            self.event_callback(event_type, data)
    
    def _get_system_prompt(self, base_prompt: str) -> str:
        """Add token limit instructions to system prompt."""
        if self.max_tokens <= 200:
            return f"{base_prompt}\n\nIMPORTANT: Keep your response very concise (under {self.max_tokens} tokens). Complete your thought fully within this limit."
        elif self.max_tokens <= 400:
            return f"{base_prompt}\n\nKeep your response concise (under {self.max_tokens} tokens). Complete your thought fully within this limit."
        else:
            return base_prompt
    
    def invoke(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Invoke the model and return response with metrics."""
        start_time = time.time()
        
        try:
            # Add token limit instructions to system prompt
            enhanced_prompt = self._get_system_prompt(system_prompt)
            
            messages = [
                SystemMessage(content=enhanced_prompt),
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
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500, use_search: bool = False):
        super().__init__(config, debate_config, "proposer", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_PROPOSER_PROMPT
        self.use_search = use_search
    
    def generate_argument(self, topic: str, round_num: int = 1, proposer_id: int = 1) -> Dict[str, Any]:
        """Generate the initial argument on the topic."""
        search_results = ""
        if self.use_search and round_num == 1:
            print(f"[{self.role}] Searching for evidence on: {topic}")
            # Emit event for tool use visibility
            self._emit_event("SEARCH_START", {"proposer_id": proposer_id, "topic": topic})
            
            # Improved query for better results
            search_query = f"{topic} facts evidence scientific consensus"
            search_results = search_tool.run(search_query)
            
            self._emit_event("SEARCH_COMPLETE", {"proposer_id": proposer_id, "results": search_results})
            print(f"[{self.role}] Search results obtained: {len(search_results)} chars")
        
        prompt = f"Topic: {topic}\n\n"
        if search_results:
            prompt += f"Background Information/Search Results:\n{search_results}\n\n"
            prompt += "CRITICAL: You MUST incorporate the facts and evidence from the search results above into your argument. Cite specific details.\n\n"
        
        prompt += "Generate your argument."
        return self.invoke(prompt, self.system_prompt)


class CriticAgent(DebateAgent):
    """Critic agent that identifies fallacies in the proposer's argument."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500, use_search: bool = False):
        super().__init__(config, debate_config, "critic", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_CRITIC_PROMPT
        self.use_search = use_search
    
    def critique(self, proposer_argument: str, topic: str = "", round_num: int = 1) -> Dict[str, Any]:
        """Critique the proposer's argument."""
        search_results = ""
        if self.use_search:
            search_query = f"counter arguments and critiques for: {topic}"
            self._emit_event("SEARCH_START", {"role": "critic", "topic": topic, "query": search_query})
            search_results = search_tool.run(search_query)
            self._emit_event("SEARCH_COMPLETE", {"role": "critic", "results": search_results})

        prompt = ""
        if search_results:
            prompt += f"Background Research on Counter-Arguments:\n{search_results}\n\n"
            prompt += "Use the research above to find specific weaknesses or overlooked facts.\n\n"
            
        prompt += f"Proposer's Argument:\n{proposer_argument}\n\nProvide your critique."
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
    """Orchestrates the multi-agent debate."""
    
    def __init__(self, config: DebateConfig, max_tokens: int = 500, proposer_configs: list = None, num_rounds: int = 1, use_search: bool = False):
        self.config = config
        self.max_tokens = max_tokens
        self.num_rounds = num_rounds
        self.use_search = use_search
        
        # Fresh model initialization for each agent to flush context
        if proposer_configs:
            self.proposers = [ProposerAgent(cfg, config, max_tokens, use_search=use_search) for cfg in proposer_configs]
        else:
            self.proposers = [ProposerAgent(config.proposer, config, max_tokens, use_search=use_search)]
        
        for p in self.proposers:
            p.set_event_callback(self._emit_event)
            
        self.critic = CriticAgent(config.critic, config, max_tokens, use_search=use_search)
        self.critic.set_event_callback(self._emit_event)
        
        self.judge = JudgeAgent(config.judge, config, max_tokens)
        self.judge.set_event_callback(self._emit_event)
        self.session_id = str(uuid.uuid4())
        self.events = []


    def _save_session(self, topic: str):
        """Persist events to database."""
        try:
            # We also include results if the debate is complete
            save_debate_session(self.session_id, topic, self.events)
        except Exception as e:
            print(f"Error saving session to DB: {e}")

    def _emit_event(self, event_type: str, data: Dict[str, Any], topic: str = None):
        """Emit an event for streaming."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self.events.append(event)
        
        # Use provided topic or find it from state
        if not topic:
            if event_type == "DEBATE_START":
                topic = data.get("topic", "Unknown Topic")
            elif self.events:
                # Try to find topic from the first event
                first_event = self.events[0]
                if first_event["event_type"] == "DEBATE_START":
                    topic = first_event["data"].get("topic", "Unknown Topic")
        
        # Ensure we have a topic before saving
        current_topic = topic or "Unknown Topic"
        self._save_session(current_topic)

    def run_debate(self, topic: str):
        """Run the complete debate workflow with multiple proposers and rounds."""
        try:
            print(f"[{self.session_id}] Starting debate on topic: {topic}")
            # Emit first event with topic immediately for persistence visibility
            self._emit_event("DEBATE_START", {"topic": topic})
            
            print(f"[{self.session_id}] Number of proposers: {len(self.proposers)}, Rounds: {self.num_rounds}")
            
            # Store all proposer arguments and critic critiques across rounds
            all_proposer_arguments = []  # List of lists: [[round1_args], [round2_args], ...]
            all_critic_critiques = []     # List of lists: [[round1_critiques], [round2_critiques], ...]
            
            for round_num in range(1, self.num_rounds + 1):
                print(f"[{self.session_id}] Round {round_num}/{self.num_rounds}")
                self._emit_event("ROUND_START", {"round": round_num, "total_rounds": self.num_rounds})
                
                # All proposers generate arguments in parallel
                round_proposer_results = []
                for i, proposer in enumerate(self.proposers):
                    print(f"[{self.session_id}] Proposer {i+1} generating argument...")
                    self._emit_event("PROPOSER_START", {"proposer_id": i+1, "round": round_num, "topic": topic})
                    self._emit_event("PROPOSER_THOUGHT", {"proposer_id": i+1, "round": round_num, "thought": "Analyzing topic and constructing argument..."})
                    
                    if round_num == 1:
                        result = proposer.generate_argument(topic, round_num=round_num, proposer_id=i+1)
                    else:
                        # In later rounds, respond to previous critique and don't repeat yourself
                        previous_critique = "\n\n".join(all_critic_critiques[round_num-2])
                        previous_argument = all_proposer_arguments[round_num-2][i]
                        result = proposer.generate_argument(
                            f"Topic: {topic}\n\n"
                            f"Your Previous Argument:\n{previous_argument}\n\n"
                            f"Critic's Critique:\n{previous_critique}\n\n"
                            f"IMPORTANT: Do not repeat your previous points. Respond to the critique, "
                            f"address the weaknesses identified, and provide new supporting evidence or "
                            f"refined reasoning. Build upon your previous argument rather than restating it.",
                            round_num=round_num,
                            proposer_id=i+1
                        )
                    
                    round_proposer_results.append(result)
                    self._emit_event("PROPOSER_FINAL", {
                        "proposer_id": i+1,
                        "round": round_num,
                        "response": result["content"],
                        "latency": result["latency"],
                        "syntactic_valid": result["syntactic_valid"]
                    })
                    print(f"[{self.session_id}] Proposer {i+1} complete")
                
                all_proposer_arguments.append([r["content"] for r in round_proposer_results])
                
                # Critic critiques all proposer arguments
                print(f"[{self.session_id}] Critic analyzing...")
                self._emit_event("CRITIC_START", {"round": round_num})
                self._emit_event("CRITIC_THOUGHT", {"round": round_num, "thought": "Identifying weaknesses and fallacies..."})
                
                combined_args = "\n\n".join([f"Proposer {idx+1}: {arg}" for idx, arg in enumerate(round_proposer_results)])
                critic_result = self.critic.critique(combined_args, topic=topic, round_num=round_num)
                
                all_critic_critiques.append([critic_result["content"]])
                self._emit_event("CRITIC_FINAL", {
                    "round": round_num,
                    "response": critic_result["content"],
                    "latency": critic_result["latency"],
                    "syntactic_valid": critic_result["syntactic_valid"]
                })
                print(f"[{self.session_id}] Critic complete")
            
            # Judge synthesizes all arguments and critiques
            print(f"[{self.session_id}] Judge synthesizing debate...")
            self._emit_event("JUDGE_START", {})
            
            debate_summary = ""
            for round_num, (args, critique) in enumerate(zip(all_proposer_arguments, all_critic_critiques), 1):
                debate_summary += f"\n=== Round {round_num} ===\n"
                for i, arg in enumerate(args):
                    debate_summary += f"\nProposer {i+1}:\n{arg}\n"
                debate_summary += f"\nCritic:\n{critique}\n"
            
            self._emit_event("JUDGE_THOUGHT", {"thought": "Synthesizing all arguments and critiques from all rounds..."})
            judge_result = self.judge.judge(debate_summary, "")
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
                "verdict": verdict,
                "num_proposers": len(self.proposers),
                "num_rounds": self.num_rounds
            })
            print(f"[{self.session_id}] Debate complete")
            
            return {
                "session_id": self.session_id,
                "proposer_responses": all_proposer_arguments,
                "critic_responses": all_critic_critiques,
                "judge_response": judge_result["content"],
                "consensus_score": consensus_score,
                "verdict": verdict,
                "num_proposers": len(self.proposers),
                "num_rounds": self.num_rounds,
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
            return min(max(score, 0), 100)
        return 50  # Default middle score
    
    def _extract_verdict(self, judge_response: str) -> str:
        """Extract verdict from judge response."""
        if "proposer" in judge_response.lower():
            return "Proposer"
        elif "critic" in judge_response.lower():
            return "Critic"
        return "Inconclusive"
