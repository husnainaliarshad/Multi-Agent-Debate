from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal, Dict, Any, Optional
import time
import json
import uuid
import os
from utils.tools import search_tool
from core.config import AgentConfig, DebateConfig, DEFAULT_PROPOSER_PROMPT, DEFAULT_CRITIC_PROMPT, DEFAULT_JUDGE_PROMPT
from core.evaluation import DebateMetrics, calculate_turn_faithfulness
from services.rag_service import RAGService
from core.stability import StabilityMonitor
import random


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
            
        # Handle Groq provider specifically
        api_key = debate_config.api_key
        base_url = debate_config.base_url
        provider = debate_config.model_provider
        
        if provider == "groq":
            api_key = debate_config.groq_api_key
            # init_chat_model for groq might not need base_url unless it's a proxy
            base_url = None 
            
        self.model = init_chat_model(
            config.model,
            model_provider=provider,
            base_url=base_url,
            api_key=api_key,
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
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500, use_search: bool = False, rag_service: RAGService = None):
        super().__init__(config, debate_config, "proposer", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_PROPOSER_PROMPT
        self.use_search = use_search
        self.rag_service = rag_service
    
    def generate_argument(
        self,
        topic: str,
        round_num: int = 1,
        proposer_id: int = 1,
        include_search: bool = False,
        search_query: Optional[str] = None,
        include_rag: bool = False,
        rag_query: Optional[str] = None,
        extra_context: str = "",
    ) -> Dict[str, Any]:
        """Generate an argument on the topic with optional search and retrieval evidence."""
        search_results = ""
        if include_search and search_query:
            print(f"[{self.role}] Searching for evidence on: {search_query}")
            self._emit_event("SEARCH_START", {"proposer_id": proposer_id, "topic": topic, "query": search_query})
            
            search_results = search_tool.run(search_query)
            
            self._emit_event(
                "SEARCH_COMPLETE",
                {"proposer_id": proposer_id, "query": search_query, "results": search_results, "evidence_type": "web"},
            )
            print(f"[{self.role}] Search results obtained: {len(search_results)} chars")
        
        rag_results = ""
        if include_rag and self.rag_service and rag_query:
            print(f"[{self.role}] Querying LegalBench RAG for: {rag_query}")
            self._emit_event("RETRIEVAL_START", {"proposer_id": proposer_id, "query": rag_query})
            rag_results = self.rag_service.query(rag_query)
            if rag_results:
                self._emit_event(
                    "RETRIEVAL_COMPLETE",
                    {"proposer_id": proposer_id, "query": rag_query, "results": f"RAG Results:\n{rag_results}", "evidence_type": "rag"},
                )

        prompt = f"Topic: {topic}\n\n"
        if extra_context:
            prompt += f"{extra_context.strip()}\n\n"

        if search_results:
            prompt += f"Background Information/Search Results:\n{search_results}\n\n"
            prompt += "CRITICAL: You MUST incorporate the facts and evidence from the search results above into your argument. Cite specific details.\n\n"
        
        if rag_results:
            prompt += f"LegalBench Reference Information:\n{rag_results}\n\n"
            prompt += "CRITICAL: You MUST incorporate relevant legal principles or case details from the LegalBench references above.\n\n"
        
        prompt += "Generate your argument."
        result = self.invoke(prompt, self.system_prompt)
        result["search_results"] = search_results
        result["rag_results"] = rag_results
        return result


class CriticAgent(DebateAgent):
    """Critic agent that identifies fallacies in the proposer's argument."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500, use_search: bool = False):
        super().__init__(config, debate_config, "critic", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_CRITIC_PROMPT
        self.use_search = use_search
    
    def critique(
        self,
        proposer_argument: str,
        topic: str = "",
        round_num: int = 1,
        include_search: bool = False,
        search_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Critique the proposer's argument."""
        search_results = ""
        if include_search and search_query:
            self._emit_event("SEARCH_START", {"role": "critic", "topic": topic, "query": search_query})
            search_results = search_tool.run(search_query)
            self._emit_event("SEARCH_COMPLETE", {"role": "critic", "query": search_query, "results": search_results, "evidence_type": "web"})

        prompt = ""
        if search_results:
            prompt += f"Background Research on Counter-Arguments:\n{search_results}\n\n"
            prompt += "Use the research above to find specific weaknesses or overlooked facts.\n\n"
            
        prompt += f"Proposer's Argument:\n{proposer_argument}\n\nProvide your critique."
        result = self.invoke(prompt, self.system_prompt)
        result["search_results"] = search_results
        return result


class JudgeAgent(DebateAgent):
    """Judge agent that synthesizes both sides and provides a verdict."""
    
    def __init__(self, config: AgentConfig, debate_config: DebateConfig, max_tokens: int = 500):
        super().__init__(config, debate_config, "judge", max_tokens)
        self.system_prompt = config.system_prompt or DEFAULT_JUDGE_PROMPT
    
    def judge(self, proposer_argument: str, critic_argument: str, mode: str = "normal") -> Dict[str, Any]:
        """Judge the debate and provide a verdict."""
        if mode == "irac":
            prompt = proposer_argument # The summary is already formatted for IRAC
        else:
            prompt = f"""Proposer's Argument:\n{proposer_argument}\n\nCritic's Critique:\n{critic_argument}\n\nProvide your verdict and consensus score."""
        return self.invoke(prompt, self.system_prompt)


class DebateOrchestrator:
    """Orchestrates the multi-agent debate."""
    
    def __init__(self, config: DebateConfig, max_tokens: int = 500, proposer_configs: list = None, num_rounds: int = 1, use_search: bool = False, use_position_swap: bool = True, use_info_gain: bool = True, use_faithfulness: bool = True, use_summary_relay: bool = True, use_rag: bool = False, rag_service: RAGService = None):
        self.config = config
        self.max_tokens = max_tokens
        self.num_rounds = num_rounds
        self.use_search = use_search
        self.use_position_swap = use_position_swap
        self.use_info_gain = use_info_gain
        self.use_faithfulness = use_faithfulness
        self.use_summary_relay = use_summary_relay
        self.use_rag = use_rag
        self.mode = config.mode if hasattr(config, 'mode') else "hybrid"
        self.mode_capabilities = self._resolve_mode_capabilities()
        self.use_stability = True
        self.stability_monitor = StabilityMonitor()
        
        # Initialize retrieval whenever the selected mode requires LegalBench support.
        requires_rag = self.mode_capabilities["shared_initial_rag"] or self.mode_capabilities["iterative_rag"] or use_rag
        self.rag_service = rag_service if rag_service else (RAGService() if requires_rag else None)
        
        # Fresh model initialization for each agent to flush context
        if proposer_configs:
            self.proposers = [ProposerAgent(cfg, config, max_tokens, use_search=False, rag_service=self.rag_service) for cfg in proposer_configs]
        else:
            self.proposers = [ProposerAgent(config.proposer, config, max_tokens, use_search=False, rag_service=self.rag_service)]
        
        for p in self.proposers:
            p.set_event_callback(self._emit_event)
            
        self.critic = CriticAgent(config.critic, config, max_tokens, use_search=False)
        self.critic.set_event_callback(self._emit_event)
        
        self.judge = JudgeAgent(config.judge, config, max_tokens)
        self.judge.set_event_callback(self._emit_event)
        self.session_id = str(uuid.uuid4())
        self.events = []
        
        # Initialize evaluation metrics
        self.metrics = DebateMetrics()

    def _resolve_mode_capabilities(self) -> Dict[str, Any]:
        if self.mode == "baseline":
            return {
                "single_agent": True,
                "proposer_search": False,
                "critic_search": False,
                "shared_initial_rag": False,
                "iterative_rag": False,
            }
        if self.mode == "react_only":
            return {
                "single_agent": False,
                "proposer_search": True,
                "critic_search": True,
                "shared_initial_rag": False,
                "iterative_rag": False,
            }
        if self.mode == "naive_rag":
            return {
                "single_agent": False,
                "proposer_search": False,
                "critic_search": False,
                "shared_initial_rag": True,
                "iterative_rag": False,
            }
        if self.mode == "active_rag":
            return {
                "single_agent": False,
                "proposer_search": False,
                "critic_search": False,
                "shared_initial_rag": False,
                "iterative_rag": True,
            }

        return {
            "single_agent": False,
            "proposer_search": self.use_search,
            "critic_search": self.use_search,
            "shared_initial_rag": False,
            "iterative_rag": self.use_rag,
        }

    def _prepare_shared_rag_context(self, topic: str) -> str:
        if not self.mode_capabilities["shared_initial_rag"] or not self.rag_service:
            return ""

        self._emit_event("RETRIEVAL_START", {"role": "shared", "query": topic, "mode": self.mode})
        context = self.rag_service.query(topic)
        if context:
            self._emit_event(
                "RETRIEVAL_COMPLETE",
                {"role": "shared", "query": topic, "results": f"RAG Results:\n{context}", "mode": self.mode},
            )
            return f"Shared LegalBench Context:\n{context}"
        return ""

    def _build_proposer_search_query(self, topic: str, round_num: int, previous_critique: str = "") -> Optional[str]:
        if not self.mode_capabilities["proposer_search"]:
            return None
        if round_num == 1:
            return f"{topic} legal analysis facts evidence precedent"
        critique_snippet = previous_critique[:240]
        return f"{topic} address critique with legal evidence and counterpoints {critique_snippet}"

    def _build_proposer_rag_query(self, topic: str, round_num: int, previous_critique: str = "") -> Optional[str]:
        if not self.mode_capabilities["iterative_rag"] or not self.rag_service:
            return None
        if round_num == 1:
            return topic
        critique_snippet = previous_critique[:320]
        return f"{topic}\n\nNeed legal support to address this critique:\n{critique_snippet}"

    def _build_critic_search_query(self, topic: str, proposer_argument: str) -> Optional[str]:
        if not self.mode_capabilities["critic_search"]:
            return None
        argument_snippet = proposer_argument[:260]
        return f"counter arguments and critiques for: {topic}. Focus on weaknesses in: {argument_snippet}"



    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event for streaming."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self.events.append(event)

    def run_debate(self, topic: str):
        """Run the complete debate workflow with multiple proposers and rounds."""
        try:
            print(f"[{self.session_id}] Starting debate on topic: {topic}")
            # Emit first event with topic immediately for persistence visibility
            self._emit_event("DEBATE_START", {"topic": topic})
            self._emit_event("MODE_SELECTED", {"mode": self.mode, **self.mode_capabilities})
            
            print(f"[{self.session_id}] Number of proposers: {len(self.proposers)}, Rounds: {self.num_rounds}")
            base_topic = topic
            shared_rag_context = self._prepare_shared_rag_context(base_topic)
            
            # Store all proposer arguments, critic critiques, and search results across rounds
            all_proposer_arguments = []  # List of lists: [[round1_args], [round2_args], ...]
            all_critic_critiques = []     # List of lists: [[round1_critiques], [round2_critiques], ...]
            all_search_results = []       # List of lists: [[round1_search], [round2_search], ...]
            self.round_summaries = []     # List of dicts for summary relay
            baseline_judge_input = ""
            
            for round_num in range(1, self.num_rounds + 1):
                print(f"[{self.session_id}] Round {round_num}/{self.num_rounds}")
                self._emit_event("ROUND_START", {"round": round_num, "total_rounds": self.num_rounds})
                
                # RESEARCH MODE: Baseline (Single Agent, No Debate)
                if self.mode == "baseline":
                    print(f"[{self.session_id}] Mode: Baseline (Single Agent)")
                    res = self.proposers[0].generate_argument(
                        base_topic,
                        round_num=1,
                        proposer_id=1,
                        include_search=False,
                        include_rag=False,
                    )
                    all_proposer_arguments.append([res["content"]])
                    if self.use_info_gain:
                        self.metrics.add_proposer_response(res["content"])
                    self.metrics.format_adherence["total"] += 1
                    if res.get("syntactic_valid", False):
                        self.metrics.format_adherence["valid"] += 1
                    self._emit_event("PROPOSER_FINAL", {
                        "proposer_id": 1,
                        "round": 1,
                        "response": res["content"],
                        "latency": res["latency"],
                        "syntactic_valid": res["syntactic_valid"]
                    })
                    baseline_judge_input = f"Topic: {base_topic}\n\nModel Response:\n{res['content']}"
                    break
                
                # All proposers generate arguments in parallel
                round_proposer_results = []
                round_search_results = []
                for i, proposer in enumerate(self.proposers):
                    print(f"[{self.session_id}] Proposer {i+1} generating argument...")
                    self._emit_event("PROPOSER_START", {"proposer_id": i+1, "round": round_num, "topic": topic})
                    self._emit_event("PROPOSER_THOUGHT", {"proposer_id": i+1, "round": round_num, "thought": "Analyzing topic and constructing argument..."})
                    
                    if round_num == 1:
                        round_topic = base_topic
                        previous_critique = ""
                    else:
                        # In later rounds, respond to previous critique and don't repeat yourself
                        if self.use_summary_relay and round_num > 1 and len(self.round_summaries) >= round_num - 1:
                            previous_critique = self.round_summaries[round_num-2]['critic']
                            previous_argument = self.round_summaries[round_num-2]['proposer'][i]
                        else:
                            previous_critique = "\n\n".join(all_critic_critiques[round_num-2])
                            previous_argument = all_proposer_arguments[round_num-2][i]
                            
                        round_topic = (
                            f"Topic: {base_topic}\n\n"
                            f"Your Previous Argument:\n{previous_argument}\n\n"
                            f"Critic's Critique:\n{previous_critique}\n\n"
                            f"IMPORTANT: Do not repeat your previous points. Respond to the critique, "
                            f"address the weaknesses identified, and provide new supporting evidence or "
                            f"refined reasoning. Build upon your previous argument rather than restating it."
                        )

                    result = proposer.generate_argument(
                        round_topic,
                        round_num=round_num,
                        proposer_id=i+1,
                        include_search=bool(self._build_proposer_search_query(base_topic, round_num, previous_critique)),
                        search_query=self._build_proposer_search_query(base_topic, round_num, previous_critique),
                        include_rag=bool(self._build_proposer_rag_query(base_topic, round_num, previous_critique)),
                        rag_query=self._build_proposer_rag_query(base_topic, round_num, previous_critique),
                        extra_context=shared_rag_context,
                    )
                    
                    # Authorship Obfuscation (Simple perturbation for SPB mitigation)
                    result["content"] = self._obfuscate_authorship(result["content"])
                    
                    round_proposer_results.append(result)
                    
                    search_content = result.get("search_results", "")
                    rag_content = result.get("rag_results", "")
                    combined_evidence = "\n\n".join([content for content in [search_content, rag_content] if content])
                    if search_content:
                        self.metrics.search_efficiency["total_searches"] += 1
                        if len(search_content.strip()) < 20:
                            self.metrics.search_efficiency["empty_searches"] += 1
                    round_search_results.append(combined_evidence)
                    
                    # Track Faithfulness
                    if self.use_faithfulness and combined_evidence:
                        faithfulness_score = calculate_turn_faithfulness(result["content"], combined_evidence)
                        self.metrics.turn_faithfulness.append(faithfulness_score)

                    # Track Format Adherence
                    self.metrics.format_adherence["total"] += 1
                    if result.get("syntactic_valid", False):
                        self.metrics.format_adherence["valid"] += 1

                    self._emit_event("PROPOSER_FINAL", {
                        "proposer_id": i+1,
                        "round": round_num,
                        "response": result["content"],
                        "latency": result["latency"],
                        "syntactic_valid": result["syntactic_valid"]
                    })
                    # Track metrics if enabled
                    if self.use_info_gain:
                        self.metrics.add_proposer_response(result["content"])
                    print(f"[{self.session_id}] Proposer {i+1} complete")
                
                all_proposer_arguments.append([r["content"] for r in round_proposer_results])
                
                # ADAPTIVE STOPPING: Check stability after each round (except last)
                if self.num_rounds > 1 and round_num < self.num_rounds and self.mode != "baseline":
                    current_stances = [self._extract_consensus_score(r["content"]) for r in round_proposer_results]
                    if self.stability_monitor.check_stability(current_stances):
                        print(f"[{self.session_id}] Adaptive Stopping triggered: Opinion stability reached at round {round_num}")
                        self._emit_event("ADAPTIVE_STOPPING", {"round": round_num})
                        break
                all_search_results.append(round_search_results)
                
                # Critic critiques all proposer arguments
                print(f"[{self.session_id}] Critic analyzing...")
                self._emit_event("CRITIC_START", {"round": round_num})
                self._emit_event("CRITIC_THOUGHT", {"round": round_num, "thought": "Identifying weaknesses and fallacies..."})
                
                if self.use_summary_relay and round_num > 1 and len(self.round_summaries) >= round_num - 1:
                    combined_args = "\n\n".join([f"Proposer {idx+1} (Summary): {arg}" for idx, arg in enumerate(self.round_summaries[round_num-2]['proposer'])])
                else:
                    combined_args = "\n\n".join([f"Proposer {idx+1}: {arg['content']}" for idx, arg in enumerate(round_proposer_results)])
                    
                critic_search_query = self._build_critic_search_query(base_topic, combined_args)
                critic_result = self.critic.critique(
                    combined_args,
                    topic=base_topic,
                    round_num=round_num,
                    include_search=bool(critic_search_query),
                    search_query=critic_search_query,
                )
                
                # Format adherence tracking
                self.metrics.format_adherence["total"] += 1
                if critic_result.get("syntactic_valid", False):
                    self.metrics.format_adherence["valid"] += 1
                    
                critic_search = critic_result.get("search_results", "")
                if critic_search:
                    self.metrics.search_efficiency["total_searches"] += 1
                    if len(critic_search.strip()) < 20:
                        self.metrics.search_efficiency["empty_searches"] += 1
                
                all_critic_critiques.append([critic_result["content"]])
                self._emit_event("CRITIC_FINAL", {
                    "round": round_num,
                    "response": critic_result["content"],
                    "latency": critic_result["latency"],
                    "syntactic_valid": critic_result["syntactic_valid"]
                })
                # Track metrics if enabled
                if self.use_info_gain:
                    self.metrics.add_critic_response(critic_result["content"])
                print(f"[{self.session_id}] Critic complete")

                # Summary-Based Relay
                if self.use_summary_relay and round_num < self.num_rounds:
                    print(f"[{self.session_id}] Generating round summaries for relay...")
                    self._emit_event("JUDGE_THOUGHT", {"thought": f"Summarizing Round {round_num} for relay to next round..."})
                    
                    round_summary = {'proposer': [], 'critic': ""}
                    for idx, arg in enumerate(round_proposer_results):
                        summ_res = self.judge.invoke(f"Summarize this argument concisely while preserving key claims and evidence:\n\n{arg['content']}", "You are a Summarizer. Output only the condensed summary.")
                        round_summary['proposer'].append(summ_res['content'])
                    
                    crit_summ_res = self.judge.invoke(f"Summarize this critique concisely:\n\n{critic_result['content']}", "You are a Summarizer. Output only the condensed summary.")
                    round_summary['critic'] = crit_summ_res['content']
                    
                    self.round_summaries.append(round_summary)
            
            # Judge synthesizes all arguments and critiques
            print(f"[{self.session_id}] Judge synthesizing debate...")
            self._emit_event("JUDGE_START", {})
            
            if self.mode == "baseline":
                debate_summary = baseline_judge_input
            else:
                debate_summary = ""
                for round_num, (args, critique) in enumerate(zip(all_proposer_arguments, all_critic_critiques), 1):
                    debate_summary += f"\n=== Round {round_num} ===\n"
                    for i, arg in enumerate(args):
                        debate_summary += f"\nProposer {i+1}:\n{arg}\n"
                    debate_summary += f"\nCritic:\n{critique}\n"
            
            # Position Swapping: Run judge twice with swapped argument order if enabled
            if self.use_position_swap:
                self._emit_event("JUDGE_THOUGHT", {"thought": "Running position-swapped evaluation to reduce bias..."})
                
                # Cognitive Load Decomposition: IRAC Judge Prompting
                irac_summary = self._format_irac_summary(debate_summary)
                
                # Normal run
                judge_result_normal = self.judge.judge(irac_summary, "", mode="irac")
                consensus_normal = self._extract_consensus_score(judge_result_normal["content"])
                verdict_normal = self._extract_verdict(judge_result_normal["content"])
                
                # Second run: swapped order (Critic then Proposer)
                if self.mode == "baseline":
                    debate_summary_swapped = debate_summary
                else:
                    debate_summary_swapped = ""
                    for round_num, (args, critique) in enumerate(zip(all_proposer_arguments, all_critic_critiques), 1):
                        debate_summary_swapped += f"\n=== Round {round_num} ===\n"
                        debate_summary_swapped += f"\nCritic:\n{critique}\n"
                        for i, arg in enumerate(args):
                            debate_summary_swapped += f"\nProposer {i+1}:\n{arg}\n"
                
                judge_result_swapped = self.judge.judge(debate_summary_swapped, "")
                consensus_swapped = self._extract_consensus_score(judge_result_swapped["content"])
                verdict_swapped = self._extract_verdict(judge_result_swapped["content"])
                
                # Average the scores
                consensus_score = int((consensus_normal + consensus_swapped) / 2)
                
                # Use the verdict from the normal run (or could implement voting logic)
                verdict = verdict_normal
                
                # Store position swap scores in metrics
                if self.use_info_gain:
                    self.metrics.position_swap_scores.append({
                        "normal": {"consensus": consensus_normal, "verdict": verdict_normal},
                        "swapped": {"consensus": consensus_swapped, "verdict": verdict_swapped},
                        "averaged": {"consensus": consensus_score, "verdict": verdict}
                    })
                
                self._emit_event("JUDGE_FINAL", {
                    "response": judge_result_normal["content"],
                    "latency": judge_result_normal["latency"],
                    "syntactic_valid": judge_result_normal["syntactic_valid"],
                    "position_swap": {
                        "normal_consensus": consensus_normal,
                        "swapped_consensus": consensus_swapped,
                        "averaged_consensus": consensus_score
                    }
                })
                print(f"[{self.session_id}] Judge complete (Position Swap: {consensus_normal} -> {consensus_swapped} -> {consensus_score})")
            else:
                # Normal single-run evaluation
                self._emit_event("JUDGE_THOUGHT", {"thought": "Synthesizing all arguments and critiques from all rounds..."})
                judge_result_normal = self.judge.judge(debate_summary, "")
                consensus_normal = self._extract_consensus_score(judge_result_normal["content"])
                verdict_normal = self._extract_verdict(judge_result_normal["content"])
                consensus_score = consensus_normal
                verdict = verdict_normal
                
                self._emit_event("JUDGE_FINAL", {
                    "response": judge_result_normal["content"],
                    "latency": judge_result_normal["latency"],
                    "syntactic_valid": judge_result_normal["syntactic_valid"]
                })
                print(f"[{self.session_id}] Judge complete (Single run: {consensus_score})")
            
            self._emit_event("DEBATE_COMPLETE", {
                "consensus_score": consensus_score,
                "verdict": verdict,
                "num_proposers": len(self.proposers),
                "num_rounds": self.num_rounds
            })
            print(f"[{self.session_id}] Debate complete")
            
            result = {
                "session_id": self.session_id,
                "topic": base_topic,
                "mode": self.mode,
                "provider": self.config.model_provider,
                "use_search": self.mode_capabilities["proposer_search"] or self.mode_capabilities["critic_search"],
                "use_rag": self.mode_capabilities["shared_initial_rag"] or self.mode_capabilities["iterative_rag"],
                "proposer_responses": all_proposer_arguments,
                "critic_responses": all_critic_critiques,
                "search_results": all_search_results,
                "judge_response": judge_result_normal["content"],
                "consensus_score": consensus_score,
                "verdict": verdict,
                "num_proposers": len(self.proposers),
                "num_rounds": self.num_rounds,
                "events": self.events
            }
            
            # Always include metrics since we now track many aspects
            result["metrics"] = self.metrics.to_dict()
            
            return result
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

    def _obfuscate_authorship(self, text: str) -> str:
        """Apply simple perturbations to obfuscate authorship."""
        replacements = {
            "I believe": "It is argued that",
            "In my opinion": "Analysis suggests",
            "I strongly agree": "There is strong support for",
            "Furthermore": "In addition",
            "However": "Conversely"
        }
        obfuscated = text
        for old, new in replacements.items():
            obfuscated = obfuscated.replace(old, new)
        return obfuscated

    def _format_irac_summary(self, summary: str) -> str:
        """Helper to structure the debate summary for IRAC evaluation."""
        return f"""Please evaluate the following legal debate using the IRAC framework:
        
{summary}

Evaluate each of the following components separately:
1. ISSUE: Did the proposer correctly identify the legal issue?
2. RULE: Was the cited legal rule/statute accurate and relevant?
3. APPLICATION: Was the rule logically applied to the facts?
4. CONCLUSION: Is the final conclusion legally sound?
"""
