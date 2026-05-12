import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import pandas as pd
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Debate Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# API base URL. In Docker, this is injected as http://backend:8001.
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8001")


@st.cache_data(ttl=300)
def fetch_models_for_dashboard() -> Dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"models": ["liquid/lfm2.5-1.2b"], "groq_models": []}


def fetch_experiment_results_dataframe(experiment_id: str):
    """Load experiment results from the API into a DataFrame."""
    try:
        r = requests.get(f"{API_BASE}/experiments/{experiment_id}/results", timeout=120)
        if r.status_code != 200:
            return None, r.text or str(r.status_code)
        payload = r.json()
        rows = payload.get("rows") or []
        cols = payload.get("columns") or []
        if cols:
            df = pd.DataFrame(rows, columns=cols)
        else:
            df = pd.DataFrame(rows)
        return df, None
    except Exception as exc:
        return None, str(exc)


def experiment_row_label(ex: Dict[str, Any]) -> str:
    """Human-friendly label: name, short id, status."""
    name = (ex.get("name") or "Unnamed").strip() or "Unnamed"
    eid = ex.get("id") or ""
    short = (eid[:8] + "…") if len(eid) > 8 else eid
    st = (ex.get("status") or "?").upper()
    return f"{name}  ({short})  ·  {st}"


def build_experiment_model_configs(
    selected_models: list,
    research_profiles: list,
    baseline_model: str,
    baseline_provider: str,
) -> list:
    """Build model_configs for /experiments/run (shared by Experiments and KnK tabs)."""
    model_configs = []
    for profile in research_profiles:
        if "Baseline" in profile:
            model_configs.append(
                {
                    "proposer_model": baseline_model,
                    "critic_model": baseline_model,
                    "judge_model": baseline_model,
                    "provider": baseline_provider,
                    "mode": "baseline",
                }
            )
        elif "ReAct Only" in profile:
            for model_name in selected_models:
                model_configs.append(
                    {
                        "mode": "react_only",
                        "provider": "openai",
                        "proposer_model": model_name,
                        "critic_model": model_name,
                        "judge_model": model_name,
                    }
                )
        elif "Naive RAG" in profile:
            for model_name in selected_models:
                model_configs.append(
                    {
                        "mode": "naive_rag",
                        "provider": "openai",
                        "proposer_model": model_name,
                        "critic_model": model_name,
                        "judge_model": model_name,
                    }
                )
        elif "Active RAG" in profile:
            for model_name in selected_models:
                model_configs.append(
                    {
                        "mode": "active_rag",
                        "provider": "openai",
                        "proposer_model": model_name,
                        "critic_model": model_name,
                        "judge_model": model_name,
                    }
                )
        elif "Hybrid" in profile:
            for model_name in selected_models:
                model_configs.append(
                    {
                        "mode": "hybrid",
                        "provider": "openai",
                        "proposer_model": model_name,
                        "critic_model": model_name,
                        "judge_model": model_name,
                    }
                )
    return model_configs


def display_event(event: Dict[str, Any], event_index: int = 0):
    """Display a single event with appropriate styling."""
    event_type = event["event_type"]
    data = event["data"]
    
    if event_type == "DEBATE_START":
        st.info(f"🚀 **Debate Started:** {data.get('topic', '')}")
    elif event_type == "ROUND_START":
        st.markdown("---")
        st.info(f"🔄 **Round {data.get('round', 1)}/{data.get('total_rounds', 1)}**")
    elif event_type == "PROPOSER_START":
        proposer_id = data.get("proposer_id", 1)
        round_num = data.get("round", 1)
        st.info(f"🗣️ Proposer {proposer_id} (Round {round_num}) is analyzing the topic...")
    elif event_type == "PROPOSER_THOUGHT":
        proposer_id = data.get("proposer_id", 1)
        with st.expander(f"💭 Proposer {proposer_id}'s Thought Process"):
            st.text(data.get("thought", ""))
    elif event_type == "PROPOSER_FINAL":
        proposer_id = data.get("proposer_id", 1)
        round_num = data.get("round", 1)
        st.success(f"✅ Proposer {proposer_id}'s argument complete (Round {round_num})")
        with st.expander(f"View Proposer {proposer_id}'s Response"):
            st.markdown(data.get("response", ""))
    elif event_type == "SEARCH_START":
        proposer_id = data.get("proposer_id", 1)
        st.warning(f"🔍 Proposer {proposer_id} is searching the web for evidence on: {data.get('topic', '')}...")
    elif event_type == "SEARCH_COMPLETE":
        proposer_id = data.get("proposer_id", 1)
        st.info(f"✅ Proposer {proposer_id} found relevant evidence.")
        with st.expander(f"View Search Results (Proposer {proposer_id})"):
            st.text(data.get("results", ""))
    elif event_type == "RETRIEVAL_START":
        proposer_id = data.get("proposer_id")
        if proposer_id:
            st.warning(f"Knowledge-base retrieval started for proposer {proposer_id}.")
        else:
            st.warning("Knowledge-base retrieval started.")
    elif event_type == "RETRIEVAL_COMPLETE":
        proposer_id = data.get("proposer_id")
        label = f"Proposer {proposer_id}" if proposer_id else "Shared Retrieval"
        st.info(f"{label} received retrieved evidence.")
        with st.expander(f"View retrieved evidence ({label})"):
            st.text(data.get("results", ""))
    elif event_type == "CRITIC_START":
        round_num = data.get("round", 1)
        st.info(f"🔍 Critic is analyzing arguments (Round {round_num})...")
    elif event_type == "CRITIC_THOUGHT":
        with st.expander("💭 Critic's Thought Process"):
            st.text(data.get("thought", ""))
    elif event_type == "CRITIC_FINAL":
        round_num = data.get("round", 1)
        st.success(f"✅ Critic's analysis complete (Round {round_num})")
        with st.expander("View Critic's Response"):
            st.markdown(data.get("response", ""))
    elif event_type == "JUDGE_START":
        st.info("⚖️ Judge is synthesizing the debate...")
    elif event_type == "JUDGE_THOUGHT":
        with st.expander("💭 Judge's Thought Process"):
            st.text(data.get("thought", ""))
    elif event_type == "JUDGE_FINAL":
        st.success("✅ Judge's verdict complete")
        with st.expander("View Judge's Response"):
            st.markdown(data.get("response", ""))
    elif event_type == "ADAPTIVE_STOPPING":
        round_num = data.get("round", 1)
        st.success(f"🛑 Adaptive Stopping Triggered: Consensus reached at Round {round_num}.")
    elif event_type == "DEBATE_COMPLETE":
        st.balloons()
        st.info(f"🎉 Debate complete with {data.get('num_proposers', 1)} proposer(s) and {data.get('num_rounds', 1)} round(s)")
    elif event_type == "ERROR":
        st.error(f"❌ Error: {data.get('error', 'Unknown error')}")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debate_events" not in st.session_state:
    st.session_state.debate_events = []
if "debate_result" not in st.session_state:
    st.session_state.debate_result = None
if "debate_complete" not in st.session_state:
    st.session_state.debate_complete = False

# Sidebar - Recent Debates
with st.sidebar:
    st.markdown("### 📜 Recent Debates")
    if st.button("🔄 Refresh History"):
        st.rerun()
        
    try:
        recent_resp = requests.get(f"{API_BASE}/debates/recent", timeout=2)
        if recent_resp.status_code == 200:
            recent_debates = recent_resp.json().get("sessions", [])
            if not recent_debates:
                st.write("No recent debates found.")
            for rd in recent_debates:
                topic_display = f"{rd['topic'][:50]}..." if len(rd['topic']) > 50 else rd['topic']
                btn_label = f"💾 {topic_display}"
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(btn_label, key=f"hist_btn_{rd['session_id']}", help=rd['topic']):
                        st.session_state.session_id = rd['session_id']
                        st.session_state.debate_events = []
                        st.session_state.debate_result = None
                        st.session_state.debate_complete = False
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{rd['session_id']}", help="Delete this debate"):
                        delete_resp = requests.delete(f"{API_BASE}/debate/{rd['session_id']}")
                        if delete_resp.status_code == 200:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete")
    except Exception as e:
        st.write("Could not load recent debates.")

# Main title
st.title("⚖️ Multi-Agent Debate Research Platform")

# Two-column layout
col_config, col_debate = st.columns([1, 1])

# LEFT COLUMN: Configuration
with col_config:
    st.markdown("## ⚙️ Debate Configuration")
    
    # Topic input at the very top
    st.markdown("### 💬 Debate Topic")
    topic = st.text_input(
        "Enter a topic for the debate",
        placeholder="e.g., 'Should AI be granted legal personhood?'",
        value="",
        label_visibility="collapsed",
        key="debate_topic"
    )
    
    st.markdown("---")
    
    # Organize parameters into Tabs
    tab_struct, tab_agents, tab_eval, tab_bench, tab_adv, tab_exp = st.tabs([
        "🏗️ Structure",
        "🤖 Agents",
        "📊 Evaluation",
        "🧩 KnK benchmark",
        "⚙️ Advanced",
        "🔬 Experiments",
    ])
    
    with tab_struct:
        st.markdown("### 🔄 Debate Structure")
        num_proposers = st.slider("Number of Proposers", 1, 5, 1, 1, key="num_proposers")
        max_rounds = st.slider("Number of Rounds", 1, 5, 1, 1, key="max_rounds")
        use_search = st.checkbox("🔍 Enable Internet Search (DuckDuckGo)", value=True, help="Allow proposers to search for evidence online", key="use_search")
        use_rag = st.checkbox(
            "📚 Enable knowledge-base RAG (LegalBench)",
            value=False,
            help="Optional: retrieve passages from a local LegalBench index when enabled",
            key="use_rag",
        )

        st.markdown("---")
        st.markdown("### 🚫 Anti-Loop Features (prevent repetitive arguments)")
        force_different_proposers = st.checkbox("🎭 Force Different Proposer Perspectives", value=False, help="Each proposer adopts a unique viewpoint (pro-plaintiff, pro-defendant, neutral, etc.) to reduce redundancy", key="force_different_proposers")
        force_different_rounds = st.checkbox("🔁 Force Different Arguments Per Round", value=False, help="Stronger anti-repetition instruction for multi-round debates to prevent restating previous points", key="force_different_rounds")
        critic_repetition_check = st.checkbox("🔍 Critic Repetition Check", value=False, help="Critic explicitly identifies if proposers repeated points from previous rounds", key="critic_repetition_check")
        negative_constraints = st.checkbox("🚫 Negative Constraints List", value=False, help="Explicitly list points already made and tell proposers to avoid them", key="negative_constraints")
        round_specific_topics = st.checkbox("🎯 Round-Specific Topics", value=False, help="Each round focuses on a different aspect (legal, factual, policy, ethical, economic)", key="round_specific_topics")
        temperature_annealing = st.checkbox("🌡️ Temperature Annealing", value=False, help="Gradually increase temperature in later rounds to force more creative responses", key="temperature_annealing")
        judge_intervention = st.checkbox("⚖️ Judge Mid-Debate Intervention", value=False, help="Judge can intervene mid-debate if looping is detected", key="judge_intervention")
        perspective_rotation = st.checkbox("🔄 Proposer Perspective Rotation", value=False, help="Proposers switch perspectives each round instead of keeping the same one", key="perspective_rotation")
        contradiction_detection = st.checkbox("⚡ Contradiction Detection", value=False, help="Check if proposers contradict their previous stances across rounds", key="contradiction_detection")
        early_termination_loop = st.checkbox("🛑 Early Termination on Loop", value=False, help="Stop debate early if high similarity with previous round is detected", key="early_termination_loop")
        
        st.markdown("#### 🧩 Logical reasoning benchmark")
        st.caption(
            "Use the **KnK benchmark** tab to run experiments on "
            "[K-and-K/knights-and-knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves) (Hugging Face)."
        )
        if st.button("🔎 Quick preview (3 puzzles)", key="struct_knk_preview"):
            try:
                r = requests.get(
                    f"{API_BASE}/benchmarks/knk/preview",
                    params={"config_name": "test", "split": "2ppl", "limit": 3, "offset": 0},
                    timeout=120,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"Split **{data.get('split')}** — {data.get('total_rows', '?')} rows total.")
                    for it in data.get("items", []):
                        with st.expander(f"Puzzle index {it.get('index')}"):
                            st.write(it.get("topic_preview", ""))
                            st.caption(f"Gold: {it.get('solution_text', '')}")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))
    
    with tab_agents:
        st.markdown("### 🤖 Model Selection")
        
        model_data = fetch_models_for_dashboard()
        available_models = model_data.get("models", [])
        groq_models = model_data.get("groq_models", [])
        
        provider_options = ["LM Studio (Local)"]
        if groq_models:
            provider_options.append("Groq (Cloud)")
            
        model_provider_ui = st.selectbox("Model Provider", provider_options, index=0, key="model_provider")
        model_provider = "openai" if model_provider_ui == "LM Studio (Local)" else "groq"
        
        # Filter models based on provider
        if model_provider == "groq":
            available_models = groq_models
        else:
            # Filter out groq models from local list if they were mixed
            available_models = [m for m in available_models if m not in groq_models]

        col_refresh, col_label = st.columns([1, 5])
        with col_refresh:
            refresh_models = st.button("🔄", help="Refresh models from LM Studio")
        with col_label:
            st.write("")
        
        if refresh_models:
            fetch_models_for_dashboard.clear()
            st.rerun()
            
        # available_models is already set above based on provider
        
        if len(available_models) > 0:
            critic_model = st.selectbox("Critic Model", available_models, index=0, key="critic_model")
            judge_model = st.selectbox("Judge Model", available_models, index=0, key="judge_model")
        else:
            st.error("No models available. Check LM Studio connection.")
            critic_model = "liquid/lfm2.5-1.2b"
            judge_model = "liquid/lfm2.5-1.2b"
            
        st.markdown("### 👨‍⚖️ Judge Profile")
        judge_profile = st.selectbox(
            "Judge Reasoning Style",
            ["default", "logical_thinker", "robust_reasoner", "deductive_reasoner"],
            index=0,
            help="Select the judge's reasoning approach",
            key="judge_profile"
        )
            
        st.markdown("### 📝 System Prompts")
        proposer_configs = []
        for i in range(num_proposers):
            with st.expander(f"Proposer {i+1} Configuration", expanded=i == 0):
                proposer_model = st.selectbox(f"Proposer {i+1} Model", available_models, index=0, key=f"proposer_{i}_model")
                proposer_temp = st.slider(f"Proposer {i+1} Temperature", 0.0, 1.0, 0.7, 0.1, key=f"proposer_{i}_temp")
                proposer_prompt = st.text_area(
                    f"Proposer {i+1} System Prompt",
                    value="You are a Proposer in a structured debate. Your role is to generate a well-reasoned legal argument on the given topic.",
                    height=80,
                    key=f"proposer_{i}_prompt"
                )
                proposer_configs.append({
                    "model": proposer_model,
                    "temperature": proposer_temp,
                    "system_prompt": proposer_prompt
                })
        
        with st.expander("Critic Prompt", expanded=False):
            critic_prompt = st.text_area("Edit Critic System Prompt", value="You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument.", height=100, key="critic_prompt")
        
        with st.expander("Judge Prompt", expanded=False):
            judge_prompt = st.text_area("Edit Judge System Prompt", value="You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict.", height=100, key="judge_prompt")
            
    with tab_eval:
        st.markdown("### 📊 Evaluation Features")
        use_position_swap = st.checkbox("🔄 Enable Position Swapping (reduce judge bias)", value=True, help="Run judge evaluation twice with swapped argument order", key="use_position_swap")
        use_info_gain = st.checkbox("📈 Enable Information Gain Metric", value=True, help="Track cosine dissimilarity between consecutive responses", key="use_info_gain")
        use_faithfulness = st.checkbox("🔎 Enable Turn Faithfulness Metric", value=True, help="Calculate percentage of arguments grounded in search results", key="use_faithfulness")
        use_summary_relay = st.checkbox("📝 Enable Summary-Based Relay", value=True, help="Condense previous rounds into summaries to reduce token bloat", key="use_summary_relay")

    with tab_bench:
        st.markdown("### 🧩 Knights & Knaves (logical reasoning)")
        st.markdown(
            "Evaluate your debate stack on the "
            "[K-and-K/knights-and-knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves) "
            "dataset (requires the `datasets` package on the backend and network on first download)."
        )

        model_data_b = fetch_models_for_dashboard()
        avail_b = model_data_b.get("models", [])
        groq_b = model_data_b.get("groq_models", [])
        prov_opts_b = ["LM Studio (Local)"]
        if groq_b:
            prov_opts_b.append("Groq (Cloud)")
        knk_provider_ui = st.selectbox("Provider", prov_opts_b, index=0, key="knk_provider_ui")
        knk_provider = "openai" if knk_provider_ui == "LM Studio (Local)" else "groq"
        if knk_provider == "groq":
            avail_b = groq_b or avail_b
        else:
            avail_b = [m for m in avail_b if m not in groq_b]

        c1, c2, c3 = st.columns(3)
        with c1:
            knk_config = st.selectbox("HF config", ["test", "train"], index=0, key="knk_config_name")
        with c2:
            knk_split = st.selectbox(
                "Split (people)",
                ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"],
                index=0,
                key="knk_split",
            )
        with c3:
            knk_use_full = st.checkbox("Use entire split", value=False, key="knk_use_full_split")

        c4, c5, c6 = st.columns(3)
        with c4:
            knk_limit = st.number_input(
                "Max puzzles (ignored if full split)",
                min_value=1,
                max_value=5000,
                value=10,
                step=1,
                key="knk_limit",
            )
        with c5:
            knk_offset = st.number_input("Offset", min_value=0, max_value=100000, value=0, step=1, key="knk_offset")
        with c6:
            knk_shuffle = st.checkbox("Shuffle before slice", value=False, key="knk_shuffle")

        knk_seed = st.number_input("Shuffle seed (optional)", value=0, step=1, key="knk_seed")
        knk_add_suffix = st.checkbox("Append answer-format instructions to each puzzle", value=True, key="knk_add_suffix")

        st.markdown("---")
        st.markdown("#### Preview")
        if st.button("Load preview from API", key="knk_preview_btn"):
            try:
                pr = requests.get(
                    f"{API_BASE}/benchmarks/knk/preview",
                    params={
                        "config_name": knk_config,
                        "split": knk_split,
                        "limit": 8,
                        "offset": knk_offset,
                    },
                    timeout=120,
                )
                if pr.status_code == 200:
                    st.session_state["knk_preview"] = pr.json()
                else:
                    st.error(pr.text)
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("knk_preview"):
            pv = st.session_state["knk_preview"]
            st.caption(f"Total rows in split: **{pv.get('total_rows', '?')}** — showing **{pv.get('preview_count', 0)}** previews.")
            for it in pv.get("items", []):
                with st.expander(f"Index {it.get('index')}"):
                    st.text(it.get("topic_preview", ""))
                    st.caption(f"Gold: {it.get('solution_text', '')}")

        st.markdown("---")
        st.markdown("#### Run batch experiment on this dataset")
        knk_exp_name = st.text_input("Experiment name", "KnK logical benchmark", key="knk_exp_name")
        knk_rounds = st.slider("Rounds per puzzle", 1, 5, 1, key="knk_max_rounds")
        knk_repeats = st.slider("Repeats per config", 1, 5, 1, key="knk_repeats")

        knk_baseline_model = st.selectbox(
            "Baseline (70B-style) model",
            groq_b if (groq_b and knk_provider == "groq") else (avail_b or ["liquid/lfm2.5-1.2b"]),
            index=0,
            key="knk_baseline_model",
        )
        knk_selected_models = st.multiselect(
            "SLM models (for non-baseline profiles)",
            avail_b or ["liquid/lfm2.5-1.2b"],
            default=[(avail_b or ["liquid/lfm2.5-1.2b"])[0]],
            key="knk_selected_models",
        )
        knk_profiles = st.multiselect(
            "Profiles to run",
            [
                "Baseline (Single 70B Model)",
                "SLM MAD (ReAct Only)",
                "SLM MAD (Naive RAG)",
                "SLM MAD (Active RAG)",
                "SLM MAD (Hybrid / Proposed)",
            ],
            default=["Baseline (Single 70B Model)"],
            key="knk_profiles",
        )

        if st.button("🚀 Start KnK experiment", type="primary", key="knk_start_exp"):
            if not knk_profiles:
                st.error("Select at least one profile.")
            elif not knk_selected_models and any("Baseline" not in p for p in knk_profiles):
                st.error("Select at least one SLM model for non-baseline profiles.")
            else:
                sm = knk_selected_models if knk_selected_models else [(avail_b or ["liquid/lfm2.5-1.2b"])[0]]
                mcfgs = build_experiment_model_configs(
                    sm,
                    knk_profiles,
                    knk_baseline_model,
                    knk_provider,
                )
                if not mcfgs:
                    st.error("No model configurations produced.")
                else:
                    knk_body = {
                        "name": knk_exp_name,
                        "topics": [],
                        "knk_dataset": {
                            "config_name": knk_config,
                            "split": knk_split,
                            "limit": None if knk_use_full else int(knk_limit),
                            "offset": int(knk_offset),
                            "shuffle": bool(knk_shuffle),
                            "seed": int(knk_seed) if knk_shuffle else None,
                            "add_topic_suffix": bool(knk_add_suffix),
                        },
                        "model_configs": mcfgs,
                        "max_rounds": int(knk_rounds),
                        "repeats": int(knk_repeats),
                        "use_rag": False,
                        "use_search": False,
                    }
                    try:
                        resp = requests.post(f"{API_BASE}/experiments/run", json=knk_body, timeout=30)
                        if resp.status_code == 200:
                            eid = resp.json().get("experiment_id", "")
                            st.session_state["knk_last_experiment_id"] = eid
                            st.success(f"Started experiment **{eid}**. Results: `backend/data/experiments/{eid}/`")
                        else:
                            st.error(resp.text)
                    except Exception as e:
                        st.error(str(e))

        if st.session_state.get("knk_last_experiment_id"):
            eid = st.session_state["knk_last_experiment_id"]
            st.info(f"Last started KnK experiment id: `{eid}` — poll status via API or check the Experiments tab.")

    with tab_adv:
        st.markdown("### 🌡️ Temperature Settings")
        critic_temp = st.slider("Critic Temperature", 0.0, 1.0, 0.7, 0.1, key="critic_temp")
        judge_temp = st.slider("Judge Temperature", 0.0, 1.0, 0.5, 0.1, key="judge_temp")
        
        st.markdown("### ⚡ Performance Settings")
        max_tokens = st.slider("Max Tokens (lower = faster)", 100, 2000, 300, 50, key="max_tokens")
        
        st.markdown("---")
        if st.button("🧪 Load Dummy Debate (Test)", key="load_dummy"):
            try:
                dummy_response = requests.get(f"{API_BASE}/debate/dummy")
                if dummy_response.status_code == 200:
                    dummy_data = dummy_response.json()
                    st.session_state.session_id = dummy_data["session_id"]
                    st.session_state.debate_events = dummy_data["events"]
                    st.session_state.debate_result = dummy_data
                    st.session_state.debate_complete = True
                    st.success("✅ Dummy debate loaded!")
                    st.rerun()
                else:
                    st.error(f"Failed to load dummy debate: {dummy_response.text}")
            except Exception as e:
                st.error(f"Error loading dummy debate: {str(e)}")
                
    with tab_exp:
        st.markdown("### 🔬 Batch Experiment Builder")
        exp_name = st.text_input("Experiment Name", "Model Comparison Study", key="exp_name")

        exp_use_knk = st.checkbox(
            "Use Hugging Face **Knights & Knaves** instead of manual topics",
            value=False,
            help="Loads K-and-K/knights-and-knaves puzzles as debate topics (same API as KnK benchmark tab).",
            key="exp_use_knk",
        )
        if exp_use_knk:
            ek1, ek2, ek3 = st.columns(3)
            with ek1:
                exp_knk_config = st.selectbox("HF config", ["test", "train"], index=0, key="exp_knk_config")
            with ek2:
                exp_knk_split = st.selectbox(
                    "Split",
                    ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"],
                    index=0,
                    key="exp_knk_split",
                )
            with ek3:
                exp_knk_full = st.checkbox("Use entire split", value=False, key="exp_knk_full")
            exp_knk_limit = st.number_input(
                "Max puzzles (if not full split)",
                1,
                5000,
                20,
                key="exp_knk_limit",
            )
            exp_knk_offset = st.number_input("KnK offset", 0, 100000, 0, key="exp_knk_offset")
            exp_knk_shuffle = st.checkbox("Shuffle KnK slice", value=False, key="exp_knk_shuffle")
            exp_knk_seed = st.number_input("KnK shuffle seed", 0, 2**31 - 1, 42, key="exp_knk_seed")
            exp_topics_str = ""
            exp_topics = []
            st.caption("Manual topic list is disabled while KnK is enabled.")
        else:
            exp_topics_str = st.text_area(
                "Debate Topics (one per line)",
                "Should AI be granted legal personhood?\nIs digital privacy a human right?",
                key="exp_topics",
            )
            exp_topics = [t.strip() for t in exp_topics_str.split("\n") if t.strip()]
        
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            exp_rounds = st.slider("Rounds per Debate", 1, 5, 2, key="exp_rounds")
            exp_repeats = st.slider("Repeats per Config", 1, 5, 1, key="exp_repeats")
        with col_exp2:
            exp_pilot_mode = st.checkbox("Pilot Mode", value=True, help="Run a small subset first: first 2 topics, 1 repeat, 1 round", key="exp_pilot_mode")
            exp_validate_after = st.checkbox("Validate Results After Run", value=True, help="Check generated outputs for missing metadata, duplicates, and mode mismatches", key="exp_validate_after")
            
        st.markdown("#### 🤖 Model Backbone (for SLM profiles)")
        selected_models = st.multiselect("Select SLM models to use", available_models, default=[available_models[0]] if available_models else [], key="selected_models")
        
        st.markdown("#### Baseline Configuration")
        baseline_provider_ui = st.selectbox(
            "Baseline Provider",
            provider_options,
            index=1 if "Groq (Cloud)" in provider_options else 0,
            key="baseline_provider_ui"
        )
        baseline_provider = "openai" if baseline_provider_ui == "LM Studio (Local)" else "groq"
        baseline_model_options = groq_models if baseline_provider == "groq" and groq_models else available_models
        baseline_model = st.selectbox(
            "Baseline Model",
            baseline_model_options if baseline_model_options else ["liquid/lfm2.5-1.2b"],
            index=0,
            key="baseline_model"
        )
        
        st.markdown("#### ⚖️ Research Configuration Profiles")
        research_profiles = st.multiselect("Select Profiles to Benchmark", [
            "Baseline (Single 70B Model)",
            "SLM MAD (ReAct Only)",
            "SLM MAD (Naive RAG)",
            "SLM MAD (Active RAG)",
            "SLM MAD (Hybrid / Proposed)"
        ], default=["Baseline (Single 70B Model)", "SLM MAD (Hybrid / Proposed)"], key="research_profiles")
        
        effective_topics = exp_topics[:2] if (exp_pilot_mode and not exp_use_knk) else exp_topics
        effective_rounds = 1 if exp_pilot_mode else exp_rounds
        effective_repeats = 1 if exp_pilot_mode else exp_repeats
        if exp_use_knk:
            effective_knk_limit = (
                2
                if (exp_pilot_mode and not exp_knk_full)
                else (None if exp_knk_full else int(exp_knk_limit))
            )
        else:
            effective_knk_limit = None
        slm_profile_count = len([p for p in research_profiles if "Baseline" not in p])
        matrix_config_count = (1 if any("Baseline" in p for p in research_profiles) else 0) + (len(selected_models) * slm_profile_count)
        if exp_use_knk:
            if exp_knk_full:
                st.caption(
                    f"Planned matrix: **full** KnK split × {matrix_config_count} config(s) × {effective_repeats} repeat(s) "
                    f"(total runs = rows in split × configs × repeats)"
                )
            else:
                planned_n = int(effective_knk_limit) if effective_knk_limit is not None else int(exp_knk_limit)
                total_planned_runs = planned_n * matrix_config_count * effective_repeats
                st.caption(
                    f"Planned matrix: {planned_n} KnK puzzle(s) × {matrix_config_count} config(s) × {effective_repeats} repeat(s) = {total_planned_runs} run(s)"
                )
        else:
            planned_n = len(effective_topics)
            total_planned_runs = planned_n * matrix_config_count * effective_repeats
            st.caption(
                f"Planned matrix: {planned_n} manual topic(s) × {matrix_config_count} config(s) × {effective_repeats} repeat(s) = {total_planned_runs} run(s)"
            )

        st.markdown("---")
        st.markdown("### 🚫 Anti-Loop Features (prevent repetitive arguments in experiments)")
        exp_force_different_proposers = st.checkbox("🎭 Force Different Proposer Perspectives", value=False, help="Each proposer adopts a unique viewpoint", key="exp_force_different_proposers")
        exp_force_different_rounds = st.checkbox("🔁 Force Different Arguments Per Round", value=False, help="Stronger anti-repetition instruction", key="exp_force_different_rounds")
        exp_critic_repetition_check = st.checkbox("🔍 Critic Repetition Check", value=False, help="Critic identifies repeated points", key="exp_critic_repetition_check")
        exp_negative_constraints = st.checkbox("🚫 Negative Constraints List", value=False, help="List points to avoid", key="exp_negative_constraints")
        exp_round_specific_topics = st.checkbox("🎯 Round-Specific Topics", value=False, help="Each round focuses on different aspect", key="exp_round_specific_topics")
        exp_temperature_annealing = st.checkbox("🌡️ Temperature Annealing", value=False, help="Increase temperature in later rounds", key="exp_temperature_annealing")
        exp_judge_intervention = st.checkbox("⚖️ Judge Mid-Debate Intervention", value=False, help="Judge can intervene on loops", key="exp_judge_intervention")
        exp_perspective_rotation = st.checkbox("🔄 Proposer Perspective Rotation", value=False, help="Proposers switch perspectives each round", key="exp_perspective_rotation")
        exp_contradiction_detection = st.checkbox("⚡ Contradiction Detection", value=False, help="Check stance contradictions", key="exp_contradiction_detection")
        exp_early_termination_loop = st.checkbox("🛑 Early Termination on Loop", value=False, help="Stop early if high similarity detected", key="exp_early_termination_loop")

        if st.button("🚀 Start Research Experiment", type="primary"):
            if not research_profiles:
                st.error("Please select at least one profile.")
            elif not exp_use_knk and not exp_topics:
                st.error("Provide manual topics or enable Knights & Knaves.")
            elif exp_use_knk and not selected_models and any("Baseline" not in p for p in research_profiles):
                st.error("Select at least one SLM model for non-baseline profiles.")
            else:
                with st.spinner("Starting research experiment..."):
                    sm = selected_models if selected_models else [available_models[0] if available_models else "liquid/lfm2.5-1.2b"]
                    model_configs = build_experiment_model_configs(
                        sm,
                        research_profiles,
                        baseline_model,
                        baseline_provider,
                    )

                    try:
                        exp_payload = {
                            "name": exp_name,
                            "max_rounds": effective_rounds,
                            "repeats": effective_repeats,
                            "use_rag": False,
                            "use_search": False,
                            "force_different_proposers": exp_force_different_proposers,
                            "force_different_rounds": exp_force_different_rounds,
                            "critic_repetition_check": exp_critic_repetition_check,
                            "negative_constraints": exp_negative_constraints,
                            "round_specific_topics": exp_round_specific_topics,
                            "temperature_annealing": exp_temperature_annealing,
                            "judge_intervention": exp_judge_intervention,
                            "perspective_rotation": exp_perspective_rotation,
                            "contradiction_detection": exp_contradiction_detection,
                            "early_termination_loop": exp_early_termination_loop,
                            "model_configs": model_configs,
                        }
                        if exp_use_knk:
                            exp_payload["topics"] = []
                            exp_payload["knk_dataset"] = {
                                "config_name": exp_knk_config,
                                "split": exp_knk_split,
                                "limit": effective_knk_limit,
                                "offset": int(exp_knk_offset),
                                "shuffle": bool(exp_knk_shuffle),
                                "seed": int(exp_knk_seed) if exp_knk_shuffle else None,
                                "add_topic_suffix": True,
                            }
                        else:
                            exp_payload["topics"] = effective_topics

                        exp_resp = requests.post(f"{API_BASE}/experiments/run", json=exp_payload, timeout=30)
                        if exp_resp.status_code == 200:
                            st.success(f"✅ Research experiment started! ID: {exp_resp.json()['experiment_id']}")
                        else:
                            st.error(f"Failed: {exp_resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("### 📋 Active & Past Experiments")
        exp_name_filter = st.text_input(
            "Filter by experiment name",
            "",
            key="exp_name_filter",
            help="Case-insensitive substring; list is sorted A→Z by name.",
        )

        @st.cache_data(ttl=5)
        def get_experiment_catalog():
            try:
                response = requests.get(f"{API_BASE}/experiments/catalog", timeout=30)
                if response.status_code == 200:
                    return response.json().get("experiments", [])
            except Exception:
                pass
            return []

        try:
            exps = get_experiment_catalog()
            q = (exp_name_filter or "").strip().lower()
            if q:
                exps = [e for e in exps if q in (e.get("name") or "").lower()]
            if exps:
                for ex in exps:
                    with st.expander(experiment_row_label(ex), expanded=False):
                        st.write(f"ID: {ex['id']}")
                        ntop = len(ex.get("config", {}).get("topics") or [])
                        st.write(f"Topics / puzzles: {ntop}")
                        st.write(f"Total Runs: {ex['total_runs']}")
                        if ex["status"] == "running":
                            st.info("Running…")
                            st.progress(min(100, max(0, int(ex.get("progress", 0)))))
                        elif ex["status"] == "completed":
                            st.success("Completed!")
                        if ex["status"] in ("completed", "running"):
                            if st.button("📊 View results table", key=f"view_{ex['id']}"):
                                st.session_state["exp_results_view_id"] = ex["id"]
                                st.rerun()
                            if st.session_state.get("exp_results_view_id") == ex["id"]:
                                df_res, err = fetch_experiment_results_dataframe(ex["id"])
                                if df_res is not None and len(df_res) > 0:
                                    st.caption(
                                        f"{len(df_res)} row(s) · status: **{ex['status']}**"
                                    )
                                    st.dataframe(
                                        df_res,
                                        use_container_width=True,
                                        height=min(520, 120 + 28 * len(df_res)),
                                    )
                                    try:
                                        raw = requests.get(
                                            f"{API_BASE}/experiments/{ex['id']}/results.csv",
                                            timeout=120,
                                        )
                                        if raw.status_code == 200:
                                            st.download_button(
                                                label="📥 Download results.csv",
                                                data=raw.content,
                                                file_name=f"experiment_{ex['id']}_results.csv",
                                                mime="text/csv",
                                                key=f"dlcsv_{ex['id']}",
                                            )
                                        else:
                                            st.download_button(
                                                label="📥 Download results.csv",
                                                data=df_res.to_csv(index=False).encode("utf-8"),
                                                file_name=f"experiment_{ex['id']}_results.csv",
                                                mime="text/csv",
                                                key=f"dlcsv_fb_{ex['id']}",
                                            )
                                    except Exception:
                                        st.download_button(
                                            label="📥 Download results.csv",
                                            data=df_res.to_csv(index=False).encode("utf-8"),
                                            file_name=f"experiment_{ex['id']}_results.csv",
                                            mime="text/csv",
                                            key=f"dlcsv_fb2_{ex['id']}",
                                        )
                                    if st.button("✖ Close table", key=f"close_tbl_{ex['id']}"):
                                        st.session_state.pop("exp_results_view_id", None)
                                        st.rerun()
                                elif err:
                                    st.warning(
                                        f"Could not load results (experiment may still be starting): {err}"
                                    )
                                else:
                                    st.info("No rows in results.csv yet.")
                        if ex["status"] == "completed" and exp_validate_after:
                            if st.button("Validate Outputs", key=f"validate_{ex['id']}"):
                                try:
                                    validate_resp = requests.get(
                                        f"{API_BASE}/experiments/validate/{ex['id']}", timeout=10
                                    )
                                    if validate_resp.status_code == 200:
                                        validation = validate_resp.json()
                                        if validation.get("errors"):
                                            st.error("Validation errors found:")
                                            for issue in validation["errors"]:
                                                st.write(f"- {issue}")
                                        else:
                                            st.success("No validation errors found.")
                                        if validation.get("warnings"):
                                            st.warning("Validation warnings:")
                                            for issue in validation["warnings"]:
                                                st.write(f"- {issue}")
                                    else:
                                        st.error(f"Validation failed: {validate_resp.text}")
                                except Exception as e:
                                    st.error(f"Validation error: {e}")
            else:
                st.write("No experiment data found yet.")
        except Exception as e:
            st.error(f"Error loading experiment list: {e}")
                                
        st.markdown("---")
        with st.expander("📊 Research Results Visualizer", expanded=False):
            st.markdown("### 📊 Research Results Visualizer")
            st.caption(
                "Loads `results.csv` via the API. Catalog is sorted by **name** and includes "
                "`experiments_index.json` plus on-disk runs (survives backend restarts)."
            )

            cat_viz = get_experiment_catalog()
            viz_id_labels = {
                e["id"]: experiment_row_label(e) for e in cat_viz if e.get("id")
            }
            exp_ids_visual = list(viz_id_labels.keys())
            if not exp_ids_visual:
                st.info("No experiments in catalog yet.")
            else:
                selected_vid = st.selectbox(
                    "Select experiment",
                    exp_ids_visual,
                    key="viz_exp_select",
                    format_func=lambda x: viz_id_labels.get(x, x),
                )
                df_v, err_v = fetch_experiment_results_dataframe(selected_vid)
                if df_v is not None and len(df_v) > 0:
                    st.write(f"**{len(df_v)}** run(s) loaded.")
                    st.dataframe(
                        df_v,
                        use_container_width=True,
                        height=min(480, 140 + 26 * len(df_v)),
                    )
                    try:
                        raw_v = requests.get(
                            f"{API_BASE}/experiments/{selected_vid}/results.csv",
                            timeout=120,
                        )
                        if raw_v.status_code == 200:
                            st.download_button(
                                "📥 Download results.csv",
                                raw_v.content,
                                file_name=f"experiment_{selected_vid}_results.csv",
                                mime="text/csv",
                                key="viz_dl_csv",
                            )
                        else:
                            st.download_button(
                                "📥 Download results.csv",
                                df_v.to_csv(index=False).encode("utf-8"),
                                file_name=f"experiment_{selected_vid}_results.csv",
                                mime="text/csv",
                                key="viz_dl_fb",
                            )
                    except Exception:
                        st.download_button(
                            "📥 Download results.csv",
                            df_v.to_csv(index=False).encode("utf-8"),
                            file_name=f"experiment_{selected_vid}_results.csv",
                            mime="text/csv",
                            key="viz_dl_fb2",
                        )

                    group_col = "mode" if "mode" in df_v.columns else ("config" if "config" in df_v.columns else None)
                    if group_col and "consensus_score" in df_v.columns:
                        st.markdown(f"#### Consensus score by {group_col}")
                        plot_df = df_v.copy()
                        plot_df["consensus_score"] = pd.to_numeric(plot_df["consensus_score"], errors="coerce")
                        avg_scores = plot_df.groupby(group_col)["consensus_score"].mean().reset_index()
                        st.bar_chart(avg_scores.set_index(group_col))

                    if {"benchmark_type", "config", "knk_gold_match"}.issubset(df_v.columns):
                        st.markdown("#### KNK accuracy by configuration")
                        knk_df = df_v[df_v["benchmark_type"] == "knk"].copy()
                        if not knk_df.empty:
                            knk_df["knk_gold_match"] = pd.to_numeric(knk_df["knk_gold_match"], errors="coerce")
                            knk_acc = knk_df.groupby("config")["knk_gold_match"].mean().mul(100).reset_index()
                            st.bar_chart(knk_acc.set_index("config"))

                    if {"config", "duration_seconds"}.issubset(df_v.columns):
                        st.markdown("#### Latency by configuration")
                        lat_df = df_v.copy()
                        lat_df["duration_seconds"] = pd.to_numeric(lat_df["duration_seconds"], errors="coerce")
                        latency = lat_df.groupby("config")["duration_seconds"].mean().reset_index()
                        st.bar_chart(latency.set_index("config"))

                    if {"config", "spb_score"}.issubset(df_v.columns):
                        st.markdown("#### Position-swap bias by configuration")
                        spb_df = df_v.copy()
                        spb_df["spb_score"] = pd.to_numeric(spb_df["spb_score"], errors="coerce")
                        spb_scores = spb_df.groupby("config")["spb_score"].mean().reset_index()
                        st.bar_chart(spb_scores.set_index("config"))

                    if {"rounds", "config", "consensus_score"}.issubset(df_v.columns):
                        st.markdown("#### Consensus by rounds and configuration")
                        rounds_df = df_v.copy()
                        rounds_df["consensus_score"] = pd.to_numeric(rounds_df["consensus_score"], errors="coerce")
                        rounds_pivot = rounds_df.pivot_table(
                            index="rounds",
                            columns="config",
                            values="consensus_score",
                            aggfunc="mean",
                        )
                        st.line_chart(rounds_pivot)

                    if "avg_info_gain" in df_v.columns:
                        st.markdown("#### Information gain per run")
                        st.line_chart(df_v["avg_info_gain"])
                    col_v1, col_v2 = st.columns(2)
                    if "faithfulness" in df_v.columns:
                        with col_v1:
                            st.markdown("#### Turn faithfulness")
                            st.bar_chart(df_v["faithfulness"])
                    if "format_adherence" in df_v.columns:
                        with col_v2:
                            st.markdown("#### Format adherence (%)")
                            st.bar_chart(df_v["format_adherence"])
                elif err_v:
                    st.warning(f"Could not load results: {err_v}")
                else:
                    st.info("Empty results file.")

        st.markdown("---")
        if st.button("📝 Export System Prompts for Submission"):
            try:
                resp = requests.get(f"{API_BASE}/prompts/export", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("✅ Prompts retrieved from backend!")
                    st.download_button("📥 Download prompts.txt", data["content"], file_name=data["filename"])
                else:
                    st.error(f"Failed to export prompts: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Initialize debate button
    st.markdown("---")
    start_btn = st.button("🚀 Start Debate", type="primary", use_container_width=True)
    
    if st.session_state.session_id:
        st.info(f"Active Session: {st.session_state.session_id}")
    
    # Start debate
    if start_btn:
        if not topic or not topic.strip():
            st.error("❌ Please enter a debate topic before starting.")
        else:
            with st.spinner("Initializing debate..."):
                try:
                    # First check if server is running
                    health_check = requests.get(f"{API_BASE}/", timeout=2)
                    if health_check.status_code != 200:
                        st.error(f"Server not responding correctly. Status: {health_check.status_code}")
                        st.info("Make sure you're running: uvicorn main:app --reload")
                        st.stop()
                    response = requests.post(
                        f"{API_BASE}/debate/init",
                        json={
                            "topic": topic,
                            "proposers": proposer_configs,
                            "critic_model": critic_model,
                            "judge_model": judge_model,
                            "critic_temperature": critic_temp,
                            "judge_temperature": judge_temp,
                            "critic_prompt": critic_prompt,
                            "judge_prompt": judge_prompt,
                            "judge_profile": judge_profile,
                            "use_position_swap": use_position_swap,
                            "use_info_gain": use_info_gain,
                            "use_faithfulness": use_faithfulness,
                            "use_summary_relay": use_summary_relay,
                            "max_rounds": max_rounds,
                            "max_tokens": max_tokens,
                            "use_search": use_search,
                            "use_rag": use_rag,
                            "model_provider": model_provider,
                            "force_different_proposers": force_different_proposers,
                            "force_different_rounds": force_different_rounds,
                            "critic_repetition_check": critic_repetition_check,
                            "negative_constraints": negative_constraints,
                            "round_specific_topics": round_specific_topics,
                            "temperature_annealing": temperature_annealing,
                            "judge_intervention": judge_intervention,
                            "perspective_rotation": perspective_rotation,
                            "contradiction_detection": contradiction_detection,
                            "early_termination_loop": early_termination_loop
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.debate_events = []
                        st.session_state.debate_result = None
                        st.session_state.debate_complete = False
                        st.success(f"Debate initialized! Session ID: {data['session_id']}")
                        st.rerun()
                    else:
                        st.error(f"Failed to initialize debate. Status: {response.status_code}")
                        st.error(f"Response: {response.text}")
                        st.info("Make sure you're running main.py (not main_simple.py)")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server.")
                    st.info(f"Expected server at: {API_BASE}")
                    st.info("Start the server with: uvicorn main:app --reload")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

# RIGHT COLUMN: Debate Results
with col_debate:
    st.markdown("## 📡 Debate Output")
    
    # Poll debate events (non-blocking)
    if st.session_state.session_id and not st.session_state.debate_result and not st.session_state.debate_complete:
        st.markdown("### Real-time Debate Log")

        # Prominent warning for long debates
        st.warning(
            f"⏳ Debate **{st.session_state.session_id[:8]}...** is running. "
            "Do not refresh this page. If you accidentally refresh, check the sidebar Recent Debates — "
            "the result will appear there automatically when finished.",
            icon="⚠️"
        )
        st.info(
            "This page auto-refreshes every 2 seconds while the debate is active. "
            "All debates are automatically saved to the database when they complete."
        )

        try:
            resp = requests.get(f"{API_BASE}/debate/events/{st.session_state.session_id}", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                events = data.get("events", [])
                is_complete = data.get("complete", False)

                # Update events if new ones arrived
                if len(events) > len(st.session_state.debate_events):
                    st.session_state.debate_events = events

                # Render all events
                for event in st.session_state.debate_events:
                    display_event(event)

                if is_complete:
                    result_resp = requests.get(f"{API_BASE}/debate/result/{st.session_state.session_id}")
                    if result_resp.status_code == 200:
                        st.session_state.debate_result = result_resp.json()
                        st.session_state.debate_complete = True
                        st.session_state.debate_events = events
                        st.rerun()
                    else:
                        st.warning(f"Debate complete but result fetch failed: {result_resp.status_code}. Retrying...")
            else:
                st.warning(f"Event poll failed: {resp.status_code}. Retrying...")
        except Exception as e:
            st.warning(f"Poll error: {str(e)}. Retrying...")

        # Schedule rerun to poll again (maintains session state unlike meta-refresh)
        # Use time-based throttling to prevent tight loop
        if not st.session_state.debate_complete:
            if "last_rerun" not in st.session_state or time.time() - st.session_state.last_rerun > 2:
                st.session_state.last_rerun = time.time()
                st.rerun()

    # Display final results
    if st.session_state.debate_result:
        st.markdown("### 🏁 Final Results")
        st.success("✅ This debate was automatically saved to the database.")

        # Manual save button (redundant but kept as explicit backup)
        if st.button("💾 Save Again (Manual Backup)", key="save_debate"):
            try:
                # Get topic from events
                topic = "Unknown Topic"
                for event in st.session_state.debate_events:
                    if event["event_type"] == "DEBATE_START":
                        topic = event["data"].get("topic", "Unknown Topic")
                        break
                
                save_response = requests.post(
                    f"{API_BASE}/debate/save",
                    json={
                        "session_id": st.session_state.session_id,
                        "topic": topic,
                        "events": st.session_state.debate_events,
                        "result": st.session_state.debate_result
                    }
                )
                
                if save_response.status_code == 200:
                    st.success("✅ Debate saved successfully!")
                    st.balloons()
                else:
                    st.error(f"Failed to save debate: {save_response.text}")
            except Exception as e:
                st.error(f"Error saving debate: {str(e)}")
        
        # Key metrics prominently displayed
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Verdict", st.session_state.debate_result.get("verdict", "N/A"))
        with col2:
            st.metric("Consensus Score", st.session_state.debate_result.get("consensus_score", 0))
        with col3:
            faithfulness_scores = st.session_state.debate_result.get('metrics', {}).get('turn_faithfulness', [0])
            faithfulness_pct = faithfulness_scores[-1] * 100 if faithfulness_scores else 0
            st.metric("Turn Faithfulness", f"{faithfulness_pct:.0f}%")
        with col4:
            st.metric("Proposers", st.session_state.debate_result.get("num_proposers", 1))
        with col5:
            # Show judge bias if position swap was used
            metrics = st.session_state.debate_result.get('metrics', {})
            if 'position_swap_scores' in metrics and metrics['position_swap_scores']:
                normal_cons = metrics['position_swap_scores'][-1].get('normal', {}).get('consensus', 0)
                swapped_cons = metrics['position_swap_scores'][-1].get('swapped', {}).get('consensus', 0)
                bias = abs(normal_cons - swapped_cons)
                st.metric("Judge Bias", f"{bias:.0f}")
            else:
                st.metric("Judge Bias", "N/A")
            st.metric("Rounds", st.session_state.debate_result.get("num_rounds", 1))
            
        # Display Evaluation Metrics
        metrics = st.session_state.debate_result.get("metrics")
        if metrics:
            st.markdown("### 📊 Evaluation Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Avg Information Gain", f"{metrics.get('average_information_gain', 0):.2f}")
                adherence = metrics.get('format_adherence_percent', 0)
                st.metric("Format Adherence", f"{adherence:.1f}%")
            with m_col2:
                faithfulness_scores = metrics.get('turn_faithfulness', [])
                avg_faith = (sum(faithfulness_scores) / len(faithfulness_scores)) * 100 if faithfulness_scores else 0
                st.metric("Avg Turn Faithfulness", f"{avg_faith:.1f}%")
                
                search_eff = metrics.get('search_efficiency', {})
                total_s = search_eff.get('total_searches', 0)
                empty_s = search_eff.get('empty_searches', 0)
                st.metric("Search Efficiency", f"{total_s - empty_s}/{total_s} effective")
            with m_col3:
                ps_scores = metrics.get('position_swap_scores', [])
                if ps_scores:
                    last_swap = ps_scores[-1]
                    normal_c = last_swap.get('normal', {}).get('consensus', 0)
                    swapped_c = last_swap.get('swapped', {}).get('consensus', 0)
                    st.metric("Position Swap Delta", f"{abs(normal_c - swapped_c)} pts")
                else:
                    st.metric("Position Swap Delta", "N/A")
                
                spb = metrics.get('spb_score', 0)
                st.metric("SPB Bias Score", f"{spb:.1f}")
                    
            if metrics.get("is_repetitive_loop"):
                st.warning("⚠️ Warning: Debate detected as a repetitive loop (low information gain).")
        
        # Full responses
        st.markdown("### 📝 Full Debate Transcript")
        
        # Display all proposer responses by round
        proposer_responses = st.session_state.debate_result.get("proposer_responses", [])
        critic_responses = st.session_state.debate_result.get("critic_responses", [])
        search_results = st.session_state.debate_result.get("search_results", [])
        
        for round_num, (round_props, round_critique) in enumerate(zip(proposer_responses, critic_responses), 1):
            with st.expander(f"🔄 Round {round_num}", expanded=round_num == 1):
                # Round search results (if available)
                if round_num <= len(search_results):
                    round_searches = search_results[round_num-1]
                    for i, s_res in enumerate(round_searches, 1):
                        if s_res:
                            with st.container():
                                st.markdown(f"**🔍 Proposer {i} Research Sources:**")
                                st.info(s_res)
                
                st.markdown("---")
                
                for i, prop_response in enumerate(round_props, 1):
                    st.markdown(f"**🗣️ Proposer {i}:**")
                    st.markdown(prop_response)
                    st.markdown("---")
                
                st.markdown(f"**🔍 Critic's Critique:**")
                st.markdown(round_critique)
        
        # Judge's verdict
        with st.expander("⚖️ Judge's Verdict", expanded=True):
            judge_resp = st.session_state.debate_result.get("judge_response", "No response")
            if not judge_resp or judge_resp == "No response":
                st.warning("Judge response not available. Check if debate completed successfully.")
            st.markdown(judge_resp)
    
    # Display events if no result yet
    if st.session_state.debate_events and not st.session_state.debate_result:
        st.markdown("### Real-time Debate Log")
        for event in st.session_state.debate_events:
            display_event(event)
