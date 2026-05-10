import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Debate Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8001"

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
            st.warning(f"LegalBench retrieval started for proposer {proposer_id}.")
        else:
            st.warning("LegalBench retrieval started.")
    elif event_type == "RETRIEVAL_COMPLETE":
        proposer_id = data.get("proposer_id")
        label = f"Proposer {proposer_id}" if proposer_id else "Shared Retrieval"
        st.info(f"{label} received LegalBench evidence.")
        with st.expander(f"View LegalBench Retrieval ({label})"):
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
                btn_label = f"{rd['topic'][:50]}..." if len(rd['topic']) > 50 else rd['topic']
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(btn_label, key=f"hist_btn_{rd['session_id']}", help=rd['topic']):
                        st.session_state.session_id = rd['session_id']
                        st.session_state.debate_events = []
                        st.session_state.debate_result = None
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
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Organize parameters into Tabs
    tab_struct, tab_agents, tab_eval, tab_adv, tab_exp = st.tabs([
        "🏗️ Structure", "🤖 Agents", "📊 Evaluation", "⚙️ Advanced", "🔬 Experiments"
    ])
    
    with tab_struct:
        st.markdown("### 🔄 Debate Structure")
        num_proposers = st.slider("Number of Proposers", 1, 5, 1, 1)
        max_rounds = st.slider("Number of Rounds", 1, 5, 1, 1)
        use_search = st.checkbox("🔍 Enable Internet Search (DuckDuckGo)", value=True, help="Allow proposers to search for evidence online")
        use_rag = st.checkbox("📚 Enable LegalBench RAG", value=False, help="Use local LegalBench dataset for legal context")
        
        if st.button("📊 Analyze LegalBench Dataset"):
            st.markdown("#### 📚 Dataset Analytics (LegalBench-RAG)")
            # Mock analytics for now, or real ones if RAG service available
            col1, col2 = st.columns(2)
            col1.metric("Total Documents", "698")
            col2.metric("Total Tokens", "~79M")
            st.info("Dataset includes: NDAs, M&A agreements, commercial contracts, and privacy policies.")
            
            # Simple bar chart of contract types (mock data based on LegalBench spec)
            data = {"Contract Type": ["NDA", "M&A", "Privacy", "Commercial"], "Count": [210, 150, 180, 158]}
            df_rag = pd.DataFrame(data)
            st.bar_chart(df_rag.set_index("Contract Type"))
    
    with tab_agents:
        # Fetch available models from LM Studio
        @st.cache_data(ttl=300)
        def get_model_data():
            """Fetch models and provider info from the backend."""
            try:
                response = requests.get(f"{API_BASE}/models", timeout=5)
                if response.status_code == 200:
                    return response.json()
                return {"models": ["liquid/lfm2.5-1.2b"]}
            except Exception:
                return {"models": ["liquid/lfm2.5-1.2b"]}

        st.markdown("### 🤖 Model Selection")
        
        model_data = get_model_data()
        available_models = model_data.get("models", [])
        groq_models = model_data.get("groq_models", [])
        
        provider_options = ["LM Studio (Local)"]
        if groq_models:
            provider_options.append("Groq (Cloud)")
            
        model_provider_ui = st.selectbox("Model Provider", provider_options, index=0)
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
            st.cache_data.clear()
            st.rerun()
            
        # available_models is already set above based on provider
        
        if len(available_models) > 0:
            critic_model = st.selectbox("Critic Model", available_models, index=0)
            judge_model = st.selectbox("Judge Model", available_models, index=0)
        else:
            st.error("No models available. Check LM Studio connection.")
            critic_model = "liquid/lfm2.5-1.2b"
            judge_model = "liquid/lfm2.5-1.2b"
            
        st.markdown("### 👨‍⚖️ Judge Profile")
        judge_profile = st.selectbox(
            "Judge Reasoning Style",
            ["default", "logical_thinker", "robust_reasoner", "deductive_reasoner"],
            index=0,
            help="Select the judge's reasoning approach"
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
            critic_prompt = st.text_area("Edit Critic System Prompt", value="You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument.", height=100)
        
        with st.expander("Judge Prompt", expanded=False):
            judge_prompt = st.text_area("Edit Judge System Prompt", value="You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict.", height=100)
            
    with tab_eval:
        st.markdown("### 📊 Evaluation Features")
        use_position_swap = st.checkbox("🔄 Enable Position Swapping (reduce judge bias)", value=True, help="Run judge evaluation twice with swapped argument order")
        use_info_gain = st.checkbox("📈 Enable Information Gain Metric", value=True, help="Track cosine dissimilarity between consecutive responses")
        use_faithfulness = st.checkbox("🔎 Enable Turn Faithfulness Metric", value=True, help="Calculate percentage of arguments grounded in search results")
        use_summary_relay = st.checkbox("📝 Enable Summary-Based Relay", value=True, help="Condense previous rounds into summaries to reduce token bloat")
        
    with tab_adv:
        st.markdown("### 🌡️ Temperature Settings")
        critic_temp = st.slider("Critic Temperature", 0.0, 1.0, 0.7, 0.1)
        judge_temp = st.slider("Judge Temperature", 0.0, 1.0, 0.5, 0.1)
        
        st.markdown("### ⚡ Performance Settings")
        max_tokens = st.slider("Max Tokens (lower = faster)", 100, 2000, 500, 50)
        
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
        exp_name = st.text_input("Experiment Name", "Model Comparison Study")
        
        # Multi-topic selection
        exp_topics_str = st.text_area("Debate Topics (one per line)", 
            "Should AI be granted legal personhood?\nIs digital privacy a human right?")
        exp_topics = [t.strip() for t in exp_topics_str.split("\n") if t.strip()]
        
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            exp_rounds = st.slider("Rounds per Debate", 1, 5, 2)
            exp_repeats = st.slider("Repeats per Config", 1, 5, 1)
        with col_exp2:
            exp_pilot_mode = st.checkbox("Pilot Mode", value=True, help="Run a small subset first: first 2 topics, 1 repeat, 1 round")
            exp_validate_after = st.checkbox("Validate Results After Run", value=True, help="Check generated outputs for missing metadata, duplicates, and mode mismatches")
            
        st.markdown("#### 🤖 Model Backbone (for SLM profiles)")
        selected_models = st.multiselect("Select SLM models to use", available_models, default=[available_models[0]] if available_models else [])
        
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
        ], default=["Baseline (Single 70B Model)", "SLM MAD (Hybrid / Proposed)"])
        
        effective_topics = exp_topics[:2] if exp_pilot_mode else exp_topics
        effective_rounds = 1 if exp_pilot_mode else exp_rounds
        effective_repeats = 1 if exp_pilot_mode else exp_repeats
        slm_profile_count = len([p for p in research_profiles if "Baseline" not in p])
        matrix_config_count = (1 if any("Baseline" in p for p in research_profiles) else 0) + (len(selected_models) * slm_profile_count)
        total_planned_runs = len(effective_topics) * matrix_config_count * effective_repeats
        st.caption(f"Planned matrix: {len(effective_topics)} topic(s) x {matrix_config_count} config(s) x {effective_repeats} repeat(s) = {total_planned_runs} run(s)")
        
        if st.button("🚀 Start Research Experiment", type="primary"):
            if not exp_topics or not research_profiles:
                st.error("Please provide topics and select profiles.")
            else:
                with st.spinner("Starting research experiment..."):
                    model_configs = []
                    for profile in research_profiles:
                        if "Baseline" in profile:
                            model_configs.append({
                                "proposer_model": baseline_model,
                                "critic_model": baseline_model,
                                "judge_model": baseline_model,
                                "provider": baseline_provider,
                                "mode": "baseline"
                            })
                        elif "ReAct Only" in profile:
                            for model_name in selected_models:
                                model_configs.append({
                                    "mode": "react_only",
                                    "provider": "openai",
                                    "proposer_model": model_name,
                                    "critic_model": model_name,
                                    "judge_model": model_name
                                })
                        elif "Naive RAG" in profile:
                            for model_name in selected_models:
                                model_configs.append({
                                    "mode": "naive_rag",
                                    "provider": "openai",
                                    "proposer_model": model_name,
                                    "critic_model": model_name,
                                    "judge_model": model_name
                                })
                        elif "Active RAG" in profile:
                            for model_name in selected_models:
                                model_configs.append({
                                    "mode": "active_rag",
                                    "provider": "openai",
                                    "proposer_model": model_name,
                                    "critic_model": model_name,
                                    "judge_model": model_name
                                })
                        elif "Hybrid" in profile:
                            for model_name in selected_models:
                                model_configs.append({
                                    "mode": "hybrid",
                                    "provider": "openai",
                                    "proposer_model": model_name,
                                    "critic_model": model_name,
                                    "judge_model": model_name
                                })

                    try:
                        exp_resp = requests.post(f"{API_BASE}/experiments/run", json={
                            "name": exp_name,
                            "topics": effective_topics,
                            "model_configs": model_configs,
                            "max_rounds": effective_rounds,
                            "repeats": effective_repeats,
                            "use_rag": False,
                            "use_search": False
                        })
                        if exp_resp.status_code == 200:
                            st.success(f"✅ Research experiment started! ID: {exp_resp.json()['experiment_id']}")
                        else:
                            st.error(f"Failed: {exp_resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("### 📋 Active & Past Experiments")
        @st.cache_data(ttl=10)
        def get_experiments():
            try:
                response = requests.get(f"{API_BASE}/experiments/list", timeout=2)
                if response.status_code == 200:
                    return response.json().get("experiments", [])
                return []
            except:
                return []

        try:
            exps = get_experiments()
            if exps:
                for ex in reversed(exps):
                        with st.expander(f"{ex['name']} - {ex['status'].upper()}"):
                            st.write(f"ID: {ex['id']}")
                            st.write(f"Topics: {len(ex['config']['topics'])}")
                            st.write(f"Total Runs: {ex['total_runs']}")
                            if ex['status'] == 'completed':
                                # Provide link to CSV results (assuming server serves it or we show a preview)
                                st.success("Completed!")
                                if st.button("📊 View Results Summary", key=f"view_{ex['id']}"):
                                    st.info("Results table feature coming soon! Check backend/data/experiments/ for CSV.")
                                if exp_validate_after and st.button("Validate Outputs", key=f"validate_{ex['id']}"):
                                    try:
                                        validate_resp = requests.get(f"{API_BASE}/experiments/validate/{ex['id']}", timeout=10)
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
                            elif ex['status'] == 'running':
                                st.info("Running...")
                                st.progress(ex.get('progress', 0))
            else:
                st.write("No experiment data found yet.")
        except Exception as e:
            st.error(f"Error loading experiment list: {e}")
                                
        st.markdown("---")
        with st.expander("📊 Research Results Visualizer", expanded=False):
            st.markdown("### 📊 Research Results Visualizer")
        
        # Look for experiment folders
        exp_base = "backend/data/experiments"
        if os.path.exists(exp_base):
            exp_folders = [d for d in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, d))]
            if exp_folders:
                selected_exp = st.selectbox("Select Experiment to Visualize", exp_folders)
                csv_path = f"{exp_base}/{selected_exp}/results.csv"
                
                if os.path.exists(csv_path):
                    @st.cache_data(ttl=60)
                    def load_exp_results(path):
                        return pd.read_csv(path)
                        
                    df = load_exp_results(csv_path)
                    st.write(f"Showing results for {len(df)} runs.")
                    
                    # Visualization 1: Consensus Score by Configuration
                    st.markdown("#### Consensus Score by Research Configuration")
                    if "mode" in df.columns:
                        avg_scores = df.groupby("mode")["consensus_score"].mean().reset_index()
                        st.bar_chart(avg_scores.set_index("mode"))
                    
                    # Visualization 2: Information Gain (Debate Depth)
                    st.markdown("#### Average Information Gain per Run")
                    st.line_chart(df["avg_info_gain"])
                    
                    # Visualization 3: Faithfulness & Efficiency
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.markdown("#### Turn Faithfulness")
                        st.bar_chart(df["faithfulness"])
                    with col_v2:
                        st.markdown("#### Format Adherence (%)")
                        st.bar_chart(df["format_adherence"])
                        
                    st.download_button("📥 Download Aggregated CSV", 
                                      data=df.to_csv(index=False), 
                                      file_name=f"experiment_{selected_exp}.csv")
                else:
                    st.write("No results.csv found in this folder.")
            else:
                st.write("No experiment data found yet.")
        
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
    if start_btn and topic:
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
                        "model_provider": model_provider
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.debate_events = []
                    st.session_state.debate_result = None
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
    
    # Poll debate events
    if st.session_state.session_id and not st.session_state.debate_result and not st.session_state.debate_complete:
        st.markdown("### Real-time Debate Log")
        
        # Check if already complete first (Optimization)
        try:
            status_resp = requests.get(f"{API_BASE}/debate/events/{st.session_state.session_id}", timeout=2)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                if status_data.get("complete"):
                    result_response = requests.get(f"{API_BASE}/debate/result/{st.session_state.session_id}")
                    if result_response.status_code == 200:
                        st.session_state.debate_result = result_response.json()
                        st.session_state.debate_complete = True
                        st.session_state.debate_events = status_data.get("events", [])
                        st.rerun()
        except:
            pass

        # Create a container for events
        events_container = st.container()
        
        # Poll for events
        max_polls = 100  # Prevent infinite polling
        poll_count = 0
        
        while poll_count < max_polls and not st.session_state.debate_result and not st.session_state.debate_complete:
            try:
                response = requests.get(f"{API_BASE}/debate/events/{st.session_state.session_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get("events", [])
                    is_complete = data.get("complete", False)
                    
                    # Display only new events
                    if len(events) > len(st.session_state.debate_events):
                        new_events = events[len(st.session_state.debate_events):]
                        st.session_state.debate_events = events
                        
                        # Display new events
                        for event in new_events:
                            display_event(event)
                    
                    # Check if complete
                    if is_complete:
                        # Get final result
                        result_response = requests.get(f"{API_BASE}/debate/result/{st.session_state.session_id}")
                        if result_response.status_code == 200:
                            st.session_state.debate_result = result_response.json()
                            st.session_state.debate_complete = True
                            st.success("🎉 Debate Complete!")
                            st.rerun()
                        else:
                            st.error(f"Failed to fetch result: {result_response.text}")
                        break
                    
                    poll_count += 1
                    time.sleep(1)  # Poll every second
                else:
                    st.error(f"Error fetching events: {response.text}")
                    break
            except Exception as e:
                st.error(f"Error polling events: {str(e)}")
                break
    
    # Display final results
    if st.session_state.debate_result:
        st.markdown("### 🏁 Final Results")
        
        # Save button
        if st.button("💾 Save Debate to Database", key="save_debate"):
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
                else:
                    st.error(f"Failed to save debate: {save_response.text}")
            except Exception as e:
                st.error(f"Error saving debate: {str(e)}")
        
        # Verdict and consensus score
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Verdict", st.session_state.debate_result.get("verdict", "N/A"))
        with col2:
            st.metric("Consensus Score", st.session_state.debate_result.get("consensus_score", 0))
        with col3:
            st.metric("Proposers", st.session_state.debate_result.get("num_proposers", 1))
        with col4:
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
