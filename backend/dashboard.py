import streamlit as st
import requests
import json
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Debate Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000"

def display_event(event: Dict[str, Any]):
    """Display a single event with appropriate styling."""
    event_type = event["event_type"]
    data = event["data"]
    
    if event_type == "ROUND_START":
        st.markdown(f"---")
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

# Main title
st.title("⚖️ Multi-Agent Debate Research Platform")

# Two-column layout
col_config, col_debate = st.columns([1, 1])

# LEFT COLUMN: Configuration
with col_config:
    st.markdown("## ⚙️ Configuration")
    
    # Fetch available models from LM Studio
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_models():
        """Fetch available models from the backend."""
        try:
            response = requests.get(f"{API_BASE}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "warning" in data:
                    st.warning(data["warning"])
                return data.get("models", ["liquid/lfm2.5-1.2b"])
            return ["liquid/lfm2.5-1.2b"]
        except:
            return ["liquid/lfm2.5-1.2b"]
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    
    # Add refresh button
    col_refresh, col_label = st.columns([1, 5])
    with col_refresh:
        refresh_models = st.button("🔄", help="Refresh models from LM Studio")
    with col_label:
        st.write("")
    
    # Clear cache and re-fetch if refresh button clicked
    if refresh_models:
        st.cache_data.clear()
        st.rerun()
    
    available_models = get_available_models()
    
    if len(available_models) > 0:
        critic_model = st.selectbox(
            "Critic Model",
            available_models,
            index=0
        )
        judge_model = st.selectbox(
            "Judge Model",
            available_models,
            index=0
        )
    else:
        st.error("No models available. Check LM Studio connection.")
        critic_model = "liquid/lfm2.5-1.2b"
        judge_model = "liquid/lfm2.5-1.2b"
    
    # Temperature sliders
    st.markdown("### 🌡️ Temperature Settings")
    critic_temp = st.slider("Critic Temperature", 0.0, 1.0, 0.7, 0.1)
    judge_temp = st.slider("Judge Temperature", 0.0, 1.0, 0.5, 0.1)
    
    # Max tokens for speed control
    st.markdown("### ⚡ Performance Settings")
    max_tokens = st.slider("Max Tokens (lower = faster)", 100, 2000, 500, 50)
    
    # Debate structure settings
    st.markdown("### 🔄 Debate Structure")
    num_proposers = st.slider("Number of Proposers", 1, 5, 1, 1)
    max_rounds = st.slider("Number of Rounds", 1, 5, 1, 1)
    
    # System prompts
    st.markdown("### 📝 System Prompts")
    
    # Proposer configurations (dynamic based on num_proposers)
    proposer_configs = []
    for i in range(num_proposers):
        with st.expander(f"Proposer {i+1} Configuration", expanded=i == 0):
            proposer_model = st.selectbox(
                f"Proposer {i+1} Model",
                available_models,
                index=0,
                key=f"proposer_{i}_model"
            )
            proposer_temp = st.slider(
                f"Proposer {i+1} Temperature",
                0.0, 1.0, 0.7, 0.1,
                key=f"proposer_{i}_temp"
            )
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
        critic_prompt = st.text_area(
            "Edit Critic System Prompt",
            value="You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument.",
            height=100
        )
    
    with st.expander("Judge Prompt", expanded=False):
        judge_prompt = st.text_area(
            "Edit Judge System Prompt",
            value="You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict.",
            height=100
        )
    
    # Topic input
    st.markdown("### 💬 Debate Topic")
    topic = st.text_input(
        "Enter a topic for the debate",
        placeholder="e.g., 'Should AI be granted legal personhood?'",
        value=""
    )
    
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
                        "max_rounds": max_rounds,
                        "max_tokens": max_tokens
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
    if st.session_state.session_id and not st.session_state.debate_result:
        st.markdown("### Real-time Debate Log")
        
        # Create placeholder for events
        event_placeholder = st.empty()
        
        # Poll for events
        max_polls = 100  # Prevent infinite polling
        poll_count = 0
        
        while poll_count < max_polls and not st.session_state.debate_result:
            try:
                response = requests.get(f"{API_BASE}/debate/events/{st.session_state.session_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get("events", [])
                    is_complete = data.get("complete", False)
                    
                    # Update events
                    if len(events) > len(st.session_state.debate_events):
                        st.session_state.debate_events = events
                    
                    # Display current events
                    with event_placeholder.container():
                        for event in st.session_state.debate_events:
                            display_event(event)
                    
                    # Check if complete
                    if is_complete:
                        # Get final result
                        result_response = requests.get(f"{API_BASE}/debate/result/{st.session_state.session_id}")
                        if result_response.status_code == 200:
                            st.session_state.debate_result = result_response.json()
                            st.success("🎉 Debate Complete!")
                            st.rerun()
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
        
        # Full responses
        st.markdown("### 📝 Full Debate Transcript")
        
        # Display all proposer responses by round
        proposer_responses = st.session_state.debate_result.get("proposer_responses", [])
        critic_responses = st.session_state.debate_result.get("critic_responses", [])
        
        for round_num, (round_props, round_critique) in enumerate(zip(proposer_responses, critic_responses), 1):
            with st.expander(f"🔄 Round {round_num}", expanded=round_num == 1):
                for i, prop_response in enumerate(round_props, 1):
                    st.markdown(f"**Proposer {i}:**")
                    st.markdown(prop_response)
                    st.markdown("---")
                
                st.markdown(f"**Critic's Critique:**")
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
