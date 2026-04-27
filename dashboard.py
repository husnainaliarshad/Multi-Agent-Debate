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
    
    if event_type == "PROPOSER_START":
        st.info("🗣️ Proposer is analyzing the topic...")
    elif event_type == "PROPOSER_THOUGHT":
        with st.expander("💭 Proposer's Thought Process"):
            st.text(data.get("thought", ""))
    elif event_type == "PROPOSER_FINAL":
        st.success("✅ Proposer's argument complete")
        with st.expander("View Proposer's Response"):
            st.markdown(data.get("response", ""))
    elif event_type == "CRITIC_START":
        st.info("🔍 Critic is analyzing the argument...")
    elif event_type == "CRITIC_THOUGHT":
        with st.expander("💭 Critic's Thought Process"):
            st.text(data.get("thought", ""))
    elif event_type == "CRITIC_FINAL":
        st.success("✅ Critic's analysis complete")
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
    elif event_type == "ERROR":
        st.error(f"❌ Error: {data.get('error', 'Unknown error')}")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debate_events" not in st.session_state:
    st.session_state.debate_events = []
if "debate_result" not in st.session_state:
    st.session_state.debate_result = None

# Sidebar configuration
st.sidebar.title("⚙️ Debate Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
proposer_model = st.sidebar.selectbox(
    "Proposer Model",
    ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"],
    index=0
)
critic_model = st.sidebar.selectbox(
    "Critic Model",
    ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"],
    index=0
)
judge_model = st.sidebar.selectbox(
    "Judge Model",
    ["liquid/lfm2.5-1.2b", "liquid/lfm2.5-3b", "llama-3.2-3b"],
    index=0
)

# Temperature sliders
st.sidebar.subheader("Temperature Settings")
proposer_temp = st.sidebar.slider("Proposer Temperature", 0.0, 1.0, 0.7, 0.1)
critic_temp = st.sidebar.slider("Critic Temperature", 0.0, 1.0, 0.7, 0.1)
judge_temp = st.sidebar.slider("Judge Temperature", 0.0, 1.0, 0.5, 0.1)

# Max tokens for speed control
st.sidebar.subheader("Performance Settings")
max_tokens = st.sidebar.slider("Max Tokens (lower = faster)", 100, 2000, 500, 50)

# System prompts
st.sidebar.subheader("System Prompts")
with st.sidebar.expander("Proposer Prompt"):
    proposer_prompt = st.text_area(
        "Edit Proposer System Prompt",
        value="You are a Proposer in a structured debate. Your role is to generate a well-reasoned legal argument on the given topic.",
        height=100
    )

with st.sidebar.expander("Critic Prompt"):
    critic_prompt = st.text_area(
        "Edit Critic System Prompt",
        value="You are a Critic in a structured debate. Your role is to identify logical fallacies, counter-points, and weaknesses in the Proposer's argument.",
        height=100
    )

with st.sidebar.expander("Judge Prompt"):
    judge_prompt = st.text_area(
        "Edit Judge System Prompt",
        value="You are a Judge in a structured debate. Your role is to synthesize both the Proposer's and Critic's arguments and provide a balanced verdict.",
        height=100
    )

# Main area
st.title("⚖️ Multi-Agent Debate Research Platform")

# Topic input
topic = st.text_input(
    "Debate Topic",
    placeholder="Enter a topic for the debate (e.g., 'Should AI be granted legal personhood?')",
    value=""
)

# Initialize debate button
col1, col2 = st.columns([1, 5])
with col1:
    start_btn = st.button("🚀 Start Debate", type="primary")

with col2:
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
                    "proposer_model": proposer_model,
                    "critic_model": critic_model,
                    "judge_model": judge_model,
                    "proposer_temperature": proposer_temp,
                    "critic_temperature": critic_temp,
                    "judge_temperature": judge_temp,
                    "proposer_prompt": proposer_prompt,
                    "critic_prompt": critic_prompt,
                    "judge_prompt": judge_prompt,
                    "max_rounds": 1,
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

# Poll debate events
if st.session_state.session_id and not st.session_state.debate_result:
    st.subheader("📡 Real-time Debate Log")
    
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
    st.subheader("🏁 Final Results")
    
    # Verdict and consensus score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verdict", st.session_state.debate_result.get("verdict", "N/A"))
    with col2:
        st.metric("Consensus Score", st.session_state.debate_result.get("consensus_score", 0))
    with col3:
        st.metric("Session ID", st.session_state.session_id[:8] + "...")
    
    # Metrics table
    st.subheader("📊 Performance Metrics")
    metrics_data = []
    
    # Proposer metrics
    proposer_events = [e for e in st.session_state.debate_events if e["event_type"] == "PROPOSER_FINAL"]
    if proposer_events:
        p_data = proposer_events[0]["data"]
        metrics_data.append({
            "Agent": "Proposer",
            "Latency (s)": f"{p_data.get('latency', 0):.2f}",
            "Valid JSON": "✓" if p_data.get('syntactic_valid', False) else "✗"
        })
    
    # Critic metrics
    critic_events = [e for e in st.session_state.debate_events if e["event_type"] == "CRITIC_FINAL"]
    if critic_events:
        c_data = critic_events[0]["data"]
        metrics_data.append({
            "Agent": "Critic",
            "Latency (s)": f"{c_data.get('latency', 0):.2f}",
            "Valid JSON": "✓" if c_data.get('syntactic_valid', False) else "✗"
        })
    
    # Judge metrics
    judge_events = [e for e in st.session_state.debate_events if e["event_type"] == "JUDGE_FINAL"]
    if judge_events:
        j_data = judge_events[0]["data"]
        metrics_data.append({
            "Agent": "Judge",
            "Latency (s)": f"{j_data.get('latency', 0):.2f}",
            "Valid JSON": "✓" if j_data.get('syntactic_valid', False) else "✗"
        })
    
    if metrics_data:
        st.table(metrics_data)
    
    # Full responses
    st.subheader("📝 Full Debate Transcript")
    
    with st.expander("🗣️ Proposer's Argument", expanded=True):
        st.markdown(st.session_state.debate_result.get("proposer_response", "No response"))
    
    with st.expander("🔍 Critic's Critique", expanded=True):
        st.markdown(st.session_state.debate_result.get("critic_response", "No response"))
    
    with st.expander("⚖️ Judge's Verdict", expanded=True):
        judge_resp = st.session_state.debate_result.get("judge_response", "No response")
        if not judge_resp or judge_resp == "No response":
            st.warning("Judge response not available. Check if debate completed successfully.")
        st.markdown(judge_resp)

# Display events if no result yet
if st.session_state.debate_events and not st.session_state.debate_result:
    st.subheader("📡 Real-time Debate Log")
    for event in st.session_state.debate_events:
        display_event(event)
