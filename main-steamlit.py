# main-streamlit.py

import os
import re
import sys
import uuid
import asyncio
import traceback
import pandas as pd
import streamlit as st

# --- Path Setup & Imports ---
# Add the parent folder to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT_DIR, "agents")

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from agents.iris import IrisOrchestrator
    from agents import db_ltm
except ImportError as e:
    st.error(f"**Fatal Error:** Could not import required modules. Details: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="IRIS", layout="wide", initial_sidebar_state="expanded")

# --- Generative UI Parser (Unchanged) ---
class GenerativeUI:
    def _render_metrics(self, text_response: str) -> bool:
        metric_pattern = r"\[METRIC\]\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*)"
        metrics = re.findall(metric_pattern, text_response)
        if not metrics: return False
        cols = st.columns(len(metrics))
        for i, (label, value, delta) in enumerate(metrics):
            with cols[i]:
                st.metric(label=label.strip(), value=value.strip(), delta=delta.strip() or None)
        return True

    def _render_dataframe(self, text_response: str) -> bool:
        table_pattern = r"(\|[^\n]+\|\r?\n)((?:\|:?[-]+:?)+\|)(\n(?:\|[^\n]+\|\r?\n?)*)"
        table_match = re.search(table_pattern, text_response)
        if not table_match: return False
        try:
            table_str = "".join(table_match.groups())
            lines = table_str.strip().split('\n')
            header = [h.strip() for h in lines[0].strip('|').split('|')]
            data = [[d.strip() for d in row.strip('|').split('|')] for row in lines[2:]]
            df = pd.DataFrame(data, columns=header)
            st.dataframe(df, use_container_width=True)
            return True
        except: return False

    def render(self, text_response: str):
        cleaned_response = text_response.strip()
        metric_pattern = r"\[METRIC\]\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*)"
        table_pattern = r"(\|[^\n]+\|\r?\n)((?:\|:?[-]+:?)+\|)(\n(?:\|[^\n]+\|\r?\n?)*)"
        metrics_found = self._render_metrics(cleaned_response)
        table_found = self._render_dataframe(cleaned_response)
        final_text = cleaned_response
        if metrics_found: final_text = re.sub(metric_pattern, '', final_text).strip()
        if table_found: final_text = re.sub(table_pattern, '', final_text).strip()
        if final_text: st.markdown(final_text, unsafe_allow_html=True)

gen_ui = GenerativeUI()

# --- THE DEFINITIVE FIX: Localized Async Execution ---
async def get_iris_response(payload, config):
    """
    This self-contained async function correctly initializes the checkpointer
    and runs the agent for a single request.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    
    # The checkpointer MUST be created inside an async context manager
    async with AsyncSqliteSaver.from_conn_string("iris_stm_checkpoint.sqlite") as memory:
        orchestrator = IrisOrchestrator(checkpointer=memory)
        # Await the agent's response
        result = await orchestrator.ainvoke(payload, config)
        return result

# --- Session Management & UI Rendering ---
def start_new_chat():
    st.session_state.thread_id = f"thread_st_{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.rerun()

def load_chat_session(session_id: int, thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.messages = db_ltm.get_session_chat_logs(session_id)
    st.rerun()

def render_styled_logo(font_size="3rem", padding_bottom="1rem"):
    st.markdown(f"""
    <style>.title-logo {{ font-size: {font_size}; font-weight: bold; background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: {padding_bottom}; }}</style>
    <p class="title-logo">IRIS</p>
    """, unsafe_allow_html=True)

def render_login_page():
    render_styled_logo()
    st.subheader("Your Personal AI Financial Analyst")
    with st.form("login_form"):
        st.write("Log in with your email or register a new account.")
        email = st.text_input("Email", key="login_email").lower().strip()
        name = st.text_input("Name (required for first-time login)", key="login_name").strip()
        submitted = st.form_submit_button("Continue")
        if submitted:
            if not email or "@" not in email: st.error("A valid email is required.")
            else:
                with st.spinner("Authenticating..."):
                    user_data = db_ltm.authenticate_and_get_user(email=email, name=name)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.success(f"Welcome, {user_data.get('name') or email}!")
                        start_new_chat()
                    else: st.error("Login failed. Please check your email and name.")

def render_chat_page():
    with st.sidebar:
        render_styled_logo(font_size="1.75rem", padding_bottom="0.5rem")
        st.markdown(f"**{st.session_state.user.get('name') or st.session_state.user.get('email')}**")
        st.divider()
        if st.button("➕ New Chat", use_container_width=True): start_new_chat()
        st.markdown("##### Chat History")
        sessions = db_ltm.get_user_sessions(st.session_state.user['id'])
        for session in sessions:
            session_label = f"Chat from {session['started_at'].strftime('%b %d, %H:%M')}"
            if st.button(session_label, key=session['id'], use_container_width=True, type="secondary"):
                load_chat_session(session['id'], session['thread_id'])

    st.title("IRIS")
    st.caption("AI-Powered Financial Analyst, ready to assist.")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            gen_ui.render(msg["content"])

    if prompt := st.chat_input("Ask about stocks, markets, or set your preferences..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    if st.session_state.get("messages") and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("IRIS is thinking..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    payload = {
                        "user_identifier": st.session_state.user['email'],
                        "thread_id": st.session_state.thread_id,
                        "user_input": st.session_state.messages[-1]["content"]
                    }
                    # Use asyncio.run() to execute the self-contained async function
                    response_state = asyncio.run(get_iris_response(payload, config))
                    response_content = response_state.get("final_response", "Sorry, an error occurred.")
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    st.rerun()
                except Exception as e:
                    error_message = "An unexpected error occurred. Please check the terminal logs for details."
                    st.error(error_message)
                    print(traceback.format_exc()) # Print full error to terminal
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Main App Logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_login_page()
else:
    render_chat_page()